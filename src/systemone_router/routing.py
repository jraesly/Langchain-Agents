"""Map a request to a model tier using a System One decision model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .decision import ChoiceDecision, DecisionClient, DecisionError


@dataclass(frozen=True)
class ModelTier:
    """One rung on the complexity ladder.

    ``description`` is what the decision model reads when deciding whether a
    request belongs in this tier, so write it as concrete examples.
    """

    name: str
    model: str
    description: str
    effort: str | None = None
    max_tokens: int = 16_000


DEFAULT_TIERS: tuple[ModelTier, ...] = (
    ModelTier(
        name="simple",
        model="claude-haiku-5-5",
        description=(
            "greetings and small talk, a short factual question, rewording a sentence, "
            "a yes/no or one-line answer, a single obvious step"
        ),
        effort="low",
    ),
    ModelTier(
        name="moderate",
        model="claude-sonnet-5-5",
        description=(
            "everyday coding or scripting, summarising or drafting a document, "
            "a well-specified multi-step task, a question that needs some reasoning"
        ),
        effort="medium",
    ),
    ModelTier(
        name="complex",
        model="claude-opus-5-5",
        description=(
            "open-ended research or analysis, debugging a hard or subtle problem, "
            "system architecture or design trade-offs, long multi-step agentic work, "
            "maths, logic, or proofs that need careful reasoning"
        ),
        effort="high",
    ),
)

DEFAULT_INSTRUCTIONS = (
    "How difficult is this request for an AI assistant to answer well? "
    "Pick the lowest tier whose examples match the request."
)


@dataclass(frozen=True)
class RoutingDecision:
    """The outcome of routing one request."""

    tier: ModelTier
    reason: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    decided_tier: ModelTier | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.name,
            "model": self.tier.model,
            "effort": self.tier.effort,
            "reason": self.reason,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "decided_tier": self.decided_tier.name if self.decided_tier else None,
            "error": self.error,
        }


def condense_state(text: str, max_chars: int) -> str:
    """Trim long inputs to fit a decision model's small context window.

    Keeps the start (where the ask usually is) and the end (where a question
    following pasted material usually is).
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head - len(" ... ")
    return text[:head] + " ... " + text[-tail:]


class ComplexityRouter:
    """Ask a decision model which tier a request belongs to.

    Args:
        client: Any :class:`DecisionClient`.
        tiers: Ordered from least to most capable.
        min_confidence: If the chosen tier's probability is below this, the
            request is bumped one tier up. Cheap mistakes are the expensive
            kind here, so uncertainty escalates rather than downgrades.
        fallback_tier: Tier name used when the decision model fails or the
            input is empty. Defaults to the most capable tier.
        instructions: The question put to the decision model.
        max_state_chars: Inputs longer than this are condensed first.
    """

    def __init__(
        self,
        client: DecisionClient,
        tiers: Sequence[ModelTier] = DEFAULT_TIERS,
        *,
        min_confidence: float = 0.5,
        fallback_tier: str | None = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
        max_state_chars: int = 1_500,
    ) -> None:
        if len(tiers) < 2:
            raise ValueError("ComplexityRouter needs at least two tiers")
        names = [t.name for t in tiers]
        if len(set(names)) != len(names):
            raise ValueError(f"tier names must be unique, got {names}")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self.client = client
        self.tiers = tuple(tiers)
        self.min_confidence = min_confidence
        self.instructions = instructions
        self.max_state_chars = max_state_chars
        self._by_name = {t.name: t for t in self.tiers}
        self.fallback_tier = self._by_name[fallback_tier] if fallback_tier else self.tiers[-1]

    @property
    def criteria(self) -> dict[str, str]:
        return {t.name: t.description for t in self.tiers}

    def tier(self, name: str) -> ModelTier:
        return self._by_name[name]

    def route(self, text: str) -> RoutingDecision:
        state = condense_state(text, self.max_state_chars)
        if not state:
            return RoutingDecision(tier=self.fallback_tier, reason="empty_input")
        try:
            decision = self.client.choose(state, self.instructions, self.criteria)
        except DecisionError as exc:
            return RoutingDecision(tier=self.fallback_tier, reason="decision_model_error", error=str(exc))
        return self._finalize(decision)

    async def aroute(self, text: str) -> RoutingDecision:
        state = condense_state(text, self.max_state_chars)
        if not state:
            return RoutingDecision(tier=self.fallback_tier, reason="empty_input")
        try:
            decision = await self.client.achoose(state, self.instructions, self.criteria)
        except DecisionError as exc:
            return RoutingDecision(tier=self.fallback_tier, reason="decision_model_error", error=str(exc))
        return self._finalize(decision)

    def _finalize(self, decision: ChoiceDecision) -> RoutingDecision:
        decided = self._by_name[decision.choice]
        index = self.tiers.index(decided)
        if decision.confidence < self.min_confidence and index < len(self.tiers) - 1:
            return RoutingDecision(
                tier=self.tiers[index + 1],
                reason="low_confidence_escalation",
                probabilities=decision.probabilities,
                confidence=decision.confidence,
                decided_tier=decided,
            )
        return RoutingDecision(
            tier=decided,
            reason="decision",
            probabilities=decision.probabilities,
            confidence=decision.confidence,
            decided_tier=decided,
        )
