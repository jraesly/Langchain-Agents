"""Turn logged requests into labeled data for evaluating or fine-tuning the decision model.

The label for a request is *the cheapest tier whose answer passes a quality bar*.
Each request is answered once per tier, each answer is judged by a judge model,
and the first passing tier (least capable first) becomes the label. If nothing
passes, the most capable tier is the label.

Outputs two files:

* a ``laya-train`` file: one ``{"text": ..., "label": ...}`` row per request
* a ``laya-evals`` file: one ``{"state": ..., "questions": ..., "expected": ...}`` row
  per request, using the router's own question so the eval measures routing directly

Answers are generated single-turn with no tools, so this measures answer
quality, not tool use. Requests that need tools are still labeled, but judge
them with that in mind.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .decision import QUESTION_ID
from .models import build_chat_model
from .routing import DEFAULT_INSTRUCTIONS, DEFAULT_TIERS, ModelTier, condense_state
from .routing_log import read_routing_log

DEFAULT_JUDGE_MODEL = "claude-opus-5-5"
DEFAULT_QUALITY_BAR = (
    "The answer is correct, complete for what was asked, and would not need a "
    "follow-up from a careful user. Minor stylistic differences do not matter."
)


class Judgment(BaseModel):
    """Structured verdict from the judge model."""

    passes: bool = Field(description="True if the answer meets the quality bar for this request.")
    rationale: str = Field(description="One or two sentences on why.")


class Judge(Protocol):
    async def __call__(self, request: str, answer: str) -> Judgment: ...


@dataclass
class LabeledExample:
    text: str
    label: str
    answers: dict[str, str] = field(default_factory=dict)
    judgments: dict[str, dict[str, Any]] = field(default_factory=dict)
    router_tier: str | None = None
    router_confidence: float | None = None

    @property
    def router_agreed(self) -> bool | None:
        return None if self.router_tier is None else self.router_tier == self.label

    def to_train_row(self) -> dict[str, Any]:
        return {"text": self.text, "label": self.label, "router_tier": self.router_tier}

    def to_evals_row(self, criteria: dict[str, str], instructions: str) -> dict[str, Any]:
        return {
            "state": self.text,
            "questions": {QUESTION_ID: {"type": "choice", "instructions": instructions, "criteria": criteria}},
            "expected": {QUESTION_ID: self.label},
            "tags": [f"router:{self.router_tier}"] if self.router_tier else [],
        }


def cheapest_passing_tier(tiers: Sequence[ModelTier], passes: dict[str, bool]) -> ModelTier:
    """First tier (least capable first) whose answer passed, else the last tier."""
    for tier in tiers:
        if passes.get(tier.name):
            return tier
    return tiers[-1]


def make_llm_judge(model: BaseChatModel, quality_bar: str = DEFAULT_QUALITY_BAR) -> Judge:
    """Wrap a chat model as a structured-output judge."""
    structured = model.with_structured_output(Judgment, method="json_schema")
    system = SystemMessage(
        "You grade whether an AI assistant's answer meets a quality bar for the request. "
        "Judge only the answer in front of you. Do not solve the request yourself.\n\n"
        f"Quality bar: {quality_bar}"
    )

    async def judge(request: str, answer: str) -> Judgment:
        prompt = HumanMessage(f"<request>\n{request}\n</request>\n\n<answer>\n{answer}\n</answer>")
        result = await structured.ainvoke([system, prompt])
        if isinstance(result, Judgment):
            return result
        return Judgment.model_validate(result)

    return judge


async def label_one(
    text: str,
    tiers: Sequence[ModelTier],
    answer_models: dict[str, BaseChatModel],
    judge: Judge,
    *,
    router_tier: str | None = None,
    router_confidence: float | None = None,
) -> LabeledExample:
    """Answer ``text`` on every tier, judge each answer, and label it."""
    answers_list = await asyncio.gather(
        *(answer_models[t.name].ainvoke([HumanMessage(text)]) for t in tiers)
    )
    answers = {t.name: msg.text for t, msg in zip(tiers, answers_list)}
    judgments_list = await asyncio.gather(*(judge(text, answers[t.name]) for t in tiers))
    judgments = {t.name: j.model_dump() for t, j in zip(tiers, judgments_list)}
    label = cheapest_passing_tier(tiers, {name: j["passes"] for name, j in judgments.items()})
    return LabeledExample(
        text=text,
        label=label.name,
        answers=answers,
        judgments=judgments,
        router_tier=router_tier,
        router_confidence=router_confidence,
    )


async def label_requests(
    requests: Iterable[dict[str, Any]],
    tiers: Sequence[ModelTier] = DEFAULT_TIERS,
    *,
    judge: Judge,
    answer_model_factory: Callable[[ModelTier], BaseChatModel] = build_chat_model,
    concurrency: int = 4,
    max_state_chars: int = 1_500,
) -> list[LabeledExample]:
    """Label many requests. Each request dict needs ``text`` and may carry ``routing``."""
    answer_models = {t.name: answer_model_factory(t) for t in tiers}
    semaphore = asyncio.Semaphore(concurrency)

    async def run(req: dict[str, Any]) -> LabeledExample:
        routing = req.get("routing") or {}
        async with semaphore:
            return await label_one(
                condense_state(req["text"], max_state_chars),
                tiers,
                answer_models,
                judge,
                router_tier=routing.get("tier"),
                router_confidence=routing.get("confidence"),
            )

    return await asyncio.gather(*(run(r) for r in requests))


def load_requests(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Read a routing log or any JSONL with a ``text`` key; dedupe on text."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for entry in read_routing_log(path):
        text = (entry.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(entry)
        if limit is not None and len(out) >= limit:
            break
    return out


def summarize(examples: Sequence[LabeledExample], tiers: Sequence[ModelTier]) -> dict[str, Any]:
    labels = {t.name: sum(1 for e in examples if e.label == t.name) for t in tiers}
    graded = [e for e in examples if e.router_tier is not None]
    agreed = sum(1 for e in graded if e.router_agreed)
    too_cheap = sum(1 for e in graded if _tier_index(tiers, e.router_tier) < _tier_index(tiers, e.label))
    too_expensive = sum(1 for e in graded if _tier_index(tiers, e.router_tier) > _tier_index(tiers, e.label))
    return {
        "examples": len(examples),
        "label_counts": labels,
        "router_graded": len(graded),
        "router_agreement": (agreed / len(graded)) if graded else None,
        "router_too_cheap": too_cheap,
        "router_too_expensive": too_expensive,
    }


def _tier_index(tiers: Sequence[ModelTier], name: str | None) -> int:
    for i, t in enumerate(tiers):
        if t.name == name:
            return i
    return -1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="systemone-label",
        description="Label logged requests with the cheapest tier whose answer passes a quality bar.",
    )
    parser.add_argument("input", type=Path, help="routing log JSONL (or any JSONL with a 'text' field)")
    parser.add_argument("--train-out", type=Path, default=Path("data/train.jsonl"), help="laya-train rows")
    parser.add_argument("--evals-out", type=Path, default=Path("data/evals.jsonl"), help="laya-evals rows")
    parser.add_argument("--limit", type=int, default=50, help="max requests to label (default 50)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--quality-bar", default=DEFAULT_QUALITY_BAR)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--yes", action="store_true", help="run without the cost confirmation")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover
        pass

    requests = load_requests(args.input, args.limit)
    if not requests:
        print(f"no requests with text found in {args.input}", file=sys.stderr)
        return 1

    tiers = DEFAULT_TIERS
    calls = len(requests) * len(tiers) * 2
    print(f"{len(requests)} requests x {len(tiers)} tiers = {len(requests) * len(tiers)} answers + as many judge calls ({calls} API calls total)")
    if not args.yes:
        print("re-run with --yes to proceed", file=sys.stderr)
        return 2
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 1

    judge = make_llm_judge(build_chat_model(ModelTier("judge", args.judge_model, "", effort="medium")), args.quality_bar)
    examples = asyncio.run(label_requests(requests, tiers, judge=judge, concurrency=args.concurrency))

    criteria = {t.name: t.description for t in tiers}
    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.evals_out.parent.mkdir(parents=True, exist_ok=True)
    with args.train_out.open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(e.to_train_row(), ensure_ascii=False) + "\n")
    with args.evals_out.open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(e.to_evals_row(criteria, DEFAULT_INSTRUCTIONS), ensure_ascii=False) + "\n")
    details = args.train_out.with_suffix(".details.jsonl")
    with details.open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    print(json.dumps(summarize(examples, tiers), indent=2))
    print(f"wrote {args.train_out}, {args.evals_out}, {details}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
