from __future__ import annotations

from collections.abc import Mapping

import pytest

from systemone_router.decision import ChoiceDecision, DecisionError


class FakeDecisionClient:
    """Scripted decision model. Records what it was asked."""

    def __init__(self, choice: str, probabilities: dict[str, float] | None = None, *, fail: bool = False):
        self.choice = choice
        self.probabilities = probabilities
        self.fail = fail
        self.calls: list[tuple[object, str, dict[str, str]]] = []

    def choose(self, state, instructions: str, criteria: Mapping[str, str]) -> ChoiceDecision:
        self.calls.append((state, instructions, dict(criteria)))
        if self.fail:
            raise DecisionError("boom")
        probs = self.probabilities or {name: (1.0 if name == self.choice else 0.0) for name in criteria}
        return ChoiceDecision(choice=self.choice, probabilities=probs)

    async def achoose(self, state, instructions: str, criteria: Mapping[str, str]) -> ChoiceDecision:
        return self.choose(state, instructions, criteria)


@pytest.fixture
def fake_client_factory():
    return FakeDecisionClient
