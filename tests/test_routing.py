from __future__ import annotations

import pytest

from systemone_router.routing import DEFAULT_TIERS, ComplexityRouter, ModelTier, condense_state


def test_routes_to_chosen_tier(fake_client_factory):
    router = ComplexityRouter(fake_client_factory("moderate", {"simple": 0.1, "moderate": 0.8, "complex": 0.1}))
    decision = router.route("write a bash script that renames files")
    assert decision.tier.name == "moderate"
    assert decision.tier.model == "claude-sonnet-5-5"
    assert decision.reason == "decision"
    assert decision.confidence == pytest.approx(0.8)


def test_sends_tier_descriptions_as_criteria(fake_client_factory):
    client = fake_client_factory("simple")
    ComplexityRouter(client).route("hello")
    state, instructions, criteria = client.calls[0]
    assert state == "hello"
    assert set(criteria) == {t.name for t in DEFAULT_TIERS}
    assert criteria["complex"] == DEFAULT_TIERS[-1].description
    assert "tier" in instructions


def test_low_confidence_escalates_one_tier(fake_client_factory):
    router = ComplexityRouter(fake_client_factory("simple", {"simple": 0.4, "moderate": 0.35, "complex": 0.25}), min_confidence=0.5)
    decision = router.route("hmm")
    assert decision.tier.name == "moderate"
    assert decision.decided_tier.name == "simple"
    assert decision.reason == "low_confidence_escalation"


def test_low_confidence_on_top_tier_stays_put(fake_client_factory):
    router = ComplexityRouter(fake_client_factory("complex", {"simple": 0.3, "moderate": 0.3, "complex": 0.4}), min_confidence=0.5)
    decision = router.route("hmm")
    assert decision.tier.name == "complex"
    assert decision.reason == "decision"


def test_decision_error_falls_back_to_top_tier(fake_client_factory):
    decision = ComplexityRouter(fake_client_factory("simple", fail=True)).route("anything")
    assert decision.tier.name == "complex"
    assert decision.reason == "decision_model_error"
    assert "boom" in decision.error


def test_custom_fallback_tier(fake_client_factory):
    decision = ComplexityRouter(fake_client_factory("simple", fail=True), fallback_tier="moderate").route("x")
    assert decision.tier.name == "moderate"


def test_empty_input_uses_fallback_without_calling_model(fake_client_factory):
    client = fake_client_factory("simple")
    decision = ComplexityRouter(client).route("   ")
    assert decision.reason == "empty_input"
    assert client.calls == []


async def test_async_route(fake_client_factory):
    decision = await ComplexityRouter(fake_client_factory("complex")).aroute("design a distributed cache")
    assert decision.tier.model == "claude-opus-5-5"


def test_condense_state_keeps_head_and_tail():
    text = "A" * 2000 + "B" * 2000
    out = condense_state(text, 1000)
    assert len(out) == 1000
    assert out.startswith("A") and out.endswith("B") and " ... " in out


def test_long_input_is_condensed_before_routing(fake_client_factory):
    client = fake_client_factory("simple")
    ComplexityRouter(client, max_state_chars=100).route("x" * 500)
    assert len(client.calls[0][0]) == 100


def test_custom_tiers(fake_client_factory):
    tiers = [ModelTier("cheap", "claude-haiku-5-5", "easy"), ModelTier("strong", "claude-opus-5-5", "hard", effort="xhigh")]
    decision = ComplexityRouter(fake_client_factory("strong"), tiers).route("x")
    assert decision.tier.effort == "xhigh"


def test_rejects_bad_config(fake_client_factory):
    with pytest.raises(ValueError):
        ComplexityRouter(fake_client_factory("a"), [ModelTier("a", "m", "d")])
    with pytest.raises(ValueError):
        ComplexityRouter(fake_client_factory("a"), [ModelTier("a", "m", "d"), ModelTier("a", "m", "d")])
    with pytest.raises(ValueError):
        ComplexityRouter(fake_client_factory("a"), min_confidence=1.5)


def test_to_dict_round_trip(fake_client_factory):
    data = ComplexityRouter(fake_client_factory("moderate")).route("x").to_dict()
    assert data["tier"] == "moderate"
    assert data["model"] == "claude-sonnet-5-5"
    assert data["effort"] == "medium"
