from __future__ import annotations

import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from systemone_router.decision import QUESTION_ID
from systemone_router.labeling import (
    Judgment,
    LabeledExample,
    cheapest_passing_tier,
    label_requests,
    load_requests,
    main,
    summarize,
)
from systemone_router.routing import DEFAULT_TIERS, ModelTier


def test_cheapest_passing_tier_picks_first_pass():
    assert cheapest_passing_tier(DEFAULT_TIERS, {"simple": False, "moderate": True, "complex": True}).name == "moderate"


def test_cheapest_passing_tier_defaults_to_top_when_nothing_passes():
    assert cheapest_passing_tier(DEFAULT_TIERS, {}).name == "complex"


def _answer_factory(tier: ModelTier):
    return GenericFakeChatModel(messages=iter([AIMessage(f"{tier.name} answer")] * 100))


def _judge_passing_from(min_tier: str):
    order = [t.name for t in DEFAULT_TIERS]

    async def judge(request: str, answer: str) -> Judgment:
        tier = answer.split()[0]
        return Judgment(passes=order.index(tier) >= order.index(min_tier), rationale="scripted")

    return judge


async def test_label_requests_end_to_end():
    requests = [
        {"text": "hi", "routing": {"tier": "simple", "confidence": 0.9}},
        {"text": "write a parser", "routing": {"tier": "simple", "confidence": 0.4}},
    ]
    examples = await label_requests(requests, DEFAULT_TIERS, judge=_judge_passing_from("moderate"), answer_model_factory=_answer_factory)

    assert [e.label for e in examples] == ["moderate", "moderate"]
    assert examples[0].answers["complex"] == "complex answer"
    assert examples[0].judgments["simple"]["passes"] is False
    assert examples[0].router_tier == "simple"
    assert examples[0].router_agreed is False

    summary = summarize(examples, DEFAULT_TIERS)
    assert summary["label_counts"] == {"simple": 0, "moderate": 2, "complex": 0}
    assert summary["router_too_cheap"] == 2
    assert summary["router_too_expensive"] == 0
    assert summary["router_agreement"] == 0.0


def test_rows_match_laya_formats():
    e = LabeledExample(text="t", label="complex", router_tier="moderate")
    assert e.to_train_row() == {"text": "t", "label": "complex", "router_tier": "moderate"}
    row = e.to_evals_row({"simple": "a", "moderate": "b", "complex": "c"}, "how hard?")
    assert row["state"] == "t"
    assert row["questions"][QUESTION_ID]["type"] == "choice"
    assert row["expected"] == {QUESTION_ID: "complex"}
    assert row["tags"] == ["router:moderate"]


def test_load_requests_dedupes_and_limits(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(json.dumps({"text": t}) for t in ["a", "b", "a", "", "c"]) + "\n")
    assert [r["text"] for r in load_requests(p, None)] == ["a", "b", "c"]
    assert [r["text"] for r in load_requests(p, 2)] == ["a", "b"]


def test_cli_stops_before_spending_without_yes(tmp_path, capsys):
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"text": "hi"}) + "\n")
    assert main([str(p), "--train-out", str(tmp_path / "t.jsonl"), "--evals-out", str(tmp_path / "e.jsonl")]) == 2
    out = capsys.readouterr()
    assert "6 API calls" in out.out
    assert "--yes" in out.err


def test_cli_errors_on_empty_input(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text("")
    assert main([str(p)]) == 1
