from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from systemone_router import ROUTING_STATE_KEY, RoutingLogger, create_routed_agent, read_routing_log
from test_agent import make_factory


def test_logger_appends_jsonl(tmp_path):
    log = RoutingLogger(tmp_path / "nested" / "log.jsonl")
    log.record({"text": "a", "routing": {"tier": "simple"}})
    log.record({"text": "b"})
    rows = list(read_routing_log(log.path))
    assert [r["text"] for r in rows] == ["a", "b"]
    assert all("ts" in r for r in rows)


def test_read_log_skips_corrupt_lines(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text('{"text": "ok"}\nnot json\n\n{"text": "also ok"}\n')
    assert [r["text"] for r in read_routing_log(p)] == ["ok", "also ok"]


def test_agent_logs_each_invocation(tmp_path, fake_client_factory):
    built = {}
    log_path = tmp_path / "routing.jsonl"
    agent = create_routed_agent(
        tools=[],
        decision_client=fake_client_factory("moderate", {"simple": 0.2, "moderate": 0.7, "complex": 0.1}),
        model_factory=make_factory(built),
        log_path=str(log_path),
    )
    result = agent.invoke({"messages": [HumanMessage("summarise the quarterly report")]})

    rows = list(read_routing_log(log_path))
    assert len(rows) == 1
    row = rows[0]
    assert row["text"] == "summarise the quarterly report"
    assert row["routing"]["tier"] == "moderate"
    assert row["routing"]["confidence"] == 0.7
    assert "state" not in row["routing"]  # hoisted to "text"
    assert row["answer"] == "answered by moderate"
    assert row["message_count"] == len(result["messages"])
    assert result[ROUTING_STATE_KEY]["state"] == "summarise the quarterly report"


async def test_agent_logs_on_async_path(tmp_path, fake_client_factory):
    built = {}
    log_path = tmp_path / "routing.jsonl"
    agent = create_routed_agent(tools=[], decision_client=fake_client_factory("simple"), model_factory=make_factory(built), log_path=str(log_path))
    await agent.ainvoke({"messages": [HumanMessage("hi")]})
    await agent.ainvoke({"messages": [HumanMessage("hello")]})
    assert [json.loads(l)["text"] for l in log_path.read_text().splitlines()] == ["hi", "hello"]
