from __future__ import annotations

import json

import httpx
import pytest

from systemone_router.decision import (
    DecisionError,
    SystemOneServerClient,
    VercelGatewayDecisionClient,
    decision_client_from_env,
    parse_choice_answer,
)

CRITERIA = {"simple": "easy", "moderate": "medium", "complex": "hard"}


def _mock_transport(captured: dict, response_body: dict, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(status, json=response_body)

    return httpx.MockTransport(handler)


def test_parse_choice_answer_normalises_probabilities():
    payload = {"answers": {"complexity": {"type": "choice", "choice": "moderate", "probabilities": {"moderate": 0.8, "simple": 0.2}}}}
    decision = parse_choice_answer(payload, CRITERIA)
    assert decision.choice == "moderate"
    assert decision.probabilities == {"simple": 0.2, "moderate": 0.8, "complex": 0.0}
    assert decision.confidence == pytest.approx(0.8)


def test_parse_choice_answer_rejects_unknown_option():
    with pytest.raises(DecisionError):
        parse_choice_answer({"answers": {"complexity": {"choice": "other", "probabilities": {}}}}, CRITERIA)


def test_parse_choice_answer_rejects_missing_answer():
    with pytest.raises(DecisionError):
        parse_choice_answer({"answers": {}}, CRITERIA)


def test_gateway_client_request_and_response():
    captured: dict = {}
    body = {"model": "convaiinnovations/laya", "answers": {"complexity": {"type": "choice", "choice": "complex", "probabilities": {"simple": 0.05, "moderate": 0.15, "complex": 0.8}}}}
    transport = _mock_transport(captured, body)

    client = VercelGatewayDecisionClient("vck_test", provider_options={"gateway": {"only": ["boundless"]}}, transport=transport)
    decision = client.choose("prove the Riemann hypothesis", "How hard?", CRITERIA)

    assert decision.choice == "complex"
    assert captured["url"] == "https://ai-gateway.vercel.sh/v1/evaluate"
    assert captured["headers"]["authorization"] == "Bearer vck_test"
    assert captured["body"]["model"] == "convaiinnovations/laya"
    assert captured["body"]["state"] == "prove the Riemann hypothesis"
    assert captured["body"]["questions"]["complexity"] == {"type": "choice", "instructions": "How hard?", "criteria": CRITERIA}
    assert captured["body"]["providerOptions"] == {"gateway": {"only": ["boundless"]}}


async def test_gateway_client_async():
    captured: dict = {}
    body = {"answers": {"complexity": {"choice": "simple", "probabilities": {"simple": 0.9}}}}
    transport = _mock_transport(captured, body)

    decision = await VercelGatewayDecisionClient("vck_test", transport=transport).achoose("hi", "How hard?", CRITERIA)
    assert decision.choice == "simple"


def test_server_client_targets_systemone_endpoint():
    captured: dict = {}
    body = {"answers": {"complexity": {"type": "choice", "choice": "moderate", "probabilities": {"moderate": 0.7}, "confidence": 0.7}}}
    transport = _mock_transport(captured, body)

    client = SystemOneServerClient("http://laya.internal:8000/", api_key="secret", transport=transport)
    decision = client.choose({"document": "write a script"}, "How hard?", CRITERIA)

    assert decision.choice == "moderate"
    assert captured["url"] == "http://laya.internal:8000/v1/systemone"
    assert captured["headers"]["authorization"] == "Bearer secret"
    assert "model" not in captured["body"]
    assert captured["body"]["state"] == {"document": "write a script"}


def test_http_error_becomes_decision_error():
    transport = _mock_transport({}, {"error": "nope"}, status=503)
    with pytest.raises(DecisionError):
        VercelGatewayDecisionClient("vck_test", transport=transport).choose("x", "q", CRITERIA)


def test_from_env_gateway():
    client = decision_client_from_env({"AI_GATEWAY_API_KEY": "k", "SYSTEMONE_MODEL": "typesafe-ai/jev"})
    assert isinstance(client, VercelGatewayDecisionClient)
    assert client.model == "typesafe-ai/jev"


def test_from_env_gateway_requires_key():
    with pytest.raises(ValueError):
        decision_client_from_env({"SYSTEMONE_BACKEND": "gateway"})


def test_from_env_server():
    client = decision_client_from_env({"SYSTEMONE_BACKEND": "server", "SYSTEMONE_SERVER_URL": "http://h:1"})
    assert isinstance(client, SystemOneServerClient)


def test_from_env_unknown_backend():
    with pytest.raises(ValueError):
        decision_client_from_env({"SYSTEMONE_BACKEND": "carrier-pigeon"})
