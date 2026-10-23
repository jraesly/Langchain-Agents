"""Clients for System One decision models.

A System One model does not generate text. It reads a *state* (the text under
consideration) and answers typed questions about it in a single forward pass,
returning calibrated probabilities. This module only needs one question type,
``choice``, which is enough to grade a request's complexity.

Three backends are provided. All of them return the same :class:`ChoiceDecision`
so the router does not care which one is in use:

* :class:`VercelGatewayDecisionClient` - hosted, ``POST /v1/evaluate`` on Vercel
  AI Gateway. Works with any decision model the gateway lists; the default is
  Laya (``convaiinnovations/laya``), which is open-weight under Apache 2.0.
* :class:`SystemOneServerClient` - any Jev-compatible ``POST /v1/systemone``
  server, such as ``laya-serve`` running on your own machine.
* :class:`LocalLayaClient` - the ``laya`` Python package in-process.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import httpx

DEFAULT_GATEWAY_EVALUATE_URL = "https://ai-gateway.vercel.sh/v1/evaluate"
DEFAULT_GATEWAY_MODEL = "convaiinnovations/laya"
DEFAULT_SERVER_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 15.0

State = str | Mapping[str, Any] | list[Any]


class DecisionError(RuntimeError):
    """The decision model could not be reached or returned an unusable answer."""


@dataclass(frozen=True)
class ChoiceDecision:
    """One answer to a ``choice`` question."""

    choice: str
    probabilities: dict[str, float]
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def confidence(self) -> float:
        """Probability the model assigned to the option it picked."""
        return float(self.probabilities.get(self.choice, 0.0))


@runtime_checkable
class DecisionClient(Protocol):
    """Anything that can answer a single ``choice`` question about a state."""

    def choose(
        self, state: State, instructions: str, criteria: Mapping[str, str]
    ) -> ChoiceDecision: ...

    async def achoose(
        self, state: State, instructions: str, criteria: Mapping[str, str]
    ) -> ChoiceDecision: ...


QUESTION_ID = "complexity"


def _choice_question(instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
    return {
        QUESTION_ID: {
            "type": "choice",
            "instructions": instructions,
            "criteria": dict(criteria),
        }
    }


def parse_choice_answer(payload: Mapping[str, Any], criteria: Mapping[str, str]) -> ChoiceDecision:
    """Normalise a gateway, laya-serve, or in-process answer into a ChoiceDecision.

    Every backend returns ``{"answers": {<qid>: {"choice": ..., "probabilities": {...}}}}``.
    """
    answers = payload.get("answers")
    if not isinstance(answers, Mapping) or QUESTION_ID not in answers:
        raise DecisionError(f"decision response has no answer for {QUESTION_ID!r}: {payload!r}")
    answer = answers[QUESTION_ID]
    choice = answer.get("choice")
    probabilities = answer.get("probabilities") or {}
    if choice not in criteria:
        raise DecisionError(f"decision model picked unknown option {choice!r}; expected one of {list(criteria)}")
    probs = {name: float(probabilities.get(name, 0.0)) for name in criteria}
    return ChoiceDecision(choice=str(choice), probabilities=probs, raw=payload)


class _HttpDecisionClient:
    """Shared plumbing for the two HTTP backends."""

    def __init__(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = url
        self._headers = dict(headers)
        self._timeout = timeout
        self._transport = transport  # for tests; httpx.MockTransport serves both sync and async

    def _body(self, state: State, instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
        raise NotImplementedError

    def choose(self, state: State, instructions: str, criteria: Mapping[str, str]) -> ChoiceDecision:
        body = self._body(state, instructions, criteria)
        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as http:
                response = http.post(self._url, json=body, headers=self._headers)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise DecisionError(f"decision request to {self._url} failed: {exc}") from exc
        return parse_choice_answer(payload, criteria)

    async def achoose(self, state: State, instructions: str, criteria: Mapping[str, str]) -> ChoiceDecision:
        body = self._body(state, instructions, criteria)
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as http:
                response = await http.post(self._url, json=body, headers=self._headers)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise DecisionError(f"decision request to {self._url} failed: {exc}") from exc
        return parse_choice_answer(payload, criteria)


class VercelGatewayDecisionClient(_HttpDecisionClient):
    """Hosted decision models via Vercel AI Gateway's ``POST /v1/evaluate``.

    Any model with the gateway's *Decision* capability works here. Laya is the
    default because it is the open-weight option.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_GATEWAY_MODEL,
        url: str = DEFAULT_GATEWAY_EVALUATE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        provider_options: Mapping[str, Any] | None = None,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("VercelGatewayDecisionClient needs an AI Gateway API key")
        super().__init__(url, {"Authorization": f"Bearer {api_key}"}, timeout, transport)
        self.model = model
        self._provider_options = dict(provider_options) if provider_options else None

    def _body(self, state: State, instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "state": state,
            "questions": _choice_question(instructions, criteria),
        }
        if self._provider_options:
            body["providerOptions"] = self._provider_options
        return body


class SystemOneServerClient(_HttpDecisionClient):
    """Any Jev-compatible ``POST /v1/systemone`` server, e.g. ``laya-serve``."""

    def __init__(
        self,
        base_url: str = DEFAULT_SERVER_URL,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        super().__init__(base_url.rstrip("/") + "/v1/systemone", headers, timeout, transport)
        self.model = model

    def _body(self, state: State, instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
        body: dict[str, Any] = {"state": state, "questions": _choice_question(instructions, criteria)}
        if self.model:
            body["model"] = self.model
        return body


class LocalLayaClient:
    """Run Laya in-process with the ``laya`` package (``pip install laya``).

    The model loads lazily on the first call. Pass ``router`` to reuse a
    ``laya.Router`` you have already configured (device, preloading, etc.).
    """

    def __init__(self, router: Any | None = None, **router_kwargs: Any) -> None:
        self._router = router
        self._router_kwargs = router_kwargs

    def _get_router(self) -> Any:
        if self._router is None:
            try:
                from laya import Router  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise DecisionError(
                    "LocalLayaClient needs the `laya` package: pip install 'systemone-router[local]'"
                ) from exc
            self._router = Router(**self._router_kwargs)
        return self._router

    def choose(self, state: State, instructions: str, criteria: Mapping[str, str]) -> ChoiceDecision:
        try:
            payload = self._get_router().predict(state, _choice_question(instructions, criteria))
        except DecisionError:
            raise
        except Exception as exc:  # the laya package raises plain exceptions
            raise DecisionError(f"local laya prediction failed: {exc}") from exc
        return parse_choice_answer(payload, criteria)

    async def achoose(self, state: State, instructions: str, criteria: Mapping[str, str]) -> ChoiceDecision:
        return await asyncio.to_thread(self.choose, state, instructions, criteria)


def decision_client_from_env(environ: Mapping[str, str] | None = None) -> DecisionClient:
    """Build a client from ``SYSTEMONE_BACKEND`` and friends (see ``.env.example``)."""
    env = os.environ if environ is None else environ
    backend = env.get("SYSTEMONE_BACKEND", "gateway").strip().lower()
    if backend == "gateway":
        api_key = env.get("AI_GATEWAY_API_KEY", "")
        if not api_key:
            raise ValueError("SYSTEMONE_BACKEND=gateway needs AI_GATEWAY_API_KEY")
        return VercelGatewayDecisionClient(
            api_key,
            model=env.get("SYSTEMONE_MODEL", DEFAULT_GATEWAY_MODEL),
            url=env.get("AI_GATEWAY_EVALUATE_URL", DEFAULT_GATEWAY_EVALUATE_URL),
        )
    if backend == "server":
        return SystemOneServerClient(
            env.get("SYSTEMONE_SERVER_URL", DEFAULT_SERVER_URL),
            api_key=env.get("SYSTEMONE_API_KEY") or None,
            model=env.get("SYSTEMONE_MODEL") or None,
        )
    if backend == "local":
        return LocalLayaClient()
    raise ValueError(f"unknown SYSTEMONE_BACKEND {backend!r}; expected gateway, server, or local")
