"""LangChain agent middleware that picks the model per request."""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

from .models import build_chat_model
from .routing import ComplexityRouter, ModelTier, RoutingDecision
from .routing_log import RoutingLogger

ROUTING_STATE_KEY = "routing"


class RoutedAgentState(AgentState):
    """Agent state plus the routing decision for the current invocation."""

    routing: NotRequired[dict[str, Any]]


def latest_human_text(messages: Sequence[AnyMessage]) -> str:
    """Text of the most recent human message, or an empty string."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


class SystemOneRoutingMiddleware(AgentMiddleware[RoutedAgentState, Any]):
    """Route each agent invocation to a model sized for its complexity.

    ``before_agent`` runs once per invocation: it sends the latest human message
    to the decision model and stores the resulting :class:`RoutingDecision` in
    state under ``routing``. ``wrap_model_call`` then swaps the request's model
    for that tier's model on every model call of the invocation, including the
    calls that follow tool use, so one invocation sticks to one model.

    Args:
        router: The complexity router.
        model_factory: Builds a chat model for a tier. Defaults to
            :func:`build_chat_model`; tests inject fakes here.
        text_for_routing: Extracts the text the decision model should read
            from the agent state. Defaults to the latest human message.
        log: Optional :class:`RoutingLogger`. When set, every invocation
            appends one line with the request, the decision, and the answer.
    """

    state_schema = RoutedAgentState

    def __init__(
        self,
        router: ComplexityRouter,
        *,
        model_factory: Callable[[ModelTier], BaseChatModel] = build_chat_model,
        text_for_routing: Callable[[RoutedAgentState], str] | None = None,
        log: RoutingLogger | None = None,
    ) -> None:
        super().__init__()
        self.router = router
        self.log = log
        self._model_factory = model_factory
        self._text_for_routing = text_for_routing or (lambda state: latest_human_text(state["messages"]))
        self._models: dict[str, BaseChatModel] = {}
        self._lock = threading.Lock()

    # -- model cache -------------------------------------------------------

    def model_for(self, tier: ModelTier) -> BaseChatModel:
        """The (cached) chat model for ``tier``."""
        with self._lock:
            model = self._models.get(tier.name)
            if model is None:
                model = self._model_factory(tier)
                self._models[tier.name] = model
            return model

    # -- hooks -------------------------------------------------------------

    def before_agent(self, state: RoutedAgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        decision = self.router.route(self._text_for_routing(state))
        return {ROUTING_STATE_KEY: decision.to_dict()}

    async def abefore_agent(self, state: RoutedAgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        decision = await self.router.aroute(self._text_for_routing(state))
        return {ROUTING_STATE_KEY: decision.to_dict()}

    def after_agent(self, state: RoutedAgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        self._log(state)
        return None

    async def aafter_agent(self, state: RoutedAgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        self._log(state)
        return None

    def _log(self, state: RoutedAgentState) -> None:
        if self.log is None:
            return
        routing = dict(state.get(ROUTING_STATE_KEY) or {})
        messages = state.get("messages") or []
        self.log.record(
            {
                "text": routing.pop("state", "") or latest_human_text(messages),
                "routing": routing,
                "answer": messages[-1].text if messages else "",
                "message_count": len(messages),
            }
        )

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(self._apply_routing(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        return await handler(self._apply_routing(request))

    def _apply_routing(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        routing = request.state.get(ROUTING_STATE_KEY)
        if not routing:
            return request
        tier = self.router.tier(routing["tier"])
        return request.override(model=self.model_for(tier))

