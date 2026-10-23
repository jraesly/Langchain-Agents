"""One-call agent factory."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from .decision import DecisionClient, decision_client_from_env
from .middleware import SystemOneRoutingMiddleware
from .models import build_chat_model
from .routing import DEFAULT_TIERS, ComplexityRouter, ModelTier
from .routing_log import RoutingLogger


def create_routed_agent(
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    decision_client: DecisionClient | None = None,
    tiers: Sequence[ModelTier] = DEFAULT_TIERS,
    min_confidence: float = 0.5,
    fallback_tier: str | None = None,
    server_side_fallbacks: bool = True,
    model_factory: Callable[[ModelTier], BaseChatModel] | None = None,
    log_path: str | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    **create_agent_kwargs: Any,
):
    """Build a LangChain agent whose model is chosen per request by a System One model.

    Args:
        tools: Tools for the agent, as for ``langchain.agents.create_agent``.
        system_prompt: System prompt for the agent.
        decision_client: Which decision model answers the routing question.
            Defaults to :func:`decision_client_from_env`.
        tiers: Complexity tiers, least to most capable.
        min_confidence: Below this, the decision is escalated one tier.
        fallback_tier: Tier used when the decision model fails. Defaults to
            the most capable tier.
        server_side_fallbacks: Enable Claude's server-side refusal fallback on
            the models that support it.
        model_factory: Override how tiers become chat models.
        log_path: Append one JSONL line per invocation (request, decision,
            answer) here. Feed it to ``systemone-label`` to build training and
            eval data for the decision model.
        middleware: Extra middleware, run after the routing middleware.
        **create_agent_kwargs: Passed through to ``create_agent`` (checkpointer,
            response_format, state_schema, ...).
    """
    client = decision_client or decision_client_from_env()
    router = ComplexityRouter(client, tiers, min_confidence=min_confidence, fallback_tier=fallback_tier)
    factory = model_factory or partial(build_chat_model, server_side_fallbacks=server_side_fallbacks)
    log = RoutingLogger(log_path) if log_path else None
    routing = SystemOneRoutingMiddleware(router, model_factory=factory, log=log)
    # The base model only matters if routing is somehow absent from state; use
    # the fallback tier so that path is still the safe one.
    base_model = routing.model_for(router.fallback_tier)
    return create_agent(
        base_model,
        tools,
        system_prompt=system_prompt,
        middleware=[routing, *middleware],
        **create_agent_kwargs,
    )
