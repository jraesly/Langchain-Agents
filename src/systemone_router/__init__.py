"""Route LangChain agent calls through a System One decision model to a right-sized Claude model."""

from .agent import create_routed_agent
from .decision import (
    ChoiceDecision,
    DecisionClient,
    DecisionError,
    LocalLayaClient,
    SystemOneServerClient,
    VercelGatewayDecisionClient,
    decision_client_from_env,
)
from .middleware import ROUTING_STATE_KEY, RoutedAgentState, SystemOneRoutingMiddleware
from .models import build_chat_model
from .routing import DEFAULT_TIERS, ComplexityRouter, ModelTier, RoutingDecision

__all__ = [
    "DEFAULT_TIERS",
    "ROUTING_STATE_KEY",
    "ChoiceDecision",
    "ComplexityRouter",
    "DecisionClient",
    "DecisionError",
    "LocalLayaClient",
    "ModelTier",
    "RoutedAgentState",
    "RoutingDecision",
    "SystemOneRoutingMiddleware",
    "SystemOneServerClient",
    "VercelGatewayDecisionClient",
    "build_chat_model",
    "create_routed_agent",
    "decision_client_from_env",
]
