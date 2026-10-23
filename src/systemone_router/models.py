"""Build the Claude chat model for a tier."""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from .routing import ModelTier

# Models that accept the server-side refusal fallback parameter on the Claude API.
SERVER_SIDE_FALLBACK_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)
SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def build_chat_model(
    tier: ModelTier,
    *,
    server_side_fallbacks: bool = True,
    **overrides: Any,
) -> BaseChatModel:
    """Create a ``ChatAnthropic`` configured for ``tier``.

    Thinking is left at the model default (adaptive on every current model);
    ``tier.effort`` sets ``output_config.effort`` to control how hard it thinks.
    When ``server_side_fallbacks`` is on and the model supports it, a request
    the safety classifiers decline is re-run server-side on a fallback model
    instead of surfacing a refusal.
    """
    kwargs: dict[str, Any] = {"model": tier.model, "max_tokens": tier.max_tokens}
    if tier.effort:
        kwargs["output_config"] = {"effort": tier.effort}
    if server_side_fallbacks and tier.model in SERVER_SIDE_FALLBACK_MODELS:
        kwargs["betas"] = [SERVER_SIDE_FALLBACK_BETA]
        kwargs["model_kwargs"] = {"fallbacks": "default"}
    kwargs.update(overrides)
    return ChatAnthropic(**kwargs)
