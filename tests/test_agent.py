from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from systemone_router import (
    ROUTING_STATE_KEY,
    ComplexityRouter,
    ModelTier,
    SystemOneRoutingMiddleware,
    create_routed_agent,
)


class EchoTierModel(GenericFakeChatModel):
    """Fake chat model that answers with the name of the tier it was built for."""

    tier_name: str = ""
    call_count: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.call_count += 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return self


def make_factory(built: dict[str, EchoTierModel]):
    def factory(tier: ModelTier) -> EchoTierModel:
        model = EchoTierModel(
            tier_name=tier.name,
            messages=iter([AIMessage(f"answered by {tier.name}")] * 50),
        )
        built[tier.name] = model
        return model

    return factory


def test_agent_uses_tier_model_and_exposes_routing(fake_client_factory):
    built: dict[str, EchoTierModel] = {}
    agent = create_routed_agent(
        tools=[],
        decision_client=fake_client_factory("simple", {"simple": 0.9, "moderate": 0.05, "complex": 0.05}),
        model_factory=make_factory(built),
    )
    result = agent.invoke({"messages": [HumanMessage("hi")]})

    assert result["messages"][-1].content == "answered by simple"
    assert built["simple"].call_count == 1
    assert built["complex"].call_count == 0  # base model built for fallback, never called
    assert result[ROUTING_STATE_KEY]["tier"] == "simple"
    assert result[ROUTING_STATE_KEY]["model"] == "claude-haiku-5-5"


async def test_agent_async_path(fake_client_factory):
    built: dict[str, EchoTierModel] = {}
    agent = create_routed_agent(
        tools=[],
        decision_client=fake_client_factory("moderate"),
        model_factory=make_factory(built),
    )
    result = await agent.ainvoke({"messages": [HumanMessage("summarise this doc")]})
    assert result["messages"][-1].content == "answered by moderate"
    assert result[ROUTING_STATE_KEY]["tier"] == "moderate"


def test_same_model_used_across_tool_loop(fake_client_factory):
    @tool
    def lookup(x: str) -> str:
        """Look something up."""
        return f"looked up {x}"

    built: dict[str, EchoTierModel] = {}

    def factory(tier: ModelTier) -> EchoTierModel:
        responses = [
            AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"x": "a"}, "id": "call_1", "type": "tool_call"}]),
            AIMessage(f"done by {tier.name}"),
        ]
        model = EchoTierModel(tier_name=tier.name, messages=iter(responses))
        built[tier.name] = model
        return model

    agent = create_routed_agent(
        tools=[lookup],
        decision_client=fake_client_factory("complex"),
        model_factory=factory,
    )
    result = agent.invoke({"messages": [HumanMessage("research a then report")]})

    assert result["messages"][-1].content == "done by complex"
    assert built["complex"].call_count == 2
    assert "simple" not in built  # models are built lazily, only for tiers that are used


def test_routes_on_latest_human_message(fake_client_factory):
    client = fake_client_factory("simple")
    built: dict[str, EchoTierModel] = {}
    agent = create_routed_agent(tools=[], decision_client=client, model_factory=make_factory(built))
    agent.invoke({"messages": [HumanMessage("first"), AIMessage("reply"), HumanMessage("second question")]})
    assert client.calls[-1][0] == "second question"


def test_decision_failure_routes_to_fallback(fake_client_factory):
    built: dict[str, EchoTierModel] = {}
    agent = create_routed_agent(
        tools=[],
        decision_client=fake_client_factory("simple", fail=True),
        model_factory=make_factory(built),
    )
    result = agent.invoke({"messages": [HumanMessage("hi")]})
    assert result["messages"][-1].content == "answered by complex"
    assert result[ROUTING_STATE_KEY]["reason"] == "decision_model_error"


def test_middleware_model_cache_is_per_tier(fake_client_factory):
    built: dict[str, EchoTierModel] = {}
    mw = SystemOneRoutingMiddleware(ComplexityRouter(fake_client_factory("simple")), model_factory=make_factory(built))
    tier = mw.router.tier("moderate")
    assert mw.model_for(tier) is mw.model_for(tier)
    assert list(built) == ["moderate"]
