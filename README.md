# systemone-router

A LangChain agent that sends every request through an open-source **System One
decision model** first, then hands the request to the Claude model that fits its
complexity. Easy requests go to Haiku, everyday ones to Sonnet, hard ones to Opus.

System One models (Laya, Jev, Microsoft Decision-1, ...) don't generate text. They
read a piece of text and answer typed questions about it in one forward pass,
returning calibrated probabilities. That makes them fast and cheap enough to sit in
front of every call. [Laya](https://huggingface.co/convaiinnovations/laya) from
ConvAI Innovations is the open-weight one (Apache 2.0), so it is the default here.
Any other System One model works through the same interface.

```
user message ──► System One model ──► tier ──► ChatAnthropic(tier.model, effort) ──► agent loop
                 "how hard is this?"   simple │ moderate │ complex
```

## How it works

1. `SystemOneRoutingMiddleware.before_agent` runs once per invocation. It sends the
   latest human message to the decision model as a `choice` question whose options
   are the tier descriptions, and stores the answer in agent state under `routing`.
2. `wrap_model_call` swaps the request's model for the chosen tier's model on every
   model call in that invocation, including calls after tool use. One invocation,
   one model.
3. Uncertain decisions escalate. If the chosen tier's probability is below
   `min_confidence` (default 0.5) the request moves one tier up. If the decision
   model is unreachable, the request goes to the fallback tier (the most capable one
   by default). Mistakes cost more in the cheap direction.

Default tiers:

| Tier     | Model              | Effort | Routed here when the request looks like |
|----------|--------------------|--------|------------------------------------------|
| simple   | `claude-haiku-5-5` | low    | greetings, short factual lookups, one-line answers |
| moderate | `claude-sonnet-5-5`| medium | everyday coding, drafting, well-specified multi-step tasks |
| complex  | `claude-opus-5-5`  | high   | open-ended analysis, hard debugging, design trade-offs, long agentic work |

Thinking stays at each model's default (adaptive). `effort` is the lever. On the
models that support it (Opus 5.5, Sonnet 5.5, Opus 5, Fable 5.1), Claude's
server-side refusal fallback is enabled by default; pass
`server_side_fallbacks=False` to turn it off.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add [local] to run Laya in-process (installs torch)
cp .env.example .env             # fill in keys
```

## Pick a System One backend

Set `SYSTEMONE_BACKEND` in `.env`:

| Backend   | What it talks to | Needs |
|-----------|------------------|-------|
| `gateway` (default) | Vercel AI Gateway `POST /v1/evaluate`. Any model with the gateway's *Decision* capability; default `convaiinnovations/laya`, free there through 2026-10-31 | `AI_GATEWAY_API_KEY` |
| `server`  | Any Jev-compatible `POST /v1/systemone` server, e.g. `pip install "laya[serve]" && laya-serve` | `SYSTEMONE_SERVER_URL` (default `http://localhost:8000`) |
| `local`   | The `laya` package in-process | `pip install "systemone-router[local]"` |

To use a different decision model on the gateway, set `SYSTEMONE_MODEL`
(for example `typesafe-ai/jev`). To plug in something else entirely, implement the
two-method `DecisionClient` protocol in `src/systemone_router/decision.py`.

## Use

```python
from langchain_core.messages import HumanMessage
from systemone_router import ROUTING_STATE_KEY, create_routed_agent

agent = create_routed_agent(tools=[...], system_prompt="You are a helpful assistant.")

result = agent.invoke({"messages": [HumanMessage("Explain CAP theorem trade-offs for a payments ledger")]})
print(result[ROUTING_STATE_KEY])
# {'tier': 'complex', 'model': 'claude-opus-5-5', 'effort': 'high', 'reason': 'decision',
#  'probabilities': {'simple': 0.02, 'moderate': 0.21, 'complex': 0.77}, 'confidence': 0.77, ...}
print(result["messages"][-1].text)
```

`create_routed_agent` accepts everything `langchain.agents.create_agent` does
(`checkpointer`, `response_format`, extra `middleware`, ...), plus:

- `tiers` — your own ladder of `ModelTier(name, model, description, effort, max_tokens)`.
  The `description` is what the decision model reads, so write it as concrete examples.
- `min_confidence`, `fallback_tier`, `server_side_fallbacks`
- `decision_client` — an explicit backend instead of reading the environment.

Or compose the pieces yourself:

```python
from systemone_router import ComplexityRouter, SystemOneRoutingMiddleware, VercelGatewayDecisionClient
from langchain.agents import create_agent

router = ComplexityRouter(VercelGatewayDecisionClient(api_key), min_confidence=0.6)
middleware = SystemOneRoutingMiddleware(router)
agent = create_agent(middleware.model_for(router.fallback_tier), tools, middleware=[middleware])
```

`ComplexityRouter.route(text)` is also usable on its own, outside an agent, whenever
you want a `RoutingDecision` for a string.

## Try it

```bash
python examples/run_agent.py      # four prompts of rising difficulty; prints tier, model, answer
pytest                            # offline: fake decision model + fake chat models
```

## Notes

- Laya's English checkpoint reads about 512 tokens, so inputs are condensed to
  `max_state_chars` (default 1500) before routing, keeping the start and end of the
  message.
- The routing decision is tier-level only. If you want the decision model to also
  answer "does this need tools?" or "is this on-policy?", add questions to the same
  request; one forward pass answers them all.
- The shipped Laya checkpoints are reported as somewhat over-confident. If routing
  skews, tune `min_confidence` or the tier descriptions against a sample of your own
  traffic before changing models.

## Layout

```
src/systemone_router/
  decision.py    DecisionClient protocol + gateway / server / local backends
  routing.py     ModelTier, DEFAULT_TIERS, ComplexityRouter, RoutingDecision
  models.py      build_chat_model(tier) -> ChatAnthropic
  middleware.py  SystemOneRoutingMiddleware (before_agent + wrap_model_call)
  agent.py       create_routed_agent(...)
examples/run_agent.py
tests/
```
