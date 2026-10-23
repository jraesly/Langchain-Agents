"""Send a few prompts of different difficulty through the routed agent.

Needs ANTHROPIC_API_KEY plus a System One backend (see .env.example).
"""

from __future__ import annotations

import datetime as dt

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from systemone_router import ROUTING_STATE_KEY, create_routed_agent


@tool
def current_time() -> str:
    """Return the current UTC time."""
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


PROMPTS = [
    "hi there!",
    "What time is it right now?",
    "Write a Python function that parses ISO-8601 dates and returns the weekday name.",
    (
        "We run a multi-region Postgres setup with async replicas and are seeing "
        "occasional stale reads after failover. Walk through the likely causes, how to "
        "confirm each one, and the trade-offs of the fixes."
    ),
]


def main() -> None:
    load_dotenv()
    agent = create_routed_agent(
        tools=[current_time, word_count],
        system_prompt="You are a concise, helpful assistant.",
    )
    for prompt in PROMPTS:
        result = agent.invoke({"messages": [HumanMessage(prompt)]})
        routing = result[ROUTING_STATE_KEY]
        print("=" * 78)
        print(f"PROMPT : {prompt[:70]}{'...' if len(prompt) > 70 else ''}")
        print(
            f"ROUTED : {routing['tier']} -> {routing['model']} "
            f"(effort={routing['effort']}, confidence={routing['confidence']:.2f}, reason={routing['reason']})"
        )
        print(f"ANSWER : {result['messages'][-1].text[:300]}")


if __name__ == "__main__":
    main()
