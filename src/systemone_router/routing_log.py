"""Append-only JSONL log of routing decisions and outcomes.

Each line is one agent invocation::

    {"ts": "...", "text": "<condensed request>", "routing": {...}, "answer": "...", "message_count": 4}

The log is the raw material for the labeling pipeline in ``labeling.py`` and
for spotting drift in tier mix over time.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RoutingLogger:
    """Thread-safe JSONL appender."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, entry: dict[str, Any]) -> None:
        line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def read_routing_log(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield entries from a routing log, skipping blank or corrupt lines."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
