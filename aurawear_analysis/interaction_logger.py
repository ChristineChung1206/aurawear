"""Append-only JSONL interaction logger for AuraWear sessions.

Each line in the log file is a self-contained JSON object with:
  - ts: ISO-8601 timestamp
  - event: event type string
  - session_id: user session
  - payload: event-specific data

Log path defaults to ``logs/interactions.jsonl`` (append mode).
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_DEFAULT_LOG_PATH = "logs/interactions.jsonl"

_lock = threading.Lock()


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def log_event(
    event: str,
    *,
    session_id: str = "",
    payload: Optional[Dict[str, Any]] = None,
    log_path: str = _DEFAULT_LOG_PATH,
) -> None:
    """Append one JSON line to the interaction log (thread-safe)."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event,
        "session_id": session_id,
        **(payload or {}),
    }
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    p = Path(log_path)
    with _lock:
        _ensure_dir(p)
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)


class Timer:
    """Simple context-manager timer that records elapsed seconds."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = round(time.perf_counter() - self._start, 3)
