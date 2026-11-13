"""Rolling in-memory log of recent notable events, shown in the live UI."""

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class EventLogEntry:
    timestamp: float
    text: str


class EventLog:
    def __init__(self, max_entries: int):
        self._entries: Deque[EventLogEntry] = deque(maxlen=max_entries)

    def add(self, text: str) -> None:
        self._entries.append(EventLogEntry(timestamp=time.time(), text=text))

    def recent(self):
        """Return entries newest-first."""
        return list(reversed(self._entries))
