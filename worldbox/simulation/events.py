"""World history: typed events and a bounded event log.

The event log is the main read-only channel between the simulation and any
frontend. Subscribers can also be registered so a future GUI can react to events
live without polling.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Deque, Dict, Iterable, List, Optional


class EventKind(Enum):
    """Category of a recorded event."""

    BIRTH = "birth"
    DEATH = "death"
    FOOD = "food"
    DISCOVERY = "discovery"
    MILESTONE = "milestone"
    SOCIETY = "society"  # Tribes founded, split, or died out.
    INVENTION = "invention"  # A tribe invented a technology.
    WAR = "war"  # Wars declared, battles fought, peace made.
    PLAGUE = "plague"  # Outbreaks and their end.
    SYSTEM = "system"
    ERROR = "error"


@dataclass(frozen=True)
class Event:
    """One thing that happened on one day."""

    day: int
    kind: EventKind
    message: str
    agent_id: Optional[int] = None

    def __str__(self) -> str:
        return f"[Day {self.day}] {self.message}"


EventListener = Callable[[Event], None]


class EventLog:
    """A bounded, append-only history with optional live subscribers."""

    def __init__(self, capacity: int = 500) -> None:
        if capacity < 1:
            raise ValueError("Event log capacity must be at least 1.")
        self._events: Deque[Event] = deque(maxlen=capacity)
        self._listeners: List[EventListener] = []
        self._counts: Dict[EventKind, int] = {kind: 0 for kind in EventKind}

    def record(
        self,
        day: int,
        kind: EventKind,
        message: str,
        agent_id: Optional[int] = None,
    ) -> Event:
        """Append an event and notify subscribers."""
        event = Event(day=day, kind=kind, message=message, agent_id=agent_id)
        self._events.append(event)
        self._counts[kind] += 1
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:  # A broken frontend must not stop the simulation.
                continue
        return event

    def subscribe(self, listener: EventListener) -> None:
        """Register a callback invoked for every future event."""
        self._listeners.append(listener)

    def unsubscribe(self, listener: EventListener) -> None:
        """Remove a previously registered callback (no-op if absent)."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def recent(
        self, count: int = 10, kinds: Optional[Iterable[EventKind]] = None
    ) -> List[Event]:
        """The most recent ``count`` events, newest last, optionally filtered."""
        if kinds is not None:
            wanted = set(kinds)
            selected = [event for event in self._events if event.kind in wanted]
        else:
            selected = list(self._events)
        return selected[-count:] if count > 0 else []

    def total_recorded(self, kind: EventKind) -> int:
        """Lifetime count of a kind, including events evicted from the buffer."""
        return self._counts[kind]

    def clear(self) -> None:
        """Drop all stored events and reset lifetime counters."""
        self._events.clear()
        self._counts = {kind: 0 for kind in EventKind}

    def __len__(self) -> int:
        return len(self._events)
