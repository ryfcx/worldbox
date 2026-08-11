"""The Chronicle: a permanent record of a civilisation's turning points.

The :class:`~worldbox.simulation.events.EventLog` is a ring buffer of everything
that happens, so after a few thousand days the early history has scrolled away.
That is fine for a live feed and useless for watching a civilisation develop.

The Chronicle keeps only *turning points* -- first inventions, settlements
founded and grown, wars, plagues, population milestones -- and keeps them
forever. It is small enough to hold a hundred thousand days of history and is
what the ``chronicle`` command reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence


class Milestone(Enum):
    """The kind of turning point being recorded."""

    ERA = "era"  # The world entered a new technological era.
    FIRST_TECH = "first_tech"  # A technology was invented for the first time.
    SETTLEMENT = "settlement"  # A settlement was founded or grew.
    POPULATION = "population"  # A population landmark was passed.
    WAR = "war"
    PLAGUE = "plague"
    TRIBE = "tribe"  # A tribe was founded or died out.
    COLLAPSE = "collapse"  # Something was lost.

    @property
    def label(self) -> str:
        """Human-readable milestone kind."""
        return self.value.replace("_", " ").title()


@dataclass(frozen=True)
class ChronicleEntry:
    """One permanent line of history."""

    day: int
    year: int
    kind: Milestone
    message: str

    def __str__(self) -> str:
        return f"Year {self.year:>4} (day {self.day:>6}) | {self.message}"


# Population landmarks worth recording, in ascending order.
POPULATION_LANDMARKS: Sequence[int] = (150, 250, 500, 750, 1000, 1500, 2500, 5000)


@dataclass
class Chronicle:
    """The permanent history of a world."""

    entries: List[ChronicleEntry] = field(default_factory=list)

    # Internal bookkeeping so each landmark is only recorded once.
    _seen_techs: set = field(default_factory=set, repr=False)
    _seen_eras: set = field(default_factory=set, repr=False)
    _peak_population: int = field(default=0, repr=False)
    _seen_landmarks: set = field(default_factory=set, repr=False)
    _seen_settlement_levels: Dict[int, int] = field(default_factory=dict, repr=False)

    def record(self, day: int, year: int, kind: Milestone, message: str) -> ChronicleEntry:
        """Add a line to history."""
        entry = ChronicleEntry(day=day, year=year, kind=kind, message=message)
        self.entries.append(entry)
        return entry

    # -- landmark detectors -------------------------------------------------

    def note_invention(
        self, day: int, year: int, tribe: str, tech_id: str, tech_name: str, era: str
    ) -> Optional[ChronicleEntry]:
        """Record a technology the first time anyone in the world invents it."""
        if tech_id in self._seen_techs:
            return None
        self._seen_techs.add(tech_id)
        entry = self.record(
            day, year, Milestone.FIRST_TECH, f"The {tribe} invented {tech_name}"
        )
        if era not in self._seen_eras:
            self._seen_eras.add(era)
            self.record(day, year, Milestone.ERA, f"The world entered the {era}")
        return entry

    def note_population(self, day: int, year: int, population: int) -> Optional[ChronicleEntry]:
        """Record the first time the population passes a landmark."""
        self._peak_population = max(self._peak_population, population)
        for landmark in POPULATION_LANDMARKS:
            if population >= landmark and landmark not in self._seen_landmarks:
                self._seen_landmarks.add(landmark)
                return self.record(
                    day, year, Milestone.POPULATION, f"World population passed {landmark:,}"
                )
        return None

    def note_settlement(
        self, day: int, year: int, settlement_id: int, name: str, level: int, level_name: str,
        tribe: str,
    ) -> Optional[ChronicleEntry]:
        """Record a settlement being founded or reaching a new level."""
        if self._seen_settlement_levels.get(settlement_id, -1) >= level:
            return None
        self._seen_settlement_levels[settlement_id] = level
        if level == 0:
            return self.record(
                day, year, Milestone.SETTLEMENT, f"The {tribe} founded {name}"
            )
        return self.record(
            day, year, Milestone.SETTLEMENT, f"{name} grew into a {level_name}"
        )

    # -- reading ------------------------------------------------------------

    def recent(self, count: int = 20, kinds: Optional[Sequence[Milestone]] = None) -> List[ChronicleEntry]:
        """The most recent entries, newest last, optionally filtered by kind."""
        if kinds is not None:
            wanted = set(kinds)
            selected = [entry for entry in self.entries if entry.kind in wanted]
        else:
            selected = self.entries
        return selected[-count:] if count > 0 else []

    def timeline(self) -> List[ChronicleEntry]:
        """Just the era advances -- the spine of the world's history."""
        return [entry for entry in self.entries if entry.kind is Milestone.ERA]

    @property
    def peak_population(self) -> int:
        """The largest population the world has ever held."""
        return self._peak_population

    def clear(self) -> None:
        """Wipe all history (used when the simulation resets)."""
        self.entries.clear()
        self._seen_techs.clear()
        self._seen_eras.clear()
        self._seen_landmarks.clear()
        self._seen_settlement_levels.clear()
        self._peak_population = 0

    def __len__(self) -> int:
        return len(self.entries)
