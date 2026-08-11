"""Export a simulation run to JSON for visualisation.

The engine stays headless: this module *observes* a run and records compact
snapshots, which a viewer (the HTML canvas page, or any future GUI) can replay.
Nothing here feeds back into the simulation.

Positions are stored as flat integer arrays rather than objects -- a 300-agent
world recorded every 50 days for 20,000 days is a few hundred kilobytes that
way, and tens of megabytes as JSON objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .simulation.engine import SimulationEngine
from .world.terrain import TerrainType

# Terrain is exported as small integers; the viewer maps them to colours.
TERRAIN_CODES: Dict[str, int] = {
    TerrainType.WATER.value: 0,
    TerrainType.GRASS.value: 1,
    TerrainType.FOREST.value: 2,
    TerrainType.MOUNTAIN.value: 3,
}


@dataclass
class Frame:
    """One recorded moment in the world's history."""

    day: int
    year: int
    population: int
    # Flat triples: x, y, tribe_index (-1 when unaffiliated).
    agents: List[int] = field(default_factory=list)
    # Flat quads: x, y, level, tribe_index.
    settlements: List[int] = field(default_factory=list)
    tribes: int = 0
    techs: int = 0
    era: str = ""
    wars: int = 0
    ill: int = 0
    war_deaths: int = 0
    plague_deaths: int = 0
    food_fraction: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Compact dictionary form for JSON."""
        return {
            "d": self.day,
            "y": self.year,
            "p": self.population,
            "a": self.agents,
            "s": self.settlements,
            "t": self.tribes,
            "k": self.techs,
            "e": self.era,
            "w": self.wars,
            "i": self.ill,
            "wd": self.war_deaths,
            "pd": self.plague_deaths,
            "f": round(self.food_fraction, 3),
        }


class RunRecorder:
    """Records frames from a running simulation.

    Tribe identities are assigned stable indices as they appear, so the viewer
    can give each tribe a consistent colour across the whole run even though
    tribes are founded and die out constantly.
    """

    # Event kinds worth putting in a viewer's log; routine foraging is noise.
    LOGGED_KINDS = ("birth", "death", "war", "plague", "invention", "society", "milestone")

    def __init__(self, engine: SimulationEngine, every: int = 50, log_limit: int = 1200) -> None:
        self.engine = engine
        self.every = max(1, every)
        self.frames: List[Frame] = []
        self.tribe_index: Dict[int, int] = {}
        self.tribe_names: List[str] = []
        self.log: List[Dict[str, Any]] = []
        self.log_limit = log_limit
        # Subscribe once so the log is captured as the run happens, rather than
        # scraped afterwards from a ring buffer that has already dropped things.
        engine.events.subscribe(self._on_event)

    def _on_event(self, event) -> None:
        """Collect notable events as they are recorded."""
        if event.kind.value not in self.LOGGED_KINDS:
            return
        if len(self.log) >= self.log_limit:
            return
        self.log.append({"d": event.day, "k": event.kind.value, "m": event.message})

    def _index_for(self, group_id: Optional[int]) -> int:
        """Stable colour index for a tribe, allocated on first sight."""
        if group_id is None:
            return -1
        if group_id not in self.tribe_index:
            group = self.engine.groups.get(group_id)
            self.tribe_index[group_id] = len(self.tribe_names)
            self.tribe_names.append(group.name if group else f"Tribe {group_id}")
        return self.tribe_index[group_id]

    def capture(self) -> Frame:
        """Take a snapshot of the world right now."""
        engine = self.engine
        stats = engine.stats()

        agents: List[int] = []
        for agent in engine.agents:
            agents.extend((agent.x, agent.y, self._index_for(agent.group_id)))

        settlements: List[int] = []
        for settlement in engine.settlements.active():
            settlements.extend(
                (
                    settlement.x,
                    settlement.y,
                    int(settlement.level),
                    self._index_for(settlement.group_id),
                )
            )

        capacity = stats.food_capacity or 1.0
        frame = Frame(
            day=stats.day,
            year=stats.year,
            population=stats.population,
            agents=agents,
            settlements=settlements,
            tribes=stats.tribes,
            techs=stats.technologies_known,
            era=stats.most_advanced_era,
            wars=stats.active_wars,
            ill=stats.ill,
            war_deaths=stats.war_deaths,
            plague_deaths=stats.plague_deaths,
            food_fraction=stats.total_food / capacity,
        )
        self.frames.append(frame)
        return frame

    def run(self, days: int) -> None:
        """Simulate ``days`` days, capturing a frame every ``every`` days."""
        self.capture()
        remaining = days
        while remaining > 0 and self.engine.agents:
            chunk = min(self.every, remaining)
            self.engine.run(chunk)
            self.capture()
            remaining -= chunk

    # -- output -------------------------------------------------------------

    def terrain_rows(self) -> List[str]:
        """Terrain as one digit string per row -- compact and human-readable."""
        return [
            "".join(str(TERRAIN_CODES[tile.value]) for tile in row)
            for row in self.engine.world.grid
        ]

    def to_dict(self) -> Dict[str, Any]:
        """The complete recording, ready to be embedded in a viewer."""
        engine = self.engine
        return {
            "seed": engine.seed,
            "width": engine.world.width,
            "height": engine.world.height,
            "daysPerYear": engine.config.simulation.days_per_year,
            "terrain": self.terrain_rows(),
            "tribeNames": self.tribe_names,
            "frames": [frame.to_dict() for frame in self.frames],
            "log": self.log,
            "narration": getattr(self, "narration", []),
            "chronicle": [
                {"day": entry.day, "year": entry.year, "kind": entry.kind.value,
                 "message": entry.message}
                for entry in engine.chronicle.entries
            ],
            "finalTribes": [
                {
                    "name": group.name,
                    "size": group.size,
                    "era": group.era,
                    "techs": len(group.knowledge.known),
                    "index": self.tribe_index.get(group.id, -1),
                }
                for group in sorted(engine.groups.active(), key=lambda g: -g.size)
            ],
        }

    def write_json(self, path: Path) -> Path:
        """Write the recording to a JSON file and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")))
        return path


def record_run(
    engine: SimulationEngine, days: int, every: int = 50
) -> RunRecorder:
    """Convenience helper: record ``days`` days of ``engine`` and return the recorder."""
    recorder = RunRecorder(engine, every=every)
    recorder.run(days)
    return recorder
