"""Settlements: the visible shape of a civilisation on the map.

A tribe that grows large enough stops being a roaming band and founds a fixed
settlement. From then on the settlement -- not the wandering centroid -- is the
tribe's home, so its people cluster around it and hold real territory.

Settlements climb five levels, each gated on both population and technology:

    Camp -> Hamlet -> Village -> Town -> City

They also hold a **food store**. Farmers harvest surrounding tiles into it, and
hungry members draw on it when the ground nearby is bare. That granary is what
lets a settlement survive a bad season, and its collapse is what turns a bad
season into a famine.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

from ..agents.agent import Agent
from ..agents.naming import settlement_name
from ..config import SettlementConfig
from .groups import Group
from .technology import Knowledge


class SettlementLevel(IntEnum):
    """How far a settlement has developed."""

    CAMP = 0
    HAMLET = 1
    VILLAGE = 2
    TOWN = 3
    CITY = 4


@dataclass(frozen=True)
class LevelSpec:
    """What a settlement level requires and what it grants."""

    level: SettlementLevel
    name: str
    min_population: int
    required_tech: Tuple[str, ...]
    store_capacity: float
    research_bonus: float
    fertility_bonus: float


# Each level demands both people and knowledge -- a tribe cannot build a city
# by breeding alone.
LEVEL_SPECS: Tuple[LevelSpec, ...] = (
    LevelSpec(SettlementLevel.CAMP, "Camp", 0, (), 60.0, 0.00, 0.00),
    LevelSpec(SettlementLevel.HAMLET, "Hamlet", 10, ("shelter",), 150.0, 0.05, 0.05),
    LevelSpec(SettlementLevel.VILLAGE, "Village", 18, ("agriculture",), 400.0, 0.12, 0.12),
    LevelSpec(SettlementLevel.TOWN, "Town", 30, ("pottery", "writing"), 900.0, 0.25, 0.18),
    LevelSpec(SettlementLevel.CITY, "City", 45, ("mathematics", "engineering"), 2000.0, 0.40, 0.25),
)

SPEC_BY_LEVEL: Dict[SettlementLevel, LevelSpec] = {spec.level: spec for spec in LEVEL_SPECS}


@dataclass
class Settlement:
    """A tribe's permanent home."""

    id: int
    group_id: int
    name: str
    x: int
    y: int
    founded_day: int
    level: SettlementLevel = SettlementLevel.CAMP
    food_store: float = 0.0
    last_upgraded_day: int = 0
    peak_level: SettlementLevel = SettlementLevel.CAMP
    peak_population: int = 0

    @property
    def spec(self) -> LevelSpec:
        """The rules for this settlement's current level."""
        return SPEC_BY_LEVEL[self.level]

    @property
    def position(self) -> Tuple[int, int]:
        """Where the settlement stands."""
        return (self.x, self.y)

    @property
    def level_name(self) -> str:
        """Human-readable level, e.g. ``"Village"``."""
        return self.spec.name

    def store_fraction(self) -> float:
        """How full the granary is, 0..1."""
        capacity = self.spec.store_capacity
        return self.food_store / capacity if capacity else 0.0


class SettlementSystem:
    """Owns every settlement and advances them one day at a time."""

    def __init__(self, config: SettlementConfig) -> None:
        self.config = config
        self.settlements: Dict[int, Settlement] = {}
        self.by_group: Dict[int, int] = {}
        self.abandoned: List[Settlement] = []
        self._next_id = 1

    # -- lookup -------------------------------------------------------------

    def for_group(self, group_id: Optional[int]) -> Optional[Settlement]:
        """The settlement belonging to a tribe, if it has founded one."""
        if group_id is None:
            return None
        settlement_id = self.by_group.get(group_id)
        return self.settlements.get(settlement_id) if settlement_id is not None else None

    def active(self) -> List[Settlement]:
        """Every standing settlement, ordered by id."""
        return [self.settlements[sid] for sid in sorted(self.settlements)]

    def largest(self) -> Optional[Settlement]:
        """The most developed settlement in the world."""
        active = self.active()
        if not active:
            return None
        return max(active, key=lambda s: (int(s.level), s.food_store))

    # -- founding and growth ------------------------------------------------

    def try_found(
        self, group: Group, world, day: int, rng: random.Random
    ) -> Optional[Settlement]:
        """Found a settlement for a tribe that has grown large enough."""
        if not self.config.enabled or group.size < self.config.min_population_to_found:
            return None
        if self.for_group(group.id) is not None:
            return None

        x, y = group.centre
        if not world.is_passable(x, y):
            # Settle the nearest walkable ground instead.
            options = world.neighbours(x, y)
            if not options:
                return None
            x, y = options[0]

        settlement = Settlement(
            id=self._next_id,
            group_id=group.id,
            name=settlement_name(rng, group.naming_style),
            x=x,
            y=y,
            founded_day=day,
            peak_population=group.size,
        )
        self.settlements[settlement.id] = settlement
        self.by_group[group.id] = settlement.id
        self._next_id += 1
        return settlement

    def try_upgrade(
        self, settlement: Settlement, group: Group, knowledge: Knowledge, day: int
    ) -> Optional[LevelSpec]:
        """Advance a settlement one level if it now qualifies."""
        next_level = SettlementLevel(min(int(settlement.level) + 1, int(SettlementLevel.CITY)))
        if next_level == settlement.level:
            return None
        spec = SPEC_BY_LEVEL[next_level]
        if group.size < spec.min_population:
            return None
        if not all(knowledge.knows(tech) for tech in spec.required_tech):
            return None
        # A settlement must be able to feed itself before it grows.
        if settlement.store_fraction() < self.config.upgrade_store_fraction:
            return None

        settlement.level = next_level
        settlement.last_upgraded_day = day
        settlement.peak_level = max(settlement.peak_level, next_level)
        return spec

    def abandon(self, group_id: int, day: int) -> Optional[Settlement]:
        """Abandon a tribe's settlement, e.g. when the tribe dies out."""
        settlement_id = self.by_group.pop(group_id, None)
        if settlement_id is None:
            return None
        settlement = self.settlements.pop(settlement_id, None)
        if settlement is not None:
            self.abandoned.append(settlement)
        return settlement

    # -- the granary --------------------------------------------------------

    def harvest(
        self,
        settlement: Settlement,
        farmers: Sequence[Agent],
        world,
        knowledge: Knowledge,
    ) -> float:
        """Farmers gather food from around the settlement into its store.

        Food is *moved* from the map into the granary, never created, so a
        settlement can still farm its surroundings bare.
        """
        if not farmers:
            return 0.0
        spec = settlement.spec
        capacity = spec.store_capacity
        room = capacity - settlement.food_store
        if room <= 0:
            return 0.0

        efficiency = 1.0 + knowledge.effects.food_yield
        wanted = min(room, len(farmers) * self.config.harvest_per_farmer)
        gathered = 0.0
        for x, y in world.tiles_within(
            settlement.x, settlement.y, self.config.harvest_radius
        ):
            if gathered >= wanted:
                break
            gathered += world.resources.take(x, y, self.config.harvest_per_tile)

        stored = min(room, gathered * efficiency)
        settlement.food_store += stored
        return stored

    def cultivate(
        self,
        settlement: Settlement,
        farmers: Sequence[Agent],
        world,
        knowledge: Knowledge,
    ) -> float:
        """Farmers improve the land around their settlement.

        This is the mechanism by which technology raises carrying capacity:
        better farming knowledge lifts the fertility ceiling, cultivated tiles
        both hold and regrow more food, and the population that land can
        support rises with it.
        """
        if not farmers:
            return 0.0
        config = self.config
        ceiling = config.max_fertility + knowledge.effects.food_yield * config.fertility_per_tech
        tiles = list(world.tiles_within(settlement.x, settlement.y, config.cultivation_radius))
        if not tiles:
            return 0.0

        # Effort is shared across the fields, so a hamlet improves its few tiles
        # quickly while a city spreads the same work thinner.
        effort = len(farmers) * config.cultivation_per_farmer
        per_tile = effort / len(tiles)
        for x, y in tiles:
            world.resources.cultivate(x, y, per_tile, ceiling)
        return effort

    def draw_ration(self, settlement: Settlement, amount: float) -> float:
        """Take a ration out of the granary, returning what was actually available."""
        taken = min(settlement.food_store, max(0.0, amount))
        settlement.food_store -= taken
        return taken

    def spoil(self, settlement: Settlement, knowledge: Knowledge) -> None:
        """Apply daily spoilage; pottery and storage tech slow it down."""
        rate = self.config.spoilage_rate * (
            1.0 - min(0.8, knowledge.effects.disease_resistance)
        )
        settlement.food_store = max(0.0, settlement.food_store * (1.0 - rate))
