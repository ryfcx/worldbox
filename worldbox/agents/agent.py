"""The agent: identity, state, and a small personal memory.

:class:`Agent` is deliberately close to plain data. Decision-making lives in
:mod:`worldbox.agents.behavior` and world mutation lives in the engine, so an
agent can be inspected, serialised or copied without dragging along behaviour.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

from ..config import AgentConfig
from .naming import full_name
from .needs import Needs

if TYPE_CHECKING:  # Type-only import; avoids an agents <-> society import cycle.
    from ..society.epidemics import Infection

Coord = Tuple[int, int]



@dataclass
class Memory:
    """What an agent personally knows about the world.

    Intentionally small for this first version: remembered food locations, the
    terrain types it has encountered, and a short personal event log.
    """

    birthplace: Coord = (0, 0)
    known_food: Dict[Coord, int] = field(default_factory=dict)
    known_terrain: set[str] = field(default_factory=set)
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=8))

    def remember_food(self, location: Coord, day: int, config: AgentConfig) -> None:
        """Record a food location, evicting the stalest entry when full."""
        self.known_food[location] = day
        if len(self.known_food) > config.memory_capacity:
            oldest = min(self.known_food, key=self.known_food.__getitem__)
            del self.known_food[oldest]

    def forget_food(self, location: Coord) -> None:
        """Drop a location that turned out to be empty."""
        self.known_food.pop(location, None)

    def forget_stale_food(self, day: int, config: AgentConfig) -> None:
        """Drop memories older than ``memory_decay_days``."""
        cutoff = day - config.memory_decay_days
        for location in [loc for loc, seen in self.known_food.items() if seen < cutoff]:
            del self.known_food[location]

    def note(self, day: int, message: str) -> None:
        """Append a line to this agent's personal history."""
        self.log.append(f"Day {day}: {message}")


@dataclass
class Agent:
    """A single autonomous inhabitant of the world."""

    id: int
    name: str
    x: int
    y: int
    age_days: int
    lifespan_days: int
    needs: Needs = field(default_factory=Needs)
    memory: Memory = field(default_factory=Memory)
    goal: str = "wander"  # Current goal name; see behavior.Goal.
    alive: bool = True
    cause_of_death: Optional[str] = None
    parents: Tuple[Optional[int], Optional[int]] = (None, None)
    children: List[int] = field(default_factory=list)
    last_birth_day: int = -10**9  # Effectively "never" until first child.
    birth_day: int = 0

    # -- society ------------------------------------------------------------
    group_id: Optional[int] = None  # Tribe membership; None = unaffiliated.
    role: str = "forager"  # Profession within the tribe; see roles.Role.
    family_name: str = ""  # Inherited surname, so lineages are visible.
    aggression: float = 0.35  # 0.0 peaceful .. 1.0 warlike; inherited.
    kills: int = 0
    wounded_on_day: int = -10**9  # Last day this agent took a wound.
    wounded_by_group: Optional[int] = None  # Tribe that inflicted it.

    # -- illness ------------------------------------------------------------
    infection: Optional["Infection"] = None
    outbreak_id: Optional[int] = None
    # Disease id -> day the immunity expires (or None for permanent immunity).
    immunities: Dict[str, Optional[int]] = field(default_factory=dict)

    # -- derived properties -------------------------------------------------

    @property
    def position(self) -> Coord:
        """Current ``(x, y)`` position."""
        return (self.x, self.y)

    def age_years(self, days_per_year: int) -> float:
        """Age converted to years for display."""
        return self.age_days / days_per_year

    def is_adult(self, config: AgentConfig, days_per_year: int) -> bool:
        """True once old enough to reproduce."""
        return self.age_days >= config.adult_age_years * days_per_year

    def distance_to(self, x: int, y: int) -> int:
        """Chebyshev distance, matching 8-way movement."""
        return max(abs(self.x - x), abs(self.y - y))

    @property
    def full_name(self) -> str:
        """Given name plus family name, e.g. ``"Kaelira Stonewarden"``."""
        return f"{self.name} {self.family_name}".strip()

    @property
    def is_ill(self) -> bool:
        """True while the agent is carrying a disease."""
        return self.infection is not None

    # -- illness ------------------------------------------------------------

    def is_susceptible_to(self, disease_id: str, day: int) -> bool:
        """True if this agent can catch the given disease right now."""
        if not self.alive or self.infection is not None:
            return False
        if disease_id not in self.immunities:
            return True
        expires = self.immunities[disease_id]
        if expires is None:  # Permanent immunity.
            return False
        if day >= expires:
            del self.immunities[disease_id]
            return True
        return False

    def infect(
        self, disease_id: str, duration_days: int, day: int, outbreak_id: Optional[int]
    ) -> None:
        """Give the agent a disease (the caller checks susceptibility first)."""
        from ..society.epidemics import Infection  # Imported late to avoid a cycle.

        self.infection = Infection(
            disease_id=disease_id, days_left=duration_days, caught_on_day=day
        )
        self.outbreak_id = outbreak_id

    def recover_from_illness(self, day: int, immunity_days: int) -> None:
        """Clear the current illness and grant immunity to it."""
        if self.infection is None:
            return
        disease_id = self.infection.disease_id
        self.immunities[disease_id] = None if immunity_days <= 0 else day + immunity_days
        self.infection = None
        self.outbreak_id = None
        self.memory.note(day, f"recovered from {disease_id}")

    # -- mutation -----------------------------------------------------------

    def move_to(self, x: int, y: int) -> None:
        """Place the agent on a new tile (validity is the caller's job)."""
        self.x = x
        self.y = y

    def die(self, cause: str, day: int) -> None:
        """Mark the agent dead and record why."""
        self.alive = False
        self.cause_of_death = cause
        self.memory.note(day, f"died of {cause}")


def create_agent(
    agent_id: int,
    x: int,
    y: int,
    age_days: int,
    rng: random.Random,
    config: AgentConfig,
    days_per_year: int,
    day: int = 0,
    parents: Tuple[Optional[int], Optional[int]] = (None, None),
    needs: Optional[Needs] = None,
    group_id: Optional[int] = None,
    inherited_aggression: Optional[float] = None,
    naming_style: Optional[int] = None,
    inherited_family: Optional[str] = None,
) -> Agent:
    """Construct an agent with a randomised name, lifespan and temperament.

    Children inherit their parents' average aggression plus a little drift, so
    a tribe's temperament develops over generations rather than being fixed.
    """
    lifespan_years = max(
        config.lifespan_min_years,
        rng.gauss(config.lifespan_mean_years, config.lifespan_stddev_years),
    )
    if inherited_aggression is None:
        aggression = rng.gauss(config.aggression_mean, config.aggression_stddev)
    else:
        aggression = rng.gauss(inherited_aggression, config.aggression_inheritance_drift)
    aggression = max(0.0, min(1.0, aggression))

    given, family = full_name(rng, naming_style, inherited_family)
    agent = Agent(
        id=agent_id,
        name=given,
        x=x,
        y=y,
        age_days=age_days,
        lifespan_days=int(lifespan_years * days_per_year),
        needs=needs if needs is not None else Needs(),
        memory=Memory(birthplace=(x, y), log=deque(maxlen=config.personal_log_size)),
        parents=parents,
        birth_day=day,
        group_id=group_id,
        family_name=family,
        aggression=aggression,
    )
    return agent
