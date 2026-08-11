"""Relations between tribes, and the wars that come of them.

Every pair of tribes has a relation score from -100 (blood feud) to +100
(close allies), starting at neutral. Three pressures move it:

    * **Territory** -- tribes whose centres are close compete for the same
      ground, and competition bites harder when the world is running short of
      food.
    * **Violence** -- every individual skirmish sours relations, so fighting
      can drag two tribes into a war neither side declared.
    * **Time** -- relations always drift back toward neutral, so grudges fade
      if the tribes drift apart.

When a relation falls past ``war_threshold`` the tribes are at war. A war ends
when relations recover, or when one side has lost enough of its members to be
exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import DiplomacyConfig
from .groups import Group, GroupRegistry

Pair = Tuple[int, int]


def pair_key(a: int, b: int) -> Pair:
    """Order-independent key for a pair of tribe ids."""
    return (a, b) if a < b else (b, a)


@dataclass
class War:
    """An active or concluded war between two tribes."""

    tribes: Pair
    started_day: int
    ended_day: Optional[int] = None
    casualties: Dict[int, int] = field(default_factory=dict)
    battles: int = 0

    @property
    def active(self) -> bool:
        """True while the war is still being fought."""
        return self.ended_day is None

    def total_casualties(self) -> int:
        """Dead on both sides."""
        return sum(self.casualties.values())

    def record_death(self, group_id: int) -> None:
        """Attribute one death to a side."""
        self.casualties[group_id] = self.casualties.get(group_id, 0) + 1


class Diplomacy:
    """Tracks relations between every pair of tribes, and their wars."""

    def __init__(self, config: DiplomacyConfig) -> None:
        self.config = config
        self.relations: Dict[Pair, float] = {}
        self.wars: Dict[Pair, War] = {}
        self.history: List[War] = []

    # -- relations ----------------------------------------------------------

    def relation(self, a: int, b: int) -> float:
        """Current relation score between two tribes."""
        if a == b:
            return 100.0
        return self.relations.get(pair_key(a, b), self.config.starting_relation)

    def adjust(self, a: int, b: int, delta: float) -> float:
        """Shift a relation and return the new value, clamped to -100..100."""
        if a == b:
            return 100.0
        key = pair_key(a, b)
        value = self.relations.get(key, self.config.starting_relation) + delta
        value = max(-100.0, min(100.0, value))
        self.relations[key] = value
        return value

    def at_war(self, a: Optional[int], b: Optional[int]) -> bool:
        """True if the two tribes are currently at war."""
        if a is None or b is None or a == b:
            return False
        war = self.wars.get(pair_key(a, b))
        return war is not None and war.active

    def are_hostile(self, a: Optional[int], b: Optional[int]) -> bool:
        """True if the tribes are at war or their relation is already sour."""
        if a is None or b is None or a == b:
            return False
        return self.at_war(a, b) or self.relation(a, b) <= self.config.hostility_threshold

    def forget_group(self, group_id: int) -> None:
        """Drop all relations involving a tribe that no longer exists."""
        for key in [key for key in self.relations if group_id in key]:
            del self.relations[key]
        for key, war in list(self.wars.items()):
            if group_id in key and war.active:
                war.ended_day = -1  # Ended by extinction rather than treaty.
                self.history.append(war)
                del self.wars[key]

    # -- daily update -------------------------------------------------------

    def update(
        self,
        registry: GroupRegistry,
        day: int,
        food_fraction: float,
    ) -> Tuple[List[War], List[War]]:
        """Advance relations one day.

        Returns ``(declared, concluded)`` -- the wars that started and ended
        today, so the engine can turn them into events.

        ``food_fraction`` is how full the world's larder is (0..1); scarcity
        makes every territorial dispute sharper.
        """
        declared: List[War] = []
        concluded: List[War] = []
        if not self.config.enabled:
            return declared, concluded

        groups = registry.active()
        scarcity = 1.0 + self.config.scarcity_multiplier * (1.0 - max(0.0, min(1.0, food_fraction)))

        for index, first in enumerate(groups):
            for second in groups[index + 1 :]:
                key = pair_key(first.id, second.id)
                distance = max(
                    abs(first.centre[0] - second.centre[0]),
                    abs(first.centre[1] - second.centre[1]),
                )

                # Grudges always fade, proportionally to how strong they are.
                # This is what stops every pair of neighbours from sliding
                # inexorably to -100 and staying permanently at war: relations
                # settle at an equilibrium set by how hard the tribes are
                # actually pressing on each other.
                current = self.relation(first.id, second.id)
                self.adjust(first.id, second.id, -current * self.config.neutral_drift)

                if distance <= self.config.territory_pressure_radius:
                    # Closer tribes press harder on each other.
                    closeness = 1.0 - distance / max(1, self.config.territory_pressure_radius)
                    self.adjust(
                        first.id,
                        second.id,
                        -self.config.pressure_per_day * closeness * scarcity,
                    )

                relation = self.relation(first.id, second.id)
                war = self.wars.get(key)

                if war is not None and war.active:
                    if self._should_end(war, first, second, relation, day):
                        war.ended_day = day
                        self.history.append(war)
                        del self.wars[key]
                        self.relations[key] = self.config.post_war_relation
                        concluded.append(war)
                elif relation <= self.config.war_threshold:
                    war = War(tribes=key, started_day=day)
                    self.wars[key] = war
                    declared.append(war)

        return declared, concluded

    def _should_end(
        self, war: War, first: Group, second: Group, relation: float, day: int
    ) -> bool:
        """True when a war has run its course."""
        if day - war.started_day < self.config.min_war_days:
            return False
        if relation >= self.config.peace_threshold:
            return True
        # War exhaustion: either side losing too large a share of its people.
        for group in (first, second):
            losses = war.casualties.get(group.id, 0)
            strength = group.size + losses
            if strength > 0 and losses / strength >= self.config.war_exhaustion_fraction:
                return True
        return False

    # -- reporting ----------------------------------------------------------

    def active_wars(self) -> List[War]:
        """Every war currently being fought, oldest first."""
        return sorted(self.wars.values(), key=lambda war: war.started_day)

    def relation_summary(self, group_id: int, registry: GroupRegistry) -> List[Tuple[Group, float]]:
        """This tribe's relations with every other, worst first."""
        others = [group for group in registry.active() if group.id != group_id]
        scored = [(group, self.relation(group_id, group.id)) for group in others]
        return sorted(scored, key=lambda item: item[1])
