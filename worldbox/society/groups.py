"""Tribes: emergent groups of agents that hold territory and split apart.

Nothing here is pre-assigned. Agents begin unaffiliated; two who meet may found
a tribe, neighbours get recruited into it, and children inherit their parents'
tribe. Once a tribe outgrows :attr:`GroupConfig.split_size`, the members
furthest from its centre splinter off and found their own -- which is where the
map's patchwork of rival packs comes from.

A tribe's "territory" is simply the centroid of its living members plus a
radius; it is recomputed each day rather than stored as owned tiles, which keeps
territory fluid and cheap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..agents.agent import Agent
from ..agents.naming import STYLE_COUNT, tribe_name
from ..config import GroupConfig
from .technology import Knowledge

Coord = Tuple[int, int]

_TRIBE_PREFIXES = (
    "Ka", "Mor", "El", "Tar", "Sil", "Bran", "Ny", "Ors", "Vel", "Dun",
    "Ash", "Ith", "Ro", "Zan", "Ler", "Mir", "Ost", "Fen", "Ura", "Cal",
    "Grim", "Tho", "Wyn", "Skal", "Bor", "Hal", "Ver", "Dra", "Kor", "Ael",
)
_TRIBE_SUFFIXES = ("ni", "dar", "vek", "thi", "mar", "sk", "ra", "gan", "len", "tor")
_TRIBE_KINDS = ("Tribe", "Clan", "Folk", "Kin", "Host")


def generate_tribe_name(rng: random.Random) -> str:
    """Build a tribe name such as ``"Kaerani Clan"``."""
    return (
        rng.choice(_TRIBE_PREFIXES)
        + rng.choice(_TRIBE_SUFFIXES)
        + " "
        + rng.choice(_TRIBE_KINDS)
    )


@dataclass
class Group:
    """A tribe: a named group of agents with shared knowledge and territory."""

    id: int
    name: str
    founded_day: int
    knowledge: Knowledge = field(default_factory=Knowledge)

    # Recomputed every day from the living membership.
    member_ids: List[int] = field(default_factory=list)
    centre: Coord = (0, 0)
    average_hunger: float = 0.0
    average_aggression: float = 0.0

    # Lifetime history.
    parent_group_id: Optional[int] = None
    naming_style: int = 0  # Which syllable pool this tribe's names come from.
    chieftain_id: Optional[int] = None
    settlement_id: Optional[int] = None
    role_counts: Dict[str, int] = field(default_factory=dict)
    healer_resistance: float = 0.0  # Disease resistance from the tribe's healers.
    battles_won: int = 0
    battles_lost: int = 0
    war_dead: int = 0
    plague_dead: int = 0

    @property
    def size(self) -> int:
        """Number of living members."""
        return len(self.member_ids)

    @property
    def era(self) -> str:
        """The tribe's technological era."""
        return self.knowledge.era

    def distance_to(self, x: int, y: int) -> int:
        """Chebyshev distance from the tribe's centre to a point."""
        return max(abs(self.centre[0] - x), abs(self.centre[1] - y))


class GroupRegistry:
    """Owns every tribe and the rules by which they form, grow and split."""

    def __init__(self, config: GroupConfig) -> None:
        self.config = config
        self.groups: Dict[int, Group] = {}
        self._next_id = 1

    # -- lookup -------------------------------------------------------------

    def get(self, group_id: Optional[int]) -> Optional[Group]:
        """Look up a tribe by id, tolerating ``None``."""
        if group_id is None:
            return None
        return self.groups.get(group_id)

    def active(self) -> List[Group]:
        """Every surviving tribe, ordered by id for deterministic iteration."""
        return [self.groups[gid] for gid in sorted(self.groups)]

    def largest(self) -> Optional[Group]:
        """The most populous tribe, or ``None`` when there are none."""
        active = self.active()
        return max(active, key=lambda group: group.size) if active else None

    # -- creation -----------------------------------------------------------

    def create(
        self,
        day: int,
        rng: random.Random,
        parent: Optional[Group] = None,
    ) -> Group:
        """Found a new tribe, optionally inheriting a parent tribe's knowledge."""
        # A splinter tribe keeps its parent's naming style: the people sound
        # related because they are.
        style = parent.naming_style if parent is not None else rng.randrange(STYLE_COUNT)
        group = Group(
            id=self._next_id,
            name=tribe_name(rng, style),
            founded_day=day,
            parent_group_id=parent.id if parent else None,
            naming_style=style,
        )
        if parent is not None:
            # A splinter group leaves knowing what its parent knew.
            group.knowledge.known = set(parent.knowledge.known)
            group.knowledge.recompute_effects()
        self.groups[group.id] = group
        self._next_id += 1
        return group

    def dissolve(self, group_id: int) -> Optional[Group]:
        """Remove a tribe that has no living members left."""
        return self.groups.pop(group_id, None)

    # -- daily maintenance --------------------------------------------------

    def refresh(self, agents: Sequence[Agent]) -> List[Group]:
        """Recompute membership, centres and averages; return extinct tribes.

        Called once per day before any group logic runs, so every later system
        sees consistent membership.
        """
        for group in self.groups.values():
            group.member_ids.clear()

        sums: Dict[int, List[float]] = {}
        for agent in agents:
            group = self.get(agent.group_id)
            if group is None:
                agent.group_id = None
                continue
            group.member_ids.append(agent.id)
            totals = sums.setdefault(group.id, [0.0, 0.0, 0.0, 0.0])
            totals[0] += agent.x
            totals[1] += agent.y
            totals[2] += agent.needs.hunger
            totals[3] += agent.aggression

        for group_id, totals in sums.items():
            group = self.groups[group_id]
            count = float(group.size or 1)
            group.centre = (int(round(totals[0] / count)), int(round(totals[1] / count)))
            group.average_hunger = totals[2] / count
            group.average_aggression = totals[3] / count

        extinct = [group for group in self.active() if group.size == 0]
        for group in extinct:
            self.dissolve(group.id)
        return extinct

    # -- membership changes -------------------------------------------------

    def try_form_or_join(
        self,
        agent: Agent,
        neighbours: Iterable[Agent],
        day: int,
        rng: random.Random,
    ) -> Optional[Tuple[Group, bool]]:
        """Let an unaffiliated agent join a nearby tribe or found a new one.

        Returns ``(group, founded)`` where ``founded`` is True for a brand new
        tribe, or ``None`` if nothing happened today.
        """
        if agent.group_id is not None:
            return None

        companions = [other for other in neighbours if other.id != agent.id and other.alive]
        if not companions:
            return None

        # Prefer joining an existing tribe over founding a rival one.
        for other in companions:
            group = self.get(other.group_id)
            if group is not None and group.size < self.config.split_size:
                if rng.random() < self.config.recruit_chance:
                    agent.group_id = group.id
                    group.member_ids.append(agent.id)
                    return group, False
                return None

        # Otherwise two unaffiliated agents may found a tribe together.
        unaffiliated = [other for other in companions if other.group_id is None]
        if unaffiliated and rng.random() < self.config.founding_chance:
            group = self.create(day, rng)
            founder = unaffiliated[0]
            agent.group_id = group.id
            founder.group_id = group.id
            group.member_ids.extend([agent.id, founder.id])
            group.centre = (agent.x, agent.y)
            return group, True
        return None

    def find_unviable(self, day: int) -> List[Group]:
        """Tribes too small to survive, past their founding grace period."""
        return [
            group
            for group in self.active()
            if group.size < self.config.min_viable_size
            and day - group.founded_day > self.config.disband_grace_days
        ]

    def disband(self, group: Group, members: Sequence[Agent]) -> None:
        """Dissolve a failing tribe; its people become unaffiliated again."""
        for agent in members:
            agent.group_id = None
        self.dissolve(group.id)

    def try_split(
        self,
        group: Group,
        members: Sequence[Agent],
        day: int,
        rng: random.Random,
    ) -> Optional[Group]:
        """Split an oversized tribe, returning the new splinter tribe.

        The members furthest from the centre leave -- the periphery is exactly
        where a group's cohesion is weakest.
        """
        if group.size < self.config.split_size:
            return None
        leaving_count = int(group.size * self.config.split_fraction)
        if leaving_count < self.config.min_split_size:
            return None

        ranked = sorted(
            members,
            key=lambda agent: (-group.distance_to(agent.x, agent.y), agent.id),
        )
        leaving = ranked[:leaving_count]
        splinter = self.create(day, rng, parent=group)
        leaving_ids = {agent.id for agent in leaving}
        for agent in leaving:
            agent.group_id = splinter.id
        # Keep membership consistent immediately, rather than waiting for the
        # next daily refresh, so the rest of today's phases see the truth.
        splinter.member_ids = sorted(leaving_ids)
        group.member_ids = [
            member_id for member_id in group.member_ids if member_id not in leaving_ids
        ]
        centre_x = sum(agent.x for agent in leaving) // len(leaving)
        centre_y = sum(agent.y for agent in leaving) // len(leaving)
        splinter.centre = (centre_x, centre_y)
        return splinter
