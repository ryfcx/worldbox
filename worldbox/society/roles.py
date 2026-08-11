"""Professions: the division of labour inside a tribe.

A band of identical foragers is not a civilisation. Once a tribe has the
technology for it, its members specialise -- and specialisation is what turns
technology into a compounding advantage rather than a flat bonus:

    FORAGER  the default; better at finding wild food
    FARMER   needs Agriculture; harvests into the settlement's granary
    WARRIOR  fights harder, and more of them are called up during a war
    SCHOLAR  needs Writing; multiplies the tribe's research
    HEALER   needs Herbal Medicine; protects the tribe against disease
    CHILD    too young to work
    ELDER    past working age, but contributes accumulated wisdom

Roles are reassigned as the tribe's needs change: a tribe at war calls up
warriors, a starving tribe puts people back in the fields.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, List, Sequence

from ..agents.agent import Agent
from ..config import AgentConfig, RoleConfig
from .technology import Knowledge


class Role(Enum):
    """What an agent does for their tribe."""

    CHILD = "child"
    FORAGER = "forager"
    FARMER = "farmer"
    WARRIOR = "warrior"
    SCHOLAR = "scholar"
    HEALER = "healer"
    ELDER = "elder"

    @property
    def label(self) -> str:
        """Human-readable role name."""
        return self.value.capitalize()


# A role is only available once its enabling technology is known.
ROLE_REQUIREMENTS: Dict[Role, str] = {
    Role.FARMER: "agriculture",
    Role.SCHOLAR: "writing",
    Role.HEALER: "medicine",
}


def assign_roles(
    members: Sequence[Agent],
    knowledge: Knowledge,
    at_war: bool,
    starving: bool,
    config: RoleConfig,
    agent_config: AgentConfig,
    days_per_year: int,
) -> Dict[Role, int]:
    """Assign every member of a tribe a role for today.

    Selection is by suitability -- the most aggressive become warriors, the
    oldest become scholars -- and entirely deterministic, so a given seed always
    produces the same society.

    Returns a count of each role, which the engine feeds back into research,
    combat and disease.
    """
    counts: Dict[Role, int] = {role: 0 for role in Role}

    workers: List[Agent] = []
    for agent in members:
        years = agent.age_days / days_per_year
        if years < agent_config.adult_age_years:
            agent.role = Role.CHILD.value
            counts[Role.CHILD] += 1
        elif years > config.elder_age_years:
            agent.role = Role.ELDER.value
            counts[Role.ELDER] += 1
        else:
            workers.append(agent)

    if not workers:
        return counts

    available = len(workers)
    assigned: set[int] = set()

    def take(role: Role, fraction: float, key) -> None:
        """Assign the best-suited unassigned workers to ``role``."""
        if role in ROLE_REQUIREMENTS and not knowledge.knows(ROLE_REQUIREMENTS[role]):
            return
        wanted = int(available * fraction)
        if wanted <= 0:
            return
        candidates = sorted(
            (worker for worker in workers if worker.id not in assigned),
            key=key,
        )
        for worker in candidates[:wanted]:
            worker.role = role.value
            assigned.add(worker.id)
            counts[role] += 1

    # Warriors first: a tribe at war calls up far more of them.
    warrior_share = config.warrior_share_at_war if at_war else config.warrior_share_at_peace
    take(Role.WARRIOR, warrior_share, key=lambda a: (-a.aggression, a.id))

    # A starving tribe puts everyone it can back into food production.
    farmer_share = config.farmer_share_when_starving if starving else config.farmer_share
    take(Role.FARMER, farmer_share, key=lambda a: (a.aggression, a.id))

    take(Role.HEALER, config.healer_share, key=lambda a: (a.aggression, a.id))
    take(Role.SCHOLAR, config.scholar_share, key=lambda a: (-a.age_days, a.id))

    # Everyone left forages.
    for worker in workers:
        if worker.id not in assigned:
            worker.role = Role.FORAGER.value
            counts[Role.FORAGER] += 1

    return counts


def research_multiplier(counts: Dict[Role, int], config: RoleConfig) -> float:
    """How much a tribe's scholars and elders speed up its research."""
    return (
        1.0
        + counts.get(Role.SCHOLAR, 0) * config.scholar_research_bonus
        + counts.get(Role.ELDER, 0) * config.elder_research_bonus
    )


def disease_resistance_bonus(counts: Dict[Role, int], population: int, config: RoleConfig) -> float:
    """Extra disease resistance granted by the tribe's healers."""
    if population <= 0:
        return 0.0
    healers = counts.get(Role.HEALER, 0)
    return min(config.max_healer_resistance, healers / population * config.healer_resistance)


def combat_multiplier(agent: Agent, config: RoleConfig) -> float:
    """How much better a warrior fights than an ordinary tribe member."""
    if agent.role == Role.WARRIOR.value:
        return 1.0 + config.warrior_combat_bonus
    if agent.role in (Role.CHILD.value, Role.ELDER.value):
        return config.noncombatant_penalty
    return 1.0


def foraging_vision_bonus(agent: Agent, config: RoleConfig) -> int:
    """Extra sight radius a forager has when looking for wild food."""
    return config.forager_vision_bonus if agent.role == Role.FORAGER.value else 0
