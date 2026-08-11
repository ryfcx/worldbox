"""Fights between members of rival tribes.

Two agents standing next to each other come to blows if their tribes are at war,
or -- less often -- if one of them is aggressive enough to start something
unprovoked. Unprovoked skirmishes damage relations, which is how peacetime
violence escalates into an actual war.

Strength is drawn from health, age, the tribe's weapons technology and the
agent's inherited aggression, with a random factor so the stronger side is
favoured but never guaranteed to win.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..agents.agent import Agent
from ..config import AgentConfig, CombatConfig, RoleConfig
from ..agents.needs import clamp
from .diplomacy import Diplomacy
from .groups import Group, GroupRegistry
from .roles import combat_multiplier


@dataclass
class BattleResult:
    """The outcome of a single fight."""

    winner: Agent
    loser: Agent
    winner_group: Optional[Group]
    loser_group: Optional[Group]
    fatal: bool
    provoked: bool  # True if the tribes were already at war.


def combat_strength(
    agent: Agent,
    group: Optional[Group],
    config: CombatConfig,
    agent_config: AgentConfig,
    role_config: RoleConfig,
    days_per_year: int,
    rng: random.Random,
) -> float:
    """How hard an agent hits, all factors combined.

    Children and the very old fight poorly; health, weapons technology and a
    warlike temperament all help.
    """
    years = agent.age_days / days_per_year
    if years < agent_config.adult_age_years:
        age_factor = 0.35 + 0.65 * (years / max(1.0, agent_config.adult_age_years))
    elif years > agent_config.max_reproduction_age_years:
        decline = (years - agent_config.max_reproduction_age_years) / 40.0
        age_factor = max(0.3, 1.0 - decline)
    else:
        age_factor = 1.0

    tech_bonus = 0.0
    if group is not None:
        tech_bonus = min(config.max_tech_combat_bonus, group.knowledge.effects.combat)

    return (
        (agent.needs.health / 100.0)
        * age_factor
        * (1.0 + tech_bonus)
        * combat_multiplier(agent, role_config)
        * (0.6 + 0.8 * agent.aggression)
        * rng.uniform(0.8, 1.2)
    ) or 0.01


def resolve_fight(
    attacker: Agent,
    defender: Agent,
    registry: GroupRegistry,
    config: CombatConfig,
    agent_config: AgentConfig,
    role_config: RoleConfig,
    days_per_year: int,
    day: int,
    provoked: bool,
    rng: random.Random,
) -> BattleResult:
    """Fight one round between two agents and apply the damage."""
    attacker_group = registry.get(attacker.group_id)
    defender_group = registry.get(defender.group_id)

    attacker_strength = combat_strength(
        attacker, attacker_group, config, agent_config, role_config, days_per_year, rng
    )
    defender_strength = combat_strength(
        defender, defender_group, config, agent_config, role_config, days_per_year, rng
    )

    if attacker_strength >= defender_strength:
        winner, loser = attacker, defender
        ratio = attacker_strength / max(0.01, defender_strength)
        winner_group, loser_group = attacker_group, defender_group
    else:
        winner, loser = defender, attacker
        ratio = defender_strength / max(0.01, attacker_strength)
        winner_group, loser_group = defender_group, attacker_group

    ratio = min(config.damage_ratio_cap, ratio)
    loser.needs.health = clamp(loser.needs.health - config.loser_damage * ratio)
    winner.needs.health = clamp(winner.needs.health - config.winner_damage)
    loser.wounded_on_day = day
    loser.wounded_by_group = winner.group_id
    winner.wounded_on_day = day
    winner.wounded_by_group = loser.group_id

    fatal = loser.needs.health <= 0.0
    if fatal:
        winner.kills += 1
        winner.memory.note(day, f"killed Agent #{loser.id} in battle")
    if winner_group is not None:
        winner_group.battles_won += 1
    if loser_group is not None:
        loser_group.battles_lost += 1

    return BattleResult(
        winner=winner,
        loser=loser,
        winner_group=winner_group,
        loser_group=loser_group,
        fatal=fatal,
        provoked=provoked,
    )


def find_enemy(
    agent: Agent,
    occupancy,
    world,
    diplomacy: Diplomacy,
    radius: int = 1,
) -> Optional[Agent]:
    """The nearest agent from a hostile tribe, or ``None``.

    Used both to pick fights and to decide when to run away.
    """
    if agent.group_id is None:
        return None
    for x, y in world.tiles_within(agent.x, agent.y, radius):
        for other in occupancy.get((x, y), ()):
            if not other.alive or other.id == agent.id or other.group_id is None:
                continue
            if other.group_id != agent.group_id and diplomacy.are_hostile(
                agent.group_id, other.group_id
            ):
                return other
    return None


def engagements(
    agents: List[Agent],
    occupancy,
    world,
    registry: GroupRegistry,
    diplomacy: Diplomacy,
    config: CombatConfig,
    rng: random.Random,
) -> List[Tuple[Agent, Agent, bool]]:
    """Pick every fight that happens today.

    Each agent fights at most once per day, and pairs are chosen in list order,
    so the result is deterministic for a given seed.
    """
    fighting: set[int] = set()
    chosen: List[Tuple[Agent, Agent, bool]] = []
    if not config.enabled:
        return chosen

    for agent in agents:
        if not agent.alive or agent.id in fighting or agent.group_id is None:
            continue
        enemy = find_enemy(agent, occupancy, world, diplomacy)
        if enemy is None or enemy.id in fighting:
            continue

        provoked = diplomacy.at_war(agent.group_id, enemy.group_id)
        if provoked:
            chance = config.war_engagement_chance
        else:
            # Only genuinely aggressive agents start unprovoked trouble.
            chance = config.skirmish_chance * agent.aggression
        if rng.random() >= chance:
            continue

        fighting.add(agent.id)
        fighting.add(enemy.id)
        chosen.append((agent, enemy, provoked))
    return chosen
