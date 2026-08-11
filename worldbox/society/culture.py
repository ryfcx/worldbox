"""Culture: how knowledge, food and leadership move between and within tribes.

Three things here keep a civilisation from being a set of sealed boxes:

* **Diffusion.** Neighbouring tribes on tolerable terms pick up each other's
  technology. A region advances together, and an isolated tribe falls behind --
  which is much closer to how human technology actually spread than every tribe
  independently reinventing agriculture.
* **Trade.** Allies send food to a neighbour whose granary has failed, so a
  friendly neighbourhood survives a famine that would kill a lone tribe.
* **Chieftains.** Each tribe is led by its most prestigious member, and the
  leader's temperament slowly pulls the tribe's own toward theirs -- so a line
  of warlike chiefs makes a warlike people.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..agents.agent import Agent
from ..config import CultureConfig
from .diplomacy import Diplomacy
from .groups import Group, GroupRegistry
from .settlements import SettlementSystem
from .technology import TECH_BY_ID, Technology


@dataclass
class DiffusionEvent:
    """One tribe learning a technology from another."""

    learner: Group
    teacher: Group
    technology: Technology


@dataclass
class TradeEvent:
    """One tribe sending food to another."""

    giver: Group
    receiver: Group
    amount: float


def choose_chieftain(members: Sequence[Agent], days_per_year: int) -> Optional[Agent]:
    """Pick a tribe's leader: the member with the most standing.

    Prestige is age plus a bounded contribution from children and victories --
    experience, lineage and prowess, in that order of weight.
    """
    adults = [agent for agent in members if agent.alive]
    if not adults:
        return None

    def prestige(agent: Agent) -> Tuple[float, int]:
        years = agent.age_days / days_per_year
        score = years + 3.0 * len(agent.children) + 2.0 * agent.kills
        return (score, -agent.id)  # -id breaks ties deterministically.

    return max(adults, key=prestige)


def apply_chieftain_influence(
    group: Group, chieftain: Optional[Agent], members: Sequence[Agent], config: CultureConfig
) -> None:
    """Nudge a tribe's members toward their leader's temperament."""
    if chieftain is None or not config.enabled:
        return
    pull = config.chieftain_influence
    for agent in members:
        if agent.id == chieftain.id:
            continue
        agent.aggression += (chieftain.aggression - agent.aggression) * pull * 0.01
        agent.aggression = max(0.0, min(1.0, agent.aggression))


def diffuse_knowledge(
    registry: GroupRegistry,
    diplomacy: Diplomacy,
    config: CultureConfig,
    rng: random.Random,
) -> List[DiffusionEvent]:
    """Let neighbouring tribes copy technologies from one another."""
    events: List[DiffusionEvent] = []
    if not config.enabled:
        return events

    groups = registry.active()
    for learner in groups:
        for teacher in groups:
            if teacher.id == learner.id:
                continue
            if diplomacy.relation(learner.id, teacher.id) < config.diffusion_min_relation:
                continue
            distance = max(
                abs(learner.centre[0] - teacher.centre[0]),
                abs(learner.centre[1] - teacher.centre[1]),
            )
            if distance > config.diffusion_radius:
                continue

            # Only technologies the learner could actually put to use.
            teachable = [
                tech_id
                for tech_id in sorted(teacher.knowledge.known - learner.knowledge.known)
                if all(
                    prerequisite in learner.knowledge.known
                    for prerequisite in TECH_BY_ID[tech_id].prerequisites
                )
            ]
            if not teachable:
                continue
            if rng.random() >= config.diffusion_chance:
                continue

            tech_id = rng.choice(teachable)
            learner.knowledge.learn(tech_id)
            events.append(DiffusionEvent(learner, teacher, TECH_BY_ID[tech_id]))
            break  # One borrowed idea per tribe per day.
    return events


def trade_food(
    registry: GroupRegistry,
    settlements: SettlementSystem,
    diplomacy: Diplomacy,
    config: CultureConfig,
) -> List[TradeEvent]:
    """Send food from full granaries to allied tribes that have run dry."""
    trades: List[TradeEvent] = []
    if not config.enabled:
        return trades

    groups = registry.active()
    for receiver in groups:
        receiving = settlements.for_group(receiver.id)
        if receiving is None or receiving.store_fraction() > 0.1:
            continue
        for giver in groups:
            if giver.id == receiver.id:
                continue
            if diplomacy.relation(giver.id, receiver.id) < config.trade_min_relation:
                continue
            giving = settlements.for_group(giver.id)
            if giving is None or giving.store_fraction() < 0.5:
                continue

            amount = giving.food_store * config.trade_fraction
            giving.food_store -= amount
            receiving.food_store = min(
                receiving.spec.store_capacity, receiving.food_store + amount
            )
            trades.append(TradeEvent(giver, receiver, amount))
            break  # One shipment per receiver per day.
    return trades
