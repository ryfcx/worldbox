"""Agent needs (hunger, energy, health) and the daily metabolism rules.

Conventions:
    hunger  0.0 = completely full      100.0 = starving
    energy  100.0 = fully rested         0.0 = exhausted
    health  100.0 = healthy              0.0 = dead

Only the *passive* daily upkeep lives here. Costs and gains caused by actions
(moving, resting, eating) are applied by :mod:`worldbox.agents.behavior` and the
engine, so this module stays a pure description of the body.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AgentConfig


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Constrain ``value`` to the ``low``..``high`` range."""
    return max(low, min(high, value))


@dataclass
class Needs:
    """The three tracked needs of a single agent."""

    hunger: float = 0.0
    energy: float = 100.0
    health: float = 100.0

    def is_hungry(self, config: AgentConfig) -> bool:
        """True once hunger crosses the threshold that motivates food seeking."""
        return self.hunger >= config.hunger_hungry_threshold

    def is_starving(self, config: AgentConfig) -> bool:
        """True once hunger is high enough to damage health."""
        return self.hunger >= config.hunger_starving_threshold

    def is_tired(self, config: AgentConfig) -> bool:
        """True once energy is low enough that resting takes priority."""
        return self.energy <= config.energy_tired_threshold

    def is_exhausted(self) -> bool:
        """True when energy has bottomed out and health starts to suffer."""
        return self.energy <= 0.0

    def is_dead(self) -> bool:
        """True when health has run out."""
        return self.health <= 0.0


def apply_daily_upkeep(
    needs: Needs, config: AgentConfig, health_bonus: float = 0.0
) -> None:
    """Apply one day of baseline metabolism (phase 2 of the tick).

    Hunger always rises and energy always falls; health then reacts to whichever
    needs are being neglected, or slowly recovers when the agent is comfortable.

    ``health_bonus`` is an additive multiplier on recovery (0.35 = +35%), which
    the engine derives from the agent's tribe's medical technology.
    """
    needs.hunger = clamp(needs.hunger + config.hunger_per_day)
    needs.energy = clamp(needs.energy - config.energy_per_day)

    health_delta = 0.0
    if needs.is_starving(config):
        health_delta -= config.health_loss_starving
    if needs.is_exhausted():
        health_delta -= config.health_loss_exhausted
    if (
        health_delta == 0.0
        and needs.hunger <= config.health_regen_max_hunger
        and needs.energy >= config.health_regen_min_energy
    ):
        health_delta += config.health_regen * (1.0 + max(0.0, health_bonus))

    needs.health = clamp(needs.health + health_delta)


def apply_movement_cost(needs: Needs, config: AgentConfig) -> None:
    """Charge the extra hunger/energy cost of moving one tile."""
    needs.hunger = clamp(needs.hunger + config.hunger_per_move)
    needs.energy = clamp(needs.energy - config.energy_per_move)


def apply_rest(needs: Needs, config: AgentConfig) -> None:
    """Restore energy for a day spent resting."""
    needs.energy = clamp(needs.energy + config.energy_restored_per_rest)


def apply_meal(needs: Needs, food_units: float, config: AgentConfig) -> None:
    """Reduce hunger in proportion to the food actually eaten."""
    needs.hunger = clamp(needs.hunger - food_units * config.hunger_restored_per_food)


def cause_of_death(needs: Needs, config: AgentConfig) -> str:
    """Best-guess description of why an agent with no health left died."""
    if needs.is_starving(config):
        return "starvation"
    if needs.is_exhausted():
        return "exhaustion"
    return "poor health"
