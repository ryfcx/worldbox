"""Agent decision-making and action execution.

The behaviour layer follows a strict rule: it may read the world and move the
agent, but it never mutates the resource field, never touches other agents, and
never writes to the event log. Anything with wider consequences is reported back
to the engine as an :class:`ActionResult`, which keeps agents, resources and
events loosely coupled.

Decision priority (simple, deliberately shallow):

    1. Enemy near and hurt -> FLEE
    2. Enemy near, willing -> FIGHT
    3. Too tired           -> REST
    4. Hungry, food here   -> EAT
    5. Hungry, no food     -> SEEK_FOOD
    6. Ready to breed      -> SEEK_MATE
    7. Otherwise           -> WANDER (drifting home if far from the tribe)

The behaviour layer is told about threats through a small ``ThreatSense``
callback rather than importing the society package, which keeps the dependency
arrow pointing one way: society -> agents, never the reverse.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from ..config import Config
from ..world.world import World
from .agent import Agent
from .needs import apply_movement_cost, apply_rest

Coord = Tuple[int, int]

# Where each agent stands, rebuilt by the engine once per day.
Occupancy = Dict[Coord, List[Agent]]

# Given an agent, return the nearest hostile agent (or None). Supplied by the
# engine so this module never has to know how tribes or wars work.
ThreatSense = Callable[[Agent], Optional[Agent]]

# Given an agent, return its tribe's centre (or None if it has no tribe).
HomeSense = Callable[[Agent], Optional[Coord]]


class Goal(Enum):
    """What an agent is currently trying to do."""

    WANDER = "wander"
    SEEK_FOOD = "seek_food"
    EAT = "eat"
    REST = "rest"
    SEEK_MATE = "seek_mate"
    FIGHT = "fight"
    FLEE = "flee"

    @property
    def label(self) -> str:
        """Human-readable goal name, e.g. ``"seek food"``."""
        return self.value.replace("_", " ")


@dataclass
class ActionResult:
    """What the engine must follow up on after an agent acted."""

    wants_to_eat: bool = False
    moved: bool = False
    discovered_terrain: Optional[str] = None


@dataclass
class SocialContext:
    """Society-derived facts about one agent's situation for one day.

    The engine assembles this from the tribe, diplomacy and combat systems and
    hands it down. Bundling it keeps the behaviour functions' signatures stable
    as more social systems are added, and keeps this module from importing any
    of them.
    """

    threat: Optional[Agent] = None  # Nearest hostile agent, if any.
    home: Optional[Coord] = None  # The agent's tribe's settlement or centre.
    reproduction_cooldown_scale: float = 1.0  # <1.0 means children come sooner.
    vision_bonus: int = 0  # Extra sight radius, e.g. for foragers.


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def can_reproduce(
    agent: Agent, config: Config, day: int, cooldown_scale: float = 1.0
) -> bool:
    """True if ``agent`` currently satisfies every reproduction precondition.

    ``cooldown_scale`` below 1.0 shortens the gap between children -- the
    engine derives it from the tribe's fertility technologies.
    """
    agent_config = config.agents
    days_per_year = config.simulation.days_per_year
    if not agent.is_adult(agent_config, days_per_year):
        return False
    if agent.age_days > agent_config.max_reproduction_age_years * days_per_year:
        return False
    cooldown = agent_config.reproduction_cooldown_days * max(0.1, cooldown_scale)
    if day - agent.last_birth_day < cooldown:
        return False
    needs = agent.needs
    return (
        needs.hunger <= agent_config.reproduction_max_hunger
        and needs.energy >= agent_config.reproduction_min_energy
        and needs.health >= agent_config.reproduction_min_health
    )


def decide(
    agent: Agent,
    world: World,
    config: Config,
    day: int,
    social: Optional[SocialContext] = None,
) -> Goal:
    """Choose this agent's goal for the day (phase 3 of the tick).

    Survival outranks every other need, so a nearby threat is considered first.
    """
    needs = agent.needs
    agent_config = config.agents
    social = social or SocialContext()
    threat = social.threat

    if threat is not None:
        # The wounded, the young and the timid run; everyone else stands.
        too_hurt = needs.health <= config.combat.flee_health
        outmatched = threat.needs.health > needs.health * 1.5
        if too_hurt or (outmatched and agent.aggression < 0.5):
            return Goal.FLEE
        if agent.is_adult(agent_config, config.simulation.days_per_year):
            return Goal.FIGHT
        return Goal.FLEE

    if needs.is_tired(agent_config):
        return Goal.REST

    if needs.is_hungry(agent_config):
        here = world.resources.food_at(agent.x, agent.y)
        if here >= config.resources.min_food_to_eat:
            return Goal.EAT
        return Goal.SEEK_FOOD

    if can_reproduce(agent, config, day, social.reproduction_cooldown_scale):
        return Goal.SEEK_MATE

    return Goal.WANDER


# ---------------------------------------------------------------------------
# Movement helpers
# ---------------------------------------------------------------------------


def _step_towards(
    agent: Agent, target: Coord, world: World, rng: random.Random
) -> bool:
    """Move one tile towards ``target``; return True if the agent moved.

    Greedy and memoryless -- no pathfinding in v1. Ties are broken randomly so
    agents do not all hug the same wall, and an agent that cannot improve its
    distance takes a random sidestep instead of freezing.
    """
    tx, ty = target
    options = world.neighbours(agent.x, agent.y)
    if not options:
        return False

    current_distance = max(abs(agent.x - tx), abs(agent.y - ty))
    best: List[Coord] = []
    best_distance = current_distance
    for nx, ny in options:
        distance = max(abs(nx - tx), abs(ny - ty))
        if distance < best_distance:
            best_distance = distance
            best = [(nx, ny)]
        elif distance == best_distance and distance < current_distance:
            best.append((nx, ny))

    destination = rng.choice(best) if best else rng.choice(options)
    agent.move_to(*destination)
    return True


def _wander(agent: Agent, world: World, rng: random.Random) -> bool:
    """Take a random step to an adjacent passable tile."""
    options = world.neighbours(agent.x, agent.y)
    if not options:
        return False
    agent.move_to(*rng.choice(options))
    return True


def _step_away_from(agent: Agent, threat: Coord, world: World, rng: random.Random) -> bool:
    """Move one tile away from a threat; wander if cornered."""
    tx, ty = threat
    options = world.neighbours(agent.x, agent.y)
    if not options:
        return False
    current = max(abs(agent.x - tx), abs(agent.y - ty))
    retreats = [
        (nx, ny) for nx, ny in options if max(abs(nx - tx), abs(ny - ty)) > current
    ]
    agent.move_to(*rng.choice(retreats if retreats else options))
    return True


def _wander_or_return_home(
    agent: Agent, home: Optional[Coord], world: World, config: Config, rng: random.Random
) -> bool:
    """Wander, but drift back toward the tribe's centre.

    This single rule is what turns a scatter of individuals into packs that
    hold recognisable territory: members who stray past the territory radius
    always turn back, and even those inside it usually stay close.
    """
    if home is None:
        return _wander(agent, world, rng)
    group_config = config.groups
    distance = max(abs(agent.x - home[0]), abs(agent.y - home[1]))
    if distance > group_config.territory_radius or rng.random() < group_config.cohesion_chance:
        return _step_towards(agent, home, world, rng)
    return _wander(agent, world, rng)


def find_food_target(
    agent: Agent, world: World, config: Config, day: int, vision_bonus: int = 0
) -> Optional[Coord]:
    """Pick the best known or visible food tile, or ``None`` if there is none.

    Memory is consulted first (it is cheap and gives agents a sense of place);
    stale entries that no longer hold food are forgotten on the spot. Otherwise
    the agent scans its vision radius and scores tiles by food per unit distance.
    """
    minimum = config.resources.min_food_to_eat

    best: Optional[Coord] = None
    best_score = 0.0
    for location in list(agent.memory.known_food):
        x, y = location
        available = world.resources.food_at(x, y)
        if available < minimum:
            agent.memory.forget_food(location)
            continue
        score = available / (1.0 + agent.distance_to(x, y))
        if score > best_score:
            best_score = score
            best = location
    if best is not None:
        return best

    vision = config.agents.vision_radius + max(0, vision_bonus)
    for x, y in world.tiles_within(agent.x, agent.y, vision):
        available = world.resources.food_at(x, y)
        if available < minimum or not world.is_passable(x, y):
            continue
        score = available / (1.0 + agent.distance_to(x, y))
        if score > best_score:
            best_score = score
            best = (x, y)

    if best is not None:
        agent.memory.remember_food(best, day, config.agents)
    return best


def find_mate_target(
    agent: Agent, occupancy: Occupancy, world: World, config: Config, day: int
) -> Optional[Coord]:
    """Nearest visible agent that is also ready to reproduce."""
    best: Optional[Coord] = None
    best_distance = 10**9
    # Mate-finding uses plain vision: the forager's bonus is for spotting food.
    for x, y in world.tiles_within(agent.x, agent.y, config.agents.vision_radius):
        for other in occupancy.get((x, y), ()):
            if other.id == agent.id or not other.alive:
                continue
            if not can_reproduce(other, config, day):
                continue
            distance = agent.distance_to(x, y)
            if distance < best_distance:
                best_distance = distance
                best = (x, y)
    return best


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute(
    agent: Agent,
    goal: Goal,
    world: World,
    occupancy: Occupancy,
    config: Config,
    rng: random.Random,
    day: int,
    social: Optional[SocialContext] = None,
) -> ActionResult:
    """Carry out ``goal`` for one day (phase 4 of the tick).

    Eating and fighting are *not* performed here: the agent only signals
    intent, and the engine resolves meals and battles afterwards, so agents
    competing for the same tile or the same opponent are handled consistently.
    """
    result = ActionResult()
    social = social or SocialContext()
    threat = social.threat

    if goal is Goal.EAT:
        result.wants_to_eat = True
        return result

    if goal is Goal.FIGHT:
        # Closing with the enemy is the action; the engine resolves the fight.
        if threat is not None and agent.distance_to(threat.x, threat.y) > 1:
            result.moved = _step_towards(agent, (threat.x, threat.y), world, rng)
        if result.moved:
            apply_movement_cost(agent.needs, config.agents)
        return result

    if goal is Goal.FLEE:
        if threat is not None:
            result.moved = _step_away_from(agent, (threat.x, threat.y), world, rng)
        else:
            result.moved = _wander(agent, world, rng)
        if result.moved:
            apply_movement_cost(agent.needs, config.agents)
        return result

    if goal is Goal.REST:
        apply_rest(agent.needs, config.agents)
        return result

    if goal is Goal.SEEK_FOOD:
        target = find_food_target(agent, world, config, day, social.vision_bonus)
        result.moved = (
            _step_towards(agent, target, world, rng)
            if target is not None
            else _wander(agent, world, rng)
        )
    elif goal is Goal.SEEK_MATE:
        target = find_mate_target(agent, occupancy, world, config, day)
        result.moved = (
            _step_towards(agent, target, world, rng)
            if target is not None
            else _wander(agent, world, rng)
        )
    else:  # Goal.WANDER
        result.moved = _wander_or_return_home(agent, social.home, world, config, rng)

    if result.moved:
        apply_movement_cost(agent.needs, config.agents)
        terrain = world.terrain_at(agent.x, agent.y)
        if terrain is not None and terrain.value not in agent.memory.known_terrain:
            agent.memory.known_terrain.add(terrain.value)
            result.discovered_terrain = terrain.value

    return result
