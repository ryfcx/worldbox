"""EDIT THIS FILE to change how agents think.

This is the one place to change agent behaviour. Everything else in the
simulation is plumbing; this is the decision-making itself.

--------------------------------------------------------------------------
HOW IT WORKS
--------------------------------------------------------------------------
Once a day, every agent scores each thing it could do on a scale of 0 to 1,
and does whichever scores highest. That's the whole model.

A scorer is a function that takes the situation and returns a number:

    @scorer(Goal.REST)
    def score_rest(s: Situation) -> float:
        return s.tiredness

Return 0 to mean "not an option right now". Return something above 1 and it
will dominate everything else, so keep scores inside 0..1 and use the WEIGHTS
below to say how much each action matters in general.

--------------------------------------------------------------------------
THREE THINGS YOU CAN DO HERE
--------------------------------------------------------------------------
1. Tune WEIGHTS below. Raise `Goal.FIGHT` and the world gets more violent.
   This is the quickest experiment and needs no code.

2. Rewrite a scorer. Want agents to keep eating until completely full?
   Change `score_eat` to return `1.0` whenever there is food underfoot.

3. Add a whole new action. Three steps, no other file needs touching:
     a. Add it to `Goal` in agents/behavior.py
     b. Write a scorer for it here
     c. Teach `execute()` in agents/behavior.py how to carry it out

--------------------------------------------------------------------------
WHAT `s` (the Situation) GIVES YOU
--------------------------------------------------------------------------
    s.hunger      0..1, curved. 1 = starving
    s.tiredness   0..1, curved. 1 = exhausted
    s.frailty     0..1. 1 = at death's door
    s.comfort     0..1. High when well fed, rested and healthy
    s.food_here   food on the agent's own tile
    s.can_eat     True if there is enough food here to bother
    s.threat      the nearest hostile agent, or None
    s.courage     0..1, from health, aggression and the caution trait
    s.can_breed   True if every reproduction condition is met
    s.is_adult    True once old enough
    s.agent       the agent itself (age, traits, tribe, memory, needs...)
    s.world       the map, for terrain and food lookups
    s.config      the full config, if you need a number from it

After changing anything here, check what it did:

    python3 tools/compare_runs.py --days 6000
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    from .agents.agent import Agent
    from .agents.behavior import Goal
    from .config import Config
    from .world.world import World


# ===========================================================================
# WEIGHTS -- how much each action matters in general.
# Multiplied by whatever its scorer returns. This is the fastest thing to tune.
# ===========================================================================

WEIGHTS: Dict[str, float] = {
    "eat": 1.00,
    "seek_food": 0.95,
    "rest": 0.90,
    "flee": 1.15,
    "fight": 0.95,
    "seek_mate": 0.60,
    # Wandering is the floor: it wins only when nothing else is pressing.
    "wander": 0.06,
}


# ===========================================================================
# THE SITUATION -- everything a scorer can see.
# ===========================================================================


@dataclass
class Situation:
    """One agent's circumstances on one day, pre-chewed for the scorers."""

    agent: "Agent"
    world: "World"
    config: "Config"
    day: int

    hunger: float  # 0..1, curved so starvation dominates
    tiredness: float  # 0..1, curved
    frailty: float  # 0..1, how close to death
    comfort: float  # 0..1, how well-off overall
    food_here: float  # Raw food on this tile
    can_eat: bool  # Enough food here to be worth eating
    threat: Optional["Agent"]  # Nearest enemy, if any
    courage: float  # 0..1, willingness to fight rather than run
    can_breed: bool  # All reproduction conditions met
    is_adult: bool


# ===========================================================================
# THE SCORERS -- one per action. Edit freely.
# ===========================================================================

# Filled in by the @scorer decorator below.
SCORERS: Dict["Goal", Callable[[Situation], float]] = {}


def scorer(goal: "Goal") -> Callable:
    """Register a function as the scorer for one goal."""

    def register(function: Callable[[Situation], float]) -> Callable[[Situation], float]:
        SCORERS[goal] = function
        return function

    return register


def install_default_scorers() -> None:
    """Register the built-in scorers.

    Called once at import time. Kept in a function so the module has no
    import-order problem with the Goal enum.
    """
    from .agents.behavior import Goal

    @scorer(Goal.EAT)
    def score_eat(s: Situation) -> float:
        """Eat what is underfoot. Hungrier means keener."""
        if not s.can_eat:
            return 0.0
        return s.hunger

    @scorer(Goal.SEEK_FOOD)
    def score_seek_food(s: Situation) -> float:
        """Go looking for food. Industrious agents set out sooner."""
        if s.can_eat:
            return 0.0  # No need to travel; there is food right here.
        eagerness = 0.55 + 0.45 * s.agent.industry
        return s.hunger * eagerness

    @scorer(Goal.REST)
    def score_rest(s: Situation) -> float:
        """Sleep it off."""
        return s.tiredness

    @scorer(Goal.FIGHT)
    def score_fight(s: Situation) -> float:
        """Stand and fight. Only against a real enemy, and only if grown."""
        if s.threat is None or not s.is_adult:
            return 0.0
        return s.courage

    @scorer(Goal.FLEE)
    def score_flee(s: Situation) -> float:
        """Run. The hurt and the timid run hardest."""
        if s.threat is None:
            return 0.0
        return min(1.0, (1.0 - s.courage) + s.frailty * 0.5)

    @scorer(Goal.SEEK_MATE)
    def score_seek_mate(s: Situation) -> float:
        """Look for a partner, but only when life is otherwise comfortable."""
        if not s.can_breed:
            return 0.0
        return s.comfort

    @scorer(Goal.WANDER)
    def score_wander(s: Situation) -> float:
        """Always available, always cheap. The fallback."""
        return 1.0


install_default_scorers()
