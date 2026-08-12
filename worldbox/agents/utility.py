"""Utility-based action selection.

The original decision function was a first-match ladder: check threat, then
tiredness, then hunger, then reproduction, else wander. Whichever test matched
first won, and nothing below it was ever considered. That has two failure modes
which showed up clearly in measurement:

* **Cliffs.** ``hunger >= 45`` meant an agent at 44 was not hungry at all and one
  at 45 was fully hungry. Around 44% of a settled population sat within a single
  day's drift of a threshold, flipping goals on trivial state changes.
* **No trade-offs.** A starving agent at energy 30 rested, because ``is_tired``
  sat higher in the ladder -- not because resting was the better choice.

Here every action is scored on 0..1 and the best one wins. Nothing is skipped, so
competing needs actually resolve against each other, and the scores are curved
rather than stepped, so behaviour changes smoothly as state changes.

The scoring functions are deliberately small and independent. Adding a new
action later -- take a job, trade, migrate -- means writing one more scorer, not
finding the right rung of a ladder and re-reasoning about everything below it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

from ..config import Config

if TYPE_CHECKING:  # Avoids a circular import at runtime.
    from ..rules import Situation
    from ..world.world import World
    from .agent import Agent
    from .behavior import Goal, SocialContext


def clamp01(value: float) -> float:
    """Constrain a score to 0..1."""
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def curve(value: float, exponent: float) -> float:
    """Bend a 0..1 need so urgency rises steeply rather than linearly.

    An exponent above 1 keeps mild needs cheap and makes severe ones dominate,
    which is what stops an agent abandoning a meal because it is slightly tired.
    """
    return clamp01(value) ** exponent


@dataclass
class Decision:
    """The chosen action and the scores it beat.

    Keeping the full score table makes the agent's reasoning inspectable, which
    is the whole point of moving off a ladder -- you can ask *why* it chose that.
    """

    goal: "Goal"
    score: float
    scores: Dict["Goal", float] = field(default_factory=dict)

    def ranked(self):
        """Every action, best first."""
        return sorted(self.scores.items(), key=lambda item: -item[1])


def build_situation(
    agent: "Agent",
    world: "World",
    config: Config,
    day: int,
    social: Optional["SocialContext"] = None,
) -> "Situation":
    """Work out everything the scorers in :mod:`worldbox.rules` need.

    All the arithmetic lives here so that the scorers themselves stay short
    enough to read and edit without knowing how the simulation is wired.
    """
    from ..rules import Situation
    from .behavior import SocialContext, can_reproduce

    social = social or SocialContext()
    weights = config.utility
    needs = agent.needs
    agent_config = config.agents

    food_here = world.resources.food_at(agent.x, agent.y)
    threat = social.threat

    # Willingness to stand and fight, rather than a fixed health cutoff.
    courage = 0.0
    if threat is not None:
        relative_strength = needs.health / max(1.0, threat.needs.health)
        courage = clamp01(
            0.5 * clamp01(relative_strength)
            + 0.5 * agent.aggression
            - weights.caution_weight * agent.caution
        )
        if not agent.is_adult(agent_config, config.simulation.days_per_year):
            courage *= weights.child_courage  # Children rarely hold their ground.

    return Situation(
        agent=agent,
        world=world,
        config=config,
        day=day,
        hunger=curve(needs.hunger / 100.0, weights.hunger_exponent),
        tiredness=curve(1.0 - needs.energy / 100.0, weights.energy_exponent),
        frailty=1.0 - clamp01(needs.health / 100.0),
        comfort=clamp01(
            (1.0 - needs.hunger / 100.0)
            * (needs.energy / 100.0)
            * (needs.health / 100.0)
        ),
        food_here=food_here,
        can_eat=food_here >= config.resources.min_food_to_eat,
        threat=threat,
        courage=courage,
        can_breed=can_reproduce(agent, config, day, social.reproduction_cooldown_scale),
        is_adult=agent.is_adult(agent_config, config.simulation.days_per_year),
    )


def score_actions(
    agent: "Agent",
    world: "World",
    config: Config,
    day: int,
    social: Optional["SocialContext"] = None,
) -> Dict["Goal", float]:
    """Score every action this agent could take today, on 0..1.

    The scoring itself lives in :mod:`worldbox.rules`, which is the file meant
    to be edited. Each registered scorer is multiplied by its weight, and any
    scoring zero is dropped as "not an option right now".
    """
    from ..rules import SCORERS, WEIGHTS

    situation = build_situation(agent, world, config, day, social)

    scores: Dict[Goal, float] = {}
    for goal, score_function in SCORERS.items():
        try:
            raw = score_function(situation)
        except Exception:
            # A broken custom scorer should sideline that one action, not kill
            # the agent -- this file is meant to be edited by hand.
            continue
        if raw <= 0.0:
            continue
        scores[goal] = WEIGHTS.get(goal.value, 1.0) * raw

    if not scores:  # Every scorer declined; wandering is always possible.
        from .behavior import Goal as _Goal

        scores[_Goal.WANDER] = WEIGHTS.get("wander", 0.06)
    return scores


def choose(
    agent: "Agent",
    world: "World",
    config: Config,
    day: int,
    social: Optional["SocialContext"] = None,
) -> Decision:
    """Score every action and pick the best.

    Ties break toward the action declared first in :class:`Goal`, so the result
    is deterministic for a given world state.
    """
    scores = score_actions(agent, world, config, day, social)
    goal, score = max(scores.items(), key=lambda item: (item[1], -list(scores).index(item[0])))
    return Decision(goal=goal, score=score, scores=scores)
