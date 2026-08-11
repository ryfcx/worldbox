"""Technology: a small prerequisite tree that tribes advance through.

Knowledge belongs to a *tribe*, not an individual, which keeps the bookkeeping
tractable and makes technology a genuinely social phenomenon: a tribe that grows
and feeds itself researches faster, and a tribe that is wiped out takes its
knowledge with it.

Research points accumulate daily from population and food security. When a tribe
can afford one of the technologies whose prerequisites it already holds, it
invents it. Every technology contributes modifiers that are summed into a
:class:`TechEffects` for the whole tribe.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..config import TechnologyConfig


@dataclass(frozen=True)
class Technology:
    """One entry in the tech tree.

    The modifier fields are *additive bonuses* (0.25 = +25%) that are summed
    across everything a tribe knows.
    """

    id: str
    name: str
    era: str
    cost: float
    prerequisites: Tuple[str, ...] = ()

    food_yield: float = 0.0  # More nutrition from the same food.
    combat: float = 0.0  # Fighting strength.
    research: float = 0.0  # Speed of future research.
    health: float = 0.0  # Daily health recovery.
    fertility: float = 0.0  # Shorter gap between children.
    disease_resistance: float = 0.0  # Resistance to infection and its mortality.


# The tree itself. Roots have no prerequisites; each era gates the next.
TECH_TREE: Tuple[Technology, ...] = (
    # --- Stone Age --------------------------------------------------------
    Technology("fire", "Fire", "Stone Age", 960, (), food_yield=0.15, health=0.10),
    Technology("stone_tools", "Stone Tools", "Stone Age", 1080, (), food_yield=0.20, combat=0.15),
    Technology("shelter", "Shelter", "Stone Age", 1440, ("fire",), health=0.20, fertility=0.10),
    Technology(
        "hunting", "Organised Hunting", "Stone Age", 1920, ("stone_tools",),
        food_yield=0.25, combat=0.10,
    ),
    Technology(
        "burial", "Burial Rites", "Stone Age", 2280, ("shelter",),
        research=0.10, disease_resistance=0.15,
    ),
    # --- Agricultural Age -------------------------------------------------
    Technology(
        "agriculture", "Agriculture", "Agricultural Age", 3600, ("fire", "stone_tools"),
        food_yield=0.45, fertility=0.20,
    ),
    Technology(
        "pottery", "Pottery", "Agricultural Age", 4320, ("agriculture",),
        food_yield=0.20, disease_resistance=0.15,
    ),
    Technology(
        "weaving", "Weaving", "Agricultural Age", 4560, ("shelter",),
        health=0.15, fertility=0.10,
    ),
    Technology(
        "medicine", "Herbal Medicine", "Agricultural Age", 6240, ("burial", "weaving"),
        health=0.35, disease_resistance=0.35,
    ),
    Technology(
        "writing", "Writing", "Agricultural Age", 7200, ("pottery", "shelter"),
        research=0.35,
    ),
    # --- Bronze Age -------------------------------------------------------
    Technology(
        "bronze_working", "Bronze Working", "Bronze Age", 9120, ("agriculture", "pottery"),
        combat=0.35, food_yield=0.10,
    ),
    Technology(
        "the_wheel", "The Wheel", "Bronze Age", 10080, ("bronze_working",),
        food_yield=0.20, research=0.10,
    ),
    Technology(
        "mathematics", "Mathematics", "Bronze Age", 12000, ("writing",),
        research=0.40,
    ),
    Technology(
        "sailing", "Sailing", "Bronze Age", 12960, ("the_wheel", "weaving"),
        food_yield=0.25, research=0.10,
    ),
    # --- Iron Age ---------------------------------------------------------
    Technology(
        "iron_working", "Iron Working", "Iron Age", 16800, ("bronze_working", "mathematics"),
        combat=0.50,
    ),
    Technology(
        "engineering", "Engineering", "Iron Age", 21600, ("mathematics", "iron_working"),
        food_yield=0.30, health=0.20, research=0.20,
    ),
)

TECH_BY_ID: Dict[str, Technology] = {tech.id: tech for tech in TECH_TREE}
ERAS: Tuple[str, ...] = ("Stone Age", "Agricultural Age", "Bronze Age", "Iron Age")


@dataclass(frozen=True)
class TechEffects:
    """The summed modifiers granted by everything a tribe knows."""

    food_yield: float = 0.0
    combat: float = 0.0
    research: float = 0.0
    health: float = 0.0
    fertility: float = 0.0
    disease_resistance: float = 0.0


@dataclass
class Knowledge:
    """One tribe's technological state."""

    known: Set[str] = field(default_factory=set)
    research_points: float = 0.0
    effects: TechEffects = field(default_factory=TechEffects)

    def knows(self, tech_id: str) -> bool:
        """True if this tribe has invented the given technology."""
        return tech_id in self.known

    @property
    def era(self) -> str:
        """The most advanced era this tribe has reached."""
        best = ERAS[0]
        for tech_id in self.known:
            era = TECH_BY_ID[tech_id].era
            if ERAS.index(era) > ERAS.index(best):
                best = era
        return best

    def recompute_effects(self) -> None:
        """Re-sum the modifiers after learning something new."""
        totals = {
            "food_yield": 0.0,
            "combat": 0.0,
            "research": 0.0,
            "health": 0.0,
            "fertility": 0.0,
            "disease_resistance": 0.0,
        }
        for tech_id in self.known:
            tech = TECH_BY_ID[tech_id]
            for key in totals:
                totals[key] += getattr(tech, key)
        self.effects = TechEffects(**totals)

    def learn(self, tech_id: str) -> None:
        """Add a technology and refresh the tribe's modifiers."""
        self.known.add(tech_id)
        self.recompute_effects()


def available_technologies(knowledge: Knowledge) -> List[Technology]:
    """Technologies whose prerequisites are met but which are not yet known.

    Returned in tree order, so selection is deterministic.
    """
    return [
        tech
        for tech in TECH_TREE
        if tech.id not in knowledge.known
        and all(prerequisite in knowledge.known for prerequisite in tech.prerequisites)
    ]


def daily_research(
    knowledge: Knowledge,
    population: int,
    average_hunger: float,
    config: TechnologyConfig,
) -> float:
    """Research points earned by a tribe in one day.

    Scales sub-linearly with population (a tribe twice the size is not twice as
    inventive) and falls off as the tribe goes hungry.
    """
    if population < config.min_population_to_research:
        return 0.0
    hunger_factor = 1.0 - config.hunger_research_penalty * min(1.0, average_hunger / 100.0)
    return (
        config.base_research_rate
        * (population**config.population_exponent)
        * (1.0 + knowledge.effects.research)
        * max(0.0, hunger_factor)
    )


def try_invent(
    knowledge: Knowledge,
    population: int,
    average_hunger: float,
    config: TechnologyConfig,
    rng: random.Random,
) -> Optional[Technology]:
    """Accumulate a day of research and invent something if affordable.

    Returns the technology invented, or ``None``. Cheaper technologies are
    weighted more heavily, so tribes tend to broaden before they deepen.
    """
    knowledge.research_points += daily_research(knowledge, population, average_hunger, config)

    candidates = [
        tech for tech in available_technologies(knowledge) if tech.cost <= knowledge.research_points
    ]
    if not candidates:
        return None

    weights = [1.0 / tech.cost for tech in candidates]
    chosen = rng.choices(candidates, weights=weights, k=1)[0]
    knowledge.research_points -= chosen.cost
    knowledge.learn(chosen.id)
    return chosen


def world_technology_summary(all_knowledge: Sequence[Knowledge]) -> Dict[str, int]:
    """How many tribes know each technology, keyed by technology id."""
    counts: Dict[str, int] = {}
    for knowledge in all_knowledge:
        for tech_id in knowledge.known:
            counts[tech_id] = counts.get(tech_id, 0) + 1
    return counts
