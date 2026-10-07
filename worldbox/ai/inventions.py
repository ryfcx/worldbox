"""Custom inventions proposed by tribe minds, paid for with research.

The stock tech tree still runs. On top of it, a mind can name one new craft
built from materials that are actually around the tribe -- river, forest,
granary, a war, a technology they already hold. Research points buy it the
same way they buy Fire. The modifiers land in :class:`TechEffects`, so a
tribe that invents fish traps really does eat better.

Invalid proposals are dropped, not applied. A mind cannot will iron out of
grassland.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import AIConfig
from ..society.technology import (
    ERAS,
    TECH_BY_ID,
    TECH_TREE,
    Technology,
    available_technologies,
    register_technology,
)

PHYSICAL = ("grass", "forest", "mountain", "water", "food", "settlement", "war", "plague")
EFFECT_ALIASES = {
    "food": "food_yield",
    "food_yield": "food_yield",
    "farm": "food_yield",
    "fish": "food_yield",
    "combat": "combat",
    "fight": "combat",
    "war": "combat",
    "weapon": "combat",
    "weapons": "combat",
    "research": "research",
    "knowledge": "research",
    "learn": "research",
    "health": "health",
    "heal": "health",
    "healing": "health",
    "fertility": "fertility",
    "children": "fertility",
    "disease": "disease_resistance",
    "medicine": "disease_resistance",
    "plague": "disease_resistance",
}
STOCK_NAMES = {tech.name.lower() for tech in TECH_TREE}


@dataclass
class InventionProposal:
    """One custom craft a mind wants its people to work toward."""

    name: str
    tech_id: str
    materials: List[str]
    effect: str
    bonus: float
    cost: float
    era: str
    why: str = ""


@dataclass
class MaterialSurvey:
    """What a tribe can currently honestly claim to work with."""

    materials: List[str]
    terrain: Dict[str, int]
    food_nearby: float
    granary: float
    research_points: float
    available_stock: List[Tuple[str, float]]


def slugify(name: str) -> str:
    """Stable id for a custom technology, e.g. ``custom_fish_traps``."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    slug = slug[:40] or "craft"
    return f"custom_{slug}"


def scan_materials(engine: Any, group: Any, radius: int = 8) -> MaterialSurvey:
    """Walk the ground around a tribe and list what they could craft from."""
    settlement = engine.settlements.for_group(group.id)
    origin = settlement.position if settlement is not None else group.centre
    terrain: Dict[str, int] = {}
    food = 0.0
    for x, y in engine.world.tiles_within(origin[0], origin[1], radius):
        tile = engine.world.terrain_at(x, y)
        if tile is not None:
            terrain[tile.value] = terrain.get(tile.value, 0) + 1
        food += engine.world.resources.food_at(x, y)

    found: List[str] = []
    if terrain.get("grass", 0) >= 4:
        found.append("grass")
    if terrain.get("forest", 0) >= 3:
        found.append("forest")
    if terrain.get("mountain", 0) >= 2:
        found.append("mountain")
    if terrain.get("water", 0) >= 3:
        found.append("water")
    granary = settlement.food_store if settlement is not None else 0.0
    if food >= 8.0 or granary >= 12.0:
        found.append("food")
    if settlement is not None:
        found.append("settlement")
    at_war = any(
        war.active for key, war in engine.diplomacy.wars.items() if group.id in key
    )
    if at_war:
        found.append("war")
    if engine.diseases.active_outbreaks():
        found.append("plague")
    for tech_id in sorted(group.knowledge.known):
        if tech_id in TECH_BY_ID and not tech_id.startswith("custom_"):
            found.append(tech_id)

    stock = [
        (tech.name, tech.cost)
        for tech in available_technologies(group.knowledge)
    ][:6]
    return MaterialSurvey(
        materials=found,
        terrain=terrain,
        food_nearby=food,
        granary=granary,
        research_points=group.knowledge.research_points,
        available_stock=stock,
    )


def parse_invention(raw: Any, era: str, research_stance: bool) -> Optional[InventionProposal]:
    """Read an ``invent`` object out of a mind's JSON, or None if it is junk."""
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name", "")).strip()
    name = re.sub(r"\s+", " ", name)
    if not 2 <= len(name) <= 32:
        return None
    if len(name.split()) > 5:
        return None
    if name.lower() in STOCK_NAMES:
        return None
    if not re.search(r"[A-Za-z]", name):
        return None

    listed = raw.get("from") or raw.get("materials") or []
    if isinstance(listed, str):
        listed = [listed]
    materials: List[str] = []
    if isinstance(listed, list):
        for item in listed:
            token = str(item).strip().lower().replace(" ", "_")
            if token and token not in materials:
                materials.append(token)
    if not materials:
        return None

    effect = EFFECT_ALIASES.get(str(raw.get("effect", "food")).strip().lower())
    if effect is None:
        return None

    why = str(raw.get("why", "")).strip().replace("\n", " ")
    if len(why) > 140:
        why = why[:137] + "..."

    bonus = 0.10 + 0.03 * min(4, len(materials))
    if research_stance:
        bonus += 0.02
    bonus = min(0.28, bonus)
    # Local crafts are cheaper than stock techs like Fire (960): they should
    # show up in a live run, not after a decade of saving.
    cost = 240.0 + 1400.0 * bonus
    if research_stance:
        cost *= 0.8

    era_name = era if era in ERAS else ERAS[0]
    return InventionProposal(
        name=name.title(),
        tech_id=slugify(name),
        materials=materials[:5],
        effect=effect,
        bonus=round(bonus, 3),
        cost=round(cost, 1),
        era=era_name,
        why=why,
    )


def custom_count(group: Any) -> int:
    """How many custom crafts this tribe already knows."""
    return sum(1 for tech_id in group.knowledge.known if str(tech_id).startswith("custom_"))


def validate_proposal(
    proposal: InventionProposal,
    group: Any,
    survey: MaterialSurvey,
    config: AIConfig,
) -> Optional[str]:
    """Return an error string if this invention cannot be worked on, else None."""
    if not config.allow_inventions:
        return "inventions disabled"
    if custom_count(group) >= config.max_custom_techs:
        return "this tribe already has enough unique crafts"
    allowed: Set[str] = set(survey.materials)
    if not proposal.materials or not set(proposal.materials) <= allowed:
        return "materials are not actually at hand"
    if not any(item in PHYSICAL or item in allowed for item in proposal.materials):
        return "needs a real material"
    if proposal.tech_id in group.knowledge.known:
        return "already known"
    existing = TECH_BY_ID.get(proposal.tech_id)
    if existing is not None and not existing.id.startswith("custom_"):
        return "collides with the stock tree"
    return None


def realize(
    group: Any,
    proposal: InventionProposal,
    survey: MaterialSurvey,
    config: AIConfig,
) -> Optional[Technology]:
    """Spend research and learn the custom tech if the tribe can still afford it."""
    error = validate_proposal(proposal, group, survey, config)
    if error:
        return None
    if group.knowledge.research_points < proposal.cost:
        return None
    kwargs = {proposal.effect: proposal.bonus}
    tech = Technology(
        id=proposal.tech_id,
        name=proposal.name,
        era=proposal.era,
        cost=proposal.cost,
        **kwargs,
    )
    register_technology(tech)
    group.knowledge.research_points -= proposal.cost
    group.knowledge.learn(tech.id)
    return tech


def briefing_lines(survey: MaterialSurvey) -> List[str]:
    """Extra prompt lines so a mind knows what it is allowed to invent."""
    lines = [
        "Materials at hand: " + (", ".join(survey.materials) if survey.materials else "none") + ".",
        f"Research stored: {survey.research_points:.0f} points. Granary {survey.granary:.0f}. "
        f"Food nearby {survey.food_nearby:.0f}.",
    ]
    if survey.available_stock:
        stock = ", ".join(f"{name} ({cost:.0f} pts)" for name, cost in survey.available_stock)
        lines.append("Stock tech available: " + stock + ".")
    else:
        lines.append("Stock tech available: none yet (keep accumulating research).")
    lines.append(
        'You may set "invent" to ONE new craft using ONLY materials at hand, '
        "or null if you cannot honestly make anything new. "
        'Example: {"name":"Fish Traps","from":["water","forest"],"effect":"food",'
        '"why":"The river can feed us if we weave traps."}'
    )
    return lines
