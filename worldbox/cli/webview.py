"""A live map in the browser, served from the running simulation.

The recorded viewer replays a finished run. This serves the world *as it is
happening*: a tiny standard-library HTTP server hands the page a JSON snapshot
whenever it asks, and the page redraws. Start it at day zero and watch the whole
history unfold.

The separation still holds. The server only reads the engine through the same
lock the terminal uses, and the engine has no idea it exists.

Endpoints:
    GET /            the page itself
    GET /world       terrain and dimensions, fetched once
    GET /state       the current snapshot: agents, settlements, stats, log
    GET /control     pause, resume or change speed
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from ..simulation.engine import SimulationEngine, SimulationRunner
from ..simulation.events import EventKind
from ..society.technology import lookup_tech
from ..world.terrain import TerrainType

TERRAIN_CODES: Dict[str, int] = {
    TerrainType.WATER.value: 0,
    TerrainType.GRASS.value: 1,
    TerrainType.FOREST.value: 2,
    TerrainType.MOUNTAIN.value: 3,
}

ROLE_CODES: Dict[str, int] = {
    "child": 0,
    "forager": 1,
    "farmer": 2,
    "warrior": 3,
    "scholar": 4,
    "healer": 5,
    "elder": 6,
}
HOUSE_CODES: Dict[str, int] = {"": 0, "deepseek": 1, "openai": 2}


# Map sprites for the things a tribe actually builds. The hall itself is drawn
# from the settlement level; these are the extra structures around it.
STRUCTURE_HUT = 0
STRUCTURE_FARM = 1
STRUCTURE_GRANARY = 2
STRUCTURE_WORKSHOP = 3
STRUCTURE_WALL = 4
STRUCTURE_SHRINE = 5
STRUCTURE_DOCK = 6
STRUCTURE_CRAFT = 7


def _structure_spots(world: Any, x: int, y: int, radius: int, taken: set) -> List[Tuple[int, int]]:
    """Passable tiles around a settlement, nearest ring first."""
    spots: List[Tuple[int, int]] = []
    for ring in range(1, radius + 1):
        for ty in range(y - ring, y + ring + 1):
            for tx in range(x - ring, x + ring + 1):
                if max(abs(tx - x), abs(ty - y)) != ring:
                    continue
                if (tx, ty) in taken or not world.is_passable(tx, ty):
                    continue
                spots.append((tx, ty))
    return spots


def _plan_structures(
    world: Any,
    settlement: Any,
    known: set,
    names: Dict[str, str],
    taken: set,
) -> List[Tuple[int, int, int, str]]:
    """Buildings this settlement should show, in stable positions."""
    level = int(settlement.level)
    wants: List[Tuple[int, str]] = []
    huts = min(6, 1 + level + (1 if "shelter" in known else 0))
    wants.extend((STRUCTURE_HUT, "Hut") for _ in range(huts))
    if "agriculture" in known:
        wants.extend((STRUCTURE_FARM, "Farm") for _ in range(2 + (1 if level >= 2 else 0)))
    if "pottery" in known or level >= 2:
        wants.append((STRUCTURE_GRANARY, "Granary"))
    if "weaving" in known:
        wants.append((STRUCTURE_WORKSHOP, "Loom"))
    if "bronze_working" in known or "iron_working" in known:
        wants.append((STRUCTURE_WORKSHOP, "Workshop"))
    if "burial" in known:
        wants.append((STRUCTURE_SHRINE, "Shrine"))
    if "engineering" in known or "iron_working" in known:
        wants.extend((STRUCTURE_WALL, "Wall") for _ in range(4))
    if "sailing" in known:
        wants.append((STRUCTURE_DOCK, "Dock"))
    for tech_id in sorted(tech for tech in known if str(tech).startswith("custom_"))[:4]:
        wants.append((STRUCTURE_CRAFT, names.get(tech_id, "Craft")))

    radius = min(4, 2 + level)
    spots = _structure_spots(world, settlement.x, settlement.y, radius, taken)
    if not spots:
        return []

    def _near_water(spot: Tuple[int, int]) -> bool:
        sx, sy = spot
        for nx, ny in world.neighbours(sx, sy, passable_only=False):
            tile = world.terrain_at(nx, ny)
            if tile is not None and tile.value == "water":
                return True
        return False

    placed: List[Tuple[int, int, int, str]] = []
    used: set = set()
    dock = next(((i, item) for i, item in enumerate(wants) if item[0] == STRUCTURE_DOCK), None)
    if dock is not None:
        index, item = dock
        wants.pop(index)
        water_spots = [spot for spot in spots if _near_water(spot)]
        if water_spots:
            spot = water_spots[0]
            used.add(spot)
            placed.append((spot[0], spot[1], item[0], item[1]))

    walls = [item for item in wants if item[0] == STRUCTURE_WALL]
    rest = [item for item in wants if item[0] != STRUCTURE_WALL]
    free = [spot for spot in spots if spot not in used]
    outer = list(reversed(free))
    for item, spot in zip(walls, outer):
        used.add(spot)
        placed.append((spot[0], spot[1], item[0], item[1]))
    free = [spot for spot in free if spot not in used]
    for item, spot in zip(rest, free):
        placed.append((spot[0], spot[1], item[0], item[1]))
    for x, y, _kind, _name in placed:
        taken.add((x, y))
    taken.add((settlement.x, settlement.y))
    return placed


def _pack_look(agent: Any, is_chief: bool) -> int:
    """Pack role, LLM house and flags into one int for the live map sprites."""
    role = ROLE_CODES.get(getattr(agent, "role", "") or "forager", 1)
    house = HOUSE_CODES.get(getattr(agent, "ai_house", "") or "", 0)
    flags = 0
    if is_chief:
        flags |= 1
    if getattr(agent, "is_ill", False):
        flags |= 2
    health = getattr(getattr(agent, "needs", None), "health", 100.0)
    if health < 35:
        flags |= 4
    return role | (house << 4) | (flags << 8)

# Event kinds worth showing in the live feed.
LOGGED_KINDS = ("birth", "death", "war", "plague", "invention", "society", "milestone", "mind")


class LiveState:
    """Reads snapshots out of a running engine, safely.

    Tribe colour indices are allocated on first sight and kept, so a tribe keeps
    its colour for as long as it exists.
    """

    def __init__(self, engine: SimulationEngine, runner: SimulationRunner) -> None:
        self.engine = engine
        self.runner = runner
        self.tribe_index: Dict[int, int] = {}
        self.log: List[Dict[str, Any]] = []
        self._next_index = 0
        engine.events.subscribe(self._on_event)

    def _on_event(self, event) -> None:
        """Keep a rolling window of notable events for the feed."""
        if event.kind.value not in LOGGED_KINDS:
            return
        self.log.append({"d": event.day, "k": event.kind.value, "m": event.message})
        if len(self.log) > 400:
            del self.log[:-300]

    def _index_for(self, group_id: Optional[int]) -> int:
        if group_id is None:
            return -1
        if group_id not in self.tribe_index:
            self.tribe_index[group_id] = self._next_index
            self._next_index += 1
        return self.tribe_index[group_id]

    def world(self) -> Dict[str, Any]:
        """Terrain and dimensions. Constant for the life of a world."""
        with self.runner.lock:
            grid = self.engine.world.grid
            from ..ai.client import available_houses

            return {
                "width": self.engine.world.width,
                "height": self.engine.world.height,
                "seed": self.engine.seed,
                "houses": [
                    {"house": spec["house"], "label": spec["label"]}
                    for spec in available_houses()
                ],
                "terrain": [
                    "".join(str(TERRAIN_CODES[tile.value]) for tile in row) for row in grid
                ],
            }

    def snapshot(self, since_day: int = -1) -> Dict[str, Any]:
        """Everything the page needs to draw one frame."""
        with self.runner.lock:
            engine = self.engine
            stats = engine.stats()

            chiefs = {
                group.chieftain_id
                for group in engine.groups.active()
                if group.chieftain_id is not None
            }
            agents: List[int] = []
            for agent in engine.agents:
                agents.extend(
                    (
                        agent.x,
                        agent.y,
                        self._index_for(agent.group_id),
                        _pack_look(agent, agent.id in chiefs),
                    )
                )

            settlements: List[int] = []
            structures: List[Any] = []
            taken: set = set()
            for settlement in engine.settlements.active():
                color = self._index_for(settlement.group_id)
                settlements.extend(
                    (settlement.x, settlement.y, int(settlement.level), color)
                )
                group = engine.groups.get(settlement.group_id)
                known = group.knowledge.known if group is not None else set()
                labels = {
                    tech_id: (lookup_tech(tech_id).name if lookup_tech(tech_id) else tech_id)
                    for tech_id in known
                }
                for sx, sy, kind, label in _plan_structures(
                    engine.world, settlement, known, labels, taken
                ):
                    structures.extend((sx, sy, kind, color, label))

            fresh = [entry for entry in self.log if entry["d"] > since_day][-60:]
            capacity = stats.food_capacity or 1.0
            from ..ai.client import ENV_KEY, OPENAI_ENV_KEY, available_houses, is_available

            minds = []
            civs = []
            for row in engine.minds.snapshot(engine):
                group = engine.groups.get(row.group_id)
                settlement = engine.settlements.for_group(row.group_id) if group else None
                entry = {
                    "id": row.group_id,
                    "name": row.name,
                    "stance": row.stance,
                    "intent": row.intent,
                    "inventing": row.inventing,
                    "thinking": row.thinking,
                    "error": row.error,
                    "house": row.house,
                    "model": row.model,
                    "color": self._index_for(row.group_id),
                    "pop": group.size if group else 0,
                    "era": group.era if group else "",
                    "x": (settlement.x if settlement else group.centre[0]) if group else 0,
                    "y": (settlement.y if settlement else group.centre[1]) if group else 0,
                    "town": settlement.name if settlement else "",
                    "research": round(group.knowledge.research_points) if group else 0,
                }
                minds.append(entry)
                civs.append(entry)
            houses = [
                {"house": spec["house"], "label": spec["label"]}
                for spec in available_houses()
            ]
            return {
                "day": stats.day,
                "year": stats.year,
                "running": self.runner.running,
                "population": stats.population,
                "tribes": stats.tribes,
                "settlements": stats.settlements,
                "techs": stats.technologies_known,
                "era": stats.most_advanced_era,
                "wars": stats.active_wars,
                "ill": stats.ill,
                "warDeaths": stats.war_deaths,
                "plagueDeaths": stats.plague_deaths,
                "births": stats.total_births,
                "deaths": stats.total_deaths,
                "health": round(stats.average_health, 1),
                "hunger": round(stats.average_hunger, 1),
                "food": round(stats.total_food / capacity, 3),
                "agents": agents,
                "settlementList": settlements,
                "structures": structures,
                "log": fresh,
                "discoveries": list(engine.discoveries[-16:]),
                "civs": civs,
                "houses": houses,
                "delay": self.runner.delay,
                "ai": {
                    "available": is_available() and engine.config.ai.enabled,
                    "key": ENV_KEY,
                    "openaiKey": OPENAI_ENV_KEY,
                    "minds": stats.ai_minds,
                    "thinking": stats.ai_thinking,
                    "tribes": minds,
                    "houses": houses,
                },
            }

    def inspect(self, x: int, y: int) -> Dict[str, Any]:
        """Describe whatever is at a tile: settlement, tribe and people."""
        with self.runner.lock:
            engine = self.engine
            dpy = engine.config.simulation.days_per_year
            result: Dict[str, Any] = {"x": x, "y": y}

            terrain = engine.world.terrain_at(x, y)
            result["terrain"] = terrain.label if terrain else "void"
            result["food"] = round(engine.world.resources.food_at(x, y), 1)
            if 0 <= y < engine.world.height and 0 <= x < engine.world.width:
                result["fertility"] = round(engine.world.resources.fertility[y][x], 2)

            # Nearest settlement within a few tiles, so clicking is forgiving.
            best, best_distance = None, 4
            for settlement in engine.settlements.active():
                d = max(abs(settlement.x - x), abs(settlement.y - y))
                if d <= best_distance:
                    best, best_distance = settlement, d

            if best is not None:
                from ..ai.client import house_spec

                group = engine.groups.get(best.group_id)
                chief = engine.get_agent(group.chieftain_id) if group and group.chieftain_id else None
                house = group.ai_house if group else ""
                result["settlement"] = {
                    "name": best.name,
                    "level": best.level_name,
                    "founded": best.founded_day,
                    "store": round(best.food_store),
                    "storePct": round(best.store_fraction() * 100),
                    "tribe": group.name if group else "abandoned",
                    "population": group.size if group else 0,
                    "era": group.era if group else "",
                    "house": house,
                    "model": house_spec(house)["label"] if house else "",
                    "research": round(group.knowledge.research_points) if group else 0,
                    "techs": [
                        (tech.name if tech else tech_id)
                        for tech_id in sorted(group.knowledge.known)
                        for tech in [lookup_tech(tech_id)]
                    ] if group else [],
                    "chieftain": chief.full_name if chief else None,
                    "chieftainAge": round(chief.age_years(dpy)) if chief else None,
                    "battlesWon": group.battles_won if group else 0,
                    "battlesLost": group.battles_lost if group else 0,
                    "warDead": group.war_dead if group else 0,
                    "plagueDead": group.plague_dead if group else 0,
                    "roles": {k: v for k, v in (group.role_counts or {}).items() if v} if group else {},
                    "structures": [],
                    "mind": None,
                }
                if group is not None:
                    labels = {
                        tech_id: (lookup_tech(tech_id).name if lookup_tech(tech_id) else tech_id)
                        for tech_id in group.knowledge.known
                    }
                    built = _plan_structures(
                        engine.world, best, group.knowledge.known, labels, set()
                    )
                    result["settlement"]["structures"] = [name for _x, _y, _k, name in built]
                if group is not None:
                    policy = engine.minds.policy_for(group.id)
                    if policy is not None:
                        result["settlement"]["mind"] = {
                            "stance": policy.stance,
                            "intent": policy.intent,
                            "inventing": policy.invent.name if policy.invent else "",
                            "fight": policy.fight,
                            "flee": policy.flee,
                            "house": group.ai_house,
                        }

            # A couple of the people standing here.
            people = []
            for agent in engine.agents:
                if max(abs(agent.x - x), abs(agent.y - y)) <= 1:
                    tribe = engine.groups.get(agent.group_id)
                    people.append({
                        "id": agent.id,
                        "name": agent.full_name,
                        "age": round(agent.age_years(dpy)),
                        "role": agent.role,
                        "goal": agent.goal,
                        "health": round(agent.needs.health),
                        "hunger": round(agent.needs.hunger),
                        "energy": round(agent.needs.energy),
                        "house": agent.ai_house,
                        "family": agent.family_name,
                        "chief": bool(tribe and tribe.chieftain_id == agent.id),
                        "tribe": tribe.name if tribe else None,
                        "ill": agent.infection.disease.name if agent.infection else None,
                    })
                    if len(people) >= 8:
                        break
            result["people"] = people
            return result

    def explain(self) -> Dict[str, Any]:
        """Ask the AI what has been happening, from the chronicle.

        Runs outside the simulation lock: it is a network call taking seconds,
        and holding the lock would freeze the world while it waited.
        """
        from ..ai import narrator

        if not narrator.is_available():
            return {"ok": False, "error":
                    f"No API key. Put {narrator.ENV_KEY} in .env or export it."}
        with self.runner.lock:
            entries = list(self.engine.chronicle.entries)
            year = self.engine.clock.year
        if not entries:
            return {"ok": False, "error": "Nothing has happened yet."}
        result = narrator.narrate(entries, max_entries=80)
        if not result.ok:
            return {"ok": False, "error": result.error}
        return {"ok": True, "year": year, "text": result.text}

    def ask(self, question: str) -> Dict[str, Any]:
        """Answer one question about the world from the chronicle and a briefing.

        The network call runs outside the simulation lock.
        """
        from ..ai import narrator

        cleaned = " ".join((question or "").split())[:600]
        if not cleaned:
            return {"ok": False, "error": "Type a question first."}
        if not narrator.is_available():
            return {
                "ok": False,
                "error": f"No API key. Put {narrator.ENV_KEY} or OPENAI_API_KEY in .env.",
            }
        with self.runner.lock:
            engine = self.engine
            stats = engine.stats()
            entries = list(engine.chronicle.entries)
            lines = [
                f"Year {engine.clock.year}, day {engine.clock.day}.",
                f"Population {stats.population}. Tribes {stats.tribes}. "
                f"Settlements {stats.settlements}. Wars {stats.active_wars}. "
                f"Ill {stats.ill}. Most advanced era: {stats.most_advanced_era}. "
                f"Distinct technologies: {stats.technologies_known}.",
                "Tribes:",
            ]
            for group in engine.groups.active()[:24]:
                policy = engine.minds.policy_for(group.id)
                settlement = engine.settlements.for_group(group.id)
                line = f"- {group.name}: {group.size} people, {group.era}"
                if settlement is not None:
                    line += f", home {settlement.name} ({settlement.level_name})"
                if policy is not None and policy.stance:
                    line += f", stance {policy.stance}"
                if policy is not None and policy.intent:
                    line += f". Intent: {policy.intent}"
                if policy is not None and policy.invent is not None:
                    line += f". Working on {policy.invent.name}"
                lines.append(line)
            if engine.discoveries:
                lines.append("Recent inventions:")
                for row in engine.discoveries[-12:]:
                    lines.append(
                        f"- Year {row.get('year')}: {row.get('tribe')} invented {row.get('name')}"
                    )
            recent = self.log[-24:]
            if recent:
                lines.append("Recent events:")
                lines.extend(f"- Day {entry['d']} ({entry['k']}): {entry['m']}" for entry in recent)
            briefing = "\n".join(lines)
            year = engine.clock.year
        result = narrator.answer(cleaned, entries, briefing)
        if not result.ok:
            return {"ok": False, "error": result.error}
        return {"ok": True, "year": year, "question": cleaned, "text": result.text, "model": result.model}

    def act(self, tool: str, x: int, y: int, radius: int = 3) -> Dict[str, Any]:
        """Apply a god-tool to the world.

        These deliberately break determinism -- a seed no longer reproduces a
        run once you have interfered with it. That is the trade for being able
        to poke the world and see what happens, and it is why the simulation
        itself never calls any of this.
        """
        from ..agents.agent import create_agent
        from ..agents.needs import Needs
        from ..society.epidemics import DISEASES, Outbreak

        with self.runner.lock:
            engine = self.engine
            world = engine.world
            config = engine.config
            rng = engine.rng
            day = engine.clock.day
            affected = 0

            def tiles():
                return list(world.tiles_within(x, y, radius))

            if tool == "spawn":
                for _ in range(6):
                    spot = [(tx, ty) for tx, ty in tiles() if world.is_passable(tx, ty)]
                    if not spot:
                        break
                    tx, ty = rng.choice(spot)
                    agent = create_agent(
                        agent_id=engine._next_agent_id,
                        x=tx, y=ty,
                        age_days=int(rng.uniform(16, 30) * config.simulation.days_per_year),
                        rng=rng, config=config.agents,
                        days_per_year=config.simulation.days_per_year, day=day,
                        needs=Needs(hunger=20.0, energy=90.0, health=100.0),
                    )
                    from ..ai.client import house_for_index

                    agent.ai_house = house_for_index(agent.id)
                    engine._next_agent_id += 1
                    engine.agents.append(agent)
                    affected += 1
                engine.events.record(day, EventKind.SYSTEM,
                                     f"{affected} people appeared at ({x}, {y})")

            elif tool == "bless":
                ceiling = config.settlements.max_fertility + 1.0
                for tx, ty in tiles():
                    world.resources.cultivate(tx, ty, 0.6, ceiling)
                    cap = world.resources.effective_capacity(tx, ty)
                    if cap > 0:
                        world.resources.food[ty][tx] = cap
                        affected += 1
                engine.events.record(day, EventKind.SYSTEM,
                                     f"The land around ({x}, {y}) flourished")

            elif tool == "blight":
                for tx, ty in tiles():
                    if 0 <= ty < world.height and 0 <= tx < world.width:
                        world.resources.food[ty][tx] = 0.0
                        world.resources.fertility[ty][tx] = 1.0
                        affected += 1
                engine.events.record(day, EventKind.SYSTEM,
                                     f"The land around ({x}, {y}) withered")

            elif tool == "smite":
                for agent in engine.agents:
                    if max(abs(agent.x - x), abs(agent.y - y)) <= radius:
                        agent.needs.health = 0.0
                        agent.wounded_on_day = day
                        affected += 1
                engine.events.record(day, EventKind.SYSTEM,
                                     f"Fire fell on ({x}, {y}); {affected} died")

            elif tool == "plague":
                nearby = [a for a in engine.agents
                          if max(abs(a.x - x), abs(a.y - y)) <= radius]
                disease = rng.choice(DISEASES)
                candidates = [a for a in nearby if a.is_susceptible_to(disease.id, day)]
                if candidates:
                    outbreak = Outbreak(
                        id=engine.diseases._next_id, disease_id=disease.id,
                        started_day=day, origin=(x, y), infections=1,
                    )
                    engine.diseases.outbreaks[outbreak.id] = outbreak
                    engine.diseases._next_id += 1
                    patient = rng.choice(candidates)
                    patient.infect(disease.id, disease.duration_days, day, outbreak.id)
                    affected = 1
                    engine.events.record(day, EventKind.PLAGUE,
                                         f"{disease.name} was loosed at ({x}, {y})")

            elif tool == "heal":
                for agent in engine.agents:
                    if max(abs(agent.x - x), abs(agent.y - y)) <= radius:
                        agent.needs.health = 100.0
                        agent.needs.hunger = 0.0
                        agent.needs.energy = 100.0
                        if agent.infection is not None:
                            agent.recover_from_illness(day, config.disease.immunity_days)
                        affected += 1
                engine.events.record(day, EventKind.SYSTEM,
                                     f"{affected} people were restored at ({x}, {y})")
            else:
                return {"ok": False, "error": f"Unknown tool: {tool}"}

            return {"ok": True, "tool": tool, "affected": affected}

    def found_civ(
        self,
        x: int,
        y: int,
        name: str,
        house: str,
        people: int = 10,
    ) -> Dict[str, Any]:
        """Plant a named civilisation at a tile, played by a chosen LLM house."""
        from ..agents.agent import create_agent
        from ..agents.needs import Needs
        from ..ai.client import available_houses, house_for_index, house_spec

        cleaned = "".join(
            ch for ch in name.strip()[:40] if ch.isalnum() or ch in " -'"
        ).strip()
        count = max(6, min(18, int(people)))
        requested = (house or "").strip().lower()
        roster = available_houses()
        mixed = requested in {"mix", "mixed", "both"}
        chosen = ""
        if mixed:
            chosen = house_for_index(x + y + count) if roster else ""
        elif requested:
            spec = house_spec(requested)
            if roster and any(row["house"] == spec["house"] for row in roster):
                chosen = spec["house"]
            elif roster:
                chosen = roster[0]["house"]
        elif roster:
            chosen = roster[0]["house"]

        with self.runner.lock:
            engine = self.engine
            world = engine.world
            if not world.in_bounds(x, y):
                return {"ok": False, "error": "That tile is off the map."}
            if not world.is_passable(x, y):
                nearby = [
                    (tx, ty)
                    for tx, ty in world.tiles_within(x, y, 4)
                    if world.is_passable(tx, ty)
                ]
                if not nearby:
                    return {"ok": False, "error": "No walkable ground here."}
                x, y = nearby[0]

            day = engine.clock.day
            rng = engine.rng
            dpy = engine.config.simulation.days_per_year
            group = engine.groups.create(day, rng)
            if cleaned:
                kind = group.name.split()[-1] if " " in group.name else "Folk"
                group.name = cleaned if cleaned.lower().endswith(
                    ("tribe", "clan", "folk", "kin", "host")
                ) else f"{cleaned} {kind}"
            group.ai_house = chosen
            group.centre = (x, y)

            spots = [
                (tx, ty)
                for tx, ty in world.tiles_within(x, y, 2)
                if world.is_passable(tx, ty)
            ] or [(x, y)]
            family = cleaned.split()[0] if cleaned else group.name.split()[0]
            spawned = []
            for index in range(count):
                tx, ty = spots[index % len(spots)]
                agent = create_agent(
                    agent_id=engine._next_agent_id,
                    x=tx,
                    y=ty,
                    age_days=int(rng.uniform(16, 32) * dpy),
                    rng=rng,
                    config=engine.config.agents,
                    days_per_year=dpy,
                    day=day,
                    group_id=group.id,
                    inherited_family=family,
                    needs=Needs(hunger=12.0, energy=95.0, health=100.0),
                )
                if mixed and roster:
                    agent.ai_house = roster[index % len(roster)]["house"]
                else:
                    agent.ai_house = chosen or house_for_index(agent.id)
                engine._next_agent_id += 1
                engine.agents.append(agent)
                spawned.append(agent)
            group.member_ids = [agent.id for agent in spawned]
            group.chieftain_id = spawned[0].id
            spawned[0].role = "warrior"

            ceiling = engine.config.settlements.max_fertility + 1.0
            for tx, ty in world.tiles_within(x, y, 3):
                world.resources.cultivate(tx, ty, 0.5, ceiling)
                cap = world.resources.effective_capacity(tx, ty)
                if cap > 0:
                    world.resources.food[ty][tx] = cap

            settlement = engine.settlements.found_at(group, x, y, day, rng, food_store=50.0)
            label = house_spec(chosen)["label"] if chosen else "utility rules"
            town = settlement.name if settlement else "open ground"
            engine.events.record(
                day,
                EventKind.SOCIETY,
                f"You founded the {group.name} at {town} ({label}, {count} people)",
            )
            if settlement is not None:
                engine.chronicle.note_settlement(
                    day,
                    engine.clock.year,
                    settlement.id,
                    settlement.name,
                    int(settlement.level),
                    settlement.level_name,
                    group.name,
                )
            return {
                "ok": True,
                "id": group.id,
                "name": group.name,
                "house": chosen,
                "model": label,
                "people": count,
                "x": x,
                "y": y,
            }

    def overlay(self, kind: str) -> Dict[str, Any]:
        """A per-tile data layer for the map: fertility, food or crowding."""
        with self.runner.lock:
            world = self.engine.world
            res = world.resources
            rows: List[List[float]] = []
            if kind == "fertility":
                rows = [[round(v, 2) for v in row] for row in res.fertility]
            elif kind == "food":
                rows = [
                    [round(res.food[y][x] / c, 2) if (c := res.capacity[y][x]) > 0 else 0.0
                     for x in range(world.width)]
                    for y in range(world.height)
                ]
            elif kind == "territory":
                grid = [[-1] * world.width for _ in range(world.height)]
                for settlement in self.engine.settlements.active():
                    group = self.engine.groups.get(settlement.group_id)
                    reach = 6 + (2 * int(settlement.level))
                    idx = self._index_for(settlement.group_id)
                    for tx, ty in world.tiles_within(settlement.x, settlement.y, reach):
                        grid[ty][tx] = idx
                return {"kind": kind, "rows": grid}
            return {"kind": kind, "rows": rows}

    def control(self, action: str, value: Optional[str]) -> Dict[str, Any]:
        """Pause, resume or set the speed from the browser."""
        if action == "pause":
            self.runner.pause()
        elif action == "resume":
            self.runner.start()
        elif action == "speed" and value is not None:
            try:
                self.runner.set_delay(float(value))
            except ValueError:
                pass
        elif action == "step":
            days = 1
            try:
                days = max(1, min(200, int(value or "1")))
            except ValueError:
                days = 1
            self.runner.pause()
            with self.runner.lock:
                for _ in range(days):
                    if not self.engine.agents:
                        break
                    self.engine.step()
        return {"running": self.runner.running, "delay": self.runner.delay, "day": self.engine.clock.day}


def _handler_factory(state: LiveState, page_loader):
    """Build a request handler bound to one simulation."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # noqa: D102
            pass  # Silence per-request logging; it would spam the terminal.

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Dict[str, Any]) -> None:
            self._send(json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self._send(page_loader().encode("utf-8"), "text/html; charset=utf-8")
                elif parsed.path == "/world":
                    self._json(state.world())
                elif parsed.path == "/state":
                    since = int(query.get("since", ["-1"])[0])
                    self._json(state.snapshot(since))
                elif parsed.path == "/inspect":
                    x = int(query.get("x", ["0"])[0])
                    y = int(query.get("y", ["0"])[0])
                    self._json(state.inspect(x, y))
                elif parsed.path == "/explain":
                    self._json(state.explain())
                elif parsed.path == "/ask":
                    self._json(state.ask(query.get("q", [""])[0]))
                elif parsed.path == "/act":
                    self._json(state.act(
                        query.get("tool", [""])[0],
                        int(query.get("x", ["0"])[0]),
                        int(query.get("y", ["0"])[0]),
                        int(query.get("radius", ["3"])[0]),
                    ))
                elif parsed.path == "/found":
                    self._json(state.found_civ(
                        int(query.get("x", ["0"])[0]),
                        int(query.get("y", ["0"])[0]),
                        query.get("name", [""])[0],
                        query.get("house", [""])[0],
                        int(query.get("people", ["10"])[0]),
                    ))
                elif parsed.path == "/overlay":
                    self._json(state.overlay(query.get("kind", ["fertility"])[0]))
                elif parsed.path == "/control":
                    action = query.get("action", [""])[0]
                    value = query.get("value", [None])[0]
                    self._json(state.control(action, value))
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Browser navigated away mid-response.
            except Exception as error:  # Never let one bad request kill the server.
                try:
                    self.send_error(500, str(error))
                except OSError:
                    pass

    return Handler


class LiveServer:
    """Serves the live map on localhost while the simulation runs."""

    def __init__(
        self, engine: SimulationEngine, runner: SimulationRunner, port: int = 8765
    ) -> None:
        self.state = LiveState(engine, runner)
        self.port = port
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        """Where the page is served."""
        return f"http://127.0.0.1:{self.port}/"

    def start(self) -> str:
        """Start serving in a background thread; returns the URL.

        If the port is busy, tries the next few before giving up.
        """
        handler = _handler_factory(self.state, load_page)
        last_error: Optional[OSError] = None
        for offset in range(8):
            try:
                self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port + offset), handler)
                self.port += offset
                break
            except OSError as error:
                last_error = error
        if self.httpd is None:
            raise OSError(f"Could not bind a port near {self.port}: {last_error}")

        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self.url

    def stop(self) -> None:
        """Shut the server down."""
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parent / "static"
PAGE_FILE = STATIC_DIR / "live.html"


def load_page() -> str:
    """Read the UI from disk.

    Kept as a real HTML file rather than a Python string so the interface can be
    edited (and reloaded with a browser refresh) without touching Python. Falls
    back to a plain message if the file is missing, rather than failing to boot.
    """
    try:
        return PAGE_FILE.read_text(encoding="utf-8")
    except OSError:
        return (
            "<!doctype html><meta charset='utf-8'><title>Worldbox</title>"
            f"<p style='font-family:system-ui;padding:2rem'>Missing UI file: {PAGE_FILE}</p>"
        )
