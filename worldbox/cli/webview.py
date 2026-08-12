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
from ..world.terrain import TerrainType

TERRAIN_CODES: Dict[str, int] = {
    TerrainType.WATER.value: 0,
    TerrainType.GRASS.value: 1,
    TerrainType.FOREST.value: 2,
    TerrainType.MOUNTAIN.value: 3,
}

# Event kinds worth showing in the live feed.
LOGGED_KINDS = ("birth", "death", "war", "plague", "invention", "society", "milestone")


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
            return {
                "width": self.engine.world.width,
                "height": self.engine.world.height,
                "seed": self.engine.seed,
                "terrain": [
                    "".join(str(TERRAIN_CODES[tile.value]) for tile in row) for row in grid
                ],
            }

    def snapshot(self, since_day: int = -1) -> Dict[str, Any]:
        """Everything the page needs to draw one frame."""
        with self.runner.lock:
            engine = self.engine
            stats = engine.stats()

            agents: List[int] = []
            for agent in engine.agents:
                agents.extend((agent.x, agent.y, self._index_for(agent.group_id)))

            settlements: List[int] = []
            for settlement in engine.settlements.active():
                settlements.extend(
                    (settlement.x, settlement.y, int(settlement.level),
                     self._index_for(settlement.group_id))
                )

            fresh = [entry for entry in self.log if entry["d"] > since_day][-60:]
            capacity = stats.food_capacity or 1.0
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
                "log": fresh,
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
                group = engine.groups.get(best.group_id)
                chief = engine.get_agent(group.chieftain_id) if group and group.chieftain_id else None
                result["settlement"] = {
                    "name": best.name,
                    "level": best.level_name,
                    "founded": best.founded_day,
                    "store": round(best.food_store),
                    "storePct": round(best.store_fraction() * 100),
                    "tribe": group.name if group else "abandoned",
                    "population": group.size if group else 0,
                    "era": group.era if group else "",
                    "techs": sorted(group.knowledge.known) if group else [],
                    "chieftain": chief.full_name if chief else None,
                    "chieftainAge": round(chief.age_years(dpy)) if chief else None,
                    "battlesWon": group.battles_won if group else 0,
                    "battlesLost": group.battles_lost if group else 0,
                    "warDead": group.war_dead if group else 0,
                    "plagueDead": group.plague_dead if group else 0,
                    "roles": {k: v for k, v in (group.role_counts or {}).items() if v} if group else {},
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
                        "tribe": tribe.name if tribe else None,
                        "ill": agent.infection.disease.name if agent.infection else None,
                    })
                    if len(people) >= 6:
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
                    f"No API key. Set {narrator.ENV_KEY} before starting Worldbox."}
        with self.runner.lock:
            entries = list(self.engine.chronicle.entries)
            year = self.engine.clock.year
        if not entries:
            return {"ok": False, "error": "Nothing has happened yet."}
        result = narrator.narrate(entries, max_entries=80)
        if not result.ok:
            return {"ok": False, "error": result.error}
        return {"ok": True, "year": year, "text": result.text}

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
        return {"running": self.runner.running, "delay": self.runner.delay}


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
                elif parsed.path == "/act":
                    self._json(state.act(
                        query.get("tool", [""])[0],
                        int(query.get("x", ["0"])[0]),
                        int(query.get("y", ["0"])[0]),
                        int(query.get("radius", ["3"])[0]),
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
