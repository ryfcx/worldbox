"""Tribe minds: DeepSeek Flash playing the civilisation, not just narrating it.

Each tribe that forms is handed to a language model. Periodically the model
looks at the tribe's situation -- food, neighbours, wars, granary, tech -- and
returns a policy. That policy is applied to real simulation systems:

* profession shares (warriors, farmers, scholars, healers)
* how aggressively members fight, flee, and forage
* how tightly the tribe holds territory
* daily diplomacy toward named neighbours (war, peace, trade)

The bodies still score actions every day. The mind sets the weights and the
politics those scores live inside, which is what "trapping AIs in a simulator"
actually means here: they do not get a chat log, they get a people.

Network calls run on a background thread pool. The day loop never waits, so a
slow reply costs sim-days rather than freezing the live map. No key, or a
failed call, leaves the tribe on the ordinary utility rules.
"""

from __future__ import annotations

import json
import queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..config import AIConfig, Config
from ..simulation.events import EventKind
from .client import DEFAULT_MODEL, ChatResult, chat, house_spec, is_available
from .inventions import (
    InventionProposal,
    briefing_lines,
    parse_invention,
    scan_materials,
    validate_proposal,
)

STANCES = ("survive", "farm", "expand", "defend", "research", "raid", "trade")
TOWARD_ACTIONS = ("war", "peace", "trade", "ignore")

STANCE_ALIASES = {
    "war": "raid",
    "attack": "raid",
    "conquer": "raid",
    "peace": "trade",
    "ally": "trade",
    "food": "farm",
    "grow": "expand",
    "settle": "expand",
    "study": "research",
    "hide": "survive",
    "flee": "survive",
}
TOWARD_ALIASES = {
    "attack": "war",
    "raid": "war",
    "fight": "war",
    "ally": "peace",
    "friend": "peace",
    "gift": "trade",
}

SYSTEM_PROMPT = """\
You are the mind of a tribe trapped in a civilisation simulator. You do not \
describe history; you decide what this people will do next. Your choices \
change the world's logic: who farms, who fights, who they attack, how far \
they roam, and what they invent from the materials actually around them.

Reply with one JSON object and nothing else. Example:
{"stance":"farm","warrior_share":0.12,"farmer_share":0.40,"scholar_share":0.10,\
"healer_share":0.08,"cohesion":0.7,"fight":0.8,"flee":1.1,"seek_food":1.2,\
"seek_mate":1.0,"toward":[{"tribe_id":2,"action":"trade"}],\
"invent":{"name":"Fish Traps","from":["water","forest"],"effect":"food",\
"why":"The river can feed us if we weave traps."},\
"intent":"Store grain and build traps on the river."}

Rules:
- stance is one of: survive, farm, expand, defend, research, raid, trade
- shares are 0..1 fractions of working adults; leftover people forage
- fight/flee/seek_food/seek_mate multiply how strongly people pick that action
- cohesion 0 = scatter and claim ground, 1 = huddle at home
- toward.action is war, peace, trade, or ignore; use real tribe_id values
- invent is one new craft or null. from/effect must use only materials listed \
in the briefing. effect is food, combat, research, health, fertility, or medicine.
- Do not invent tribes, places, or materials you were not given.
- intent is one short sentence a chieftain would actually say
"""

HOUSE_FLAVOR = {
    "deepseek": "You are DeepSeek Flash: terse, practical, invent from what is underfoot.",
    "openai": "You are GPT-5 Mini: ambitious, social, name crafts and push the tribe to grow.",
}


@dataclass
class TribePolicy:
    """What a mind has decided its tribe should do, until it thinks again."""

    stance: str = "survive"
    warrior_share: float = 0.10
    farmer_share: float = 0.32
    scholar_share: float = 0.14
    healer_share: float = 0.08
    cohesion: float = 0.55
    fight: float = 1.0
    flee: float = 1.0
    seek_food: float = 1.0
    seek_mate: float = 1.0
    toward: List[Tuple[int, str]] = field(default_factory=list)
    invent: Optional[InventionProposal] = None
    intent: str = ""
    thought_day: int = 0
    model: str = DEFAULT_MODEL

    @property
    def territory_radius(self) -> int:
        """How far members may stray before they are always pulled home."""
        return int(round(8 + (1.0 - self.cohesion) * 16))


@dataclass
class MindState:
    """Public snapshot of one tribe mind, for the terminal and live map."""

    group_id: int
    name: str
    stance: str = ""
    intent: str = ""
    inventing: str = ""
    house: str = ""
    model: str = ""
    thinking: bool = False
    thought_day: int = 0
    error: str = ""


def _clamp(value: float, low: float, high: float) -> float:
    """Pin a number to an inclusive range."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    return low if number < low else (high if number > high else number)


def _share(value: Any, default: float) -> float:
    """Read a 0..1 labour share; values above 1 are treated as percents."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number > 1.0:
        number /= 100.0
    return _clamp(number, 0.0, 0.8)


def _bias(value: Any, default: float = 1.0) -> float:
    """Read an action multiplier, kept in a range that cannot freeze a tribe."""
    return _clamp(value if value is not None else default, 0.35, 2.2)


def extract_json(text: str) -> Dict[str, Any]:
    """Pull a JSON object out of a model reply, even if it used fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("JSON was not an object")
    return parsed


def parse_policy(
    raw: Dict[str, Any],
    day: int,
    model: str,
    era: str = "Stone Age",
) -> TribePolicy:
    """Turn the model's JSON into a policy the engine can apply."""
    stance = str(raw.get("stance", "survive")).strip().lower()
    stance = STANCE_ALIASES.get(stance, stance)
    if stance not in STANCES:
        stance = "survive"

    toward: List[Tuple[int, str]] = []
    listed = raw.get("toward") or []
    if isinstance(listed, dict):
        listed = [listed]
    if isinstance(listed, list):
        for item in listed:
            if not isinstance(item, dict):
                continue
            try:
                other_id = int(item.get("tribe_id", item.get("id")))
            except (TypeError, ValueError):
                continue
            action = str(item.get("action", "ignore")).strip().lower()
            action = TOWARD_ALIASES.get(action, action)
            if action not in TOWARD_ACTIONS:
                action = "ignore"
            toward.append((other_id, action))

    intent = str(raw.get("intent", "")).strip().replace("\n", " ")
    if len(intent) > 180:
        intent = intent[:177] + "..."

    invent = parse_invention(
        raw.get("invent"), era=era, research_stance=(stance == "research")
    )

    return TribePolicy(
        stance=stance,
        warrior_share=_share(raw.get("warrior_share"), 0.10),
        farmer_share=_share(raw.get("farmer_share"), 0.32),
        scholar_share=_share(raw.get("scholar_share"), 0.14),
        healer_share=_share(raw.get("healer_share"), 0.08),
        cohesion=_clamp(raw.get("cohesion", 0.55), 0.05, 0.95),
        fight=_bias(raw.get("fight")),
        flee=_bias(raw.get("flee")),
        seek_food=_bias(raw.get("seek_food")),
        seek_mate=_bias(raw.get("seek_mate")),
        toward=toward[:6],
        invent=invent,
        intent=intent,
        thought_day=day,
        model=model,
    )


def _brief_tribe(engine: Any, group: Any) -> str:
    """A compact situation report one mind can decide from."""
    config: Config = engine.config
    dpy = config.simulation.days_per_year
    members = engine._members_of(group)
    chief = engine.get_agent(group.chieftain_id) if group.chieftain_id else None
    settlement = engine.settlements.for_group(group.id)
    known = sorted(group.knowledge.known)

    spec = house_spec(getattr(group, "ai_house", "") or DEFAULT_MODEL)
    lines = [
        f"Year {engine.clock.year}, day {engine.clock.day}.",
        f"You are the mind of the {group.name} (tribe_id {group.id}).",
        HOUSE_FLAVOR.get(spec["house"], ""),
        f"Your model is {spec['label']}.",
        f"People: {group.size}. Hunger: {group.average_hunger:.0f}/100. "
        f"Aggression: {group.average_aggression:.2f}. Era: {group.era}.",
    ]
    if chief is not None:
        lines.append(
            f"Chieftain: {chief.full_name}, "
            f"age {chief.age_years(dpy):.0f}, role {chief.role}."
        )
    if settlement is not None:
        lines.append(
            f"Home: {settlement.name} ({settlement.level_name}) at "
            f"({settlement.x}, {settlement.y}). Granary "
            f"{settlement.food_store:.0f}/{settlement.spec.store_capacity:.0f} "
            f"({settlement.store_fraction() * 100:.0f}% full)."
        )
    else:
        lines.append(f"Nomadic. Centre tile {group.centre}.")

    roles = ", ".join(
        f"{name} {count}" for name, count in sorted(group.role_counts.items()) if count
    )
    if roles:
        lines.append(f"Jobs: {roles}.")
    lines.append("Technologies: " + (", ".join(known) if known else "none") + ".")
    lines.append(
        f"Record: {group.battles_won} battles won, {group.battles_lost} lost, "
        f"{group.war_dead} war dead, {group.plague_dead} plague dead."
    )

    neighbours = engine.diplomacy.relation_summary(group.id, engine.groups)[:8]
    if neighbours:
        lines.append("Neighbours:")
        for other, score in neighbours:
            distance = max(
                abs(group.centre[0] - other.centre[0]),
                abs(group.centre[1] - other.centre[1]),
            )
            war = " AT WAR" if engine.diplomacy.at_war(group.id, other.id) else ""
            other_town = engine.settlements.for_group(other.id)
            home = other_town.level_name if other_town is not None else "nomadic"
            lines.append(
                f"  tribe_id {other.id} {other.name}: {other.size} people, "
                f"{home}, {other.era}, relation {score:.0f}, {distance} tiles{war}."
            )
    else:
        lines.append("No other tribes known.")

    world_food = engine.world.resources.total_food()
    world_cap = engine.world.resources.total_capacity() or 1.0
    lines.append(
        f"World: {len(engine.agents)} people, {len(engine.groups.active())} tribes, "
        f"{len(engine.diplomacy.active_wars())} wars, "
        f"food {world_food / world_cap * 100:.0f}% of wild capacity."
    )

    recent = [
        entry.message
        for entry in engine.chronicle.entries[-12:]
        if group.name in entry.message
    ]
    if recent:
        lines.append("Recent history: " + " | ".join(recent))

    policy = engine.minds.policies.get(group.id)
    if policy is not None and policy.intent:
        lines.append(f"Your last intent (day {policy.thought_day}): {policy.intent}")
        if policy.invent is not None:
            lines.append(
                f"You are currently working toward {policy.invent.name} "
                f"({policy.invent.cost:.0f} research)."
            )
    lines.extend(briefing_lines(scan_materials(engine, group)))
    return "\n".join(lines)


def _think(briefing: str, timeout: float, model: str = DEFAULT_MODEL) -> ChatResult:
    """One blocking API call. Runs on a worker thread."""
    spec = house_spec(model)
    flavor = HOUSE_FLAVOR.get(spec["house"], "")
    system = SYSTEM_PROMPT if not flavor else SYSTEM_PROMPT + " " + flavor
    return chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": briefing},
        ],
        timeout=timeout,
        retries=1,
        backoff=1.0,
        temperature=0.6,
        max_tokens=700,
        json_mode=True,
        thinking=False,
        model=spec["model"],
    )


class MindDirector:
    """Owns every tribe mind and applies their policies to the world.

    The engine calls :meth:`tick` once per day. HTTP happens off-thread.
    """

    def __init__(self, config: AIConfig) -> None:
        self.config = config
        self.policies: Dict[int, TribePolicy] = {}
        self.last_thought: Dict[int, int] = {}
        self.errors: Dict[int, str] = {}
        self._inbox: "queue.Queue[Tuple[int, int, str, Any]]" = queue.Queue()
        self._pending: Dict[int, bool] = {}
        self._lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None
        self._last_error: str = ""

    def reset(self) -> None:
        """Drop every mind. Called when the world is rebuilt."""
        with self._lock:
            self.policies.clear()
            self.last_thought.clear()
            self.errors.clear()
            self._pending.clear()
        while True:
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                break
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        self._last_error = ""

    def available(self) -> bool:
        """True when minds should actually call the model."""
        return bool(self.config.enabled) and is_available()

    def thinking_count(self) -> int:
        """How many minds currently have an API call in flight."""
        with self._lock:
            return len(self._pending)

    def policy_for(self, group_id: Optional[int]) -> Optional[TribePolicy]:
        """The current policy of a tribe, or None if it has no mind yet."""
        if group_id is None:
            return None
        return self.policies.get(group_id)

    def snapshot(self, engine: Any) -> List[MindState]:
        """One row per active tribe, for the live map and ``minds`` command."""
        with self._lock:
            pending = set(self._pending)
            policies = dict(self.policies)
            errors = dict(self.errors)
        rows: List[MindState] = []
        for group in engine.groups.active():
            policy = policies.get(group.id)
            rows.append(
                MindState(
                    group_id=group.id,
                    name=group.name,
                    stance=policy.stance if policy else "",
                    intent=policy.intent if policy else "",
                    inventing=(
                        policy.invent.name
                        if policy is not None and policy.invent is not None
                        else ""
                    ),
                    house=getattr(group, "ai_house", "") or "",
                    model=house_spec(getattr(group, "ai_house", "") or DEFAULT_MODEL)["label"]
                    if getattr(group, "ai_house", "")
                    else "",
                    thinking=group.id in pending,
                    thought_day=policy.thought_day if policy else 0,
                    error=errors.get(group.id, ""),
                )
            )
        rows.sort(key=lambda row: (-(1 if row.stance else 0), row.name))
        return rows

    def tick(self, engine: Any, day: int) -> None:
        """Apply finished thoughts, push diplomacy, and start new ones."""
        if not self.config.enabled:
            return
        self._forget_dead(engine)
        self._drain(engine, day)
        if not self.available():
            return
        self._nudge_diplomacy(engine)
        self._schedule(engine, day)

    def _pool_or_start(self) -> ThreadPoolExecutor:
        """Lazy thread pool so a world with no key never opens one."""
        if self._pool is None:
            workers = max(1, self.config.max_concurrent)
            self._pool = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="worldbox-mind"
            )
        return self._pool

    def _forget_dead(self, engine: Any) -> None:
        """Drop policies for tribes that no longer exist."""
        living = {group.id for group in engine.groups.active()}
        with self._lock:
            for store in (self.policies, self.last_thought, self.errors, self._pending):
                for group_id in [gid for gid in store if gid not in living]:
                    del store[group_id]

    def _drain(self, engine: Any, day: int) -> None:
        """Fold finished API calls into live policies."""
        while True:
            try:
                group_id, thought_day, briefing, result = self._inbox.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                self._pending.pop(group_id, None)
            group = engine.groups.get(group_id)
            if group is None:
                continue
            if not isinstance(result, ChatResult) or not result.ok:
                error = result.error if isinstance(result, ChatResult) else str(result)
                self.errors[group_id] = error
                self._last_error = error
                continue
            try:
                policy = parse_policy(
                    extract_json(result.text),
                    thought_day,
                    result.model,
                    era=group.era,
                )
            except (ValueError, json.JSONDecodeError) as error:
                self.errors[group_id] = f"Could not parse policy: {error}"
                continue
            if policy.invent is not None:
                survey = scan_materials(engine, group)
                problem = validate_proposal(
                    policy.invent, group, survey, engine.config.ai
                )
                if problem:
                    policy.invent = None
            previous = self.policies.get(group_id)
            self.policies[group_id] = policy
            self.last_thought[group_id] = thought_day
            self.errors.pop(group_id, None)
            if previous is None or previous.stance != policy.stance or previous.intent != policy.intent:
                intent = policy.intent or policy.stance
                engine.events.record(
                    day,
                    EventKind.MIND,
                    f"The {group.name} chose to {policy.stance}: {intent}",
                )
            if policy.invent is not None and (
                previous is None
                or previous.invent is None
                or previous.invent.tech_id != policy.invent.tech_id
            ):
                engine.events.record(
                    day,
                    EventKind.MIND,
                    f"The {group.name} began work on {policy.invent.name}",
                )

    def _nudge_diplomacy(self, engine: Any) -> None:
        """Push relations according to each mind's current toward-list."""
        if not engine.config.diplomacy.enabled:
            return
        deltas = {"war": -3.2, "peace": 2.4, "trade": 1.6, "ignore": 0.0}
        for group in engine.groups.active():
            policy = self.policies.get(group.id)
            if policy is None:
                continue
            # A raid stance with no target still leans on whoever is closest.
            targets = list(policy.toward)
            if policy.stance == "raid" and not targets:
                neighbours = engine.diplomacy.relation_summary(group.id, engine.groups)
                if neighbours:
                    targets = [(neighbours[0][0].id, "war")]
            if policy.stance == "trade" and not targets:
                neighbours = engine.diplomacy.relation_summary(group.id, engine.groups)
                if neighbours:
                    targets = [(neighbours[0][0].id, "trade")]
            for other_id, action in targets:
                if other_id == group.id or engine.groups.get(other_id) is None:
                    continue
                delta = deltas.get(action, 0.0)
                if delta:
                    engine.diplomacy.adjust(group.id, other_id, delta)

    def _schedule(self, engine: Any, day: int) -> None:
        """Ask idle minds that are due to think to look at the world again."""
        tribes = sorted(engine.groups.active(), key=lambda group: -group.size)
        eligible = [
            group
            for group in tribes
            if group.size >= self.config.min_tribe_size
        ][: self.config.max_minds]

        due: List[Any] = []
        with self._lock:
            for group in eligible:
                if group.id in self._pending:
                    continue
                last = self.last_thought.get(group.id)
                if last is None or day - last >= self.config.think_interval_days:
                    due.append(group)
                    self._pending[group.id] = True

        if not due:
            return
        pool = self._pool_or_start()
        timeout = self.config.timeout_seconds
        for group in due:
            briefing = _brief_tribe(engine, group)
            group_id = group.id

            def _done(future, gid=group_id, thought_day=day, text=briefing) -> None:
                try:
                    result = future.result()
                except Exception as error:  # Worker faults must not kill the pool.
                    result = ChatResult(error=str(error), model=DEFAULT_MODEL)
                self._inbox.put((gid, thought_day, text, result))

            if not getattr(group, "ai_house", ""):
                from .client import house_for_index

                group.ai_house = house_for_index(group.id)
            future = pool.submit(
                _think, briefing, timeout, group.ai_house or DEFAULT_MODEL
            )
            future.add_done_callback(_done)
