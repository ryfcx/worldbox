"""The terminal interface for Worldbox.

This is the only module that knows about printing, prompts and commands. It
talks to the simulation exclusively through :class:`SimulationEngine`'s public
read-only API (``stats()``, ``get_agent()``, ``events.recent()``) and through
:class:`SimulationRunner` for start/pause -- so replacing it with a GUI later
requires no change to the simulation.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..agents.agent import Agent
from ..agents.behavior import Goal
from ..simulation.chronicle import ChronicleEntry
from ..simulation.engine import SimulationEngine, SimulationRunner, WorldStats
from ..simulation.events import Event
from ..ai import narrator as ai_narrator
from .live import LiveView

RULE_CHAR = "─"
MAX_RULE_WIDTH = 60


def _rule() -> str:
    """A horizontal rule sized to the terminal, within sane limits."""
    width = min(shutil.get_terminal_size((80, 24)).columns, MAX_RULE_WIDTH)
    return RULE_CHAR * max(20, width)


def render_dashboard(stats: WorldStats, events: List[Event]) -> str:
    """Render the main status screen from a stats snapshot and recent events.

    Pure function: give it data, get back text. Handy for tests and for any
    future frontend that wants the same summary.
    """
    rule = _rule()
    lines = [
        "WORLDBOX",
        rule,
        "",
        f"Day: {stats.day}",
        f"Population: {stats.population}",
        f"Births: {stats.births_today}",
        f"Deaths: {stats.deaths_today}",
        "",
        f"Average Age: {stats.average_age_years:.1f}",
        f"Average Health: {stats.average_health:.1f}",
        f"Hungry: {stats.hungry}",
        f"Resting: {stats.resting}",
        f"Searching for Food: {stats.seeking_food}",
        "",
        "Recent Events:",
    ]
    if events:
        lines.extend(f"  • {event.message}" for event in events)
    else:
        lines.append("  (nothing yet)")
    lines.extend(["", rule])
    return "\n".join(lines)


def render_stats(stats: WorldStats) -> str:
    """Render the detailed world statistics screen."""
    rule = _rule()
    terrain = stats.terrain_counts
    total_tiles = sum(terrain.values()) or 1
    food_pct = (stats.total_food / stats.food_capacity * 100.0) if stats.food_capacity else 0.0

    lines = [
        "WORLD STATISTICS",
        rule,
        f"Seed: {stats.seed}",
        f"Day: {stats.day}   Year: {stats.year}",
        "",
        "Population",
        f"  Living agents:      {stats.population}",
        f"  Births (total):     {stats.total_births}",
        f"  Deaths (total):     {stats.total_deaths}",
        f"  Average age:        {stats.average_age_years:.1f} years",
        f"  Oldest agent:       {stats.oldest_age_years:.1f} years",
        "",
        "Condition",
        f"  Average health:     {stats.average_health:.1f}",
        f"  Average hunger:     {stats.average_hunger:.1f}",
        f"  Average energy:     {stats.average_energy:.1f}",
        f"  Hungry agents:      {stats.hungry}",
        "",
        "Activity",
        f"  Resting:            {stats.resting}",
        f"  Searching for food: {stats.seeking_food}",
        f"  Eating:             {stats.eating}",
        f"  Seeking a mate:     {stats.seeking_mate}",
        f"  Wandering:          {stats.wandering}",
        "",
        "Civilisation",
        f"  Tribes:             {stats.tribes} ({stats.unaffiliated} unaffiliated)",
        f"  Largest tribe:      {stats.largest_tribe or 'none'} ({stats.largest_tribe_size})",
        f"  Settlements:        {stats.settlements}"
        + (
            f", largest {stats.largest_settlement} ({stats.largest_settlement_level})"
            if stats.largest_settlement
            else ""
        ),
        f"  Era:                {stats.most_advanced_era}",
        f"  Technologies:       {stats.technologies_known} / 16 discovered",
        f"  Food stored:        {stats.total_food_stored:,.0f}",
        f"  Chronicle entries:  {stats.chronicle_entries}",
        "",
        "Conflict & disease",
        f"  Active wars:        {stats.active_wars}",
        f"  Battles (total):    {stats.total_battles}",
        f"  War dead:           {stats.war_deaths}",
        f"  Average aggression: {stats.average_aggression:.2f}",
        f"  Currently ill:      {stats.ill}"
        + (f" ({stats.current_plague})" if stats.current_plague else ""),
        f"  Plague dead:        {stats.plague_deaths}",
        "",
        "Environment",
        f"  Food available:     {stats.total_food:,.0f} / {stats.food_capacity:,.0f} ({food_pct:.1f}%)",
    ]
    for name in sorted(terrain):
        count = terrain[name]
        lines.append(f"  {name.capitalize():<18} {count:>6} tiles ({count / total_tiles * 100:.1f}%)")
    lines.append(rule)
    return "\n".join(lines)


def render_agent(
    agent: Agent,
    days_per_year: int,
    tribe_name: Optional[str] = None,
    is_chieftain: bool = False,
) -> str:
    """Render the inspection screen for a single agent."""
    rule = _rule()
    needs = agent.needs
    goal = Goal(agent.goal).label if agent.goal in {g.value for g in Goal} else agent.goal
    parents = ", ".join(str(p) for p in agent.parents if p is not None) or "none (founder)"
    known_food = ", ".join(f"({x},{y})" for x, y in list(agent.memory.known_food)[:6]) or "none"
    known_terrain = ", ".join(sorted(agent.memory.known_terrain)) or "none"

    allegiance = tribe_name or "unaffiliated"
    if is_chieftain:
        allegiance += "  [CHIEFTAIN]"
    illness = (
        f"{agent.infection.disease.name} ({agent.infection.days_left}d left)"
        if agent.infection
        else "healthy"
    )

    lines = [
        f"AGENT #{agent.id} - {agent.full_name}",
        rule,
        f"Age:        {agent.age_years(days_per_year):.1f} years "
        f"(lifespan ~{agent.lifespan_days / days_per_year:.1f})",
        f"Position:   ({agent.x}, {agent.y})",
        f"Goal:       {goal}",
        f"Tribe:      {allegiance}",
        f"Role:       {agent.role}",
        f"Temperament:{agent.aggression:>6.2f} aggression, {agent.kills} kills",
        f"Condition:  {illness}",
        "",
        f"Health:     {needs.health:.1f}",
        f"Hunger:     {needs.hunger:.1f}   (0 = full, 100 = starving)",
        f"Energy:     {needs.energy:.1f}   (100 = rested)",
        "",
        f"Born:       day {agent.birth_day} at {agent.memory.birthplace}",
        f"Parents:    {parents}",
        f"Children:   {len(agent.children)}"
        + (f" ({', '.join(str(c) for c in agent.children)})" if agent.children else ""),
        "",
        "Memory",
        f"  Known food:    {known_food}",
        f"  Known terrain: {known_terrain}",
        "  Personal log:",
    ]
    if agent.memory.log:
        lines.extend(f"    • {entry}" for entry in agent.memory.log)
    else:
        lines.append("    (nothing notable yet)")
    lines.append(rule)
    return "\n".join(lines)


def render_events(events: List[Event]) -> str:
    """Render a list of historical events."""
    rule = _rule()
    lines = ["RECENT EVENTS", rule]
    if events:
        lines.extend(f"  • {event}" for event in events)
    else:
        lines.append("  (no events recorded)")
    lines.append(rule)
    return "\n".join(lines)


def render_tribes(engine: SimulationEngine) -> str:
    """Render the roster of tribes: size, era, settlement, leader and wars."""
    rule = _rule()
    lines = ["TRIBES", rule]
    tribes = sorted(engine.groups.active(), key=lambda g: -g.size)
    if not tribes:
        lines += ["  (no tribes have formed yet)", rule]
        return "\n".join(lines)

    for group in tribes:
        settlement = engine.settlements.for_group(group.id)
        chief = engine.get_agent(group.chieftain_id) if group.chieftain_id else None
        wars = [w for key, w in engine.diplomacy.wars.items() if group.id in key and w.active]
        home = f"{settlement.name} ({settlement.level_name})" if settlement else "nomadic"
        lines.append(f"  #{group.id} {group.name}")
        lines.append(
            f"      {group.size} people | {group.era} | {len(group.knowledge.known)} techs | {home}"
        )
        if chief is not None:
            lines.append(
                f"      Chieftain: {chief.full_name} "
                f"(age {chief.age_years(engine.config.simulation.days_per_year):.0f})"
            )
        if settlement is not None:
            lines.append(
                f"      Granary: {settlement.food_store:.0f} "
                f"({settlement.store_fraction() * 100:.0f}% full)"
            )
        roles = ", ".join(
            f"{name} {count}" for name, count in sorted(group.role_counts.items()) if count
        )
        if roles:
            lines.append(f"      {roles}")
        if wars:
            lines.append(f"      AT WAR ({len(wars)})")
    lines.append(rule)
    return "\n".join(lines)


def render_tribe(engine: SimulationEngine, group_id: int) -> str:
    """Render one tribe in detail, including its relations with the others."""
    group = engine.groups.get(group_id)
    if group is None:
        return f"No tribe with id {group_id}."
    rule = _rule()
    settlement = engine.settlements.for_group(group.id)
    chief = engine.get_agent(group.chieftain_id) if group.chieftain_id else None
    known = sorted(group.knowledge.known)

    lines = [
        f"TRIBE #{group.id} - {group.name}",
        rule,
        f"Founded:    day {group.founded_day}",
        f"Members:    {group.size}",
        f"Era:        {group.era}",
        f"Chieftain:  {chief.full_name if chief else 'none'}",
        f"Aggression: {group.average_aggression:.2f}   Hunger: {group.average_hunger:.1f}",
        f"Record:     {group.battles_won} battles won, {group.battles_lost} lost, "
        f"{group.war_dead} war dead, {group.plague_dead} lost to plague",
        "",
    ]
    if settlement is not None:
        lines += [
            f"Settlement: {settlement.name} ({settlement.level_name}) at "
            f"({settlement.x}, {settlement.y})",
            f"Granary:    {settlement.food_store:.0f} / "
            f"{settlement.spec.store_capacity:.0f}",
            "",
        ]
    lines.append(f"Technology ({len(known)}):")
    lines.append("  " + (", ".join(known) if known else "none"))
    lines.append("")
    lines.append("Relations:")
    for other, score in engine.diplomacy.relation_summary(group.id, engine.groups)[:8]:
        state = "WAR" if engine.diplomacy.at_war(group.id, other.id) else ""
        lines.append(f"  {other.name:<28} {score:>6.1f}  {state}")
    lines.append(rule)
    return "\n".join(lines)


def render_technology(engine: SimulationEngine) -> str:
    """Render the world's technological progress, era by era."""
    from ..society.technology import ERAS, TECH_TREE

    rule = _rule()
    counts = {}
    for group in engine.groups.active():
        for tech_id in group.knowledge.known:
            counts[tech_id] = counts.get(tech_id, 0) + 1
    tribes = max(1, len(engine.groups.active()))

    lines = ["TECHNOLOGY", rule]
    for era in ERAS:
        lines.append(f"  {era}")
        for tech in TECH_TREE:
            if tech.era != era:
                continue
            holders = counts.get(tech.id, 0)
            mark = "#" if holders else "."
            share = f"{holders}/{tribes} tribes" if holders else "undiscovered"
            lines.append(f"    [{mark}] {tech.name:<20} {share}")
    lines.append(rule)
    return "\n".join(lines)


def render_wars(engine: SimulationEngine) -> str:
    """Render active wars and the most recent concluded ones."""
    rule = _rule()
    lines = ["WARS", rule]
    active = engine.diplomacy.active_wars()
    if active:
        lines.append("  Active:")
        for war in active:
            first = engine.groups.get(war.tribes[0])
            second = engine.groups.get(war.tribes[1])
            names = f"{first.name if first else '?'} vs {second.name if second else '?'}"
            lines.append(
                f"    {names} | started day {war.started_day} | "
                f"{war.battles} battles | {war.total_casualties()} dead"
            )
    else:
        lines.append("  No wars are being fought.")

    if engine.diplomacy.history:
        lines += ["", "  Recent history:"]
        for war in engine.diplomacy.history[-8:]:
            ended = "extinction" if war.ended_day == -1 else f"day {war.ended_day}"
            lines.append(
                f"    days {war.started_day}-{ended} | {war.battles} battles | "
                f"{war.total_casualties()} dead"
            )
    lines.append(rule)
    return "\n".join(lines)


def render_settlements(engine: SimulationEngine) -> str:
    """Render every standing settlement."""
    rule = _rule()
    lines = ["SETTLEMENTS", rule]
    settlements = sorted(engine.settlements.active(), key=lambda s: (-int(s.level), s.name))
    if not settlements:
        lines += ["  (none founded yet)", rule]
        return "\n".join(lines)
    for settlement in settlements:
        group = engine.groups.get(settlement.group_id)
        lines.append(
            f"  {settlement.name:<22} {settlement.level_name:<9} "
            f"({settlement.x:>3},{settlement.y:>3}) | "
            f"{group.size if group else 0:>3} people | "
            f"granary {settlement.store_fraction() * 100:>3.0f}% | "
            f"{group.name if group else 'abandoned'}"
        )
    lines.append(rule)
    return "\n".join(lines)


def render_chronicle(entries: List[ChronicleEntry], total: int) -> str:
    """Render the permanent history of the world."""
    rule = _rule()
    lines = ["CHRONICLE", rule]
    if not entries:
        lines += ["  (nothing of note has happened yet)", rule]
        return "\n".join(lines)
    for entry in entries:
        lines.append(f"  {entry}")
    lines += [f"  ({len(entries)} of {total} recorded entries)", rule]
    return "\n".join(lines)


HELP_TEXT = """\
Commands
  start | live       Open the live dashboard (everything updates in place)
  pause              Pause a running simulation
  step               Advance one day
  advance <n>        Advance n days (e.g. advance 10 / 100 / 1000)
  10 | 100 | 1000    Shortcut for 'advance n'
  reset              Restart the world with the current seed
  seed <n>           Change the random seed and restart the world
  agent <id>         Inspect an individual agent
  stats              Display world statistics
  tribes             List every tribe: size, era, leader, settlement
  tribe <id>         Inspect one tribe in detail, including its relations
  tech               World technology tree and who knows what
  wars               Active wars and recent war history
  towns              Every settlement and its granary
  chronicle [n]      The permanent history of the world (default 30)
  narrate            Have an AI write up the chronicle as a history
  livemap            Open a browser map that updates as the simulation runs
  view [days]        Build a recorded map of the run so far and open it
  export [days]      Record a run to JSON for the visual map viewer
  events [n]         Display the n most recent events (default 15)
  help               Show this help
  exit               Quit Worldbox

In the live dashboard
  space              Run / pause
  s                  Advance one day while paused
  + / -              Faster / slower
  q                  Pause and return to this prompt\
"""


class Terminal:
    """The interactive Worldbox command loop."""

    def __init__(self, engine: SimulationEngine) -> None:
        self.engine = engine
        self.runner = SimulationRunner(engine)
        self.should_exit = False
        self._live_server = None  # Created lazily by the livemap command.
        self.commands: Dict[str, Callable[[List[str]], None]] = {
            "start": self.cmd_start,
            "live": self.cmd_start,
            "pause": self.cmd_pause,
            "step": self.cmd_step,
            "advance": self.cmd_advance,
            "reset": self.cmd_reset,
            "seed": self.cmd_seed,
            "agent": self.cmd_agent,
            "inspect": self.cmd_agent,
            "stats": self.cmd_stats,
            "tribes": self.cmd_tribes,
            "groups": self.cmd_tribes,
            "tribe": self.cmd_tribe,
            "tech": self.cmd_tech,
            "wars": self.cmd_wars,
            "towns": self.cmd_towns,
            "settlements": self.cmd_towns,
            "chronicle": self.cmd_chronicle,
            "narrate": self.cmd_narrate,
            "export": self.cmd_export,
            "view": self.cmd_view,
            "livemap": self.cmd_livemap,
            "live": self.cmd_livemap,
            "gui": self.cmd_view,
            "map": self.cmd_view,
            "history": self.cmd_chronicle,
            "events": self.cmd_events,
            "help": self.cmd_help,
            "exit": self.cmd_exit,
            "quit": self.cmd_exit,
        }

    # -- main loop ----------------------------------------------------------

    def run(self) -> None:
        """Show the dashboard and process commands until the user exits."""
        self.show_dashboard()
        print("\nType 'help' for commands.\n")
        while not self.should_exit:
            try:
                raw = input("worldbox> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not raw:
                continue
            self.dispatch(raw)
        self.runner.pause()
        if self._live_server is not None:
            self._live_server.stop()
        print("Goodbye.")

    def dispatch(self, raw: str) -> None:
        """Parse and run a single command line."""
        parts = raw.split()
        name, args = parts[0].lower(), parts[1:]

        # Bare numbers are a shortcut for 'advance n'.
        if name.isdigit():
            name, args = "advance", [name]

        handler = self.commands.get(name)
        if handler is None:
            print(f"Unknown command: {name!r}. Type 'help' for the command list.")
            return
        try:
            handler(args)
        except Exception as error:  # Never let a bad command kill the session.
            print(f"Command failed: {error}")

    # -- output helpers -----------------------------------------------------

    def show_dashboard(self) -> None:
        """Print the main status screen."""
        with self.runner.lock:
            stats = self.engine.stats()
            events = self.engine.events.recent(6)
        print(render_dashboard(stats, events))

    def _require_paused(self) -> bool:
        """Warn and return False if the simulation is running unattended."""
        if self.runner.running:
            print("The simulation is running. Type 'pause' first.")
            return False
        return True

    @staticmethod
    def _parse_int(args: List[str], name: str, default: Optional[int] = None) -> int:
        """Parse a single integer argument with a friendly error message."""
        if not args:
            if default is None:
                raise ValueError(f"Expected a {name}, e.g. '{name} 10'.")
            return default
        try:
            return int(args[0])
        except ValueError:
            raise ValueError(f"{args[0]!r} is not a valid {name}.") from None

    # -- commands -----------------------------------------------------------

    def cmd_start(self, args: List[str]) -> None:
        """Open the live dashboard and start running.

        Control returns here when the user presses 'q', with the simulation
        paused so the static dashboard below is never stale.
        """
        LiveView(self.engine, self.runner).run(start_running=True)
        self.show_dashboard()

    def cmd_pause(self, args: List[str]) -> None:
        """Pause a continuously running simulation."""
        if self.runner.pause():
            print("Paused.")
            self.show_dashboard()
        else:
            print("Simulation is not running.")

    def cmd_step(self, args: List[str]) -> None:
        """Advance exactly one day."""
        self.cmd_advance(["1"])

    def cmd_advance(self, args: List[str]) -> None:
        """Advance the simulation by n days."""
        if not self._require_paused():
            return
        days = self._parse_int(args, "number of days", default=1)
        if days <= 0:
            print("Number of days must be positive.")
            return
        with self.runner.lock:
            simulated = self.engine.run(days)
        if simulated < days:
            print(f"Stopped early after {simulated} days: the population died out.")
        self.show_dashboard()

    def cmd_reset(self, args: List[str]) -> None:
        """Rebuild the world from the current seed."""
        self.runner.pause()
        with self.runner.lock:
            self.engine.reset()
        print(f"World reset with seed {self.engine.seed}.")
        self.show_dashboard()

    def cmd_seed(self, args: List[str]) -> None:
        """Change the random seed and rebuild the world."""
        seed = self._parse_int(args, "seed")
        self.runner.pause()
        with self.runner.lock:
            self.engine.reset(seed)
        print(f"Seed changed to {seed}; world regenerated.")
        self.show_dashboard()

    def cmd_agent(self, args: List[str]) -> None:
        """Inspect a single agent by id."""
        agent_id = self._parse_int(args, "agent id")
        with self.runner.lock:
            agent = self.engine.get_agent(agent_id)
            days_per_year = self.engine.config.simulation.days_per_year
            tribe = self.engine.groups.get(agent.group_id) if agent else None
            tribe_name = tribe.name if tribe else None
            is_chief = bool(tribe and tribe.chieftain_id == agent_id)
        if agent is None:
            print(f"No living agent with id {agent_id}.")
            return
        print(render_agent(agent, days_per_year, tribe_name, is_chief))

    def cmd_stats(self, args: List[str]) -> None:
        """Show detailed world statistics."""
        with self.runner.lock:
            stats = self.engine.stats()
        print(render_stats(stats))

    def cmd_tribes(self, args: List[str]) -> None:
        """List every tribe."""
        with self.runner.lock:
            print(render_tribes(self.engine))

    def cmd_tribe(self, args: List[str]) -> None:
        """Inspect a single tribe."""
        tribe_id = self._parse_int(args, "tribe id")
        with self.runner.lock:
            print(render_tribe(self.engine, tribe_id))

    def cmd_tech(self, args: List[str]) -> None:
        """Show the world's technological progress."""
        with self.runner.lock:
            print(render_technology(self.engine))

    def cmd_wars(self, args: List[str]) -> None:
        """Show active and recent wars."""
        with self.runner.lock:
            print(render_wars(self.engine))

    def cmd_towns(self, args: List[str]) -> None:
        """Show every settlement."""
        with self.runner.lock:
            print(render_settlements(self.engine))

    def cmd_chronicle(self, args: List[str]) -> None:
        """Show the permanent history of the world."""
        count = self._parse_int(args, "entry count", default=30)
        with self.runner.lock:
            entries = self.engine.chronicle.recent(max(1, count))
            total = len(self.engine.chronicle)
        print(render_chronicle(entries, total))

    def cmd_narrate(self, args: List[str]) -> None:
        """Ask an AI to write the world's chronicle up as a history."""
        if not ai_narrator.is_available():
            print(
                f"Narration needs an API key. Set {ai_narrator.ENV_KEY} in your\n"
                "environment, then run 'narrate' again."
            )
            return
        with self.runner.lock:
            entries = list(self.engine.chronicle.entries)
            year = self.engine.clock.year
        print("Writing the history... (this takes a few seconds)")
        result = ai_narrator.narrate(entries)
        if not result.ok:
            print(f"Narration failed: {result.error}")
            return
        rule = _rule()
        print(f"\nA HISTORY OF THE WORLD, TO YEAR {year}")
        print(rule)
        for line in ai_narrator.wrap(result.text):
            print(f"  {line}")
        print(rule)

    def cmd_livemap(self, args: List[str]) -> None:
        """Serve a browser map that updates while the simulation runs."""
        import webbrowser

        from .webview import LiveServer

        if self._live_server is None:
            self._live_server = LiveServer(self.engine, self.runner)
            try:
                url = self._live_server.start()
            except OSError as error:
                self._live_server = None
                print(f"Could not start the live map: {error}")
                return
            print(f"Live map serving at {url}")
            webbrowser.open(url)
        else:
            print(f"Live map already serving at {self._live_server.url}")

        # A live map with a paused world shows nothing moving, so start ticking.
        if not self.runner.running:
            self.runner.start()
            print("Simulation running. Use the browser controls, or 'pause' here.")

    def cmd_view(self, args: List[str]) -> None:
        """Build the graphical map from the current world and open it."""
        from ..main import run_viewer

        days = self._parse_int(args, "number of days", default=0)
        self.runner.pause()
        with self.runner.lock:
            run_viewer(self.engine, days, open_browser=True)

    def cmd_export(self, args: List[str]) -> None:
        """Record a run to JSON for the visual viewer."""
        from ..export import RunRecorder

        days = self._parse_int(args, "number of days", default=0)
        with self.runner.lock:
            recorder = RunRecorder(self.engine, every=max(10, max(days, 1) // 200))
            if days > 0:
                recorder.run(days)
            else:
                recorder.capture()
            path = recorder.write_json(Path("worldbox_run.json"))
        print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KB, "
              f"{len(recorder.frames)} frames).")

    def cmd_events(self, args: List[str]) -> None:
        """Show the most recent events."""
        count = self._parse_int(args, "event count", default=15)
        with self.runner.lock:
            events = self.engine.events.recent(max(1, count))
        print(render_events(events))

    def cmd_help(self, args: List[str]) -> None:
        """List the available commands."""
        print(HELP_TEXT)

    def cmd_exit(self, args: List[str]) -> None:
        """Leave Worldbox."""
        self.should_exit = True
