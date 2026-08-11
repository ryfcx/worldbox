"""The live, continuously repainting dashboard.

The screen is redrawn several times a second while the simulation ticks on its
background thread, so every statistic updates in place. Input is read a single
keypress at a time (no Enter needed), which avoids the usual problem of a typed
line being mangled by a concurrent repaint.

Like the rest of :mod:`worldbox.cli`, this module only reads from the engine --
it never mutates the world directly.

Terminal handling:
    * The alternate screen buffer is used so the normal scrollback is restored
      untouched on exit.
    * Frames are painted by homing the cursor and clearing each line, rather
      than clearing the whole screen, which avoids flicker.
    * Terminal state is always restored, including on exceptions and Ctrl-C.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections import deque
from contextlib import contextmanager
from typing import Deque, Iterator, List, Optional, Tuple

from ..simulation.engine import SimulationEngine, SimulationRunner, WorldStats
from ..simulation.events import Event

# ANSI control sequences.
ALT_SCREEN_ON = "\033[?1049h"
ALT_SCREEN_OFF = "\033[?1049l"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CURSOR_HOME = "\033[H"
CLEAR_LINE = "\033[K"
CLEAR_BELOW = "\033[J"

MAX_WIDTH = 78
MIN_WIDTH = 40

# Optional raw-terminal support. Absent on Windows, where the view degrades to
# a repainting display that runs until Ctrl-C.
try:  # pragma: no cover - platform dependent
    import select
    import termios
    import tty

    RAW_INPUT_AVAILABLE = True
except ImportError:  # pragma: no cover - Windows
    RAW_INPUT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Terminal plumbing
# ---------------------------------------------------------------------------


@contextmanager
def live_terminal(interactive: bool) -> Iterator[None]:
    """Switch to the alternate screen (and cbreak mode) for the duration.

    ``cbreak`` rather than full raw mode, so Ctrl-C still raises
    :class:`KeyboardInterrupt` and the user is never trapped.
    """
    fd = sys.stdin.fileno() if interactive else -1
    saved: Optional[list] = None
    if interactive and RAW_INPUT_AVAILABLE:
        saved = termios.tcgetattr(fd)
    sys.stdout.write(ALT_SCREEN_ON + HIDE_CURSOR)
    sys.stdout.flush()
    try:
        if saved is not None:
            tty.setcbreak(fd)
        yield
    finally:
        if saved is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        sys.stdout.write(SHOW_CURSOR + ALT_SCREEN_OFF)
        sys.stdout.flush()


def read_key(timeout: float) -> Optional[str]:
    """Wait up to ``timeout`` seconds for a keypress; ``None`` if none arrived."""
    if not RAW_INPUT_AVAILABLE:
        time.sleep(timeout)
        return None
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    try:
        data = os.read(sys.stdin.fileno(), 8)
    except OSError:
        return None
    if not data:
        return None
    return data.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------


def _width() -> int:
    """Usable frame width, clamped to something readable."""
    return max(MIN_WIDTH, min(shutil.get_terminal_size((80, 24)).columns - 1, MAX_WIDTH))


def _bar(value: float, maximum: float, width: int) -> str:
    """A proportional bar, e.g. ``████████░░░░``."""
    if maximum <= 0 or width <= 0:
        return ""
    filled = int(round(width * max(0.0, min(1.0, value / maximum))))
    return "█" * filled + "░" * (width - filled)


def _activity_row(label: str, count: int, population: int, width: int) -> str:
    """One ``label  count  bar`` line of the activity breakdown."""
    bar_width = max(8, width - 32)
    return f"  {label:<20}{count:>5}  {_bar(count, population or 1, bar_width)}"


def render_frame(
    stats: WorldStats,
    events: List[Event],
    running: bool,
    speed: str,
    rate: float,
    rows: int,
) -> str:
    """Build one complete frame of the live view.

    Pure function of its inputs, so it can be tested without a terminal. The
    event feed is trimmed to whatever space is left in ``rows``.
    """
    width = _width()
    rule = "─" * width
    status = "▶ RUNNING" if running else "❚❚ PAUSED"
    right = f"{status}   {rate:6.1f} days/s   speed {speed}"
    headline = "WORLDBOX" + " " * max(1, width - 8 - len(right)) + right

    food_pct = (stats.total_food / stats.food_capacity * 100.0) if stats.food_capacity else 0.0
    terrain = stats.terrain_counts

    # Always shown: the header, the core numbers, and the control footer.
    core = [
        headline,
        rule,
        f"  Day {stats.day:<10,} Year {stats.year:<8} Seed {stats.seed}",
        "",
        f"  Population        {stats.population:>6,}"
        f"     Births today  {stats.births_today:>4}",
        f"  Total births      {stats.total_births:>6,}"
        f"     Deaths today  {stats.deaths_today:>4}",
        f"  Total deaths      {stats.total_deaths:>6,}"
        f"     Oldest        {stats.oldest_age_years:>6.1f}y",
        "",
        f"  Average Age       {stats.average_age_years:>6.1f}"
        f"     Avg Hunger  {stats.average_hunger:>6.1f}",
        f"  Average Health    {stats.average_health:>6.1f}"
        f"     Avg Energy  {stats.average_energy:>6.1f}",
    ]
    footer = [
        rule,
        "  [space] run/pause  [s] step  [+/-] speed  [q] prompt",
    ]

    activity = [
        "",
        "  ACTIVITY",
        _activity_row("Hungry", stats.hungry, stats.population, width),
        _activity_row("Resting", stats.resting, stats.population, width),
        _activity_row("Searching for Food", stats.seeking_food, stats.population, width),
        _activity_row("Eating", stats.eating, stats.population, width),
        _activity_row("Seeking a Mate", stats.seeking_mate, stats.population, width),
        _activity_row("Fighting", stats.fighting, stats.population, width),
        _activity_row("Wandering", stats.wandering, stats.population, width),
    ]
    civilisation = [
        "",
        "  CIVILISATION",
        f"  {'Era':<20}{stats.most_advanced_era}",
        f"  {'Tribes':<20}{stats.tribes:>5}   settlements {stats.settlements}"
        + (
            f" (largest: {stats.largest_settlement_level})"
            if stats.largest_settlement_level
            else ""
        ),
        f"  {'Technologies':<20}{stats.technologies_known:>5} / 16"
        f"   stored food {stats.total_food_stored:,.0f}",
        f"  {'Wars':<20}{stats.active_wars:>5}   battles today {stats.battles_today}"
        f"   war dead {stats.war_deaths}",
        f"  {'Ill':<20}{stats.ill:>5}"
        + (f"   ({stats.current_plague})" if stats.current_plague else "")
        + f"   plague dead {stats.plague_deaths}",
    ]
    environment = [
        "",
        "  ENVIRONMENT",
        f"  {'Food remaining':<20}{food_pct:>4.0f}%  "
        f"{_bar(stats.total_food, stats.food_capacity, max(8, width - 32))}",
        "  " + "  ".join(f"{name} {count}" for name, count in sorted(terrain.items())),
    ]

    # Sections are dropped, largest-optional-first, until the frame fits the
    # window. A frame taller than the terminal would scroll, which breaks
    # in-place repainting -- so it must never happen.
    lines = list(core)
    budget = rows - len(core) - len(footer)
    if budget >= len(activity):
        lines += activity
        budget -= len(activity)
    if budget >= len(civilisation):
        lines += civilisation
        budget -= len(civilisation)
    if budget >= len(environment):
        lines += environment
        budget -= len(environment)
    if budget >= 3 and events:
        lines += ["", "  RECENT EVENTS"]
        budget -= 2
        feed = events[-budget:]
        lines += [f"  • {event.message}" for event in feed]
        budget -= len(feed)
    lines += [""] * max(0, budget)
    lines += footer

    return "\n".join(line[:width] for line in lines[:rows])


# ---------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------


class LiveView:
    """A continuously repainting dashboard with single-key controls."""

    def __init__(self, engine: SimulationEngine, runner: SimulationRunner) -> None:
        self.engine = engine
        self.runner = runner
        self.config = engine.config.simulation
        self.speed_index = self.config.default_speed_index
        self._samples: Deque[Tuple[float, int]] = deque(maxlen=40)

    # -- speed --------------------------------------------------------------

    @property
    def delay(self) -> float:
        """Current tick delay in seconds."""
        return self.config.speed_steps[self.speed_index]

    @property
    def speed_label(self) -> str:
        """Speed shown in the header, e.g. ``"4/8"`` or ``"MAX"``."""
        highest = len(self.config.speed_steps) - 1
        if self.speed_index == highest and self.delay <= 0:
            return "MAX"
        return f"{self.speed_index + 1}/{highest + 1}"

    def change_speed(self, direction: int) -> None:
        """Step the speed up (+1) or down (-1) through the configured steps."""
        highest = len(self.config.speed_steps) - 1
        self.speed_index = max(0, min(highest, self.speed_index + direction))
        self.runner.set_delay(self.delay)

    def _measure_rate(self, day: int) -> float:
        """Observed simulation speed in days per second, over a short window."""
        now = time.monotonic()
        self._samples.append((now, day))
        while len(self._samples) > 1 and now - self._samples[0][0] > 2.0:
            self._samples.popleft()
        if len(self._samples) < 2:
            return 0.0
        elapsed = self._samples[-1][0] - self._samples[0][0]
        if elapsed <= 0:
            return 0.0
        return (self._samples[-1][1] - self._samples[0][1]) / elapsed

    # -- main loop ----------------------------------------------------------

    def run(self, start_running: bool = True) -> None:
        """Take over the screen until the user presses ``q`` (or Ctrl-C).

        Leaving the live view always pauses the simulation, so the command
        prompt never shows a display that is silently going stale.
        """
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
        self.runner.set_delay(self.delay)
        if start_running:
            self.runner.start()

        try:
            with live_terminal(interactive):
                self._loop(interactive)
        except KeyboardInterrupt:
            pass
        finally:
            self.runner.pause()

        if not interactive:
            print("Live view needs an interactive terminal; simulation paused.")

    def _loop(self, interactive: bool) -> None:
        refresh = self.config.live_refresh_seconds
        first_frame = True

        while True:
            with self.runner.lock:
                stats = self.engine.stats()
                events = self.engine.events.recent(12)
                extinct = not self.engine.agents
            rows = max(20, shutil.get_terminal_size((80, 24)).lines - 1)
            frame = render_frame(
                stats,
                events,
                self.runner.running,
                self.speed_label,
                self._measure_rate(stats.day),
                rows,
            )
            self._paint(frame, full=first_frame)
            first_frame = False

            if extinct:
                self.runner.pause()

            keys = read_key(refresh)
            if keys and not self._handle_keys(keys):
                return

    def _paint(self, frame: str, full: bool) -> None:
        """Write a frame in place, clearing each line as it goes."""
        out = [CURSOR_HOME]
        out.extend(line + CLEAR_LINE + "\n" for line in frame.split("\n"))
        out.append(CLEAR_BELOW)
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def _handle_keys(self, data: str) -> bool:
        """Act on everything in one read; False when the view should close.

        A single read can contain several keypresses when the user types
        quickly, so each character is handled in turn. Multi-byte escape
        sequences (arrow keys, function keys) are ignored as a unit -- only a
        lone Escape means "quit".
        """
        if data.startswith("\x1b"):
            return len(data) > 1
        for key in data:
            if not self._handle_key(key):
                return False
        return True

    def _handle_key(self, key: str) -> bool:
        """Act on a single keypress. Returns False when the view should close."""
        if key in ("q", "Q", "\x03"):
            return False
        if key == " ":
            if self.runner.running:
                self.runner.pause()
            else:
                self.runner.start()
        elif key in ("s", "S"):
            if not self.runner.running:
                with self.runner.lock:
                    self.engine.step()
        elif key in ("+", "="):
            self.change_speed(+1)
        elif key in ("-", "_"):
            self.change_speed(-1)
        return True
