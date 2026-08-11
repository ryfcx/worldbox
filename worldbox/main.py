"""Worldbox entry point.

Usage::

    python -m worldbox.main              # interactive terminal session
    python -m worldbox.main --seed 42    # start from a specific seed
    python -m worldbox.main --days 1000  # headless run, print a final summary
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from typing import List, Optional

from .cli.terminal import Terminal, render_dashboard, render_stats
from .config import Config
from .simulation.engine import SimulationEngine
from .world.terrain import WorldGenerationError


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="worldbox",
        description="Worldbox - a terminal-based artificial civilisation simulation.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for world generation."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Run this many days without interaction, print a summary, then exit.",
    )
    parser.add_argument(
        "--view",
        type=int,
        nargs="?",
        const=20000,
        default=None,
        metavar="DAYS",
        help="Simulate DAYS days (default 20000), build the map viewer and open it.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="With --view, write the page but do not open a browser.",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=None,
        help="Override the starting population.",
    )
    parser.add_argument(
        "--width", type=int, default=None, help="Override the world width in tiles."
    )
    parser.add_argument(
        "--height", type=int, default=None, help="Override the world height in tiles."
    )
    return parser


def build_config(args: argparse.Namespace) -> Config:
    """Apply command-line overrides on top of the default configuration."""
    config = Config()
    if args.width is not None or args.height is not None:
        config = dataclasses.replace(
            config,
            world=dataclasses.replace(
                config.world,
                width=args.width if args.width is not None else config.world.width,
                height=args.height if args.height is not None else config.world.height,
            ),
        )
    if args.population is not None:
        config = dataclasses.replace(
            config,
            agents=dataclasses.replace(config.agents, initial_population=args.population),
        )
    return config


def run_headless(engine: SimulationEngine, days: int) -> None:
    """Simulate ``days`` days with no interaction and print the result."""
    simulated = engine.run(days)
    print(render_dashboard(engine.stats(), engine.events.recent(10)))
    print()
    print(render_stats(engine.stats()))
    if simulated < days:
        print(f"\nStopped after {simulated} days: the population died out.")


def run_viewer(engine: SimulationEngine, days: int, open_browser: bool = True) -> int:
    """Simulate, build the standalone map page, and open it.

    This is the whole graphical path in one call, so seeing a world takes a
    single command rather than an export, a build step and a file hunt.
    """
    import webbrowser
    from pathlib import Path

    from .export import RunRecorder

    # tools/ is not a package, so put it on the path to reuse the builder
    # rather than duplicating the template-substitution logic here.
    tools = Path(__file__).resolve().parent.parent / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    try:
        from build_viewer import build
    except ImportError:
        print(f"Could not find the viewer builder in {tools}.", file=sys.stderr)
        return 1

    # Aim for ~200 frames whatever the run length, so the scrubber stays smooth.
    every = max(10, days // 200) if days > 0 else 10
    if days > 0:
        print(f"Simulating {days:,} days...")
    recorder = RunRecorder(engine, every=every)
    if days > 0:
        recorder.run(days)
    else:
        recorder.capture()

    data_path = Path("worldbox_run.json")
    page = Path("worldbox.html")
    try:
        recorder.write_json(data_path)
        build(data_path, page)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Could not build the viewer: {error}", file=sys.stderr)
        return 1

    stats = engine.stats()
    print(
        f"Year {stats.year}: {stats.population} people, {stats.tribes} tribes, "
        f"{stats.settlements} settlements, {stats.technologies_known}/16 technologies."
    )
    print(f"Wrote {page} ({page.stat().st_size / 1024:.0f} KB).")

    if open_browser:
        webbrowser.open(page.resolve().as_uri())
        print("Opened it in your browser.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Program entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        config = build_config(args)
        engine = SimulationEngine(config=config, seed=args.seed)
    except (ValueError, WorldGenerationError) as error:
        print(f"Could not start Worldbox: {error}", file=sys.stderr)
        return 1

    if args.view is not None:
        return run_viewer(engine, args.view, open_browser=not args.no_open)

    if args.days is not None:
        if args.days < 0:
            print("--days must not be negative.", file=sys.stderr)
            return 1
        run_headless(engine, args.days)
        return 0

    Terminal(engine).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
