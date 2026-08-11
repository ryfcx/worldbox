#!/usr/bin/env python3
"""Compare two configurations across several seeds.

This is the regression harness for behaviour changes: run the same seeds under
two configs and print the difference in the numbers that matter. It is
deliberately small, and it is the seed of the larger experiment system.

    python3 tools/compare_runs.py --days 10000 --seeds 1337 2024 7

By default it compares the utility decision system against the original
first-match ladder, which is the change it was written to validate.
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

# Running this as a script puts tools/ on sys.path, not the repo root, so the
# package would not import. Add the repo root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldbox.config import Config  # noqa: E402
from worldbox.simulation.engine import SimulationEngine
from worldbox.simulation.events import EventKind


def measure(config: Config, seed: int, days: int) -> Dict[str, float]:
    """Run one simulation and return the numbers worth comparing."""
    engine = SimulationEngine(config=config, seed=seed)
    started = time.monotonic()
    engine.run(days)
    elapsed = time.monotonic() - started

    stats = engine.stats()
    agents = engine.agents
    population = len(agents) or 1
    agent_config = config.agents

    # How many agents sit within a single day's drift of a decision boundary.
    # Under a threshold ladder this clusters hard; under utility it should not.
    near = sum(
        1
        for a in agents
        if abs(a.needs.hunger - agent_config.hunger_hungry_threshold) < agent_config.hunger_per_day
        or abs(a.needs.energy - agent_config.energy_tired_threshold) < agent_config.energy_per_day
    )

    idle = sum(1 for a in agents if a.goal == "wander")

    return {
        "population": stats.population,
        "idle_pct": idle / population * 100.0,
        "boundary_pct": near / population * 100.0,
        "births": stats.total_births,
        "deaths": stats.total_deaths,
        "tribes": stats.tribes,
        "settlements": stats.settlements,
        "techs": stats.technologies_known,
        "war_dead": stats.war_deaths,
        "plague_dead": stats.plague_deaths,
        "avg_health": stats.average_health,
        "avg_hunger": stats.average_hunger,
        "errors": engine.events.total_recorded(EventKind.ERROR),
        "seconds": elapsed,
    }


def aggregate(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each measured value across seeds."""
    return {key: statistics.fmean([row[key] for row in rows]) for key in rows[0]}


def build_configs(baseline_flag: str) -> Dict[str, Config]:
    """The two configurations being compared."""
    utility_on = Config()
    utility_off = dataclasses.replace(
        utility_on, utility=dataclasses.replace(utility_on.utility, enabled=False)
    )
    return {baseline_flag: utility_off, "utility": utility_on}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two Worldbox configurations.")
    parser.add_argument("--days", type=int, default=6000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 2024, 7])
    args = parser.parse_args(argv)

    configs = build_configs("rules")
    results: Dict[str, Dict[str, float]] = {}

    for label, config in configs.items():
        rows = []
        for seed in args.seeds:
            print(f"  {label:<8} seed {seed} ...", end="", flush=True)
            row = measure(config, seed, args.days)
            print(f" pop {row['population']:.0f}  ({row['seconds']:.0f}s)")
            rows.append(row)
        results[label] = aggregate(rows)

    keys = [k for k in results["rules"] if k != "seconds"]
    label_a, label_b = "rules", "utility"
    width = max(len(k) for k in keys) + 2

    print(f"\n{args.days:,} days, {len(args.seeds)} seeds, means:\n")
    print(f"{'metric':<{width}}{label_a:>12}{label_b:>12}{'change':>12}")
    print("-" * (width + 36))
    for key in keys:
        a, b = results[label_a][key], results[label_b][key]
        if a:
            change = f"{(b - a) / abs(a) * 100:+.0f}%"
        else:
            change = "n/a" if not b else "new"
        print(f"{key:<{width}}{a:>12.1f}{b:>12.1f}{change:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
