"""
Command-line entry point for running the F1 race strategy simulator.
"""

from __future__ import annotations

from typing import List, Tuple

from .montecarlo import MonteCarloSimulator, SimulationStats
from .strategy import Strategy, get_preset_strategies

RACE_LAPS = 58
RUNS = 5000


def format_stats(strategy: Strategy, stats: SimulationStats) -> str:
    return (
        f"{strategy.name:<20}"
        f"{stats.mean:>10.3f}s"
        f"{stats.median:>10.3f}s"
        f"{stats.p05:>10.3f}s"
        f"{stats.p95:>10.3f}s"
    )


def run() -> None:
    strategies = get_preset_strategies(RACE_LAPS)
    simulator = MonteCarloSimulator(race_laps=RACE_LAPS, runs=RUNS)

    results: List[Tuple[Strategy, SimulationStats]] = []
    print(f"Running {RUNS} simulations per strategy over {RACE_LAPS} laps...\n")
    for strategy in strategies:
        stats = simulator.run_strategy(strategy)
        results.append((strategy, stats))
        print(format_stats(strategy, stats))

    best_strategy, best_stats = min(results, key=lambda item: item[1].mean)

    print("\nBest strategy based on mean race time:")
    print(f"- {best_strategy.name} ({best_stats.mean:.3f}s average)")


if __name__ == "__main__":
    run()

