"""
Command-line entry point for running the F1 race strategy simulator.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .montecarlo import MonteCarloSimulator, SimulationStats, StrategyResult
from .plotting import generate_all_plots
from .strategy import Strategy, get_preset_strategies

RACE_LAPS = 58
RUNS = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 race strategy simulator")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (requires matplotlib) and save them to the plots/ directory.",
    )
    parser.add_argument(
        "--plot-dir",
        default="plots",
        help="Destination directory for generated plots (default: plots/).",
    )
    return parser.parse_args()


def format_stats(result: StrategyResult) -> str:
    stats = result.stats
    return (
        f"{result.strategy.name:<20}"
        f"{stats.mean:>10.3f}s"
        f"{stats.median:>10.3f}s"
        f"{stats.p05:>10.3f}s"
        f"{stats.p95:>10.3f}s"
    )


def run() -> None:
    args = parse_args()
    strategies = get_preset_strategies(RACE_LAPS)
    simulator = MonteCarloSimulator(race_laps=RACE_LAPS, runs=RUNS)

    results: List[StrategyResult] = []
    print(f"Running {RUNS} simulations per strategy over {RACE_LAPS} laps...\n")
    for strategy in strategies:
        result = simulator.run_strategy(strategy)
        results.append(result)
        print(format_stats(result))

    best_result = min(results, key=lambda item: item.stats.mean)
    best_strategy, best_stats = best_result.strategy, best_result.stats

    print("\nBest strategy based on mean race time:")
    print(f"- {best_strategy.name} ({best_stats.mean:.3f}s average)")

    if args.plot:
        samples = {result.strategy.name: result.samples for result in results}
        output_paths = generate_all_plots(samples, output_dir=args.plot_dir)
        print("\nPlots saved to:")
        for path in output_paths:
            print(f"- {Path(path).resolve()}")


if __name__ == "__main__":
    run()

