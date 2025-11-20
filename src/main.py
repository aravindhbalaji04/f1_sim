"""
Command-line entry point for running the F1 race strategy simulator.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from .lap_model import set_degradation_mode
from .montecarlo import MonteCarloSimulator, SimulationStats, StrategyResult
from .optimization import (
    OptimizationConfig,
    OptimizationResult,
    build_optimizer,
)
from .plotting import generate_all_plots
from .strategy import Strategy, get_preset_strategies

RACE_LAPS = 58
RUNS = 5000
DRIVER_SKILL_SIGMA = {
    "elite": 0.08,
    "average": 0.20,
}


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
    parser.add_argument(
        "--driver-skill",
        choices=sorted(DRIVER_SKILL_SIGMA),
        default="elite",
        help="Driver skill preset affecting variance (elite=0.08s, average=0.20s).",
    )
    parser.add_argument(
        "--optimize",
        choices=["none", "bayesian", "genetic", "annealing"],
        default="none",
        help="Search for an optimal strategy using the selected meta-heuristic.",
    )
    parser.add_argument(
        "--optimization-iterations",
        type=int,
        default=60,
        help="Iterations/evaluations for the chosen optimizer.",
    )
    parser.add_argument(
        "--optimization-eval-runs",
        type=int,
        default=600,
        help="Monte Carlo runs per optimization evaluation (lower = faster).",
    )
    parser.add_argument(
        "--max-stops",
        type=int,
        default=3,
        help="Maximum number of pit stops the optimizer may schedule.",
    )
    parser.add_argument(
        "--min-stint",
        type=int,
        default=8,
        help="Minimum stint length enforced during optimization.",
    )
    parser.add_argument(
        "--with-presets",
        action="store_true",
        help="When optimizing, also evaluate the preset strategies for comparison.",
    )
    parser.add_argument(
        "--degradation-model",
        choices=["analytical", "ml"],
        default="analytical",
        help="Choose between the closed-form or ML-based degradation estimates.",
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


def load_strategies(
    args: argparse.Namespace, driver_sigma: float
) -> Tuple[List[Strategy], Optional[OptimizationResult]]:
    if args.optimize == "none":
        return get_preset_strategies(RACE_LAPS), None

    config = OptimizationConfig(
        race_laps=RACE_LAPS,
        max_stops=args.max_stops,
        min_stint_laps=args.min_stint,
        evaluation_runs=args.optimization_eval_runs,
        driver_sigma=driver_sigma,
    )
    optimizer = build_optimizer(args.optimize, config)
    optimization_result = optimizer.optimize(iterations=args.optimization_iterations)

    strategies = [optimization_result.strategy]
    if args.with_presets:
        strategies.extend(get_preset_strategies(RACE_LAPS))
    return strategies, optimization_result


def run() -> None:
    args = parse_args()
    set_degradation_mode(args.degradation_model)
    driver_sigma = DRIVER_SKILL_SIGMA[args.driver_skill]
    strategies, optimization = load_strategies(args, driver_sigma)
    simulator = MonteCarloSimulator(race_laps=RACE_LAPS, runs=RUNS, driver_sigma=driver_sigma)

    results: List[StrategyResult] = []
    print(
        f"Running {RUNS} simulations per strategy over {RACE_LAPS} laps "
        f"(driver: {args.driver_skill}, σ={driver_sigma:.2f})...\n"
    )
    for strategy in strategies:
        result = simulator.run_strategy(strategy)
        results.append(result)
        print(format_stats(result))

    best_result = min(results, key=lambda item: item.stats.mean)
    best_strategy, best_stats = best_result.strategy, best_result.stats

    print("\nBest strategy based on mean race time:")
    print(f"- {best_strategy.name} ({best_stats.mean:.3f}s average)")

    if optimization:
        print(
            f"\nOptimizer summary: {optimization.algorithm} best estimate "
            f"{optimization.score:.3f}s after {len(optimization.history)} evaluations."
        )

    if args.plot:
        samples = {result.strategy.name: result.samples for result in results}
        output_paths = generate_all_plots(samples, output_dir=args.plot_dir)
        print("\nPlots saved to:")
        for path in output_paths:
            print(f"- {Path(path).resolve()}")


if __name__ == "__main__":
    run()
