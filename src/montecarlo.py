"""
Monte Carlo race simulation utilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, median
from typing import Dict, Iterable, List, Sequence

from .lap_model import simulate_lap
from .strategy import Strategy


@dataclass(frozen=True)
class SimulationStats:
    """Aggregate statistics for Monte Carlo runs."""

    mean: float
    median: float
    p05: float
    p95: float
    samples: int


@dataclass(frozen=True)
class StrategyResult:
    """Encapsulates raw samples and statistics for a strategy."""

    strategy: Strategy
    samples: List[float]
    stats: SimulationStats


class MonteCarloSimulator:
    """Run multiple race simulations for supplied strategies."""

    def __init__(self, race_laps: int, runs: int = 5000) -> None:
        if race_laps <= 0:
            raise ValueError("race_laps must be positive.")
        if runs <= 0:
            raise ValueError("runs must be positive.")
        self.race_laps = race_laps
        self.runs = runs

    def run_strategy(self, strategy: Strategy) -> StrategyResult:
        """Simulate a single strategy multiple times and summarize the result."""
        self._validate_strategy(strategy)
        samples = sorted(self._simulate_runs(strategy))
        stats = SimulationStats(
            mean=fmean(samples),
            median=median(samples),
            p05=_percentile(samples, 5.0),
            p95=_percentile(samples, 95.0),
            samples=self.runs,
        )
        return StrategyResult(strategy=strategy, samples=samples, stats=stats)

    def run_all(self, strategies: Sequence[Strategy]) -> Dict[str, StrategyResult]:
        """Run simulations for all provided strategies."""
        return {strategy.name: self.run_strategy(strategy) for strategy in strategies}

    def _simulate_runs(self, strategy: Strategy) -> List[float]:
        return [self._simulate_single_race(strategy) for _ in range(self.runs)]

    def _simulate_single_race(self, strategy: Strategy) -> float:
        total_time = 0.0
        completed_laps = 0
        for stint in strategy.stints:
            for lap_in_stint in range(stint.length):
                completed_laps += 1
                fuel_laps = self.race_laps - completed_laps
                total_time += simulate_lap(
                    compound=stint.compound,
                    lap_age=lap_in_stint,
                    fuel_laps=fuel_laps,
                )
        return total_time

    def _validate_strategy(self, strategy: Strategy) -> None:
        total = strategy.total_laps()
        if total != self.race_laps:
            raise ValueError(
                f"Strategy '{strategy.name}' covers {total} laps but race requires {self.race_laps}."
            )


def _percentile(data: List[float], percentile: float) -> float:
    if not data:
        raise ValueError("Cannot compute percentile of empty data.")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100.")

    if len(data) == 1:
        return data[0]

    k = (len(data) - 1) * (percentile / 100)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return data[int(k)]
    fraction = k - lower
    return data[lower] + fraction * (data[upper] - data[lower])


__all__ = ["MonteCarloSimulator", "SimulationStats", "StrategyResult"]

