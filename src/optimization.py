"""
Heuristic search algorithms for discovering race strategies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .montecarlo import MonteCarloSimulator, UncertaintyBounds
from .strategy import Strategy, StintPlan, VALID_COMPOUNDS

COMPOUND_INDEX: Dict[str, int] = {compound: idx for idx, compound in enumerate(VALID_COMPOUNDS)}


@dataclass
class StrategyCandidate:
    """Pit stop (lap) plan and compound assignment."""

    pit_laps: List[int]
    compounds: List[str]

    def stint_lengths(self, race_laps: int) -> List[int]:
        laps = self.pit_laps + [race_laps]
        previous = 0
        lengths: List[int] = []
        for lap in laps:
            lengths.append(lap - previous)
            previous = lap
        return lengths

    def to_strategy(self, name: str, race_laps: int) -> Strategy:
        lengths = self.stint_lengths(race_laps)
        stints = [
            StintPlan(compound=self.compounds[idx], length=lengths[idx])
            for idx in range(len(lengths))
        ]
        return Strategy(name=name, stints=stints)


@dataclass
class OptimizationConfig:
    race_laps: int
    max_stops: int = 3
    min_stint_laps: int = 8
    evaluation_runs: int = 600
    driver_sigma: float = 0.08
    uncertainty: UncertaintyBounds | None = None

    def __post_init__(self) -> None:
        if self.max_stops < 1:
            raise ValueError("max_stops must be at least 1.")
        if self.min_stint_laps < 1:
            raise ValueError("min_stint_laps must be positive.")


@dataclass
class EvaluationRecord:
    candidate: StrategyCandidate
    score: float


@dataclass
class OptimizationResult:
    strategy: Strategy
    score: float
    history: List[float]
    algorithm: str


class StrategyEvaluator:
    """Shared Monte Carlo engine for optimization runs."""

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.simulator = MonteCarloSimulator(
            race_laps=config.race_laps,
            runs=config.evaluation_runs,
            driver_sigma=config.driver_sigma,
            uncertainty=config.uncertainty,
        )

    def evaluate(self, candidate: StrategyCandidate, label: str) -> EvaluationRecord:
        strategy = candidate.to_strategy(label, self.config.race_laps)
        result = self.simulator.run_strategy(strategy)
        return EvaluationRecord(candidate=candidate, score=result.stats.mean)


class StrategyOptimizer:
    """Base optimizer interface."""

    def __init__(self, config: OptimizationConfig, evaluator: StrategyEvaluator, name: str) -> None:
        self.config = config
        self.evaluator = evaluator
        self.name = name
        self.records: List[EvaluationRecord] = []
        self.history: List[float] = []
        self.evaluations = 0

    def optimize(self, iterations: int) -> OptimizationResult:  # pragma: no cover - interface
        raise NotImplementedError

    # Helper utilities -------------------------------------------------
    def _random_candidate(self) -> StrategyCandidate:
        stops = random.randint(1, self.config.max_stops)
        stints = stops + 1
        lengths = _random_partition(self.config.race_laps, stints, self.config.min_stint_laps)
        pit_laps = []
        acc = 0
        for length in lengths[:-1]:
            acc += length
            pit_laps.append(acc)
        compounds = [random.choice(VALID_COMPOUNDS) for _ in range(stints)]
        return StrategyCandidate(pit_laps=pit_laps, compounds=compounds)

    def _record(self, record: EvaluationRecord) -> None:
        self.records.append(record)
        best = min(self.records, key=lambda rec: rec.score)
        self.history.append(best.score)

    def _evaluate(self, candidate: StrategyCandidate) -> EvaluationRecord:
        self.evaluations += 1
        label = f"{self.name}-cand-{self.evaluations}"
        record = self.evaluator.evaluate(candidate, label=label)
        self._record(record)
        return record

    def _best_result(self) -> OptimizationResult:
        best = min(self.records, key=lambda rec: rec.score)
        strategy = best.candidate.to_strategy(f"{self.name}-optimized", self.config.race_laps)
        return OptimizationResult(strategy=strategy, score=best.score, history=self.history, algorithm=self.name)


class BayesianOptimizer(StrategyOptimizer):
    """Lightweight kernel regression optimizer with LCB acquisition."""

    def __init__(
        self,
        config: OptimizationConfig,
        evaluator: StrategyEvaluator,
        kernel_sigma: float = 0.35,
        exploration: float = 0.20,
        initial_samples: int = 8,
        candidate_pool: int = 60,
    ) -> None:
        super().__init__(config, evaluator, name="bayesian")
        self.kernel_sigma = kernel_sigma
        self.exploration = exploration
        self.initial_samples = initial_samples
        self.candidate_pool = candidate_pool
        self.max_stints = self.config.max_stops + 1

    def optimize(self, iterations: int) -> OptimizationResult:
        for _ in range(self.initial_samples):
            self._evaluate(self._random_candidate())

        for _ in range(iterations):
            candidate = self._acquire_candidate()
            self._evaluate(candidate)

        return self._best_result()

    def _acquire_candidate(self) -> StrategyCandidate:
        best_score = float("inf")
        best_candidate = None
        for _ in range(self.candidate_pool):
            candidate = self._random_candidate()
            mean, std = self._predict(candidate)
            acquisition = mean - self.exploration * std
            if acquisition < best_score:
                best_score = acquisition
                best_candidate = candidate
        assert best_candidate is not None
        return best_candidate

    def _predict(self, candidate: StrategyCandidate) -> Tuple[float, float]:
        vector = _candidate_vector(candidate, self.config, self.max_stints)
        weights: List[float] = []
        values: List[float] = []
        for record in self.records:
            other_vec = _candidate_vector(record.candidate, self.config, self.max_stints)
            distance = _squared_distance(vector, other_vec)
            weight = math.exp(-distance / (2 * self.kernel_sigma**2))
            weights.append(weight)
            values.append(record.score)

        if not weights or sum(weights) == 0:
            best = min(self.records, key=lambda rec: rec.score)
            return best.score, 1.5

        total_weight = sum(weights)
        mean = sum(w * v for w, v in zip(weights, values)) / total_weight
        variance = (
            sum(w * (v - mean) ** 2 for w, v in zip(weights, values)) / total_weight
        )
        std = math.sqrt(max(variance, 1e-6))
        return mean, std


class GeneticAlgorithmOptimizer(StrategyOptimizer):
    """Simple generational genetic algorithm."""

    def __init__(
        self,
        config: OptimizationConfig,
        evaluator: StrategyEvaluator,
        population_size: int = 26,
        mutation_rate: float = 0.25,
        elite_count: int = 2,
    ) -> None:
        super().__init__(config, evaluator, name="genetic")
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count

    def optimize(self, iterations: int) -> OptimizationResult:
        population = [self._random_candidate() for _ in range(self.population_size)]
        scores: Dict[Tuple, float] = {}

        def fitness(candidate: StrategyCandidate) -> float:
            key = _candidate_key(candidate)
            if key not in scores:
                record = self._evaluate(candidate)
                scores[key] = record.score
            return scores[key]

        for _ in range(iterations):
            population.sort(key=fitness)

            next_generation = population[: self.elite_count]
            while len(next_generation) < self.population_size:
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                child = self._crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = _mutate_candidate(child, self.config)
                next_generation.append(child)
            population = next_generation

        # Ensure scores captured for final population
        for candidate in population:
            fitness(candidate)

        return self._best_result()

    def _tournament_selection(self, population: List[StrategyCandidate], fitness) -> StrategyCandidate:
        competitors = random.sample(population, k=min(3, len(population)))
        competitors.sort(key=fitness)
        return competitors[0]

    def _crossover(self, a: StrategyCandidate, b: StrategyCandidate) -> StrategyCandidate:
        len_a = len(a.compounds)
        len_b = len(b.compounds)
        if len_a != len_b:
            template = random.choice([a, b])
            return StrategyCandidate(pit_laps=template.pit_laps[:], compounds=template.compounds[:])

        cut = random.randint(1, len_a - 1)
        lengths_a = a.stint_lengths(self.config.race_laps)
        lengths_b = b.stint_lengths(self.config.race_laps)
        child_lengths = lengths_a[:cut] + lengths_b[cut:]
        child_lengths = _rebalance_lengths(child_lengths, self.config)

        child_compounds = a.compounds[:cut] + b.compounds[cut:]
        return _candidate_from_lengths(child_lengths, child_compounds)


class SimulatedAnnealingOptimizer(StrategyOptimizer):
    """Simulated annealing with simple neighborhood mutations."""

    def __init__(
        self,
        config: OptimizationConfig,
        evaluator: StrategyEvaluator,
        initial_temperature: float = 2.0,
        cooling_rate: float = 0.95,
    ) -> None:
        super().__init__(config, evaluator, name="annealing")
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def optimize(self, iterations: int) -> OptimizationResult:
        current = self._random_candidate()
        current_record = self._evaluate(current)
        best_record = current_record
        temperature = self.initial_temperature

        for _ in range(iterations):
            neighbor = _mutate_candidate(current, self.config)
            neighbor_record = self._evaluate(neighbor)

            delta = neighbor_record.score - current_record.score
            if delta < 0 or math.exp(-delta / max(temperature, 1e-6)) > random.random():
                current = neighbor
                current_record = neighbor_record
                if neighbor_record.score < best_record.score:
                    best_record = neighbor_record

            temperature *= self.cooling_rate

        return self._best_result()


# Helper functions ------------------------------------------------------------

def _random_partition(total: int, parts: int, minimum: int) -> List[int]:
    if parts * minimum > total:
        raise ValueError("Cannot partition total laps with current constraints.")
    remaining = total - parts * minimum
    lengths = [minimum] * parts
    for _ in range(remaining):
        idx = random.randrange(parts)
        lengths[idx] += 1
    random.shuffle(lengths)
    # Ensure order reconstitutes race distance
    return lengths


def _candidate_from_lengths(lengths: List[int], compounds: List[str]) -> StrategyCandidate:
    if len(compounds) < len(lengths):
        compounds = compounds + [
            random.choice(VALID_COMPOUNDS) for _ in range(len(lengths) - len(compounds))
        ]
    else:
        compounds = compounds[: len(lengths)]
    pit_laps = []
    acc = 0
    for length in lengths[:-1]:
        acc += length
        pit_laps.append(acc)
    return StrategyCandidate(pit_laps=pit_laps, compounds=compounds)


def _rebalance_lengths(lengths: List[int], config: OptimizationConfig) -> List[int]:
    lengths = lengths[:]
    total = sum(lengths)
    target = config.race_laps

    if total > target:
        excess = total - target
        while excess > 0:
            idx = random.randrange(len(lengths))
            if lengths[idx] > config.min_stint_laps:
                lengths[idx] -= 1
                excess -= 1
    elif total < target:
        deficit = target - total
        while deficit > 0:
            idx = random.randrange(len(lengths))
            lengths[idx] += 1
            deficit -= 1

    diff = sum(lengths) - target
    if diff != 0:
        lengths[-1] -= diff

    if lengths[-1] < config.min_stint_laps:
        shortfall = config.min_stint_laps - lengths[-1]
        lengths[-1] = config.min_stint_laps
        idx = 0
        while shortfall > 0 and idx < len(lengths) - 1:
            available = lengths[idx] - config.min_stint_laps
            take = min(available, shortfall)
            if take > 0:
                lengths[idx] -= take
                shortfall -= take
            idx += 1
    return lengths


def _mutate_candidate(candidate: StrategyCandidate, config: OptimizationConfig) -> StrategyCandidate:
    lengths = candidate.stint_lengths(config.race_laps)
    if len(lengths) > 1:
        i = random.randrange(len(lengths))
        j = (i + 1) % len(lengths)
        delta = random.choice([-2, -1, 1, 2])
        if lengths[i] + delta >= config.min_stint_laps and lengths[j] - delta >= config.min_stint_laps:
            lengths[i] += delta
            lengths[j] -= delta

    compounds = candidate.compounds[:]
    idx = random.randrange(len(compounds))
    compounds[idx] = random.choice(VALID_COMPOUNDS)

    return _candidate_from_lengths(lengths, compounds)


def _candidate_key(candidate: StrategyCandidate) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
    return (tuple(candidate.pit_laps), tuple(candidate.compounds))


def _candidate_vector(candidate: StrategyCandidate, config: OptimizationConfig, max_stints: int) -> List[float]:
    lengths = candidate.stint_lengths(config.race_laps)
    norm_lengths = [length / config.race_laps for length in lengths]
    norm_lengths += [0.0] * (max_stints - len(norm_lengths))

    compounds = [
        COMPOUND_INDEX.get(compound, 0) / max(len(VALID_COMPOUNDS) - 1, 1)
        for compound in candidate.compounds
    ]
    compounds += [0.0] * (max_stints - len(compounds))
    return norm_lengths + compounds


def _squared_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(vec_a, vec_b))


def build_optimizer(kind: str, config: OptimizationConfig) -> StrategyOptimizer:
    evaluator = StrategyEvaluator(config)
    kind = kind.lower()
    if kind == "bayesian":
        return BayesianOptimizer(config, evaluator)
    if kind == "genetic":
        return GeneticAlgorithmOptimizer(config, evaluator)
    if kind == "annealing":
        return SimulatedAnnealingOptimizer(config, evaluator)
    raise ValueError(f"Unknown optimizer '{kind}'.")


__all__ = [
    "OptimizationConfig",
    "OptimizationResult",
    "StrategyOptimizer",
    "BayesianOptimizer",
    "GeneticAlgorithmOptimizer",
    "SimulatedAnnealingOptimizer",
    "build_optimizer",
]

