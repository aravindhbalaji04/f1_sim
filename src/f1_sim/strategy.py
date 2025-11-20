"""
Strategy definitions and helpers for race planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


VALID_COMPOUNDS = ("soft", "medium", "hard")


@dataclass(frozen=True)
class StintPlan:
    """Single stint specification."""

    compound: str
    length: int

    def __post_init__(self) -> None:
        if self.compound not in VALID_COMPOUNDS:
            raise ValueError(f"Invalid compound '{self.compound}'.")
        if self.length <= 0:
            raise ValueError("Stint length must be positive.")


@dataclass(frozen=True)
class Strategy:
    """A race strategy composed of ordered stints."""

    name: str
    stints: Sequence[StintPlan]

    def total_laps(self) -> int:
        """Return total laps covered by the strategy."""
        return sum(stint.length for stint in self.stints)


def _make_strategy(name: str, definition: Iterable[tuple[str, int]]) -> Strategy:
    stints = [StintPlan(compound=compound, length=length) for compound, length in definition]
    return Strategy(name=name, stints=stints)


def get_preset_strategies(race_laps: int) -> List[Strategy]:
    """
    Generate preset 1-stop, 2-stop, and 3-stop strategies for the given race.

    Args:
        race_laps: Total laps in the race.
    """
    base = [
        ("soft", int(race_laps * 0.35)),
        ("medium", int(race_laps * 0.35)),
        ("hard", int(race_laps * 0.3)),
    ]

    # Ensure total equals race distance
    def adjust(stints: List[StintPlan]) -> List[StintPlan]:
        delta = race_laps - sum(stint.length for stint in stints)
        if delta != 0:
            stints[-1] = StintPlan(stints[-1].compound, stints[-1].length + delta)
        return stints

    one_stop = adjust([StintPlan("soft", race_laps // 2), StintPlan("hard", race_laps - race_laps // 2)])

    two_stop_def = [
        ("soft", int(race_laps * 0.3)),
        ("medium", int(race_laps * 0.4)),
        ("hard", race_laps - int(race_laps * 0.7)),
    ]
    two_stop = adjust([StintPlan(*t) for t in two_stop_def])

    three_stop = adjust([StintPlan(*t) for t in base if t[1] > 0])

    return [
        Strategy("1-stop aggressive", one_stop),
        Strategy("2-stop balanced", two_stop),
        Strategy("3-stop push", three_stop),
    ]


__all__ = ["Strategy", "StintPlan", "get_preset_strategies", "VALID_COMPOUNDS"]

