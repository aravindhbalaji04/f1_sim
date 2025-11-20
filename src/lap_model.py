"""
Lap time simulation that incorporates base pace, fuel effect, tire degradation,
and random variability to emulate on-track conditions.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

from .degradation import get_degradation

BASE_LAP_TIMES: Dict[str, float] = {
    "soft": 88.5,
    "medium": 89.3,
    "hard": 90.1,
}

FUEL_COEFFICIENT = 0.035  # seconds of penalty per lap of fuel remaining
VARIANCE = 0.12  # standard deviation for stochastic component
PIT_STOP_LOSS = 21.0  # stationary time per pit stop
UNDERCUT_THRESHOLD = 4  # lap delta required for undercut bonus
UNDERCUT_BONUS = 0.15  # seconds gained during undercut window
UNDERCUT_LAPS = 3  # number of laps the bonus applies


def simulate_lap(
    compound: str,
    lap_age: float,
    fuel_laps: float,
    opponent_lap_age: Optional[float] = None,
) -> float:
    """
    Simulate the lap time for the given compound and conditions.

    Args:
        compound: Tire compound in use.
        lap_age: Laps already completed on the current tire set.
        fuel_laps: Estimated laps of fuel remaining (higher => heavier car).
        opponent_lap_age: Estimated lap age of the primary competitor's tires.

    Returns:
        Lap time in seconds.
    """
    key = compound.strip().lower()
    if key not in BASE_LAP_TIMES:
        raise ValueError(
            f"Unknown compound '{compound}'. Expected one of {list(BASE_LAP_TIMES)}."
        )

    base_time = BASE_LAP_TIMES[key]
    fuel_penalty = FUEL_COEFFICIENT * max(fuel_laps, 0)
    tire_deg = get_degradation(key, lap_age)
    opponent_age = opponent_lap_age if opponent_lap_age is not None else lap_age
    undercut_bonus = (
        UNDERCUT_BONUS
        if lap_age < UNDERCUT_LAPS and (opponent_age - lap_age) >= UNDERCUT_THRESHOLD
        else 0.0
    )
    noise = random.gauss(0.0, VARIANCE)

    return base_time + fuel_penalty + tire_deg + noise - undercut_bonus


__all__ = [
    "simulate_lap",
    "BASE_LAP_TIMES",
    "FUEL_COEFFICIENT",
    "VARIANCE",
    "PIT_STOP_LOSS",
]

