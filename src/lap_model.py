"""
Lap time simulation that incorporates base pace, fuel effect, tire degradation,
and random variability to emulate on-track conditions.
"""

from __future__ import annotations

import random
from typing import Dict

from .degradation import get_degradation

BASE_LAP_TIMES: Dict[str, float] = {
    "soft": 88.5,
    "medium": 89.3,
    "hard": 90.1,
}

FUEL_COEFFICIENT = 0.035  # seconds of penalty per lap of fuel remaining
VARIANCE = 0.12  # standard deviation for stochastic component


def simulate_lap(compound: str, lap_age: float, fuel_laps: float) -> float:
    """
    Simulate the lap time for the given compound and conditions.

    Args:
        compound: Tire compound in use.
        lap_age: Number of laps already completed on the current tire set.
        fuel_laps: Estimated laps of fuel remaining (higher => heavier car).

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
    noise = random.gauss(0.0, VARIANCE)

    return base_time + fuel_penalty + tire_deg + noise


__all__ = ["simulate_lap", "BASE_LAP_TIMES", "FUEL_COEFFICIENT", "VARIANCE"]

