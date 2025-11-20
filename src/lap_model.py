"""
Lap time simulation that incorporates base pace, fuel effect, tire degradation,
thermal behavior, track evolution, weather, and stochastic driver variability.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

from .degradation import get_degradation
from .ml_model import predict_degradation as predict_learned_degradation

BASE_LAP_TIMES: Dict[str, float] = {
    "soft": 88.5,
    "medium": 89.3,
    "hard": 90.1,
}

FUEL_COEFFICIENT = 0.035  # seconds of penalty per lap of fuel remaining
VARIANCE = 0.12  # baseline stochastic component
PIT_STOP_LOSS = 21.0  # stationary time per pit stop
UNDERCUT_THRESHOLD = 4  # lap delta required for undercut bonus
UNDERCUT_BONUS = 0.15  # seconds gained during undercut window
UNDERCUT_LAPS = 3  # number of laps the bonus applies
TRACK_IMPROVEMENT_RATE = -0.005  # seconds per lap

COMPOUND_HEATING = {
    "soft": 6.0,
    "medium": 5.0,
    "hard": 4.0,
}
OPTIMAL_TEMP = 95.0
MAX_TEMP = 110.0
COLD_PENALTY = 0.18
OVERHEAT_FACTOR = 0.01

DRS_BONUS = 0.2
ERS_BONUS = 0.25

DEGRADATION_MODE = "analytical"
VALID_DEGRADATION_MODES = ("analytical", "ml")
def set_degradation_mode(mode: str) -> None:
    """
    Switch between analytical and machine-learned degradation estimators.
    """
    normalized = mode.strip().lower()
    if normalized not in VALID_DEGRADATION_MODES:
        raise ValueError(f"Invalid degradation mode '{mode}'.")
    global DEGRADATION_MODE
    DEGRADATION_MODE = normalized


def _compute_tire_degradation(
    compound: str, lap_age: float, track_temp: float, weather_penalty: float
) -> float:
    if DEGRADATION_MODE == "ml":
        return predict_learned_degradation(compound, lap_age, track_temp, weather_penalty)
    return get_degradation(compound, lap_age)


def estimate_tire_temp(compound: str, lap_age: float, track_temp: float) -> float:
    heating = COMPOUND_HEATING.get(compound, 5.0)
    return track_temp + heating * min(lap_age, 8.0)


def temperature_penalty(compound: str, lap_age: float, track_temp: float) -> float:
    temp = estimate_tire_temp(compound, lap_age, track_temp)
    penalty = 0.0
    if lap_age < 2.0:
        penalty += COLD_PENALTY * (2.0 - lap_age)
    if temp > MAX_TEMP:
        penalty += (temp - MAX_TEMP) * OVERHEAT_FACTOR
    return penalty


def simulate_lap(
    compound: str,
    lap_age: float,
    fuel_laps: float,
    opponent_lap_age: Optional[float] = None,
    lap_number: int = 1,
    track_temp: float = 40.0,
    weather_penalty: float = 0.0,
    drs_active: bool = False,
    ers_active: bool = False,
    driver_sigma: float = 0.08,
    weather_variance_scale: float = 1.0,
) -> float:
    """
    Simulate the lap time for the given compound and conditions.

    Args:
        compound: Tire compound in use.
        lap_age: Laps already completed on the current tire set.
        fuel_laps: Estimated laps of fuel remaining (higher => heavier car).
        opponent_lap_age: Estimated lap age of the primary competitor's tires.
        lap_number: Absolute lap number in the race (1-indexed).
        track_temp: Current track temperature in Celsius.
        weather_penalty: Additional penalty driven by weather (e.g., rain).
        drs_active: Whether DRS is available on this lap.
        ers_active: Whether ERS deployment is used on this lap.
        driver_sigma: Driver-specific noise parameter.

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
    tire_deg = _compute_tire_degradation(key, lap_age, track_temp, weather_penalty)
    temp_penalty = temperature_penalty(key, lap_age, track_temp)
    track_evolution = TRACK_IMPROVEMENT_RATE * lap_number
    opponent_age = opponent_lap_age if opponent_lap_age is not None else lap_age
    undercut_bonus = (
        UNDERCUT_BONUS
        if lap_age < UNDERCUT_LAPS and (opponent_age - lap_age) >= UNDERCUT_THRESHOLD
        else 0.0
    )
    drs_bonus = DRS_BONUS if drs_active else 0.0
    ers_bonus = ERS_BONUS if ers_active else 0.0
    stochastic = random.gauss(0.0, VARIANCE) + random.gauss(
        0.0, driver_sigma * weather_variance_scale
    )

    lap_time = (
        base_time
        + fuel_penalty
        + tire_deg
        + temp_penalty
        + weather_penalty
        + track_evolution
        + stochastic
        - undercut_bonus
        - drs_bonus
        - ers_bonus
    )
    return max(lap_time, 0.0)


__all__ = [
    "simulate_lap",
    "BASE_LAP_TIMES",
    "FUEL_COEFFICIENT",
    "VARIANCE",
    "PIT_STOP_LOSS",
    "TRACK_IMPROVEMENT_RATE",
    "set_degradation_mode",
]

