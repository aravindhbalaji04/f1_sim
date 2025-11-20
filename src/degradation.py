"""
Tire degradation models for different compounds.

The degradation model follows a simple polynomial relationship:

    deg = a * lap_age**1.2 + b * lap_age

The coefficients (a, b) are tuned per compound to mimic the relative
durability of soft, medium, and hard tires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TireCompoundModel:
    """Encapsulates the coefficients for a tire compound degradation model."""

    name: str
    a: float
    b: float

    def degradation(self, lap_age: float) -> float:
        """Compute the degradation time loss for the given lap age."""
        if lap_age < 0:
            raise ValueError("lap_age must be non-negative")
        return self.a * lap_age ** 1.2 + self.b * lap_age


COMPOUND_MODELS: Dict[str, TireCompoundModel] = {
    "soft": TireCompoundModel("soft", a=0.020, b=0.035),
    "medium": TireCompoundModel("medium", a=0.015, b=0.025),
    "hard": TireCompoundModel("hard", a=0.010, b=0.020),
}


def get_degradation(compound: str, lap_age: float) -> float:
    """
    Compute tire degradation for the requested compound and lap age.

    Args:
        compound: One of "soft", "medium", or "hard" (case-insensitive).
        lap_age: Number of laps completed on the current tire set.

    Returns:
        The degradation penalty in seconds for the next lap.
    """
    key = compound.strip().lower()
    if key not in COMPOUND_MODELS:
        raise ValueError(
            f"Unknown compound '{compound}'. Expected one of {list(COMPOUND_MODELS)}."
        )
    return COMPOUND_MODELS[key].degradation(lap_age)


__all__ = ["get_degradation", "COMPOUND_MODELS", "TireCompoundModel"]

