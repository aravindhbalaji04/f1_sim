"""
Interactive dashboard for the F1 Race Strategy Simulator.

Run with: `streamlit run streamlit_app.py`
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# Ensure the src package is importable when launching via Streamlit
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.lap_model import set_degradation_mode
from src.montecarlo import (
    MonteCarloSimulator,
    StrategyResult,
    UncertaintyBounds,
    WeatherState,
)
from src.strategy import Strategy, get_preset_strategies

RACE_LAPS = 58
DRIVER_SKILLS = {
    "Elite (0.08s)": 0.08,
    "Pro (0.12s)": 0.12,
    "Average (0.20s)": 0.20,
}

WEATHER_OPTIONS = {
    "Sunny": {"lap_penalty": 0.0, "variance": 1.0, "wet": False},
    "Cloudy": {"lap_penalty": 0.15, "variance": 0.9, "wet": False},
    "Light Rain": {"lap_penalty": 1.2, "variance": 0.8, "wet": True},
    "Heavy Rain": {"lap_penalty": 2.5, "variance": 0.7, "wet": True},
}


def main() -> None:
    st.set_page_config(page_title="F1 Strategy Dashboard", layout="wide")
    st.title("F1 Race Strategy Dashboard")
    st.caption(
        "Explore race outcomes under different weather, driver, and uncertainty scenarios. "
        "All computations use the Monte Carlo simulator defined in `src/`."
    )

    preset_strategies = get_preset_strategies(RACE_LAPS)
    strategy_map = {strategy.name: strategy for strategy in preset_strategies}

    with st.sidebar:
        st.header("Simulation Controls")
        selected_names = st.multiselect(
            "Select strategies", list(strategy_map.keys()), default=list(strategy_map.keys())
        )
        runs = st.slider("Monte Carlo runs", 200, 5000, 1500, step=100)
        driver_skill_label = st.selectbox("Driver skill", list(DRIVER_SKILLS.keys()))
        driver_sigma = DRIVER_SKILLS[driver_skill_label]
        degradation_model = st.selectbox("Degradation model", ["analytical", "ml"])

        st.subheader("Environment")
        weather_label = st.selectbox("Weather", list(WEATHER_OPTIONS.keys()))
        track_temp = st.slider("Track temperature (°C)", 25.0, 55.0, 40.0, step=0.5)
        custom_penalty = st.slider("Weather lap penalty (s)", 0.0, 3.0, WEATHER_OPTIONS[weather_label]["lap_penalty"], 0.05)
        weather_variance = st.slider(
            "Random variance scale", 0.5, 1.2, WEATHER_OPTIONS[weather_label]["variance"], 0.05
        )

        st.subheader("Parameter Uncertainty")
        pit_range = st.slider("Pit stop loss (s)", 19.0, 24.0, (20.0, 22.0), 0.1)
        deg_range = st.slider("Degradation scale", 0.7, 1.3, (0.9, 1.1), 0.01)
        sc_range = st.slider("Safety car probability", 0.0, 0.4, (0.08, 0.18), 0.01)

        run_button = st.button("Run Simulation", type="primary")

    if not selected_names:
        st.warning("Select at least one strategy to begin.")
        return

    strategies = [strategy_map[name] for name in selected_names]

    if not run_button:
        st.info("Adjust parameters in the sidebar and click **Run Simulation**.")
        return

    # Configure runtime
    set_degradation_mode(degradation_model)
    weather = WeatherState(
        condition=weather_label,
        track_temp=track_temp,
        lap_penalty=custom_penalty,
        variance_scale=weather_variance,
        wet=WEATHER_OPTIONS[weather_label]["wet"],
    )
    bounds = UncertaintyBounds(
        pit_stop_range=_ordered_tuple(pit_range),
        degradation_scale_range=_ordered_tuple(deg_range),
        safety_car_probability_range=_ordered_tuple(sc_range),
    )

    simulator = MonteCarloSimulator(
        race_laps=RACE_LAPS,
        runs=runs,
        driver_sigma=driver_sigma,
        uncertainty=bounds,
        forced_weather=weather,
    )

    results = [simulator.run_strategy(strategy) for strategy in strategies]
    lap_sequences = {
        strategy.name: simulator.simulate_lap_sequence(strategy) for strategy in strategies
    }

    render_summary(results)
    render_win_probabilities(results)
    render_lap_times(lap_sequences)
    render_compound_usage(strategies)


def render_summary(results: List[StrategyResult]) -> None:
    st.subheader("Race Outcome Summary")
    data = [
        {
            "Strategy": res.strategy.name,
            "Mean (s)": res.stats.mean,
            "Median (s)": res.stats.median,
            "P05 (s)": res.stats.p05,
            "P95 (s)": res.stats.p95,
        }
        for res in results
    ]
    st.dataframe(pd.DataFrame(data).set_index("Strategy"), use_container_width=True)


def render_win_probabilities(results: List[StrategyResult]) -> None:
    st.subheader("Probability of Winning")
    samples = np.vstack([res.samples for res in results])
    best_indices = np.argmin(samples, axis=0)
    counts = np.bincount(best_indices, minlength=len(results))
    probs = counts / samples.shape[1]
    df = pd.DataFrame(
        {"Strategy": [res.strategy.name for res in results], "Win Probability": probs}
    ).set_index("Strategy")
    st.bar_chart(df)


def render_lap_times(sequences: Dict[str, List[float]]) -> None:
    st.subheader("Lap Time Trajectories")
    max_len = max(len(seq) for seq in sequences.values())
    data = {"Lap": list(range(1, max_len + 1))}
    for name, seq in sequences.items():
        padded = seq + [None] * (max_len - len(seq))
        data[name] = padded
    df = pd.DataFrame(data).set_index("Lap")
    st.line_chart(df)


def render_compound_usage(strategies: List[Strategy]) -> None:
    st.subheader("Compound Usage per Strategy")
    rows = []
    for strategy in strategies:
        for idx, stint in enumerate(strategy.stints, start=1):
            rows.append(
                {
                    "Strategy": strategy.name,
                    "Stint": idx,
                    "Compound": stint.compound.title(),
                    "Laps": stint.length,
                }
            )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


def _ordered_tuple(values) -> tuple[float, float]:
    a, b = values
    return (a, b) if a <= b else (b, a)


if __name__ == "__main__":
    main()

