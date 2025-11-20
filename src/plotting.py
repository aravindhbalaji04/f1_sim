"""
Visualization helpers for race simulation outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt


def _ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_distribution_curves(
    strategy_samples: Mapping[str, Sequence[float]],
    output_dir: str | Path = "plots",
    bins: int = 80,
) -> Path:
    """
    Plot probability density-style histograms for each strategy.

    Args:
        strategy_samples: Mapping of strategy name to race-time samples.
        output_dir: Folder where the plot image will be saved.
        bins: Histogram bins used to approximate the distribution.
    """
    output_path = _ensure_output_dir(output_dir) / "race_time_distributions.png"
    plt.figure(figsize=(10, 6))
    for name, samples in strategy_samples.items():
        plt.hist(
            samples,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=name,
        )
    plt.xlabel("Total race time (s)")
    plt.ylabel("Density")
    plt.title("Race Time Distributions by Strategy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_strategy_boxplot(
    strategy_samples: Mapping[str, Sequence[float]],
    output_dir: str | Path = "plots",
) -> Path:
    """
    Generate a boxplot comparing overall race time distribution per strategy.
    """
    output_path = _ensure_output_dir(output_dir) / "race_time_boxplot.png"
    plt.figure(figsize=(8, 6))
    labels = list(strategy_samples.keys())
    data = [strategy_samples[label] for label in labels]
    plt.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanline=True,
        patch_artist=True,
    )
    plt.ylabel("Total race time (s)")
    plt.title("Strategy Comparison (Boxplot)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def generate_all_plots(
    strategy_samples: Mapping[str, Sequence[float]],
    output_dir: str | Path = "plots",
) -> Iterable[Path]:
    """
    Convenience wrapper that produces all available plots.
    """
    return [
        plot_distribution_curves(strategy_samples, output_dir=output_dir),
        plot_strategy_boxplot(strategy_samples, output_dir=output_dir),
    ]


__all__ = [
    "plot_distribution_curves",
    "plot_strategy_boxplot",
    "generate_all_plots",
]

