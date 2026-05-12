"""Calibration plotting: empirical curve per lead time, with bias-adjusted normal overlay."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .calibration import per_lead_error_stats, theoretical_curve


def plot_calibration(
    cal_df: pd.DataFrame, dataset_df: pd.DataFrame, out_path: Path | str,
    title: str = "Forecast calibration — NYC (KLGA)",
) -> None:
    """Render the calibration figure. One subplot, lines per lead time."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    cmap = plt.get_cmap("viridis")
    lead_times = sorted(cal_df["lead_time"].unique())
    stats = per_lead_error_stats(dataset_df)

    for i, lt in enumerate(lead_times):
        color = cmap(i / max(1, len(lead_times) - 1))
        sub = cal_df[cal_df["lead_time"] == lt].sort_values("bin_center")
        bias = float(stats.loc[lt, "bias"])
        sigma = float(stats.loc[lt, "std"])
        mae = float(stats.loc[lt, "mae"])
        label = (f"T={lt}h  n={int(sub['n'].sum())}  "
                 f"MAE={mae:.2f}°F  bias={bias:+.2f}°F  σ={sigma:.2f}°F")
        ax.plot(sub["bin_center"], sub["p_hat"], "-o", color=color, label=label)
        ax.fill_between(sub["bin_center"], sub["lo"], sub["hi"], color=color, alpha=0.15)

        xs, ys = theoretical_curve(sigma_f=sigma, bias_f=bias)
        ax.plot(xs, ys, "--", color=color, alpha=0.7)

    ax.axhline(0.5, color="gray", linewidth=0.5)
    ax.axvline(0.0, color="gray", linewidth=0.5)
    ax.set_xlabel("spread = forecast_high − threshold (°F)")
    ax.set_ylabel("empirical P(YES)")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.text(0.01, 0.01,
             "Dashed: bias-adjusted normal Φ((spread − bias)/σ). "
             "Solid: empirical with Wilson 95% CI.",
             fontsize=7, color="gray")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
