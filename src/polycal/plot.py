"""Calibration plotting: empirical curve per lead time, with theoretical overlay."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .calibration import per_lead_mae, theoretical_curve


def plot_calibration(
    cal_df: pd.DataFrame, dataset_df: pd.DataFrame, out_path: Path | str,
    title: str = "Forecast calibration — NYC (KLGA)",
) -> None:
    """Render the calibration figure. One subplot, lines per lead time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    lead_times = sorted(cal_df["lead_time"].unique())
    mae_map = per_lead_mae(dataset_df)

    for i, lt in enumerate(lead_times):
        color = cmap(i / max(1, len(lead_times) - 1))
        sub = cal_df[cal_df["lead_time"] == lt].sort_values("bin_center")
        ax.plot(sub["bin_center"], sub["p_hat"], "-o", color=color,
                label=f"T={lt}h (n={int(sub['n'].sum())}, MAE={mae_map.get(lt, float('nan')):.2f}°F)")
        ax.fill_between(sub["bin_center"], sub["lo"], sub["hi"], color=color, alpha=0.15)

        mae = mae_map.get(lt)
        if mae and mae > 0:
            xs, ys = theoretical_curve(mae)
            ax.plot(xs, ys, "--", color=color, alpha=0.6)

    ax.axhline(0.5, color="gray", linewidth=0.5)
    ax.axvline(0.0, color="gray", linewidth=0.5)
    ax.set_xlabel("spread = forecast_high − threshold (°F)")
    ax.set_ylabel("empirical P(YES)")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
