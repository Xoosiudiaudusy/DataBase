"""Market-calibration plot: empirical winrate vs Polymarket YES price."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_market_calibration(
    cal_df: pd.DataFrame, joined_df: pd.DataFrame, out_path: Path | str,
    title: str = "Polymarket NYC temp — empirical winrate vs YES price",
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    cmap = plt.get_cmap("viridis")
    lead_times = sorted(cal_df["lead_time"].unique())

    ax.plot([0, 1], [0, 1], color="black", linewidth=0.7, linestyle=":",
            label="perfect calibration (y = x)")

    for i, lt in enumerate(lead_times):
        color = cmap(i / max(1, len(lead_times) - 1))
        sub = cal_df[cal_df["lead_time"] == lt].sort_values("bin_center")
        n_total = int(joined_df[joined_df["lead_time"] == lt].shape[0])
        ax.plot(sub["mean_price"], sub["p_hat"], "-o", color=color,
                label=f"T={lt}h  n={n_total}")
        ax.fill_between(sub["mean_price"], sub["lo"], sub["hi"],
                        color=color, alpha=0.15)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Polymarket YES price at T hours before peak heating")
    ax.set_ylabel("empirical P(YES)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.text(0.01, 0.01,
             "Below diagonal at high price → market overprices YES tail. "
             "Bands: Wilson 95% CI.", fontsize=7, color="gray")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
