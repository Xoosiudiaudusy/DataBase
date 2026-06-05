"""XGBoost binary classifier with walk-forward CV and isotonic calibration.

Predicts P(YES) = probability the bucket [lo, hi] contains the realized
daily high, given features observable at decision_moment.

Walk-forward protocol:
  - Sort all rows by resolve_date.
  - For each fold k: train on dates < cutoff[k], evaluate on dates in
    [cutoff[k], cutoff[k+1]).
  - Default: 30-day folds. ~16 months of data → ~12-14 folds.
  - Calibration (isotonic) is fit on a holdout slice of the training fold.

Metrics per fold:
  - log_loss, Brier score, AUC (ranking)
  - Strategy ROI: bet YES on rows where model_p > market_p + margin.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from .features import FEATURE_COLS


def _prep_X(df: pd.DataFrame, train_cities: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y). X has FEATURE_COLS + one-hot cities; y = yes_won."""
    X = df[FEATURE_COLS].copy()
    # One-hot city (consistent column order across folds)
    city_cols = sorted(df["city"].unique() if train_cities is None else train_cities)
    for c in city_cols:
        X[f"city_{c}"] = (df["city"] == c).astype(int)
    y = df["yes_won"].astype(int)
    return X, y


def walk_forward_folds(df: pd.DataFrame, fold_days: int = 30,
                       min_train_days: int = 90, min_test_rows: int = 0):
    """Generator of (train_mask, test_mask, cutoff_start, cutoff_end) by time.

    Folds are strictly forward — no leakage. `min_test_rows` filters out folds
    that don't have enough test data for stable metrics (default 0 = keep all).
    """
    df = df.sort_values("resolve_date").reset_index(drop=True)
    dates = pd.to_datetime(df["resolve_date"])
    d_min, d_max = dates.min(), dates.max()
    cutoffs = pd.date_range(d_min + pd.Timedelta(days=min_train_days),
                            d_max + pd.Timedelta(days=1), freq=f"{fold_days}D")
    for c in cutoffs:
        nxt = c + pd.Timedelta(days=fold_days)
        train = dates < c
        test = (dates >= c) & (dates < nxt)
        if test.sum() < max(1, min_test_rows):
            continue
        yield train, test, c, nxt


def fit_xgb(X_train, y_train, X_val=None, y_val=None, **params):
    p = dict(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        max_depth=4, eta=0.05, n_estimators=400,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=10, tree_method="hist",
        verbosity=0,
    )
    p.update(params)
    model = xgb.XGBClassifier(**p)
    eval_set = [(X_val, y_val)] if X_val is not None else None
    fit_kw = {}
    if eval_set is not None:
        fit_kw["eval_set"] = eval_set
        # XGB >=2.0 dropped early_stopping kwarg; we just use a fixed n_estimators
    model.fit(X_train, y_train, **fit_kw)
    return model


def calibrate(p_raw: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw, y)
    return iso


def strategy_pnl(model_p: np.ndarray, market_p: np.ndarray, yes_won: np.ndarray,
                 margin: float = 0.05, side: str = "yes") -> dict:
    """Bet $1 of YES (or NO) whenever model says it's mispriced by `margin`."""
    if side == "yes":
        bets = model_p > market_p + margin
        cost = market_p[bets]
        wins = yes_won[bets]
    else:  # no-side
        bets = model_p < market_p - margin
        cost = 1 - market_p[bets]
        wins = 1 - yes_won[bets]
    n = int(bets.sum())
    if n == 0:
        return {"n": 0, "win_rate": None, "roi_pct": None, "pnl": 0.0}
    pnl = float(wins.sum() - cost.sum())
    return {
        "n": n, "win_rate": float(wins.mean()),
        "avg_cost": float(cost.mean()),
        "pnl": pnl,
        "roi_pct": pnl / float(cost.sum()) * 100 if cost.sum() > 0 else 0.0,
    }


def run(dataset_path: str, out_path: str = "results/ml_walkforward_report.parquet"):
    df = pd.read_parquet(dataset_path)
    df["resolve_date"] = pd.to_datetime(df["resolve_date"])
    # Need market_yes_price for strategy eval — left as None if absent
    has_mkt = "market_yes_price" in df.columns

    fold_metrics = []
    train_cities = sorted(df["city"].unique())

    for train_mask, test_mask, c0, c1 in walk_forward_folds(df):
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        # Split a calibration holdout from training
        train_idx, cal_idx = train_test_split(
            np.arange(len(train_df)), test_size=0.15, random_state=42,
            stratify=train_df["yes_won"])
        cal_df = train_df.iloc[cal_idx]
        fit_df = train_df.iloc[train_idx]

        X_fit, y_fit = _prep_X(fit_df, train_cities)
        X_cal, y_cal = _prep_X(cal_df, train_cities)
        X_test, y_test = _prep_X(test_df, train_cities)

        model = fit_xgb(X_fit, y_fit, X_cal, y_cal)
        p_cal_raw = model.predict_proba(X_cal)[:, 1]
        iso = calibrate(p_cal_raw, y_cal.values)

        p_test_raw = model.predict_proba(X_test)[:, 1]
        p_test = iso.transform(p_test_raw)

        m = {
            "fold_from": str(c0.date()), "fold_to": str(c1.date()),
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "logloss": float(log_loss(y_test, p_test.clip(1e-3, 1 - 1e-3))),
            "brier": float(brier_score_loss(y_test, p_test)),
            "auc": float(roc_auc_score(y_test, p_test)) if y_test.nunique() > 1 else None,
        }
        if has_mkt:
            for margin in (0.05, 0.10):
                for side in ("yes", "no"):
                    r = strategy_pnl(
                        p_test, test_df["market_yes_price"].values,
                        y_test.values, margin=margin, side=side)
                    m[f"{side}_m{int(margin*100):02d}_n"] = r["n"]
                    m[f"{side}_m{int(margin*100):02d}_roi"] = r["roi_pct"]
        fold_metrics.append(m)

    out = pd.DataFrame(fold_metrics)
    out.to_parquet(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {out_path}")
    return out


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "results/ml_dataset_2025-01-02_2026-05-05.parquet"
    run(p)
