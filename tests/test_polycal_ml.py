"""Smoke tests for polycal_ml pipeline.
We don't have all the data caches in tests, so we patch where needed."""
import datetime as dt
from pathlib import Path
import pandas as pd
import pytest

from polycal_ml.features import (
    FeatureRow, build_features, nbm_forecast_max, FEATURE_COLS
)


def test_feature_row_keys_complete():
    """Even without any cached data, build_features should return a dict
    with all expected keys (values may be None for missing inputs)."""
    r = FeatureRow(
        city="nyc", resolve_date=dt.date(2026, 4, 15),
        bucket_lo=84, bucket_hi=85, decision_lead_h=6, yes_won=0,
    )
    feats = build_features(r)
    assert "city" in feats and feats["city"] == "nyc"
    assert feats["bucket_lo"] == 84 and feats["bucket_hi"] == 85
    assert feats["bucket_width"] == 1
    assert feats["bucket_mid"] == 84.5
    assert feats["month"] == 4
    assert feats["day_of_year"] == 105
    assert feats["lat"] > 40 and feats["lat"] < 41  # KLGA lat
    # All declared feature cols should be present (even if None)
    for col in FEATURE_COLS:
        assert col in feats, f"missing feature {col}"


def test_nbm_max_missing_returns_none(tmp_path, monkeypatch):
    """With both the consolidated parquet and per-file extracts absent,
    nbm_forecast_max returns None gracefully."""
    import polycal_ml.features as feat
    monkeypatch.setattr(feat, "GRIB_EXT", tmp_path)
    monkeypatch.setattr(feat, "NBM_PARQUET", tmp_path / "nope.parquet")
    monkeypatch.setattr(feat, "_NBM_CACHE", None)  # force reload of empty index
    assert nbm_forecast_max("nyc", dt.date(2099, 1, 1), 6) is None


def test_walk_forward_folds():
    from polycal_ml.train import walk_forward_folds
    # Synthetic 200 days
    df = pd.DataFrame({
        "resolve_date": pd.date_range("2025-01-01", periods=200),
        "yes_won": [0, 1] * 100,
        "city": ["nyc"] * 200,
    })
    folds = list(walk_forward_folds(df, fold_days=30, min_train_days=60))
    assert len(folds) >= 3
    for tr, te, c0, c1 in folds:
        assert tr.sum() > 0 and te.sum() > 0
        # No overlap, train strictly before test
        train_dates = df.loc[tr, "resolve_date"]
        test_dates = df.loc[te, "resolve_date"]
        assert train_dates.max() < test_dates.min()


def test_strategy_pnl_yes_side_logic():
    """Verify the bet-accounting math."""
    import numpy as np
    from polycal_ml.train import strategy_pnl
    # 3 markets:
    #   #0: model says 0.4, market 0.2, yes_won=1 → bet at $0.2, win $1, pnl=+0.8
    #   #1: model says 0.3, market 0.2, yes_won=0 → bet at $0.2, lose, pnl=-0.2
    #   #2: model says 0.1, market 0.5, yes_won=1 → NO BET (model < market)
    p = np.array([0.4, 0.3, 0.1])
    mp = np.array([0.2, 0.2, 0.5])
    yw = np.array([1, 0, 1])
    r = strategy_pnl(p, mp, yw, margin=0.05, side="yes")
    assert r["n"] == 2
    assert abs(r["pnl"] - 0.6) < 1e-9
    assert r["win_rate"] == 0.5


def test_discover_event_buckets_smoke():
    """If canonical list exists, discovery returns non-empty result."""
    if not Path("results/weather_markets_canonical.parquet").exists():
        pytest.skip("canonical list missing — run HF download first")
    from polycal_ml.data_builder import discover_event_buckets
    df = discover_event_buckets(dt.date(2026, 4, 1), dt.date(2026, 4, 5))
    assert len(df) > 0
    assert df["yes_won"].isin([0, 1]).all()
    # Expect ~5 days * 9 cities * ~7 buckets = ~315 rows
    assert 100 < len(df) < 1000
