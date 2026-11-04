"""
Tests for MuniPricer. Run: pytest -q

The slow end-to-end test trains on a small synthetic file; it runs in about
20 seconds and is what CI uses to catch regressions.

Author: Anubhav Awasthi
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import features as F  # noqa: E402
from features import build_features, load_trades, modified_duration, price_from_yield, stale_bucket  # noqa: E402


# ----------------------------------------------------------------------------- bond math
def test_par_bond_prices_at_100():
    assert price_from_yield(4.0, 4.0, 10)[0] == pytest.approx(100.0, abs=1e-9)


def test_price_falls_when_yield_rises():
    p_lo, p_hi = price_from_yield([4.0, 4.0], [3.0, 5.0], [10, 10])
    assert p_lo > 100 > p_hi


def test_duration_is_positive_and_grows_with_maturity():
    d5, d20 = modified_duration([4, 4], [4, 4], [5, 20])
    assert 0 < d5 < d20


def test_stale_bucket_handles_nan_and_keeps_index():
    s = pd.Series([3, 20, 60, 200, np.nan], index=[10, 11, 12, 13, 14])
    b = stale_bucket(s)
    assert list(b) == ["<1 wk", "1-4 wk", "1-3 mo", "3-12 mo", "never traded"]
    assert list(b.index) == [10, 11, 12, 13, 14]


# ----------------------------------------------------------------------------- features
@pytest.fixture(scope="module")
def small_trades(tmp_path_factory):
    """A compact synthetic trade file: 400 bonds, 8,000 trades."""
    d = tmp_path_factory.mktemp("data")
    env = dict(os.environ, MUNI_N_BONDS="400", MUNI_N_TRADES="8000", MUNI_OUT_DIR=str(d))
    subprocess.run([sys.executable, "make_synthetic.py"], cwd=ROOT, env=env, check=True, capture_output=True)
    return str(d / "trades_synthetic.csv")


def test_features_do_not_peek_into_the_future(small_trades):
    df = build_features(load_trades(small_trades))
    # last trade of the same CUSIP must be strictly earlier
    g = df.groupby("cusip")["trade_date"]
    prev = g.shift(1)
    ok = prev.isna() | (prev <= df["trade_date"])
    assert ok.all()
    # the first row of the file has no benchmark history, so the target still exists (fallback)
    assert df[F.TARGET].notna().all()
    # a bond's first appearance has no last-trade features
    first = df.groupby("cusip").head(1)
    assert first["last_spread_same_cusip"].isna().all()
    assert first["days_since_last_trade"].isna().all()


def test_spread_target_is_level_free(small_trades):
    df = build_features(load_trades(small_trades))
    # spread + benchmark reconstructs yield exactly
    assert np.allclose(df[F.TARGET] + df["bench"], df["yield"])
    # the benchmark tracks the market: correlation with daily mean yield should be high
    daily = df.groupby(df["trade_date"].dt.date)["yield"].mean().rolling(7).mean()
    bench_daily = df.groupby(df["trade_date"].dt.date)["bench"].mean()
    assert daily.corr(bench_daily) > 0.8


def test_load_trades_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"cusip": ["X"], "trade_date": ["2026-01-01"], "maturity_date": ["2030-01-01"],
                  "dated_date": ["2020-01-01"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_trades(str(p))


# ----------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def trained(small_trades, tmp_path_factory):
    import train
    out = str(tmp_path_factory.mktemp("models"))
    f, m = train.main(small_trades, out, quiet=True)
    return f, m, out


def test_lgb_beats_baseline_and_range_is_calibrated(trained):
    f, m, out = trained
    assert m["lgb_mae_price"] < m["baseline_mae_price"]
    assert 0.65 <= m["range_coverage"] <= 0.95          # small data, loose bounds
    assert np.isfinite(f.widen_global)
    for k in ["lgb_yield.txt", "lgb_q10.txt", "lgb_q90.txt", "meta.json", "bond_reference.csv", "benchmark.csv"]:
        assert os.path.exists(os.path.join(out, k))
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["target"] == "spread" and "bench_latest" in meta


def test_predict_lookup_and_manual(trained):
    from predict import MuniPricer
    _, _, out = trained
    p = MuniPricer(out)
    row = p.lookup(p.ref["cusip"].iloc[0])
    r = p.price(row.to_dict())
    assert r["price_low"] < r["price"] < r["price_high"]
    assert 50 < r["price"] < 160
    assert r["confidence"] in ("high", "medium", "low")
    assert "benchmark" in p.explain(r, row.to_dict())
    # manual entry, no history, unknown CUSIP
    b = dict(coupon=4.0, maturity_date="2036-10-01", rating="AA", state=p.meta["categories"]["state"][0],
             sector=p.meta["categories"]["sector"][0], issue_size_mm=25, callable=True, trade_type="S")
    r2 = p.price(b)
    assert np.isfinite(r2["price"]) and r2["price_low"] < r2["price"] < r2["price_high"]
    # MSRB: S = dealer sells to you (you pay the markup), P = dealer buys from you. S price must be higher.
    b["trade_type"] = "P"
    assert p.price(b)["price"] < r2["price"]
    # unknown rating falls back to NR instead of crashing
    b["rating"] = "ZZZ"
    assert np.isfinite(p.price(b)["price"])


def test_predict_uses_last_trade_when_given(trained):
    from predict import MuniPricer
    _, _, out = trained
    p = MuniPricer(out)
    b = dict(coupon=4.0, maturity_date="2036-10-01", rating="AA", state=p.meta["categories"]["state"][0],
             sector=p.meta["categories"]["sector"][0], issue_size_mm=25, callable=True, trade_type="D")
    base = p.price(b)["price"]
    cheap = dict(b, last_yield=base_y(p, b) + 1.0, last_trade_date=str((p.as_of - pd.Timedelta(days=30)).date()))
    rich = dict(b, last_yield=base_y(p, b) - 1.0, last_trade_date=str((p.as_of - pd.Timedelta(days=30)).date()))
    assert p.price(cheap)["price"] < base < p.price(rich)["price"]


def base_y(p, b):
    return p.price(b)["yield"]
