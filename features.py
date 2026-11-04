"""
Shared feature engineering for training and inference.

Design (v2): the model predicts a bond's SPREAD over a trailing market
benchmark, not its absolute yield. Trees cannot extrapolate, so a model that
learns absolute yield is anchored to the rate level it was trained on and
drifts as rates move. In spread space the benchmark carries the level and the
model only has to learn what is bond-specific. The history features (this
bond's last trade, similar bonds recently) are expressed as spreads too, for
the same reason.

All history features use only trades strictly before each row (no peeking).

Author: Anubhav Awasthi
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RATING_ORDER = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "NR"]
RATING_NUM = {r: i for i, r in enumerate(RATING_ORDER)}
TRADE_TYPES = ["D", "P", "S"]

BENCH_WINDOW = "3D"          # trailing market benchmark window (mean yield of all trades)
TARGET = "spread"            # what the model predicts: yield - bench

CAT_COLS = ["state", "sector", "trade_type"]
NUM_COLS = ["coupon", "years_to_maturity", "rating_num", "log_issue_size", "callable", "bank_qualified",
            "insured", "log_par", "last_spread_same_cusip", "days_since_last_trade", "bucket_spread_30d",
            "market_momentum_7d"]
FEATURES = NUM_COLS + CAT_COLS

# how long since the bond last traded; used for reporting and for the per-bucket range calibration
STALE_BINS = [-1, 7, 30, 90, 365, 1e9]
STALE_LABELS = ["<1 wk", "1-4 wk", "1-3 mo", "3-12 mo", "never traded"]

FRIENDLY = {
    "coupon": "coupon rate",
    "years_to_maturity": "years to maturity",
    "rating_num": "credit rating",
    "log_issue_size": "issue size",
    "callable": "callable",
    "bank_qualified": "bank qualified",
    "insured": "bond insurance",
    "log_par": "trade size",
    "last_spread_same_cusip": "this bond's last trade",
    "days_since_last_trade": "time since last trade",
    "bucket_spread_30d": "recent trades in similar bonds",
    "market_momentum_7d": "market direction this week",
    "state": "state",
    "sector": "sector",
    "trade_type": "trade type (buy/sell/dealer)",
}

REQUIRED_COLS = ["cusip", "trade_date", "trade_type", "par_traded", "price", "yield", "coupon", "maturity_date",
                 "dated_date", "state", "sector", "rating", "issue_size_mm", "callable", "bank_qualified", "insured"]


def stale_bucket(days_since_last_trade) -> pd.Series:
    """Bucket days-since-last-trade; NaN (no prior trade in the data) -> 'never traded'."""
    idx = days_since_last_trade.index if isinstance(days_since_last_trade, pd.Series) else None
    s = pd.Series(np.asarray(days_since_last_trade, dtype=float), index=idx).fillna(9999)
    return pd.cut(s, STALE_BINS, labels=STALE_LABELS).astype(str)


def load_trades(path: str) -> pd.DataFrame:
    """Load a trade file in the MuniPricer column layout (see README). Works for
    the synthetic file and for the output of load_msrb.py."""
    df = pd.read_csv(path, parse_dates=["trade_date", "maturity_date", "dated_date"], dtype={"cusip": str})
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; run load_msrb.py on raw WRDS exports first")
    for c in ["callable", "bank_qualified", "insured"]:
        if df[c].dtype != bool:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1", "y", "yes"])
    return df


# ----------------------------------------------------------------------------- bond math
def price_from_yield(coupon, ytm, years):
    """Clean price per 100 par, semiannual coupons, whole periods."""
    coupon = np.atleast_1d(np.asarray(coupon, dtype=float))
    ytm = np.atleast_1d(np.asarray(ytm, dtype=float))
    years = np.atleast_1d(np.asarray(years, dtype=float))
    n = np.maximum(np.round(years * 2), 1).astype(int)
    out = np.empty(len(ytm))
    for i in range(len(ytm)):
        periods = np.arange(1, n[i] + 1)
        disc = (1 + ytm[i] / 200.0) ** -periods
        out[i] = (coupon[i] / 2.0) * disc.sum() + 100 * disc[-1]
    return out


def modified_duration(coupon, ytm, years):
    """Price sensitivity: fractional price change per 1 percentage point of yield (e.g. 0.14 = 14%)."""
    base = price_from_yield(coupon, ytm, years)
    up = price_from_yield(coupon, np.asarray(ytm) + 0.01, years)
    return (base - up) / base / 0.01


# ----------------------------------------------------------------------------- benchmark
def market_benchmark(df: pd.DataFrame, window: str = BENCH_WINDOW) -> pd.Series:
    """Trailing mean yield of every trade in the market, strictly before each row.
    Early rows with no history fall back to the expanding mean, then the global mean."""
    s = df.set_index("trade_date")["yield"]
    bench = pd.Series(s.rolling(window, closed="left").mean().values, index=df.index)
    bench = bench.fillna(pd.Series(s.expanding().mean().shift(1).values, index=df.index))
    return bench.fillna(df["yield"].mean())


def daily_benchmark(df: pd.DataFrame, window: str = BENCH_WINDOW) -> pd.DataFrame:
    """One benchmark value per calendar day (used by predict.py to look up the
    benchmark on a user-supplied last-trade date)."""
    s = df.set_index("trade_date")["yield"].sort_index()
    daily = s.resample("D").mean()
    days = int(window.rstrip("D"))
    out = daily.rolling(days, min_periods=1).mean().ffill()
    return out.rename("bench").reset_index().rename(columns={"trade_date": "date"})


# ----------------------------------------------------------------------------- features
def build_features(df: pd.DataFrame, bench_window: str = BENCH_WINDOW) -> pd.DataFrame:
    """Add model features and the target. Sorted by trade_date on return."""
    df = df.sort_values(["trade_date", "cusip"], kind="stable").reset_index(drop=True).copy()
    df["years_to_maturity"] = ((df["maturity_date"] - df["trade_date"]).dt.days / 365.25).clip(0.25, 40)
    df["rating_num"] = df["rating"].map(RATING_NUM).fillna(RATING_NUM["NR"]).astype(int)
    df["log_issue_size"] = np.log1p(df["issue_size_mm"].astype(float))
    df["log_par"] = np.log(df["par_traded"].clip(lower=1000))
    for c in ["callable", "bank_qualified", "insured"]:
        df[c] = df[c].astype(int)

    # market benchmark (level) and the target
    df["bench"] = market_benchmark(df, bench_window)
    df[TARGET] = df["yield"] - df["bench"]

    # last trade of the same CUSIP, strictly earlier, as a spread over the benchmark at that time
    g = df.groupby("cusip")
    df["last_spread_same_cusip"] = g[TARGET].shift(1)
    df["last_yield_same_cusip"] = g["yield"].shift(1)           # kept for reporting only
    df["days_since_last_trade"] = (df["trade_date"] - g["trade_date"].shift(1)).dt.days

    # trailing 30-day mean spread of similar bonds (same rating bucket, same maturity bucket), strictly earlier
    df["_rb"] = df["rating_num"] // 3
    df["_mb"] = pd.cut(df["years_to_maturity"], [0, 3, 7, 12, 20, 40], labels=False)
    df["bucket_spread_30d"] = np.nan
    for _, idx in df.groupby(["_rb", "_mb"]).groups.items():
        sub = df.loc[idx, ["trade_date", TARGET]].set_index("trade_date")[TARGET]
        df.loc[idx, "bucket_spread_30d"] = sub.rolling("30D", closed="left").mean().values
    df = df.drop(columns=["_rb", "_mb"])

    # market direction: 7-day trailing mean vs the short benchmark (positive = yields were higher last week = falling)
    m7 = df.set_index("trade_date")["yield"].rolling("7D", closed="left").mean().values
    df["market_momentum_7d"] = m7 - df["bench"]

    for c in CAT_COLS:
        cats = TRADE_TYPES if c == "trade_type" else sorted(df[c].dropna().astype(str).unique())
        df[c] = pd.Categorical(df[c].astype(str), categories=cats)
    return df
