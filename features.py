"""
Shared feature engineering for training and inference.

Author: Anubhav Awasthi
"""
import numpy as np
import pandas as pd

RATING_ORDER = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "NR"]
RATING_NUM = {r: i for i, r in enumerate(RATING_ORDER)}
TRADE_TYPES = ["D", "P", "S"]

CAT_COLS = ["state", "sector", "trade_type"]
NUM_COLS = ["coupon", "years_to_maturity", "rating_num", "log_issue_size", "callable", "bank_qualified",
            "insured", "log_par", "last_yield_same_cusip", "days_since_last_trade", "bucket_yield_30d", "market_yield_7d"]
FEATURES = NUM_COLS + CAT_COLS

FRIENDLY = {
    "coupon": "coupon rate",
    "years_to_maturity": "years to maturity",
    "rating_num": "credit rating",
    "log_issue_size": "issue size",
    "callable": "callable",
    "bank_qualified": "bank qualified",
    "insured": "bond insurance",
    "log_par": "trade size",
    "days_index": "market level on trade date",
    "last_yield_same_cusip": "this bond's last trade",
    "days_since_last_trade": "time since last trade",
    "bucket_yield_30d": "recent trades in similar bonds",
    "market_yield_7d": "overall market level this week",
    "state": "state",
    "sector": "sector",
    "trade_type": "trade type (buy/sell/dealer)",
}


def load_trades(path: str) -> pd.DataFrame:
    """Load a trade file. Works for the synthetic file and for a WRDS MSRB export
    once the columns are renamed per README."""
    df = pd.read_csv(path, parse_dates=["trade_date", "maturity_date", "dated_date"])
    for c in ["callable", "bank_qualified", "insured"]:
        if df[c].dtype != bool:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1", "y", "yes"])
    return df


def price_from_yield(coupon, ytm, years):
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


def build_features(df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Add model features. Requires df sorted by trade_date. The two history
    features use only trades strictly before each row (no peeking)."""
    df = df.sort_values("trade_date").reset_index(drop=True).copy()
    t0 = df["trade_date"].min() if as_of is None else as_of
    df["years_to_maturity"] = ((df["maturity_date"] - df["trade_date"]).dt.days / 365.25).clip(0.25, 40)
    df["rating_num"] = df["rating"].map(RATING_NUM).fillna(RATING_NUM["NR"]).astype(int)
    df["log_issue_size"] = np.log1p(df["issue_size_mm"])
    df["log_par"] = np.log(df["par_traded"].clip(lower=1000))
    df["days_index"] = (df["trade_date"] - t0).dt.days
    for c in ["callable", "bank_qualified", "insured"]:
        df[c] = df[c].astype(int)

    # last trade of the same CUSIP, strictly earlier
    g = df.groupby("cusip")
    df["last_yield_same_cusip"] = g["yield"].shift(1)
    df["days_since_last_trade"] = (df["trade_date"] - g["trade_date"].shift(1)).dt.days

    # trailing 30-day mean yield of similar bonds (same rating bucket, same maturity bucket), strictly earlier
    df["_rb"] = df["rating_num"] // 3
    df["_mb"] = pd.cut(df["years_to_maturity"], [0, 3, 7, 12, 20, 40], labels=False)
    df["bucket_yield_30d"] = np.nan
    for _, idx in df.groupby(["_rb", "_mb"]).groups.items():
        sub = df.loc[idx, ["trade_date", "yield"]].set_index("trade_date")["yield"]
        roll = sub.rolling("30D", closed="left").mean()
        df.loc[idx, "bucket_yield_30d"] = roll.values
    df = df.drop(columns=["_rb", "_mb"])

    # trailing 7-day mean yield across the whole market, strictly earlier (catches rate moves fast)
    allq = df.set_index("trade_date")["yield"].rolling("7D", closed="left").mean()
    df["market_yield_7d"] = allq.values

    for c in CAT_COLS:
        df[c] = pd.Categorical(df[c], categories=sorted(df[c].dropna().unique()) if c != "trade_type" else TRADE_TYPES)
    return df
