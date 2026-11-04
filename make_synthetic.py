"""
Generate a realistic synthetic municipal bond trade dataset.

Column layout mirrors the WRDS MSRB transaction file so that train.py can
load either this file or a real WRDS export without changes.

Author: Anubhav Awasthi
"""
import os
import numpy as np
import pandas as pd
from datetime import date, timedelta

rng = np.random.default_rng(42)

# overridable for tests and CI: MUNI_N_BONDS=400 MUNI_N_TRADES=8000 MUNI_OUT_DIR=/tmp/x python make_synthetic.py
N_BONDS = int(os.environ.get("MUNI_N_BONDS", 3000))
N_TRADES = int(os.environ.get("MUNI_N_TRADES", 60000))
OUT_DIR = os.environ.get("MUNI_OUT_DIR", "data")
AS_OF = date(2026, 10, 1)
os.makedirs(OUT_DIR, exist_ok=True)

STATES = ["CA", "NY", "TX", "IL", "FL", "PA", "OH", "NJ", "MA", "WA", "MI", "GA", "NC", "VA", "AZ"]
STATE_W = np.array([18, 14, 11, 8, 7, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3], dtype=float)
STATE_W /= STATE_W.sum()

SECTORS = ["General Obligation", "Water & Sewer", "School District", "Transportation", "Healthcare", "Housing", "Higher Ed", "Electric Utility"]
SECTOR_W = np.array([30, 14, 16, 12, 8, 6, 8, 6], dtype=float)
SECTOR_W /= SECTOR_W.sum()

# S&P style rating buckets and the credit spread (bps) each adds over AAA
RATINGS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "NR"]
RATING_W = np.array([10, 14, 20, 14, 10, 9, 6, 5, 4, 3, 5], dtype=float)
RATING_W /= RATING_W.sum()
RATING_SPREAD = {"AAA": 0, "AA+": 8, "AA": 15, "AA-": 25, "A+": 38, "A": 52, "A-": 70,
                 "BBB+": 95, "BBB": 125, "BBB-": 165, "NR": 140}

# A plausible AAA tax-exempt muni yield curve (percent) by years to maturity
CURVE_PTS = np.array([[0.5, 2.70], [1, 2.75], [2, 2.72], [3, 2.70], [5, 2.75], [7, 2.90],
                      [10, 3.15], [15, 3.55], [20, 3.85], [25, 4.05], [30, 4.15]])


def aaa_yield(years):
    return np.interp(years, CURVE_PTS[:, 0], CURVE_PTS[:, 1])


def price_from_yield(coupon, ytm, years):
    """Clean price per 100 par, semiannual coupons."""
    n = np.maximum(np.round(years * 2), 1).astype(int)
    c = coupon / 2.0
    r = ytm / 200.0
    out = np.empty_like(ytm, dtype=float)
    for i in range(len(ytm)):
        periods = np.arange(1, n[i] + 1)
        disc = (1 + r[i]) ** -periods
        out[i] = c[i] * disc.sum() + 100 * disc[-1]
    return out


def make_cusip(i):
    # 9-char fake CUSIPs; first 6 identify the issuer
    issuer = f"{rng.integers(100000, 999999)}"
    tail = f"{rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'))}{rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'))}{i % 10}"
    return issuer + tail


# ---------------- bonds ----------------
bonds = pd.DataFrame({
    "cusip": [make_cusip(i) for i in range(N_BONDS)],
    "state": rng.choice(STATES, N_BONDS, p=STATE_W),
    "sector": rng.choice(SECTORS, N_BONDS, p=SECTOR_W),
    "rating": rng.choice(RATINGS, N_BONDS, p=RATING_W),
})
bonds["coupon"] = rng.choice([2.0, 2.5, 3.0, 3.25, 3.5, 4.0, 4.25, 4.5, 5.0, 5.25], N_BONDS,
                             p=[0.04, 0.06, 0.12, 0.06, 0.08, 0.22, 0.06, 0.08, 0.24, 0.04])
years_to_mat = np.clip(rng.gamma(2.2, 5.0, N_BONDS), 0.5, 30)
bonds["maturity_date"] = [AS_OF + timedelta(days=int(y * 365.25)) for y in years_to_mat]
issue_age = np.clip(rng.gamma(2.0, 2.5, N_BONDS), 0.1, 20)
bonds["dated_date"] = [AS_OF - timedelta(days=int(a * 365.25)) for a in issue_age]
bonds["issue_size_mm"] = np.round(np.exp(rng.normal(3.0, 1.1, N_BONDS)), 1)  # $MM, median ~20
bonds["callable"] = rng.random(N_BONDS) < 0.55
bonds["bank_qualified"] = rng.random(N_BONDS) < 0.25
bonds["insured"] = rng.random(N_BONDS) < 0.12
# per-bond idiosyncratic spread the model cannot see from reference fields. In the real market two bonds with
# the same rating, sector and state trade tens of bps apart, and the only way to know is the bond's own trade
# history. 35 bps keeps the history features honest; v1 used 12, which made reference fields almost sufficient.
bonds["_issuer_eps"] = rng.normal(0, 35, N_BONDS)
# liquidity: bigger, higher rated, GO bonds trade more
lam = 0.6 * np.log1p(bonds["issue_size_mm"]) + np.where(bonds["sector"] == "General Obligation", 0.4, 0) \
    + bonds["rating"].map(lambda r: 0.5 if r.startswith("AA") else 0).values
# heavy-tailed activity, as in the real market: a few bonds trade daily, most trade a handful of times a year.
# The lognormal factor gives roughly 40% of bonds fewer than 5 trades a year, so the stale buckets are populated.
bonds["_trade_rate"] = np.exp(lam + rng.normal(0, 1.3, N_BONDS))
bonds["_trade_rate"] /= bonds["_trade_rate"].mean()

# ---------------- trades ----------------
bond_idx = rng.choice(N_BONDS, N_TRADES, p=bonds["_trade_rate"] / bonds["_trade_rate"].sum())
t = bonds.iloc[bond_idx].reset_index(drop=True)

days_back = rng.integers(0, 365, N_TRADES)
t["trade_date"] = [AS_OF - timedelta(days=int(d)) for d in days_back]
t["years_to_maturity"] = [(m - d).days / 365.25 for m, d in zip(t["maturity_date"], t["trade_date"])]
t["years_to_maturity"] = t["years_to_maturity"].clip(0.25, 30)

# market level drifts over the year: a slow sine plus a small random walk
market_shift = 0.25 * np.sin(days_back / 365 * 2 * np.pi) + rng.normal(0, 0.04, N_TRADES)

# trade type and size, MSRB convention: D = inter-dealer, P = dealer PURCHASE from a customer (the customer
# sells, gets a lower price = higher yield), S = dealer SALE to a customer (the customer buys, pays a markup).
# Small trades carry a bigger markup/markdown.
t["trade_type"] = rng.choice(["D", "P", "S"], N_TRADES, p=[0.25, 0.35, 0.40])
par = np.exp(rng.normal(np.log(25000), 1.0, N_TRADES))
t["par_traded"] = np.round(par / 5000) * 5000
t["par_traded"] = t["par_traded"].clip(5000, 5_000_000)

size_bps = np.where(t["par_traded"] < 25000, 18, np.where(t["par_traded"] < 100000, 10, np.where(t["par_traded"] < 1_000_000, 4, 0)))
type_bps = np.where(t["trade_type"] == "P", +size_bps, np.where(t["trade_type"] == "S", -size_bps, 0))

ytm = (aaa_yield(t["years_to_maturity"].values)
       + market_shift
       + t["rating"].map(RATING_SPREAD).values / 100
       + np.where(t["callable"], 0.12, 0)
       + np.where(t["insured"], -0.10, 0)
       + np.where(t["bank_qualified"], -0.08, 0)
       + np.where(t["sector"].isin(["Healthcare", "Housing"]), 0.20, 0)
       + np.where(t["sector"] == "General Obligation", -0.05, 0)
       + np.where(t["state"].isin(["IL", "NJ"]), 0.15, 0)
       + np.where(t["state"].isin(["CA", "NY"]), -0.05, 0)
       - 0.02 * np.log1p(t["issue_size_mm"].values)
       + t["_issuer_eps"].values / 100
       + type_bps / 100
       + rng.normal(0, 0.06, N_TRADES))
t["yield"] = np.round(ytm, 3)
t["price"] = np.round(price_from_yield(t["coupon"].values, t["yield"].values, t["years_to_maturity"].values), 3)

cols = ["cusip", "trade_date", "trade_type", "par_traded", "price", "yield", "coupon", "maturity_date", "dated_date",
        "state", "sector", "rating", "issue_size_mm", "callable", "bank_qualified", "insured"]
t = t[cols].sort_values("trade_date").reset_index(drop=True)
t.to_csv(os.path.join(OUT_DIR, "trades_synthetic.csv"), index=False)
bonds[["cusip", "state", "sector", "rating", "coupon", "maturity_date", "dated_date", "issue_size_mm", "callable", "bank_qualified", "insured"]].to_csv(os.path.join(OUT_DIR, "bonds_synthetic.csv"), index=False)
print(t.head())
print(len(t), "trades across", t.cusip.nunique(), "bonds")
print(t[["price", "yield", "years_to_maturity" if "years_to_maturity" in t else "coupon"]].describe().round(2))
