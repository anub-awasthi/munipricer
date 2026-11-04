"""
Inference: price one bond, with a range and a plain-English explanation.

The model predicts the bond's spread over the trailing market benchmark;
this module adds the benchmark back, converts yield to price, and turns the
SHAP contributions into sentences.

  python predict.py <CUSIP>

Author: Anubhav Awasthi
"""
from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import (CAT_COLS, FEATURES, FRIENDLY, RATING_NUM, RATING_ORDER, modified_duration,
                      price_from_yield, stale_bucket)

MAT_EDGES = [0, 3, 7, 12, 20, 40]


class MuniPricer:
    def __init__(self, model_dir: str = "models"):
        self.model = lgb.Booster(model_file=f"{model_dir}/lgb_yield.txt")
        self.q10 = lgb.Booster(model_file=f"{model_dir}/lgb_q10.txt")
        self.q90 = lgb.Booster(model_file=f"{model_dir}/lgb_q90.txt")
        self.meta = json.load(open(f"{model_dir}/meta.json"))
        self.ref = pd.read_csv(f"{model_dir}/bond_reference.csv", parse_dates=["maturity_date", "last_trade_date"],
                               dtype={"cusip": str})
        self.ref["cusip"] = self.ref["cusip"].str.upper()
        bpath = f"{model_dir}/benchmark.csv"
        self.bench_hist = (pd.read_csv(bpath, parse_dates=["date"]).set_index("date")["bench"]
                           if os.path.exists(bpath) else pd.Series(dtype=float))
        self.as_of = pd.Timestamp(self.meta["as_of"])
        self.bench_latest = float(self.meta.get("bench_latest", np.nan))
        self.widen = float(self.meta.get("range_widen", 0.0))
        self.widen_by_stale = self.meta.get("range_widen_by_staleness", {})
        self.bucket_latest = self.meta.get("bucket_spread_latest", {})
        self.synthetic = bool(self.meta.get("synthetic", True))

    # ------------------------------------------------------------------ helpers
    def lookup(self, cusip: str):
        row = self.ref[self.ref["cusip"] == cusip.strip().upper()]
        return None if row.empty else row.iloc[0]

    def benchmark_on(self, date) -> float:
        """Market benchmark on a given date (nearest earlier day in the saved history)."""
        if self.bench_hist.empty:
            return self.bench_latest
        d = pd.Timestamp(date)
        hist = self.bench_hist[self.bench_hist.index <= d]
        return float(hist.iloc[-1]) if len(hist) else float(self.bench_hist.iloc[0])

    def bucket_spread(self, rating: str, years: float) -> float:
        rb = RATING_NUM.get(rating, RATING_NUM["NR"]) // 3
        mb = int(np.searchsorted(MAT_EDGES, years, side="right") - 1)
        mb = min(max(mb, 0), len(MAT_EDGES) - 2)
        return float(self.bucket_latest.get(f"{rb}-{mb}", np.nan))

    def _widen(self, days_since_last_trade) -> float:
        if not self.widen_by_stale:
            return self.widen
        return float(self.widen_by_stale.get(stale_bucket([days_since_last_trade]).iloc[0], self.widen))

    def _frame(self, b: dict) -> tuple[pd.DataFrame, float]:
        """Build a one-row feature frame from a bond dict. Returns (X, benchmark)."""
        as_of = pd.Timestamp(b.get("as_of", self.as_of))
        bench = float(b.get("bench", self.bench_latest))
        years = max((pd.Timestamp(b["maturity_date"]) - as_of).days / 365.25, 0.25)
        rating = b.get("rating", "NR") if b.get("rating", "NR") in RATING_ORDER else "NR"

        last_date = b.get("last_trade_date")
        has_last = last_date is not None and not pd.isna(last_date)
        if "last_spread" in b and not pd.isna(b["last_spread"]):
            last_spread = float(b["last_spread"])
        elif has_last and "last_yield" in b and not pd.isna(b["last_yield"]):
            last_spread = float(b["last_yield"]) - self.benchmark_on(last_date)
        else:
            last_spread = np.nan
        bucket = b.get("bucket_spread_30d", np.nan)
        if pd.isna(bucket):
            bucket = self.bucket_spread(rating, years)

        row = {
            "coupon": float(b["coupon"]),
            "years_to_maturity": years,
            "rating_num": RATING_NUM[rating],
            "log_issue_size": np.log1p(float(b.get("issue_size_mm", 20))),
            "callable": int(bool(b.get("callable", False))),
            "bank_qualified": int(bool(b.get("bank_qualified", False))),
            "insured": int(bool(b.get("insured", False))),
            "log_par": np.log(max(float(b.get("par_traded", 25000)), 1000)),
            "last_spread_same_cusip": last_spread,
            "days_since_last_trade": (as_of - pd.Timestamp(last_date)).days if has_last else np.nan,
            "bucket_spread_30d": bucket,
            "market_momentum_7d": float(b.get("market_momentum_7d", self.meta.get("market_momentum_latest", 0.0))),
            "state": b.get("state", "CA"),
            "sector": b.get("sector", "General Obligation"),
            "trade_type": b.get("trade_type", "S"),
        }
        X = pd.DataFrame([row])
        for c in CAT_COLS:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.meta["categories"][c])
        return X[FEATURES], bench

    # ------------------------------------------------------------------ price
    def price(self, b: dict) -> dict:
        X, bench = self._frame(b)
        spread = float(self.model.predict(X)[0])
        w = self._widen(X["days_since_last_trade"].iloc[0])
        y = bench + spread
        lo_y = bench + float(self.q10.predict(X)[0]) - w
        hi_y = bench + float(self.q90.predict(X)[0]) + w
        coupon, yrs = float(X["coupon"].iloc[0]), float(X["years_to_maturity"].iloc[0])
        px = float(price_from_yield(coupon, y, yrs)[0])
        px_hi = float(price_from_yield(coupon, lo_y, yrs)[0])   # lower yield -> higher price
        px_lo = float(price_from_yield(coupon, hi_y, yrs)[0])
        dur = float(modified_duration(coupon, y, yrs)[0])

        # SHAP contributions in spread (yield) space, converted to price via duration
        contrib = self.model.predict(X, pred_contrib=True)[0]
        base_spread = float(contrib[-1])
        parts = []
        for name, c in zip(FEATURES, contrib[:-1]):
            if abs(c) < 1e-4:
                continue
            parts.append({"feature": name, "label": FRIENDLY.get(name, name), "yield_pts": float(c),
                          "price_pts": float(-c * dur * px), "value": X[name].iloc[0]})
        parts.sort(key=lambda p: -abs(p["price_pts"]))
        # confidence from the yield range (bps), so long bonds are not penalised for having more duration
        width_bps = (hi_y - lo_y) * 100
        confidence = "high" if width_bps < 25 else "medium" if width_bps < 45 else "low"
        return {"price": px, "price_low": px_lo, "price_high": px_hi, "yield": y, "yield_low": lo_y,
                "yield_high": hi_y, "benchmark": bench, "spread": spread, "base_spread": base_spread,
                "duration": dur, "contributions": parts, "confidence": confidence, "years_to_maturity": yrs,
                "days_since_last_trade": X["days_since_last_trade"].iloc[0], "range_margin_yield": w}

    # ------------------------------------------------------------------ explain
    @staticmethod
    def phrase(p: dict, b: dict) -> str:
        """Short description of one contribution, e.g. 'it has 22 years left to maturity'."""
        v = p["value"]; lab = p["label"]; f = p["feature"]
        if f == "last_spread_same_cusip":
            if pd.isna(v):
                return "this bond has no earlier trade on record"
            return (f"this bond last traded {abs(v) * 100:.0f} bps {'above' if v > 0 else 'below'} the market at the time")
        if f == "bucket_spread_30d":
            if pd.isna(v):
                return "there were no recent trades in similar bonds"
            return f"similar bonds traded {abs(v) * 100:.0f} bps {'above' if v > 0 else 'below'} the market in the last 30 days"
        if f == "market_momentum_7d":
            return f"yields have been {'falling' if v > 0 else 'rising'} this week"
        if f == "years_to_maturity":
            return f"it has {v:.1f} years left to maturity"
        if f == "coupon":
            return f"the coupon is {v:.2f}%"
        if f == "rating_num":
            return f"the credit rating is {b.get('rating', 'not rated')}"
        if f == "log_issue_size":
            return f"the issue size is about ${float(b.get('issue_size_mm', 20)):.0f}M"
        if f == "log_par":
            return f"the trade size is ${float(b.get('par_traded', 25000)):,.0f}"
        if f == "trade_type":
            # MSRB convention: S = dealer sale to customer (you buy), P = dealer purchase from customer (you sell)
            return {"S": "you are buying from a dealer (dealers mark up)",
                    "P": "you are selling to a dealer (dealers mark down)",
                    "D": "this is a dealer-to-dealer trade"}.get(str(v), "trade type")
        if f in ("callable", "insured", "bank_qualified"):
            return f"the bond is {'' if int(v) else 'not '}{lab}"
        if f == "days_since_last_trade":
            return f"it last traded {int(v)} days ago" if not pd.isna(v) else "it has not traded in our data"
        if f in ("state", "sector"):
            return f"the {lab} is {v}"
        return f"{lab} is {v}"

    @staticmethod
    def explain(res: dict, b: dict, top: int = 5) -> str:
        """Plain-English paragraph from the benchmark and the top contributions."""
        lines = [f"Our estimate is {res['price']:.2f} per 100 of face value, with an 80% range of "
                 f"{res['price_low']:.2f} to {res['price_high']:.2f}. That works out to a yield of about "
                 f"{res['yield']:.2f}%, which is {abs(res['spread']) * 100:.0f} bps "
                 f"{'above' if res['spread'] >= 0 else 'below'} the current market benchmark of {res['benchmark']:.2f}%."]
        if res["confidence"] == "low":
            lines.append("The range is wide, so treat this as a rough guide rather than a quote.")
        for p in res["contributions"][:top]:
            what = MuniPricer.phrase(p, b)
            d = "up" if p["price_pts"] > 0 else "down"
            lines.append(f"{what[0].upper() + what[1:]}, which pushes the price {d} by about {abs(p['price_pts']):.2f}.")
        return " ".join(lines)


if __name__ == "__main__":
    import sys
    p = MuniPricer()
    cusip = sys.argv[1] if len(sys.argv) > 1 else p.ref["cusip"].iloc[0]
    row = p.lookup(cusip)
    if row is None:
        print("CUSIP not found in reference table"); sys.exit(1)
    b = row.to_dict()
    r = p.price(b)
    print(f"{cusip}: {b['state']} {b['sector']} {b['rating']} {b['coupon']}% {row['maturity_date'].date()}")
    print(f"last traded {row['last_trade_date'].date()} at {row['last_price']:.2f} ({row['last_yield']:.2f}%)")
    print(f"predicted {r['price']:.2f}  range {r['price_low']:.2f}-{r['price_high']:.2f}  yield {r['yield']:.2f}%  confidence {r['confidence']}")
    print(p.explain(r, b))
