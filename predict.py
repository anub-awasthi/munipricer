"""
Inference: price one bond, with a range and a plain-English explanation.

Author: Anubhav Awasthi
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from features import (FEATURES, CAT_COLS, NUM_COLS, RATING_NUM, TRADE_TYPES, FRIENDLY,
                      price_from_yield, modified_duration)


class MuniPricer:
    def __init__(self, model_dir="models"):
        self.model = lgb.Booster(model_file=f"{model_dir}/lgb_yield.txt")
        self.q10 = lgb.Booster(model_file=f"{model_dir}/lgb_q10.txt")
        self.q90 = lgb.Booster(model_file=f"{model_dir}/lgb_q90.txt")
        self.meta = json.load(open(f"{model_dir}/meta.json"))
        self.ref = pd.read_csv(f"{model_dir}/bond_reference.csv", parse_dates=["maturity_date", "last_trade_date"])
        self.ref["cusip"] = self.ref["cusip"].astype(str)
        self.as_of = pd.Timestamp(self.meta["as_of"])
        self.widen = self.meta.get("range_widen", 0.0)

    # ------------------------------------------------------------------
    def lookup(self, cusip: str):
        row = self.ref[self.ref["cusip"] == cusip.strip().upper()]
        return None if row.empty else row.iloc[0]

    def _frame(self, b: dict) -> pd.DataFrame:
        """Build a one-row feature frame from a bond dict."""
        as_of = pd.Timestamp(b.get("as_of", self.as_of))
        ytm_years = max((pd.Timestamp(b["maturity_date"]) - as_of).days / 365.25, 0.25)
        last_date = b.get("last_trade_date")
        row = {
            "coupon": float(b["coupon"]),
            "years_to_maturity": ytm_years,
            "rating_num": RATING_NUM.get(b.get("rating", "NR"), RATING_NUM["NR"]),
            "log_issue_size": np.log1p(float(b.get("issue_size_mm", 20))),
            "callable": int(bool(b.get("callable", False))),
            "bank_qualified": int(bool(b.get("bank_qualified", False))),
            "insured": int(bool(b.get("insured", False))),
            "log_par": np.log(max(float(b.get("par_traded", 25000)), 1000)),
            "last_yield_same_cusip": b.get("last_yield", np.nan),
            "days_since_last_trade": (as_of - pd.Timestamp(last_date)).days if last_date is not None and not pd.isna(last_date) else np.nan,
            "bucket_yield_30d": b.get("bucket_yield_30d", np.nan),
            "market_yield_7d": b.get("market_yield_7d", self.meta.get("market_yield_latest", np.nan)),
            "state": b.get("state", "CA"),
            "sector": b.get("sector", "General Obligation"),
            "trade_type": b.get("trade_type", "P"),
        }
        X = pd.DataFrame([row])
        for c in CAT_COLS:
            X[c] = pd.Categorical(X[c], categories=self.meta["categories"][c])
        return X[FEATURES]

    # ------------------------------------------------------------------
    def price(self, b: dict) -> dict:
        X = self._frame(b)
        y = float(self.model.predict(X)[0])
        lo_y = float(self.q10.predict(X)[0]) - self.widen
        hi_y = float(self.q90.predict(X)[0]) + self.widen
        coupon, yrs = X["coupon"].iloc[0], X["years_to_maturity"].iloc[0]
        px = float(price_from_yield(coupon, y, yrs)[0])
        px_hi = float(price_from_yield(coupon, lo_y, yrs)[0])   # lower yield -> higher price
        px_lo = float(price_from_yield(coupon, hi_y, yrs)[0])
        dur = float(modified_duration(coupon, y, yrs)[0])

        # SHAP-style contributions in yield, converted to price via duration
        contrib = self.model.predict(X, pred_contrib=True)[0]
        base_y = contrib[-1]
        parts = []
        for name, c in zip(FEATURES, contrib[:-1]):
            if abs(c) < 1e-4:
                continue
            parts.append({"feature": name, "label": FRIENDLY.get(name, name),
                          "yield_pts": float(c), "price_pts": float(-c * dur * px),
                          "value": X[name].iloc[0]})
        parts.sort(key=lambda p: -abs(p["price_pts"]))
        width = px_hi - px_lo
        confidence = "high" if width < 2.0 else "medium" if width < 4.0 else "low"
        return {"price": px, "price_low": px_lo, "price_high": px_hi, "yield": y, "yield_low": lo_y, "yield_high": hi_y,
                "duration": dur, "base_yield": float(base_y), "contributions": parts, "confidence": confidence,
                "years_to_maturity": float(yrs)}

    # ------------------------------------------------------------------
    @staticmethod
    def phrase(p: dict, b: dict) -> str:
        """Short description of one contribution, e.g. 'it has 22 years left to maturity'."""
        v = p["value"]; lab = p["label"]; f = p["feature"]
        if f == "last_yield_same_cusip":
            return f"this bond last traded at a {v:.2f}% yield" if not pd.isna(v) else "this bond has no recent trade on record"
        if f == "bucket_yield_30d":
            return f"similar bonds traded near {v:.2f}% over the last 30 days" if not pd.isna(v) else "there were no recent trades in similar bonds"
        if f == "market_yield_7d":
            return f"the overall muni market is yielding about {v:.2f}% this week"
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
            return {"P": "you are buying from a dealer (dealers mark up)",
                    "S": "you are selling to a dealer (dealers mark down)",
                    "D": "this is a dealer-to-dealer trade"}.get(str(v), "trade type")
        if f in ("callable", "insured", "bank_qualified"):
            return f"the bond is {'' if int(v) else 'not '}{lab}"
        if f == "days_since_last_trade":
            return f"it last traded {int(v)} days ago" if not pd.isna(v) else "it has not traded in our data"
        return f"{lab} is {v}"

    @staticmethod
    def explain(res: dict, b: dict) -> str:
        """Plain-English paragraph from the top contributions."""
        lines = [f"Our estimate is {res['price']:.2f} per 100 of face value, with an 80% range of "
                 f"{res['price_low']:.2f} to {res['price_high']:.2f}. That works out to a yield of about {res['yield']:.2f}%."]
        if res["confidence"] == "low":
            lines.append("The range is wide, so treat this as a rough guide rather than a quote.")
        for p in res["contributions"][:5]:
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
    b["last_trade_date"] = row["last_trade_date"]
    r = p.price(b)
    print(f"{cusip}: {b['state']} {b['sector']} {b['rating']} {b['coupon']}% {row['maturity_date'].date()}")
    print(f"last traded {row['last_trade_date'].date()} at {row['last_price']:.2f}")
    print(f"predicted {r['price']:.2f}  range {r['price_low']:.2f}-{r['price_high']:.2f}  yield {r['yield']:.2f}%  confidence {r['confidence']}")
    print(p.explain(r, b))
