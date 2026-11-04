"""
Precompute predictions for every reference bond so the web demo needs no model runtime.
Author: Anubhav Awasthi
"""
import json
import numpy as np
import pandas as pd
from predict import MuniPricer
from features import FRIENDLY

p = MuniPricer("models")
out = []
for _, row in p.ref.iterrows():
    b = row.to_dict()
    rec = {"c": b["cusip"], "st": b["state"], "sec": b["sector"], "r": b["rating"], "cp": float(b["coupon"]),
           "mat": str(pd.Timestamp(b["maturity_date"]).date()), "sz": float(b["issue_size_mm"]),
           "call": bool(b["callable"]), "ins": bool(b["insured"]), "bq": bool(b["bank_qualified"]),
           "ld": str(pd.Timestamp(b["last_trade_date"]).date()), "lp": float(b["last_price"]), "ly": float(b["last_yield"]),
           "s": {}}
    for tt in ("P", "S", "D"):
        b["trade_type"] = tt
        b["par_traded"] = 25000
        r = p.price(b)
        rec["s"][tt] = {"px": round(r["price"], 2), "lo": round(r["price_low"], 2), "hi": round(r["price_high"], 2),
                        "y": round(r["yield"], 2), "ylo": round(r["yield_low"], 2), "yhi": round(r["yield_high"], 2),
                        "sp": round(r["spread"] * 100), "conf": r["confidence"], "t": p.explain(r, b),
                        "k": [[c["label"], round(c["price_pts"], 2)] for c in r["contributions"][:6]]}
    out.append(rec)

demo = {"as_of": p.meta["as_of"], "bench": round(p.bench_latest, 2), "synthetic": p.synthetic, "metrics": p.meta["metrics"], "importance": {FRIENDLY.get(k, k): v for k, v in p.meta["importance"].items()}, "bonds": out}
json.dump(demo, open("demo_data.json", "w"), separators=(",", ":"))
import os
print(len(out), "bonds,", os.path.getsize("demo_data.json") / 1e6, "MB")
