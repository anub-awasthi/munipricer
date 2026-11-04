"""
Train the muni bond pricing model.

  python train.py                      # uses data/trades_synthetic.csv
  python train.py data/wrds_msrb.csv   # real data, same columns
  python train.py data/x.csv --out models_real   # save to another folder

Steps: load -> features -> time-based split -> linear baseline -> LightGBM
(point + 10th/90th percentile) -> conformal calibration -> evaluate -> save.

Everything is a function so evaluate.py (rolling backtest, ablations) and
the tests can reuse the same code path the CLI runs.

Author: Anubhav Awasthi
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from features import (CAT_COLS, FEATURES, FRIENDLY, NUM_COLS, RATING_NUM, STALE_LABELS, TARGET, build_features, daily_benchmark,
                      load_trades, price_from_yield, stale_bucket)

LGB_PARAMS = dict(objective="huber", alpha=0.5, learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
                  feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
QUANTILE_OVERRIDES = dict(num_leaves=31, min_data_in_leaf=100)
TARGET_COVERAGE = 0.80


# ----------------------------------------------------------------------------- split
@dataclass
class Split:
    fit: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    cut_val: pd.Timestamp
    cut_test: pd.Timestamp


def split_by_time(df: pd.DataFrame, test_days: int = 60, val_days: int = 45,
                  end: pd.Timestamp | None = None) -> Split:
    """fit: up to (end - test_days - val_days); val: next val_days; test: last test_days.
    `end` lets the rolling backtest slide the window. Never shuffles."""
    end = df["trade_date"].max() if end is None else pd.Timestamp(end)
    cut_test = end - pd.Timedelta(days=test_days)
    cut_val = cut_test - pd.Timedelta(days=val_days)
    fit = df[df["trade_date"] <= cut_val]
    val = df[(df["trade_date"] > cut_val) & (df["trade_date"] <= cut_test)]
    test = df[(df["trade_date"] > cut_test) & (df["trade_date"] <= end)]
    return Split(fit, val, test, cut_val, cut_test)


def to_price(frame: pd.DataFrame, yld) -> np.ndarray:
    return price_from_yield(frame["coupon"].values, yld, frame["years_to_maturity"].values)


# ----------------------------------------------------------------------------- models
@dataclass
class Fitted:
    model: lgb.Booster
    q_models: dict
    ridge: Ridge
    medians: pd.Series
    features: list
    widen: dict = field(default_factory=dict)          # staleness bucket -> yield pts
    widen_global: float = 0.0


def fit_baseline(fit: pd.DataFrame, num_cols=NUM_COLS):
    med = fit[num_cols].median()
    ridge = Ridge(alpha=1.0).fit(fit[num_cols].fillna(med), fit[TARGET])
    return ridge, med


def fit_lgb(fit: pd.DataFrame, val: pd.DataFrame, features=FEATURES, params=None, rounds=6000, patience=100):
    params = dict(LGB_PARAMS if params is None else params)
    cats = [c for c in CAT_COLS if c in features]
    dfit = lgb.Dataset(fit[features], fit[TARGET], categorical_feature=cats, free_raw_data=False)
    dval = lgb.Dataset(val[features], val[TARGET], reference=dfit)
    return lgb.train(params, dfit, num_boost_round=rounds, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(patience, verbose=False)])


def fit_quantiles(fit, val, features=FEATURES, quantiles=(0.1, 0.9)):
    out = {}
    for q in quantiles:
        qp = dict(LGB_PARAMS, objective="quantile", alpha=q, **QUANTILE_OVERRIDES)
        out[q] = fit_lgb(fit, val, features, params=qp)
    return out


def calibrate_range(q_models, val: pd.DataFrame, features=FEATURES, target=TARGET_COVERAGE,
                    by_staleness=True, min_group=300) -> tuple[dict, float]:
    """Conformal widening so the 10-90 band covers `target` of calibration trades.

    Returns (per-bucket widen, global widen). Per-bucket widening is the fix for
    the v1 coverage gap: stale bonds have fatter errors, so one global margin
    under-covers them and over-covers liquid ones. Buckets with fewer than
    `min_group` trades fall back to the global margin."""
    lo = q_models[0.1].predict(val[features], num_iteration=q_models[0.1].best_iteration)
    hi = q_models[0.9].predict(val[features], num_iteration=q_models[0.9].best_iteration)
    score = np.maximum(lo - val[TARGET].values, val[TARGET].values - hi)
    n = len(score)
    # finite-sample correction: ceil((n+1)*target)/n keeps the guarantee honest
    level = min(np.ceil((n + 1) * target) / n, 1.0)
    widen_global = float(np.quantile(score, level))
    widen = {}
    if by_staleness:
        bucket = stale_bucket(val["days_since_last_trade"])
        for lab in STALE_LABELS:
            m = (bucket == lab).values
            if m.sum() >= min_group:
                lvl = min(np.ceil((m.sum() + 1) * target) / m.sum(), 1.0)
                widen[lab] = float(np.quantile(score[m], lvl))
            else:
                widen[lab] = widen_global
    return widen, widen_global


def widen_for(fitted: Fitted, days_since_last_trade) -> np.ndarray:
    """Vectorised lookup of the per-row conformal margin."""
    if not fitted.widen:
        return np.full(len(days_since_last_trade), fitted.widen_global)
    b = stale_bucket(pd.Series(days_since_last_trade))
    return b.map(fitted.widen).fillna(fitted.widen_global).astype(float).values


def fit_all(split: Split, features=FEATURES, by_staleness=True) -> Fitted:
    ridge, med = fit_baseline(split.fit, [c for c in NUM_COLS if c in features])
    model = fit_lgb(split.fit, split.val, features)
    qm = fit_quantiles(split.fit, split.val, features)
    widen, wg = calibrate_range(qm, split.val, features, by_staleness=by_staleness)
    return Fitted(model, qm, ridge, med, list(features), widen, wg)


# ----------------------------------------------------------------------------- evaluate
def predict_all(f: Fitted, frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[f.features]
    b = frame["bench"].values                     # model works in spread space; add the level back
    y = f.model.predict(X, num_iteration=f.model.best_iteration) + b
    w = widen_for(f, frame["days_since_last_trade"])
    lo = f.q_models[0.1].predict(X, num_iteration=f.q_models[0.1].best_iteration) - w + b
    hi = f.q_models[0.9].predict(X, num_iteration=f.q_models[0.9].best_iteration) + w + b
    num = [c for c in NUM_COLS if c in f.features]
    base = f.ridge.predict(X[num].fillna(f.medians)) + b
    return pd.DataFrame({"yield_pred": y, "yield_lo": lo, "yield_hi": hi, "yield_base": base,
                         "price_pred": to_price(frame, y), "price_base": to_price(frame, base),
                         # lower yield -> higher price
                         "price_hi": to_price(frame, lo), "price_lo": to_price(frame, hi)}, index=frame.index)


def evaluate(f: Fitted, test: pd.DataFrame) -> dict:
    p = predict_all(f, test)
    y, px = test["yield"].values, test["price"].values
    inside = (y >= p["yield_lo"]) & (y <= p["yield_hi"])
    abs_err = np.abs(px - p["price_pred"])
    bucket = stale_bucket(test["days_since_last_trade"])
    by_stale = pd.DataFrame({"abs_err": abs_err, "inside": inside, "width": p["price_hi"] - p["price_lo"]}) \
        .groupby(bucket, observed=True).agg(mae=("abs_err", "mean"), coverage=("inside", "mean"),
                                            width=("width", "mean"), n=("abs_err", "size"))
    imp = pd.Series(f.model.feature_importance("gain"), index=f.features)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    return {
        "baseline_mae_price": float(mean_absolute_error(px, p["price_base"])),
        "lgb_mae_price": float(abs_err.mean()),
        "baseline_mae_yield": float(mean_absolute_error(y, p["yield_base"])),
        "lgb_mae_yield": float(mean_absolute_error(y, p["yield_pred"])),
        "range_coverage": float(inside.mean()),
        "range_width_price": float((p["price_hi"] - p["price_lo"]).mean()),
        "n_test": int(len(test)),
        "best_iter": int(f.model.best_iteration),
        "by_staleness": by_stale,
        "importance": imp,
        "pred": p,
    }


# ----------------------------------------------------------------------------- save
def bucket_table(frame: pd.DataFrame) -> dict:
    """Mean spread by (rating bucket, maturity bucket), keyed 'rb-mb'."""
    rb = frame["rating_num"] // 3
    mb = pd.cut(frame["years_to_maturity"], [0, 3, 7, 12, 20, 40], labels=False)
    t = frame.groupby([rb, mb], observed=True)[TARGET].mean()
    return {f"{a}-{b}": float(v) for (a, b), v in t.items()}


def save(f: Fitted, df: pd.DataFrame, split: Split, metrics: dict, out_dir="models", data_path="") -> None:
    os.makedirs(out_dir, exist_ok=True)
    f.model.save_model(f"{out_dir}/lgb_yield.txt", num_iteration=f.model.best_iteration)
    for q, m in f.q_models.items():
        m.save_model(f"{out_dir}/lgb_q{int(q * 100)}.txt", num_iteration=m.best_iteration)
    cats = {c: list(df[c].cat.categories) for c in CAT_COLS}
    meta = {
        "features": f.features, "cat_cols": CAT_COLS, "categories": cats,
        "num_medians": {k: float(v) for k, v in f.medians.items()},
        "range_widen": f.widen_global, "range_widen_by_staleness": f.widen,
        "target": TARGET, "t0": str(df["trade_date"].min().date()),
        "bench_latest": float(df["bench"].iloc[-1]),
        "market_momentum_latest": float(df["market_momentum_7d"].iloc[-1]),
        "as_of": str(df["trade_date"].max().date()),
        "data_path": data_path, "synthetic": "synthetic" in os.path.basename(data_path),
        "metrics": {
            "baseline_mae_price": round(metrics["baseline_mae_price"], 3),
            "lgb_mae_price": round(metrics["lgb_mae_price"], 3),
            "range_coverage": round(metrics["range_coverage"], 3),
            "range_width_price": round(metrics["range_width_price"], 3),
            "n_fit": int(len(split.fit)), "n_valid": int(len(split.val)), "n_test": int(len(split.test)),
            "by_staleness": {str(k): {"mae": round(float(r["mae"]), 3), "coverage": round(float(r["coverage"]), 3),
                                      "n": int(r["n"])} for k, r in metrics["by_staleness"].iterrows()},
        },
        "importance": {k: round(float(v), 4) for k, v in metrics["importance"].items()},
    }
    daily_benchmark(df).to_csv(f"{out_dir}/benchmark.csv", index=False)
    # one row per bond: its static fields plus its most recent trade (yield, spread at the time, date)
    last = df.sort_values("trade_date").groupby("cusip").tail(1)
    ref = last[["cusip", "state", "sector", "rating", "coupon", "maturity_date", "issue_size_mm", "callable",
                "bank_qualified", "insured", "trade_date", "price", "yield", TARGET]].rename(
        columns={"trade_date": "last_trade_date", "price": "last_price", "yield": "last_yield", TARGET: "last_spread"})
    # the similar-bond spread each bucket shows over the last 30 days of data, so a bond priced today
    # (looked up or entered by hand) gets the same feature the model saw in training
    tail = df[df["trade_date"] > df["trade_date"].max() - pd.Timedelta(days=30)]
    bt = bucket_table(tail)
    rb = (ref["rating"].map(RATING_NUM).fillna(RATING_NUM["NR"]).astype(int) // 3)
    mb = pd.cut(((ref["maturity_date"] - df["trade_date"].max()).dt.days / 365.25).clip(0.25, 40),
                [0, 3, 7, 12, 20, 40], labels=False)
    ref["bucket_spread_30d"] = [bt.get(f"{a}-{b}", np.nan) for a, b in zip(rb, mb)]
    meta["bucket_spread_latest"] = bt
    json.dump(meta, open(f"{out_dir}/meta.json", "w"), indent=2)
    ref.to_csv(f"{out_dir}/bond_reference.csv", index=False)


# ----------------------------------------------------------------------------- main
def main(path="data/trades_synthetic.csv", out_dir="models", test_days=60, val_days=45, quiet=False):
    log = (lambda *a: None) if quiet else print
    df = build_features(load_trades(path)).dropna(subset=["price", "yield"])
    sp = split_by_time(df, test_days, val_days)
    log(f"fit {len(sp.fit):,} to {sp.cut_val.date()} | valid {len(sp.val):,} to {sp.cut_test.date()} | test {len(sp.test):,} after")
    f = fit_all(sp)
    m = evaluate(f, sp.test)
    log(f"baseline ridge      MAE yield {m['baseline_mae_yield']:.3f} pts | MAE price {m['baseline_mae_price']:.2f}")
    log(f"lightgbm            MAE yield {m['lgb_mae_yield']:.3f} pts | MAE price {m['lgb_mae_price']:.2f}  (best iter {m['best_iter']})")
    log(f"10-90 range: global widen {f.widen_global:.3f} yield pts, by staleness "
        + ", ".join(f"{k} {v:.3f}" for k, v in f.widen.items()))
    log(f"coverage on test {m['range_coverage']:.1%} (target {TARGET_COVERAGE:.0%}), mean width {m['range_width_price']:.2f} price pts")
    log("\nby time since the bond last traded:")
    log(m["by_staleness"].round(3))
    log("\ntop features by gain:")
    log(m["importance"].head(8).round(3).rename(index=FRIENDLY))
    save(f, df, sp, m, out_dir, path)
    log(f"\nsaved {out_dir}/")
    return f, m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/trades_synthetic.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--val-days", type=int, default=45)
    a = ap.parse_args()
    main(a.path, a.out, a.test_days, a.val_days)
