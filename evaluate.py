"""
Evaluate MuniPricer beyond a single train/test split.

  python evaluate.py                          # synthetic data, all sections
  python evaluate.py data/wrds_msrb.csv       # real data
  python evaluate.py --folds 3 --skip-ablations

Writes reports/evaluation.md and PNG charts to reports/. Sections:

  1. Rolling-origin backtest: slide the test window back `folds` times so the
     accuracy and the range coverage are measured on several different periods,
     not one lucky or unlucky one. Reports mean and spread across folds.
  2. Ablations: retrain with groups of features removed, plus the v1
     absolute-yield target, so the write-up can say what each piece buys.
  3. Error analysis on the final split: residual drift over time, error by
     maturity, rating, trade type, trade size and staleness, calibration by
     bucket, and the twenty worst misses.

Author: Anubhav Awasthi
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import features as F
from features import FEATURES, FRIENDLY, STALE_LABELS, build_features, load_trades, stale_bucket
from train import Split, evaluate, fit_all, split_by_time

ABLATIONS = {
    "full model": {"drop": []},
    "no bond history (last trade, days since, similar bonds)": {"drop": ["last_spread_same_cusip", "days_since_last_trade", "bucket_spread_30d"]},
    "no trade features (size, buy/sell)": {"drop": ["log_par", "trade_type"]},
    "no market momentum": {"drop": ["market_momentum_7d"]},
    "no reference fields (rating, size, call, insured, BQ)": {"drop": ["rating_num", "log_issue_size", "callable", "insured", "bank_qualified"]},
    "v1 target: absolute yield, no benchmark": {"drop": [], "bench_window": None},
}


# ----------------------------------------------------------------------------- 1. backtest
def backtest(df: pd.DataFrame, folds=4, test_days=30, val_days=45, log=print) -> pd.DataFrame:
    rows = []
    end = df["trade_date"].max()
    for k in range(folds):
        fold_end = end - pd.Timedelta(days=test_days * k)
        sp = split_by_time(df, test_days, val_days, end=fold_end)
        if len(sp.fit) < 5000 or len(sp.test) < 200:
            log(f"fold {k}: too little data (fit {len(sp.fit)}, test {len(sp.test)}), stopping")
            break
        t0 = time.time()
        f = fit_all(sp)
        m = evaluate(f, sp.test)
        bs = m["by_staleness"]
        rows.append({"fold": k, "test_start": sp.cut_test.date() + pd.Timedelta(days=1), "test_end": fold_end.date(),
                     "n_test": m["n_test"], "baseline_mae": m["baseline_mae_price"], "lgb_mae": m["lgb_mae_price"],
                     "coverage": m["range_coverage"], "width": m["range_width_price"],
                     **{f"mae {lab}": float(bs.loc[lab, "mae"]) if lab in bs.index else np.nan for lab in STALE_LABELS},
                     **{f"cov {lab}": float(bs.loc[lab, "coverage"]) if lab in bs.index else np.nan for lab in STALE_LABELS}})
        log(f"fold {k}: test {rows[-1]['test_start']} to {rows[-1]['test_end']}  MAE {m['lgb_mae_price']:.3f} "
            f"(baseline {m['baseline_mae_price']:.3f})  coverage {m['range_coverage']:.1%}  [{time.time() - t0:.0f}s]")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 2. ablations
def ablations(raw: pd.DataFrame, test_days=60, val_days=45, log=print) -> pd.DataFrame:
    rows = []
    for name, cfg in ABLATIONS.items():
        t0 = time.time()
        bw = cfg.get("bench_window", F.BENCH_WINDOW)
        df = build_features(raw, bench_window=bw).dropna(subset=["price", "yield"])
        feats = [c for c in FEATURES if c not in cfg["drop"]]
        sp = split_by_time(df, test_days, val_days)
        f = fit_all(sp, features=feats)
        m = evaluate(f, sp.test)
        bs = m["by_staleness"]
        rows.append({"variant": name, "lgb_mae": m["lgb_mae_price"], "coverage": m["range_coverage"],
                     "width": m["range_width_price"],
                     "mae stale (1-3 mo)": float(bs.loc["1-3 mo", "mae"]) if "1-3 mo" in bs.index else np.nan,
                     "mae stale (3-12 mo)": float(bs.loc["3-12 mo", "mae"]) if "3-12 mo" in bs.index else np.nan})
        log(f"{name:58s} MAE {m['lgb_mae_price']:.3f}  coverage {m['range_coverage']:.1%}  [{time.time() - t0:.0f}s]")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 3. error analysis
def error_analysis(df: pd.DataFrame, test_days=60, val_days=45, log=print) -> dict:
    sp = split_by_time(df, test_days, val_days)
    f = fit_all(sp)
    m = evaluate(f, sp.test)
    t = sp.test.copy()
    p = m["pred"]
    t["pred"] = p["price_pred"]; t["resid_px"] = t["price"] - p["price_pred"]
    t["resid_y"] = t["yield"] - p["yield_pred"]
    t["abs_err"] = t["resid_px"].abs()
    t["inside"] = (t["yield"] >= p["yield_lo"]) & (t["yield"] <= p["yield_hi"])
    t["width"] = p["price_hi"] - p["price_lo"]

    def by(col, bins=None, labels=None):
        key = pd.cut(t[col], bins, labels=labels) if bins is not None else t[col]
        g = t.groupby(key, observed=True).agg(mae=("abs_err", "mean"), bias_yield=("resid_y", "mean"),
                                              coverage=("inside", "mean"), width=("width", "mean"), n=("abs_err", "size"))
        return g.round(3)

    out = {
        "metrics": m, "fitted": f, "split": sp,
        "drift": t.set_index("trade_date")["resid_y"].resample("W").agg(["mean", "count"]).round(3)
            .rename_axis("week ending").pipe(lambda d: d.set_index(d.index.strftime("%Y-%m-%d"))),
        "by_stale": t.assign(b=stale_bucket(t["days_since_last_trade"])).groupby("b", observed=True)
            .agg(mae=("abs_err", "mean"), bias_yield=("resid_y", "mean"), coverage=("inside", "mean"), width=("width", "mean"), n=("abs_err", "size")).round(3),
        "by_maturity": by("years_to_maturity", [0, 3, 7, 12, 20, 40], ["0-3y", "3-7y", "7-12y", "12-20y", "20y+"]),
        "by_rating": by("rating"),
        "by_trade_type": by("trade_type"),
        "by_size": by("par_traded", [0, 25000, 100000, 1e6, 1e12], ["<25k", "25-100k", "100k-1M", ">1M"]),
        "by_sector": by("sector"),
        "worst": t.nlargest(20, "abs_err")[["cusip", "trade_date", "trade_type", "par_traded", "rating", "years_to_maturity",
                                           "days_since_last_trade", "price", "pred", "resid_px", "width"]],
        "test": t,
    }
    return out


# ----------------------------------------------------------------------------- charts
def charts(bt: pd.DataFrame | None, ea: dict, out_dir: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    t = ea["test"]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    d = ea["drift"]
    ax.plot(d.index, d["mean"], marker="o", color="#2b6cb0")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_title("Weekly mean residual (actual minus predicted yield, pts) on the test window")
    ax.set_ylabel("yield pts")
    fig.tight_layout(); fn = f"{out_dir}/residual_drift.png"; fig.savefig(fn, dpi=130); plt.close(fig); files.append(fn)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    bs = ea["by_stale"].reindex([l for l in STALE_LABELS if l in ea["by_stale"].index])
    ax.bar(bs.index, bs["mae"], color="#2b6cb0")
    for i, (v, n) in enumerate(zip(bs["mae"], bs["n"])):
        ax.text(i, v, f"{v:.2f}\nn={n}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Price MAE by time since the bond last traded"); ax.set_ylabel("points per 100")
    fig.tight_layout(); fn = f"{out_dir}/mae_by_staleness.png"; fig.savefig(fn, dpi=130); plt.close(fig); files.append(fn)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    bs2 = ea["by_stale"].reindex(bs.index)
    ax.bar(bs2.index, bs2["coverage"], color="#718096")
    ax.axhline(0.8, color="#c53030", ls="--", lw=1, label="target 80%")
    ax.set_ylim(0, 1); ax.legend(); ax.set_title("80% range: actual coverage by staleness")
    fig.tight_layout(); fn = f"{out_dir}/coverage_by_staleness.png"; fig.savefig(fn, dpi=130); plt.close(fig); files.append(fn)

    if bt is not None and len(bt):
        fig, ax = plt.subplots(figsize=(7, 3.2))
        x = [str(s) for s in bt["test_start"]]
        ax.plot(x, bt["lgb_mae"], marker="o", label="LightGBM", color="#2b6cb0")
        ax.plot(x, bt["baseline_mae"], marker="s", label="ridge baseline", color="#a0aec0")
        ax.set_title("Rolling backtest: price MAE by test window"); ax.set_ylabel("points per 100"); ax.legend()
        fig.tight_layout(); fn = f"{out_dir}/backtest_mae.png"; fig.savefig(fn, dpi=130); plt.close(fig); files.append(fn)

    fig, ax = plt.subplots(figsize=(5, 5))
    s = t.sample(min(3000, len(t)), random_state=0)
    ax.scatter(s["price"], s["pred"], s=5, alpha=0.4, color="#2b6cb0")
    lim = [s["price"].min() - 2, s["price"].max() + 2]
    ax.plot(lim, lim, color="#999", lw=0.8); ax.set_xlabel("actual price"); ax.set_ylabel("predicted"); ax.set_title("Actual vs predicted (test)")
    fig.tight_layout(); fn = f"{out_dir}/actual_vs_predicted.png"; fig.savefig(fn, dpi=130); plt.close(fig); files.append(fn)
    return files


# ----------------------------------------------------------------------------- report
def md(df: pd.DataFrame, floatfmt=".3f") -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    cols = [str(c) for c in d.columns]
    idx_name = d.index.name or ""
    lines = ["| " + idx_name + " | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in d.iterrows():
        lines.append("| " + str(i) + " | " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def write_report(path, data_path, bt, ab, ea, files):
    m = ea["metrics"]; f = ea["fitted"]; sp = ea["split"]
    synthetic = "synthetic" in os.path.basename(data_path)
    L = [f"# MuniPricer evaluation", "",
         f"Data: `{data_path}` ({'synthetic' if synthetic else 'real MSRB'}). Generated by `evaluate.py` on {pd.Timestamp.today().date()}.", ""]
    if synthetic:
        L += ["> These numbers describe the pipeline on simulated trades. They say the code works and which design choices matter; "
              "they are not a claim about the real market. Retrain on WRDS/MSRB data before quoting accuracy.", ""]
    L += ["## Headline (final split)", "",
          f"- Fit {len(sp.fit):,} trades to {sp.cut_val.date()}, calibrate on {len(sp.val):,} to {sp.cut_test.date()}, test on {len(sp.test):,} after.",
          f"- Price MAE: **{m['lgb_mae_price']:.2f}** points per 100 (ridge baseline {m['baseline_mae_price']:.2f}, "
          f"{1 - m['lgb_mae_price'] / m['baseline_mae_price']:.0%} lower).",
          f"- 80% range: covers **{m['range_coverage']:.1%}** of test trades, mean width {m['range_width_price']:.2f} points.",
          f"- Conformal margin (yield pts): global {f.widen_global:.3f}; by staleness " + ", ".join(f"{k} {v:.3f}" for k, v in f.widen.items()) + ".", ""]
    if bt is not None and len(bt):
        L += ["## Rolling backtest", "",
              f"{len(bt)} test windows of {int((pd.Timestamp(bt['test_end'].iloc[0]) - pd.Timestamp(bt['test_start'].iloc[0])).days) + 1} days each, "
              "each with its own fit, calibration and test slice.", "",
              md(bt[["fold", "test_start", "test_end", "n_test", "baseline_mae", "lgb_mae", "coverage", "width"]].set_index("fold")), "",
              f"Across folds: MAE {bt['lgb_mae'].mean():.3f} ± {bt['lgb_mae'].std():.3f}, coverage {bt['coverage'].mean():.1%} ± {bt['coverage'].std():.1%}.", ""]
        stale_cols = [c for c in bt.columns if c.startswith("mae ")]
        L += ["MAE by staleness, per fold:", "", md(bt[["fold"] + stale_cols].set_index("fold")), ""]
    if ab is not None and len(ab):
        L += ["## Ablations", "", "Each row retrains from scratch on the final split with a group of features removed.", "",
              md(ab.set_index("variant")), "",
              "Read the gap between a row and the full model as what that feature group is worth.", ""]
    L += ["## Error analysis (final split)", "",
          "### Residual drift over the test window", "", "Mean (actual minus predicted) yield by week. Values near zero mean the benchmark is carrying the rate level.", "",
          md(ea["drift"]), "",
          "### By time since the bond last traded", "", md(ea["by_stale"]), "",
          "### By years to maturity", "", md(ea["by_maturity"]), "",
          "### By rating", "", md(ea["by_rating"]), "",
          "### By trade type (MSRB: D inter-dealer, P dealer buys from customer, S dealer sells to customer)", "", md(ea["by_trade_type"]), "",
          "### By trade size", "", md(ea["by_size"]), "",
          "### By sector", "", md(ea["by_sector"]), "",
          "### Twenty worst misses", "", md(ea["worst"].set_index("cusip"), ".2f"), "",
          "### Feature importance (gain share)", "", md(m["importance"].rename(index=FRIENDLY).rename("gain").to_frame()), ""]
    if files:
        L += ["## Charts", ""] + [f"![{os.path.basename(x)}]({os.path.basename(x)})" for x in files] + [""]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/trades_synthetic.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--fold-days", type=int, default=30)
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--val-days", type=int, default=45)
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--skip-ablations", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = load_trades(a.path)
    df = build_features(raw).dropna(subset=["price", "yield"])
    bt = ab = None
    if not a.skip_backtest:
        print("== rolling backtest"); bt = backtest(df, a.folds, a.fold_days, a.val_days)
        bt.to_csv(f"{a.out}/backtest.csv", index=False)
    if not a.skip_ablations:
        print("== ablations"); ab = ablations(raw, a.test_days, a.val_days)
        ab.to_csv(f"{a.out}/ablations.csv", index=False)
    print("== error analysis"); ea = error_analysis(df, a.test_days, a.val_days)
    files = charts(bt, ea, a.out)
    write_report(f"{a.out}/evaluation.md", a.path, bt, ab, ea, files)
    print(f"wrote {a.out}/evaluation.md and {len(files)} charts")


if __name__ == "__main__":
    main()
