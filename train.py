"""
Train the muni bond pricing model.

  python train.py                      # uses data/trades_synthetic.csv
  python train.py data/wrds_msrb.csv   # real data, same columns

Steps: load -> features -> time-based split -> linear baseline -> LightGBM
(median + 10th/90th percentile) -> evaluate -> save to models/.

Author: Anubhav Awasthi
"""
import json
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from features import load_trades, build_features, price_from_yield, FEATURES, CAT_COLS, NUM_COLS, FRIENDLY

path = sys.argv[1] if len(sys.argv) > 1 else "data/trades_synthetic.csv"
df = build_features(load_trades(path))
df = df.dropna(subset=["price", "yield"])

# ---- time-based split (no peeking at the future) ----
#   fit:   everything up to 105 days before the end
#   valid: next 45 days   (early stopping + range calibration)
#   test:  last 60 days   (never touched until final scoring)
end = df["trade_date"].max()
cut_test = end - pd.Timedelta(days=60)
cut_val = cut_test - pd.Timedelta(days=45)
fit = df[df["trade_date"] <= cut_val]
val = df[(df["trade_date"] > cut_val) & (df["trade_date"] <= cut_test)]
test = df[df["trade_date"] > cut_test]
train = pd.concat([fit, val])
print(f"fit {len(fit):,} to {cut_val.date()} | valid {len(val):,} to {cut_test.date()} | test {len(test):,} after")

X_fit, X_val, X_te = fit[FEATURES], val[FEATURES], test[FEATURES]
y_fit, y_val, y_te = fit["yield"], val["yield"], test["yield"]


def to_price(frame, yld):
    return price_from_yield(frame["coupon"].values, yld, frame["years_to_maturity"].values)


# ---- baseline: ridge regression on numeric features (fill NaN with median) ----
med = X_fit[NUM_COLS].median()
ridge = Ridge(alpha=1.0).fit(X_fit[NUM_COLS].fillna(med), y_fit)
base_pred = ridge.predict(X_te[NUM_COLS].fillna(med))
base_mae_px = mean_absolute_error(test["price"], to_price(test, base_pred))
print(f"baseline ridge      MAE yield {mean_absolute_error(y_te, base_pred):.3f} pts | MAE price {base_mae_px:.2f}")

# ---- LightGBM point model (predicts yield) ----
params = dict(objective="huber", alpha=0.5, learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
              feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
dfit = lgb.Dataset(X_fit, y_fit, categorical_feature=CAT_COLS)
dval = lgb.Dataset(X_val, y_val, reference=dfit)
model = lgb.train(params, dfit, num_boost_round=6000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
pred = model.predict(X_te, num_iteration=model.best_iteration)
mae_px = mean_absolute_error(test["price"], to_price(test, pred))
print(f"lightgbm            MAE yield {mean_absolute_error(y_te, pred):.3f} pts | MAE price {mae_px:.2f}  (best iter {model.best_iteration})")

# ---- quantile models for the range, then conformal calibration on the valid slice ----
q_models = {}
for q in (0.1, 0.9):
    qp = dict(params, objective="quantile", alpha=q, num_leaves=31, min_data_in_leaf=100)
    q_models[q] = lgb.train(qp, dfit, num_boost_round=6000, valid_sets=[dval],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
lo_v, hi_v = q_models[0.1].predict(X_val), q_models[0.9].predict(X_val)
score = np.maximum(lo_v - y_val, y_val - hi_v)          # how far outside the band each valid trade fell
widen = float(np.quantile(score, 0.85))                  # widen both sides so 80% of valid trades are inside
lo = q_models[0.1].predict(X_te) - widen
hi = q_models[0.9].predict(X_te) + widen
coverage = np.mean((y_te >= lo) & (y_te <= hi))
print(f"10-90 range: widened by {widen:.3f} yield pts; coverage on test {coverage:.1%} (target 80%)")

# ---- accuracy by how recently the bond traded (the whole point of the tool) ----
test = test.assign(abs_err=np.abs(test["price"] - to_price(test, pred)))
stale = pd.cut(test["days_since_last_trade"].fillna(9999), [-1, 7, 30, 90, 365, 1e9],
               labels=["<1 wk", "1-4 wk", "1-3 mo", "3-12 mo", "never traded"])
print("\nprice MAE by time since the bond last traded:")
print(test.groupby(stale, observed=True)["abs_err"].agg(["mean", "count"]).round(2))

# ---- feature importance ----
imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
print("\ntop features by gain:")
print((imp / imp.sum()).head(8).round(3).rename(index=FRIENDLY))

# ---- save ----
model.save_model("models/lgb_yield.txt", num_iteration=model.best_iteration)
for q, m in q_models.items():
    m.save_model(f"models/lgb_q{int(q*100)}.txt", num_iteration=m.best_iteration)
cats = {c: list(df[c].cat.categories) for c in CAT_COLS}
meta = {
    "features": FEATURES, "cat_cols": CAT_COLS, "categories": cats, "num_medians": {k: float(v) for k, v in med.items()}, "range_widen": widen,
    "t0": str(df["trade_date"].min().date()), "market_yield_latest": float(df["yield"].tail(500).mean()), "as_of": str(df["trade_date"].max().date()),
    "metrics": {"baseline_mae_price": round(float(base_mae_px), 3), "lgb_mae_price": round(float(mae_px), 3),
                "range_coverage": round(float(coverage), 3), "n_fit": int(len(fit)), "n_valid": int(len(val)), "n_test": int(len(test))},
    "importance": {k: round(float(v), 4) for k, v in (imp / imp.sum()).items()},
}
json.dump(meta, open("models/meta.json", "w"), indent=2)
# bond reference table + history so the app can look up a CUSIP
ref = df.sort_values("trade_date").groupby("cusip").tail(1)[
    ["cusip", "state", "sector", "rating", "coupon", "maturity_date", "issue_size_mm", "callable", "bank_qualified", "insured",
     "trade_date", "price", "yield", "bucket_yield_30d"]].rename(columns={"trade_date": "last_trade_date", "price": "last_price", "yield": "last_yield"})
ref.to_csv("models/bond_reference.csv", index=False)
print("\nsaved models/ and bond_reference.csv")
