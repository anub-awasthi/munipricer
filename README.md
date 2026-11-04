# MuniPricer

Fair-price estimates for municipal bonds that rarely trade. AI Essentials final project, Chicago Booth, Fall 2026.
Team: Anubhav Awasthi, Ibukun (IBK), Eno.

## What it does

Enter a CUSIP (or describe the bond). Get a predicted price per 100 of face value, an 80% range, and a plain-English explanation of what drove the number. Built for small holders and local treasurers who do not have a Bloomberg terminal.

## Quick start

```bash
pip install -r requirements.txt
python make_synthetic.py      # builds data/trades_synthetic.csv (skip once you have real data)
python train.py               # trains, evaluates, saves models/
streamlit run app.py          # opens the app in your browser
```

Command-line check for one bond: `python predict.py <CUSIP>`

## Files

| File | What it does |
|---|---|
| `make_synthetic.py` | Builds a realistic simulated trade file in the WRDS MSRB column layout |
| `features.py` | Feature engineering shared by training and inference (no future peeking) |
| `train.py` | Time-based split, ridge baseline, LightGBM point model, 10/90 quantile models, calibration, saves `models/` |
| `predict.py` | `MuniPricer` class: price one bond, range, SHAP-style contributions, explanation text |
| `app.py` | Streamlit app |
| `export_demo.py` | Precomputes predictions for the web demo (`demo_data.json`) |
| `web/index.html` | Shareable web demo (no model runtime, uses precomputed predictions) |

## Swapping in real MSRB data from WRDS

1. Log in to WRDS with Booth credentials, open the MSRB dataset, run a web query for a date range (3 to 12 months is plenty), export CSV.
2. Rename columns to match `data/trades_synthetic.csv`:

| Needed column | Meaning | Typical WRDS/MSRB field |
|---|---|---|
| `cusip` | 9-char ID | `cusip` |
| `trade_date` | YYYY-MM-DD | `trade_date` |
| `trade_type` | `P` customer buy, `S` customer sell, `D` dealer-dealer | `trade_type` |
| `par_traded` | face value traded | `par_traded` |
| `price` | per 100 | `dollar_price` |
| `yield` | percent | `yield` |
| `coupon` | percent | `coupon` |
| `maturity_date` | YYYY-MM-DD | `maturity_date` |
| `dated_date` | YYYY-MM-DD | `dated_date` |
| `state`, `sector`, `rating`, `issue_size_mm`, `callable`, `bank_qualified`, `insured` | bond reference fields | join from Mergent/EMMA reference data; fill `rating` with `NR` and booleans with `False` if unavailable |

3. `python train.py data/wrds_msrb.csv`, then `python export_demo.py` to refresh the web demo.

## Model notes

- Predicts **yield**, then converts to price with standard bond math (semiannual coupons). Yield is smoother across maturities than price, so the model learns faster.
- Split is **by time**: last 60 days held out as test, 45 days before that for early stopping and range calibration. No shuffling, so the model is judged on the future.
- The two strongest features are the bond's own last trade and the trailing 30-day yield of similar bonds (same rating and maturity bucket). Both are computed using only earlier trades.
- The 80% range comes from quantile models, widened by a conformal adjustment fit on the calibration slice. On synthetic test data it covers 77% of trades. Rates drifted during the test window, which is what the real world does too.
- Explanations use LightGBM's built-in SHAP contributions (`pred_contrib=True`), converted from yield to price using the bond's duration.

## Current results (synthetic data)

| | Price MAE (points) |
|---|---|
| Ridge baseline | 1.40 |
| LightGBM | 0.85 |

Error rises for bonds that have not traded in a long time (1.7 points at 3-12 months stale vs 0.9 under a week), which is exactly the segment the tool exists for and where real data will matter most.
