# MuniPricer

Fair-price estimates for municipal bonds that rarely trade. AI Essentials final project, Chicago Booth, Fall 2026.
Team: Anubhav Awasthi, Ibukun (IBK), Eno.

Enter a CUSIP (or describe the bond). Get a predicted price per 100 of face value, an 80% range, and a plain-English
explanation of what drove the number. Built for small holders and local treasurers who do not have a Bloomberg terminal.

## Quick start

Python 3.10 or newer. On a Mac, `brew install libomp` first (LightGBM needs it).

```bash
pip install -r requirements.txt
python make_synthetic.py      # builds data/trades_synthetic.csv (about a second)
python train.py               # trains, evaluates, saves models/ (about 30 seconds)
streamlit run app.py          # opens the app; it trains itself on first launch if models/ is missing
```

Other commands:

```bash
python predict.py <CUSIP>                        # price one bond from the command line
python evaluate.py                               # rolling backtest, ablations, error report -> reports/
python export_demo.py                            # refresh demo_data.json for the web demo
python explain_llm.py <CUSIP>                    # compare template vs Claude-rewritten explanation (needs ANTHROPIC_API_KEY)
pytest -q                                        # tests (about 15 seconds)
```

## Real data from WRDS

This is the critical path. Nothing in the accuracy numbers means anything until the model is retrained on real trades.

1. Log in to [WRDS](https://wrds-www.wharton.upenn.edu) with Booth credentials. If you have no account, email the Booth
   library or the Kilts Center and ask for a student account with the **MSRB** dataset. Allow one to two days.
2. Open the MSRB transaction data, pick a date range (6 to 12 months is plenty), and export as CSV. Keep every column;
   the loader picks the ones it needs. Save the export under `data/raw/`.
3. Run the loader. It matches column names loosely, so WRDS, MSRB academic-file and EMMA layouts all work:

   ```bash
   python load_msrb.py data/raw/*.csv                              # -> data/wrds_msrb.csv + reports/load_report.md
   python load_msrb.py data/raw/*.csv --ref data/raw/emma_ref.csv  # also join rating, size, call, insured, BQ by CUSIP
   ```

   Read `reports/load_report.md`. It lists what was dropped and where each reference field came from. MSRB trade data
   carries no rating, sector, issue size, call, insurance or bank-qualified field. Without a reference file the loader
   parses state and sector from the security description, sets rating to `NR`, issue size to $20M and the flags to
   false. The model still runs (trade history does most of the work) but a reference file from EMMA, Mergent or
   Bloomberg keyed by CUSIP will help a lot.
4. Retrain and evaluate:

   ```bash
   python train.py data/wrds_msrb.csv
   python evaluate.py data/wrds_msrb.csv
   python export_demo.py
   ```

   Expect worse numbers than synthetic. Real spreads are messier, and real reference data is thinner.

The trade file layout the model expects (what `load_msrb.py` produces and `make_synthetic.py` imitates):

| Column | Meaning |
|---|---|
| `cusip` | 9-character ID |
| `trade_date`, `maturity_date`, `dated_date` | YYYY-MM-DD |
| `trade_type` | MSRB convention: `D` inter-dealer, `P` dealer **purchase from** a customer (the customer sold), `S` dealer **sale to** a customer (the customer bought) |
| `par_traded` | face value traded, dollars |
| `price` | per 100 |
| `yield` | percent |
| `coupon` | percent (zero-coupon bonds are dropped) |
| `state`, `sector`, `rating`, `issue_size_mm`, `callable`, `bank_qualified`, `insured` | bond reference fields |

## How the model works

- **Target is spread, not yield.** The model predicts a bond's spread over a trailing 3-day market benchmark (the mean
  yield of every trade in the market). Trees cannot extrapolate, so a model trained on absolute yield is anchored to
  the rate level it saw in training and drifts as rates move. In spread space the benchmark carries the level and the
  model only learns what is bond-specific. On synthetic data this cut price MAE from 0.85 to 0.71 and fixed the
  range coverage. The bond-history features are spreads too, for the same reason.
- **Features.** Coupon, years to maturity, rating, issue size, callable, insured, bank qualified, trade size, trade
  type, state, sector, plus history computed only from earlier trades: this bond's last trade (as a spread), days since
  it traded, the trailing 30-day spread of similar bonds (same rating and maturity bucket), and a 7-day market
  momentum term.
- **Model.** LightGBM with a Huber loss, early-stopped on a validation slice. Ridge regression as the baseline.
- **Split by time, never shuffled.** Last 60 days are the test set, the 45 days before that are for early stopping and
  calibration, everything earlier is for fitting. `evaluate.py` also runs a rolling backtest over four 30-day windows.
- **Range.** Two more LightGBM models predict the 10th and 90th percentile of spread. A conformal adjustment fit on the
  validation slice widens the band so it covers 80% out of sample, with a separate margin for each staleness bucket
  (bonds with no trade history get a much wider range).
- **Explanations.** LightGBM's built-in SHAP contributions give each feature's push on spread, converted to a price
  impact using the bond's duration and turned into template sentences. The optional Claude rewrite (`explain_llm.py`)
  only changes wording and is checked to keep every number.

## Current results (synthetic data)

These describe the pipeline, not the market. See `reports/evaluation.md` for the full report with charts.

| | Price MAE (points per 100) |
|---|---|
| Ridge baseline | 1.39 |
| LightGBM, v1 target (absolute yield) | 0.88 |
| LightGBM, spread target | **0.71** |

| Time since the bond last traded | Price MAE | 80% range coverage |
|---|---|---|
| Under 1 week | 0.63 | 80% |
| 1 to 4 weeks | 0.78 | 79% |
| 1 to 3 months | 0.87 | 78% |
| 3 to 12 months | 1.02 | 78% |
| No trade history | 2.76 | 83% |

Rolling backtest over four 30-day windows: MAE 0.71 ± 0.02, coverage 79.4% ± 0.4%.

## Files

| File | What it does |
|---|---|
| `make_synthetic.py` | Builds a simulated trade file in the MuniPricer layout (heavy-tailed activity, 35 bps bond-specific spread) |
| `load_msrb.py` | Turns a raw WRDS/MSRB export into that layout; cleaning, reference join, load report |
| `features.py` | Feature engineering shared by training and inference, benchmark, bond math |
| `train.py` | Split, baseline, LightGBM, quantile models, per-staleness calibration, saves `models/` |
| `evaluate.py` | Rolling backtest, ablations, error analysis, charts -> `reports/` |
| `predict.py` | `MuniPricer` class: price one bond, range, contributions, explanation |
| `explain_llm.py` | Optional Claude rewrite of the explanation |
| `app.py` | Streamlit app |
| `export_demo.py` | Precomputes predictions for the web demo (`demo_data.json`) |
| `web/index.html` | Shareable web demo (no model runtime) |
| `tests/` | pytest suite, also run by GitHub Actions on every push |

`models/` and `data/trades_synthetic.csv` are not in git. They are rebuilt by the two commands in the quick start.

## Deploying the app

Streamlit Community Cloud: point it at this repo and `app.py`. The app trains on synthetic data at first launch (about a
minute), so no model files need to be committed. To use the Claude rewrite there, add `ANTHROPIC_API_KEY` under the
app's secrets.
