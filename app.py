"""
MuniPricer: fair-price estimates for municipal bonds that rarely trade.

  streamlit run app.py

Author: Anubhav Awasthi
"""
import os

import numpy as np
import pandas as pd
import streamlit as st

import explain_llm
from features import RATING_ORDER, STALE_LABELS
from predict import MuniPricer

MODEL_DIR = os.environ.get("MUNIPRICER_MODELS", "models")

st.set_page_config(page_title="MuniPricer", page_icon="📈", layout="wide")


@st.cache_resource
def load():
    return MuniPricer(MODEL_DIR)


if not os.path.exists(f"{MODEL_DIR}/meta.json"):
    # First launch on a fresh clone or on Streamlit Community Cloud (models/ is not in git): build the synthetic
    # dataset and train once. Takes about a minute. Set MUNIPRICER_NO_BOOTSTRAP=1 to disable.
    if os.environ.get("MUNIPRICER_NO_BOOTSTRAP") or MODEL_DIR != "models":
        st.error(f"No trained model in `{MODEL_DIR}/`. Run `python make_synthetic.py` then `python train.py` first.")
        st.stop()
    import subprocess
    import sys
    with st.status("No trained model found. Building the synthetic dataset and training (about a minute)...", expanded=True) as s:
        if not os.path.exists("data/trades_synthetic.csv"):
            st.write("Generating synthetic trades")
            subprocess.run([sys.executable, "make_synthetic.py"], check=True, capture_output=True)
        st.write("Training LightGBM models")
        subprocess.run([sys.executable, "train.py"], check=True, capture_output=True)
        s.update(label="Model ready", state="complete", expanded=False)
    st.rerun()

p = load()
meta = p.meta
mt = meta["metrics"]

st.title("MuniPricer")
st.caption("A fair-price estimate for any municipal bond, even one that hasn't traded in months. "
           "Built by Anubhav Awasthi, Ibukun and Eno for AI Essentials, Booth Fall 2026.")
if p.synthetic:
    st.warning("This model is trained on **simulated** trade data. The numbers show how the tool works, not what the "
               "market is doing. Retrain on MSRB data (see README) before using any price here.", icon="⚠️")

left, right = st.columns([1, 1.4], gap="large")

# ----------------------------------------------------------------------------- inputs
with left:
    mode = st.radio("How do you want to describe the bond?", ["Look up a CUSIP", "Enter the details myself"], horizontal=True)
    bond = {}
    if mode == "Look up a CUSIP":
        example = p.ref["cusip"].iloc[0]
        cusip = st.text_input("CUSIP", value=example, help="9-character bond ID. Try the example or paste your own.")
        row = p.lookup(cusip)
        if row is None:
            st.warning("That CUSIP isn't in our reference data. Switch to manual entry and describe the bond; "
                       "if you know its last trade, add it there too.")
            st.stop()
        bond = row.to_dict()
        st.markdown(
            f"**{bond['state']} · {bond['sector']} · {bond['rating']}**  \n"
            f"{bond['coupon']:.2f}% coupon, matures {pd.Timestamp(bond['maturity_date']).date()}  \n"
            f"Issue size ${bond['issue_size_mm']:.0f}M · "
            f"{'callable' if bond['callable'] else 'non-callable'}"
            f"{' · insured' if bond['insured'] else ''}{' · bank qualified' if bond['bank_qualified'] else ''}")
        days = (p.as_of - pd.Timestamp(bond["last_trade_date"])).days
        st.info(f"Last traded **{pd.Timestamp(bond['last_trade_date']).date()}** ({days} days ago) at "
                f"**{bond['last_price']:.2f}**, a {bond['last_yield']:.2f}% yield.")
    else:
        c1, c2 = st.columns(2)
        bond["coupon"] = c1.number_input("Coupon (%)", 0.25, 10.0, 4.0, 0.25)
        bond["maturity_date"] = c2.date_input("Maturity date", value=p.as_of.date() + pd.Timedelta(days=365 * 10),
                                              min_value=p.as_of.date() + pd.Timedelta(days=120))
        bond["rating"] = c1.selectbox("Rating", RATING_ORDER, index=2)
        bond["state"] = c2.selectbox("State", meta["categories"]["state"])
        bond["sector"] = st.selectbox("Sector", meta["categories"]["sector"])
        bond["issue_size_mm"] = st.number_input("Issue size ($M)", 1.0, 5000.0, 25.0, 5.0)
        c3, c4, c5 = st.columns(3)
        bond["callable"] = c3.checkbox("Callable", True)
        bond["insured"] = c4.checkbox("Insured", False)
        bond["bank_qualified"] = c5.checkbox("Bank qualified", False)
        with st.expander("Last known trade (optional, narrows the range a lot)"):
            has = st.checkbox("I know the last trade")
            if has:
                bond["last_yield"] = st.number_input("Last trade yield (%)", 0.0, 12.0, 3.5, 0.05)
                bond["last_trade_date"] = st.date_input("Last trade date", value=p.as_of.date() - pd.Timedelta(days=90),
                                                        max_value=p.as_of.date())

    st.divider()
    st.markdown("**Your side of the trade**")
    tt = st.radio("side", ["Buying from a dealer", "Selling to a dealer", "Dealer-to-dealer / neutral"], label_visibility="collapsed")
    # MSRB convention: S = dealer sells to a customer (you buy), P = dealer purchases from a customer (you sell)
    bond["trade_type"] = {"Buying from a dealer": "S", "Selling to a dealer": "P", "Dealer-to-dealer / neutral": "D"}[tt]
    bond["par_traded"] = st.select_slider("Trade size (face value)", [5000, 10000, 25000, 50000, 100000, 250000, 1000000],
                                          value=25000, format_func=lambda v: f"${v:,}")
    use_llm = False
    if explain_llm.available():
        use_llm = st.toggle("Rewrite the explanation with Claude", value=False,
                            help="Rewrites the templated paragraph for readability. Numbers are checked and kept exactly.")

res = p.price(bond)

# ----------------------------------------------------------------------------- results
with right:
    st.subheader("Estimated fair price")
    m1, m2, m3 = st.columns(3)
    m1.metric("Price (per 100)", f"{res['price']:.2f}")
    m2.metric("80% range", f"{res['price_low']:.1f} to {res['price_high']:.1f}")
    m3.metric("Yield", f"{res['yield']:.2f}%",
              delta=f"{res['spread'] * 100:+.0f} bps vs market {res['benchmark']:.2f}%", delta_color="off")
    conf_color = {"high": "green", "medium": "orange", "low": "red"}[res["confidence"]]
    st.markdown(f"Confidence: :{conf_color}[**{res['confidence']}**] "
                f"(yield range {(res['yield_high'] - res['yield_low']) * 100:.0f} bps, "
                f"price range {res['price_high'] - res['price_low']:.2f} points)")

    if mode == "Look up a CUSIP":
        delta = res["price"] - bond["last_price"]
        st.markdown(f"That is **{abs(delta):.2f} points {'above' if delta > 0 else 'below'}** the last trade. "
                    f"{'Rates or spreads have moved since then.' if abs(delta) > 1 else 'Not much has changed since then.'}")

    st.subheader("What drove this number")
    text = p.explain(res, bond)
    if use_llm:
        with st.spinner("Rewriting..."):
            text, used = explain_llm.rewrite(text, bond)
        if not used:
            st.caption("Claude rewrite unavailable or failed its number check; showing the template text.")
    st.write(text)

    contrib = pd.DataFrame(res["contributions"]).head(8)
    if not contrib.empty:
        short = {"this bond's last trade": "last trade of this bond", "recent trades in similar bonds": "similar bonds, 30d",
                 "market direction this week": "market direction", "trade type (buy/sell/dealer)": "buy/sell markup"}
        chart = (contrib.assign(label=contrib["label"].map(lambda l: short.get(l, l)))[["label", "price_pts"]]
                 .set_index("label").sort_values("price_pts"))
        st.bar_chart(chart, horizontal=True, color="#2b6cb0", height=280)
        st.caption("Price impact (points per 100) of each factor, relative to an average bond in the training data, "
                   "after the market benchmark. Positive pushes the price up, negative pushes it down.")

# ----------------------------------------------------------------------------- about
with st.expander("About the model and its accuracy"):
    bs = mt.get("by_staleness", {})
    rows = [(lab, bs[lab]["mae"], bs[lab]["coverage"], bs[lab]["n"]) for lab in STALE_LABELS if lab in bs]
    st.markdown(f"""
- **Model:** gradient-boosted trees (LightGBM) predicting the bond's **spread over a trailing market benchmark**,
  then converted to price with standard bond math. Predicting spread rather than yield keeps the model honest when
  rates move, because trees cannot extrapolate a level they were never trained on.
- **Range:** two more models predict the 10th and 90th percentile of spread; the band is then widened on held-out
  data (conformal calibration), separately for each staleness bucket, so stale bonds get wider ranges.
- **Training data:** {mt['n_fit']:,} trades (fit) + {mt['n_valid']:,} (calibration). Tested on the final {mt['n_test']:,}
  trades, which the model never saw. Data: `{meta.get('data_path', 'unknown')}`.
- **Accuracy on the test period:** average price error **{mt['lgb_mae_price']:.2f} points**, vs {mt['baseline_mae_price']:.2f}
  for a linear baseline. The 80% range contained the real price **{mt['range_coverage']:.0%}** of the time.
- **Biggest drivers overall:** {', '.join(list(meta['importance'].keys())[:4])}.
""")
    if rows:
        st.markdown("**Accuracy by how recently the bond last traded** (the whole point of the tool):")
        st.dataframe(pd.DataFrame(rows, columns=["Last traded", "Price MAE", "Range coverage", "Test trades"])
                     .set_index("Last traded").style.format({"Price MAE": "{:.2f}", "Range coverage": "{:.0%}"}),
                     width="stretch")
    st.markdown("Explanations are templated from the model's own feature contributions (SHAP), so every number comes from "
                "the model. The optional Claude rewrite only changes wording and is checked to keep every number.")
