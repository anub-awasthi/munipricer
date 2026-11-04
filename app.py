"""
MuniPricer: fair-price estimates for municipal bonds that rarely trade.

  streamlit run app.py

Author: Anubhav Awasthi
"""
import json
import numpy as np
import pandas as pd
import streamlit as st

from predict import MuniPricer
from features import RATING_ORDER

st.set_page_config(page_title="MuniPricer", page_icon="📈", layout="wide")


@st.cache_resource
def load():
    return MuniPricer("models")


p = load()
meta = p.meta

st.title("MuniPricer")
st.caption("A fair-price estimate for any municipal bond, even one that hasn't traded in months. "
           "Built by Anubhav Awasthi, Ibukun and Eno for AI Essentials, Booth Fall 2026.")

left, right = st.columns([1, 1.4], gap="large")

with left:
    mode = st.radio("How do you want to describe the bond?", ["Look up a CUSIP", "Enter the details myself"], horizontal=True)
    bond = {}
    if mode == "Look up a CUSIP":
        example = p.ref["cusip"].iloc[0]
        cusip = st.text_input("CUSIP", value=example, help="9-character bond ID. Try the example or paste your own.")
        row = p.lookup(cusip)
        if row is None:
            st.warning("That CUSIP isn't in our reference data yet. Switch to manual entry.")
            st.stop()
        bond = row.to_dict()
        st.markdown(
            f"**{bond['state']} · {bond['sector']} · {bond['rating']}**  \n"
            f"{bond['coupon']:.2f}% coupon, matures {pd.Timestamp(bond['maturity_date']).date()}  \n"
            f"Issue size ${bond['issue_size_mm']:.0f}M · "
            f"{'callable' if bond['callable'] else 'non-callable'}"
            f"{' · insured' if bond['insured'] else ''}{' · bank qualified' if bond['bank_qualified'] else ''}")
        days = (p.as_of - pd.Timestamp(bond["last_trade_date"])).days
        st.info(f"Last traded **{pd.Timestamp(bond['last_trade_date']).date()}** ({days} days ago) at **{bond['last_price']:.2f}**, "
                f"a {bond['last_yield']:.2f}% yield.")
    else:
        c1, c2 = st.columns(2)
        bond["coupon"] = c1.number_input("Coupon (%)", 0.0, 10.0, 4.0, 0.25)
        bond["maturity_date"] = c2.date_input("Maturity date", value=p.as_of.date() + pd.Timedelta(days=365 * 10))
        bond["rating"] = c1.selectbox("Rating", RATING_ORDER, index=2)
        bond["state"] = c2.selectbox("State", meta["categories"]["state"])
        bond["sector"] = st.selectbox("Sector", meta["categories"]["sector"])
        bond["issue_size_mm"] = st.number_input("Issue size ($M)", 1.0, 5000.0, 25.0, 5.0)
        c3, c4, c5 = st.columns(3)
        bond["callable"] = c3.checkbox("Callable", True)
        bond["insured"] = c4.checkbox("Insured", False)
        bond["bank_qualified"] = c5.checkbox("Bank qualified", False)
        with st.expander("Last known trade (optional, improves accuracy a lot)"):
            has = st.checkbox("I know the last trade")
            if has:
                bond["last_yield"] = st.number_input("Last trade yield (%)", 0.0, 12.0, 3.5, 0.05)
                bond["last_trade_date"] = st.date_input("Last trade date", value=p.as_of.date() - pd.Timedelta(days=90))

    st.divider()
    st.markdown("**Your side of the trade**")
    tt = st.radio("", ["Buying from a dealer", "Selling to a dealer", "Dealer-to-dealer / neutral"], label_visibility="collapsed")
    bond["trade_type"] = {"Buying from a dealer": "S", "Selling to a dealer": "P", "Dealer-to-dealer / neutral": "D"}[tt]
    bond["par_traded"] = st.select_slider("Trade size (face value)", [5000, 10000, 25000, 50000, 100000, 250000, 1000000],
                                          value=25000, format_func=lambda v: f"${v:,}")

res = p.price(bond)

with right:
    st.subheader("Estimated fair price")
    m1, m2, m3 = st.columns(3)
    m1.metric("Price (per 100)", f"{res['price']:.2f}")
    m2.metric("80% range", f"{res['price_low']:.1f} to {res['price_high']:.1f}")
    m3.metric("Yield", f"{res['yield']:.2f}%")
    conf_color = {"high": "green", "medium": "orange", "low": "red"}[res["confidence"]]
    st.markdown(f"Confidence: :{conf_color}[**{res['confidence']}**] "
                f"(range width {res['price_high'] - res['price_low']:.2f} points)")

    if mode == "Look up a CUSIP":
        delta = res["price"] - bond["last_price"]
        st.markdown(f"That is **{abs(delta):.2f} points {'above' if delta > 0 else 'below'}** the last trade. "
                    f"{'Rates or spreads have moved since then.' if abs(delta) > 1 else 'Not much has changed since then.'}")

    st.subheader("What drove this number")
    st.write(p.explain(res, bond))

    contrib = pd.DataFrame(res["contributions"]).head(8)
    if not contrib.empty:
        short = {"this bond's last trade": "last trade of this bond", "recent trades in similar bonds": "similar bonds, 30d",
                 "overall market level this week": "market this week", "trade type (buy/sell/dealer)": "buy/sell markup",
                 "market level on trade date": "market level"}
        chart = contrib.assign(label=contrib["label"].map(lambda l: short.get(l, l)))[["label", "price_pts"]].set_index("label").sort_values("price_pts")
        st.bar_chart(chart, horizontal=True, color="#2b6cb0", height=280)
        st.caption("Price impact (points per 100) of each factor, relative to an average bond in the training data. "
                   "Positive pushes the price up, negative pushes it down.")

with st.expander("About the model and its accuracy"):
    mt = meta["metrics"]
    st.markdown(f"""
- **Model:** gradient-boosted trees (LightGBM) predicting yield, converted to price with standard bond math.
  Two more models predict the 10th and 90th percentile to give the range, then the range is calibrated on held-out data.
- **Training data:** {mt['n_fit']:,} trades (fit) + {mt['n_valid']:,} (calibration). Tested on the final {mt['n_test']:,} trades, which the model never saw.
- **Accuracy on the test period:** average price error **{mt['lgb_mae_price']:.2f} points**, vs {mt['baseline_mae_price']:.2f} for a linear baseline.
- **Range honesty:** the 80% range contained the real trade price {mt['range_coverage']:.0%} of the time on test data.
- **Biggest drivers overall:** {', '.join(list(meta['importance'].keys())[:4])}.
- **Data note:** currently trained on a simulated dataset built to mimic MSRB trade data. Swap in a WRDS/MSRB export to retrain on real trades (see README).
""")
