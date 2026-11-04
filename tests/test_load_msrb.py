"""
Tests for load_msrb.py against a mock export shaped like a WRDS MSRB file.

Author: Anubhav Awasthi
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import load_msrb as L  # noqa: E402


def make_mock_wrds(path, n=3000, seed=1):
    """A raw file in the WRDS/MSRB academic layout, with the usual dirt."""
    rng = np.random.default_rng(seed)
    cusips = [f"{rng.integers(100000, 999999)}AB{i % 10}" for i in range(300)]
    desc_pool = ["CALIFORNIA ST GO BDS", "NEW YORK ST DORM AUTH REVS", "TEXAS WTR DEV BRD", "CHICAGO ILL BRD ED",
                 "FLORIDA ST TPK AUTH", "OHIO ST UNIV GEN RCPTS", "HOUSTON TEX INDPT SCH DIST", "MASS HLTH & EDL FACS",
                 "WASH ST HSG FIN COMMN", "PENNSYLVANIA ST TPK COMMN", "ARIZONA ST", "N Y C MUN WTR FIN AUTH"]
    desc = {c: desc_pool[i % len(desc_pool)] for i, c in enumerate(cusips)}
    idx = rng.integers(0, len(cusips), n)
    dates = pd.Timestamp("2026-01-05") + pd.to_timedelta(rng.integers(0, 200, n), "D")
    df = pd.DataFrame({
        "CUSIP": [cusips[i] for i in idx],
        "Trade Date": dates.strftime("%Y%m%d"),
        "Time of Trade": "103000",
        "Trade Type Indicator": rng.choice(["D", "P", "S"], n),
        "Par Traded": rng.choice([5000, 10000, 25000, 50000, 100000], n).astype(object),
        "Dollar Price": np.round(rng.normal(102, 4, n), 3),
        "Yield (if applicable)": np.round(rng.normal(3.4, 0.5, n), 3),
        "Coupon (if available)": rng.choice([3.0, 4.0, 5.0], n).astype(object),
        "Maturity Date": (dates + pd.to_timedelta(rng.integers(400, 9000, n), "D")).strftime("%Y%m%d"),
        "Dated Date": "20200101",
        "When-Issued Indicator": "",
        "Weighted Price Indicator": "",
        "Security Description": [desc[cusips[i]] for i in idx],
    })
    # dirt: MM+ pars, when-issued rows, weighted prices, zero coupons, a bad cusip, a duplicate
    df.loc[:19, "Par Traded"] = "MM+"
    df.loc[20:29, "When-Issued Indicator"] = "Y"
    df.loc[30:39, "Weighted Price Indicator"] = "Y"
    df.loc[40:49, "Coupon (if available)"] = ""
    df.loc[50, "CUSIP"] = "BAD"
    df = pd.concat([df, df.iloc[[100]]], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def test_column_matching_is_loose():
    cols = ["CUSIP", "Trade Date", "DOLLAR_PRICE", "par_traded", "yield", "Coupon (if available)", "MaturityDate", "trade_type"]
    cm = L.match_columns(cols, L.TRADE_ALIASES)
    assert cm["price"] == "DOLLAR_PRICE" and cm["maturity_date"] == "MaturityDate" and cm["coupon"] == "Coupon (if available)"


def test_description_parsing():
    s = pd.Series(["CALIFORNIA ST GO BDS", "HOUSTON TEX INDPT SCH DIST", "N Y C MUN WTR FIN AUTH", "SOMEWHERE ELSE"])
    assert list(L.parse_state(s)) == ["CA", "TX", "NY", "UNK"]
    assert list(L.parse_sector(s)) == ["General Obligation", "School District", "Water & Sewer", "General Obligation"]


def test_rating_normalisation():
    r = L.norm_rating(pd.Series(["AA+", "Aa2", "A3", "BBB- (neg)", "junk", "", "AAA"]))
    assert list(r) == ["AA+", "AA", "A-", "BBB-", "NR", "NR", "AAA"]


def test_end_to_end_without_reference(tmp_path):
    raw = tmp_path / "msrb_raw.csv"
    make_mock_wrds(str(raw))
    out = tmp_path / "wrds_msrb.csv"
    rep = tmp_path / "report.md"
    df = L.run([str(raw)], str(out), None, str(rep))
    assert list(df.columns) == L.OUT_COLS
    # the dirt was removed: 20 MM+, 10 WI, 10 weighted, 10 zero coupon, 1 bad cusip, 1 duplicate (+ any out-of-range)
    assert len(df) <= 3001 - 52
    assert df["trade_type"].isin(["D", "P", "S"]).all()
    assert (df["state"] != "UNK").mean() > 0.9
    assert df["rating"].eq("NR").all() and (df["issue_size_mm"] == 20.0).all()
    txt = rep.read_text()
    assert "par masked" in txt and "Column mapping" in txt
    # and it trains
    from features import build_features, load_trades
    feat = build_features(load_trades(str(out)))
    assert feat["spread"].notna().all()


def test_reference_join_overrides_defaults(tmp_path):
    raw = tmp_path / "msrb_raw.csv"
    mock = make_mock_wrds(str(raw))
    cus = mock["CUSIP"].unique()[:100]
    ref = pd.DataFrame({"Cusip": cus, "S&P Rating": "A+", "Issue Amount": 55_000_000, "Callable": "Y",
                        "Use of Proceeds": "HOSPITAL", "State": "oh"})
    rp = tmp_path / "ref.csv"
    ref.to_csv(rp, index=False)
    df = L.run([str(raw)], str(tmp_path / "o.csv"), str(rp), str(tmp_path / "r.md"))
    hit = df["cusip"].isin(cus)
    assert hit.any()
    assert (df.loc[hit, "rating"] == "A+").all() and (df.loc[hit, "issue_size_mm"] == 55.0).all()
    assert df.loc[hit, "callable"].all() and (df.loc[hit, "sector"] == "Healthcare").all()
    assert (df.loc[hit, "state"] == "OH").all()
    assert (df.loc[~hit, "rating"] == "NR").all()
