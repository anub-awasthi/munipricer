"""Tests for the pure helpers in pull_wrds.py (no WRDS connection needed).

Author: Anubhav Awasthi
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pull_wrds as P  # noqa: E402


def test_pick_table():
    assert P.pick_table(["msrb"]) == "msrb"
    assert P.pick_table(["issues", "trades"]) == "trades"
    assert P.pick_table(["a", "b"], "b") == "b"
    with pytest.raises(SystemExit):
        P.pick_table(["a", "b"])


def test_find_date_col():
    assert P.find_date_col(["cusip", "TRADE_DATE", "dollar_price"]) == "TRADE_DATE"
    with pytest.raises(SystemExit):
        P.find_date_col(["cusip", "price"])


def test_month_windows_cover_range_without_gaps():
    w = P.month_windows("2026-01-15", "2026-03-31")
    assert w[0][0] == "2026-01-15" and w[-1][1] == "2026-04-01"
    for (_, b), (c, _) in zip(w[:-1], w[1:]):
        assert b == c
