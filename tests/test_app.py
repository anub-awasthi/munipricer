"""
Headless Streamlit test: the app renders in both modes against a small model dir.

Author: Anubhav Awasthi
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="module")
def small_models(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    m = tmp_path_factory.mktemp("models")
    env = dict(os.environ, MUNI_N_BONDS="400", MUNI_N_TRADES="8000", MUNI_OUT_DIR=str(d))
    subprocess.run([sys.executable, "make_synthetic.py"], cwd=ROOT, env=env, check=True, capture_output=True)
    subprocess.run([sys.executable, "train.py", str(d / "trades_synthetic.csv"), "--out", str(m)],
                   cwd=ROOT, check=True, capture_output=True)
    return str(m)


def test_app_renders_lookup_and_manual(small_models, monkeypatch):
    monkeypatch.setenv("MUNIPRICER_MODELS", small_models)
    monkeypatch.chdir(ROOT)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.metric) == 3 and float(at.metric[0].value) > 50
    at.radio[0].set_value("Enter the details myself").run()
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.metric) == 3
    # selling to a dealer must price below buying from one
    buy = float(at.metric[0].value)
    at.radio[1].set_value("Selling to a dealer").run()
    assert float(at.metric[0].value) < buy
