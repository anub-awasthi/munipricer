"""
Optional LLM rewrite of the explanation paragraph.

The model's own explanation (predict.py) is built from templated sentences
over the SHAP contributions. That is deliberate: it never hallucinates a
number. This module asks Claude to rewrite that paragraph for readability
only, with the rules that every number must be kept exactly and nothing may
be added. It is off unless an API key is present, and any failure falls back
to the template text, so the app never depends on the network.

  export ANTHROPIC_API_KEY=...            # or put it in .streamlit/secrets.toml
  python explain_llm.py <CUSIP>           # prints template vs rewritten

Author: Anubhav Awasthi
"""
from __future__ import annotations

import os

MODEL = os.environ.get("MUNIPRICER_LLM_MODEL", "claude-haiku-5-5")

SYSTEM = (
    "You rewrite a short, machine-generated explanation of a municipal bond price estimate so a non-expert "
    "investor can read it comfortably. Rules: keep every number, percentage and date exactly as given; do not add "
    "facts, advice, or claims that are not in the text; keep the 80% range and the benchmark sentence; keep it to "
    "one paragraph of plain English, no bullet points, no headings, no preamble. Basis points may be written as bps."
)


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def rewrite(template_text: str, bond: dict | None = None, timeout: float = 15.0) -> tuple[str, bool]:
    """Returns (text, used_llm). Falls back to the template on any problem."""
    if not available():
        return template_text, False
    try:
        import anthropic
        client = anthropic.Anthropic(timeout=timeout)
        context = ""
        if bond:
            context = (f"Bond: {bond.get('state', '')} {bond.get('sector', '')}, rating {bond.get('rating', 'NR')}, "
                       f"{bond.get('coupon', '')}% coupon.\n\n")
        resp = client.messages.create(
            model=MODEL, max_tokens=400, system=SYSTEM,
            messages=[{"role": "user", "content": context + "Rewrite this:\n\n" + template_text}],
        )
        text = "".join(getattr(b, "text", "") for b in resp.content).strip()
        return (text, True) if _numbers_preserved(template_text, text) else (template_text, False)
    except Exception:
        return template_text, False


def _numbers_preserved(src: str, out: str) -> bool:
    """Guardrail: every number in the template must appear in the rewrite."""
    import re
    nums = set(re.findall(r"\d+\.\d+|\d+", src))
    return all(n in out for n in nums)


if __name__ == "__main__":
    import sys
    from predict import MuniPricer
    p = MuniPricer()
    cusip = sys.argv[1] if len(sys.argv) > 1 else p.ref["cusip"].iloc[0]
    row = p.lookup(cusip)
    b = row.to_dict()
    r = p.price(b)
    t = p.explain(r, b)
    print("TEMPLATE:\n" + t + "\n")
    out, used = rewrite(t, b)
    print(("LLM (" + MODEL + "):\n" if used else "LLM unavailable, fallback:\n") + out)
