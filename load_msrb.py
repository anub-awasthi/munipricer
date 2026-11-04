"""
Turn a raw WRDS / MSRB trade export into the MuniPricer trade layout.

  python load_msrb.py data/raw/msrb_export.csv                          # -> data/wrds_msrb.csv
  python load_msrb.py data/raw/*.csv --ref data/raw/emma_ref.csv        # join bond reference fields
  python load_msrb.py data/raw/msrb_export.csv --out data/wrds_msrb.csv --report reports/load_report.md

What it does
  1. Reads one or more raw files (CSV, CSV.GZ, or Parquet). Column names are
     matched loosely (case, underscores and spaces ignored) against known
     WRDS / MSRB names, so "Dollar Price", "dollar_price" and "DOLLAR_PRICE"
     all work. Unknown layouts fail with a message naming what was not found.
  2. Cleans: drops trades with no yield, no price, no coupon (zero-coupon
     bonds, which the bond math does not handle), "MM+" masked par amounts,
     when-issued trades, weighted-average prices, implausible prices or
     yields, and exact duplicates.
  3. Adds the reference fields the model expects (state, sector, rating,
     issue_size_mm, callable, bank_qualified, insured):
       - from a reference CSV keyed by cusip if --ref is given (EMMA, Mergent,
         Bloomberg, anything with a cusip column; names matched loosely), and
       - otherwise parsed from the MSRB security description for state and
         sector, with rating NR, issue_size_mm = the median synthetic size (20),
         and the three flags False.
     The report says how many rows came from each source so the team knows
     how thin the reference data is.
  4. Writes the cleaned file and a markdown report.

Author: Anubhav Awasthi
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

OUT_COLS = ["cusip", "trade_date", "trade_type", "par_traded", "price", "yield", "coupon", "maturity_date",
            "dated_date", "state", "sector", "rating", "issue_size_mm", "callable", "bank_qualified", "insured"]

# loose aliases: normalised (lowercase alphanumerics only) names seen in WRDS, MSRB academic files and EMMA
TRADE_ALIASES = {
    "cusip": ["cusip", "cusipid", "cusip9", "cusipno"],
    "trade_date": ["tradedate", "trddate", "trade_dt", "tradedt", "date"],
    "trade_type": ["tradetype", "tradetypeindicator", "trdtype", "tradetypeind", "side"],
    "par_traded": ["partraded", "par", "paramount", "paramt", "quantity", "parvalue"],
    "price": ["dollarprice", "price", "dollar_price", "prc", "tradeprice"],
    "yield": ["yield", "yld", "yieldifapplicable", "tradeyield", "yieldtoworst", "ytw"],
    "coupon": ["coupon", "couponifavailable", "couponrate", "cpn", "interestrate", "rate"],
    "maturity_date": ["maturitydate", "maturity", "matdate", "maturitydt"],
    "dated_date": ["dateddate", "dated", "dateddt"],
    "security_description": ["securitydescription", "secdesc", "description", "issuedescription", "secdescription"],
    "when_issued": ["whenissuedindicator", "whenissued", "wi", "wiindicator"],
    "weighted_price": ["weightedpriceindicator", "weightedprice", "wtdprice"],
}
REF_ALIASES = {
    "cusip": ["cusip", "cusipid", "cusip9"],
    "state": ["state", "st", "statecode", "issuerstate", "stateabbrev"],
    "sector": ["sector", "purpose", "usofproceeds", "useofproceeds", "sectordescription", "industry", "purposeclass"],
    "rating": ["rating", "sprating", "spratingcurrent", "spltrating", "moodysrating", "moodys", "fitchrating", "ratingsp"],
    "issue_size_mm": ["issuesizemm", "issuesize", "amountissued", "issueamount", "originalissueamount", "parissued",
                      "totalissuesize", "dealsize", "offeringamount"],
    "callable": ["callable", "iscallable", "callflag", "callind", "callabletype"],
    "bank_qualified": ["bankqualified", "bq", "bqflag", "bankqualifiedflag"],
    "insured": ["insured", "insurance", "insurer", "bondinsurance", "insuredflag"],
    "coupon": ["coupon", "couponrate"],
    "maturity_date": ["maturitydate", "maturity"],
}

MOODYS_TO_SP = {"Aaa": "AAA", "Aa1": "AA+", "Aa2": "AA", "Aa3": "AA-", "A1": "A+", "A2": "A", "A3": "A-",
                "Baa1": "BBB+", "Baa2": "BBB", "Baa3": "BBB-", "Ba1": "BB+", "Ba2": "BB", "Ba3": "BB-"}
SP_RATINGS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-"]

# state tokens as they appear at the start of MSRB security descriptions (e.g. "CALIFORNIA ST", "N Y C", "TEX")
STATE_PATTERNS = [
    ("CA", r"\b(?:CALIFORNIA|CALIF|CA)\b"), ("NY", r"\b(?:NEW YORK|N Y C?|NY|NYC)\b"), ("TX", r"\b(?:TEXAS|TEX|TX)\b"),
    ("IL", r"\b(?:ILLINOIS|ILL|IL|CHICAGO)\b"), ("FL", r"\b(?:FLORIDA|FLA|FL)\b"), ("PA", r"\b(?:PENNSYLVANIA|PA|PENN)\b"),
    ("OH", r"\b(?:OHIO|OH)\b"), ("NJ", r"\b(?:NEW JERSEY|N J|NJ)\b"), ("MA", r"\b(?:MASSACHUSETTS|MASS|MA)\b"),
    ("WA", r"\b(?:WASHINGTON ST|WASH|WA)\b"), ("MI", r"\b(?:MICHIGAN|MICH|MI)\b"), ("GA", r"\b(?:GEORGIA|GA)\b"),
    ("NC", r"\b(?:NORTH CAROLINA|N C|NC)\b"), ("VA", r"\b(?:VIRGINIA|VA)\b"), ("AZ", r"\b(?:ARIZONA|ARIZ|AZ)\b"),
    ("CO", r"\b(?:COLORADO|COLO|CO)\b"), ("MN", r"\b(?:MINNESOTA|MINN|MN)\b"), ("WI", r"\b(?:WISCONSIN|WIS|WI)\b"),
    ("MD", r"\b(?:MARYLAND|MD)\b"), ("MO", r"\b(?:MISSOURI|MO)\b"), ("IN", r"\b(?:INDIANA|IND|IN)\b"),
    ("TN", r"\b(?:TENNESSEE|TENN|TN)\b"), ("CT", r"\b(?:CONNECTICUT|CONN|CT)\b"), ("OR", r"\b(?:OREGON|ORE|OR)\b"),
    ("SC", r"\b(?:SOUTH CAROLINA|S C|SC)\b"), ("LA", r"\b(?:LOUISIANA|LA)\b"), ("KY", r"\b(?:KENTUCKY|KY)\b"),
    ("AL", r"\b(?:ALABAMA|ALA|AL)\b"), ("OK", r"\b(?:OKLAHOMA|OKLA|OK)\b"), ("UT", r"\b(?:UTAH|UT)\b"),
    ("NV", r"\b(?:NEVADA|NEV|NV)\b"), ("IA", r"\b(?:IOWA|IA)\b"), ("KS", r"\b(?:KANSAS|KANS|KS)\b"),
    ("AR", r"\b(?:ARKANSAS|ARK|AR)\b"), ("MS", r"\b(?:MISSISSIPPI|MISS|MS)\b"), ("NE", r"\b(?:NEBRASKA|NEB|NE)\b"),
    ("NM", r"\b(?:NEW MEXICO|N M|NM)\b"), ("HI", r"\b(?:HAWAII|HI)\b"), ("DC", r"\b(?:DISTRICT OF COLUMBIA|D C|DC)\b"),
    ("PR", r"\b(?:PUERTO RICO|P R|PR)\b"), ("WV", r"\b(?:WEST VIRGINIA|W VA|WV)\b"), ("ID", r"\b(?:IDAHO|ID)\b"),
    ("NH", r"\b(?:NEW HAMPSHIRE|N H|NH)\b"), ("ME", r"\b(?:MAINE|ME)\b"), ("RI", r"\b(?:RHODE ISLAND|R I|RI)\b"),
    ("MT", r"\b(?:MONTANA|MONT|MT)\b"), ("DE", r"\b(?:DELAWARE|DEL|DE)\b"), ("SD", r"\b(?:SOUTH DAKOTA|S D|SD)\b"),
    ("ND", r"\b(?:NORTH DAKOTA|N D|ND)\b"), ("AK", r"\b(?:ALASKA|AK)\b"), ("VT", r"\b(?:VERMONT|VT)\b"), ("WY", r"\b(?:WYOMING|WYO|WY)\b"),
]
SECTOR_PATTERNS = [
    ("School District", r"\b(?:SCH|SCHOOL|SCHS|SCHOOLS|ISD|USD|CSD|UNIF SCH|UNION FREE)\b"),
    ("Higher Ed", r"\b(?:UNIV|UNIVERSITY|COLLEGE|CLG|COMMUNITY COLLEGE|HIGHER ED|EDL FACS)\b"),
    ("Healthcare", r"\b(?:HOSP|HOSPITAL|HLTH|HEALTH|MED CTR|MEDICAL|HEALTHCARE|HLTH FACS)\b"),
    ("Housing", r"\b(?:HSG|HOUSING|HOUS FIN|MTG|MORTGAGE|SINGLE FAMILY|MULTIFAMILY)\b"),
    ("Water & Sewer", r"\b(?:WTR|WATER|SWR|SEWER|WASTEWATER|SANITATION|SAN DIST|UTIL DIST)\b"),
    ("Transportation", r"\b(?:TRANSN|TRANSPORTATION|TRANSIT|TPK|TURNPIKE|TOLL|HWY|HIGHWAY|AIRPT|AIRPORT|PORT AUTH|BRDG|BRIDGE|RAIL)\b"),
    ("Electric Utility", r"\b(?:ELEC|ELECTRIC|PWR|POWER|PUB PWR|MUN UTIL|ENERGY)\b"),
    ("General Obligation", r"\b(?:GO|G O|GEN OBLIG|GENERAL OBLIGATION|UNLTD TAX|LTD TAX|CNTY|COUNTY|CITY|TOWN|TWP|VILLAGE|BORO)\b"),
]


# ----------------------------------------------------------------------------- column matching
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def match_columns(columns, aliases: dict) -> dict:
    """Map our names -> the file's column names using loose matching."""
    normed = {_norm(c): c for c in columns}
    out = {}
    for ours, alts in aliases.items():
        for a in [ours] + alts:
            if _norm(a) in normed:
                out[ours] = normed[_norm(a)]
                break
    return out


def read_any(path: str) -> pd.DataFrame:
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


# ----------------------------------------------------------------------------- parsing helpers
def to_date(s: pd.Series) -> pd.Series:
    """Dates arrive as YYYYMMDD, YYYY-MM-DD, MM/DD/YYYY, or SAS numbers. Try the common ones."""
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d%b%Y", "%Y-%m-%d %H:%M:%S"):
        miss = out.isna()
        if not miss.any():
            break
        out[miss] = pd.to_datetime(s[miss], format=fmt, errors="coerce")
    return out


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("$", "").str.strip(), errors="coerce")


def to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "y", "yes", "t", "callable", "insured", "bq"])


def norm_rating(s: pd.Series) -> pd.Series:
    r = s.astype(str).str.strip().str.upper()
    r = r.str.replace(r"^(SP|S&P|MOODYS|MOODY'S|FITCH)[:\s]+", "", regex=True)
    r = r.str.replace(r"\s*\(.*\)$", "", regex=True)                       # "BBB- (neg)" -> "BBB-"
    r = r.str.replace(r"\s+(NEG|POS|STABLE|\*|WATCH).*$", "", regex=True).str.strip()
    moodys = {k.upper(): v for k, v in MOODYS_TO_SP.items()}
    r = r.map(lambda x: moodys.get(x, x))
    r = r.where(r.isin(SP_RATINGS), "NR")
    return r


def parse_state(desc: pd.Series) -> pd.Series:
    """State from the first few tokens of the MSRB security description."""
    head = desc.fillna("").astype(str).str.upper().str.slice(0, 40)
    out = pd.Series("UNK", index=desc.index)
    done = pd.Series(False, index=desc.index)
    for code, pat in STATE_PATTERNS:
        m = head.str.contains(pat, regex=True) & ~done
        out[m] = code
        done |= m
    return out


def parse_sector(desc: pd.Series) -> pd.Series:
    text = desc.fillna("").astype(str).str.upper()
    out = pd.Series("General Obligation", index=desc.index)
    done = pd.Series(False, index=desc.index)
    for sector, pat in SECTOR_PATTERNS:
        m = text.str.contains(pat, regex=True) & ~done
        out[m] = sector
        done |= m
    return out


# ----------------------------------------------------------------------------- main steps
def load_raw(paths: list[str]) -> tuple[pd.DataFrame, dict]:
    frames = []
    for p in paths:
        raw = read_any(p)
        cm = match_columns(raw.columns, TRADE_ALIASES)
        need = ["cusip", "trade_date", "trade_type", "par_traded", "price", "yield", "coupon", "maturity_date"]
        missing = [c for c in need if c not in cm]
        if missing:
            raise SystemExit(f"{p}: could not find columns for {missing}. Columns present: {list(raw.columns)[:40]}")
        df = pd.DataFrame({ours: raw[theirs] for ours, theirs in cm.items()})
        df["_source"] = os.path.basename(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df, cm


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    drops = Counter()
    n0 = len(df)
    df = df.copy()
    df["cusip"] = df["cusip"].astype(str).str.strip().str.upper()
    df["trade_type"] = df["trade_type"].astype(str).str.strip().str.upper().str[:1]
    df["trade_date"] = to_date(df["trade_date"])
    df["maturity_date"] = to_date(df["maturity_date"])
    df["dated_date"] = to_date(df["dated_date"]) if "dated_date" in df else pd.NaT
    df["par_traded"] = to_num(df["par_traded"])          # "MM+" masked rows become NaN
    df["price"] = to_num(df["price"])
    df["yield"] = to_num(df["yield"])
    df["coupon"] = to_num(df["coupon"])

    def drop(mask, why):
        drops[why] += int(mask.sum())
        return df[~mask]

    df = drop(~df["cusip"].str.fullmatch(r"[0-9A-Z]{9}"), "bad cusip")
    df = drop(~df["trade_type"].isin(["D", "P", "S"]), "trade type not D/P/S")
    df = drop(df["trade_date"].isna() | df["maturity_date"].isna(), "unparseable date")
    df = drop(df["par_traded"].isna(), "par masked (MM+) or missing")
    df = drop(df["price"].isna() | df["yield"].isna(), "missing price or yield")
    df = drop(df["coupon"].isna() | (df["coupon"] <= 0), "zero or missing coupon")
    if "when_issued" in df:
        df = drop(df["when_issued"].astype(str).str.upper().str.startswith("Y"), "when-issued")
    if "weighted_price" in df:
        df = drop(df["weighted_price"].astype(str).str.upper().str.startswith("Y"), "weighted-average price")
    df = drop((df["price"] < 40) | (df["price"] > 175), "price outside 40-175")
    df = drop((df["yield"] < -1) | (df["yield"] > 15), "yield outside -1 to 15")
    df = drop(df["maturity_date"] <= df["trade_date"], "matured before trade")
    df = drop(df.duplicated(subset=["cusip", "trade_date", "trade_type", "par_traded", "price", "yield"]), "exact duplicate")
    drops["_kept"] = len(df)
    drops["_in"] = n0
    # dated_date is only used for display; fill from maturity when missing
    df["dated_date"] = df["dated_date"].fillna(df["trade_date"] - pd.Timedelta(days=365))
    return df.reset_index(drop=True), drops


def add_reference(df: pd.DataFrame, ref_path: str | None, default_issue_size=20.0) -> tuple[pd.DataFrame, dict]:
    """Attach state/sector/rating/size/flags, from a reference file where possible, else parsed/defaults."""
    src = {}
    df = df.copy()
    desc = df["security_description"] if "security_description" in df else pd.Series("", index=df.index)
    df["state"] = parse_state(desc)
    df["sector"] = parse_sector(desc)
    df["rating"] = "NR"
    df["issue_size_mm"] = np.nan
    df["callable"] = False
    df["bank_qualified"] = False
    df["insured"] = False
    src["state"] = "parsed from description"
    src["sector"] = "parsed from description"
    src["rating"] = "default NR"
    src["issue_size_mm"] = f"default {default_issue_size}"
    src["callable"] = src["bank_qualified"] = src["insured"] = "default False"
    joined = 0
    if ref_path:
        ref = read_any(ref_path)
        cm = match_columns(ref.columns, REF_ALIASES)
        if "cusip" not in cm:
            raise SystemExit(f"{ref_path}: no cusip column found")
        r = pd.DataFrame({ours: ref[theirs] for ours, theirs in cm.items()})
        r["cusip"] = r["cusip"].astype(str).str.strip().str.upper()
        r = r.drop_duplicates("cusip").set_index("cusip")
        hit = df["cusip"].isin(r.index)
        joined = int(hit.sum())
        if "state" in r:
            v = df["cusip"].map(r["state"].astype(str).str.upper().str.strip())
            ok = v.notna() & v.str.fullmatch(r"[A-Z]{2}")
            df.loc[ok, "state"] = v[ok]; src["state"] = f"reference ({int(ok.sum()):,} rows), else parsed"
        if "sector" in r:
            v = df["cusip"].map(r["sector"])
            df.loc[v.notna(), "sector"] = parse_sector(v[v.notna()]).values
            src["sector"] = f"reference ({int(v.notna().sum()):,} rows, mapped to our 8 sectors), else parsed"
        if "rating" in r:
            v = df["cusip"].map(r["rating"])
            df.loc[v.notna(), "rating"] = norm_rating(v[v.notna()]).values
            src["rating"] = f"reference ({int(v.notna().sum()):,} rows), else NR"
        if "issue_size_mm" in r:
            v = to_num(df["cusip"].map(r["issue_size_mm"]))
            v = np.where(v > 20000, v / 1e6, v)          # dollars -> $MM when obviously in dollars
            v = pd.Series(v, index=df.index)
            df.loc[v.notna(), "issue_size_mm"] = v[v.notna()]
            src["issue_size_mm"] = f"reference ({int(v.notna().sum()):,} rows), else default"
        for flag in ("callable", "bank_qualified", "insured"):
            if flag in r:
                v = df["cusip"].map(r[flag])
                df.loc[v.notna(), flag] = to_bool(v[v.notna()]).values
                src[flag] = f"reference ({int(v.notna().sum()):,} rows), else False"
    df["issue_size_mm"] = df["issue_size_mm"].fillna(default_issue_size)
    src["_ref_rows_matched"] = joined
    return df, src


def write_report(path, paths, cm, drops, src, df):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = ["# MSRB load report", "", f"Inputs: {', '.join(os.path.basename(p) for p in paths)}", "",
             "## Column mapping", "", "| MuniPricer column | Source column |", "|---|---|"]
    lines += [f"| `{k}` | `{v}` |" for k, v in cm.items()]
    lines += ["", "## Cleaning", "", f"Rows in: {drops['_in']:,}. Rows kept: {drops['_kept']:,} "
              f"({drops['_kept'] / max(drops['_in'], 1):.1%}).", "", "| Dropped because | Rows |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in drops.items() if not k.startswith("_") and v]
    lines += ["", "## Reference fields", "", "| Field | Source |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in src.items() if not k.startswith("_")]
    lines += ["", "## Output summary", "",
              f"- {len(df):,} trades across {df['cusip'].nunique():,} CUSIPs, "
              f"{df['trade_date'].min().date()} to {df['trade_date'].max().date()}",
              f"- trade type mix: " + ", ".join(f"{k} {v:.0%}" for k, v in df['trade_type'].value_counts(normalize=True).items()),
              f"- state: {df['state'].nunique()} values, top: " + ", ".join(f"{k} {v:.0%}" for k, v in df['state'].value_counts(normalize=True).head(6).items()),
              f"- sector: " + ", ".join(f"{k} {v:.0%}" for k, v in df['sector'].value_counts(normalize=True).items()),
              f"- rating: " + ", ".join(f"{k} {v:.0%}" for k, v in df['rating'].value_counts(normalize=True).head(6).items()),
              f"- price median {df['price'].median():.2f}, yield median {df['yield'].median():.2f}%, "
              f"par median ${df['par_traded'].median():,.0f}", ""]
    if (df["state"] == "UNK").mean() > 0.2:
        lines.append(f"- WARNING: {(df['state'] == 'UNK').mean():.0%} of rows have no state. Provide --ref or extend STATE_PATTERNS.")
    if (df["rating"] == "NR").mean() > 0.9:
        lines.append("- WARNING: almost no ratings. The model will lean on description-derived sector and the trade history. "
                     "An EMMA or Mergent reference file with ratings would help a lot.")
    open(path, "w").write("\n".join(lines))


def run(paths: list[str], out="data/wrds_msrb.csv", ref=None, report="reports/load_report.md") -> pd.DataFrame:
    df, cm = load_raw(paths)
    df, drops = clean(df)
    df, src = add_reference(df, ref)
    df = df[OUT_COLS].sort_values(["trade_date", "cusip"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    write_report(report, paths, cm, drops, src, df)
    print(f"kept {drops['_kept']:,} of {drops['_in']:,} rows -> {out}; report in {report}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="raw WRDS/MSRB export(s): csv, csv.gz or parquet; globs ok")
    ap.add_argument("--ref", help="bond reference CSV keyed by cusip (EMMA, Mergent, ...)")
    ap.add_argument("--out", default="data/wrds_msrb.csv")
    ap.add_argument("--report", default="reports/load_report.md")
    a = ap.parse_args()
    files = sorted(sum([glob.glob(p) for p in a.inputs], []))
    if not files:
        sys.exit("no input files matched")
    run(files, a.out, a.ref, a.report)
