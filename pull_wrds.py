"""
Pull MSRB municipal bond trades from WRDS into data/raw/, ready for load_msrb.py.

Run this on your own machine (it needs your WRDS login and Duo approval):

  pip install -r requirements-data.txt
  python pull_wrds.py --list                                   # show MSRB tables and their columns
  python pull_wrds.py --start 2025-11-01 --end 2026-10-31      # pull 12 months -> data/raw/msrb_<start>_<end>.csv
  python pull_wrds.py --start 2026-09-01 --end 2026-09-30 --sample 5000   # quick test pull

Then:
  python load_msrb.py data/raw/msrb_*.csv
  python train.py data/wrds_msrb.csv

How it works
  - Connects with the official `wrds` Python package. The first time it asks for your
    WRDS username and password, approves via Duo, and offers to save a ~/.pgpass file
    so later runs do not ask again.
  - Finds the MSRB trade table in the `msrb` library. If there is more than one
    candidate it lists them and asks you to choose with --table.
  - Finds the trade-date column by name, then pulls one month at a time (the full
    table is large) and appends to a single CSV. Every column is kept;
    load_msrb.py picks the ones it needs.

Raw WRDS data stays in data/raw/, which is gitignored. Do not commit it or publish it:
WRDS terms do not allow redistribution.

Author: Anubhav Awasthi
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from load_msrb import TRADE_ALIASES, _norm

LIBRARY = "msrb"


def pick_table(tables: list[str], requested: str | None = None) -> str:
    """Choose the trade table. Prefer an explicit --table, else a single name containing 'trade' or 'msrb'."""
    if requested:
        if requested not in tables:
            raise SystemExit(f"table '{requested}' not in library {LIBRARY}: {tables}")
        return requested
    if len(tables) == 1:
        return tables[0]
    hits = [t for t in tables if "trade" in t.lower()] or [t for t in tables if "msrb" in t.lower()]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"Several MSRB tables found: {tables}. Rerun with --table <name> "
                     f"(run with --list to see each table's columns).")


def find_date_col(columns: list[str]) -> str:
    wanted = {_norm(a) for a in ["trade_date"] + TRADE_ALIASES["trade_date"]}
    for c in columns:
        if _norm(c) in wanted:
            return c
    raise SystemExit(f"No trade-date column found among {columns}. Pass --date-col <name>.")


def month_windows(start: str, end: str) -> list[tuple[str, str]]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    edges = list(pd.date_range(s, e, freq="MS"))
    if not edges or edges[0] != s:
        edges = [s] + edges
    edges.append(e + pd.Timedelta(days=1))
    return [(str(a.date()), str(b.date())) for a, b in zip(edges[:-1], edges[1:]) if a < b]


def connect(username: str | None):
    try:
        import wrds
    except ImportError:
        raise SystemExit("The wrds package is not installed. Run: pip install -r requirements-data.txt")
    return wrds.Connection(wrds_username=username) if username else wrds.Connection()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list MSRB tables and columns, then exit")
    ap.add_argument("--start", help="first trade date, YYYY-MM-DD")
    ap.add_argument("--end", help="last trade date, YYYY-MM-DD")
    ap.add_argument("--table", help="MSRB table name if auto-detection is ambiguous")
    ap.add_argument("--date-col", help="trade date column name if auto-detection fails")
    ap.add_argument("--sample", type=int, help="stop after this many rows (for a quick test)")
    ap.add_argument("--username", help="WRDS username (otherwise you are prompted)")
    ap.add_argument("--out", help="output CSV (default data/raw/msrb_<start>_<end>.csv)")
    a = ap.parse_args()

    db = connect(a.username)
    tables = sorted(db.list_tables(library=LIBRARY))
    if not tables:
        raise SystemExit(f"No tables visible in the '{LIBRARY}' library. Check that your WRDS account includes MSRB.")

    if a.list:
        for t in tables:
            cols = db.describe_table(library=LIBRARY, table=t)
            print(f"\n{LIBRARY}.{t}")
            print(cols[["name", "type"]].to_string(index=False))
        return

    if not (a.start and a.end):
        raise SystemExit("Pass --start and --end (YYYY-MM-DD), or --list to explore the tables.")

    table = pick_table(tables, a.table)
    columns = list(db.describe_table(library=LIBRARY, table=table)["name"])
    date_col = a.date_col or find_date_col(columns)
    out = a.out or f"data/raw/msrb_{a.start}_{a.end}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if os.path.exists(out):
        os.remove(out)

    print(f"pulling {LIBRARY}.{table} on {date_col} from {a.start} to {a.end} -> {out}")
    total = 0
    for lo, hi in month_windows(a.start, a.end):
        limit = f" LIMIT {int(a.sample - total)}" if a.sample else ""
        sql = (f"SELECT * FROM {LIBRARY}.{table} "
               f"WHERE {date_col} >= %(lo)s AND {date_col} < %(hi)s{limit}")
        df = db.raw_sql(sql, params={"lo": lo, "hi": hi})
        df.to_csv(out, mode="a", header=total == 0, index=False)
        total += len(df)
        print(f"  {lo} to {hi}: {len(df):,} rows (total {total:,})")
        if a.sample and total >= a.sample:
            break
    db.close()
    if total == 0:
        sys.exit("No rows returned. Check the date range (WRDS updates MSRB with a lag) or run --list.")
    print(f"\ndone: {total:,} rows in {out}\nnext: python load_msrb.py {out}")


if __name__ == "__main__":
    main()
