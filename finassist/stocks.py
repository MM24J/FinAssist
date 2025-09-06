import re
import pandas as pd

def parse_tickers(text: str):
    toks = re.findall(r"\b[A-Z]{1,5}\b", text)
    ignore = {"USD", "ETF", "YTD", "YOY", "Q", "VS"}
    return [t for t in toks if t not in ignore] or None

def parse_period(text: str, default="1y"):
    q = text.lower()
    if "ytd" in q:
        return "ytd"
    m = re.search(r"\b(\d+)\s*([dwmy])\b", q)
    if not m:
        return default
    n, u = m.group(1), m.group(2)
    if u == "m":  # yfinance uses "mo"
        u = "mo"
    return f"{n}{u}"

def stock_summary(tickers, period):
    # stub: return dummy data frame
    return pd.DataFrame([{
        "ticker": t,
        "return_pct": 10.0,
        "start": 100.0,
        "end": 110.0,
        "last_close": 110.0
    } for t in tickers])

def summarize_returns(results, period_label="1y"):
    if results.empty:
        print("No data available.")
        return
    print(f"\n--- Investment Summary for {period_label} ---\n")
    for r in results.itertuples(index=False):
        print(f"{r.ticker}: {r.return_pct:,.2f}% "
              f"(Start ${r.start:,.2f} → End ${r.end:,.2f}; Last ${r.last_close:,.2f})")
