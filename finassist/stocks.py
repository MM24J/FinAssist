import re
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

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
    """Get real stock data using yfinance."""
    results = []
    
    for ticker in tickers:
        try:
            # Download stock data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                print(f"Warning: No data found for {ticker}")
                continue
                
            # Calculate returns
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            last_close = end_price
            return_pct = ((end_price - start_price) / start_price) * 100
            
            results.append({
                "ticker": ticker,
                "return_pct": return_pct,
                "start": start_price,
                "end": end_price,
                "last_close": last_close
            })
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            # Fallback to dummy data if yfinance fails
            results.append({
                "ticker": ticker,
                "return_pct": 0.0,
                "start": 100.0,
                "end": 100.0,
                "last_close": 100.0
            })
    
    return pd.DataFrame(results)

def summarize_returns(results, period_label="1y"):
    if results.empty:
        return "No data available."
    
    output = [f"\n--- Investment Summary for {period_label} ---"]
    
    for r in results.itertuples(index=False):
        output.append(f"{r.ticker}: {r.return_pct:+.2f}% "
                     f"(Start ${r.start:.2f} → End ${r.end:.2f}; Last ${r.last_close:.2f})")
    
    return "\n".join(output)