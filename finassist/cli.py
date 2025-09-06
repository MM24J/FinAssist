import os, re
import matplotlib.pyplot as plt
from finassist.rag import rag_answer
from .budget import parse_month, month_report, budget_summary, human_month
from .stocks import parse_tickers, parse_period, stock_summary, summarize_returns
from .advice import looks_like_advice, looks_like_budget, looks_like_invest
from .data import tx

def ask(q: str):
    ql = q.lower()
    
    # Advice/RAG
    if looks_like_advice(ql):
        return rag_answer(q, k=3)
    
    # Budget
    if looks_like_budget(ql):
        month = parse_month(q) or (tx.groupby("Month")["Spend"].sum().pipe(lambda s: s[s>0]).index.max())
        rep = month_report(month)
        m = re.search(r"top\s+(\d+)", ql); topn = int(m.group(1)) if m else 5
        return budget_summary(rep, topn=topn, month_label=human_month(month))
    
    # Stocks
    if looks_like_invest(ql):
        tickers = parse_tickers(q) or ["SPY"]
        period = parse_period(q or "6mo")
        results = stock_summary(tickers, period)
        return summarize_returns(results, period)
    
    return "Try:\n • Where am I over budget in 2019-09?\n • Compare AAPL and MSFT over 1y\n • How can I reduce restaurant spending?"