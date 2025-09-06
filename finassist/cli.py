import os, re
import matplotlib.pyplot as plt
from finassist.rag import rag_answer
from .budget import parse_month, month_report, budget_summary, human_month
from .stocks import parse_tickers, parse_period, stock_summary, summarize_returns
from .advice import looks_like_advice, looks_like_budget, looks_like_invest
from .data import tx

def ask(q: str):
    ql = q.lower()
    
    # Debug classification
    print(f"DEBUG CLI: Question='{q}'")
    print(f"DEBUG CLI: looks_like_advice={looks_like_advice(ql)}")
    print(f"DEBUG CLI: looks_like_budget={looks_like_budget(ql)}")
    print(f"DEBUG CLI: looks_like_invest={looks_like_invest(ql)}")
    
    # Check Budget FIRST (most specific)
    if looks_like_budget(ql):
        print("DEBUG CLI: Going to Budget")
        month = parse_month(q) or (tx.groupby("Month")["Spend"].sum().pipe(lambda s: s[s>0]).index.max())
        rep = month_report(month)
        m = re.search(r"top\s+(\d+)", ql); topn = int(m.group(1)) if m else 5
        return budget_summary(rep, topn=topn, month_label=human_month(month))
    
    # Check Investment (specific)
    if looks_like_invest(ql):
        print("DEBUG CLI: Going to Investment")
        tickers = parse_tickers(q) or ["SPY"]
        period = parse_period(q or "6mo")
        results = stock_summary(tickers, period)
        return summarize_returns(results, period)
    
    # Check Advice/RAG LAST (most general)
    if looks_like_advice(ql):
        print("DEBUG CLI: Going to RAG")
        return rag_answer(q, k=3)
    
    return "Try:\n • Where am I over budget in 2019-09?\n • Compare AAPL and MSFT over 1y\n • How can I reduce restaurant spending?"