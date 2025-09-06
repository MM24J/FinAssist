import re
from datetime import datetime
import pandas as pd

def human_month(ym: str) -> str:
    try:
        return datetime.strptime(ym, "%Y-%m").strftime("%B %Y")
    except Exception:
        return ym

def parse_month(text: str):
    m = re.search(r"(20\d{2}-\d{2})", text)
    return m.group(1) if m else None

def budget_summary(rep: pd.DataFrame, top=5, month_label="(month)"):
    over = rep[rep["Variance"] > 0].sort_values("Variance", ascending=False).head(top)
    under = rep[rep["Variance"] < 0].sort_values("Variance", ascending=True).head(top)

    print(f"--- Budget Summary for {month_label} ---\n")
    print(f"Top {len(over)} over-budget categories:")
    for r in over.itertuples():
        print(f"  - {r.Category}: +${r.Variance:,.2f} "
              f"(Actual ${r.Actual:,.2f} vs Budget ${r.Budget:,.2f})")

    print(f"\nTop {len(under)} under-budget categories:")
    for r in under.itertuples():
        print(f"  - {r.Category}: ${r.Variance:,.2f} "
              f"(Actual ${r.Actual:,.2f} vs Budget ${r.Budget:,.2f})")

def month_report(month: str) -> pd.DataFrame:
    # stub: replace with real logic from notebook
    data = {"Category": ["Restaurants", "Internet"],
            "Budget": [150, 75],
            "Actual": [172.34, 75],
            "Variance": [22.34, 0]}
    return pd.DataFrame(data)
