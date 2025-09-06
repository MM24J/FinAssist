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

def budget_summary(rep: pd.DataFrame, topn=5, month_label="(month)"):
    """Return formatted budget summary instead of printing."""
    over = rep[rep["Variance"] > 0].sort_values("Variance", ascending=False).head(topn)
    under = rep[rep["Variance"] < 0].sort_values("Variance", ascending=True).head(topn)

    output = [f"--- Budget Summary for {month_label} ---\n"]
    
    if len(over) > 0:
        output.append(f"Top {len(over)} over-budget categories:")
        for r in over.itertuples():
            output.append(f"  • {r.Category}: +${r.Variance:,.2f} "
                         f"(Actual ${r.Actual:,.2f} vs Budget ${r.Budget:,.2f})")
    else:
        output.append("No over-budget categories found.")

    if len(under) > 0:
        output.append(f"\nTop {len(under)} under-budget categories:")
        for r in under.itertuples():
            output.append(f"  • {r.Category}: ${r.Variance:,.2f} "
                         f"(Actual ${r.Actual:,.2f} vs Budget ${r.Budget:,.2f})")
    else:
        output.append("\nNo under-budget categories found.")
    
    return "\n".join(output)

def month_report(month: str) -> pd.DataFrame:
    """Analyze actual transaction data for the given month."""
    try:
        from .data import tx
        
        # Filter transactions for the specified month
        month_data = tx[tx['Month'] == month] if not tx.empty else pd.DataFrame()
        
        if month_data.empty:
            print(f"No transaction data found for {month}")
            return pd.DataFrame(columns=["Category", "Budget", "Actual", "Variance"])
        
        # Group by category and sum spending (use 'Spend' column from your data.py)
        actual_spending = month_data.groupby('Category')['Spend'].sum().reset_index()
        actual_spending.columns = ['Category', 'Actual']
        
        # Simple budget defaults (you can load Budget.csv later)
        budget_defaults = {
            'Restaurants': 200, 'Coffee': 50, 'Internet': 75, 
            'Groceries': 300, 'Gas': 100, 'Shopping': 250
        }
        
        # Create budget data for categories found in transactions
        categories = actual_spending['Category'].unique()
        budget_data = pd.DataFrame([
            {'Category': cat, 'Budget': budget_defaults.get(cat, 100)} 
            for cat in categories
        ])
        
        # Merge and calculate variance
        report = pd.merge(budget_data, actual_spending, on='Category', how='outer')
        report['Budget'] = report['Budget'].fillna(100)
        report['Actual'] = report['Actual'].fillna(0)
        report['Variance'] = report['Actual'] - report['Budget']
        
        return report[report['Actual'] > 0]  # Only show categories with spending
        
    except Exception as e:
        print(f"Error in budget analysis: {e}")
        return pd.DataFrame(columns=["Category", "Budget", "Actual", "Variance"])