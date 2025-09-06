import pandas as pd
import numpy as np
import os

# Assume your repo has a data/ folder with CSVs
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

try:
    tx = pd.read_csv(os.path.join(DATA_DIR, "personal_transactions.csv"), parse_dates=["Date"])
except FileNotFoundError:
    # fallback dummy data
    tx = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=5),
        "Amount": [-50, -20, -30, 1000, -15],
        "Transaction Type": ["debit", "debit", "debit", "credit", "debit"],
        "Category": ["Restaurants", "Internet", "Restaurants", "Paycheck", "Coffee"]
    })

tx["Spend"] = np.where(tx["Transaction Type"].str.lower() == "debit", tx["Amount"].astype(float).abs(), 0.0)
tx["Income"] = np.where(tx["Transaction Type"].str.lower() == "credit", tx["Amount"].astype(float), 0.0)
tx["Month"] = pd.to_datetime(tx["Date"]).dt.to_period("M").astype(str)
