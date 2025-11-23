# processing/SPX_data_download.py

import yfinance as yf
import pandas as pd
import os

spx_raw = yf.download("^SPX", start="1996-01-04", end="2024-01-01", auto_adjust=False)

print("COLUMNS:", spx_raw.columns)

if isinstance(spx_raw.columns, pd.MultiIndex):
    spx_raw.columns = [c[0] for c in spx_raw.columns]

spx = spx_raw.reset_index()[["Date", "Close"]]

os.makedirs("data/spx_data", exist_ok=True)
spx.to_csv("data/spx_data/spx.csv", index=False)

print("Saved to data/spx_data/spx.csv")
