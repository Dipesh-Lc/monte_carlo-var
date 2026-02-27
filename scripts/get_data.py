from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "JPM", "TLT"]
WEIGHTS = np.array([0.18, 0.18, 0.18, 0.18, 0.18, 0.10], dtype=float)
WEIGHTS = WEIGHTS / WEIGHTS.sum()

START_DATE = "2018-01-01"
END_DATE = None          # e.g. "2025-12-31" or None
INTERVAL = "1d"

DATA_DIR = Path("data")
PRICES_PATH = DATA_DIR / "prices.csv"
RETURNS_PATH = DATA_DIR / "returns.csv"
WEIGHTS_JSON_PATH = Path("weights.json")


def download_prices(tickers: list[str], start: str, end: str | None, interval: str = "1d") -> pd.DataFrame:
    """
    Download adjusted close prices via yfinance and return a DataFrame:
      index = Date
      columns = tickers
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance not installed. Run: pip install yfinance") from e

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if df.empty:
        raise ValueError("Downloaded data is empty. Check tickers/date range.")

    # Handle MultiIndex vs single ticker
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            prices = df["Close"].copy()
        else:
            prices = df.xs(df.columns.levels[0][0], axis=1, level=0)
    else:
        # single ticker format
        if "Close" not in df.columns:
            raise ValueError("Unexpected yfinance format: 'Close' not found.")
        prices = df["Close"].to_frame(name=tickers[0])

    prices.index.name = "Date"
    prices = prices.dropna(how="all")
    return prices


def make_returns(prices: pd.DataFrame, log_returns: bool = False) -> pd.DataFrame:
    """
    Convert price series to returns.
    """
    prices = prices.dropna()
    if prices.shape[1] < 1:
        raise ValueError("No price columns found.")
    if prices.shape[0] < 5:
        raise ValueError("Need at least 5 rows of prices to compute returns.")

    if log_returns:
        rets = np.log(prices).diff().dropna()
    else:
        rets = prices.pct_change().dropna()

    rets.index.name = "Date"
    return rets


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Print weights table 
    wdf = pd.DataFrame({"Ticker": TICKERS, "Weight": WEIGHTS})
    print("\nPortfolio:")
    print(wdf.to_string(index=False))
    print(f"Sum weights: {WEIGHTS.sum():.6f}")

    # Download prices
    prices = download_prices(TICKERS, start=START_DATE, end=END_DATE, interval=INTERVAL)
    prices.to_csv(PRICES_PATH)
    print(f"\nSaved prices:  {PRICES_PATH}  (rows={prices.shape[0]}, cols={prices.shape[1]})")

    # Compute returns
    returns = make_returns(prices, log_returns=False)
    returns.to_csv(RETURNS_PATH)
    print(f"Saved returns: {RETURNS_PATH} (rows={returns.shape[0]}, cols={returns.shape[1]})")

    # Write weights.json 
    weights_dict = {t: float(w) for t, w in zip(TICKERS, WEIGHTS)}
    WEIGHTS_JSON_PATH.write_text(json.dumps(weights_dict, indent=2), encoding="utf-8")
    print(f"Saved weights: {WEIGHTS_JSON_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()