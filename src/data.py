import numpy as np
import pandas as pd
import yfinance as yf

def download_prices(tickers, start="2018-01-01", end=None):
    """
    Download adjusted prices (auto_adjust=True) from Yahoo Finance.
    Returns DataFrame: rows=dates, cols=tickers.
    """
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    # If multiple tickers, yfinance returns MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"].copy()
    else:
        prices = df[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="any")
    return prices

def compute_returns(prices, method="log"):
    """
    method='log' or 'simple'
    """
    if method == "log":
        rets = np.log(prices / prices.shift(1))
    elif method == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")

    return rets.dropna()