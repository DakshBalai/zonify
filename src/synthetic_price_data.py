"""
synthetic_price_data.py
Generates realistic OHLCV candle data with DELIBERATELY engineered
market structure (clean uptrends, downtrends, and reversals) so the
structure-detection engine (swings, BOS, CHoCH, IDM) can be tested
against data where we already know the "correct" answer.

This is a development/testing tool only -- the real project uses actual
NSE data via yfinance (see data_loader.py). Not part of the final app.
"""

import numpy as np
import pandas as pd


def generate_trending_candles(n=300, start_price=100.0, seed=7):
    rng = np.random.default_rng(seed)

    legs = []
    remaining = n
    direction = 1
    while remaining > 0:
        length = int(rng.integers(15, 35))
        legs.append((direction, min(length, remaining)))
        remaining -= length
        direction *= -1

    opens, highs, lows, closes = [], [], [], []
    price = start_price

    for direction, length in legs:
        for i in range(length):
            is_pullback = rng.random() < 0.3
            step_dir = -direction if is_pullback else direction
            body = abs(rng.normal(0.6, 0.3)) * step_dir
            wick_extra = abs(rng.normal(0.25, 0.15))

            open_price = price
            close_price = open_price + body
            if step_dir > 0:
                high = close_price + wick_extra
                low = open_price - wick_extra * 0.4
            else:
                high = open_price + wick_extra * 0.4
                low = close_price - wick_extra

            opens.append(open_price)
            closes.append(close_price)
            highs.append(max(high, open_price, close_price))
            lows.append(min(low, open_price, close_price))
            price = close_price

    dates = pd.date_range("2023-01-01", periods=len(opens), freq="D")
    df = pd.DataFrame({
        "date": dates, "open": opens, "high": highs, "low": lows, "close": closes,
    })
    df["volume"] = rng.integers(100000, 900000, len(df))
    return df


if __name__ == "__main__":
    from pathlib import Path

    df = generate_trending_candles()
    out_path = Path(__file__).resolve().parents[1] / "data" / "synthetic_test_candles.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} candles -> {out_path}")
    print(df.head())
