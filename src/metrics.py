from __future__ import annotations
import pandas as pd
import numpy as np

def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0

def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())

def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    total_ret = float(equity.iloc[-1] / equity.iloc[0])
    years = len(equity) / periods_per_year
    return total_ret**(1/years) - 1 if years > 0 else 0.0

def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    ex = returns - rf/periods_per_year
    mu, sigma = ex.mean(), ex.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return float(np.sqrt(periods_per_year) * mu / sigma)

def profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return float(gains / losses) if losses > 0 else float("inf")

def mar(equity: pd.Series, periods_per_year: int = 252) -> float:
    c = cagr(equity, periods_per_year)
    mdd = abs(max_drawdown(equity))
    return c / mdd if mdd > 0 else 0.0