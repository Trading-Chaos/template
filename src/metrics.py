from __future__ import annotations
import pandas as pd
import numpy as np
'''
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

'''
def summarize(bt: pd.DataFrame, trades: list):
    res = {}
    res["final_equity"] = float(bt["Equity"].iloc[-1])

    # --- устойчивый расчёт CAGR ---
    dt = pd.to_datetime(bt["DateTime"], errors="coerce")
    # используем первые и последние валидные значения
    valid = dt.notna()
    if valid.sum() >= 2:
        t0 = dt[valid].iloc[0]
        t1 = dt[valid].iloc[-1]
        seconds = (t1 - t0).total_seconds()
        years = max(seconds / (365.25 * 24 * 3600), 1e-9)
        res["CAGR"] = res["final_equity"] ** (1 / years) - 1
    else:
        # fallback: если времени нет, оценим по числу баров (без календарной привязки)
        bars = len(bt)
        # допустим дневные данные: 252 бара в год; при другом ТФ поменяй коэффициент
        years = max(bars / 252.0, 1e-9)
        res["CAGR"] = res["final_equity"] ** (1 / years) - 1

    # --- макс. просадка ---
    roll_max = bt["Equity"].cummax()
    dd = bt["Equity"] / roll_max - 1.0
    res["MaxDD"] = float(dd.min())

    # --- сделки ---
    res["NumTrades"] = len(trades)
    if trades:
        wins, rets = 0, []
        for t in trades:
            r = (t["exit_price"]/t["entry_price"] - 1.0) if t["side"]=="LONG" else (t["entry_price"]/t["exit_price"] - 1.0)
            rets.append(r)
            if r > 0: wins += 1
        res["WinRate"] = wins / len(trades)
        res["AvgTradeRet"] = float(np.mean(rets))
        res["MedianTradeRet"] = float(np.median(rets))
    else:
        res["WinRate"] = res["AvgTradeRet"] = res["MedianTradeRet"] = np.nan
    return res
