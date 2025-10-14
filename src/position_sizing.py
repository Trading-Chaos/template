from __future__ import annotations
import pandas as pd

def risk_position_size(
    equity: float,
    risk_per_trade: float,
    atr: pd.Series,
    stop_atr_mult: float,
    tick_value: float = 1.0
) -> pd.Series:
    """Размер позиции по риску: риск_в_деньгах / (стоп_в_деньгах)."""
    risk_cash = equity * float(risk_per_trade)
    stop_cash = atr * float(stop_atr_mult) * float(tick_value)
    size = (risk_cash / stop_cash).clip(lower=0)
    return size