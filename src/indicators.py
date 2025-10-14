from __future__ import annotations
import pandas as pd
import numpy as np

def _midprice(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"]) / 2.0

def smma(s: pd.Series, period: int) -> pd.Series:
    """Smoothed moving average (как у Вильямса)."""
    s = s.astype(float)
    out = s.copy()
    alpha = 1.0 / period
    for i in range(1, len(s)):
        out.iat[i] = out.iat[i-1]*(1-alpha) + s.iat[i]*alpha
    return out

def alligator(
    df: pd.DataFrame,
    jaw: int = 13, teeth: int = 8, lips: int = 5,
    shift_jaw: int = 8, shift_teeth: int = 5, shift_lips: int = 3,
    use_smma: bool = True
) -> pd.DataFrame:
    mp = _midprice(df)
    if use_smma:
        j = smma(mp, jaw).shift(shift_jaw)
        t = smma(mp, teeth).shift(shift_teeth)
        l = smma(mp, lips).shift(shift_lips)
    else:
        j = mp.ewm(span=jaw, adjust=False).mean().shift(shift_jaw)
        t = mp.ewm(span=teeth, adjust=False).mean().shift(shift_teeth)
        l = mp.ewm(span=lips, adjust=False).mean().shift(shift_lips)
    return pd.DataFrame({"jaw": j, "teeth": t, "lips": l})

def fractals(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Классические фракталы Б. Уильямса (center=True)."""
    hh = df["high"].rolling(lookback, center=True).max()
    ll = df["low"].rolling(lookback, center=True).min()
    up = df["high"].eq(hh)
    dn = df["low"].eq(ll)
    return pd.DataFrame({"fractal_up": up, "fractal_dn": dn})

def ao(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.Series:
    mp = _midprice(df)
    return mp.rolling(fast).mean() - mp.rolling(slow).mean()

def ac(ao_s: pd.Series, smooth: int = 5) -> pd.Series:
    return ao_s - ao_s.rolling(smooth).mean()

def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - prev_close).abs(),
                               (df["low"] - prev_close).abs()))
    return tr.rolling(period).mean()

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"].fillna(0.0)).cumsum()