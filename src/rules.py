from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from .indicators import alligator, fractals, ao, ac, ema, atr, obv

@dataclass
class RuleParams:
    score_threshold: int = 2
    fractal_lookback: int = 5
    ema_filter: int = 200
    use_trend_filter: bool = True
    use_obv: bool = True

def compute_signals(df: pd.DataFrame, ind_cfg: dict, rules: RuleParams) -> pd.DataFrame:
    alli = alligator(df, **ind_cfg.get("alligator", {}))
    fr = fractals(df, rules.fractal_lookback)
    ao_s = ao(df, **ind_cfg.get("ao", {}))
    ac_s = ac(ao_s, **ind_cfg.get("ac", {}))
    ema_f = ema(df["close"], rules.ema_filter)
    atr_s = atr(df, **ind_cfg.get("atr", {}))
    obv_s = obv(df) if rules.use_obv and ind_cfg.get("obv", {}).get("enabled", True) else None

    # условия long
    gator_up = (alli["lips"] > alli["teeth"]) & (alli["teeth"] > alli["jaw"])
    ao_green = ao_s > 0
    ao_rising = ao_s.diff() > 0
    ac_green = ac_s > 0
    obv_rising = (obv_s.diff() > 0) if obv_s is not None else False
    trend_ok = (df["close"] > ema_f) if rules.use_trend_filter else True

    # фрактальные уровни (последние зафиксированные)
    res_up = df["high"].where(fr["fractal_up"]).ffill()
    res_dn = df["low"].where(fr["fractal_dn"]).ffill()
    breakout_up = df["close"] > res_up.shift(1)
    breakout_dn = df["close"] < res_dn.shift(1)

    score_long = (
        gator_up.astype(int) + breakout_up.astype(int) +
        ao_green.astype(int) + ao_rising.astype(int) +
        ac_green.astype(int) + (obv_rising.astype(int) if obv_s is not None else 0)
    )

    # условия short
    gator_dn = (alli["lips"] < alli["teeth"]) & (alli["teeth"] < alli["jaw"])
    ao_red = ao_s < 0
    ao_falling = ao_s.diff() < 0
    ac_red = ac_s < 0
    obv_falling = (obv_s.diff() < 0) if obv_s is not None else False

    score_short = (
        gator_dn.astype(int) + breakout_dn.astype(int) +
        ao_red.astype(int) + ao_falling.astype(int) +
        ac_red.astype(int) + (obv_falling.astype(int) if obv_s is not None else 0)
    )

    long_entry = (score_long >= rules.score_threshold) & (trend_ok if isinstance(trend_ok, pd.Series) else True)
    short_entry = (score_short >= rules.score_threshold) & (~trend_ok if rules.use_trend_filter else True)

    return pd.DataFrame({
        "score_long": score_long,
        "score_short": score_short,
        "long_entry": long_entry.fillna(False),
        "short_entry": short_entry.fillna(False),
        "ema_filter": ema_f,
        "atr": atr_s
    }, index=df.index)