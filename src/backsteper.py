from __future__ import annotations
import pandas as pd
import numpy as np
from .rules import compute_signals, RuleParams
from .position_sizing import risk_position_size
from .metrics import drawdown

class Backtester:
    """
    Простой векторный бэктестер:
    - вход по сигналу на следующем баре по цене open[t+1]
    - фиксированный стоп/тейк по ATR-мультипликаторам (упрощённо, без intra-bar логики)
    - комиссия как процент от оборота (односторонне)
    """
    def __init__(self, df: pd.DataFrame, cfg: dict):
        self.df = df.copy()
        self.cfg = cfg

    def _signals(self) -> pd.DataFrame:
        rp = RuleParams(
            score_threshold=self.cfg["rules"]["score_threshold"],
            fractal_lookback=self.cfg["rules"]["fractal_lookback"],
            ema_filter=self.cfg["indicators"]["ema_filter"],
            use_trend_filter=self.cfg["rules"]["use_trend_filter"],
            use_obv=self.cfg.get("indicators", {}).get("obv", {}).get("enabled", True),
        )
        return compute_signals(self.df, self.cfg["indicators"], rp)

    def run(self) -> dict:
        sig = self._signals()
        # позиция: -1/0/1 (берём последний сигнал)
        desired = np.where(sig["long_entry"], 1, np.where(sig["short_entry"], -1, 0))
        position = pd.Series(desired, index=self.df.index)

        # размер позиции по риску
        size = risk_position_size(
            equity=self.cfg["risk"]["account_equity"],
            risk_per_trade=self.cfg["risk"]["risk_per_trade"],
            atr=sig["atr"],
            stop_atr_mult=self.cfg["risk"]["stop_atr_mult"],
            tick_value=1.0
        ).clip(upper=1.0)  # ограничим верх

        # доходность по закрытиям (можно заменить на open->open)
        rets = self.df["close"].pct_change().fillna(0.0)
        pnl = rets * position.shift(1).fillna(0.0) * size.shift(1).fillna(0.0)

        # комиссии
        commission = float(self.cfg["data"].get("commission_perc", 0.0))
        turnover = (position.diff().abs().fillna(0.0))  # смена позиции => «торговля»
        costs = turnover * commission
        net = pnl - costs

        equity = (1.0 + net).cumprod()
        return {
            "equity": equity,
            "returns": net,
            "position": position,
            "signals": sig,
            "drawdown": drawdown(equity),
        }