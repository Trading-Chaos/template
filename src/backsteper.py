'''бэкстепер'''

# === Breakout-вход + добор 30% на первом подтверждённом фрактале + два жёстких стопа (2% на вход и 2% на добор)
# Комиссия 0.035% per side (≈0.07% за круг). Выход по flip аллигатора или по ближайшему стопу.

# -------- Параметры --------
START_EQUITY       = 100_000.0
STOP_RISK_PCT_MAIN = 0.02           # 2% от equity на момент ВХОДА
STOP_RISK_PCT_ADD  = 0.02           # 2% от equity на момент ДОБОРА
EXPOSURE_FRACTION  = 1.0            # доля капитала на главный вход
ADDON_RATIO        = 0.30           # 30% от главного объёма
MULTIPLIER         = 1.0
EXEC_EXIT          = "open_next"    # "open_next" | "close_signal"
EXEC_ADDON         = "open_next"    # как исполняем добор (обычно open следующего бара)
FEE_SIDE_RATE      = 0.00035

def fee_cash(notional): return float(notional) * float(FEE_SIDE_RATE)

# -------- Подготовка данных --------
df_bt = df.copy().sort_values("DateTime").reset_index(drop=True)
df_bt["DateTime"] = pd.to_datetime(df_bt["DateTime"], errors="coerce")
df_bt = df_bt[df_bt["DateTime"].notna()].reset_index(drop=True)

req = ["Open","High","Low","Close","EntrySignal",
       "Alligator_Jaw","Alligator_Teeth","Alligator_Lips","Fractal_Up","Fractal_Down"]
missing = [c for c in req if c not in df_bt.columns]
if missing: raise ValueError(f"Нет колонок: {missing}")

# подтверждённые фракталы (подтверждение через 2 бара)
if "Fractal_Up_conf" not in df_bt.columns:
    df_bt["Fractal_Up_conf"] = df_bt["Fractal_Up"].shift(2).fillna(0).astype(int)
if "Fractal_Down_conf" not in df_bt.columns:
    df_bt["Fractal_Down_conf"] = df_bt["Fractal_Down"].shift(2).fillna(0).astype(int)

# --- Аллигатор и Exit по flip ---
bull = (df_bt["Alligator_Lips"] > df_bt["Alligator_Teeth"]) & (df_bt["Alligator_Teeth"] > df_bt["Alligator_Jaw"])
bear = (df_bt["Alligator_Jaw"]  > df_bt["Alligator_Teeth"]) & (df_bt["Alligator_Teeth"] > df_bt["Alligator_Lips"])
bull_flip = (bull & ~bull.shift(1).fillna(False))
bear_flip = (bear & ~bear.shift(1).fillna(False))

df_bt["ExitSignal"] = 0; _pos=0
for i in range(len(df_bt)):
    if _pos==1 and bear_flip.iloc[i]: df_bt.at[i,"ExitSignal"]=1; _pos=0
    elif _pos==-1 and bull_flip.iloc[i]: df_bt.at[i,"ExitSignal"]=-1; _pos=0
    if _pos==0 and df_bt.at[i,"ExitSignal"]==0:
        s = int(df_bt.at[i,"EntrySignal"])
        if s!=0: _pos=s

def px_exit(i):
    if EXEC_EXIT=="close_signal" or i+1>=len(df_bt): return float(df_bt.at[i,"Close"])
    return float(df_bt.at[i+1,"Open"])

def px_addon(i):
    if EXEC_ADDON=="close_signal" or i+1>=len(df_bt): return float(df_bt.at[i,"Close"])
    return float(df_bt.at[i+1,"Open"])

# -------- Бэктест --------
equity = START_EQUITY
equity_curve, trades = [], []

pos=0; in_trade=False
units_main=0.0; units_add=0.0
entry_i=None; entry_px=None; entry_eq=None
fee_in_main=0.0; fee_in_add=0.0
# два независимых стоп-уровня (главный и добор)
stop_px_main=None; stop_px_add=None
prev_close=float(df_bt.at[0,"Close"])

# pending стоп-заявка на ВХОД (breakout) — активна только на следующем баре
pending = None  # dict(side, level, idx)
# «маяк» на добор: ждём первый подтверждённый фрактал после входа
addon_done=False
wait_addon=False  # True после входа, пока не доберём

for i in range(len(df_bt)):
    op = float(df_bt.at[i,"Open"])
    hi = float(df_bt.at[i,"High"])
    lo = float(df_bt.at[i,"Low"])
    cl = float(df_bt.at[i,"Close"])

    # 0) Сработала ли PENDING-заявка на ВХОД на этом баре?
    if (not in_trade) and (pending is not None) and (i == pending["idx"] + 1):
        side = pending["side"]; level = pending["level"]
        trig = (side==1 and hi>=level) or (side==-1 and lo<=level)
        if trig:
            fill = max(level, op) if side==1 else min(level, op)
            # размер главного лота
            units_main = (equity * EXPOSURE_FRACTION) / (fill * MULTIPLIER)
            if units_main > 0:
                # комиссия на вход (покупка/продажа) — берём всегда per side
                fee_in_main = fee_cash(fill * units_main * MULTIPLIER)
                equity -= fee_in_main

                pos = side; in_trade=True
                entry_i = i; entry_px = float(fill); entry_eq = float(equity)
                # главный стоп 2% от equity на момент входа
                risk_main = STOP_RISK_PCT_MAIN * equity
                stop_dist = risk_main / (units_main * MULTIPLIER)
                stop_px_main = entry_px - stop_dist if pos==1 else entry_px + stop_dist

                # включаем ожидание первого подтверждённого фрактала для добора
                addon_done=False; wait_addon=True
                units_add=0.0; fee_in_add=0.0; stop_px_add=None
        pending=None

    # 1) Жёсткий STOP-MARKET внутри бара (учитываем ближайший из двух стопов)
    if in_trade and (stop_px_main is not None or stop_px_add is not None):
        if pos==1:
            # для лонга триггерится ближайший (наибольший) из доступных стопов
            trigger_level = max([x for x in [stop_px_main, stop_px_add] if x is not None], default=None)
            hit = (trigger_level is not None) and (lo <= trigger_level)
            if hit:
                stop_fill = min(trigger_level, op)  # не лучше открытия
        else:
            # для шорта — наименьший из стопов выше цены
            trigger_level = min([x for x in [stop_px_main, stop_px_add] if x is not None], default=None)
            hit = (trigger_level is not None) and (hi >= trigger_level)
            if hit:
                stop_fill = max(trigger_level, op)
        if hit:
            # доучёт PnL от prev_close до stop_fill на общий объём
            total_units = units_main + units_add
            equity += (stop_fill - prev_close) * total_units * pos * MULTIPLIER
            prev_close = stop_fill
            # комиссия на выход (side-agnostic per side)
            fee_out = fee_cash(stop_fill * total_units * MULTIPLIER)
            equity -= fee_out

            trades.append({
                "side": "LONG" if pos==1 else "SHORT",
                "entry_time": df_bt.at[entry_i,"DateTime"],
                "exit_time":  df_bt.at[i,"DateTime"],
                "entry_price": float(entry_px),
                "exit_price":  float(stop_fill),
                "units_main": float(units_main),
                "units_add":  float(units_add),
                "reason": "hard_stop_any_leg",
                "fee_in_main": float(fee_in_main),
                "fee_in_add":  float(fee_in_add),
                "fee_out": float(fee_out),
                "stop_px_main": float(stop_px_main) if stop_px_main is not None else np.nan,
                "stop_px_add":  float(stop_px_add)  if stop_px_add  is not None else np.nan,
            })
            # сброс позиции
            pos=0; in_trade=False
            units_main=units_add=0.0
            entry_i=entry_px=entry_eq=None
            fee_in_main=fee_in_add=0.0
            stop_px_main=stop_px_add=None
            addon_done=False; wait_addon=False
            equity_curve.append(equity)
            continue

    # 2) MTM close→close (если позиция активна и стоп не сработал)
    if i>0 and in_trade:
        total_units = units_main + units_add
        equity += (cl - prev_close) * total_units * pos * MULTIPLIER
    prev_close = cl

    # 3) Плановый выход по flip аллигатора
    ex = int(df_bt.at[i,"ExitSignal"])
    if in_trade and ((pos==1 and ex==1) or (pos==-1 and ex==-1)):
        exit_price = px_exit(i)
        total_units = units_main + units_add
        # доучёт PnL до цены выхода
        equity += (exit_price - prev_close) * total_units * pos * MULTIPLIER
        prev_close = exit_price
        # комиссия на выход
        fee_out = fee_cash(exit_price * total_units * MULTIPLIER)
        equity -= fee_out

        trades.append({
            "side":"LONG" if pos==1 else "SHORT",
            "entry_time": df_bt.at[entry_i,"DateTime"],
            "exit_time":  df_bt.at[i,"DateTime"],
            "entry_price": float(entry_px),
            "exit_price":  float(exit_price),
            "units_main": float(units_main),
            "units_add":  float(units_add),
            "reason":"alligator_flip_exit",
            "fee_in_main": float(fee_in_main),
            "fee_in_add":  float(fee_in_add),
            "fee_out": float(fee_out),
            "stop_px_main": float(stop_px_main) if stop_px_main is not None else np.nan,
            "stop_px_add":  float(stop_px_add)  if stop_px_add  is not None else np.nan,
        })
        pos=0; in_trade=False
        units_main=units_add=0.0
        entry_i=entry_px=entry_eq=None
        fee_in_main=fee_in_add=0.0
        stop_px_main=stop_px_add=None
        addon_done=False; wait_addon=False
        equity_curve.append(equity)
        continue

    # 4) ДОБОР: первый подтверждённый фрактал после входа (один раз)
    if in_trade and wait_addon and (not addon_done):
        upc = int(df_bt.at[i,"Fractal_Up_conf"])
        dnc = int(df_bt.at[i,"Fractal_Down_conf"])
        trig_add = (pos==1 and upc==1) or (pos==-1 and dnc==1)
        if trig_add:
            fill_add = px_addon(i)
            # объём добора = 30% от главного объёма (округление вниз до целых, если нужно)
            units_add = float(int(np.floor(units_main * ADDON_RATIO)))
            if units_add <= 0:
                addon_done=True; wait_addon=False
            else:
                # комиссия на вход добора
                fee_in_add = fee_cash(fill_add * units_add * MULTIPLIER)
                equity -= fee_in_add
                # стоп для добора: 2% от equity на момент добора
                risk_add = STOP_RISK_PCT_ADD * equity
                stop_dist_add = risk_add / (units_add * MULTIPLIER)
                stop_px_add = fill_add - stop_dist_add if pos==1 else fill_add + stop_dist_add
                addon_done=True; wait_addon=False

    # 5) Постановка НОВОЙ стоп-заявки на ВХОД при сигнале (если flat и без pending)
    if (not in_trade) and (pending is None) and ex==0:
        s = int(df_bt.at[i,"EntrySignal"])
        if s!=0 and i+1 < len(df_bt):
            level = float(df_bt.at[i,"High"]) if s==1 else float(df_bt.at[i,"Low"])
            pending = {"side": s, "level": level, "idx": i}

    equity_curve.append(equity)

# -------- Статистика --------
bt = df_bt[["DateTime"]].copy()
bt["Equity"] = equity_curve

def full_stats(bt: pd.DataFrame, trades: list):
    out = {}
    out["start_date"]   = bt["DateTime"].iloc[0]
    out["end_date"]     = bt["DateTime"].iloc[-1]
    out["start_equity"] = START_EQUITY
    out["final_equity"] = float(bt["Equity"].iloc[-1])
    out["total_return_%"] = (out["final_equity"]/out["start_equity"] - 1.0) * 100.0
    years = max((bt["DateTime"].iloc[-1]-bt["DateTime"].iloc[0]).days/365.25, 1e-9)
    out["CAGR_%"] = ((out["final_equity"]/out["start_equity"])**(1/years) - 1) * 100.0
    roll = bt["Equity"].cummax(); dd = bt["Equity"]/roll - 1.0
    out["MaxDD_%"] = float(dd.min()*100.0)
    out["NumTrades"] = len(trades)
    if trades:
        tr = pd.DataFrame(trades)
        tr["ret_trade_%"] = np.where(
            tr["side"]=="LONG",
            (tr["exit_price"]/tr["entry_price"] - 1.0)*100.0,
            (tr["entry_price"]/tr["exit_price"] - 1.0)*100.0
        )
        tr["dur_days"] = (pd.to_datetime(tr["exit_time"])-pd.to_datetime(tr["entry_time"])).dt.days
        out["WinRate_%"] = float((tr["ret_trade_%"]>0).mean()*100.0)
        out["AvgTradeRet_%"] = float(tr["ret_trade_%"].mean())
        out["MedianTradeRet_%"] = float(tr["ret_trade_%"].median())
        out["AvgDur_days"] = float(tr["dur_days"].mean())
        gp = tr.loc[tr["ret_trade_%"]>0,"ret_trade_%"].sum()
        gl = -tr.loc[tr["ret_trade_%"]<=0,"ret_trade_%"].sum()
        out["ProfitFactor"] = float(gp/gl) if gl>0 else np.nan
        out["Fees_total"]   = float(tr.get("fee_in_main",0).sum() + tr.get("fee_in_add",0).sum() + tr.get("fee_out",0).sum())
        out["Fees_per_trade_mean"] = float((tr.get("fee_in_main",0) + tr.get("fee_in_add",0) + tr.get("fee_out",0)).mean())
        out["AddOn_used_%"] = 100.0 * float((tr["units_add"]>0).mean())
    else:
        out.update({"WinRate_%":np.nan,"AvgTradeRet_%":np.nan,"MedianTradeRet_%":np.nan,
                    "AvgDur_days":np.nan,"ProfitFactor":np.nan,"Fees_total":0.0,
                    "Fees_per_trade_mean":0.0,"AddOn_used_%":0.0})
    return out

stats = full_stats(bt, trades)
print("=== FULL-PERIOD STATS (breakout + addon@first fractal + dual hard stops + fee 0.035%/side) ===")
for k,v in stats.items():
    print(f"{k}: {v}")

##################################################################################

# === Подготовка данных + БЭКТЕСТ (совместим с твоей логикой) ===
import numpy as np
import pandas as pd

# Параметры по умолчанию (их можно переопределять через аргументы run_breakout_backtest(...))
DEFAULTS = dict(
    START_EQUITY       = 100_000.0,
    STOP_RISK_PCT_MAIN = 0.02,     # 2% от equity на момент входа
    STOP_RISK_PCT_ADD  = 0.02,     # 2% от equity на момент добора
    EXPOSURE_FRACTION  = 1.0,      # доля капитала на главный вход
    ADDON_RATIO        = 0.30,     # 30% от главного объёма
    MULTIPLIER         = 1.0,
    EXEC_EXIT          = "open_next",   # "open_next" | "close_signal"
    EXEC_ADDON         = "open_next",   # как исполняем добор
    FEE_SIDE_RATE      = 0.00035        # 0.035% на сторону
)

def fee_cash(notional, fee_rate):
    return float(notional) * float(fee_rate)

def prepare_df_bt(df: pd.DataFrame) -> pd.DataFrame:
    # Требуемые колонки как у тебя
    req = ["Open","High","Low","Close","EntrySignal",
           "Alligator_Jaw","Alligator_Teeth","Alligator_Lips","Fractal_Up","Fractal_Down","DateTime"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Нет колонок: {missing}")

    df_bt = df.copy().sort_values("DateTime").reset_index(drop=True)
    df_bt["DateTime"] = pd.to_datetime(df_bt["DateTime"], errors="coerce")
    df_bt = df_bt[df_bt["DateTime"].notna()].reset_index(drop=True)

    # подтвержденные фракталы (подтверждение через 2 бара)
    if "Fractal_Up_conf" not in df_bt.columns:
        df_bt["Fractal_Up_conf"] = df_bt["Fractal_Up"].shift(2).fillna(0).astype(int)
    if "Fractal_Down_conf" not in df_bt.columns:
        df_bt["Fractal_Down_conf"] = df_bt["Fractal_Down"].shift(2).fillna(0).astype(int)

    # Аллигатор и Exit по flip
    bull = (df_bt["Alligator_Lips"] > df_bt["Alligator_Teeth"]) & (df_bt["Alligator_Teeth"] > df_bt["Alligator_Jaw"])
    bear = (df_bt["Alligator_Jaw"]  > df_bt["Alligator_Teeth"]) & (df_bt["Alligator_Teeth"] > df_bt["Alligator_Lips"])
    bull_flip = (bull & ~bull.shift(1).fillna(False))
    bear_flip = (bear & ~bear.shift(1).fillna(False))

    df_bt = df_bt.copy()
    df_bt["ExitSignal"] = 0
    _pos = 0
    for i in range(len(df_bt)):
        if _pos==1 and bear_flip.iloc[i]:
            df_bt.at[i,"ExitSignal"]=1
            _pos=0
        elif _pos==-1 and bull_flip.iloc[i]:
            df_bt.at[i,"ExitSignal"]=-1
            _pos=0
        if _pos==0 and df_bt.at[i,"ExitSignal"]==0:
            s = int(df_bt.at[i,"EntrySignal"])
            if s!=0:
                _pos=s
    return df_bt

def _px_exit(df_bt, i, exec_exit):
    if exec_exit=="close_signal" or i+1>=len(df_bt):
        return float(df_bt.at[i,"Close"])
    return float(df_bt.at[i+1,"Open"])

def _px_addon(df_bt, i, exec_addon):
    if exec_addon=="close_signal" or i+1>=len(df_bt):
        return float(df_bt.at[i,"Close"])
    return float(df_bt.at[i+1,"Open"])

def run_breakout_backtest(df: pd.DataFrame, **params):
    # слияние параметров
    P = DEFAULTS.copy()
    P.update(params or {})

    # подготовка данных (как у тебя)
    df_bt = prepare_df_bt(df)

    START_EQUITY       = float(P["START_EQUITY"])
    STOP_RISK_PCT_MAIN = float(P["STOP_RISK_PCT_MAIN"])
    STOP_RISK_PCT_ADD  = float(P["STOP_RISK_PCT_ADD"])
    EXPOSURE_FRACTION  = float(P["EXPOSURE_FRACTION"])
    ADDON_RATIO        = float(P["ADDON_RATIO"])
    MULTIPLIER         = float(P["MULTIPLIER"])
    EXEC_EXIT          = str(P["EXEC_EXIT"])
    EXEC_ADDON         = str(P["EXEC_ADDON"])
    FEE_SIDE_RATE      = float(P["FEE_SIDE_RATE"])

    equity = START_EQUITY
    equity_curve, trades = [], []

    pos=0; in_trade=False
    units_main=0.0; units_add=0.0
    entry_i=None; entry_px=None; entry_eq=None
    fee_in_main=0.0; fee_in_add=0.0
    stop_px_main=None; stop_px_add=None
    prev_close=float(df_bt.at[0,"Close"])

    pending = None  # отложка на вход breakout (активна только на следующем баре)
    addon_done=False
    wait_addon=False

    for i in range(len(df_bt)):
        op = float(df_bt.at[i,"Open"])
        hi = float(df_bt.at[i,"High"])
        lo = float(df_bt.at[i,"Low"])
        cl = float(df_bt.at[i,"Close"])

        # 0) проверка срабатывания отложенной заявки на вход
        if (not in_trade) and (pending is not None) and (i == pending["idx"] + 1):
            side = pending["side"]; level = pending["level"]
            trig = (side==1 and hi>=level) or (side==-1 and lo<=level)
            if trig:
                fill = max(level, op) if side==1 else min(level, op)
                units_main = (equity * EXPOSURE_FRACTION) / (fill * MULTIPLIER)
                if units_main > 0:
                    fee_in_main = fee_cash(fill * units_main * MULTIPLIER, FEE_SIDE_RATE)
                    equity -= fee_in_main

                    pos = side; in_trade=True
                    entry_i = i; entry_px = float(fill); entry_eq = float(equity)
                    # стоп для главного объёма
                    risk_main = STOP_RISK_PCT_MAIN * equity
                    stop_dist = risk_main / (units_main * MULTIPLIER)
                    stop_px_main = entry_px - stop_dist if pos==1 else entry_px + stop_dist

                    addon_done=False; wait_addon=True
                    units_add=0.0; fee_in_add=0.0; stop_px_add=None
            pending=None

        # 1) жёсткие стопы (ближайший из двух)
        if in_trade and (stop_px_main is not None or stop_px_add is not None):
            hit=False
            if pos==1:
                trigger_level = max([x for x in [stop_px_main, stop_px_add] if x is not None], default=None)
                hit = (trigger_level is not None) and (lo <= trigger_level)
                if hit:
                    stop_fill = min(trigger_level, op)
            else:
                trigger_level = min([x for x in [stop_px_main, stop_px_add] if x is not None], default=None)
                hit = (trigger_level is not None) and (hi >= trigger_level)
                if hit:
                    stop_fill = max(trigger_level, op)
            if hit:
                total_units = units_main + units_add
                equity += (stop_fill - prev_close) * total_units * pos * MULTIPLIER
                prev_close = stop_fill
                fee_out = fee_cash(stop_fill * total_units * MULTIPLIER, FEE_SIDE_RATE)
                equity -= fee_out

                trades.append({
                    "side": "LONG" if pos==1 else "SHORT",
                    "entry_time": df_bt.at[entry_i,"DateTime"],
                    "exit_time":  df_bt.at[i,"DateTime"],
                    "entry_price": float(entry_px),
                    "exit_price":  float(stop_fill),
                    "units_main": float(units_main),
                    "units_add":  float(units_add),
                    "reason": "hard_stop_any_leg",
                    "fee_in_main": float(fee_in_main),
                    "fee_in_add":  float(fee_in_add),
                    "fee_out": float(fee_out),
                    "stop_px_main": float(stop_px_main) if stop_px_main is not None else np.nan,
                    "stop_px_add":  float(stop_px_add)  if stop_px_add  is not None else np.nan,
                })
                pos=0; in_trade=False
                units_main=units_add=0.0
                entry_i=entry_px=entry_eq=None
                fee_in_main=fee_in_add=0.0
                stop_px_main=stop_px_add=None
                addon_done=False; wait_addon=False
                equity_curve.append(equity)
                continue

        # 2) MTM close→close
        if i>0 and in_trade:
            total_units = units_main + units_add
            equity += (cl - prev_close) * total_units * pos * MULTIPLIER
        prev_close = cl

        # 3) Выход по flip аллигатора
        ex = int(df_bt.at[i,"ExitSignal"])
        if in_trade and ((pos==1 and ex==1) or (pos==-1 and ex==-1)):
            exit_price = _px_exit(df_bt, i, EXEC_EXIT)
            total_units = units_main + units_add
            equity += (exit_price - prev_close) * total_units * pos * MULTIPLIER
            prev_close = exit_price
            fee_out = fee_cash(exit_price * total_units * MULTIPLIER, FEE_SIDE_RATE)
            equity -= fee_out

            trades.append({
                "side":"LONG" if pos==1 else "SHORT",
                "entry_time": df_bt.at[entry_i,"DateTime"],
                "exit_time":  df_bt.at[i,"DateTime"],
                "entry_price": float(entry_px),
                "exit_price":  float(exit_price),
                "units_main": float(units_main),
                "units_add":  float(units_add),
                "reason":"alligator_flip_exit",
                "fee_in_main": float(fee_in_main),
                "fee_in_add":  float(fee_in_add),
                "fee_out": float(fee_out),
                "stop_px_main": float(stop_px_main) if stop_px_main is not None else np.nan,
                "stop_px_add":  float(stop_px_add)  if stop_px_add  is not None else np.nan,
            })
            pos=0; in_trade=False
            units_main=units_add=0.0
            entry_i=entry_px=entry_eq=None
            fee_in_main=fee_in_add=0.0
            stop_px_main=stop_px_add=None
            addon_done=False; wait_addon=False
            equity_curve.append(equity)
            continue

        # 4) ДОБОР по первому подтверждённому фракталу
        if in_trade and wait_addon and (not addon_done):
            upc = int(df_bt.at[i,"Fractal_Up_conf"])
            dnc = int(df_bt.at[i,"Fractal_Down_conf"])
            trig_add = (pos==1 and upc==1) or (pos==-1 and dnc==1)
            if trig_add:
                fill_add = _px_addon(df_bt, i, EXEC_ADDON)
                units_add = float(int(np.floor(units_main * ADDON_RATIO)))
                if units_add <= 0:
                    addon_done=True; wait_addon=False
                else:
                    fee_in_add = fee_cash(fill_add * units_add * MULTIPLIER, FEE_SIDE_RATE)
                    equity -= fee_in_add
                    risk_add = STOP_RISK_PCT_ADD * equity
                    stop_dist_add = risk_add / (units_add * MULTIPLIER)
                    stop_px_add = fill_add - stop_dist_add if pos==1 else fill_add + stop_dist_add
                    addon_done=True; wait_addon=False

        # 5) Новая стоп-заявка на ВХОД при сигнале (flat, нет pending)
        if (not in_trade) and (pending is None) and ex==0:
            s = int(df_bt.at[i,"EntrySignal"])
            if s!=0 and i+1 < len(df_bt):
                level = float(df_bt.at[i,"High"]) if s==1 else float(df_bt.at[i,"Low"])
                pending = {"side": s, "level": level, "idx": i}

        equity_curve.append(equity)

    # отчёт
    bt = df_bt[["DateTime"]].copy()
    bt["Equity"] = equity_curve

    def full_stats(bt: pd.DataFrame, trades: list):
        out = {}
        out["start_date"]   = bt["DateTime"].iloc[0]
        out["end_date"]     = bt["DateTime"].iloc[-1]
        out["start_equity"] = START_EQUITY
        out["final_equity"] = float(bt["Equity"].iloc[-1])
        out["total_return_%"] = (out["final_equity"]/out["start_equity"] - 1.0) * 100.0
        years = max((bt["DateTime"].iloc[-1]-bt["DateTime"].iloc[0]).days/365.25, 1e-9)
        out["CAGR_%"] = ((out["final_equity"]/out["start_equity"])**(1/years) - 1) * 100.0
        roll = bt["Equity"].cummax(); dd = bt["Equity"]/roll - 1.0
        out["MaxDD_%"] = float(dd.min()*100.0)
        out["NumTrades"] = len(trades)
        if trades:
            tr = pd.DataFrame(trades)
            tr["ret_trade_%"] = np.where(
                tr["side"]=="LONG",
                (tr["exit_price"]/tr["entry_price"] - 1.0)*100.0,
                (tr["entry_price"]/tr["exit_price"] - 1.0)*100.0
            )
            tr["dur_days"] = (pd.to_datetime(tr["exit_time"])-pd.to_datetime(tr["entry_time"])).dt.days
            out["WinRate_%"] = float((tr["ret_trade_%"]>0).mean()*100.0)
            out["AvgTradeRet_%"] = float(tr["ret_trade_%"].mean())
            out["MedianTradeRet_%"] = float(tr["ret_trade_%"].median())
            out["AvgDur_days"] = float(tr["dur_days"].mean())
            gp = tr.loc[tr["ret_trade_%"]>0,"ret_trade_%"].sum()
            gl = -tr.loc[tr["ret_trade_%"]<=0,"ret_trade_%"].sum()
            out["ProfitFactor"] = float(gp/gl) if gl>0 else np.nan
            out["Fees_total"]   = float(tr.get("fee_in_main",0).sum() + tr.get("fee_in_add",0).sum() + tr.get("fee_out",0).sum())
            out["Fees_per_trade_mean"] = float((tr.get("fee_in_main",0) + tr.get("fee_in_add",0) + tr.get("fee_out",0)).mean())
            out["AddOn_used_%"] = 100.0 * float((tr["units_add"]>0).mean())
        else:
            out.update({"WinRate_%":np.nan,"AvgTradeRet_%":np.nan,"MedianTradeRet_%":np.nan,
                        "AvgDur_days":np.nan,"ProfitFactor":np.nan,"Fees_total":0.0,
                        "Fees_per_trade_mean":0.0,"AddOn_used_%":0.0})
        return out

    stats = full_stats(bt, trades)
    return bt, pd.DataFrame(trades) if trades else pd.DataFrame(columns=["side"]), stats