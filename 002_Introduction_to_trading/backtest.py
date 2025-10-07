from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import ta

from position import Position

__all__ = ["backtest"]

def backtest(
    data_sig: pd.DataFrame,
    stop_loss: float,
    take_profit: float,
    risk_fraction: float,
    fee: float = 0.00125,
    initial_capital: float = 1_000_000.0,
    allow_short: bool = True,
    short_mode: str = "no_leverage",
    # Gestión opcional
    breakeven_trigger: float | None = 0.0,
    trail_atr_mult: float | None = 0.0,
    time_stop_bars: int | None = 0,
    atr_window_for_trail: int = 14,
    target_vol: float | None = None,
    vol_lookback: int = 24*30,
    max_rf_scale: float = 3.0
):
    """
    Backtest con largos y cortos sin apalancamiento (no_leverage).
    Espera en data_sig:
      - 'Close' (y opcionalmente 'High','Low' si usas trailing por ATR)
      - 'signal' ∈ {-1, 0, 1}  (si no existe, intenta derivarlo de buy/sell_signal)
      - (opcional) '_rf_t' (riesgo por barra). Si no lo calculas fuera, se arma aquí
        con risk_fraction y/o target_vol.
    Devuelve:
      equity: pd.Series (índice tiempo, columna 'equity')
      trades: pd.DataFrame con fills de operaciones
      final_cash: float
    """
    # ---------- 0) Normalización ----------
    df = data_sig.copy().sort_index()
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="first")]
    if "Close" not in df.columns:
        raise ValueError("Falta columna 'Close' en data_sig.")

    # Señal 1/0/-1
    if "signal" in df.columns:
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0).astype(int)
    else:
        buy  = df.get("buy_signal",  pd.Series(False, index=df.index)).astype(bool)
        sell = df.get("sell_signal", pd.Series(False, index=df.index)).astype(bool)
        df["signal"] = np.where(buy & ~sell, 1, np.where(sell & ~buy, -1, 0)).astype(int)

    # Series auxiliares para trailing por ATR
    if trail_atr_mult and trail_atr_mult > 0:
        df["_atr"] = ta.volatility.AverageTrueRange(
            df.get("High", df["Close"]), df.get("Low", df["Close"]), df["Close"],
            window=atr_window_for_trail
        ).average_true_range()

    # Riesgo fraccional por barra (_rf_t): fijo o “vol targeting”
    if target_vol is not None and target_vol > 0:
        ret = df["Close"].pct_change()
        # Vol anualizada aprox. para series intradía (asume 24*365 barras/año; ajusta si necesitas)
        realized_vol_ann = (ret.rolling(vol_lookback).std() * np.sqrt(24*365)).clip(lower=1e-8)
        rf_scale = (target_vol / realized_vol_ann).clip(upper=float(max_rf_scale))
        df["_rf_t"] = (risk_fraction * rf_scale).fillna(risk_fraction)
    else:
        df["_rf_t"] = float(risk_fraction)

    # ---------- 1) Estado ----------
    cash = float(initial_capital)
    pos: Optional[Position] = None
    equity_vals, equity_idx = [], []
    trades: list[dict] = []
    bars_in_trade = 0

    # ---------- 2) Loop ----------
    for ts, row in df.iterrows():
        px = float(row["Close"])
        if not np.isfinite(px) or px <= 0:
            eq_now = cash if pos is None else (cash + (pos.reserve if pos.side == "short" else 0.0) + pos.qty * px)
            equity_vals.append(eq_now); equity_idx.append(ts)
            continue

        # ===== Gestionar posición abierta =====
        if pos is not None:
            bars_in_trade += 1

            # Breakeven
            if breakeven_trigger and breakeven_trigger > 0:
                if pos.side == "long":
                    if pos.tp > pos.entry_price and px >= pos.entry_price + breakeven_trigger * (pos.tp - pos.entry_price):
                        pos.sl = max(pos.sl, pos.entry_price)
                else:
                    if pos.tp < pos.entry_price and px <= pos.entry_price - breakeven_trigger * (pos.entry_price - pos.tp):
                        pos.sl = min(pos.sl, pos.entry_price)

            # Trailing por ATR
            if trail_atr_mult and trail_atr_mult > 0:
                atr_now = float(row.get("_atr", np.nan))
                if np.isfinite(atr_now):
                    if pos.side == "long":
                        pos.sl = max(pos.sl, px - trail_atr_mult * atr_now)
                    else:
                        pos.sl = min(pos.sl, px + trail_atr_mult * atr_now)

            # Cierres por SL/TP
            if pos.side == "long":
                hit_sl = px <= pos.sl
                hit_tp = px >= pos.tp
                if hit_sl or hit_tp:
                    exec_px = pos.sl if hit_sl else pos.tp
                    proceeds = exec_px * pos.qty * (1 - fee)
                    cash += proceeds
                    trades.append({
                        "entry_time": pos.entry_time, "exit_time": ts, "side": "long",
                        "qty": pos.qty, "entry_price": pos.entry_price, "exit_price": exec_px,
                        "pnl": proceeds - (pos.entry_price * pos.qty * (1 + fee)),
                        "pnl_pct": (exec_px / pos.entry_price - 1.0),
                    })
                    pos = None; bars_in_trade = 0
            else:
                hit_sl = px >= pos.sl
                hit_tp = px <= pos.tp
                if hit_sl or hit_tp:
                    exec_px = pos.sl if hit_sl else pos.tp
                    shares_abs = -pos.qty
                    # no_leverage: liberar colateral y pagar recompra
                    exit_cost = exec_px * shares_abs * (1 + fee)
                    cash += pos.reserve
                    cash -= exit_cost
                    proceeds = pos.entry_price * shares_abs * (1 - fee)
                    pnl = proceeds - exit_cost
                    trades.append({
                        "entry_time": pos.entry_time, "exit_time": ts, "side": "short",
                        "qty": pos.qty, "entry_price": pos.entry_price, "exit_price": exec_px,
                        "pnl": pnl,
                        "pnl_pct": (pos.entry_price / exec_px - 1.0),
                    })
                    pos = None; bars_in_trade = 0

            # Time stop
            if pos is not None and time_stop_bars and time_stop_bars > 0 and bars_in_trade >= time_stop_bars:
                if pos.side == "long":
                    proceeds = px * pos.qty * (1 - fee)
                    cash += proceeds
                    trades.append({
                        "entry_time": pos.entry_time, "exit_time": ts, "side": "long",
                        "qty": pos.qty, "entry_price": pos.entry_price, "exit_price": px,
                        "pnl": proceeds - (pos.entry_price * pos.qty * (1 + fee)),
                        "pnl_pct": (px / pos.entry_price - 1.0),
                    })
                else:
                    shares_abs = -pos.qty
                    exit_cost = px * shares_abs * (1 + fee)
                    cash += pos.reserve
                    cash -= exit_cost
                    proceeds = pos.entry_price * shares_abs * (1 - fee)
                    pnl = proceeds - exit_cost
                    trades.append({
                        "entry_time": pos.entry_time, "exit_time": ts, "side": "short",
                        "qty": pos.qty, "entry_price": pos.entry_price, "exit_price": px,
                        "pnl": pnl,
                        "pnl_pct": (pos.entry_price / px - 1.0),
                    })
                pos = None; bars_in_trade = 0

        # ===== Abrir nueva posición =====
        if pos is None:
            s = int(row["signal"])
            rf_t = float(row["_rf_t"])

            # LONG
            if s == 1:
                invest = cash * rf_t
                if invest > 0:
                    shares = invest / (px * (1 + fee))
                    cost_total = shares * px * (1 + fee)
                    if shares > 0 and cash > cost_total:
                        cash -= cost_total
                        pos = Position(
                            side="long", entry_time=ts, entry_price=px, qty=shares,
                            sl=px * (1 - float(stop_loss)), tp=px * (1 + float(take_profit)),
                            reserve=0.0
                        )
                        bars_in_trade = 0

            # SHORT (no_leverage)
            elif allow_short and s == -1:
                if short_mode != "no_leverage":
                    # Por acuerdo: NO implementamos crédito; solo 'no_leverage'
                    pass
                else:
                    invest = cash * rf_t
                    if invest > 0:
                        shares_abs = invest / (px * (1 + fee))
                        collateral = invest
                        if shares_abs > 0 and cash > collateral:
                            # entrada: recibes proceeds por la venta corta y bloqueas colateral
                            proceeds = px * shares_abs * (1 - fee)
                            cash += proceeds
                            cash -= collateral
                            pos = Position(
                                side="short", entry_time=ts, entry_price=px, qty=-shares_abs,
                                sl=px * (1 + float(stop_loss)), tp=px * (1 - float(take_profit)),
                                reserve=collateral
                            )
                            bars_in_trade = 0

        # ===== Marcar equity =====
        if pos is None:
            eq_now = cash
        else:
            # no_leverage: equity ≈ cash + (reserva si short) + MtM
            eq_now = cash + (pos.reserve if pos.side == "short" else 0.0) + pos.qty * px
        equity_vals.append(eq_now); equity_idx.append(ts)

    # ---------- 3) Liquidación final ----------
    if pos is not None:
        last_ts = df.index[-1]; last_px = float(df.iloc[-1]["Close"])
        if pos.side == "long":
            proceeds = last_px * pos.qty * (1 - fee)
            cash += proceeds
            trades.append({
                "entry_time": pos.entry_time, "exit_time": last_ts, "side": "long",
                "qty": pos.qty, "entry_price": pos.entry_price, "exit_price": last_px,
                "pnl": proceeds - (pos.entry_price * pos.qty * (1 + fee)),
                "pnl_pct": (last_px / pos.entry_price - 1.0),
            })
        else:
            shares_abs = -pos.qty
            exit_cost = last_px * shares_abs * (1 + fee)
            cash += pos.reserve
            cash -= exit_cost
            proceeds = pos.entry_price * shares_abs * (1 - fee)
            pnl = proceeds - exit_cost
            trades.append({
                "entry_time": pos.entry_time, "exit_time": last_ts, "side": "short",
                "qty": pos.qty, "entry_price": pos.entry_price, "exit_price": last_px,
                "pnl": pnl,
                "pnl_pct": (pos.entry_price / last_px - 1.0),
            })
        pos = None
        equity_vals[-1] = cash

    equity = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_idx), name="equity")
    equity.iloc[0] = float(initial_capital)
    trades = pd.DataFrame(trades)
    return equity, trades, float(cash)
