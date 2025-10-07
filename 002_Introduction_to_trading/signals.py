# =========================================================
# Señales con filtros + voto ponderado + persistencia
# Incluye adaptadores para el motor de backtest (entry/exit).
# =========================================================
from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd
import ta


def _series_or_fallback(ds: pd.DataFrame, name: str, fallback_col: str = "Close") -> pd.Series:
    """
    Devuelve ds[name] si existe; de lo contrario, usa ds[fallback_col] como proxy.
    Evita romper si el dataset no trae High/Low (por ejemplo, series de Close).
    """
    if name in ds.columns:
        return ds[name]
    if fallback_col not in ds.columns:
        raise KeyError(f"Falta la columna requerida '{name}' y tampoco existe el fallback '{fallback_col}'.")
    return ds[fallback_col]


# ---------------------------------------------------------
# Señales con filtros + voto ponderado + persistencia (TU LÓGICA)
# ---------------------------------------------------------
def generate_signals(
    data: pd.DataFrame,
    rsi_window: int,
    rsi_lower: float,
    rsi_upper: float,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    adx_window: int = 14,
    adx_threshold: float = 20.0,
    bb_window: int = 20,
    bb_dev: float = 2.0,
    stoch_window: int = 14,
    stoch_smooth: int = 3,
    use_directional_adx: bool = True,
    sma_window: int = 200,
    atr_window: int = 14,
    min_atr_pct: float = 0.003,           # ATR/Close mínimo (0.3%)
    vote_weights: Optional[Dict[str, int]] = None,
    vote_threshold_trend: int = 2,        # con-tendencia
    vote_threshold_ctr: int = 3,          # contra-tendencia
    persist_bars: int = 2                 # confirmación temporal
) -> pd.DataFrame:
    """
    Genera columnas:
      - rsi, macd_diff, adx, plus_di, minus_di, bb_l, bb_h, stoch, sma, atr_pct
      - buy_signal, sell_signal, signal ∈ {-1,0,1}

    Notas:
      * Si faltan 'High'/'Low', usa 'Close' como proxy para indicadores que lo requieren.
      * 'persist_bars'>1 exige k barras consecutivas cumpliendo condición.
    """
    ds = data.copy()

    # Aliases (con fallback por si faltan columnas OHLC)
    _close = _series_or_fallback(ds, "Close", "Close")
    _high  = _series_or_fallback(ds, "High",  "Close")
    _low   = _series_or_fallback(ds, "Low",   "Close")

    # ----------------- RSI -----------------
    rsi_ind = ta.momentum.RSIIndicator(_close, window=int(rsi_window))
    rsi = rsi_ind.rsi()
    ds["rsi"] = rsi
    rsi_buy  = rsi < rsi_lower
    rsi_sell = rsi > rsi_upper

    # ----------------- MACD ----------------
    macd_ind = ta.trend.MACD(_close, window_slow=int(macd_slow),
                             window_fast=int(macd_fast), window_sign=int(macd_signal))
    macd_diff = macd_ind.macd_diff()
    ds["macd_diff"] = macd_diff
    macd_buy  = macd_diff > 0
    macd_sell = macd_diff < 0

    # ----------------- ADX (+DI/-DI) -------
    adx_ind = ta.trend.ADXIndicator(_high, _low, _close, window=int(adx_window))
    ds["adx"]      = adx_ind.adx()
    ds["plus_di"]  = adx_ind.adx_pos()
    ds["minus_di"] = adx_ind.adx_neg()
    if use_directional_adx:
        adx_buy  = (ds["plus_di"] > ds["minus_di"]) & (ds["adx"] > adx_threshold)
        adx_sell = (ds["minus_di"] > ds["plus_di"]) & (ds["adx"] > adx_threshold)
    else:
        # Filtro de fuerza solamente: permite tanto buy como sell si hay tendencia fuerte
        strong = ds["adx"] > adx_threshold
        adx_buy  = strong
        adx_sell = strong

    # ----------------- Bollinger -----------
    bb_ind = ta.volatility.BollingerBands(_close, window=int(bb_window), window_dev=float(bb_dev))
    ds["bb_l"] = bb_ind.bollinger_lband()
    ds["bb_h"] = bb_ind.bollinger_hband()
    bb_buy  = _close < ds["bb_l"]
    bb_sell = _close > ds["bb_h"]

    # ----------------- Stochastic ----------
    st_ind = ta.momentum.StochasticOscillator(_high, _low, _close,
                                              window=int(stoch_window),
                                              smooth_window=int(stoch_smooth))
    st = st_ind.stoch()
    ds["stoch"] = st
    st_buy  = st < 20
    st_sell = st > 80

    # ----------------- Filtros de régimen --
    ds["sma"] = _close.rolling(int(sma_window)).mean()
    atr = ta.volatility.AverageTrueRange(_high, _low, _close, window=int(atr_window)).average_true_range()
    ds["atr_pct"] = (atr / (_close.replace(0, np.nan))).fillna(0.0)
    has_vol = ds["atr_pct"] >= float(min_atr_pct)

    in_uptrend   = _close > ds["sma"]
    in_downtrend = _close < ds["sma"]

    # ----------------- Votación ------------
    if vote_weights is None:
        vote_weights = dict(rsi=1, macd=2, adx=2, bb=1, st=1)  # fuerza/tendencia pesan más

    buy_score  = (vote_weights["rsi"] * rsi_buy.astype(int)  +
                  vote_weights["macd"] * macd_buy.astype(int) +
                  vote_weights["adx"] * adx_buy.astype(int)   +
                  vote_weights["bb"]  * bb_buy.astype(int)    +
                  vote_weights["st"]  * st_buy.astype(int))

    sell_score = (vote_weights["rsi"] * rsi_sell.astype(int)  +
                  vote_weights["macd"] * macd_sell.astype(int) +
                  vote_weights["adx"] * adx_sell.astype(int)   +
                  vote_weights["bb"]  * bb_sell.astype(int)    +
                  vote_weights["st"]  * st_sell.astype(int))

    # Umbral dinámico: más laxo con la tendencia, más estricto contra la tendencia
    buy_thr  = np.where(in_uptrend.values,  vote_threshold_trend, vote_threshold_ctr)
    sell_thr = np.where(in_downtrend.values, vote_threshold_trend, vote_threshold_ctr)

    raw_buy  = (buy_score.values  >= buy_thr) & has_vol.values
    raw_sell = (sell_score.values >= sell_thr) & has_vol.values

    raw_buy  = pd.Series(raw_buy,  index=ds.index)
    raw_sell = pd.Series(raw_sell, index=ds.index)

    # ----------------- Persistencia --------
    if persist_bars and persist_bars > 1:
        raw_buy  = raw_buy.rolling(int(persist_bars)).apply(lambda x: int(np.all(x)), raw=True).astype(bool)
        raw_sell = raw_sell.rolling(int(persist_bars)).apply(lambda x: int(np.all(x)), raw=True).astype(bool)

    # Señales finales
    ds["buy_signal"]  = raw_buy  & ~raw_sell
    ds["sell_signal"] = raw_sell & ~raw_buy
    ds["signal"] = np.where(ds["buy_signal"], 1, np.where(ds["sell_signal"], -1, 0)).astype(int)

    return ds


# ---------------------------------------------------------
# Adaptador para el motor de backtest que usa entry/exit
# ---------------------------------------------------------
def add_entry_exit_columns(ds: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte 'buy_signal'/'sell_signal' en columnas compatibles con el motor:
      - long_entry  := buy_signal
      - long_exit   := sell_signal
      - short_entry := sell_signal
      - short_exit  := buy_signal
    """
    out = ds.copy()
    buy  = out.get("buy_signal", pd.Series(False, index=out.index)).astype(bool)
    sell = out.get("sell_signal", pd.Series(False, index=out.index)).astype(bool)

    out["long_entry"]  = buy
    out["long_exit"]   = sell
    out["short_entry"] = sell
    out["short_exit"]  = buy
    return out


# ---------------------------------------------------------
# Helper de alto nivel: todo en uno
# ---------------------------------------------------------
def generate_signals_for_backtest(
    data: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Genera tus señales avanzadas y agrega columnas entry/exit
    para que puedas pasarlas directo al motor de backtest.
    """
    sig = generate_signals(data, **kwargs)
    sig = add_entry_exit_columns(sig)
    return sig

def apply_regime_filter(sig: pd.DataFrame, sma_window: int = 200) -> pd.DataFrame:
    s = sig.copy()
    s["SMA"] = s["Close"].rolling(sma_window, min_periods=sma_window//2).mean()
    up = s["Close"] > s["SMA"]
    dn = s["Close"] < s["SMA"]

    # sólo long si tendencia alcista, sólo short si bajista
    s["signal"] = np.where(
        s["signal"] == 1, np.where(up, 1, 0),
        np.where(s["signal"] == -1, np.where(dn, -1, 0), 0)
    ).astype(int)
    return s.drop(columns=["SMA"], errors="ignore")
