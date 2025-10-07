# plotting.py
from __future__ import annotations

from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import ta

# (Opcional) estilo si tienes seaborn:
try:
    import seaborn as sns
    sns.set_theme()
except Exception:
    pass

# Defaults de figuras
plt.rcParams.setdefault('figure.figsize', [12, 6])
plt.rcParams.setdefault('axes.titlesize', 16)
plt.rcParams.setdefault('axes.labelsize', 14)
plt.rcParams.setdefault('axes.titleweight', 'bold')
plt.rcParams.setdefault('grid.alpha', 0.3)


# ===========================
# Helpers básicos
# ===========================
def _last_n_years(series: pd.Series, n_years: int) -> pd.Series:
    if series is None or series.empty or n_years is None:
        return series
    end = series.index[-1]
    start = end - pd.DateOffset(years=int(n_years))
    return series.loc[start:]


def compute_splits_from_dfs(train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame):
    markers: List[Tuple[str, pd.Timestamp]] = []
    if isinstance(test_df, pd.DataFrame) and len(test_df):
        markers.append(("Test start", test_df.index[0]))
    if isinstance(val_df, pd.DataFrame) and len(val_df):
        markers.append(("Validation start", val_df.index[0]))
    return markers


def _fmt_money(y: float) -> str:
    y = float(y)
    if abs(y) >= 1_000_000:
        return f"{y/1_000_000:.1f}M"
    if abs(y) >= 1_000:
        return f"{y/1_000:.0f}K"
    return f"{y:.0f}"


# ===========================
# Plots de equity
# ===========================
def plot_equity(
    equity: pd.Series,
    title: str = "Evolución del Portafolio",
    logy: bool = False,
    show: bool = True,
):
    if equity is None or equity.empty:
        raise ValueError("Equity vacío.")
    eq = equity.sort_index()

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    ax.plot(eq.index, eq.values, label="Equity")

    if logy:
        ax.set_yscale("log")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: _fmt_money(y)))
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    if show:
        plt.show()
    return fig


def plot_drawdown(
    equity: pd.Series,
    title: str = "Drawdown",
    show: bool = True,
):
    if equity is None or equity.empty:
        raise ValueError("Equity vacío.")
    eq = equity.sort_index()
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0

    fig = plt.figure(figsize=(12, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(dd.index, dd.values, label="Drawdown")
    ax.fill_between(dd.index, dd.values, 0, alpha=0.25)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(min(-1.0, float(dd.min())), 0.0)
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("DD")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    if show:
        plt.show()
    return fig


def plot_equity_train_test_val(
    equity_full: pd.Series,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    title: str = "Evolución del Portafolio (Mejor estrategia)",
    colors: tuple[str, str, str] = ("goldenrod", "slateblue", "darkcyan"),  # Train, Test, Val
    resample: str | None = "1D",        # re-muestreo para suavizar (None=sin resample)
    drawdown_years: int | None = None,  # None = drawdown de TODO el histórico
    logy: bool = False,
    show_grid: bool = True,
    init_capital: float | None = 1_000_000.0,   # línea de capital inicial
    show: bool = True,
):
    """
    Dibuja TODO el histórico segmentado por Train/Test/Validation y, debajo,
    un panel de drawdown. Formato: Y en dinero, X en años.
    """
    if equity_full is None or equity_full.empty:
        raise ValueError("Equity vacío.")

    # --- 1) Re-muestreo para suavizar ---
    eq = equity_full.copy().sort_index()
    if resample:
        eq = eq.resample(resample).last().ffill()

    # --- 2) Segmentos por fechas ---
    def _seg(src, lo, hi):
        if lo is None or hi is None or len(src) == 0:
            return src.iloc[0:0]
        return src.loc[max(src.index.min(), lo): min(src.index.max(), hi)]

    train_eq = _seg(eq, eq.index.min() if len(train_df)==0 else train_df.index.min(),
                       eq.index.min() if len(train_df)==0 else train_df.index.max())
    test_eq  = _seg(eq, test_df.index.min()  if len(test_df) else None,
                       test_df.index.max()   if len(test_df) else None)
    val_eq   = _seg(eq, val_df.index.min()   if len(val_df)  else None,
                       val_df.index.max()    if len(val_df)  else None)

    # --- 3) Layout ---
    fig = plt.figure(figsize=(13, 6.5))
    gs  = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.30)

    # Panel 1: Equity
    ax1 = fig.add_subplot(gs[0, 0])
    if len(train_eq):
        ax1.plot(train_eq.index, train_eq.values, label="Train", color=colors[0], linewidth=1.6)
    if len(test_eq):
        ax1.plot(test_eq.index,  test_eq.values,  label="Test",  color=colors[1], linewidth=1.6)
    if len(val_eq):
        ax1.plot(val_eq.index,   val_eq.values,   label="Validation", color=colors[2], linewidth=1.8)

    if init_capital is not None:
        ax1.axhline(float(init_capital), color="gray", linestyle="--", alpha=0.5)
        y_ = float(init_capital)
        x_ = eq.index.min() + (eq.index.max() - eq.index.min()) * 0.02
        ax1.text(x_, y_, "Capital inicial", color="gray", fontsize=9, va="bottom")

    if logy:
        ax1.set_yscale("log")

    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y/1_000_000:.1f}M"))
    ax1.set_title(title)
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Valor del Portafolio")
    ax1.legend(loc="best")
    if show_grid:
        ax1.grid(True, alpha=0.25)

    # Panel 2: Drawdown
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    dd_src = eq
    if drawdown_years:
        cutoff = dd_src.index.max() - pd.DateOffset(years=drawdown_years)
        dd_src = dd_src.loc[cutoff:] if len(dd_src) else dd_src

    if len(dd_src):
        roll_max = dd_src.cummax()
        dd = dd_src / roll_max - 1.0
        ax2.plot(dd.index, dd.values, color="#1f77b4", linewidth=1.2, label="Drawdown")
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.25, color="#1f77b4")
        ymin = min(-1.0, float(dd.min()))
    else:
        ymin = -1.0

    ax2.axhline(0.0, linewidth=0.8, color="black")
    ax2.set_ylim(ymin, 0.0)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Fecha")
    ax2.legend(loc="best")
    if show_grid:
        ax2.grid(True, alpha=0.25)

    if show:
        plt.show()
    return fig


def plot_validation_equity(
    equity_val: pd.Series,
    title: str = "Validation (20% hold-out)",
    resample: str | None = "1D",
    logy: bool = False,
    show_grid: bool = True,
    init_capital: float | None = 1_000_000.0,
    show_drawdown: bool = True,
    show: bool = True,
):
    """
    Grafica SOLO la curva de VALIDATION. Si show_drawdown=True, añade
    un segundo panel de drawdown en la MISMA figura.
    """
    if equity_val is None or equity_val.empty:
        raise ValueError("Equity de Validation vacío.")

    eq = equity_val.copy().sort_index()
    if resample:
        eq = eq.resample(resample).last().ffill()

    # Layout (1 o 2 paneles)
    if show_drawdown:
        fig = plt.figure(figsize=(13, 5.5))
        gs  = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.30)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    else:
        fig = plt.figure(figsize=(13, 4.2))
        ax1 = fig.add_subplot(111)
        ax2 = None

    # Panel 1: Equity
    ax1.plot(eq.index, eq.values, label="Validation", color="tab:cyan", linewidth=1.8)
    if init_capital is not None:
        y0 = float(init_capital)
        ax1.axhline(y0, color="gray", linestyle="--", alpha=0.5)
        x0 = eq.index.min() + (eq.index.max() - eq.index.min()) * 0.02
        ax1.text(x0, y0, "Capital inicial", color="gray", fontsize=9, va="bottom")

    if logy:
        ax1.set_yscale("log")

    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: _fmt_money(y)))
    ax1.set_title(title)
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Valor del Portafolio")
    ax1.legend(loc="best")
    if show_grid:
        ax1.grid(True, alpha=0.25)

    # Panel 2: Drawdown
    if show_drawdown and ax2 is not None:
        roll_max = eq.cummax()
        dd = eq / roll_max - 1.0
        ax2.plot(dd.index, dd.values, color="#1f77b4", linewidth=1.2, label="Drawdown (Validation)")
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.25, color="#1f77b4")
        ax2.axhline(0.0, linewidth=0.8, color="black")
        ax2.set_ylim(min(-1.0, float(dd.min())), 0.0)
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Fecha")
        ax2.legend(loc="best")
        if show_grid:
            ax2.grid(True, alpha=0.25)

    if show:
        plt.show()
    return fig


# ===========================
# Plots de indicadores
# ===========================
def plot_indicators(
    data_sig: pd.DataFrame,
    params: dict,
    show: bool = True
) -> Dict[str, plt.Figure]:
    """
    Dibuja RSI, MACD, ADX(+DI/-DI), Bollinger y Stochastic sobre el DF que pases.
    Devuelve un dict con las figuras creadas. Si show=False, no bloquea.
    """
    ds = data_sig.copy().sort_index()
    needed_ohlc = {"Close", "High", "Low"}
    missing = needed_ohlc - set(ds.columns)
    if missing:
        raise ValueError(f"Faltan columnas OHLC en data_sig: {missing}")

    figs: Dict[str, plt.Figure] = {}

    # ---- RSI ----
    if "rsi" not in ds.columns:
        rsi = ta.momentum.RSIIndicator(ds["Close"], window=int(params["rsi_window"])).rsi()
        ds["rsi"] = rsi
    fig_rsi = plt.figure(figsize=(12, 3))
    ax = fig_rsi.add_subplot(111)
    ax.plot(ds.index, ds["rsi"], label="RSI")
    ax.axhline(params["rsi_lower"], linestyle="--", label=f"RSI lower {params['rsi_lower']}")
    ax.axhline(params["rsi_upper"], linestyle="--", label=f"RSI upper {params['rsi_upper']}")
    ax.set_title("RSI"); ax.legend(); ax.grid(True, alpha=0.25)
    figs["rsi"] = fig_rsi

    # ---- MACD ----
    macd = ta.trend.MACD(
        ds["Close"],
        window_slow=int(params["macd_slow"]),
        window_fast=int(params["macd_fast"]),
        window_sign=int(params["macd_signal"])
    )
    fig_macd = plt.figure(figsize=(12, 4))
    ax = fig_macd.add_subplot(111)
    ax.plot(ds.index, macd.macd(), label="MACD")
    ax.plot(ds.index, macd.macd_signal(), label="Signal")
    ax.bar(ds.index, macd.macd_diff(), label="Hist", width=1.0)
    ax.set_title("MACD"); ax.legend(); ax.grid(True, alpha=0.25)
    figs["macd"] = fig_macd

    # ---- ADX (+DI/-DI) ----
    adx = ta.trend.ADXIndicator(ds["High"], ds["Low"], ds["Close"], window=int(params["adx_window"]))
    fig_adx = plt.figure(figsize=(12, 3))
    ax = fig_adx.add_subplot(111)
    ax.plot(adx.adx(), label="ADX")
    ax.plot(adx.adx_pos(), label="+DI")
    ax.plot(adx.adx_neg(), label="-DI")
    ax.axhline(params["adx_threshold"], linestyle="--", label="ADX thr")
    ax.set_title("ADX / +DI / -DI"); ax.legend(); ax.grid(True, alpha=0.25)
    figs["adx"] = fig_adx

    # ---- Bollinger ----
    bb = ta.volatility.BollingerBands(
        ds["Close"], window=int(params["bb_window"]), window_dev=float(params["bb_dev"])
    )
    bb_h, bb_l, bb_m = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
    fig_bb = plt.figure(figsize=(12, 4))
    ax = fig_bb.add_subplot(111)
    ax.plot(ds.index, ds["Close"], label="Close")
    ax.plot(ds.index, bb_m, label="BB mid")
    ax.plot(ds.index, bb_h, label="BB high")
    ax.plot(ds.index, bb_l, label="BB low")
    ax.set_title("Bollinger Bands"); ax.legend(); ax.grid(True, alpha=0.25)
    figs["bollinger"] = fig_bb

    # ---- Stochastic ----
    stoch = ta.momentum.StochasticOscillator(
        ds["High"], ds["Low"], ds["Close"],
        window=int(params["stoch_window"]),
        smooth_window=int(params["stoch_smooth"])
    )
    k, d = stoch.stoch(), stoch.stoch_signal()
    fig_st = plt.figure(figsize=(12, 3))
    ax = fig_st.add_subplot(111)
    ax.plot(ds.index, k, label="%K")
    ax.plot(ds.index, d, label="%D")
    ax.axhline(20, linestyle="--", label="20")
    ax.axhline(80, linestyle="--", label="80")
    ax.set_title("Stochastic Oscillator"); ax.legend(); ax.grid(True, alpha=0.25)
    figs["stochastic"] = fig_st

    if show:
        plt.show()
    return figs
