# =========================================================
# 3) Métricas y tablas de retornos
# =========================================================
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict

from utils import infer_ppy


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
    """
    Devuelve: Sharpe, Sortino, Calmar, MaxDD, WinRate, AnnualReturn (CAGR),
              AnnualVol, FinalCapital.
    - CAGR por años reales del periodo (estable frente a huecos).
    - Sharpe/Sortino anualizados a partir de media/vol por barra.
    - Calmar = CAGR / |MaxDD| (si MaxDD < 0).
    """
    if equity is None or len(equity.dropna()) < 3:
        final_cap = float(equity.dropna().iloc[-1]) if equity is not None and len(equity.dropna()) else 0.0
        return {
            "Sharpe": 0.0, "Sortino": 0.0, "Calmar": 0.0, "MaxDD": 0.0, "WinRate": 0.0,
            "AnnualReturn": 0.0, "AnnualVol": 0.0, "FinalCapital": final_cap
        }

    eq = equity.dropna()
    rets = eq.pct_change().dropna()
    if len(rets) == 0:
        final_cap = float(eq.iloc[-1])
        return {
            "Sharpe": 0.0, "Sortino": 0.0, "Calmar": 0.0, "MaxDD": 0.0, "WinRate": 0.0,
            "AnnualReturn": 0.0, "AnnualVol": 0.0, "FinalCapital": final_cap
        }

    # Periodos/año inferidos por el índice (diario~252, horario~8760, etc.)
    ppy = infer_ppy(eq.index)

    # Años reales (más robusto cuando hay huecos)
    SEC_PER_YEAR = 365.25 * 24 * 3600
    years = (eq.index[-1] - eq.index[0]).total_seconds() / SEC_PER_YEAR
    cagr = (eq.iloc[-1] / eq.iloc[0])**(1/years) - 1.0 if years > 0 else 0.0

    # Retorno y volatilidad anualizados por barra
    mu_bar = rets.mean()
    sd_bar = rets.std(ddof=0)
    ann_ret_bar = mu_bar * ppy
    ann_vol = sd_bar * np.sqrt(ppy)

    # Sharpe / Sortino
    sharpe = float(ann_ret_bar / ann_vol) if ann_vol > 0 else 0.0
    downside_bar = rets[rets < 0].std(ddof=0)
    sortino = float(ann_ret_bar / (downside_bar * np.sqrt(ppy))) if (downside_bar is not None and np.isfinite(downside_bar) and downside_bar > 0) else 0.0

    # Max drawdown y Calmar
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    calmar = float(cagr / abs(max_dd)) if (np.isfinite(cagr) and max_dd < 0) else 0.0

    # Win rate
    win_rate = float((trades["pnl"] > 0).mean()) if isinstance(trades, pd.DataFrame) and "pnl" in trades else 0.0

    return {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "MaxDD": max_dd,
        "WinRate": win_rate,
        "AnnualReturn": float(cagr),
        "AnnualVol": float(ann_vol),
        "FinalCapital": float(eq.iloc[-1]),
    }

def _safe_calmar(eq: pd.Series, dd_floor: float = 0.05, calmar_cap: float = 10.0) -> float:
    eq = eq.dropna()
    if len(eq) < 3:
        return -1.0
    # CAGR por años reales (robusto a huecos)
    SEC_PER_YEAR = 365.25 * 24 * 3600
    years = (eq.index[-1] - eq.index[0]).total_seconds() / SEC_PER_YEAR
    if years <= 0:
        return -1.0
    cagr = (eq.iloc[-1] / eq.iloc[0])**(1/years) - 1.0
    # DD con piso
    dd = (eq / eq.cummax()) - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    denom = max(abs(max_dd), dd_floor)  # piso evita inflar Calmar
    cal = cagr / denom if denom > 0 else 0.0
    # cap razonable para que no dominen outliers
    return float(np.clip(cal, -calmar_cap, calmar_cap))

def print_equity_debug(equity: pd.Series, trades: pd.DataFrame, label: str = "(val)") -> None:
    """
    Imprime valores clave de equity y trades con el formato solicitado.
    label: texto para distinguir si es (val), (test), (full), etc.
    """
    if equity is None or len(equity.dropna()) == 0:
        print(f"[DEBUG {label}] equity vacío.")
        return

    eq = equity.dropna()
    first = float(eq.iloc[0])
    last  = float(eq.iloc[-1])
    eqmax = float(eq.max())
    xfac  = last / first if first != 0 else float("nan")

    # Max drawdown sobre la serie entregada
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0

    n_tr = int(trades.shape[0]) if isinstance(trades, pd.DataFrame) else 0

    print(f"equity[0] : {first}")
    print(f"equity[-1]: {last}")
    print(f"equity max: {eqmax}")
    print(f"x-factor  : {xfac}")
    print(f"max drawdown {label}: {max_dd}")
    print(f"n_trades {label}: {n_tr}")


def returns_tables(equity: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tablas de retornos Mensual / Trimestral / Anual a partir de equity.
    Re-muestrea a diario (último valor del día) para evitar sesgos intradía.
    """
    if equity is None or len(equity.dropna()) < 2:
        empty = pd.DataFrame()
        return empty, empty, empty

    # Serie diaria y retornos diarios
    eq_d = equity.resample("D").last().dropna()
    rets_d = eq_d.pct_change().dropna()

    # Agregaciones de rendimiento compuesto
    monthly   = rets_d.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    quarterly = rets_d.resample("QE").apply(lambda x: (1.0 + x).prod() - 1.0)
    annual    = rets_d.resample("YE").apply(lambda x: (1.0 + x).prod() - 1.0)

    # Tabla mensual Year x Month(1..12)
    m_tbl = monthly.to_frame("ret")
    m_tbl["Year"]  = m_tbl.index.year
    m_tbl["Month"] = m_tbl.index.month
    monthly_tbl = (
        m_tbl.pivot(index="Year", columns="Month", values="ret")
            .reindex(columns=range(1, 13))
            .sort_index()
    )

    # Tabla trimestral Year x Quarter(1..4)
    q_tbl = quarterly.to_frame("ret")
    q_tbl["Year"]    = q_tbl.index.year
    q_tbl["Quarter"] = q_tbl.index.quarter
    quarterly_tbl = (
        q_tbl.pivot(index="Year", columns="Quarter", values="ret")
            .reindex(columns=range(1, 5))
            .sort_index()
    )

    # Tabla anual
    annual_tbl = annual.to_frame("Annual Return")
    annual_tbl.index = annual_tbl.index.year
    annual_tbl = annual_tbl.sort_index()

    return monthly_tbl, quarterly_tbl, annual_tbl