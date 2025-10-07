# =========================================================
# 4) Optimización con Optuna — Walk-Forward conservador
# =========================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import NopPruner
from typing import Dict, Tuple, Callable

from signals import generate_signals, apply_regime_filter
from backtest import backtest
from metrics import compute_metrics, _safe_calmar
from utils import make_walkforward_chunks

FEE = 0.00125  # comisión por lado
INITIAL_CAPITAL = 1_000_000.0


# -----------------------------
# Sugerencia de hiperparámetros
# -----------------------------
def suggest_hyperparams(trial: optuna.trial.Trial) -> dict:
    params = {
        # Gestión de capital (conservador)
        "risk_fraction": trial.suggest_float("risk_fraction", 0.08, 0.13),

        # SL / TP
        "stop_loss":   trial.suggest_float("SL", 0.030, 0.05),
        "take_profit": trial.suggest_float("TP", 0.07, 0.130),

        # Indicadores
        "rsi_window":    trial.suggest_int('rsi_window', 18, 40),
        "rsi_lower":     trial.suggest_int('rsi_lower', 20, 28),
        "rsi_upper":     trial.suggest_int('rsi_upper', 72, 82),

        "macd_fast":     trial.suggest_int('macd_fast', 10, 12),
        "macd_slow":     trial.suggest_int('macd_slow', 24, 30),
        "macd_signal":   trial.suggest_int('macd_signal', 8, 10),

        "adx_window":    trial.suggest_int('adx_window', 18, 28),
        "adx_threshold": trial.suggest_int('adx_threshold', 28, 40),

        "bb_window":     trial.suggest_int('bb_window', 20, 26),
        "bb_dev":        trial.suggest_float('bb_dev', 2, 2.6, step=0.1),

        "stoch_window":  trial.suggest_int('stoch_window', 10, 14),
        "stoch_smooth":  trial.suggest_int('stoch_smooth', 2, 4),
    }

    # Gestión avanzada
    params["breakeven_trigger"] = trial.suggest_float("breakeven_trigger", 0.2, 0.45)
    params["trail_atr_mult"]    = trial.suggest_float("trail_atr_mult", 0.25, 0.65)
    params["time_stop_bars"]    = trial.suggest_int("time_stop_bars", 20, 36)

    use_tv = trial.suggest_categorical("use_target_vol", [False, True])
    params["target_vol"] = trial.suggest_float("target_vol", 0.09, 0.11) if use_tv else None

    # Fijos
    params["atr_window_for_trail"] = 14
    params["vol_lookback"]         = 24*30
    params["max_rf_scale"]         = 3.0
    return params


# -----------------------------
# Solo kwargs de gestión para backtest
# -----------------------------
def _extract_mgmt_kwargs(p: dict) -> dict:
    """Sólo kwargs de gestión que acepta el backtest."""
    return dict(
        breakeven_trigger   = p.get("breakeven_trigger", 0.0),
        trail_atr_mult      = p.get("trail_atr_mult", 0.0),
        time_stop_bars      = p.get("time_stop_bars", 0),
        atr_window_for_trail= p.get("atr_window_for_trail", 14),
        target_vol          = p.get("target_vol", None),
        vol_lookback        = p.get("vol_lookback", 24*30),
        max_rf_scale        = p.get("max_rf_scale", 3.0),
    )


# -----------------------------
# Objetivo Walk-Forward (tu lógica)
# -----------------------------
def objective(trial: optuna.trial.Trial, data: pd.DataFrame, n_chunks: int = 5, short_mode: str = "no_leverage"):
    """
    Walk-forward conservador, centrado en VALIDATION:
      - Reglas duras por chunk (VAL):
          * Calmar >= 0.0
          * Sharpe >= 0.0
          * MaxDD >= -12%
          * n_trades >= 60
          * FinalCapital > inicial  (DEBE ganar dinero)
      - Score por chunk:
          cal_v
        + 0.25 * cal_t
        + 0.03 * sharpe_val
        + 0.20 * AnnualReturn_VAL     # incentivo por retorno absoluto
        + 0.50 * min(0, AnnualReturn_VAL - AnnualVol_VAL)   # castiga fuerte Ret < Vol
        + 0.20 * max(0, WinRate_VAL - 0.45)                 # bonus solo por encima de 45%
        - 0.60 si signo Calmar_VAL != Calmar_TEST
        - penalizaciones por DD extremos (TEST/VAL < -20%)
        - penalización por RF > 10%
        - penalización por RR fuera de [1.35, 2.40]
        - penalización por sobre-operar (trades/año > 250)
      - Score global: mean - 0.15*std
      - Requiere >=60% de chunks aprobados; si no, devuelve -1e9.
      - HARD CAP: risk_fraction > 0.14 → rechazo inmediato del trial.
    """
    # --- Sugerencia de hiperparámetros y normalización SL/TP ---
    p = suggest_hyperparams(trial)
    p["stop_loss"]   = p.get("stop_loss", p.get("SL"))
    p["take_profit"] = p.get("take_profit", p.get("TP"))
    if p["stop_loss"] is None or p["take_profit"] is None:
        return -1e9

    mgmt = _extract_mgmt_kwargs(p)

    # ======= Constantes de control =======
    PASS_RATIO_MIN = 0.60          # 60% de chunks deben aprobar
    MIN_TRADES_VAL = 60
    DD_LIMIT       = -0.12         # DD mínimo aceptable en VAL
    CALMAR_MIN     = 0.10          # pedimos Calmar >= 0.1 en VAL
    LAMBDA_STD     = 0.15
    W_SHARPE       = 0.03
    CALMAR_CAP     = 2.5           # cap bajo para evitar “calmares” irreales
    DD_FLOOR       = 0.10          # floor para calmar_safe

    # RR / RF / WR
    RR_LOW, RR_HIGH    = 1.35, 2.40
    RR_PENALTY_SLOPE   = 0.70
    RF_CAP_HARD        = 0.14
    RF_CAP_SOFT        = 0.10
    RF_PENALTY_SLOPE   = 10.0
    WR_BASELINE        = 0.45
    WR_BONUS_SLOPE     = 0.20

    # Overtrading
    SEC_PER_YEAR        = 365.25 * 24 * 3600
    TRADES_PER_YEAR_CAP = 250
    TPA_PEN_SLOPE       = 0.004

    # Otros
    DD_WARN             = -0.20
    SIGN_MISMATCH_PEN   = 0.60

    # HARD cap RF
    if float(p["risk_fraction"]) > RF_CAP_HARD:
        return -1e6

    scores   = []
    approved = 0

    for train_df, test_df, val_df in make_walkforward_chunks(data, n_chunks=n_chunks):
        # ---------- TEST ----------
        test_sig = generate_signals(
            test_df,
            rsi_window=p["rsi_window"], rsi_lower=p["rsi_lower"], rsi_upper=p["rsi_upper"],
            macd_fast=p["macd_fast"], macd_slow=p["macd_slow"], macd_signal=p["macd_signal"],
            adx_window=p["adx_window"], adx_threshold=p["adx_threshold"],
            bb_window=p["bb_window"], bb_dev=p["bb_dev"],
            stoch_window=p["stoch_window"], stoch_smooth=p["stoch_smooth"],
            use_directional_adx=True
        )
        try:
            test_sig = apply_regime_filter(test_sig, sma_window=200)
        except Exception:
            pass

        eq_t, tr_t, _ = backtest(
            data_sig=test_sig,
            stop_loss=p["stop_loss"], take_profit=p["take_profit"],
            risk_fraction=p["risk_fraction"], fee=FEE,
            initial_capital=INITIAL_CAPITAL, allow_short=True,
            short_mode=short_mode, **mgmt
        )
        m_t   = compute_metrics(eq_t, tr_t)
        cal_t = _safe_calmar(eq_t, dd_floor=DD_FLOOR, calmar_cap=CALMAR_CAP)
        dd_t  = float(m_t.get("MaxDD", 0.0))

        # ---------- VALIDATION ----------
        val_sig = generate_signals(
            val_df,
            rsi_window=p["rsi_window"], rsi_lower=p["rsi_lower"], rsi_upper=p["rsi_upper"],
            macd_fast=p["macd_fast"], macd_slow=p["macd_slow"], macd_signal=p["macd_signal"],
            adx_window=p["adx_window"], adx_threshold=p["adx_threshold"],
            bb_window=p["bb_window"], bb_dev=p["bb_dev"],
            stoch_window=p["stoch_window"], stoch_smooth=p["stoch_smooth"],
            use_directional_adx=True
        )
        try:
            val_sig = apply_regime_filter(val_sig, sma_window=200)
        except Exception:
            pass

        eq_v, tr_v, _ = backtest(
            data_sig=val_sig,
            stop_loss=p["stop_loss"], take_profit=p["take_profit"],
            risk_fraction=p["risk_fraction"], fee=FEE,
            initial_capital=INITIAL_CAPITAL, allow_short=True,
            short_mode=short_mode, **mgmt
        )
        m_v   = compute_metrics(eq_v, tr_v)
        cal_v = _safe_calmar(eq_v, dd_floor=DD_FLOOR, calmar_cap=CALMAR_CAP)
        dd_v  = float(m_v.get("MaxDD", 0.0))
        sh_v  = float(m_v.get("Sharpe", 0.0))
        wr_v  = float(m_v.get("WinRate", 0.0))
        ar_v  = float(m_v.get("AnnualReturn", 0.0))
        av_v  = float(m_v.get("AnnualVol", 0.0))
        fc_v  = float(m_v.get("FinalCapital", INITIAL_CAPITAL))
        ntr_v = int(tr_v.shape[0]) if isinstance(tr_v, pd.DataFrame) else 0

        # ---------- Reglas duras ----------
        ok = (
            np.isfinite(sh_v) and np.isfinite(cal_v) and
            cal_v >= CALMAR_MIN and sh_v >= 0.0 and
            dd_v >= DD_LIMIT and
            ntr_v >= MIN_TRADES_VAL and
            fc_v > INITIAL_CAPITAL
        )
        if not ok:
            scores.append(-1.0)
            continue
        approved += 1

        # ---------- Score por chunk ----------
        score  = cal_v
        score += 0.25 * cal_t
        score += W_SHARPE * sh_v
        score += 0.20 * ar_v

        # Ret < Vol → penaliza fuerte
        if av_v > ar_v:
            score += 0.50 * (ar_v - av_v)

        # Signos distintos entre Test y Val
        if np.sign(cal_v) != np.sign(cal_t):
            score -= SIGN_MISMATCH_PEN

        # DD extremos (TEST/VAL < -20%)
        if dd_t < DD_WARN:
            score += (dd_t - DD_WARN)   # (negativo) resta
        if dd_v < DD_WARN:
            score += (dd_v - DD_WARN)

        # RR fuera de [1.35, 2.40]
        tp = float(p["take_profit"]); sl = float(p["stop_loss"])
        rr = tp / max(sl, 1e-12)
        if rr < 1e-12:
            return -1e9
        if rr < 1.35:
            score -= 0.70 * (1.35 - rr)
        elif rr > 2.40:
            score -= 0.70 * (rr - 2.40)

        # RF > 10% penaliza (soft cap)
        rf = float(p["risk_fraction"])
        score -= 10.0 * max(0.0, rf - 0.10)

        # Sobre-operar (trades/año)
        try:
            years_val = (eq_v.index[-1] - eq_v.index[0]).total_seconds() / SEC_PER_YEAR
            if years_val > 0:
                tpa = ntr_v / years_val
                if tpa > TRADES_PER_YEAR_CAP:
                    score -= 0.004 * (tpa - TRADES_PER_YEAR_CAP)
        except Exception:
            pass

        # Bonus por WR > 45%
        if wr_v > 0.45:
            score += 0.20 * (wr_v - 0.45)

        scores.append(float(score))

    # ---------- Aprobación global ----------
    pass_ratio = approved / max(1, n_chunks)
    if pass_ratio < 0.60 or not scores:
        return -1e9

    mean_s = float(np.mean(scores))
    std_s  = float(np.std(scores)) if len(scores) > 1 else 0.0
    return mean_s - 0.15 * std_s


# -----------------------------
# Runner del estudio (usa objective anterior)
# -----------------------------
def run_optuna(
    data: pd.DataFrame,
    n_trials: int = 50,
    seed: int = 42,
    n_chunks: int = 5,
    short_mode: str = "no_leverage",
) -> optuna.study.Study:
    """
    Ejecuta el estudio con TPE + NopPruner sobre el objetivo walk-forward.
    - data: DataFrame con al menos 'Close' (idealmente 'High' y 'Low' también).
    - n_chunks: nº de bloques para walk-forward.
    """
    sampler = TPESampler(seed=seed)
    pruner = NopPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(
        lambda tr: objective(tr, data=data, n_chunks=n_chunks, short_mode=short_mode),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study