from __future__ import annotations

import argparse
import optuna
import pandas as pd

from optimization import objective, _extract_mgmt_kwargs
from signals import generate_signals, apply_regime_filter
from backtest import backtest
from metrics import compute_metrics, returns_tables, print_equity_debug
from utils import temporal_split_60_20_20
from plotting import plot_validation_equity, plot_equity_train_test_val


def run_all(
    data: pd.DataFrame,
    n_trials: int = 400,
    n_chunks: int = 7,
    n_jobs: int = 1,                   # recomendable 1 por estabilidad/consistencia
    plot_validation: bool = False,
    # --- toggles/overrides coherentes con backtest ---
    allow_short: bool = True,          # Shorts ACTIVOS por defecto
    short_mode: str = "no_leverage",   # "no_leverage" (seguro) o "credit"
    use_mgmt_extras: bool = True,      # Extras de gestión ACTIVOS por defecto
    mgmt_overrides: dict | None = None,
    initial_capital: float = 1_000_000.0,
    fee: float = 0.00125,
    # --- semillado / seeds para TPE ---
    enqueue_seeds: bool = True,
    seed: int = 42,
    seed_trials: list[dict] | None = None,  # ← pasa aquí dicts para study.enqueue_trial(...)
):
    """
    1) Optuna (walk-forward) maximizando el score de `objective` (centrado en Validation).
    2) Evalúa best_params en 60/20/20 → Validation.
    3) Imprime hiperparámetros, métricas y tablas.

    Notas:
      - Shorts: ON por defecto (modo “no_leverage”).
      - Gestión avanzada: ON por defecto (breakeven, trailing, time-stop, target_vol si vino en params).
      - Puedes pasar `seed_trials=[{...}, {...}]` para encolar puntos iniciales conocidos.
    """
    # --- Copia + saneo de índice ---
    data = data.copy().sort_index()
    if not data.index.is_unique:
        data = data[~data.index.duplicated(keep="first")]

    # Silenciar warnings experimentales de Optuna si están presentes
    import warnings
    try:
        from optuna._experimental import ExperimentalWarning
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
    except Exception:
        pass

    # === Sampler y pruner ===
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=True,
        group=True,
        constant_liar=True
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=80, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    # === Enqueue de seeds (si se proporcionan) ===
    if enqueue_seeds and seed_trials:
        for d in seed_trials:
            study.enqueue_trial(d)

    # === Optimización ===
    study.optimize(
        lambda t: objective(t, data, n_chunks=n_chunks, short_mode=short_mode),
        n_trials=n_trials,
        n_jobs=n_jobs
    )
    best = study.best_params

    # Normaliza nombres SL/TP
    stop_loss   = best.get("stop_loss", best.get("SL"))
    take_profit = best.get("take_profit", best.get("TP"))
    if stop_loss is None or take_profit is None:
        raise KeyError("Faltan 'stop_loss/take_profit' (o 'SL/TP') en best_params.")
    if "risk_fraction" not in best:
        raise KeyError("Falta 'risk_fraction' en best_params.")

    # === Gestión avanzada (ON por defecto) ===
    if use_mgmt_extras:
        mgmt = _extract_mgmt_kwargs(best)
        if mgmt_overrides:
            mgmt.update({k: v for k, v in mgmt_overrides.items() if k in mgmt})
    else:
        mgmt = dict(breakeven_trigger=0.0, trail_atr_mult=0.0, time_stop_bars=0,
                    atr_window_for_trail=14, target_vol=None, vol_lookback=24*30, max_rf_scale=3.0)

    # === Split 60/20/20 para el reporte final (Validation 20%) ===
    train_df, test_df, val_df = temporal_split_60_20_20(data)

    # Señales SOLO sobre Validation
    val_sig = generate_signals(
        val_df,
        rsi_window=best["rsi_window"], rsi_lower=best["rsi_lower"], rsi_upper=best["rsi_upper"],
        macd_fast=best["macd_fast"], macd_slow=best["macd_slow"], macd_signal=best["macd_signal"],
        adx_window=best["adx_window"], adx_threshold=best["adx_threshold"],
        bb_window=best["bb_window"], bb_dev=best["bb_dev"],
        stoch_window=best["stoch_window"], stoch_smooth=best["stoch_smooth"],
        use_directional_adx=True
    )
    # Filtro de régimen SOLO en evaluación final (no en objective → evita leakage)
    try:
        val_sig = apply_regime_filter(val_sig, sma_window=200)
    except Exception:
        pass

    # === Backtest en Validation (coherente con objective) ===
    equity, trades, final_cap = backtest(
        data_sig=val_sig,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_fraction=best["risk_fraction"],
        fee=fee,
        initial_capital=initial_capital,
        allow_short=allow_short,
        short_mode=short_mode,
        **mgmt
    )

    # Métricas y tablas
    metrics = compute_metrics(equity, trades)
    monthly_tbl, quarterly_tbl, annual_tbl = returns_tables(equity)
    print_equity_debug(equity, trades, label="(val)")

    # (OPCIONAL) Plot de Validation (usamos función existente)
    if plot_validation:
        plot_validation_equity(
            equity_val=equity,
            title="Portfolio Over Time (Validation)",
            resample="1D",
            logy=False,
            show_grid=True,
            init_capital=initial_capital,
            show_drawdown=True,
        )

    # === Prints resumen ===
    print("\n=== Best value (Optuna, avg score in walk-forward) ===")
    print(study.best_value)

    print("\n=== Best Hyperparameters ===")
    for k, v in best.items():
        print(f"{k}: {v}")

    print("\n=== Validation Metrics (hold-out 20%) ===")
    for k, v in metrics.items():
        if k in ("AnnualReturn", "AnnualVol", "MaxDD"):
            print(f"{k}: {v:.2%}")
        elif k in ("Sharpe", "Sortino", "Calmar", "WinRate"):
            print(f"{k}: {v:.3f}")
        else:
            try:
                print(f"{k}: {v:,.2f}")
            except Exception:
                print(f"{k}: {v}")

    print("\n=== Monthly Returns (%) ===");    print((monthly_tbl * 100).round(2))
    print("\n=== Quarterly Returns (%) ===");  print((quarterly_tbl * 100).round(2))
    print("\n=== Annual Returns (%) ===");     print((annual_tbl * 100).round(2))

    return dict(
        study=study,
        study_best_value=study.best_value,
        best_params=best,
        equity=equity,
        trades=trades,
        metrics=metrics,
        final_capital=final_cap,
        monthly=monthly_tbl,
        quarterly=quarterly_tbl,
        annual=annual_tbl,
        mgmt_used=mgmt,
        allow_short_used=allow_short,
        short_mode_used=short_mode,
        initial_capital_used=initial_capital,
        fee_used=fee,
    )


# ===========================
# CLI para correr desde consola
# ===========================
def main():
    ap = argparse.ArgumentParser(description="Runner de backtest/optuna")
    ap.add_argument("--mode", choices=["runall"], default="runall")
    ap.add_argument("--csv", type=str, required=True, help="Ruta al CSV con OHLC (al menos Close)")
    ap.add_argument("--n_trials", type=int, default=400)
    ap.add_argument("--n_chunks", type=int, default=7)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--plot_validation", action="store_true")
    args = ap.parse_args()

    # Cargar datos
    df = pd.read_csv(args.csv)
    # Asumimos columna de fecha -> índice (ajusta si tu CSV usa otro nombre)
    for col in ["Date", "Datetime", "timestamp", "time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col).sort_index()
            break

    run_all(
        data=df,
        n_trials=args.n_trials,
        n_chunks=args.n_chunks,
        n_jobs=args.n_jobs,
        plot_validation=args.plot_validation,
    )


if __name__ == "__main__":
    main()
