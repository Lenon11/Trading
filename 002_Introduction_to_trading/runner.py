from __future__ import annotations

from pathlib import Path
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 0) Bootstrap de ruta
# ─────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
DATA_DIR = BASE / "Data"

# Ajustes visuales para reducir espacios y sangrías en ejes
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["axes.xmargin"] = 0.0
plt.rcParams["axes.ymargin"] = 0.02
plt.rcParams["figure.dpi"] = 110

# ─────────────────────────────────────────────────────────────
# 1) Imports locales (estructura plana)
# ─────────────────────────────────────────────────────────────
from main import run_all
from io_data import clean_timeseries, diagnose_index
from signals import generate_signals, apply_regime_filter
from backtest import backtest
from plotting import (
    plot_equity_train_test_val,
    plot_validation_equity,
    plot_indicators,
)
from utils import temporal_split_60_20_20
from trials import SEED_TRIALS, DEFAULT_N_TRIALS, DEFAULT_N_CHUNKS, DEFAULT_SEED


# ─────────────────────────────────────────────────────────────
# 2) Utilidades
# ─────────────────────────────────────────────────────────────
def load_csv_smart(csv_path: Path) -> pd.DataFrame:
    """
    Lector robusto para CSVs de Binance con cabeceras re-embebidas:
    - Lee con header=1 (formato típico)
    - Mantiene todo como string inicialmente
    - Quita filas que NO parecen fecha (cabeceras re-embebidas, líneas vacías, etc.)
    - Devuelve dataframe aún sin tipar (tipamos en clean_timeseries)
    """
    import pandas as pd
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {csv_path}")

    # 1) Lee TODO como string para poder filtrar sin que Pandas meta NaN/NaT antes
    df = pd.read_csv(csv_path, header=1, dtype=str, na_filter=False, low_memory=False)
    # limpia nombres de columnas (BOM, espacios)
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    if "Date" not in df.columns:
        raise ValueError("El CSV no tiene columna 'Date' (revisa el header correcto).")

    s = df["Date"].astype(str).str.strip()
    mask_date_like = (
        s.str.match(r"^\d{4}-\d{2}-\d{2}") |         # 2019-01-01...
        s.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}") |   # 1/1/2020...
        s.str.match(r"^\d{9,}$") |                   # epoch largo, por si acaso
        s.str.match(r"^\d{4}\d{2}\d{2}")             # 20190101 (sin guiones)
    )
    df = df.loc[mask_date_like].copy()

    # 3) (OPCIONAL robustez extra): si por algún motivo se coló una cabecera textual
    # porque coincida con el patrón de fecha (raro), bórrala comparando la fila con los nombres
    header_row_mask = (
        df["Date"].str.lower().eq("date") |
        (("Open" in df.columns) and df["Open"].str.lower().eq("open")) |
        (("High" in df.columns) and df["High"].str.lower().eq("high")) |
        (("Low"  in df.columns) and df["Low"].str.lower().eq("low")) |
        (("Close" in df.columns) and df["Close"].str.lower().eq("close"))
    )
    df = df.loc[~header_row_mask].copy()

    return df

def build_full_signals(best: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Genera señales en TODO el histórico con los mejores hiperparámetros + filtro de régimen."""
    sig = generate_signals(
        data,
        rsi_window=best["rsi_window"], rsi_lower=best["rsi_lower"], rsi_upper=best["rsi_upper"],
        macd_fast=best["macd_fast"], macd_slow=best["macd_slow"], macd_signal=best["macd_signal"],
        adx_window=best["adx_window"], adx_threshold=best["adx_threshold"],
        bb_window=best["bb_window"], bb_dev=best["bb_dev"],
        stoch_window=best["stoch_window"], stoch_smooth=best["stoch_smooth"],
        use_directional_adx=True,
    )
    sig = apply_regime_filter(sig, sma_window=200)
    return sig


def run_full_backtest(best: dict, data: pd.DataFrame, results: dict):
    """Backtest de TODO el histórico coherente con la evaluación de Validation."""
    full_sig = build_full_signals(best, data)
    stop_loss   = best.get("stop_loss", best.get("SL"))
    take_profit = best.get("take_profit", best.get("TP"))
    equity_full, trades_full, _ = backtest(
        data_sig=full_sig,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_fraction=best["risk_fraction"],
        fee=results.get("fee_used", 0.00125),
        initial_capital=results.get("initial_capital_used", 1_000_000.0),
        allow_short=results.get("allow_short_used", True),
        short_mode=results.get("short_mode_used", "no_leverage"),
        **results.get("mgmt_used", dict(breakeven_trigger=0.0, trail_atr_mult=0.0, time_stop_bars=0, target_vol=None)),
    )
    return full_sig, equity_full, trades_full


def apply_zero_xmargins_to_all_axes():
    """Fuerza márgenes X=0 en todas las figuras/axes ya creados."""
    for num in plt.get_fignums():
        fig = plt.figure(num)
        for ax in fig.get_axes():
            try:
                ax.margins(x=0)
                # opcional: fija límites exactos si aún ves 'espacio' al inicio
                xdata = None
                for line in ax.get_lines():
                    xd = getattr(line, "get_xdata", None)
                    if xd:
                        x = line.get_xdata()
                        if len(x):
                            xdata = x
                            break
                if xdata is not None:
                    ax.set_xlim(xdata.min(), xdata.max())
            except Exception:
                pass
        try:
            fig.tight_layout()
        except Exception:
            pass


def print_spans(data: pd.DataFrame, full_sig: pd.DataFrame, equity_full: pd.Series) -> None:
    def _span(df): return df.index.min(), df.index.max(), len(df)
    print("\nRANGOS:")
    print("DATA   :", _span(data))
    print("FULLSIG:", _span(full_sig))
    print("EQUITY :", equity_full.index.min(), equity_full.index.max(), len(equity_full))


# ─────────────────────────────────────────────────────────────
# 3) CLI
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Runner del pipeline de trading")
    ap.add_argument("--csv", default=str(DATA_DIR / "Binance_BTCUSDT_1h.csv"),
                    help="Ruta al CSV (por defecto: Data/Binance_BTCUSDT_1h.csv)")
    ap.add_argument("--n_trials", type=int, default=DEFAULT_N_TRIALS, help="N° de trials Optuna")
    ap.add_argument("--n_chunks", type=int, default=DEFAULT_N_CHUNKS, help="N° de chunks walk-forward")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla del sampler")
    ap.add_argument("--no_seeds", action="store_true", help="No encolar SEED_TRIALS")
    ap.add_argument("--plots", action="store_true", help="Generar y mostrar gráficos")
    ap.add_argument("--block", action="store_true", help="Si se pasa, plt.show() será bloqueante")
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────
# 4) Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 4.1 Carga + limpieza
    csv_path = Path(args.csv)
    print(f"[INFO] CSV: {csv_path}")
    raw = load_csv_smart(csv_path)
    data = clean_timeseries(raw)
    diagnose_index(data, "CLEAN")
    print("Rango limpio:", data.index.min(), "→", data.index.max(), f"({len(data)} filas)")

    # 4.2 Optuna + evaluación
    results = run_all(
        data=data,
        n_trials=args.n_trials,
        n_chunks=args.n_chunks,
        n_jobs=1,
        allow_short=True,
        short_mode="no_leverage",
        use_mgmt_extras=True,
        enqueue_seeds=not args.no_seeds,
        seed=args.seed,
        seed_trials=None if args.no_seeds else SEED_TRIALS,
        plot_validation=False,   # controlamos los plots aquí
    )
    best = results["best_params"]
    print("\n[INFO] Best params:")
    for k, v in best.items():
        print(f"  {k}: {v}")

    # 4.3 Backtest full histórico
    full_sig, equity_full, trades_full = run_full_backtest(best, data, results)
    print(f"\n[INFO] Trades (full): {len(trades_full)}")
    print(f"[INFO] Equity final (full): {equity_full.iloc[-1]:,.2f}")

    # 4.4 RANGOS
    print("\nRANGOS:")
    def _span(df): return df.index.min(), df.index.max(), len(df)
    print("DATA   :", _span(data))
    print("FULLSIG:", _span(full_sig))
    print("EQUITY :", equity_full.index.min(), equity_full.index.max(), len(equity_full))

    # 4.5 Plots (un solo show bloqueante)
    if args.plots:
        _ = plot_equity_train_test_val(
            equity_full,
            *temporal_split_60_20_20(data),
            drawdown_years=None, logy=False, show=False,
        )
        _ = plot_validation_equity(
            results["equity"],
            title="Validation (20% hold-out)",
            resample="1D",
            logy=False,
            init_capital=results.get("initial_capital_used", 1_000_000.0),
            show_drawdown=True,
            show=False,
        )
        _ = plot_indicators(full_sig, best, show=False)

        # reduce márgenes y ABRE las ventanas hasta que cierres manualmente
        apply_zero_xmargins_to_all_axes()
        import matplotlib.pyplot as plt
        plt.ioff()   # asegúrate de modo no-interactivo (evita autocierre en algunos backends)
        plt.show()   # <<< BLOQUEANTE, mantiene las ventanas abiertas

if __name__ == "__main__":
    main()
