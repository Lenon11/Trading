# Report/__init__.py

__all__ = [
    # submódulos
    "backtest", "config", "indicators", "io_data", "metrics",
    "optimization", "plotting", "position", "signals", "utils", "main",
    # atajos de funciones clave:
    "run_all", "run_optuna", "generate_signals", "apply_regime_filter",
    "backtest_fn", "compute_metrics", "returns_tables",
]

# ===== Atajos convenientes para import directo =====
from main import run_all
from optimization import run_optuna
from signals import generate_signals, apply_regime_filter
from backtest import backtest as backtest_fn
from metrics import compute_metrics, returns_tables
