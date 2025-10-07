# =========================================================
# Utils: anualización, splits temporales y walk-forward
# =========================================================
from __future__ import annotations
import numpy as np
import pandas as pd

def infer_ppy(index: pd.DatetimeIndex) -> float:
    """
    Infer periods-per-year (PPY) para anualizar métricas.
    - Usa el delta mediano del índice (robusto a huecos).
    - 'Snapea' a valores típicos: 1m, 5m, 15m, 30m, 1h, 4h, diario(252), semanal(52), mensual(12).
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return 252.0

    # Asegura índice ordenado y sin tz para el cálculo de deltas
    idx = index.sort_values()
    if idx.tz is not None:
        # convertir a UTC y quitar tz para evitar sorpresas con .total_seconds()
        idx = idx.tz_convert("UTC").tz_localize(None)

    # Paso mediano en segundos (robusto a outliers y huecos)
    diffs = pd.Series(idx).diff().dropna()
    if diffs.empty:
        return 252.0
    step_sec = float(pd.to_timedelta(diffs.median()).total_seconds())
    if not np.isfinite(step_sec) or step_sec <= 0:
        return 252.0

    SEC_PER_YEAR = 365.25 * 24 * 3600

    # Candidatos típicos: (segundos por barra, ppy deseado)
    # Nota: para diario devolvemos 252
    candidates = [
        (60,        SEC_PER_YEAR / 60.0),        # 1m  -> ~525,600
        (5*60,      SEC_PER_YEAR / (5*60.0)),    # 5m  -> ~105,120
        (15*60,     SEC_PER_YEAR / (15*60.0)),   # 15m -> ~35,040
        (30*60,     SEC_PER_YEAR / (30*60.0)),   # 30m -> ~17,520
        (3600,      8760.0),                     # 1h  -> 8,760
        (4*3600,    2190.0),                     # 4h  -> 2,190
        (24*3600,   252.0),                      # 1d  -> 252 (días hábiles)
        (7*24*3600, 52.0),                       # 1w  -> 52
        (30*24*3600,12.0),                       # 1m ~> 12
    ]

    # Si el step mediano está cerca de un candidato, usar su PPY “snapeado”
    tol = 0.10  # 10% de tolerancia
    for sec, ppy_snap in candidates:
        if abs(step_sec - sec) / sec <= tol:
            return float(ppy_snap)

    # Si no encaja, devolver cálculo continuo
    return float(SEC_PER_YEAR / step_sec)


def temporal_split_60_20_20(df: pd.DataFrame):
    """Split global: 60% Train, 20% Test, 20% Validation (orden temporal)."""
    df = df.copy()
    n  = len(df)
    i1 = int(n*0.6)
    i2 = int(n*0.8)
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def make_walkforward_chunks(df: pd.DataFrame, n_chunks: int, min_points: int = 1000):
    """
    Genera n_chunks secuenciales; cada chunk se parte 60/20/20 internamente.
    Se exige una longitud mínima (min_points) para evitar splits diminutos.
    """
    df = df.copy()
    n = len(df)
    if n < min_points:
        # en datasets cortos, al menos 1 chunk
        yield temporal_split_60_20_20(df)
        return

    step = max(min_points, n // n_chunks)
    for start in range(0, n - min_points + 1, step):
        end = min(n, start + step)
        sub = df.iloc[start:end]
        if len(sub) >= min_points:
            yield temporal_split_60_20_20(sub)

def _duration_years(dt_index: pd.DatetimeIndex) -> float:
    if not isinstance(dt_index, pd.DatetimeIndex) or len(dt_index) < 2:
        return 1.0
    SEC_PER_YEAR = 365.25 * 24 * 3600
    return max((dt_index[-1] - dt_index[0]).total_seconds() / SEC_PER_YEAR, 1e-6)
