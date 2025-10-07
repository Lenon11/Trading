import numpy as np
import pandas as pd
from pathlib import Path

# =======================
# Carga / Limpieza robusta
# =======================

def clean_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza conservadora:
    - Normaliza nombres de columnas.
    - Quita 'Unix' si existe.
    - Filtra de nuevo cabeceras re-embebidas (doble defensa).
    - Parsea fechas.
    - Convierte OHLC a numérico con coerce (no rompe si vienen strings).
    - Elimina filas donde falte cualquier OHLC (pero NO requiere Volume).
    - Ordena por fecha.
    - Si hay timestamps duplicados, conserva el último (keep='last').
    """
    out = df.copy()

    # 0) Normaliza nombres
    out.columns = out.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

    # 1) Columnas básicas
    out = out.drop(columns=["Unix"], errors="ignore")
    if "Date" not in out.columns:
        raise ValueError("clean_timeseries: no se encontró la columna 'Date'.")

    # 2) Defensa extra: quita filas que son cabeceras re-embebidas
    s = out["Date"].astype(str).str.strip()
    mask_date_like = (
        s.str.match(r"^\d{4}-\d{2}-\d{2}") |
        s.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}") |
        s.str.match(r"^\d{9,}$") |
        s.str.match(r"^\d{4}\d{2}\d{2}")
    )
    out = out.loc[mask_date_like].copy()

    # 3) Parseo de fecha
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).set_index("Date").sort_index()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(None)

    # 4) Tipado numérico
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])

    # 5) Timestamps duplicados → conserva el último
    out = out[~out.index.duplicated(keep="last")]

    return out.sort_index()

def diagnose_index(df: pd.DataFrame, name="DF"):
    dup_mask = df.index.duplicated(keep=False)
    print(f"[{name}] Duplicados:", int(dup_mask.sum()))
    # Gaps ≠ 1h
    if len(df.index) > 1:
        deltas = np.diff(df.index.values).astype("timedelta64[h]").astype(int)
        print(f"[{name}] Gaps ≠ 1h:", int((deltas != 1).sum()))
    else:
        print(f"[{name}] Gaps ≠ 1h: 0")


# --- Opcional: auto-prueba manual (NO se ejecuta al importar) ---
if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    csv_path = BASE / "Data" / "Binance_BTCUSDT_1h.csv"
    print(f"[SELFTEST] Leyendo: {csv_path}")
    raw = pd.read_csv(csv_path, header=1)
    data = clean_timeseries(raw)
    diagnose_index(raw, "RAW")
    diagnose_index(data, "CLEAN")
    print("Rango limpio:", data.index.min(), "→", data.index.max(), f"({len(data)} filas)")
