from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class Position:
    side: str                # 'long' o 'short'
    entry_time: pd.Timestamp
    entry_price: float
    qty: float               # >0 long, <0 short (abs(qty) = nº “shares”)
    sl: float
    tp: float
    reserve: float = 0.0     # colateral reservado para short no_leverage
