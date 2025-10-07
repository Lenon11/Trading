from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class DataCfg:
    csv_path: Path = ROOT / "Data" / "aapl_5m_train.csv"
    datetime_col: str = "Date"
    price_col: str = "Close"

@dataclass(frozen=True)
class StratCfg:
    rsi_window: int = 15
    rsi_lower: int = 30
    rsi_upper: int = 70
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_window: int = 14
    adx_min: float = 20.0

@dataclass(frozen=True)
class BtCfg:
    fee: float = 0.00125
    initial_capital: float = 1_000_000.0
    allow_short: bool = True
    short_mode: str = "no_leverage"  # dejamos la bandera, pero SIN crédito
    stop_loss: float = 0.06
    take_profit: float = 0.06
    n_shares: int = 50