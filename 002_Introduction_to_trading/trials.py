# trials.py
SEED_TRIALS = [
    {
        "risk_fraction": 0.1252595478810879,
        "SL": 0.046425109072137656, "TP": 0.10600981581711783,
        "rsi_window": 21, "rsi_lower": 27, "rsi_upper": 73,
        "macd_fast": 11, "macd_slow": 24, "macd_signal": 8,
        "adx_window": 22, "adx_threshold": 39,
        "bb_window": 21, "bb_dev": 2.4,
        "stoch_window": 11, "stoch_smooth": 4,
        "breakeven_trigger": 0.3114421972346785,
        "trail_atr_mult": 0.6496657165849747,
        "time_stop_bars": 34,
        "use_target_vol": False,
    },
    # … agrega más
]

# Defaults “de experimento”
DEFAULT_N_TRIALS = 10
DEFAULT_N_CHUNKS = 6
DEFAULT_SEED = 42
