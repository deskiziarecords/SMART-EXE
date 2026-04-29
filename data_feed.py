import random
import time
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, PAIR, MOCK_MODE

def connect():
    if MOCK_MODE or not MT5_AVAILABLE:
        print(f"  [DATA_FEED] Running in MOCK MODE (MT5 Available: {MT5_AVAILABLE})")
        return

    if not mt5.initialize():
        raise RuntimeError("MT5 initialize() failed")
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

def get_bar() -> dict | None:
    """Returns latest closed M1 bar as OHLCV dict."""
    if MOCK_MODE or not MT5_AVAILABLE:
        # Generate synthetic bar
        base = 1.0850 + (random.random() - 0.5) * 0.01
        return {
            'open':   base,
            'high':   base + random.random() * 0.001,
            'low':    base - random.random() * 0.001,
            'close':  base + (random.random() - 0.5) * 0.001,
            'volume': random.randint(10, 100),
            'time':   int(time.time()),
        }

    rates = mt5.copy_rates_from_pos(PAIR, mt5.TIMEFRAME_M1, 0, 2)
    if rates is None or len(rates) < 2:
        return None
    # index 1 = last fully closed bar, index 0 = current forming bar
    bar = rates[1]
    return {
        'open':   float(bar['open']),
        'high':   float(bar['high']),
        'low':    float(bar['low']),
        'close':  float(bar['close']),
        'volume': int(bar['tick_volume']),
        'time':   int(bar['time']),
    }

def get_price() -> float:
    if MOCK_MODE or not MT5_AVAILABLE:
        return 1.0850 + (random.random() - 0.5) * 0.01

    tick = mt5.symbol_info_tick(PAIR)
    return tick.bid if tick else 0.0
