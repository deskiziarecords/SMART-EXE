import MetaTrader5 as mt5
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, PAIR

def connect():
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize() failed")
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

def get_bar() -> dict | None:
    """Returns latest closed M1 bar as OHLCV dict."""
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
    tick = mt5.symbol_info_tick(PAIR)
    return tick.bid if tick else 0.0
