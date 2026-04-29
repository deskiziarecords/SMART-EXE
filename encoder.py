"""
encoder.py — SMART-EXE Candle Language Encoder
================================================
Converts OHLCV bars to the 7-symbol alphabet.

Symbols:
  B — Bullish strong  (body > 60% of range, close > open)
  I — Bearish strong  (body > 60% of range, close < open)
  U — Bullish weak    (body 10–60%, close > open)
  D — Bearish weak    (body 10–60%, close < open)
  W — Upper wick trap (upper wick > 60% of range)
  w — Lower wick trap (lower wick > 60% of range)
  X — Doji / indecision (body < 10% of range)

This matches the annotated_candles.csv labels exactly.
"""
from enum import Enum
import numpy as np

# Forward and reverse maps — consistent with QUIMERIA SMK + training data
SYMBOL_MAP = {0: 'B', 1: 'I', 2: 'U', 3: 'D', 4: 'W', 5: 'w', 6: 'X'}
MAP        = {'B': 0, 'I': 1, 'U': 2, 'D': 3, 'W': 4, 'w': 5, 'X': 6}
SYMBOLS    = ['B', 'I', 'U', 'D', 'W', 'w', 'X']
VOCAB      = len(SYMBOLS)

class CandleType(Enum):
    B = 0
    I = 1
    U = 2
    D = 3
    W = 4
    w = 5
    X = 6

def encode_candle(o: float, h: float, l: float, c: float) -> int:
    """
    Encode a single OHLCV bar to symbol index.

    Args:
        o: open price
        h: high price
        l: low price
        c: close price

    Returns:
        int index (0-6), use SYMBOL_MAP[idx] for the letter
    """
    rng  = h - l
    if rng < 1e-10:
        return MAP['X']  # zero-range bar = doji

    body       = abs(c - o)
    body_ratio = body / rng

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    upper_ratio = upper_wick / rng
    lower_ratio = lower_wick / rng

    # Wick-dominant bars (institutional rejection wicks)
    if upper_ratio > 0.60:
        return MAP['W']   # upper wick trap — bearish rejection
    if lower_ratio > 0.60:
        return MAP['w']   # lower wick trap — bullish rejection

    # Doji — body less than 10% of range
    if body_ratio < 0.10:
        return MAP['X']

    # Directional bars
    if c > o:
        return MAP['B'] if body_ratio > 0.60 else MAP['U']
    else:
        return MAP['I'] if body_ratio > 0.60 else MAP['D']


def encode_symbol(o: float, h: float, l: float, c: float) -> str:
    """Convenience wrapper — returns the letter directly."""
    return SYMBOL_MAP[encode_candle(o, h, l, c)]


def encode_sequence(bars: list) -> str:
    """
    Encode a list of OHLCV dicts or tuples to a symbol string.

    Accepts:
        [{'open':..,'high':..,'low':..,'close':..}, ...]
        or [(o,h,l,c), ...]
    """
    result = []
    for bar in bars:
        if isinstance(bar, dict):
            result.append(encode_symbol(
                bar['open'], bar['high'], bar['low'], bar['close']
            ))
        else:
            result.append(encode_symbol(*bar[:4]))
    return ''.join(result)


def sequence_to_indices(seq: str) -> list:
    """Convert symbol string to list of int indices for model input."""
    return [MAP[s] for s in seq if s in MAP]

class SymbolicEncoder:
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.avg_body_history = []

    def encode_candle(self, o, h, l, c) -> CandleType:
        body = abs(c - o)
        self.avg_body_history.append(body)
        if len(self.avg_body_history) > 100:
            self.avg_body_history.pop(0)

        idx = encode_candle(o, h, l, c)
        return CandleType(idx)

    def compute_rho_c_fixed(self, symbols):
        if not symbols: return 0.0
        doji_count = symbols.count(CandleType.X.value)
        return doji_count / len(symbols)

    def compute_Q_session_fixed(self, symbols):
        if not symbols: return 0.0
        n = len(symbols)
        doji_count = symbols.count(CandleType.X.value)
        clusters = self._clusters(symbols, CandleType.X.value)

        Q = (
            (doji_count / n) *
            (sum(k * clusters.get(k, 0) for k in range(3, 10)) /
             (sum((k**2) * clusters.get(k, 0) for k in range(5, 10)) + 1e-6))
        )
        return float(Q)

    def compute_eta_trend(self, k_bar, min_v, max_v):
        # Simplified trend efficiency based on avg run length
        if k_bar <= 1.0: return 0.0
        # Normalize between min_v and max_v
        return min(1.0, (k_bar - min_v) / (max_v - min_v))

    def _clusters(self, seq, symbol_val):
        clusters = {}
        count = 0
        for s in seq:
            if s == symbol_val:
                count += 1
            else:
                if count > 0:
                    clusters[count] = clusters.get(count, 0) + 1
                    count = 0
        if count > 0:
            clusters[count] = clusters.get(count, 0) + 1
        return clusters

if __name__ == '__main__':
    # Sanity check against annotated CSV labels
    tests = [
        # (o, h, l, c, expected_symbol)
        (1.0850, 1.0870, 1.0848, 1.0868, 'B'),   # strong bullish
        (1.0868, 1.0870, 1.0845, 1.0848, 'I'),   # strong bearish
        (1.0850, 1.0855, 1.0848, 1.0853, 'U'),   # weak bullish
        (1.0853, 1.0855, 1.0848, 1.0850, 'D'),   # weak bearish
        (1.0850, 1.0870, 1.0848, 1.0851, 'W'),   # upper wick trap
        (1.0850, 1.0852, 1.0830, 1.0851, 'w'),   # lower wick trap
        (1.0850, 1.0858, 1.0842, 1.08505, 'X'), # doji (3% body)
    ]
    print("Encoder sanity check:")
    all_pass = True
    for o, h, l, c, expected in tests:
        got = encode_symbol(o, h, l, c)
        status = '✔' if got == expected else '✖'
        if got != expected: all_pass = False
        print(f"  {status} ({o},{h},{l},{c}) → {got}  (expected {expected})")
    print(f"\n  {'All passed' if all_pass else 'FAILURES DETECTED'}")
