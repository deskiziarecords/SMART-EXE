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

# Forward and reverse maps — consistent with QUIMERIA SMK + training data
SYMBOL_MAP = {0: 'B', 1: 'I', 2: 'U', 3: 'D', 4: 'W', 5: 'w', 6: 'X'}
MAP        = {'B': 0, 'I': 1, 'U': 2, 'D': 3, 'W': 4, 'w': 5, 'X': 6}
SYMBOLS    = ['B', 'I', 'U', 'D', 'W', 'w', 'X']
VOCAB      = len(SYMBOLS)


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
