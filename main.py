import time
from collections import deque
from encoder import encode_candle, SYMBOL_MAP
from memory import Memory
from risk_engine import RiskEngine
from trader import Trader
from data_feed import get_price
from lambda7 import Lambda7

prices = deque(maxlen=100)
patterns = deque(maxlen=100)

memory = Memory()
risk = RiskEngine()
trader = Trader()
lambda7 = Lambda7()

def entropy(seq):
    from collections import Counter
    from math import log2

    c = Counter(seq)
    total = len(seq)
    return -sum((v/total)*log2(v/total) for v in c.values())

while True:
    p = get_price()
    prices.append(p)

    if len(prices) < 2:
        continue

    o = prices[-2]
    c = prices[-1]
    h = max(o,c)
    l = min(o,c)

    idx = encode_candle(o,h,l,c)
    sym = SYMBOL_MAP[idx]
    patterns.append(sym)

    seq = ''.join(patterns)

    lambda7.update(list(prices))

    state = {
        "entropy": entropy(seq[-50:]),
        "memory_bias": memory.query(seq),
        "direction": "LONG" if seq[-1] in ['B','U'] else "SHORT",
        "lambda7_ok": lambda7.valid("LONG")
    }

    decision = risk.evaluate(state)

    if decision["action"]=="ALLOW":
        trader.order(state["direction"], decision["size"])

    time.sleep(1)
