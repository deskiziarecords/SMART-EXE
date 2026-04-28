"""
main.py — SMART-EXE Core Trading Loop
======================================
Upgraded to use CLM + semantic memory.

Flow per bar:
    MT5 / OANDA tick
        → OHLCV bar
        → encoder.encode_symbol()     → symbol (e.g. 'B')
        → sequence buffer             → 'BUUIXIBXDDI...'
        → clm.direction_signal()      → {direction, entropy, bull_mass, ...}
        → memory.query_full()         → {bias, confidence, direction, ...}
        → risk.evaluate(state)        → ALLOW / BLOCK
        → trader.order()              → execution
        → memory.add(seq, outcome)    → learning from result

Specialist model:
    Each SMART-EXE instance runs ONE asset, ONE timeframe.
    The CLM is trained only on that pair's history.
    It speaks only EUR/USD (or whichever asset you configure).

Usage:
    python main.py                           # uses config.py settings
    python main.py --asset EURUSD --tf M1   # override
"""

import time
import argparse
from collections import deque
from datetime import datetime

from encoder import encode_candle, SYMBOL_MAP, encode_symbol
from model   import CandleLM, load_model
from memory  import Memory
from risk_engine import RiskEngine
from trader  import Trader
from data_feed import get_price, get_bar   # returns (o,h,l,c) bar
from lambda7 import Lambda7
from logger  import log_block, log_trade, log_signal

try:
    from config import (
        PAIR, TIMEFRAME, ENTROPY_THRESHOLD,
        MODEL_PATH, MEMORY_PATH,
        MAX_SEQ_LEN, WINDOW,
    )
except ImportError:
    # Defaults if config.py not present
    PAIR              = 'EUR_USD'
    TIMEFRAME         = 'M1'
    ENTROPY_THRESHOLD = 0.60
    MODEL_PATH        = 'clm_eurusd.pt'
    MEMORY_PATH       = 'memory_eurusd'
    MAX_SEQ_LEN       = 200
    WINDOW            = 10


def entropy(seq: str) -> float:
    """Raw Shannon entropy of symbol sequence (kept for logging)."""
    from collections import Counter
    from math import log2
    if not seq: return 0.0
    c = Counter(seq)
    n = len(seq)
    return -sum((v/n) * log2(v/n) for v in c.values())


def run(pair: str = PAIR, model_path: str = MODEL_PATH,
        memory_path: str = MEMORY_PATH, dry_run: bool = False):

    print("═" * 60)
    print(f"  SMART-EXE — {pair} {TIMEFRAME}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("═" * 60)

    # ── Load CLM ──────────────────────────────────────────────────────────────
    import os
    clm = None
    if os.path.exists(model_path):
        clm = load_model(model_path)
        print(f"  CLM loaded: {model_path}")
    else:
        print(f"  WARNING: no model at {model_path} — running without CLM")
        print(f"  Train first: python model.py")

    # ── Init components ───────────────────────────────────────────────────────
    memory  = Memory(dim=64 if clm is None else clm.dim, clm=clm)
    risk    = RiskEngine()
    trader  = Trader()
    lam7    = Lambda7()

    # Load persistent memory if exists
    if os.path.exists(memory_path + '.npz'):
        memory.load(memory_path)

    prices   = deque(maxlen=200)   # raw prices for λ7
    bars     = deque(maxlen=200)   # (o,h,l,c) bars
    seq_buf  = deque(maxlen=MAX_SEQ_LEN)   # symbol sequence

    open_trade: dict | None = None
    bar_count  = 0

    print(f"\n  Starting live loop — {pair} {TIMEFRAME}\n")

    while True:
        try:
            # ── Get bar ───────────────────────────────────────────────────────
            bar = get_bar()   # returns {'open','high','low','close','time'}
            if bar is None:
                time.sleep(1)
                continue

            o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
            prices.append(c)
            bars.append((o, h, l, c))
            bar_count += 1

            # ── Encode ────────────────────────────────────────────────────────
            sym = encode_symbol(o, h, l, c)
            seq_buf.append(sym)
            seq = ''.join(seq_buf)

            ts  = datetime.now().strftime('%H:%M:%S')
            print(f"  [{ts}] {pair} {sym}  seq=…{seq[-12:]}  price={c:.5f}", end='')

            if len(seq) < WINDOW:
                print()
                continue

            # ── CLM signal ────────────────────────────────────────────────────
            if clm is not None:
                clm_sig = clm.direction_signal(seq[-WINDOW:],
                                               entropy_threshold=ENTROPY_THRESHOLD)
            else:
                # Fallback: simple momentum heuristic
                bull = sum(1 for s in seq[-10:] if s in 'BU')
                bear = sum(1 for s in seq[-10:] if s in 'ID')
                clm_sig = {
                    'direction':  'LONG' if bull > bear else 'SHORT' if bear > bull else 'WAIT',
                    'confidence': max(bull, bear) / 10,
                    'entropy':    0.5,
                    'clm_ok':     True,
                    'bull_mass':  bull / 10,
                    'bear_mass':  bear / 10,
                    'next_symbol': sym,
                    'distribution': {},
                }

            # ── Memory query ──────────────────────────────────────────────────
            mem_result = memory.query_full(seq[-WINDOW:])

            # ── λ7 macro ──────────────────────────────────────────────────────
            lam7.update(list(prices))
            lam7_ok = lam7.valid(clm_sig['direction'])

            # ── Build risk state ──────────────────────────────────────────────
            state = {
                # CLM signals
                'entropy':       clm_sig['entropy'],
                'clm_ok':        clm_sig['clm_ok'],
                'clm_direction': clm_sig['direction'],
                'clm_conf':      clm_sig['confidence'],
                'bull_mass':     clm_sig['bull_mass'],
                'bear_mass':     clm_sig['bear_mass'],
                'next_symbol':   clm_sig['next_symbol'],
                'distribution':  clm_sig.get('distribution', {}),

                # Memory signals
                'memory_bias':   mem_result['bias'],
                'memory_conf':   mem_result['confidence'],
                'memory_n':      mem_result['n_similar'],
                'memory_dir':    mem_result['direction'],

                # Macro
                'lambda7_ok':    lam7_ok,
                'direction':     clm_sig['direction'],

                # Context
                'pair':          pair,
                'price':         c,
                'sequence':      seq[-20:],
                'bar_count':     bar_count,
            }

            print(f"  H={clm_sig['entropy']:.2f} "
                  f"CLM={clm_sig['direction']} "
                  f"mem={mem_result['bias']:+.2f} "
                  f"λ7={'ok' if lam7_ok else 'NO'}", end='')

            # ── Evaluate open trade ───────────────────────────────────────────
            if open_trade:
                pips = (c - open_trade['entry']) * 10000
                if open_trade['direction'] == 'SHORT':
                    pips = -pips
                print(f"  | open {open_trade['direction']} {pips:+.1f}p", end='')

                # Close conditions: opposite CLM signal or risk reversal
                should_close = (
                    (open_trade['direction'] == 'LONG'  and clm_sig['direction'] == 'SHORT') or
                    (open_trade['direction'] == 'SHORT' and clm_sig['direction'] == 'LONG')  or
                    pips > 15 or pips < -8   # TP/SL in pips
                )
                if should_close and not dry_run:
                    outcome = pips / 20   # normalise outcome to -1..+1
                    trader.close(open_trade['id'])
                    memory.add(open_trade['sequence'], outcome)
                    log_trade(open_trade, c, pips)
                    print(f"  → CLOSED {pips:+.1f}p", end='')
                    open_trade = None

            print()

            # ── Risk gate ─────────────────────────────────────────────────────
            if open_trade:
                continue   # one trade at a time

            decision = risk.evaluate(state)

            if decision['action'] == 'ALLOW':
                direction = decision['direction']
                size      = decision['size']

                log_signal(state, decision)
                print(f"  ▶ TRADE: {direction} {size} lots @ {c:.5f}")
                print(f"    reason: {decision['reason']}")

                if not dry_run:
                    trade_id = trader.order(direction, size)
                    open_trade = {
                        'id':        trade_id,
                        'direction': direction,
                        'entry':     c,
                        'size':      size,
                        'sequence':  seq[-WINDOW:],
                        'bar':       bar_count,
                    }
            else:
                print(f"  ✗ BLOCKED: {decision['reason']}")

            # ── Persist memory periodically ───────────────────────────────────
            if bar_count % 100 == 0:
                memory.persist(memory_path)
                print(f"  [Memory saved: {memory.size} patterns]")

            # ── Sleep until next bar ──────────────────────────────────────────
            time.sleep(60 if TIMEFRAME == 'M1' else 5)

        except KeyboardInterrupt:
            print("\n\n  Stopping SMART-EXE…")
            memory.persist(memory_path)
            print(f"  Session stats: {risk.stats()}")
            print(f"  Memory: {memory.size} patterns")
            break
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMART-EXE live trading')
    parser.add_argument('--asset',    default=PAIR,       help='Trading pair')
    parser.add_argument('--model',    default=MODEL_PATH, help='CLM model path')
    parser.add_argument('--memory',   default=MEMORY_PATH,help='Memory path')
    parser.add_argument('--dry-run',  action='store_true', help='No real orders')
    args = parser.parse_args()

    run(
        pair=args.asset,
        model_path=args.model,
        memory_path=args.memory,
        dry_run=args.dry_run,
    )
