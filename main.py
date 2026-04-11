# main.py - Corrected orchestration
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

from encoder import SymbolicEncoder, CandleType
from memory import FAISSMemory
from risk_engine import RiskEngine, MarketState
from lambda7 import Lambda7Engine, MacroState
from trader import OandaTrader
from logger import log_block
import argparse
import random
import asyncio
from tick_aggregator import MinuteAggregator
from oanda_stream import OandaStreamer
from config import ACCOUNT_ID, OANDA_API_KEY

class HyperionSystem:
    def __init__(self):
        self.encoder = SymbolicEncoder(lookback=20)
        self.memory = FAISSMemory(dim=16, ngram_size=5)
        self.risk = RiskEngine(adaptive_thresholds=True)
        self.lambda7 = Lambda7Engine()
        self.trader = OandaTrader()
        self.session_symbols = []
        
    def tick(self, candle, macro_data):
        """Main processing loop"""
        # 1. Encode
        symbol = self.encoder.encode_candle(
            candle['open'], candle['high'], 
            candle['low'], candle['close']
        )
        self.session_symbols.append(symbol)
        
        if len(self.session_symbols) < 20:
            return  # Warmup
        
        # 2. Compute market state M (all 7 dimensions)
        rho_c = self.encoder.compute_rho_c_fixed(self.session_symbols)
        Q_session = self.encoder.compute_Q_session_fixed(self.session_symbols)
        
        # Trend efficiency (adaptive to timeframe)
        k_bar = self.compute_avg_consecutive_run(self.session_symbols)
        eta_trend = self.encoder.compute_eta_trend(
            k_bar, timeframe_minutes=1, reference_trend_minutes=15
        )
        
        # FIXED: Context MI (n-gram, not unigram)
        I_context = self.memory.compute_context_mi(self.session_symbols)
        
        # Range extension
        delta_r = self.compute_range_extension(candle)
        
        # Stop hunt probability
        P_stop_hunt = self.detect_stop_hunt(self.session_symbols, candle)
        
        M = MarketState(
            rho_c=rho_c, delta_r=delta_r, P_stop_hunt=P_stop_hunt,
            eta_adj=eta_trend, eta_trend=eta_trend, 
            Q_session=Q_session, I_context=I_context
        )
        
        # 3. Query memory for expectation
        expected_eps, confidence = self.memory.get_bias_and_confidence(
            self.session_symbols
        )
        
        # 4. Risk evaluation (uses ALL M components)
        should_trade, direction, block_reason = self.risk.evaluate(
            M, expected_eps, confidence
        )
        
        # 5. Lambda7 causal check
        if should_trade:
            valid, causal_strength = self.lambda7.validate_direction(
                direction, macro_data
            )
            if not valid:
                should_trade = False
                block_reason = f"LAMBDA7_REJECT: DXY contradiction"
        
        # 6. Execute or log
        if should_trade:
            size = self.risk.position_size(
                expected_eps, confidence, M, 
                capital=self.trader.get_equity()
            )
            self.trader.execute(direction, size, candle['close'])
            self.log_trade(direction, size, M, expected_eps)
        else:
            self.log_blocked(block_reason, M, expected_eps)
        
        # Update adaptive thresholds
        self.risk.update_thresholds(M)
        
        # Store to memory for future reference
        self.memory.add(self.session_symbols, symbol, expected_eps)

    def compute_avg_consecutive_run(self, symbols: List[int]) -> float:
        """Computes the average length of consecutive bullish or bearish runs"""
        if not symbols:
            return 0.0
        runs = []
        count = 1

        # Move lambda outside loop for performance
        is_bull = lambda s: s in [CandleType.B, CandleType.I]
        is_bear = lambda s: s in [CandleType.X, CandleType.W]

        for i in range(1, len(symbols)):
            # Bullish: B(0), I(2); Bearish: X(1), W(3)
            curr = symbols[i]
            prev = symbols[i-1]

            if (is_bull(curr) and is_bull(prev)) or (is_bear(curr) and is_bear(prev)):
                count += 1
            else:
                if count > 1:
                    runs.append(count)
                count = 1
        if count > 1:
            runs.append(count)
        return np.mean(runs) if runs else 1.0

    def compute_range_extension(self, candle: Dict) -> float:
        """Computes current candle range relative to average body size"""
        range_ = candle['high'] - candle['low']
        avg_body = np.mean(self.encoder.avg_body_history) if self.encoder.avg_body_history else 0.0001
        return range_ / avg_body if avg_body > 0 else 1.0

    def detect_stop_hunt(self, symbols: List[int], candle: Dict) -> float:
        """Probability of a stop hunt based on recent strong move + wick/doji"""
        if len(symbols) < 2:
            return 0.0
        last = symbols[-2]
        curr = symbols[-1]

        # Strong move (B/X) followed by Doji(4) or Spinning Top(5)
        if last in [CandleType.B, CandleType.X] and curr in [CandleType.D, CandleType.S]:
            return 0.75

        # Strong move followed by weak move in opposite direction
        if last == CandleType.B and curr == CandleType.W: return 0.5
        if last == CandleType.X and curr == CandleType.I: return 0.5

        return 0.1

    def log_trade(self, direction, size, M, expected_eps):
        print(f"[{datetime.now()}] EXECUTED {direction} | Size: {size:.4f} | Epsilon: {expected_eps:.2f}")

    def log_blocked(self, reason, M, expected_eps):
        print(f"[{datetime.now()}] BLOCKED: {reason} | Epsilon: {expected_eps:.2f}")
        # Log to file via logger.py
        log_block({
            "entropy": M.I_context, # Using MI as proxy for entropy in log
            "direction": "N/A",
            "memory_bias": expected_eps
        }, reason)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperion Trading System')
    parser.add_argument('--mock', action='store_true', help='Run with mock data')
    parser.add_argument('--live', action='store_true', help='Run with live OANDA stream')
    args = parser.parse_args()

    system = HyperionSystem()

    if args.mock:
        print("Starting Hyperion with mock data...")
        price = 1.0850
        for i in range(100):
            # Generate mock candle
            open_p = price + (random.random() - 0.5) * 0.001
            high = open_p + random.random() * 0.0005
            low = open_p - random.random() * 0.0005
            close = open_p + (random.random() - 0.5) * 0.0008
            price = close

            candle = {
                'open': open_p, 'high': high, 'low': low, 'close': close,
                'timestamp': time.time()
            }

            # Generate mock macro data
            macro = MacroState(
                dxy_change=(random.random() - 0.5) * 0.2,
                dxy_trend=(random.random() - 0.5),
                spx_change=(random.random() - 0.5) * 0.5,
                yields_change=(random.random() - 0.5) * 0.1
            )

            system.tick(candle, macro)
            time.sleep(0.1) # Fast mock execution

    elif args.live:
        print("Starting Hyperion with live OANDA stream...")

        def on_candle(candle):
            # In a real scenario, you'd fetch real macro data here
            macro = MacroState(dxy_change=0.0, dxy_trend=0.0, spx_change=0.0, yields_change=0.0)
            system.tick(candle, macro)

        aggregator = MinuteAggregator(on_candle)
        streamer = OandaStreamer(ACCOUNT_ID, OANDA_API_KEY, aggregator)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(streamer.stream())

    else:
        parser.print_help()
