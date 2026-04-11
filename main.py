# main.py - Corrected orchestration
import time
from datetime import datetime

class HyperionSystem:
    def __init__(self):
        self.encoder = SymbolicEncoder(lookback=20)
        self.memory = FAISSMemory(dim=16, ngram_size=5)
        self.risk = RiskEngine(adaptive_thresholds=True)
        self.lambda7 = Lambda7Engine()
        self.trader = OandaTrader()  # Your execution module
        
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
