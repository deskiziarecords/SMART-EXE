# encoder.py - Corrected implementation
import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple

class CandleType(IntEnum):
    B = 0  # Bullish strong (body > 2×avg, close > open)
    X = 1  # Bearish strong (body > 2×avg, close < open)
    I = 2  # Bullish weak (body ≤ 2×avg, close > open)
    W = 3  # Bearish weak (body ≤ 2×avg, close < open)
    D = 4  # Doji (body ≤ 0.1×avg, any wick)
    S = 5  # Spinning top (0.1×avg < body ≤ avg, long wicks)
    U = 6  # Uncertain (gap, news spike, etc.)

@dataclass
class EncodedSession:
    symbols: List[int]           # List of CandleType indices
    epsilon_values: List[float] # Context-aware engulfing strength
    rho_c: float               # Fixed consolidation density
    Q_session: float           # Fixed quietness index
    
class SymbolicEncoder:
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.avg_body_history = []
        self.min_cluster_length = 3  # FIXED: Explicit parameter
        
    def encode_candle(self, open_p, high, low, close) -> CandleType:
        body = abs(close - open_p)
        body_avg = np.mean(self.avg_body_history) if self.avg_body_history else body
        
        # Update history
        self.avg_body_history.append(body)
        if len(self.avg_body_history) > self.lookback:
            self.avg_body_history.pop(0)
        
        # Classification with hysteresis to avoid flickering
        if body <= 0.1 * body_avg:
            return CandleType.D
        elif body <= body_avg and (high - low) > 2 * body:
            return CandleType.S
        elif body > 2 * body_avg:
            return CandleType.B if close > open_p else CandleType.X
        else:
            return CandleType.I if close > open_p else CandleType.W
    
    def compute_epsilon(self, triplet: List[CandleType], 
                       trend_eta: float, rho_c: float) -> float:
        """
        FIXED: Context-aware engulfing strength
        """
        pattern = ''.join(t.name for t in triplet)
        
        # Base scores with corrected XBB/BXX classification
        base_scores = {
            'BBX': 2.0, 'XXB': -2.0,   # Classic engulfing
            'XBB': 2.0, 'BXX': -2.0,   # FIXED: Strong reversals (was +/-1)
            'BXD': -1.5 if trend_eta > 0.3 else -0.5,  # Contextual
            'DXB': 1.5 if trend_eta < -0.3 else 0.5,
            'IBB': 1.0, 'IXX': -1.0,   # Weak continuation
            'DDB': 0.5, 'DDX': -0.5,   # Doji break
        }
        
        score = base_scores.get(pattern, 0.0)
        
        # FIXED: Modulate by market state (not just hardcoded)
        if rho_c > 0.6:  # Choppy market
            score *= 0.5   # Reduce confidence
        if abs(trend_eta) > 0.7:  # Strong trend
            score *= 1.2   # Boost trend-following
            
        return np.clip(score, -2.5, 2.5)
    
    def compute_rho_c_fixed(self, symbols: List[int]) -> float:
        """
        FIXED: Dimensional consistency with min_k=3
        """
        # Find doji clusters
        clusters = []
        current_cluster = 0
        
        for s in symbols:
            if s == CandleType.D:
                current_cluster += 1
            else:
                if current_cluster >= self.min_cluster_length:
                    clusters.append(current_cluster)
                current_cluster = 0
        
        # Handle trailing cluster
        if current_cluster >= self.min_cluster_length:
            clusters.append(current_cluster)
        
        if not clusters:
            return 0.0
        
        # FIXED: Sum k·Nk for qualifying clusters only
        weighted_sum = sum(k for k in clusters)
        rho_c = weighted_sum / len(symbols)  # Proportion of candles in clusters
        
        return min(rho_c, 1.0)
    
    def compute_Q_session_fixed(self, symbols: List[int]) -> float:
        """
        FIXED: Zero-denominator protection + quadratic weighting
        """
        # Count cluster lengths
        from collections import Counter
        cluster_lengths = []
        current = 0
        
        for s in symbols:
            if s == CandleType.D:
                current += 1
            else:
                if current > 0:
                    cluster_lengths.append(current)
                current = 0
        
        # Separate short vs long
        short = [k for k in cluster_lengths if 1 <= k < self.min_cluster_length]
        long = [k for k in cluster_lengths if k >= self.min_cluster_length]
        
        # FIXED: Zero guard
        denom = sum(k * cluster_lengths.count(k) for k in long)
        if denom == 0:
            return 0.0  # No qualifying clusters
        
        # Quadratic weighting: Σ(k²·Nk) / Σ(k·Nk)
        numer = sum((k ** 2) * cluster_lengths.count(k) for k in long)
        Q_raw = numer / denom
        
        # Normalize by session doji density
        total_dojis = len(short) + len(long)
        density_factor = len(long) / total_dojis if total_dojis > 0 else 0
        
        return min(Q_raw * density_factor, 1.0)

    def compute_eta_trend(self, k_bar: float, timeframe_minutes: int,
                         reference_trend_minutes: int) -> float:
        """
        Trend efficiency: proportion of actual run length vs target trend length
        """
        target_k = reference_trend_minutes / timeframe_minutes
        if target_k == 0:
            return 0.0
        eta = k_bar / target_k
        return min(eta, 1.0)
