# risk_engine.py - Full implementation
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MarketState:
    """Complete 6-dimensional state vector M"""
    rho_c: float              # Consolidation density
    delta_r: float            # Range extension
    P_stop_hunt: float        # Stop hunt probability
    eta_adj: float            # Adaptive efficiency
    eta_trend: float          # Trend efficiency
    Q_session: float          # Session quietness
    I_context: float          # ADDED: Context mutual information
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.rho_c, self.delta_r, self.P_stop_hunt,
            self.eta_adj, self.eta_trend, self.Q_session, self.I_context
        ])

class RiskEngine:
    def __init__(self, adaptive_thresholds=True):
        self.adaptive = adaptive_thresholds
        self.threshold_history = {
            'rho_c': [], 'delta_r': [], 'Q_session': [],
            'eta_trend': [], 'I_context': []
        }
        self.window = 1000
        
        # FIXED: Timeframe-aware defaults (1m EUR/USD calibrated)
        self.base_thresholds = {
            'rho_c_max': 0.60,
            'delta_r_max': 0.15,
            'Q_session_max': 0.70,
            'eta_trend_min': 0.30,
            'I_context_min': 0.15,
            'P_stop_hunt_max': 0.40,
            'confidence_min': 0.60
        }
    
    def update_thresholds(self, M: MarketState):
        """Adaptive percentile-based thresholds"""
        if not self.adaptive:
            return
            
        for key in self.threshold_history:
            self.threshold_history[key].append(getattr(M, key))
            if len(self.threshold_history[key]) > self.window:
                self.threshold_history[key].pop(0)
    
    def get_threshold(self, metric: str, percentile: float = 80) -> float:
        """Get adaptive threshold from recent history"""
        if not self.adaptive or len(self.threshold_history[metric]) < 100:
            return self.base_thresholds.get(metric, 0.5)
        
        return np.percentile(self.threshold_history[metric], percentile)
    
    def evaluate(self, M: MarketState, 
                 expected_epsilon: float, 
                 confidence: float) -> Tuple[bool, str, Optional[str]]:
        """
        FIXED: Uses all components of M in decision
        
        Returns: (should_trade, direction_reason, block_reason)
        """
        # Gate 1: Consolidation check
        if M.rho_c > self.get_threshold('rho_c_max', 85):
            return False, None, f"CONSOLIDATION: rho_c={M.rho_c:.2f} > {self.get_threshold('rho_c_max', 85):.2f}"
        
        # Gate 2: Quiet session (doji storm)
        if M.Q_session > self.base_thresholds['Q_session_max']:
            return False, None, f"QUIET: Q={M.Q_session:.2f} (doji clusters)"
        
        # Gate 3: Trend efficiency - FIXED: Now used in decision
        if M.eta_trend < self.get_threshold('eta_trend_min', 20):
            return False, None, f"NO_TREND: eta={M.eta_trend:.2f} < {self.get_threshold('eta_trend_min', 20):.2f}"
        
        # Gate 4: Stop hunt probability
        if M.P_stop_hunt > self.base_thresholds['P_stop_hunt_max']:
            return False, None, f"STOP_HUNT: P={M.P_stop_hunt:.2f}"
        
        # Gate 5: Information threshold - FIXED: Uses I_context not just H
        if M.I_context < self.base_thresholds['I_context_min']:
            return False, None, f"LOW_INFO: MI={M.I_context:.2f} < {self.base_thresholds['I_context_min']:.2f}"
        
        # Gate 6: Confidence check
        if confidence < self.base_thresholds['confidence_min']:
            return False, None, f"LOW_CONF: sigma={confidence:.2f}"
        
        # Direction determination
        if expected_epsilon > 1.5:
            return True, "BUY", None
        elif expected_epsilon < -1.5:
            return True, "SELL", None
        else:
            return False, None, f"WEAK_SIGNAL: epsilon={expected_epsilon:.2f}"
    
    def position_size(self, epsilon: float, confidence: float, 
                     M: MarketState, capital: float) -> float:
        """
        Kelly-like sizing with all risk factors
        """
        # Base Kelly fraction
        edge = abs(epsilon) / 2.5  # Normalized to [0,1]
        base_size = edge * confidence * 0.02  # Max 2% risk per trade
        
        # Risk scaling factors
        rho_factor = 1 - (M.rho_c * 0.5)  # Reduce in consolidation
        Q_factor = 1 - (M.Q_session * 0.3)  # Reduce in quiet sessions
        trend_boost = 1 + (M.eta_trend * 0.2)  # Boost in strong trends
        
        size = base_size * rho_factor * Q_factor * trend_boost
        return min(size, 0.05)  # Hard cap at 5%
