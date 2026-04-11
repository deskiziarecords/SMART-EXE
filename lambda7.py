# lambda7.py - DXY → EUR/USD causality
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class MacroState:
    dxy_change: float      # DXY 1h change %
    dxy_trend: float       # DXY 4h efficiency
    spx_change: float      # S&P 500 futures (risk sentiment)
    yields_change: float   # 2Y Treasury (rates differential proxy)

class Lambda7Engine:
    """
    Macro causal validation for EUR/USD
    DXY strength → EUR/USD weakness (negative correlation ~-0.85)
    """
    def __init__(self, correlation_window=50):
        self.history = []
        self.corr_window = correlation_window
        self.current_regime = "normal"  # normal | risk_on | risk_off
        
    def update(self, macro: MacroState):
        self.history.append(macro)
        if len(self.history) > self.corr_window:
            self.history.pop(0)
        
        # Detect regime
        if len(self.history) >= 10:
            recent = self.history[-10:]
            avg_dxy = np.mean([m.dxy_change for m in recent])
            avg_spx = np.mean([m.spx_change for m in recent])
            
            if avg_dxy > 0.3 and avg_spx < -0.5:
                self.current_regime = "risk_off"  # USD flight-to-safety
            elif avg_dxy < -0.2 and avg_spx > 0.3:
                self.current_regime = "risk_on"   # USD selling
            else:
                self.current_regime = "normal"
    
    def validate_direction(self, direction: str, macro: MacroState) -> Tuple[bool, float]:
        """
        Returns: (valid, causal_strength)
        """
        # DXY-EURUSD inverse relationship
        if direction == "BUY":  # EUR/USD long = USD weakness
            # Valid if DXY weakening or risk-on
            if macro.dxy_change < -0.1:
                return True, abs(macro.dxy_change) * 2  # Strong confirmation
            elif macro.dxy_change > 0.2:
                return False, 0.0  # Contradiction
            else:
                return True, 0.3  # Weak/neutral confirmation
                
        elif direction == "SELL":  # EUR/USD short = USD strength
            # Valid if DXY strengthening or risk-off
            if macro.dxy_change > 0.1:
                return True, macro.dxy_change * 2
            elif macro.dxy_change < -0.2:
                return False, 0.0
            else:
                return True, 0.3
        
        return False, 0.0
    
    def get_lambda7_signal(self) -> float:
        """Aggregate causal strength (-1 to 1)"""
        if len(self.history) < 5:
            return 0.0
        
        recent = self.history[-5:]
        dxy_momentum = np.mean([m.dxy_change for m in recent])
        
        # Normalize to [-1, 1]
        signal = np.clip(dxy_momentum * 5, -1, 1)
        return signal
