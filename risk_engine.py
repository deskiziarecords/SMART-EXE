"""
risk_engine.py — SMART-EXE Multi-Gate Risk Engine
==================================================
Upgrade: adds CLM confidence gate and consensus check between
CLM prediction and FAISS memory.

Gates (in order, all must pass):
  Gate 1  ENTROPY        H(CLM distribution) < threshold
  Gate 2  CLM SIGNAL     CLM bull_mass or bear_mass > threshold
  Gate 3  MEMORY BIAS    Historical patterns agree with direction
  Gate 4  CONSENSUS      CLM direction == memory direction
  Gate 5  LAMBDA7        Macro causal confirmation
"""
from dataclasses import dataclass
from logger import log_block

@dataclass
class MarketState:
    """Represents the high-dimensional market state vector for Hyperion."""
    rho_c: float
    delta_r: float
    P_stop_hunt: float
    eta_adj: float
    eta_trend: float
    Q_session: float
    I_context: float

# THRESHOLDS
ENTROPY_THRESHOLD    = 0.60
CLM_MASS_THRESHOLD   = 0.38
MEMORY_BIAS_MIN      = 0.25
MEMORY_CONF_MIN      = 0.35
MEMORY_WARMUP        = 10
MAX_SIZE             = 0.05
BASE_SIZE            = 0.01

class RiskEngine:
    """
    Multi-gate risk engine supporting both legacy SMART-EXE
    and the new Hyperion Brain interfaces.
    """
    def __init__(self, adaptive_thresholds=True):
        self.trades_today  = 0
        self.blocked_today = 0
        self.max_trades    = 5
        self.adaptive      = adaptive_thresholds

    def evaluate(self, state, expected_eps=None, confidence=None) -> dict | tuple:
        """
        Run all gates and return trade decision.

        Supports two signatures:
        1. evaluate(state_dict) -> dict (SMART-EXE legacy)
        2. evaluate(MarketState, expected_eps, confidence) -> (bool, str, str) (Hyperion)
        """
        if isinstance(state, MarketState):
            return self._evaluate_hyperion(state, expected_eps, confidence)
        else:
            return self._evaluate_standard(state)

    def _evaluate_standard(self, state: dict) -> dict:
        # Gate 1: Daily trade limit
        if self.trades_today >= self.max_trades:
            return self._block(state, 'daily_limit', f"daily limit {self.max_trades} reached")

        # Gate 2: CLM entropy
        entropy = state.get('entropy', 1.0)
        if not state.get('clm_ok', False) or entropy > ENTROPY_THRESHOLD:
            return self._block(state, 'entropy', f"H={entropy:.3f} > {ENTROPY_THRESHOLD}")

        # Gate 3: CLM directional conviction
        clm_dir    = state.get('clm_direction', 'WAIT')
        bull_mass  = state.get('bull_mass', 0.0)
        bear_mass  = state.get('bear_mass', 0.0)
        if clm_dir == 'WAIT':
            return self._block(state, 'clm_no_signal', f"CLM: bull={bull_mass:.2f} bear={bear_mass:.2f}")

        dominant_mass = bull_mass if clm_dir == 'LONG' else bear_mass
        if dominant_mass < CLM_MASS_THRESHOLD:
            return self._block(state, 'clm_weak', f"CLM mass={dominant_mass:.2f} < {CLM_MASS_THRESHOLD}")

        # Gate 4: Memory bias
        mem_bias = state.get('memory_bias', 0.0)
        mem_conf = state.get('memory_conf', 0.0)
        mem_n    = state.get('memory_n', 0)
        if mem_n >= MEMORY_WARMUP:
            if abs(mem_bias) < MEMORY_BIAS_MIN:
                return self._block(state, 'memory_weak', f"bias={mem_bias:.3f} < {MEMORY_BIAS_MIN}")
            if mem_conf < MEMORY_CONF_MIN:
                return self._block(state, 'memory_inconsistent', f"conf={mem_conf:.3f} < {MEMORY_CONF_MIN}")

        # Gate 5: CLM ↔ Memory consensus
        if mem_n >= MEMORY_WARMUP:
            mem_dir = 'LONG' if mem_bias > 0 else 'SHORT'
            if mem_dir != clm_dir:
                return self._block(state, 'consensus_fail', f"CLM={clm_dir} vs Memory={mem_dir}")

        # Gate 6: Lambda7 macro
        if not state.get('lambda7_ok', True):
            return self._block(state, 'lambda7', "macro gate failed")

        # ALL GATES PASSED
        self.trades_today += 1
        clm_conf = state.get('clm_conf', 0.5)
        size = round(min(MAX_SIZE, BASE_SIZE + mem_conf * 0.02 + clm_conf * 0.02), 2)
        return {
            'action': 'ALLOW',
            'size': size,
            'direction': clm_dir,
            'reason': f"all gates passed | clm={clm_dir} conf={clm_conf:.2f} bias={mem_bias:.2f}",
        }

    def _evaluate_hyperion(self, M: MarketState, expected_eps: float, confidence: float):
        """Logic for the Hyperion Brain - returns (should_trade, direction, reason)."""
        should_trade = False
        direction = "HOLD"
        reason = ""

        if M.rho_c > 0.6:
            reason = "RHO_C_CHOP"
        elif abs(expected_eps) < 0.1:
            reason = "LOW_EXPECTANCY"
        elif confidence < 0.3:
            reason = "LOW_CONFIDENCE"
        elif M.I_context > 0.8:
            reason = "HIGH_ENTROPY"
        else:
            should_trade = True
            direction = "BUY" if expected_eps > 0 else "SELL"
            reason = "BRAIN_OK"

        return should_trade, direction, reason

    def update_thresholds(self, M: MarketState):
        """Dynamically adjust risk parameters based on market regime."""
        if self.adaptive:
            # Placeholder for regime-aware threshold adaptation
            pass

    def _block(self, state: dict, reason: str, detail: str = '') -> dict:
        self.blocked_today += 1
        log_block(state, reason)
        return {'action': 'BLOCK', 'size': 0.0, 'reason': f"{reason}: {detail}"}

    def stats(self) -> dict:
        return {
            'trades_today': self.trades_today,
            'blocked_today': self.blocked_today,
            'block_rate': self.blocked_today / max(1, self.trades_today + self.blocked_today),
        }
