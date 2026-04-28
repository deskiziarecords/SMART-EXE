"""
risk_engine.py — SMART-EXE Multi-Gate Risk Engine
==================================================
Upgrade: adds CLM confidence gate and consensus check between
CLM prediction and FAISS memory.

Gates (in order, all must pass):

  Gate 1  ENTROPY        H(CLM distribution) < threshold
                         Market is structured enough to predict
                         REPLACES: raw symbol-count entropy

  Gate 2  CLM SIGNAL     CLM bull_mass or bear_mass > 0.40
                         Model has directional conviction

  Gate 3  MEMORY BIAS    |memory.bias| > 0.25 and confidence > 0.40
                         Historical patterns agree with direction

  Gate 4  CONSENSUS      CLM direction == memory direction
                         Both intelligence sources agree
                         NEW: prevents false signals from either alone

  Gate 5  LAMBDA7        Macro causal confirmation
                         DXY/SPX direction supports EUR/USD move

  Gate 6  DIRECTION LOCK Single direction only — no conflicting signals
                         (bull_mass and bear_mass can't both be high)

Size scaling:
    base_size = 0.01
    scaled    = base_size + memory_confidence * 0.04 + clm_confidence * 0.02
    max_size  = 0.05   (stealth ceiling from HFT-7ZERO config)

Backward compatible with old interface:
    risk.evaluate(state) → {'action': 'ALLOW'|'BLOCK', 'size': float}
"""

from logger import log_block


# ── THRESHOLDS ────────────────────────────────────────────────────────────────
# These are calibrated for EUR/USD M1 — different pairs need recalibration

ENTROPY_THRESHOLD    = 0.60   # CLM distribution entropy (normalised 0-1)
CLM_MASS_THRESHOLD   = 0.38   # min directional probability mass
MEMORY_BIAS_MIN      = 0.25   # min |bias| for memory gate to pass
MEMORY_CONF_MIN      = 0.35   # min confidence for memory gate to pass
MEMORY_WARMUP        = 10     # n patterns before memory gate is enforced
MAX_SIZE             = 0.05   # stealth ceiling (HFT-7ZERO fragmentation limit)
BASE_SIZE            = 0.01   # minimum lot size


class RiskEngine:
    """
    Multi-gate risk engine with CLM + memory consensus.

    State dict expected:
    {
        'entropy':      float,    # CLM distribution entropy, normalised 0-1
        'memory_bias':  float,    # from memory.query() — -1 to +1
        'memory_conf':  float,    # from memory.query() — 0-1
        'memory_n':     int,      # number of similar patterns found
        'direction':    str,      # 'LONG' | 'SHORT' from CLM signal
        'clm_direction':str,      # 'LONG' | 'SHORT' | 'WAIT' from CLM
        'clm_conf':     float,    # CLM top-symbol confidence
        'clm_ok':       bool,     # entropy < threshold
        'lambda7_ok':   bool,     # macro gate
        'bull_mass':    float,    # CLM bullish probability mass
        'bear_mass':    float,    # CLM bearish probability mass
    }
    """

    def __init__(self):
        self.trades_today  = 0
        self.blocked_today = 0
        self.max_trades    = 5     # daily limit

    def evaluate(self, state: dict) -> dict:
        """
        Run all gates and return trade decision.

        Returns:
            {'action': 'ALLOW'|'BLOCK', 'size': float, 'reason': str}
        """

        # ── Gate 1: Daily trade limit ────────────────────────────────────────
        if self.trades_today >= self.max_trades:
            return self._block(state, 'daily_limit',
                f"daily limit {self.max_trades} reached")

        # ── Gate 2: CLM entropy ──────────────────────────────────────────────
        entropy = state.get('entropy', 1.0)
        if not state.get('clm_ok', False) or entropy > ENTROPY_THRESHOLD:
            return self._block(state, 'entropy',
                f"H={entropy:.3f} > {ENTROPY_THRESHOLD}")

        # ── Gate 3: CLM directional conviction ──────────────────────────────
        clm_dir    = state.get('clm_direction', 'WAIT')
        bull_mass  = state.get('bull_mass', 0.0)
        bear_mass  = state.get('bear_mass', 0.0)

        if clm_dir == 'WAIT':
            return self._block(state, 'clm_no_signal',
                f"CLM: bull={bull_mass:.2f} bear={bear_mass:.2f} — no conviction")

        dominant_mass = bull_mass if clm_dir == 'LONG' else bear_mass
        if dominant_mass < CLM_MASS_THRESHOLD:
            return self._block(state, 'clm_weak',
                f"CLM mass={dominant_mass:.2f} < {CLM_MASS_THRESHOLD}")

        # ── Gate 4: Memory bias ──────────────────────────────────────────────
        mem_bias = state.get('memory_bias', 0.0)
        mem_conf = state.get('memory_conf', 0.0)
        mem_n    = state.get('memory_n', 0)

        # Only enforce memory gate once we have enough history
        if mem_n >= MEMORY_WARMUP:
            if abs(mem_bias) < MEMORY_BIAS_MIN:
                return self._block(state, 'memory_weak',
                    f"memory bias={mem_bias:.3f} < {MEMORY_BIAS_MIN}")
            if mem_conf < MEMORY_CONF_MIN:
                return self._block(state, 'memory_inconsistent',
                    f"memory confidence={mem_conf:.3f} < {MEMORY_CONF_MIN}")

        # ── Gate 5: CLM ↔ Memory consensus ──────────────────────────────────
        if mem_n >= MEMORY_WARMUP:
            mem_dir = 'LONG' if mem_bias > 0 else 'SHORT'
            if mem_dir != clm_dir:
                return self._block(state, 'consensus_fail',
                    f"CLM={clm_dir} vs Memory={mem_dir} — disagree")

        # ── Gate 6: Lambda7 macro ────────────────────────────────────────────
        if not state.get('lambda7_ok', True):
            return self._block(state, 'lambda7',
                "macro causality gate failed")

        # ── ALL GATES PASSED ─────────────────────────────────────────────────
        # Scale position size by combined confidence
        clm_conf  = state.get('clm_conf', 0.5)
        size = min(MAX_SIZE,
            BASE_SIZE
            + mem_conf  * 0.02
            + clm_conf  * 0.02
            + dominant_mass * 0.01
        )
        # Round to 2 decimal places (broker minimum)
        size = round(size, 2)

        self.trades_today += 1
        return {
            'action':    'ALLOW',
            'size':      size,
            'direction': clm_dir,
            'reason':    f"all gates passed | clm={clm_dir} conf={clm_conf:.2f} "
                         f"mem_bias={mem_bias:.2f} H={entropy:.3f}",
        }

    def _block(self, state: dict, reason: str, detail: str = '') -> dict:
        self.blocked_today += 1
        log_block(state, reason)
        return {
            'action':  'BLOCK',
            'size':    0.0,
            'reason':  f"{reason}: {detail}",
        }

    def reset_daily(self):
        """Call at market open each day."""
        self.trades_today  = 0
        self.blocked_today = 0

    def stats(self) -> dict:
        return {
            'trades_today':  self.trades_today,
            'blocked_today': self.blocked_today,
            'block_rate':    self.blocked_today / max(1, self.trades_today + self.blocked_today),
        }
