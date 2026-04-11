# market_state_engine.py

import numpy as np
from collections import Counter

SYMBOLS = ['I', 'B', 'X', 'U', 'D', 'W', 'w']

class MarketStateEngine:
    def __init__(self, window=120):
        self.window = window
        self.sequence = []

    # --------------------------------------------------
    # UPDATE
    # --------------------------------------------------
    def update(self, new_symbol: str):
        self.sequence.append(new_symbol)
        if len(self.sequence) > self.window:
            self.sequence.pop(0)

        return self.compute_state()

    # --------------------------------------------------
    # CORE STATE COMPUTATION
    # --------------------------------------------------
    def compute_state(self):
        seq = self.sequence
        n = len(seq)

        if n < 10:
            return {"action": "HOLD", "confidence": 0.0}

        # ===============================
        # 1. DOJI CLUSTERS
        # ===============================
        doji_count = seq.count('I')
        rho_c = doji_count / n

        clusters = self._clusters(seq, 'I')
        cluster_score = sum(k * v for k, v in clusters.items())
        
        Q_session = (
            (doji_count / n) *
            (sum(k * clusters.get(k, 0) for k in range(3, 10)) /
             (sum((k**2) * clusters.get(k, 0) for k in range(5, 10)) + 1e-6))
        )

        # ===============================
        # 2. REVERSAL DENSITY
        # ===============================
        reversal_patterns = {"XIW", "BIw", "IWI", "WIX", "wIB"}
        reversals = 0

        for i in range(2, n):
            tri = ''.join(seq[i-2:i+1])
            if tri in reversal_patterns:
                reversals += 1

        delta_r = reversals / max(1, (n - 2))

        # ===============================
        # 3. STOP HUNT PROBABILITY
        # ===============================
        strong_seq = ''.join([s for s in seq if s in ['B', 'X']])
        stop_hunts = sum(1 for i in range(len(seq)-1)
                         if seq[i] in ['B','X'] and seq[i+1] in ['W','w'])

        P_stop = stop_hunts / (len(strong_seq) + 1e-6)

        # ===============================
        # 4. ENGULFING
        # ===============================
        engulf_score = 0

        for i in range(2, n):
            tri = ''.join(seq[i-2:i+1])

            if tri in ['DXB', 'XUB']:
                engulf_score += 2
            elif tri in ['DUB', 'XBB']:
                engulf_score += 1
            elif tri in ['BXD', 'UXD']:
                engulf_score -= 2
            elif tri in ['BWD', 'BDD']:
                engulf_score -= 1

        eta_adj = engulf_score / (n - doji_count + 1e-6)

        # ===============================
        # 5. TREND EFFICIENCY
        # ===============================
        runs = self._trend_runs(seq)
        if runs:
            k_mean = np.mean(runs)
            fizzle_prob = sum(1 for r in runs if r <= 2) / len(runs)
            eta_trend = (1 / (k_mean + 1e-6)) * (1 - fizzle_prob)
        else:
            eta_trend = 0

        # ===============================
        # 6. ENTROPY
        # ===============================
        counts = Counter(seq)
        probs = np.array([counts[s]/n for s in SYMBOLS if counts[s] > 0])
        entropy = -np.sum(probs * np.log2(probs))

        # ===============================
        # FINAL DECISION
        # ===============================
        action = "HOLD"
        confidence = 0.0

        if rho_c < 0.6:  # NOT CHOP
            if eta_adj > 0.1:
                action = "BUY"
                confidence = eta_adj * (1 - rho_c) * eta_trend
            elif eta_adj < -0.1:
                action = "SELL"
                confidence = abs(eta_adj) * (1 - rho_c) * eta_trend

        return {
            "action": action,
            "confidence": round(float(confidence), 4),
            "state_vector": {
                "rho_c": rho_c,
                "delta_r": delta_r,
                "P_stop": P_stop,
                "eta_adj": eta_adj,
                "eta_trend": eta_trend,
                "Q_session": Q_session,
                "entropy": entropy
            }
        }

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def _clusters(self, seq, symbol):
        clusters = {}
        count = 0

        for s in seq:
            if s == symbol:
                count += 1
            else:
                if count > 0:
                    clusters[count] = clusters.get(count, 0) + 1
                    count = 0

        if count > 0:
            clusters[count] = clusters.get(count, 0) + 1

        return clusters

    def _trend_runs(self, seq):
        runs = []
        count = 1

        for i in range(1, len(seq)):
            if seq[i] in ['B', 'U'] and seq[i-1] in ['B', 'U']:
                count += 1
            elif seq[i] in ['X', 'D'] and seq[i-1] in ['X', 'D']:
                count += 1
            else:
                if count > 1:
                    runs.append(count)
                count = 1

        return runs
