"""
memory.py — SMART-EXE Semantic Pattern Memory
==============================================
Upgrade from raw ASCII embeddings to CLM learned embeddings.

OLD: embed(seq) = [ord(c) % 7 for c in seq]  ← meaningless geometry
NEW: embed(seq) = clm.get_embedding(seq)       ← learned semantic space

The CLM embedding space has real geometric meaning:
  - Sequences that produce similar next-symbol distributions
    are close in embedding space
  - FAISS search finds patterns that are semantically similar,
    not just character-similar

Two modes:
  1. WITH CLM (recommended): semantic embeddings via Transformer
  2. WITHOUT CLM (fallback): improved positional encoding,
     still far better than ord(c) % 7

Interface is backward compatible with old Memory class.
"""

import numpy as np
from collections import Counter
import math

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[Memory] faiss not found — using numpy fallback. pip install faiss-cpu")

from encoder import MAP, SYMBOLS, VOCAB


# ── EMBEDDING ─────────────────────────────────────────────────────────────────

def _positional_embed(seq: str, dim: int = 64) -> np.ndarray:
    """
    Improved fallback embedding when CLM is not available.

    Encodes:
    - Symbol identity (one-hot × position weight)
    - Position information (sinusoidal)
    - Bigram transitions (captures local patterns)
    - Global distribution (symbol frequency)

    Far better than ord(c) % 7 but not as good as learned CLM embeddings.
    """
    vec = np.zeros(dim, dtype=np.float32)
    n   = len(seq)
    if n == 0:
        return vec

    # Segment 1: symbol frequency distribution (7 dims)
    for s in seq:
        if s in MAP:
            vec[MAP[s]] += 1.0
    if n > 0:
        vec[:7] /= n

    # Segment 2: positional symbol encoding (position-weighted)
    for i, s in enumerate(seq[-32:]):
        if s in MAP:
            pos_weight = (i + 1) / len(seq[-32:])   # recent = higher weight
            base = 7 + MAP[s] * 4
            if base + 3 < dim:
                vec[base]     += pos_weight
                vec[base + 1] += pos_weight * np.sin(i * 0.3)
                vec[base + 2] += pos_weight * np.cos(i * 0.3)

    # Segment 3: bigram transitions
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i+1]
        if a in MAP and b in MAP:
            bigram_idx = 35 + MAP[a] * VOCAB + MAP[b]
            if bigram_idx < dim:
                vec[bigram_idx] += 1.0

    # Normalise
    norm = np.linalg.norm(vec) + 1e-9
    return vec / norm


class Memory:
    """
    FAISS-backed episodic pattern memory with CLM embeddings.

    Stores (embedding, outcome) pairs and retrieves historically
    similar patterns to estimate expected outcome of a given sequence.
    """

    def __init__(self, dim: int = 64, clm=None, decay: float = 0.95):
        """
        Args:
            dim:   embedding dimension — must match CLM dim if provided
            clm:   CandleLM instance for semantic embeddings (optional)
            decay: time-decay factor — older memories have less weight
        """
        self.dim      = dim
        self.clm      = clm
        self.decay    = decay
        self.outcomes : list  = []
        self.timestamps: list = []
        self.sequences : list = []
        self._t        : int  = 0

        if FAISS_AVAILABLE:
            # L2 index — works well for normalised CLM embeddings
            self.index = faiss.IndexFlatL2(dim)
        else:
            # Numpy fallback
            self._vectors: list = []
            self.index = None

    def _embed(self, seq: str) -> np.ndarray:
        """Get embedding vector for a sequence."""
        if self.clm is not None:
            try:
                import torch
                with torch.no_grad():
                    emb = self.clm.get_embedding(seq)
                vec = emb.numpy().astype(np.float32)
                # Normalise for cosine-like behaviour in L2 space
                norm = np.linalg.norm(vec) + 1e-9
                return vec / norm
            except Exception:
                pass  # fallback to positional

        # Adjust dim for positional embedding
        return _positional_embed(seq, self.dim)

    def add(self, seq: str, outcome: float):
        """
        Store a pattern → outcome pair.

        Args:
            seq:     symbol sequence that preceded the trade
            outcome: trade result — use pips/10000 or normalised:
                     +1.0 = strong win, -1.0 = strong loss, 0 = break-even
        """
        vec = self._embed(seq).reshape(1, -1)
        self._t += 1

        if FAISS_AVAILABLE:
            self.index.add(vec)
        else:
            self._vectors.append(vec[0])

        self.outcomes.append(float(outcome))
        self.timestamps.append(self._t)
        self.sequences.append(seq[-20:])   # keep last 20 chars for inspection

    def query(self, seq: str, k: int = 7) -> tuple:
        """
        Find k most similar historical patterns and estimate outcome.

        Returns:
            (bias, confidence, n_found)
        """
        n_stored = len(self.outcomes)
        if n_stored < max(k, 3):
            return 0.0, 0.0, 0

        vec = self._embed(seq).reshape(1, -1)
        k   = min(k, n_stored)

        if FAISS_AVAILABLE:
            distances, indices = self.index.search(vec, k)
            indices = indices[0]
            distances = distances[0]
        else:
            # Numpy L2 search
            matrix    = np.stack(self._vectors)
            dists     = np.sum((matrix - vec) ** 2, axis=1)
            indices   = np.argsort(dists)[:k]
            distances = dists[indices]

        # Time-decay weighting: recent patterns count more
        results = []
        for i, idx in enumerate(indices):
            if idx < 0 or idx >= n_stored:
                continue
            age    = self._t - self.timestamps[idx]
            weight = (self.decay ** age) / (1.0 + distances[i])
            results.append((self.outcomes[idx], weight))

        if not results:
            return 0.0, 0.0, 0

        outcomes, weights = zip(*results)
        total_w  = sum(weights)
        bias     = sum(o * w for o, w in zip(outcomes, weights)) / total_w

        # Confidence = 1 - normalised std (low std = all similar patterns agree)
        if len(outcomes) > 1:
            std        = np.std(outcomes)
            confidence = max(0.0, 1.0 - std)
        else:
            confidence = abs(bias)

        return float(bias), float(confidence), len(results)

    def query_full(self, seq: str, k: int = 7) -> dict:
        """
        Extended query returning full breakdown for logging / display.
        """
        bias, conf, n = self.query(seq, k)
        return {
            'bias':        bias,
            'confidence':  conf,
            'n_similar':   n,
            'memory_ok':   abs(bias) >= 0.25 and conf >= 0.40,
            'direction':   'LONG' if bias > 0.1 else 'SHORT' if bias < -0.1 else 'NEUTRAL',
        }

    def persist(self, path: str):
        """Save memory to disk."""
        np.savez_compressed(path,
            outcomes=np.array(self.outcomes, dtype=np.float32),
            timestamps=np.array(self.timestamps, dtype=np.int32),
        )
        if FAISS_AVAILABLE and self.index.ntotal > 0:
            faiss.write_index(self.index, path + '.faiss')
        print(f"  Memory saved → {path} ({len(self.outcomes)} patterns)")

    def load(self, path: str):
        """Load memory from disk."""
        try:
            data = np.load(path + '.npz')
            self.outcomes    = list(data['outcomes'])
            self.timestamps  = list(data['timestamps'])
            self._t          = int(max(self.timestamps)) if self.timestamps else 0
            if FAISS_AVAILABLE:
                import os
                if os.path.exists(path + '.faiss'):
                    self.index = faiss.read_index(path + '.faiss')
            print(f"  Memory loaded ← {path} ({len(self.outcomes)} patterns)")
        except Exception as e:
            print(f"  Memory load failed: {e} — starting fresh")

    @property
    def size(self) -> int:
        return len(self.outcomes)

class FAISSMemory:
    """Hyperion-compatible episodic pattern memory with JAX/FAISS acceleration."""

    def __init__(self, dim=16, ngram_size=5):
        """
        Args:
            dim:   embedding dimension
            ngram_size: n-gram window for context analysis
        """
        self.dim = dim
        self.ngram_size = ngram_size
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dim)
        else:
            self.index = None
            print("[FAISSMemory] FAISS not found - running in degraded mode")
        self.history = []
        self.outcomes = []

    def compute_context_mi(self, symbols):
        """Estimate Information Theory metrics (Mutual Information / Pattern Entropy)."""
        if not symbols: return 0.0
        n = len(symbols)
        counts = Counter(symbols)
        probs = [c/n for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        return entropy

    def get_bias_and_confidence(self, symbols):
        """Find similar historical patterns and estimate directional bias."""
        if not self.history or self.index is None or not FAISS_AVAILABLE:
            return 0.0, 0.0

        query = self._embed(symbols).reshape(1, -1)
        k = min(5, len(self.history))
        D, I = self.index.search(query, k)

        neighbor_outcomes = [self.outcomes[idx] for idx in I[0] if idx != -1]
        if not neighbor_outcomes:
            return 0.0, 0.0

        bias = sum(neighbor_outcomes) / len(neighbor_outcomes)
        conf = 1.0 / (1.0 + np.mean(D[0]))
        return float(bias), float(conf)

    def add(self, symbols, last_symbol_val, outcome):
        """Store a sequence → outcome pair in memory."""
        emb = self._embed(symbols).reshape(1, -1)
        if self.index is not None:
            self.index.add(emb)
        self.history.append(symbols)
        self.outcomes.append(outcome)

    def _embed(self, symbols):
        """Create a fixed-length numeric embedding from a sequence of symbols."""
        vec = np.zeros(self.dim, dtype=np.float32)
        # Handle both list of ints and list of objects/Enums
        for i, s in enumerate(symbols[-self.dim:]):
            try:
                val = float(s)
            except (TypeError, ValueError):
                # Fallback for Enum or complex objects
                val = float(getattr(s, 'value', 0))
            vec[i] = val

        norm = np.linalg.norm(vec) + 1e-9
        return vec / norm
