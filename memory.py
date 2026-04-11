# memory.py - Corrected mutual information
import faiss
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple

class FAISSMemory:
    def __init__(self, dim=16, ngram_size=5):  # FIXED: n-gram context, not unigram
        self.dim = dim
        self.ngram_size = ngram_size
        self.index = faiss.IndexFlatIP(dim)  # Inner product for similarity
        self.metadata = []  # Store (pattern, outcome, epsilon) tuples
        self.embedding_net = self._build_embedding()
        
    def _build_embedding(self):
        """Neural embedding replacing ASCII encoding"""
        class GRUWrapper(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
            def forward(self, x):
                out, _ = self.gru(x)
                return out[:, -1, :] # Take last hidden state

        return nn.Sequential(
            nn.Embedding(7, 32),
            GRUWrapper(32, 64),
            nn.Linear(64, self.dim)
        )
    
    def encode_sequence(self, symbols: List[int]) -> np.ndarray:
        """Encode n-gram context for FAISS"""
        with torch.no_grad():
            x = torch.tensor([symbols[-self.ngram_size:]])
            emb = self.embedding_net(x).numpy()
        return emb.flatten()
    
    def compute_context_mi(self, recent_symbols: List[int]) -> float:
        """
        FIXED: I(Sn; Sn-k:n-1) not I(Sn; Sn-1)
        Mutual information between next symbol and preceding sequence
        """
        if len(recent_symbols) < self.ngram_size + 1:
            return 0.0
        
        context = recent_symbols[-self.ngram_size-1:-1]  # Sn-k:n-1
        target = recent_symbols[-1]  # Sn
        
        # Search similar contexts in memory
        if self.index.ntotal == 0:
            return 0.0

        query = self.encode_sequence(context)
        k = min(50, self.index.ntotal)
        D, I = self.index.search(query.reshape(1, -1).astype('float32'), k=k)
        
        if len(I[0]) == 0:
            return 0.0
        
        # Compute empirical conditional distribution
        neighbors = [self.metadata[i] for i in I[0] if i >= 0 and i < len(self.metadata)]
        outcomes = [m[1] for m in neighbors]  # Next symbols after similar contexts
        
        # Estimate H(Sn | context) via neighbor entropy
        from collections import Counter
        counts = Counter(outcomes)
        total = sum(counts.values())
        probs = [c/total for c in counts.values()]
        conditional_entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        # H(Sn) marginal (from base rates in memory)
        all_outcomes = [m[1] for m in self.metadata]
        base_counts = Counter(all_outcomes)
        base_total = sum(base_counts.values())
        base_probs = [c/base_total for c in base_counts.values()]
        marginal_entropy = -sum(p * np.log2(p) for p in base_probs if p > 0)
        
        # MI = H(Sn) - H(Sn | context)
        mi = marginal_entropy - conditional_entropy
        return max(0, mi) / marginal_entropy if marginal_entropy > 0 else 0  # Normalized
    
    def get_bias_and_confidence(self, symbols: List[int]) -> Tuple[float, float]:
        """
        Returns: (expected_epsilon, confidence_sigma)
        FIXED: Explicit confidence definition
        """
        if self.index.ntotal == 0:
            return 0.0, 0.0

        query = self.encode_sequence(symbols)
        k = min(20, self.index.ntotal)
        D, I = self.index.search(query.reshape(1, -1).astype('float32'), k=k)
        
        if len(I[0]) == 0 or I[0][0] == -1:
            return 0.0, 0.0  # No memory
        
        neighbors = [self.metadata[i] for i in I[0] if i < len(self.metadata)]
        epsilons = [m[2] for m in neighbors]
        
        # Expected epsilon: weighted average by similarity
        weights = np.exp(D[0][:len(epsilons)])  # Similarity scores
        weights = weights / weights.sum()
        expected_eps = np.average(epsilons, weights=weights)
        
        # FIXED: Confidence = 1 - normalized variance (high agreement = high conf)
        variance = np.average((np.array(epsilons) - expected_eps)**2, weights=weights)
        confidence = 1.0 - min(variance / 4.0, 1.0)  # Normalize by max possible var (ε∈[-2,2])
        
        return expected_eps, confidence

    def add(self, symbols: List[int], next_symbol: int, outcome_eps: float):
        """Add new experience to memory"""
        if len(symbols) < self.ngram_size:
            return

        vector = self.encode_sequence(symbols)
        # FAISS expects float32
        self.index.add(vector.reshape(1, -1).astype('float32'))
        self.metadata.append((symbols[-self.ngram_size:], next_symbol, outcome_eps))
