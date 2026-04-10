import faiss
import numpy as np

class Memory:
    def __init__(self, dim=32):
        self.index = faiss.IndexFlatL2(dim)
        self.outcomes = []

    def embed(self, seq):
        vec = np.zeros(32, dtype="float32")
        for i, c in enumerate(seq[-32:]):
            vec[i] = ord(c) % 7
        return vec

    def add(self, seq, outcome):
        v = self.embed(seq)
        self.index.add(v.reshape(1,-1))
        self.outcomes.append(outcome)

    def query(self, seq, k=5):
        if len(self.outcomes) < k:
            return 0.0

        v = self.embed(seq)
        D,I = self.index.search(v.reshape(1,-1), k)

        res = [self.outcomes[i] for i in I[0]]
        return sum(res)/len(res)
