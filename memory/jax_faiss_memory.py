# memory/jax_faiss_memory.py
# JAX-accelerated FAISS pattern memory

import jax
import jax.numpy as jnp
from jax import jit, vmap
import faiss
from functools import partial

class JAXFAISSMemory:
    """
    FAISS index with JAX acceleration for batched queries
    """
    
    def __init__(self, dimension: int = 7):  # 7 pattern types
        self.dimension = dimension
        # CPU FAISS index (still fastest for search)
        self.index = faiss.IndexFlatL2(dimension)
        
        # JAX-compiled search functions
        self._jitted_search = jit(self._search_batch)
        
    def add_patterns(self, patterns: jnp.ndarray, outcomes: jnp.ndarray):
        """
        Add patterns to memory with their outcomes
        patterns: [n, 20] tokenized patterns
        outcomes: [n] profit/loss outcomes
        """
        # Convert to embeddings (one-hot or learned)
        embeddings = jax.nn.one_hot(patterns, 7).reshape(patterns.shape[0], -1)
        
        # Add to FAISS (on CPU)
        self.index.add(np.array(embeddings))
        
        # Store outcomes in JAX array (on GPU)
        self.outcomes = jnp.array(outcomes)
        
    @partial(jit, static_argnums=(0,))
    def _search_batch(self, queries: jnp.ndarray, k: int = 5):
        """
        JIT-compiled batch search
        """
        # Convert queries to embeddings
        query_emb = jax.nn.one_hot(queries, 7).reshape(queries.shape[0], -1)
        
        # Search (FAISS on CPU, but called from JAX)
        distances, indices = self._faiss_search(query_emb, k)
        
        # Aggregate outcomes for nearest neighbors
        neighbor_outcomes = self.outcomes[indices]
        bias = jnp.mean(neighbor_outcomes, axis=1)
        
        return bias, distances, indices
    
    def batch_memory_bias(self, sequences: jnp.ndarray) -> jnp.ndarray:
        """
        Get memory bias for all 7 assets in one JIT call
        """
        return self._jitted_search(sequences)


# Run all 7 assets through memory in one operation
memory = JAXFAISSMemory()
asset_sequences = jnp.array([...])  # 7x20

# Single call - JAX handles parallelism
biases = memory.batch_memory_bias(asset_sequences)
print(f"Memory biases for all assets: {biases}")
