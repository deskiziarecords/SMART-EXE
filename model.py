"""
model.py — SMART-EXE Candle Language Model (CLM)
=================================================
Drop-in replacement for the BiGRU PatternModel.

Upgrade path:
    OLD: PatternModel  → 3 pattern classes (LONG/SHORT/WAIT)
    NEW: CandleLM      → 7-symbol probability distribution

The CLM output is richer — instead of a hard class prediction it
gives the full probability distribution over next symbols.
That distribution IS the trading signal:

    High P(B) after squeeze  → expansion, LONG
    High P(I) at premium     → distribution, SHORT
    High entropy output      → chaotic regime, BLOCK
    Low entropy output       → structured, high-confidence

Usage (identical interface to old model):
    from model import CandleLM, load_model
    clm = CandleLM()
    load_model(clm, 'clm_eurusd.pt')

    probs = clm.predict_proba('BUUIXIBXDD')
    # → {'B':0.31, 'I':0.08, 'U':0.22, 'D':0.05, 'W':0.01, 'w':0.04, 'X':0.29}

    sym, conf, entropy = clm.predict(sequence)
    # → ('B', 0.31, 0.72)   ← symbol, confidence, H(distribution)

Requirements: pip install torch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import MAP, SYMBOL_MAP, SYMBOLS, VOCAB


# ── ARCHITECTURE ──────────────────────────────────────────────────────────────

class CandleLM(nn.Module):
    """
    Transformer-based Candle Language Model.

    Architecture:
        Token embedding (VOCAB → dim)
        + Learnable positional embedding (max_len → dim)
        → N × TransformerEncoderLayer (causal mask)
        → LayerNorm
        → Linear (dim → VOCAB)

    Compared to the old BiGRU:
        - Attention can jump to any position, not just adjacent
        - Causal mask enforces autoregressive left-to-right conditioning
        - Output is a full probability distribution, not a class index
        - 5× fewer parameters needed for the same expressiveness on short seqs
    """

    def __init__(
        self,
        vocab:    int = VOCAB,    # 7
        dim:      int = 64,       # embedding / transformer dim
        n_heads:  int = 4,        # attention heads (dim must be divisible)
        n_layers: int = 3,        # transformer depth
        max_len:  int = 64,       # maximum sequence length
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.vocab   = vocab
        self.dim     = dim
        self.max_len = max_len

        # Token embedding — each of the 7 symbols gets a learned vector
        self.tok_emb = nn.Embedding(vocab, dim)

        # Learnable positional embedding (better than sinusoidal for short seqs)
        self.pos_emb = nn.Embedding(max_len, dim)

        # Transformer encoder stack with pre-norm (more stable training)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,     # Pre-LN: better gradient flow
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f  = nn.LayerNorm(dim)
        self.head  = nn.Linear(dim, vocab, bias=False)

        # Weight tying: head shares weights with embedding (halves parameters,
        # forces the model to learn a coherent symbol space)
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (batch, seq_len) int64 token indices

        Returns:
            logits: (batch, vocab) — next-token logits from last position
        """
        B, T = idx.shape
        assert T <= self.max_len, f"Sequence too long: {T} > {self.max_len}"

        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, -1)
        x   = self.tok_emb(idx) + self.pos_emb(pos)

        # Causal attention mask — each position only sees itself and prior tokens
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x    = self.transformer(x, mask=mask, is_causal=True)
        x    = self.ln_f(x)

        # Only the last position predicts the next token
        return self.head(x[:, -1, :])   # (B, vocab)

    # ── INFERENCE API ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_proba(self, seq: str) -> dict:
        """
        Full probability distribution over next symbol.

        Args:
            seq: string of 1–max_len symbols, e.g. 'BUUIXIBXDD'

        Returns:
            dict: {'B': 0.31, 'I': 0.08, ..., 'X': 0.29}
        """
        self.eval()
        tokens = [MAP[s] for s in seq if s in MAP]
        if not tokens:
            return {s: 1/VOCAB for s in SYMBOLS}

        idx    = torch.tensor([tokens], dtype=torch.long)
        logits = self(idx)
        probs  = F.softmax(logits, dim=-1)[0]
        return {SYMBOL_MAP[i]: float(probs[i]) for i in range(self.vocab)}

    @torch.no_grad()
    def predict(self, seq: str) -> tuple:
        """
        Returns (next_symbol, confidence, entropy).

        entropy is H(distribution) normalised to [0,1].
        entropy ≈ 0 → model is very sure → low-noise regime → trade
        entropy ≈ 1 → model is confused  → chaotic regime  → block
        """
        dist    = self.predict_proba(seq)
        best    = max(dist, key=dist.get)
        conf    = dist[best]
        h       = -sum(p * math.log2(p + 1e-9) for p in dist.values())
        h_norm  = h / math.log2(self.vocab)   # normalise to [0,1]
        return best, conf, h_norm

    @torch.no_grad()
    def get_embedding(self, seq: str) -> torch.Tensor:
        """
        Extract the final hidden state as a sequence embedding.
        Used by memory.py (FAISS) for semantic similarity search.

        Returns:
            (dim,) float32 tensor
        """
        self.eval()
        tokens = [MAP[s] for s in seq if s in MAP]
        if not tokens:
            return torch.zeros(self.dim)
        idx = torch.tensor([tokens], dtype=torch.long)
        B, T = idx.shape
        pos  = torch.arange(T).unsqueeze(0)
        x    = self.tok_emb(idx) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        x    = self.transformer(x, mask=mask, is_causal=True)
        x    = self.ln_f(x)
        return x[0, -1, :].detach()   # (dim,)

    # ── DIRECTION HELPERS (matches old PatternModel interface) ─────────────
    def direction_signal(self, seq: str, entropy_threshold: float = 0.55) -> dict:
        """
        Translate CLM output into trading direction signal.
        Replaces the old 3-class classifier output.

        Returns:
            {
              'direction': 'LONG' | 'SHORT' | 'WAIT',
              'confidence': float,
              'entropy': float,       ← gate: block if > entropy_threshold
              'distribution': dict,   ← full probs for QUIMERIA sensor
              'clm_ok': bool,         ← True if entropy < threshold
            }
        """
        best, conf, entropy = self.predict(seq)
        dist = self.predict_proba(seq)

        # Aggregate bullish and bearish mass
        bull_mass = dist.get('B', 0) + dist.get('U', 0) + dist.get('w', 0)
        bear_mass = dist.get('I', 0) + dist.get('D', 0) + dist.get('W', 0)

        if entropy > entropy_threshold:
            direction = 'WAIT'
        elif bull_mass > bear_mass and bull_mass > 0.40:
            direction = 'LONG'
        elif bear_mass > bull_mass and bear_mass > 0.40:
            direction = 'SHORT'
        else:
            direction = 'WAIT'

        return {
            'direction':    direction,
            'confidence':   conf,
            'entropy':      entropy,
            'distribution': dist,
            'clm_ok':       entropy < entropy_threshold,
            'bull_mass':    bull_mass,
            'bear_mass':    bear_mass,
            'next_symbol':  best,
        }


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_clm(
    model:      CandleLM,
    sequences:  list,          # list of symbol strings
    epochs:     int   = 100,
    window:     int   = 10,
    batch_size: int   = 64,
    lr:         float = 3e-3,
    val_split:  float = 0.15,
    device:     str   = 'cpu',
    verbose:    bool  = True,
) -> dict:
    """
    Train the CLM on a list of symbol sequences.

    Args:
        sequences: list of strings, e.g. ['BUUIXIBXDDI', 'BBUIDIIX', ...]
                   Can be one long string split into chunks, or multiple sessions.

    Returns:
        dict with training history
    """
    import numpy as np
    from torch.utils.data import Dataset, DataLoader

    # Build sliding-window dataset from all sequences
    class SeqDataset(Dataset):
        def __init__(self, pairs):
            self.X = torch.tensor([p[0] for p in pairs], dtype=torch.long)
            self.y = torch.tensor([p[1] for p in pairs], dtype=torch.long)

        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i]

    pairs = []
    for seq in sequences:
        indices = [MAP[s] for s in seq if s in MAP]
        for i in range(len(indices) - window):
            pairs.append((indices[i:i+window], indices[i+window]))

    if not pairs:
        raise ValueError("No training pairs — check sequence data")

    np.random.shuffle(pairs)
    n_val  = max(1, int(len(pairs) * val_split))
    val_d  = SeqDataset(pairs[:n_val])
    trn_d  = SeqDataset(pairs[n_val:])

    trn_loader = DataLoader(trn_d, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_d, batch_size=256,        shuffle=False)

    # Class weights to handle imbalance (I is ~55% of data)
    sym_counts = [sum(1 for p in pairs if p[1] == i) for i in range(VOCAB)]
    total      = sum(sym_counts)
    weights    = torch.tensor(
        [total / (VOCAB * max(c, 1)) for c in sym_counts],
        dtype=torch.float32, device=device
    )

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine LR schedule with warmup
    total_steps  = epochs * len(trn_loader)
    warmup_steps = total_steps // 10
    scheduler    = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
    )

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val  = float('inf')
    best_state = None
    patience   = 20
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        trn_loss = 0.0
        for Xb, yb in trn_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimiser.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            trn_loss += loss.item()

        model.eval()
        val_loss, correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits  = model(Xb)
                val_loss += criterion(logits, yb).item()
                correct  += (logits.argmax(-1) == yb).sum().item()
                total_val += len(yb)

        avg_trn = trn_loss / len(trn_loader)
        avg_val = val_loss  / len(val_loader)
        acc     = correct   / total_val

        history['train_loss'].append(avg_trn)
        history['val_loss'].append(avg_val)
        history['val_acc'].append(acc)

        if avg_val < best_val:
            best_val   = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"trn={avg_trn:.4f}  val={avg_val:.4f}  "
                  f"acc={acc:.3f}  lr={scheduler.get_last_lr()[0]:.5f}")

        if no_improve >= patience:
            if verbose: print(f"  Early stop at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    return history


# ── SERIALISATION ─────────────────────────────────────────────────────────────

def save_model(model: CandleLM, path: str):
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'vocab':    model.vocab,
            'dim':      model.dim,
            'n_heads':  4,
            'n_layers': len(model.transformer.layers),
            'max_len':  model.max_len,
        }
    }, path)
    print(f"  Saved → {path}")


def load_model(path: str) -> CandleLM:
    ckpt   = torch.load(path, map_location='cpu')
    cfg    = ckpt['config']
    model  = CandleLM(**cfg)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"  Loaded ← {path}")
    return model


# ── QUICK TRAIN FROM CSV ──────────────────────────────────────────────────────

def train_from_csv(
    training_csv:  str = 'training_data.csv',
    candles_csv:   str = 'annotated_candles.csv',
    save_path:     str = 'clm_eurusd.pt',
    epochs:        int = 120,
):
    """
    One-shot train from your existing CSV files and save.
    Run from command line: python model.py
    """
    import csv

    sequences = []

    # Load training_data.csv — each row is a (input_seq, target) pair
    # Reconstruct full sequences by concatenating
    full_seq = ''
    if __import__('os').path.exists(training_csv):
        with open(training_csv) as f:
            for row in csv.DictReader(f):
                inp = row['input'].strip()
                tgt = row['target'].strip()
                if full_seq == '':
                    full_seq = inp
                full_seq += tgt
        if full_seq:
            sequences.append(full_seq)
            print(f"  Loaded {training_csv}: {len(full_seq)} symbols")

    # Load annotated_candles.csv — raw symbol column
    if __import__('os').path.exists(candles_csv):
        with open(candles_csv) as f:
            sym_str = ''.join(
                row['Pattern'].strip()
                for row in csv.DictReader(f)
                if row['Pattern'].strip() in MAP
            )
        if sym_str:
            sequences.append(sym_str)
            print(f"  Loaded {candles_csv}: {len(sym_str)} symbols")

    if not sequences:
        print("  No data found — generating synthetic sequence for demo")
        sequences = ['BUUIXIBXDDIDBIIDIIUDIIBXDDIDBIIBXDDIDBIIX' * 20]

    model = CandleLM(vocab=VOCAB, dim=64, n_heads=4, n_layers=3)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Training for {epochs} epochs…\n")

    train_clm(model, sequences, epochs=epochs, window=10)
    save_model(model, save_path)

    # Quick inference demo
    print("\n── INFERENCE DEMO ──────────────────────────────────────────────")
    test_seqs = ['BUUIXIBXDD', 'IIIIIIIIII', 'BBBBBBBBBB', 'XXXXXXXXXX']
    for seq in test_seqs:
        sym, conf, ent = model.predict(seq)
        sig = model.direction_signal(seq)
        print(f"  [{seq}] → {sym} conf={conf:.3f} H={ent:.3f} | {sig['direction']}")

    return model


if __name__ == '__main__':
    train_from_csv()
