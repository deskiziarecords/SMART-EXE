#!/usr/bin/env python3
"""
SMART-EXE  —  GRU Pattern Predictor
Trains a per-asset sequence model from sliding-window CSVs.

Input CSV format (your exact format):
    position, input, target
    0, IBXXIXXwXW, B
    1, BXXIXXwXWB, B
    ...

Output:
    model_EUR_USD_1h.pt      — TorchScript model (CPU, portable)
    model_EUR_USD_1h.json    — vocab + hyper-params for inference
    training_report.txt      — per-epoch loss, val accuracy, confusion matrix
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import json
import argparse
import time
from datetime import datetime

# ── Alphabet ────────────────────────────────────────────────────────────────
# Canonical 7-symbol alphabet.  Edit here ONLY if you change the encoder.
# B = strong bullish body
# I = strong bearish body   (NOTE: NOT doji — align with your encoder.py)
# X = neutral / inside bar
# U = small bullish (upper body)
# D = small bearish (lower body)
# W = upper wick dominant (shooting star type)
# w = lower wick dominant (hammer type)
SYMBOLS   = ['B', 'I', 'X', 'U', 'D', 'W', 'w']
SYM2IDX   = {s: i for i, s in enumerate(SYMBOLS)}
IDX2SYM   = {i: s for i, s in enumerate(SYMBOLS)}
VOCAB_SIZE = len(SYMBOLS)          # 7
PAD_IDX    = VOCAB_SIZE            # index 7 used for padding shorter seqs


# ── Dataset ──────────────────────────────────────────────────────────────────
class PatternDataset(Dataset):
    """
    Reads all CSVs from a directory and builds (sequence_tensor, target_idx) pairs.

    Each CSV row:
        input  = string of symbols e.g. "IBBXUWD"
        target = next symbol       e.g. "B"

    Unknown symbols are silently skipped (handles stray chars in raw data).
    """
    def __init__(self, csv_dir: Path, seq_len: int = 20, min_seq: int = 5):
        self.seq_len = seq_len
        self.samples = []          # list of (List[int], int)
        self.skipped = 0

        files = sorted(csv_dir.glob("*training_data.csv"))
        if not files:
            # fallback: any CSV in the folder
            files = sorted(csv_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {csv_dir}")

        print(f"  Loading {len(files)} file(s) from {csv_dir}")

        for f in files:
            df = pd.read_csv(f)
            required = {'input', 'target'}
            if not required.issubset(df.columns):
                print(f"  ⚠  Skipping {f.name} — missing columns {required - set(df.columns)}")
                continue

            for _, row in df.iterrows():
                seq_str = str(row['input']).strip()
                tgt_str = str(row['target']).strip()

                # Filter unknowns
                seq_clean = [c for c in seq_str if c in SYM2IDX]
                if len(seq_clean) < min_seq or tgt_str not in SYM2IDX:
                    self.skipped += 1
                    continue

                # Encode sequence — truncate or pad to seq_len
                indices = [SYM2IDX[c] for c in seq_clean[-seq_len:]]
                if len(indices) < seq_len:
                    indices = [PAD_IDX] * (seq_len - len(indices)) + indices

                self.samples.append((indices, SYM2IDX[tgt_str]))

        if not self.samples:
            raise ValueError("Dataset is empty after filtering — check your CSV format.")

        print(f"  {len(self.samples):,} samples  ({self.skipped} skipped)")
        self._report_balance()

    def _report_balance(self):
        targets = [t for _, t in self.samples]
        c = Counter(targets)
        total = len(targets)
        print("  Target distribution:")
        for idx in sorted(c):
            bar = '█' * int(c[idx] / total * 40)
            print(f"    {IDX2SYM[idx]}  {c[idx]:6,} ({c[idx]/total*100:5.1f}%)  {bar}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, tgt = self.samples[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────
class PatternGRU(nn.Module):
    """
    Embedding → GRU → LayerNorm → Linear → softmax (7 classes).

    Also outputs a scalar confidence = max(softmax) — used by the
    local-LLM escalation gate in the live app.

    Architecture kept intentionally small so it runs fast on CPU
    and bundles cleanly inside a PyInstaller .exe.
    """
    def __init__(
        self,
        vocab_size:  int = VOCAB_SIZE + 1,   # +1 for PAD
        embed_dim:   int = 32,
        hidden_dim:  int = 64,
        num_layers:  int = 2,
        num_classes: int = VOCAB_SIZE,        # 7 symbols
        dropout:     float = 0.25,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.gru = nn.GRU(
            input_size  = embed_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.norm   = nn.LayerNorm(hidden_dim)
        self.drop   = nn.Dropout(dropout)
        self.head   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        """x : (batch, seq_len) long tensor"""
        emb = self.embed(x)                     # (B, T, E)
        out, _ = self.gru(emb)                  # (B, T, H)
        last = self.norm(out[:, -1, :])         # take final step
        logits = self.head(self.drop(last))     # (B, num_classes)
        return logits

    @torch.no_grad()
    def predict(self, seq_indices: list[int]) -> tuple[str, float]:
        """
        Convenience method for live inference.
        Returns (predicted_symbol, confidence_0_to_1).
        """
        self.eval()
        x = torch.tensor([seq_indices], dtype=torch.long)
        logits = self.forward(x)
        probs  = torch.softmax(logits, dim=-1)[0]
        idx    = int(probs.argmax())
        return IDX2SYM[idx], float(probs[idx])


# ── Training loop ─────────────────────────────────────────────────────────────
def train(
    csv_dir:     str  = "./data/collected_data",
    output_dir:  str  = "./models",
    asset:       str  = "EUR_USD",
    timeframe:   str  = "1h",
    seq_len:     int  = 20,
    epochs:      int  = 40,
    batch_size:  int  = 256,
    lr:          float = 3e-4,
    val_split:   float = 0.15,
    hidden_dim:  int  = 64,
    num_layers:  int  = 2,
    dropout:     float = 0.25,
    seed:        int  = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    csv_path    = Path(csv_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_stem  = f"model_{asset}_{timeframe}"
    report_path = output_path / "training_report.txt"

    print("=" * 65)
    print("SMART-EXE  GRU TRAINER")
    print("=" * 65)
    print(f"Asset      : {asset}  [{timeframe}]")
    print(f"CSV dir    : {csv_path}")
    print(f"Output dir : {output_path}")
    print(f"Seq length : {seq_len}")
    print(f"Epochs     : {epochs}  |  batch {batch_size}  |  lr {lr}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────
    print("Loading dataset...")
    full_ds = PatternDataset(csv_path, seq_len=seq_len)

    n_val  = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=512,         shuffle=False, num_workers=0)
    print(f"  Train: {n_train:,}  |  Val: {n_val:,}")
    print()

    # ── Class weights (handles symbol imbalance) ───────────────────────────
    all_targets = [full_ds.samples[i][1] for i in range(len(full_ds))]
    counts = Counter(all_targets)
    total  = len(all_targets)
    weights = torch.tensor(
        [total / (VOCAB_SIZE * counts.get(i, 1)) for i in range(VOCAB_SIZE)],
        dtype=torch.float
    )
    print(f"  Class weights: { {IDX2SYM[i]: round(float(weights[i]),2) for i in range(VOCAB_SIZE)} }")
    print()

    # ── Model, loss, optimizer ─────────────────────────────────────────────
    model     = PatternGRU(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/20)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params : {total_params:,}")
    print()

    # ── Training ───────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_state   = None
    history      = []

    print(f"{'Epoch':>5}  {'Train loss':>10}  {'Val loss':>9}  {'Val acc':>8}  {'LR':>8}  {'Time':>6}")
    print("-" * 65)

    report_lines = [
        f"SMART-EXE Training Report",
        f"Generated: {datetime.now().isoformat()}",
        f"Asset: {asset}  Timeframe: {timeframe}",
        f"Samples: {len(full_ds):,}  Seq len: {seq_len}  Epochs: {epochs}",
        f"Params: {total_params:,}",
        "",
        f"{'Epoch':>5}  {'Train loss':>10}  {'Val loss':>9}  {'Val acc':>8}",
        "-" * 50,
    ]

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # -- train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # -- validate
        model.eval()
        val_loss    = 0.0
        correct     = 0
        conf_matrix = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=int)

        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss   = criterion(logits, yb)
                val_loss += loss.item() * len(xb)
                preds  = logits.argmax(dim=-1)
                correct += (preds == yb).sum().item()
                for t, p in zip(yb.numpy(), preds.numpy()):
                    conf_matrix[t][p] += 1

        val_loss /= n_val
        val_acc   = correct / n_val
        elapsed   = time.time() - t0

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        row = f"{epoch:5d}  {train_loss:10.4f}  {val_loss:9.4f}  {val_acc:8.4f}  {current_lr:8.6f}  {elapsed:5.1f}s"
        print(row)
        report_lines.append(f"{epoch:5d}  {train_loss:10.4f}  {val_loss:9.4f}  {val_acc:8.4f}")
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"        ↑ new best val_acc")

    # ── Save best model ────────────────────────────────────────────────────
    print()
    print(f"Best val accuracy: {best_val_acc:.4f}")
    model.load_state_dict(best_state)
    model.eval()

    # TorchScript — portable, no class definition needed at load time
    scripted   = torch.jit.script(model)
    model_path = output_path / f"{model_stem}.pt"
    scripted.save(str(model_path))
    print(f"Saved model  : {model_path}")

    # Companion JSON — needed by inference code + app
    meta = {
        "asset":       asset,
        "timeframe":   timeframe,
        "symbols":     SYMBOLS,
        "sym2idx":     SYM2IDX,
        "idx2sym":     {str(k): v for k, v in IDX2SYM.items()},
        "vocab_size":  VOCAB_SIZE,
        "pad_idx":     PAD_IDX,
        "seq_len":     seq_len,
        "hidden_dim":  hidden_dim,
        "num_layers":  num_layers,
        "best_val_acc": round(best_val_acc, 4),
        "trained_at":  datetime.now().isoformat(),
        "total_samples": len(full_ds),
        "history":     history,
    }
    meta_path = output_path / f"{model_stem}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta   : {meta_path}")

    # Confusion matrix + per-class metrics in report
    report_lines += [
        "",
        f"Best val accuracy: {best_val_acc:.4f}",
        "",
        "Confusion matrix (rows=true, cols=pred):",
        "     " + "  ".join(f"{s:>4}" for s in SYMBOLS),
    ]
    for i, sym in enumerate(SYMBOLS):
        row_vals = "  ".join(f"{conf_matrix[i][j]:4d}" for j in range(VOCAB_SIZE))
        report_lines.append(f"  {sym}  {row_vals}")

    report_lines += [
        "",
        "Per-class precision / recall:",
        f"  {'Symbol':>6}  {'Precision':>10}  {'Recall':>8}",
    ]
    for i, sym in enumerate(SYMBOLS):
        tp = conf_matrix[i][i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        report_lines.append(f"  {sym:>6}  {prec:10.3f}  {rec:8.3f}")

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved report : {report_path}")
    print()
    print("=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)

    return model, meta


# ── Inference helper (used by live app) ───────────────────────────────────────
class PatternPredictor:
    """
    Thin wrapper — load once at app start, call predict() on each new candle.

    Usage:
        predictor = PatternPredictor("models/model_EUR_USD_1h.pt",
                                     "models/model_EUR_USD_1h.json")
        symbol, confidence = predictor.predict("BBUIBBXIBB")

        if confidence < 0.60:
            # escalate to local LLM
            ...
    """
    ESCALATE_THRESHOLD = 0.60   # below this → local LLM takes over

    def __init__(self, model_path: str, meta_path: str):
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.sym2idx  = self.meta["sym2idx"]
        self.idx2sym  = {int(k): v for k, v in self.meta["idx2sym"].items()}
        self.seq_len  = self.meta["seq_len"]
        self.pad_idx  = self.meta["pad_idx"]
        self.model    = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()

    def _encode(self, seq_str: str) -> list[int]:
        clean = [c for c in seq_str if c in self.sym2idx][-self.seq_len:]
        indices = [self.sym2idx[c] for c in clean]
        if len(indices) < self.seq_len:
            indices = [self.pad_idx] * (self.seq_len - len(indices)) + indices
        return indices

    @torch.no_grad()
    def predict(self, seq_str: str) -> tuple[str, float]:
        """Returns (next_symbol, confidence)."""
        indices = self._encode(seq_str)
        x       = torch.tensor([indices], dtype=torch.long)
        logits  = self.model(x)
        probs   = torch.softmax(logits, dim=-1)[0]
        idx     = int(probs.argmax())
        conf    = float(probs[idx])
        return self.idx2sym[idx], conf

    def should_escalate(self, seq_str: str) -> tuple[bool, str, float]:
        """
        Returns (escalate_flag, predicted_symbol, confidence).
        If escalate_flag is True the live app should call the local LLM.
        """
        sym, conf = self.predict(seq_str)
        return conf < self.ESCALATE_THRESHOLD, sym, conf


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMART-EXE GRU trainer")
    parser.add_argument("--csv-dir",    default="./data/collected_data")
    parser.add_argument("--output-dir", default="./models")
    parser.add_argument("--asset",      default="EUR_USD")
    parser.add_argument("--timeframe",  default="1h")
    parser.add_argument("--seq-len",    type=int,   default=20)
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int,   default=64)
    parser.add_argument("--num-layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.25)
    parser.add_argument("--val-split",  type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)

    # Quick self-test flag — generates synthetic data and runs 2 epochs
    parser.add_argument("--test", action="store_true",
                        help="Run a quick self-test with synthetic data")
    args = parser.parse_args()

    if args.test:
        import tempfile, os
        print("Running self-test with synthetic data...")
        tmp = tempfile.mkdtemp()
        rows = []
        syms = list(SYM2IDX.keys())
        for i in range(2000):
            seq = "".join(np.random.choice(syms, 20))
            tgt = np.random.choice(syms)
            rows.append(f"{i},{seq},{tgt}")
        test_csv = Path(tmp) / "results_EUR_USD_1h_365days_training_data.csv"
        test_csv.write_text("position,input,target\n" + "\n".join(rows))
        train(
            csv_dir=tmp, output_dir=tmp,
            asset="EUR_USD", timeframe="1h",
            epochs=2, batch_size=64, seq_len=20,
        )
        print("\nSelf-test passed.")
    else:
        train(
            csv_dir    = args.csv_dir,
            output_dir = args.output_dir,
            asset      = args.asset,
            timeframe  = args.timeframe,
            seq_len    = args.seq_len,
            epochs     = args.epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            hidden_dim = args.hidden_dim,
            num_layers = args.num_layers,
            dropout    = args.dropout,
            val_split  = args.val_split,
            seed       = args.seed,
        )
