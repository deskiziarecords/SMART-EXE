import torch
import pandas as pd
from model import CandleLM
from encoder import MAP

# --------------------------------------------------
# 1. LOAD YOUR DATA
# --------------------------------------------------

df = pd.read_csv("training_data.csv")

print(f"Loaded {len(df)} samples")

# Convert to training pairs
pairs = []

for _, row in df.iterrows():
    seq = row["input"]
    target = row["target"]

    # convert to indices
    x = [MAP[s] for s in seq if s in MAP]
    y = MAP[target]

    if len(x) > 0:
        pairs.append((x, y))

print(f"Valid pairs: {len(pairs)}")

# --------------------------------------------------
# 2. DATASET
# --------------------------------------------------

import torch
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, pairs):
        self.X = [torch.tensor(p[0], dtype=torch.long) for p in pairs]
        self.y = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

dataset = SeqDataset(pairs)
loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=lambda b: (
    torch.nn.utils.rnn.pad_sequence([x for x,_ in b], batch_first=True),
    torch.tensor([y for _,y in b])
))

# --------------------------------------------------
# 3. MODEL
# --------------------------------------------------

model = CandleLM()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

# --------------------------------------------------
# 4. TRAIN LOOP
# --------------------------------------------------

for epoch in range(10):
    total_loss = 0

    for X, y in loader:
        logits = model(X)
        loss = torch.nn.functional.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: loss={total_loss:.4f}")

# --------------------------------------------------
# 5. SAVE
# --------------------------------------------------

torch.save(model.state_dict(), "clm_eurusd.pt")
print("✅ Saved clm_eurusd.pt")
