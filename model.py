import torch
import torch.nn as nn
import torch.nn.functional as F

class PatternModel(nn.Module):
    def __init__(self, num_patterns=3, seq_len=7, dropout=0.3):
        super().__init__()
        
        # Embeddings: 7 tokens → richer 32-dim representations
        self.embed = nn.Embedding(7, 32, padding_idx=0)
        
        # Bidirectional GRU for pattern context (before/after)
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=128,
            num_layers=2,           # Deeper for pattern complexity
            batch_first=True,
            bidirectional=True,      # Capture left+right context
            dropout=dropout if 2 > 1 else 0
        )
        
        # Attention mechanism (learn which timesteps matter)
        self.attention = nn.Sequential(
            nn.Linear(256, 64),      # 256 = 128*2 (biGRU)
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, num_patterns)  # 3 outputs, no softmax (CE loss handles it)
        )
        
    def forward(self, x):
        # x: (batch, seq_len) of token indices
        x = self.embed(x)  # (batch, seq_len, 32)
        
        # GRU features
        gru_out, _ = self.gru(x)  # (batch, seq_len, 256)
        
        # Attention-weighted pooling (better than just last timestep)
        attn_weights = F.softmax(self.attention(gru_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * gru_out, dim=1)  # (batch, 256)
        
        # Classify
        return self.classifier(context)  # (batch, 3) - raw logits
