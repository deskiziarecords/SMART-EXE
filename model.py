import torch
import torch.nn as nn

class PatternModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(7, 16)
        self.gru = nn.GRU(16, 64, batch_first=True)
        self.fc = nn.Linear(64,1)

    def forward(self,x):
        x = self.embed(x)
        o,_ = self.gru(x)
        o = o[:,-1,:]
        return torch.sigmoid(self.fc(o))
