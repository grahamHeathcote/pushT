import torch
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class Network(nn.Module):
    def __init__(self, input, hidden_1, hidden_2, output):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input, hidden_1),
            nn.SiLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.SiLU(),
            nn.Linear(hidden_2, output)
        )

    def forward(self, x):
        return self.model(x)
