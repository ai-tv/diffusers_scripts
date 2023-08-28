import torch
from torch import nn


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class ResMlpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        y = self.block(x)
        if self.in_channels == self.out_channels:
            return x + y
        else:
            return x


class ResMlp(nn.Module):

    def __init__(self, num_layers=3, num_hidden_states=768):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden_states = num_hidden_states
        self.fc1 = nn.Linear(512, num_hidden_states)
        self.act1 = nn.SiLU()
        self.blocks = nn.ModuleList([
            ResMlpBlock(num_hidden_states, num_hidden_states)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(self.num_hidden_states, self.num_hidden_states)
        # nn.init.zeros_(self.out.weight)
        # nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        for module in self.blocks:
            x = module(x)
        y = self.out(x)
        return y
