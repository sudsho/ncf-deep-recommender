"""NCF models: GMF, MLP, and the fused NeuMF.

Reference: He et al., "Neural Collaborative Filtering", WWW 2017.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class GMF(nn.Module):
    """Generalized Matrix Factorization.

    Element-wise product of user and item embeddings, then a single linear
    layer to produce a logit. With the linear weights tied to ones and a
    sigmoid output this reduces to plain MF.
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int = 16):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.out = nn.Linear(embed_dim, 1)
        self._init()

    def _init(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(users)
        i = self.item_emb(items)
        x = u * i
        return self.out(x).squeeze(-1)
