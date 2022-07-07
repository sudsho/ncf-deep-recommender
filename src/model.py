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


class MLP(nn.Module):
    """Deep MLP over concatenated user/item embeddings."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 32,
        layers: List[int] = [64, 32, 16, 8],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)

        in_dim = 2 * embed_dim
        modules: List[nn.Module] = []
        for h in layers:
            modules.append(nn.Linear(in_dim, h))
            modules.append(nn.ReLU())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            in_dim = h
        self.mlp = nn.Sequential(*modules)
        self.out = nn.Linear(in_dim, 1)
        self._init()

    def _init(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(users)
        i = self.item_emb(items)
        x = torch.cat([u, i], dim=-1)
        x = self.mlp(x)
        return self.out(x).squeeze(-1)


class NeuMF(nn.Module):
    """NeuMF: GMF and MLP towers concatenated, then a final linear head.

    The two towers learn separate user/item embeddings, as in the paper.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim_gmf: int = 16,
        embed_dim_mlp: int = 32,
        layers: List[int] = [64, 32, 16, 8],
        dropout: float = 0.0,
    ):
        super().__init__()
        # GMF tower
        self.user_gmf = nn.Embedding(num_users, embed_dim_gmf)
        self.item_gmf = nn.Embedding(num_items, embed_dim_gmf)

        # MLP tower
        self.user_mlp = nn.Embedding(num_users, embed_dim_mlp)
        self.item_mlp = nn.Embedding(num_items, embed_dim_mlp)
        in_dim = 2 * embed_dim_mlp
        modules: List[nn.Module] = []
        for h in layers:
            modules.append(nn.Linear(in_dim, h))
            modules.append(nn.ReLU())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            in_dim = h
        self.mlp = nn.Sequential(*modules)
        self.mlp_out_dim = in_dim

        # Fusion head
        self.head = nn.Linear(embed_dim_gmf + self.mlp_out_dim, 1)
        self._init()

    def _init(self):
        for emb in [self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp]:
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        # GMF branch
        ug = self.user_gmf(users)
        ig = self.item_gmf(items)
        gmf_out = ug * ig

        # MLP branch
        um = self.user_mlp(users)
        im = self.item_mlp(items)
        mlp_in = torch.cat([um, im], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Fusion
        x = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.head(x).squeeze(-1)


def build_model(num_users: int, num_items: int, cfg: dict) -> nn.Module:
    """Factory for the three model variants based on cfg['type']."""
    kind = cfg.get("type", "neumf").lower()
    if kind == "gmf":
        return GMF(num_users, num_items, embed_dim=cfg.get("embed_dim_gmf", 16))
    if kind == "mlp":
        return MLP(
            num_users,
            num_items,
            embed_dim=cfg.get("embed_dim_mlp", 32),
            layers=cfg.get("mlp_layers", [64, 32, 16, 8]),
            dropout=cfg.get("dropout", 0.0),
        )
    if kind == "neumf":
        return NeuMF(
            num_users,
            num_items,
            embed_dim_gmf=cfg.get("embed_dim_gmf", 16),
            embed_dim_mlp=cfg.get("embed_dim_mlp", 32),
            layers=cfg.get("mlp_layers", [64, 32, 16, 8]),
            dropout=cfg.get("dropout", 0.0),
        )
    raise ValueError(f"unknown model type: {kind}")
