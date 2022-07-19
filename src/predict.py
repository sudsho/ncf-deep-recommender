"""Top-N recommendations for a single user.

Loads a trained checkpoint, scores every (user, item) pair the user has
not already seen, and returns the top-N items by predicted score.
"""
from __future__ import annotations

import argparse
from typing import List, Optional, Set, Tuple

import torch

from .model import build_model


def _load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def load_user_seen_from_csv(path: str) -> dict:
    """Helper for the API: load a user_idx -> seen-items dict from a csv with
    columns user_idx,item_idx (e.g. the training split saved alongside the model).
    """
    import pandas as pd

    df = pd.read_csv(path)
    out: dict = {}
    for u, it in zip(df["user_idx"].values, df["item_idx"].values):
        out.setdefault(int(u), set()).add(int(it))
    return out


def topn_for_user(
    user_idx: int,
    n: int = 10,
    checkpoint_path: str = "artifacts/neumf.pt",
    seen: Optional[Set[int]] = None,
    return_original_ids: bool = False,
) -> List[Tuple[int, float]]:
    """Return [(item, score), ...] sorted by score descending.

    If `return_original_ids=True`, item ids are mapped back to the original
    MovieLens ids via the saved item_map; otherwise they are the dense
    contiguous indices the model uses internally.
    """
    ckpt = _load_checkpoint(checkpoint_path)
    cfg = ckpt["config"]
    num_users = ckpt["num_users"]
    num_items = ckpt["num_items"]

    if user_idx < 0 or user_idx >= num_users:
        raise ValueError(
            f"user_idx={user_idx} out of range [0, {num_users})"
        )

    model = build_model(num_users, num_items, cfg["model"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    items = torch.arange(num_items, dtype=torch.long)
    users = torch.full((num_items,), user_idx, dtype=torch.long)
    with torch.no_grad():
        scores = model(users, items).cpu().numpy()

    # mask seen items so we don't recommend things the user already has
    if seen:
        for it in seen:
            if 0 <= it < num_items:
                scores[it] = -1e9

    # top-N
    n = min(n, num_items)
    top_idx = scores.argsort()[::-1][:n]
    pairs = [(int(i), float(scores[i])) for i in top_idx]

    if return_original_ids and "item_map" in ckpt:
        # item_map is original -> dense; invert it once
        inv = {v: k for k, v in ckpt["item_map"].items()}
        pairs = [(int(inv.get(i, i)), s) for i, s in pairs]
    return pairs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--user_id", type=int, required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--checkpoint", default="artifacts/neumf.pt")
    args = p.parse_args()
    for item, score in topn_for_user(args.user_id, n=args.n, checkpoint_path=args.checkpoint):
        print(f"{item}\t{score:.4f}")
