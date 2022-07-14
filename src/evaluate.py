"""Leave-one-out HR@K and NDCG@K evaluation.

For each user we score 1 held-out positive against `num_negatives_eval`
sampled negatives, sort by predicted score, and check whether the positive
landed in the top K.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch


def hit_rate(ranked: List[int], target: int) -> int:
    return 1 if target in ranked else 0


def ndcg(ranked: List[int], target: int) -> float:
    if target not in ranked:
        return 0.0
    rank = ranked.index(target)
    return 1.0 / math.log2(rank + 2)


def evaluate_loo(
    model: torch.nn.Module,
    test_df,
    user_pos: Dict[int, set],
    num_items: int,
    top_k: int = 10,
    num_negatives: int = 99,
    device: str = "cpu",
    seed: int = 7,
) -> Tuple[float, float]:
    """Run leave-one-out evaluation.

    For each (user, positive_item) row in `test_df`, sample
    `num_negatives` items the user has not seen, score the candidate
    set with the model, and compute HR@K and NDCG@K.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    hits: List[int] = []
    ndcgs: List[float] = []

    with torch.no_grad():
        # itertuples is much faster than iterrows for big test sets
        for row in test_df.itertuples(index=False):
            u = int(getattr(row, "user_idx"))
            pos = int(getattr(row, "item_idx"))
            seen = user_pos.get(u, set())

            negatives: List[int] = []
            while len(negatives) < num_negatives:
                j = int(rng.integers(0, num_items))
                if j != pos and j not in seen and j not in negatives:
                    negatives.append(j)
            candidates = [pos] + negatives

            users_t = torch.full((len(candidates),), u, dtype=torch.long, device=device)
            items_t = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = model(users_t, items_t).detach().cpu().numpy()

            order = np.argsort(-scores)
            ranked = [candidates[k] for k in order[:top_k]]
            hits.append(hit_rate(ranked, pos))
            ndcgs.append(ndcg(ranked, pos))

    return float(np.mean(hits)), float(np.mean(ndcgs))
