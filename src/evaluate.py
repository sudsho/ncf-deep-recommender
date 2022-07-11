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
