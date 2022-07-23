"""Shared pytest fixtures."""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_ratings(tmp_path) -> str:
    """Write a tiny ratings file in ml-100k format and return its path."""
    rng = np.random.default_rng(0)
    rows = []
    seen = set()
    target = 200
    while len(rows) < target:
        u = int(rng.integers(1, 21))   # 20 users
        i = int(rng.integers(1, 31))   # 30 items
        if (u, i) in seen:
            continue
        seen.add((u, i))
        rows.append([u, i, int(rng.integers(1, 6)), 1_000_000_000 + len(rows)])
    df = pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])
    p = tmp_path / "u.data"
    df.to_csv(p, sep="\t", header=False, index=False)
    return str(p)
