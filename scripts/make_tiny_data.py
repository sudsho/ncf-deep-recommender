"""Create a tiny fake MovieLens-style ratings file for CI / smoke tests.

Writes a tab-separated file in the ml-100k format:
user, item, rating, timestamp.
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def make(n_users: int = 50, n_items: int = 80, density: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_inter = int(n_users * n_items * density)
    rows = []
    seen = set()
    while len(rows) < n_inter:
        u = int(rng.integers(1, n_users + 1))
        i = int(rng.integers(1, n_items + 1))
        if (u, i) in seen:
            continue
        seen.add((u, i))
        r = int(rng.integers(1, 6))
        ts = int(1_000_000_000 + rng.integers(0, 1_000_000))
        rows.append((u, i, r, ts))
    return rows


def write(rows, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for u, i, r, ts in rows:
            f.write(f"{u}\t{i}\t{r}\t{ts}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/tiny/u.data")
    p.add_argument("--users", type=int, default=50)
    p.add_argument("--items", type=int, default=80)
    p.add_argument("--density", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rows = make(args.users, args.items, args.density, args.seed)
    write(rows, args.out)
    print(f"wrote {len(rows)} rows to {args.out}")
