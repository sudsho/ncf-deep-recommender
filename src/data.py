"""Data loading for MovieLens.

Reads the ratings file, applies a minimum-interactions filter, remaps user
and item ids to dense integer indices, samples negatives for implicit
feedback training, and wraps everything in a torch Dataset.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_ratings(path: str, min_interactions: int = 5) -> pd.DataFrame:
    """Load a MovieLens ratings file and filter cold-start users.

    Supports both the 100K format (`u.data`, tab-separated, no header) and the
    25M format (`ratings.csv`, comma-separated, with header).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ratings file not found: {path}. "
            "run scripts/download_data.sh first."
        )

    if path.endswith(".csv"):
        df = pd.read_csv(path)
        df = df.rename(columns={"userId": "user", "movieId": "item"})
    else:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["user", "item", "rating", "timestamp"],
        )

    # drop users with too few interactions
    counts = df["user"].value_counts()
    keep = counts[counts >= min_interactions].index
    df = df[df["user"].isin(keep)].copy()
    return df


def remap_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
    """Remap user/item ids to contiguous 0..N-1 indices."""
    user_map = {u: i for i, u in enumerate(df["user"].unique())}
    item_map = {it: i for i, it in enumerate(df["item"].unique())}
    df = df.copy()
    df["user_idx"] = df["user"].map(user_map)
    df["item_idx"] = df["item"].map(item_map)
    return df, user_map, item_map


def leave_one_out_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent interaction per user as the test sample."""
    df = df.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)
    test = df.groupby("user_idx", as_index=False).tail(1)
    train = df.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_user_pos_set(df: pd.DataFrame) -> Dict[int, set]:
    """Map user_idx -> set of items they interacted with (train + test)."""
    pos: Dict[int, set] = {}
    for u, it in zip(df["user_idx"].values, df["item_idx"].values):
        pos.setdefault(int(u), set()).add(int(it))
    return pos


def sample_negatives(
    user_pos: Dict[int, set],
    num_items: int,
    num_negatives: int,
    rng: np.random.Generator,
) -> Dict[int, List[int]]:
    """For each user, sample `num_negatives` items they have NOT interacted with."""
    out: Dict[int, List[int]] = {}
    all_items = np.arange(num_items)
    for u, pos in user_pos.items():
        # rejection sampling, fast enough for ml-100k / ml-25m subsets
        seen = pos
        sampled: List[int] = []
        while len(sampled) < num_negatives:
            cand = int(rng.choice(all_items))
            if cand not in seen and cand not in sampled:
                sampled.append(cand)
        out[u] = sampled
    return out


class NCFTrainDataset(Dataset):
    """Pairs of (user, item, label) where label is 1 for positives, 0 for negatives.

    For each positive interaction we draw `num_negatives` negative items uniformly
    at random outside the user's positive set. This re-sampling is rebuilt every
    epoch by calling `resample()`.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        user_pos: Dict[int, set],
        num_items: int,
        num_negatives: int = 4,
        seed: int = 42,
    ):
        self.train_df = train_df.reset_index(drop=True)
        self.user_pos = user_pos
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)
        self.users: np.ndarray = np.array([], dtype=np.int64)
        self.items: np.ndarray = np.array([], dtype=np.int64)
        self.labels: np.ndarray = np.array([], dtype=np.float32)
        self.resample()

    def resample(self) -> None:
        pos_users = self.train_df["user_idx"].values.astype(np.int64)
        pos_items = self.train_df["item_idx"].values.astype(np.int64)
        n_pos = len(pos_users)
        n_neg = n_pos * self.num_negatives

        neg_users = np.repeat(pos_users, self.num_negatives)
        neg_items = np.empty(n_neg, dtype=np.int64)
        # cache the user->set lookup once per resample to avoid the int() each iter
        cache = self.user_pos
        N = self.num_items
        empty: set = set()
        for k in range(n_neg):
            u = int(neg_users[k])
            seen = cache.get(u, empty)
            while True:
                j = int(self.rng.integers(0, N))
                if j not in seen:
                    neg_items[k] = j
                    break

        self.users = np.concatenate([pos_users, neg_users])
        self.items = np.concatenate([pos_items, neg_items])
        self.labels = np.concatenate(
            [np.ones(n_pos, dtype=np.float32), np.zeros(n_neg, dtype=np.float32)]
        )

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )
