"""Data loading for MovieLens.

Reads the ratings file, applies a minimum-interactions filter, remaps user
and item ids to dense integer indices, and exposes the result as a pandas
DataFrame. Negative sampling and torch Dataset wrappers come later.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd


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
