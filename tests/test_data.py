"""Tests for src/data.py."""
import numpy as np
import pandas as pd

from src.data import (
    NCFTrainDataset,
    build_user_pos_set,
    leave_one_out_split,
    load_ratings,
    remap_ids,
)


def test_load_ratings_filters_cold_users(tiny_ratings):
    df = load_ratings(tiny_ratings, min_interactions=3)
    counts = df.groupby("user").size()
    assert (counts >= 3).all()


def test_remap_ids_dense(tiny_ratings):
    df = load_ratings(tiny_ratings, min_interactions=1)
    df, um, im = remap_ids(df)
    assert df["user_idx"].min() == 0
    assert df["user_idx"].max() == len(um) - 1
    assert df["item_idx"].max() == len(im) - 1


def test_leave_one_out_test_size_per_user(tiny_ratings):
    df = load_ratings(tiny_ratings, min_interactions=2)
    df, _, _ = remap_ids(df)
    train_df, test_df = leave_one_out_split(df)
    # exactly one held-out row per user
    assert test_df["user_idx"].nunique() == len(test_df)
    # train + test == original
    assert len(train_df) + len(test_df) == len(df)


def test_negatives_disjoint_from_positives(tiny_ratings):
    df = load_ratings(tiny_ratings, min_interactions=2)
    df, _, _ = remap_ids(df)
    n_items = int(df["item_idx"].max()) + 1
    pos = build_user_pos_set(df)
    ds = NCFTrainDataset(df, pos, num_items=n_items, num_negatives=3, seed=1)
    # all label-0 rows must be items NOT in the user's positive set
    is_pos = ds.labels.astype(bool)
    neg_users = ds.users[~is_pos]
    neg_items = ds.items[~is_pos]
    for u, i in zip(neg_users, neg_items):
        assert int(i) not in pos[int(u)]
