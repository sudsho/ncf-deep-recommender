"""Tests for src/evaluate.py."""
import math

import pandas as pd
import pytest
import torch

from src.evaluate import evaluate_loo, hit_rate, ndcg


def test_hit_rate():
    assert hit_rate([1, 2, 3], 2) == 1
    assert hit_rate([1, 2, 3], 9) == 0


def test_ndcg_top_position():
    # target at rank 0 -> 1 / log2(2) = 1.0
    assert math.isclose(ndcg([7, 1, 2], 7), 1.0)


def test_ndcg_lower_position():
    # target at rank 2 -> 1 / log2(4) = 0.5
    assert math.isclose(ndcg([1, 2, 7], 7), 0.5)


def test_ndcg_missing_target():
    assert ndcg([1, 2, 3], 9) == 0.0


class _PerfectModel(torch.nn.Module):
    """Always scores item==42 highest. Used to verify HR/NDCG plumbing."""

    def forward(self, users, items):
        return (items == 42).float()


def test_evaluate_loo_perfect_model():
    test_df = pd.DataFrame({"user_idx": [0, 1], "item_idx": [42, 42]})
    user_pos = {0: {42}, 1: {42}}
    hr, nd = evaluate_loo(
        _PerfectModel(),
        test_df,
        user_pos,
        num_items=200,
        top_k=5,
        num_negatives=20,
        device="cpu",
        seed=0,
    )
    assert hr == 1.0
    assert nd == 1.0


class _RandomModel(torch.nn.Module):
    """Random scores. HR with K=1 over 99 negatives + 1 positive ~ 0.01."""

    def forward(self, users, items):
        return torch.randn(items.shape[0])


def test_evaluate_loo_returns_floats():
    test_df = pd.DataFrame({"user_idx": [0], "item_idx": [5]})
    user_pos = {0: {5}}
    hr, nd = evaluate_loo(
        _RandomModel(), test_df, user_pos, num_items=50, top_k=10,
        num_negatives=20, device="cpu", seed=0,
    )
    assert isinstance(hr, float)
    assert isinstance(nd, float)
    assert 0.0 <= hr <= 1.0
    assert 0.0 <= nd <= 1.0
