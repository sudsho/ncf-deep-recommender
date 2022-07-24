"""Tests for the FastAPI service.

We train a tiny model on synthetic data, point the api at the saved
checkpoint via env vars, and hit /health, /info, /recommend.
"""
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from fastapi.testclient import TestClient

from src.data import (
    NCFTrainDataset,
    build_user_pos_set,
    leave_one_out_split,
    load_ratings,
    remap_ids,
)
from src.model import build_model


def _train_tiny_checkpoint(tmp_path: Path, ratings_path: str) -> tuple:
    df = load_ratings(ratings_path, min_interactions=1)
    df, um, im = remap_ids(df)
    n_users = int(df["user_idx"].max()) + 1
    n_items = int(df["item_idx"].max()) + 1
    train_df, _ = leave_one_out_split(df)
    pos = build_user_pos_set(df)
    ds = NCFTrainDataset(train_df, pos, num_items=n_items, num_negatives=1, seed=0)

    cfg = {
        "model": {
            "type": "neumf",
            "embed_dim_gmf": 4,
            "embed_dim_mlp": 4,
            "mlp_layers": [8, 4],
            "dropout": 0.0,
        }
    }
    model = build_model(n_users, n_items, cfg["model"])
    # 1 quick optimisation step (we do not need to converge for an API test)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    u = torch.tensor(ds.users[:32], dtype=torch.long)
    i = torch.tensor(ds.items[:32], dtype=torch.long)
    y = torch.tensor(ds.labels[:32], dtype=torch.float32)
    logits = model(u, i)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    opt.step()

    ckpt_path = tmp_path / "neumf.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_users": n_users,
            "num_items": n_items,
            "user_map": um,
            "item_map": im,
            "config": cfg,
        },
        ckpt_path,
    )
    train_csv = tmp_path / "train_split.csv"
    train_df[["user_idx", "item_idx"]].to_csv(train_csv, index=False)
    return str(ckpt_path), str(train_csv), n_users, n_items


def test_health_no_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("NCF_CHECKPOINT", str(tmp_path / "missing.pt"))
    # reload module so env var is picked up
    import importlib

    from src.api import main as api_main

    importlib.reload(api_main)
    client = TestClient(api_main.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_recommend_returns_n_items(monkeypatch, tmp_path, tiny_ratings):
    ckpt, train_csv, n_users, n_items = _train_tiny_checkpoint(tmp_path, tiny_ratings)
    monkeypatch.setenv("NCF_CHECKPOINT", ckpt)
    monkeypatch.setenv("NCF_TRAIN_SPLIT", train_csv)
    import importlib

    from src.api import main as api_main

    importlib.reload(api_main)
    client = TestClient(api_main.app)
    r = client.get("/recommend", params={"user_id": 0, "n": 5})
    assert r.status_code == 200
    payload = r.json()
    assert payload["user_id"] == 0
    assert len(payload["items"]) <= 5
    for entry in payload["items"]:
        assert 0 <= entry["item_id"] < n_items


def test_recommend_bad_user(monkeypatch, tmp_path, tiny_ratings):
    ckpt, train_csv, n_users, n_items = _train_tiny_checkpoint(tmp_path, tiny_ratings)
    monkeypatch.setenv("NCF_CHECKPOINT", ckpt)
    monkeypatch.setenv("NCF_TRAIN_SPLIT", train_csv)
    import importlib

    from src.api import main as api_main

    importlib.reload(api_main)
    client = TestClient(api_main.app)
    r = client.get("/recommend", params={"user_id": 9999, "n": 3})
    assert r.status_code == 400
