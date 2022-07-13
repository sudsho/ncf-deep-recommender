"""Training script for NCF.

Reads a YAML config, loads MovieLens, builds the chosen model,
trains with binary cross-entropy on positive + negative pairs.
Eval and MLflow tracking land in later commits.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .data import (
    NCFTrainDataset,
    build_user_pos_set,
    leave_one_out_split,
    load_ratings,
    remap_ids,
)
from .evaluate import evaluate_loo
from .model import build_model


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    torch.manual_seed(cfg["train"].get("seed", 42))

    df = load_ratings(
        cfg["data"]["path"],
        min_interactions=cfg["data"].get("min_user_interactions", 5),
    )
    df, _, _ = remap_ids(df)

    num_users = int(df["user_idx"].max()) + 1
    num_items = int(df["item_idx"].max()) + 1
    print(f"users={num_users} items={num_items} rows={len(df)}")

    user_pos = build_user_pos_set(df)
    train_df, test_df = leave_one_out_split(df)

    train_ds = NCFTrainDataset(
        train_df,
        user_pos,
        num_items=num_items,
        num_negatives=cfg["data"].get("num_negatives_train", 4),
        seed=cfg["train"].get("seed", 42),
    )
    loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
    )

    device = torch.device(cfg["train"].get("device", "cpu"))
    model = build_model(num_users, num_items, cfg["model"]).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    loss_fn = nn.BCEWithLogitsLoss()

    epochs = cfg["train"]["epochs"]
    for ep in range(1, epochs + 1):
        model.train()
        train_ds.resample()  # fresh negatives each epoch
        running = 0.0
        n_batches = 0
        t0 = time.time()
        for u, i, y in loader:
            u, i, y = u.to(device), i.to(device), y.to(device)
            logits = model(u, i)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        avg_loss = running / max(n_batches, 1)
        hr, nd = evaluate_loo(
            model,
            test_df,
            user_pos,
            num_items=num_items,
            top_k=cfg["eval"]["top_k"],
            num_negatives=cfg["data"].get("num_negatives_eval", 99),
            device=str(device),
            seed=cfg["train"].get("seed", 42),
        )
        print(
            f"epoch {ep}/{epochs} loss={avg_loss:.4f} "
            f"hr@{cfg['eval']['top_k']}={hr:.4f} ndcg@{cfg['eval']['top_k']}={nd:.4f} "
            f"time={time.time()-t0:.1f}s"
        )

    out_dir = cfg["artifacts"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, cfg["artifacts"]["model_name"])
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
            "config": cfg,
        },
        out_path,
    )
    print(f"saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    train(args.config)
