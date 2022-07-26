"""FastAPI inference service for NCF.

GET /recommend?user_id=<int>&n=<int>  -> top-N items for that user.
GET /health                           -> simple health check.
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException, Query

from ..model import build_model
from ..predict import load_user_seen_from_csv, topn_for_user
from .schemas import Recommendation, RecommendResponse

app = FastAPI(title="ncf-deep-recommender", version="0.1.0")

CHECKPOINT_PATH = os.environ.get("NCF_CHECKPOINT", "artifacts/neumf.pt")
TRAIN_SPLIT_PATH = os.environ.get("NCF_TRAIN_SPLIT", "artifacts/train_split.csv")
DEFAULT_TOP_N = int(os.environ.get("NCF_DEFAULT_N", "10"))
MAX_TOP_N = int(os.environ.get("NCF_MAX_N", "200"))

# cache the loaded model + maps so we don't re-read the checkpoint on every request
_state: dict = {"model": None, "ckpt": None, "seen": None}


def _ensure_loaded() -> dict:
    if _state["model"] is None:
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(CHECKPOINT_PATH)
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model = build_model(ckpt["num_users"], ckpt["num_items"], ckpt["config"]["model"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _state["model"] = model
        _state["ckpt"] = ckpt
        if os.path.exists(TRAIN_SPLIT_PATH):
            _state["seen"] = load_user_seen_from_csv(TRAIN_SPLIT_PATH)
        else:
            _state["seen"] = {}
    return _state


@app.get("/health")
def health() -> dict:
    info = {"status": "ok", "checkpoint_loaded": False}
    if os.path.exists(CHECKPOINT_PATH):
        info["checkpoint_loaded"] = _state["model"] is not None
        info["checkpoint_path"] = CHECKPOINT_PATH
    return info


@app.get("/info")
def info() -> dict:
    """Return model metadata: variant, num users, num items."""
    if not os.path.exists(CHECKPOINT_PATH):
        raise HTTPException(status_code=503, detail="checkpoint not found")
    s = _ensure_loaded()
    cfg = s["ckpt"]["config"]
    return {
        "model_type": cfg["model"]["type"],
        "num_users": s["ckpt"]["num_users"],
        "num_items": s["ckpt"]["num_items"],
        "embed_dim_gmf": cfg["model"].get("embed_dim_gmf"),
        "embed_dim_mlp": cfg["model"].get("embed_dim_mlp"),
        "mlp_layers": cfg["model"].get("mlp_layers"),
    }


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_id: int = Query(..., ge=0, description="User index (0..num_users-1)"),
    n: int = Query(DEFAULT_TOP_N, ge=1, le=MAX_TOP_N, description="How many items to return"),
    original_ids: bool = Query(
        False, description="If true, return original MovieLens ids."
    ),
) -> RecommendResponse:
    try:
        s = _ensure_loaded()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"checkpoint not found: {CHECKPOINT_PATH}. train a model first.",
        )
    try:
        seen_for_user = s["seen"].get(user_id) if s.get("seen") else None
        items = topn_for_user(
            user_id,
            n=n,
            checkpoint_path=CHECKPOINT_PATH,
            seen=seen_for_user,
            return_original_ids=original_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RecommendResponse(
        user_id=user_id,
        items=[Recommendation(item_id=i, score=s) for i, s in items],
    )
