"""FastAPI inference service for NCF.

GET /recommend?user_id=<int>&n=<int>  -> top-N items for that user.
GET /health                           -> simple health check.
"""
from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from ..predict import topn_for_user

app = FastAPI(title="ncf-deep-recommender", version="0.1.0")

CHECKPOINT_PATH = os.environ.get("NCF_CHECKPOINT", "artifacts/neumf.pt")
DEFAULT_TOP_N = int(os.environ.get("NCF_DEFAULT_N", "10"))
MAX_TOP_N = int(os.environ.get("NCF_MAX_N", "200"))


class Recommendation(BaseModel):
    item_id: int
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    items: List[Recommendation]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_id: int = Query(..., ge=0, description="User index (0..num_users-1)"),
    n: int = Query(DEFAULT_TOP_N, ge=1, le=MAX_TOP_N, description="How many items to return"),
    original_ids: bool = Query(
        False, description="If true, return original MovieLens ids."
    ),
) -> RecommendResponse:
    if not os.path.exists(CHECKPOINT_PATH):
        raise HTTPException(
            status_code=503,
            detail=f"checkpoint not found: {CHECKPOINT_PATH}. train a model first.",
        )
    try:
        items = topn_for_user(
            user_id,
            n=n,
            checkpoint_path=CHECKPOINT_PATH,
            return_original_ids=original_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RecommendResponse(
        user_id=user_id,
        items=[Recommendation(item_id=i, score=s) for i, s in items],
    )
