"""Pydantic v1 request/response schemas for the API."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel


class Recommendation(BaseModel):
    item_id: int
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    items: List[Recommendation]


class HealthResponse(BaseModel):
    status: str
    checkpoint_loaded: bool
