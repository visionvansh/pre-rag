from typing import Literal

from pydantic import BaseModel, Field


class LIImagePathIngestRequest(BaseModel):
    image_path: str
    asset_name: str | None = None


class LITextPathIngestRequest(BaseModel):
    text_path: str
    asset_name: str | None = None


class LISearchRequest(BaseModel):
    query: str = Field(min_length=1)
    asset_id: str | None = None
    modality: Literal["all", "image", "text"] = "all"
    limit: int = Field(default=12, ge=1, le=50)
    late_candidate_limit: int | None = Field(default=None, ge=1, le=200)
    m0_candidate_limit: int | None = Field(default=None, ge=1, le=100)
    rerank: bool = True
