from typing import Literal

from pydantic import BaseModel, Field


class PathIngestRequest(BaseModel):
    video_path: str
    transcript_path: str
    asset_name: str | None = None
    video_chunk_seconds: float | None = Field(default=None, ge=1.0, le=120.0)


class ImagePathIngestRequest(BaseModel):
    image_path: str
    asset_name: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    asset_id: str | None = None
    modality: Literal["all", "video", "transcript", "image"] = "all"
    limit: int = Field(default=12, ge=1, le=50)


class SearchResult(BaseModel):
    uuid: str
    asset_id: str
    modality: str
    chunk_id: str
    start_sec: float
    end_sec: float
    distance: float | None = None
    text: str | None = None
    speaker_ids: list[int] = []
    thumbnail_url: str | None = None
    image_url: str | None = None
