from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    jina_model_path: Path = Path(
        "/Volumes/vision/Downloads/codes_necessary/models/"
        "jina-embeddings-v5-omni-small-retrieval-mlx"
    )

    weaviate_host: str = "127.0.0.1"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051
    weaviate_collection: str = "MediaChunk"

    app_data_dir: Path = Path("./data")
    video_chunk_seconds: float = 10.0
    video_max_frames: int = 32
    transcript_chunk_tokens: int = 800
    transcript_overlap_tokens: int = 120
    search_limit: int = 12

    @property
    def uploads_dir(self) -> Path:
        return self.app_data_dir / "uploads"

    @property
    def assets_dir(self) -> Path:
        return self.app_data_dir / "assets"

    @property
    def registry_path(self) -> Path:
        return self.app_data_dir / "assets.json"


settings = Settings()
for directory in (settings.app_data_dir, settings.uploads_dir, settings.assets_dir):
    directory.mkdir(parents=True, exist_ok=True)
