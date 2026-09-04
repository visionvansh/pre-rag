from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    jina_model_path: Path = Path(
        "/Volumes/vision/Downloads/codes_necessary/models/"
        "jina-embeddings-v5-omni-small-retrieval-mlx"
    )
    jina_reranker_path: Path = Path(
        "/Volumes/vision/Downloads/codes_necessary/models/jina-reranker-m0"
    )
    jina_v4_model_path: Path = Path(
        "/Volumes/vision/Downloads/codes_necessary/models/jina-embeddings-v4"
    )
    jina_v4_revision: str = "853c867b65b749f3c3c72a06868140d842e04f06"
    jina_v4_python_path: Path = Path(".venv-jina-v4/bin/python")
    jina_v4_worker_timeout_sec: float = 1200.0
    jina_v4_device: str = "auto"
    jina_v4_attn_implementation: str = "eager"
    jina_v4_text_max_length: int = 8192

    weaviate_host: str = "127.0.0.1"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051
    weaviate_collection: str = "MediaChunk"
    li_weaviate_collection: str = "JinaV4LateChunk"

    app_data_dir: Path = Path("./data")
    video_chunk_seconds: float = 10.0
    video_max_frames: int = 32
    transcript_chunk_tokens: int = 800
    transcript_overlap_tokens: int = 120
    search_limit: int = 12

    # Parallel Jina-v4 late-interaction experiment. Keep the same initial text
    # chunk baseline as the existing app so retrieval architecture is the main variable.
    li_text_chunk_tokens: int = 800
    li_text_overlap_tokens: int = 120
    li_search_limit: int = 12
    li_late_candidate_limit: int = 64
    li_late_max_candidates: int = 200
    li_m0_candidate_limit: int = 40
    li_low_memory_mode: bool = False

    # Stage-two reranking. Dense retrieval remains the candidate generator only.
    # m0 is launched in a separate Python environment because its Qwen2-VL custom
    # code is incompatible with the newer Transformers stack needed by Jina-v5 Omni.
    reranker_enabled: bool = True
    reranker_device: str = "auto"  # auto -> MPS on Apple Silicon when available
    reranker_python_path: Path = Path(".venv-reranker/bin/python")
    reranker_worker_timeout_sec: float = 900.0
    reranker_candidate_limit: int = 64
    reranker_max_candidates: int = 200
    reranker_text_batch_size: int = 4
    reranker_image_batch_size: int = 1
    reranker_text_max_length: int = 3072
    reranker_image_max_length: int = 4096
    reranker_query_max_length: int = 512
    reranker_attn_implementation: str = "eager"

    @property
    def uploads_dir(self) -> Path:
        return self.app_data_dir / "uploads"

    @property
    def assets_dir(self) -> Path:
        return self.app_data_dir / "assets"

    @property
    def registry_path(self) -> Path:
        return self.app_data_dir / "assets.json"

    @property
    def li_data_dir(self) -> Path:
        return self.app_data_dir / "late_interaction"

    @property
    def li_uploads_dir(self) -> Path:
        return self.li_data_dir / "uploads"

    @property
    def li_assets_dir(self) -> Path:
        return self.li_data_dir / "assets"

    @property
    def li_registry_path(self) -> Path:
        return self.li_data_dir / "assets.json"

    @property
    def li_query_tmp_dir(self) -> Path:
        return self.li_data_dir / "query_tmp"


settings = Settings()
for directory in (
    settings.app_data_dir,
    settings.uploads_dir,
    settings.assets_dir,
    settings.li_data_dir,
    settings.li_uploads_dir,
    settings.li_assets_dir,
    settings.li_query_tmp_dir,
):
    directory.mkdir(parents=True, exist_ok=True)
