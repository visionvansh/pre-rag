from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import PathIngestRequest


def main() -> None:
    sample = PathIngestRequest(
        video_path="video.mp4",
        transcript_path="asr.json",
        video_chunk_seconds=6.5,
    )
    assert sample.video_chunk_seconds == 6.5

    for invalid in (0.5, 121.0):
        try:
            PathIngestRequest(
                video_path="video.mp4",
                transcript_path="asr.json",
                video_chunk_seconds=invalid,
            )
        except Exception:
            continue
        raise AssertionError(f"Expected validation failure for {invalid}")

    print("Video chunk seconds validation OK")


if __name__ == "__main__":
    main()
