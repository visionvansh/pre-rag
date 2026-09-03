from pathlib import Path

from app.registry import AssetRegistry


def test_legacy_video_registry_entry_is_migrated_publicly(tmp_path: Path):
    reg = AssetRegistry(tmp_path / "assets.json")
    reg.upsert("v1", {
        "name": "Legacy video",
        "video_path": str(tmp_path / "missing.mp4"),
        "video_chunks": 12,
        "transcript_chunks": 3,
        "weaviate_objects": 15,
    })
    row = reg.get_public("v1")
    assert row["asset_type"] == "video"
    assert row["video_chunks"] == 12
    assert row["transcript_chunks"] == 3


def test_image_registry_entry_public_counts(tmp_path: Path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    reg = AssetRegistry(tmp_path / "assets.json")
    reg.upsert("i1", {
        "name": "Images",
        "asset_type": "images",
        "image_paths": [str(image)],
        "image_count": 1,
        "weaviate_objects": 1,
    })
    row = reg.get_public("i1")
    assert row["asset_type"] == "images"
    assert row["image_count"] == 1
    assert row["media_available"] is True
