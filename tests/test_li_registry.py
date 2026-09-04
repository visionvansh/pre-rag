from pathlib import Path

from app.late_interaction.registry import LateInteractionRegistry


def test_registry_is_separate_and_round_trips(tmp_path: Path):
    registry = LateInteractionRegistry(tmp_path / "late_interaction" / "assets.json")
    registry.upsert(
        "abc",
        {
            "name": "test",
            "asset_type": "texts",
            "text_paths": [str(tmp_path / "a.txt")],
            "text_chunks": 3,
            "weaviate_objects": 3,
            "late_vectors": 900,
        },
    )
    item = registry.get_public("abc")
    assert item is not None
    assert item["asset_type"] == "texts"
    assert item["late_vectors"] == 900
    assert "late_interaction" in str(registry.path)
