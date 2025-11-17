# tests/test_entities.py
from pathlib import Path
from src.config.settings import get_settings
from src.entities.config_entities import build_entities

def test_build_entities_paths(tmp_path, monkeypatch):
    # Force artifact root to tmp dir, and deterministic timestamp
    monkeypatch.setenv("PATHS__ARTIFACT_DIR", str(tmp_path / "artifact"))
    ts = "20990101_010203"
    
    get_settings.cache_clear()

    ents = build_entities(ts=ts)

    # Training
    assert ents.training.timestamp == ts
    assert ents.training.artifact_dir == Path(tmp_path / "artifact" / ts)

    # Ingestion
    ingestion_root = ents.training.artifact_dir / "data_ingestion"
    assert ents.ingestion.data_ingestion_dir == ingestion_root
    assert ents.ingestion.feature_store_file_path.name == "data.csv"
    assert ents.ingestion.training_file_path.name == "train.csv"
    assert ents.ingestion.testing_file_path.name == "test.csv"
    assert 0 < ents.ingestion.train_test_split_ratio < 1
