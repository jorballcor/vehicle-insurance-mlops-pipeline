# tests/test_data_ingestion.py
from pathlib import Path
import pandas as pd

from src.config.settings import get_settings
from src.entities.config_entities import build_entities
from src.components.data_ingestion import DataIngestion
from tests.conftest import FakeRepo


def test_data_ingestion_happy_path(tmp_path, monkeypatch, small_df):
    # Set artifact dir to tmp for isolation
    monkeypatch.setenv("PATHS__ARTIFACT_DIR", str(tmp_path / "artifact"))
    
    get_settings.cache_clear()

    # Deterministic timestamp to make assertions easy
    ents = build_entities(ts="20990101_010203")

    di = DataIngestion(
        training_cfg=ents.training,
        ingestion_cfg=ents.ingestion,
        repo=FakeRepo(small_df),
        target_column="Response",      # also exercises stratify path
        split_random_state=123,
        shuffle=True,
    )

    artifact = di.initiate_data_ingestion()

    # Files exist
    assert artifact.trained_file_path.exists()
    assert artifact.test_file_path.exists()

    # Read back and check shapes add up
    train_df = pd.read_csv(artifact.trained_file_path)
    test_df = pd.read_csv(artifact.test_file_path)
    assert len(train_df) + len(test_df) == len(small_df)

    # Feature store file should also exist
    assert ents.ingestion.feature_store_file_path.exists()


def test_deterministic_split(tmp_path, monkeypatch, small_df):
    monkeypatch.setenv("PATHS__ARTIFACT_DIR", str(tmp_path / "artifact"))
    
    get_settings.cache_clear()
    
    ents = build_entities(ts="20990101_010203")

    # Run ingestion twice with same seed to ensure identical splits
    di1 = DataIngestion(
        training_cfg=ents.training,
        ingestion_cfg=ents.ingestion,
        repo=FakeRepo(small_df),
        split_random_state=777,
        shuffle=True,
    )
    di1.export_data_into_feature_store()
    df1 = pd.read_csv(ents.ingestion.feature_store_file_path)
    di1.split_data_as_train_test(df1)

    train1 = pd.read_csv(ents.ingestion.training_file_path)
    test1 = pd.read_csv(ents.ingestion.testing_file_path)

    # Use a different timestamp (new run context) but same seed and data
    ents2 = build_entities(ts="20990101_010204")
    di2 = DataIngestion(
        training_cfg=ents2.training,
        ingestion_cfg=ents2.ingestion,
        repo=FakeRepo(small_df),
        split_random_state=777,
        shuffle=True,
    )
    di2.export_data_into_feature_store()
    df2 = pd.read_csv(ents2.ingestion.feature_store_file_path)
    di2.split_data_as_train_test(df2)

    train2 = pd.read_csv(ents2.ingestion.training_file_path)
    test2 = pd.read_csv(ents2.ingestion.testing_file_path)

    # Deterministic: same rows in train/test (order may differ; compare sets of indices/rows)
    # Here we compare sorted tuples of rows for simplicity.
    assert sorted(map(tuple, train1.values.tolist())) == sorted(map(tuple, train2.values.tolist()))
    assert sorted(map(tuple, test1.values.tolist())) == sorted(map(tuple, test2.values.tolist()))
