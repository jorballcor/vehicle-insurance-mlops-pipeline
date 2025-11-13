# tests/test_stratify.py
import pandas as pd
from src.entities.config_entities import build_entities
from src.components.data_ingestion import DataIngestion
from tests.conftest import FakeRepo

def _class_ratio(series):
    counts = series.value_counts(normalize=True)
    return counts.to_dict()

def test_stratified_split_preserves_ratio(tmp_path, monkeypatch, imbalanced_df):
    monkeypatch.setenv("PATHS__ARTIFACT_DIR", str(tmp_path / "artifact"))

    ents = build_entities(ts="20990101_010203")

    di = DataIngestion(
        training_cfg=ents.training,
        ingestion_cfg=ents.ingestion,
        repo=FakeRepo(imbalanced_df),
        target_column="Response",     # key: enables stratify
        split_random_state=42,
        shuffle=True,
    )

    df = di.export_data_into_feature_store()
    di.split_data_as_train_test(df)

    train_df = pd.read_csv(ents.ingestion.training_file_path)
    test_df = pd.read_csv(ents.ingestion.testing_file_path)

    global_ratio = _class_ratio(df["Response"])
    train_ratio = _class_ratio(train_df["Response"])
    test_ratio  = _class_ratio(test_df["Response"])

    # With stratify, train/test ratios should be close to the global ratio
    # (exact equality is not guaranteed, use a tolerance)
    for label in global_ratio:
        assert abs(train_ratio.get(label, 0) - global_ratio[label]) < 0.05
        assert abs(test_ratio.get(label, 0) - global_ratio[label]) < 0.05
