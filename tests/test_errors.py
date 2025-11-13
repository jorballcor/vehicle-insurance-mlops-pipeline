import pandas as pd
import pytest

from src.entities.config_entities import build_entities
from src.components.data_ingestion import DataIngestion
from tests.conftest import FakeRepo


def test_empty_export_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("PATHS__ARTIFACT_DIR", str(tmp_path / "artifact"))
    ents = build_entities(ts="20990101_010203")

    empty_df = pd.DataFrame()
    di = DataIngestion(
        training_cfg=ents.training,
        ingestion_cfg=ents.ingestion,
        repo=FakeRepo(empty_df),
    )

    with pytest.raises(Exception):
        di.export_data_into_feature_store()
