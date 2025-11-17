from src.config.settings import get_settings
from src.pipeline.training_pipeline import TrainPipeline
from src.entities.config_entities import build_entities
from src.components.data_ingestion import DataIngestion
from tests.conftest import FakeRepo, small_df  


def test_pipeline_runs_ingestion_with_fake_repo(tmp_path, monkeypatch, small_df):
    # Redirigir artifacts al tmp
    monkeypatch.setenv("PATHS__ARTIFACT_DIR", str(tmp_path / "artifact"))
    
    get_settings.cache_clear()


    ents = build_entities(ts="20990101_010203")
    pipeline = TrainPipeline(entities=ents)

    # Sobrescribir solo la etapa de ingestion con un FakeRepo
    def custom_start_data_ingestion():
        di = DataIngestion(
            training_cfg=ents.training,
            ingestion_cfg=ents.ingestion,
            repo=FakeRepo(small_df),
        )
        return di.initiate_data_ingestion()

    pipeline.start_data_ingestion = custom_start_data_ingestion

    artifact = pipeline.start_data_ingestion()

    assert artifact.trained_file_path.exists()
    assert artifact.test_file_path.exists()
