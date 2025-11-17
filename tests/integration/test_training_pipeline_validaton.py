from src.config.settings import get_settings
from src.pipeline.training_pipeline import TrainPipeline
from src.entities.config_entities import build_entities
from src.components.data_validation import DataValidation
from src.entities.artifact_entity import DataIngestionArtifact
from tests.conftest import FakeRepo, small_df
import pandas as pd
from pathlib import Path


def test_pipeline_runs_validation_after_ingestion(tmp_path, monkeypatch, small_df):
    """Test that validation runs correctly after data ingestion in the pipeline"""
    # Setup environment
    monkeypatch.setenv("PATHS__ARTIFACT_DIR", str(tmp_path / "artifact"))
    get_settings.cache_clear()

    # Build pipeline entities
    ents = build_entities(ts="20990101_010203")
    pipeline = TrainPipeline(entities=ents)

    # Create a simple data ingestion artifact (simulating previous step)
    ingestion_dir = ents.ingestion.data_ingestion_dir / ents.ingestion.data_ingestion_dir
    ingestion_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = ingestion_dir / "train.csv"
    test_file = ingestion_dir / "test.csv"
    
    # Split small_df into train/test
    train_data = small_df.iloc[:4]  # First 4 rows for training
    test_data = small_df.iloc[4:]   # Last 2 rows for testing
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    # Create ingestion artifact
    ingestion_artifact = DataIngestionArtifact(
        trained_file_path=train_file,
        test_file_path=test_file
    )

    # Test data validation component
    validator = DataValidation(
        data_ingestion_artifact=ingestion_artifact,
        data_validation_config=ents.validation
    )

    # Execute validation
    validation_artifact = validator.initiate_data_validation()

    # Assertions
    assert validation_artifact.validation_status is True
    assert validation_artifact.validation_report_file_path.exists()
    assert "success" in validation_artifact.message.lower()


def test_validation_fails_with_invalid_data(tmp_path, monkeypatch):
    """Test that validation correctly fails when data doesn't match schema"""
    monkeypatch.setenv("PATHS__ARTIFACT_DIR", str(tmp_path / "artifact"))
    get_settings.cache_clear()

    ents = build_entities(ts="20990101_010203")
    
    # Create ingestion directory and invalid data files
    ingestion_dir = ents.ingestion.data_ingestion_dir / ents.ingestion.data_ingestion_dir
    ingestion_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = ingestion_dir / "train.csv"
    test_file = ingestion_dir / "test.csv"
    
    # Create data with wrong columns (doesn't match schema)
    invalid_data = pd.DataFrame({
        "wrong_column_1": [1, 2, 3, 4],
        "wrong_column_2": ["A", "B", "C", "D"]
    })
    
    invalid_data.to_csv(train_file, index=False)
    invalid_data.to_csv(test_file, index=False)

    ingestion_artifact = DataIngestionArtifact(
        trained_file_path=train_file,
        test_file_path=test_file
    )

    validator = DataValidation(
        data_ingestion_artifact=ingestion_artifact,
        data_validation_config=ents.validation
    )

    validation_artifact = validator.initiate_data_validation()

    # Should fail validation
    assert validation_artifact.validation_status is False
    assert validation_artifact.validation_report_file_path.exists()


def test_validation_in_pipeline_context(tmp_path, monkeypatch, small_df):
    """Test validation as part of the complete pipeline context"""
    monkeypatch.setenv("PATHS__ARTIFACT_DIR", str(tmp_path / "artifact"))
    get_settings.cache_clear()

    ents = build_entities(ts="20990101_010203")
    pipeline = TrainPipeline(entities=ents)

    # Simulate pipeline running data ingestion and validation
    ingestion_dir = ents.ingestion.data_ingestion_dir / ents.ingestion.data_ingestion_dir
    ingestion_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = ingestion_dir / "train.csv"
    test_file = ingestion_dir / "test.csv"
    
    # Use the small_df fixture which should match your schema
    train_data = small_df.iloc[:4]
    test_data = small_df.iloc[4:]
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    ingestion_artifact = DataIngestionArtifact(
        trained_file_path=train_file,
        test_file_path=test_file
    )

    # This is what the pipeline would call internally
    validation_artifact = pipeline.start_data_validation(ingestion_artifact)

    # Verify the validation completed successfully
    assert validation_artifact.validation_status is True
    assert validation_artifact.validation_report_file_path.exists()
    
    # Verify the report directory structure was created
    assert ents.validation.data_validation_dir.exists()