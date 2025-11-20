# tests/integration/test_training_pipeline_validaton.py
from src.config.settings import get_settings
from src.pipeline.training_pipeline import TrainPipeline
from src.entities.config_entities import build_entities
from src.components.data_validation import DataValidation
from src.entities.artifact_entity import DataIngestionArtifact
from tests.conftest import FakeRepo, small_df, insurance_small_df
import pandas as pd
from pathlib import Path
from unittest.mock import patch


def test_pipeline_runs_validation_after_ingestion(tmp_path, monkeypatch, insurance_small_df):
    """Test that validation runs correctly after data ingestion in the pipeline"""
    # Setup environment
    monkeypatch.setenv("PATHS__ARTIFACT_DIR", str(tmp_path / "artifact"))
    get_settings.cache_clear()

    # Build pipeline entities
    ents = build_entities(ts="20990101_010203")
    pipeline = TrainPipeline(entities=ents)

    # ✅ CORREGIDO: Usar el path correcto para ingested data
    ingestion_dir = ents.ingestion.data_ingestion_dir / "ingested"
    ingestion_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = ingestion_dir / "train.csv"
    test_file = ingestion_dir / "test.csv"
    
    # Split insurance_small_df into train/test (que coincide con el schema real)
    train_data = insurance_small_df.iloc[:4]  # First 4 rows for training
    test_data = insurance_small_df.iloc[4:]   # Last 2 rows for testing
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    # Create ingestion artifact
    ingestion_artifact = DataIngestionArtifact(
        trained_file_path=train_file,
        test_file_path=test_file
    )

    # Test data validation component - SIN MOCK (usa el schema real)
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
    
    # ✅ CORREGIDO: Usar el path correcto
    ingestion_dir = ents.ingestion.data_ingestion_dir / "ingested"
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


def test_validation_in_pipeline_context(tmp_path, monkeypatch, insurance_small_df):
    """Test validation as part of the complete pipeline context"""
    monkeypatch.setenv("PATHS__ARTIFACT_DIR", str(tmp_path / "artifact"))
    get_settings.cache_clear()

    ents = build_entities(ts="20990101_010203")
    pipeline = TrainPipeline(entities=ents)

    # ✅ CORREGIDO: Usar el path correcto
    ingestion_dir = ents.ingestion.data_ingestion_dir / "ingested"
    ingestion_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = ingestion_dir / "train.csv"
    test_file = ingestion_dir / "test.csv"
    
    # Use the insurance_small_df fixture que coincide con el schema real
    train_data = insurance_small_df.iloc[:4]
    test_data = insurance_small_df.iloc[4:]
    
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


# ✅ NUEVO: Test para verificar compatibilidad con el schema antiguo
def test_validation_with_old_small_df_fails(tmp_path, monkeypatch, small_df):
    """Test that the original small_df fails validation (backwards compatibility check)"""
    monkeypatch.setenv("PATHS__ARTIFACT_DIR", str(tmp_path / "artifact"))
    get_settings.cache_clear()

    ents = build_entities(ts="20990101_010203")
    
    ingestion_dir = ents.ingestion.data_ingestion_dir / "ingested"
    ingestion_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = ingestion_dir / "train.csv"
    test_file = ingestion_dir / "test.csv"
    
    # Use the original small_df (should fail)
    train_data = small_df.iloc[:4]
    test_data = small_df.iloc[4:]
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    ingestion_artifact = DataIngestionArtifact(
        trained_file_path=train_file,
        test_file_path=test_file
    )

    validator = DataValidation(
        data_ingestion_artifact=ingestion_artifact,
        data_validation_config=ents.validation
    )

    validation_artifact = validator.initiate_data_validation()

    # Should fail with original small_df
    assert validation_artifact.validation_status is False