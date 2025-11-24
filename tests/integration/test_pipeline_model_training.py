from pathlib import Path
from unittest.mock import patch

import pytest

from src.pipeline.training_pipeline import TrainPipeline
from src.entities.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)


@pytest.mark.integration
def test_run_pipeline_model_training_success(
    tmp_path,
    mock_data_ingestion_artifact,
    mock_data_transformation_config,
    model_trainer_config,
):
    """
    Integration test for TrainPipeline.run_pipeline:
    - Ensures the full pipeline runs without errors.
    - Verifies that each stage (ingestion, validation, transformation, training)
      is called once.
    """

    # ---- Arrange: create fake artifacts for each stage ----

    # 1) Data Ingestion artifact (reuse fixture with real CSV paths)
    ingestion_artifact: DataIngestionArtifact = mock_data_ingestion_artifact

    # 2) Data Validation artifact
    validation_report_path = tmp_path / "validation_report.json"
    validation_artifact = DataValidationArtifact(
        validation_status=True,
        message="Data validation completed successfully.",
        validation_report_file_path=validation_report_path,
    )

    # 3) Data Transformation artifact
    transformed_dir = tmp_path / "transformation"
    transformed_dir.mkdir(parents=True, exist_ok=True)

    transformation_artifact = DataTransformationArtifact(
        transformed_object_file_path=transformed_dir / "preprocessor.pkl",
        transformed_train_file_path=transformed_dir / "train.npy",
        transformed_test_file_path=transformed_dir / "test.npy",
    )

    # 4) Model Trainer artifact
    metric_artifact = ClassificationMetricArtifact(
        f1_score=0.9,
        precision_score=0.9,
        recall_score=0.9,
    )

    model_trainer_artifact = ModelTrainerArtifact(
        trained_model_file_path=model_trainer_config.trained_model_file_path,
        metric_artifact=metric_artifact,
    )

    # ---- Act: patch components inside the TrainPipeline module ----
    # We patch the classes imported in src.pipeline.training_pipeline:
    #   DataIngestion, DataValidation, DataTransformation, ModelTrainer

    with patch("src.pipeline.training_pipeline.DataIngestion") as MockIngestion, \
         patch("src.pipeline.training_pipeline.DataValidation") as MockValidation, \
         patch("src.pipeline.training_pipeline.DataTransformation") as MockTransformation, \
         patch("src.pipeline.training_pipeline.ModelTrainer") as MockTrainer:

        # Configure mocked instances
        MockIngestion.return_value.initiate_data_ingestion.return_value = ingestion_artifact
        MockValidation.return_value.initiate_data_validation.return_value = validation_artifact
        MockTransformation.return_value.initiate_data_transformation.return_value = transformation_artifact
        MockTrainer.return_value.initiate_model_trainer.return_value = model_trainer_artifact

        # Initialize pipeline
        pipeline = TrainPipeline()

        # Inject configs needed by start_data_transformation and start_model_trainer
        pipeline.data_transformation_config = mock_data_transformation_config
        pipeline.model_trainer_config = model_trainer_config

        # Run full pipeline (the thing we really care about)
        pipeline.run_pipeline()

    # ---- Assert: every stage was executed once ----

    # Class constructors called once
    MockIngestion.assert_called_once()
    MockValidation.assert_called_once()
    MockTransformation.assert_called_once()
    MockTrainer.assert_called_once()

    # Stage methods called once
    MockIngestion.return_value.initiate_data_ingestion.assert_called_once()
    MockValidation.return_value.initiate_data_validation.assert_called_once()
    MockTransformation.return_value.initiate_data_transformation.assert_called_once()
    MockTrainer.return_value.initiate_model_trainer.assert_called_once()
