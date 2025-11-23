import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.pipeline.training_pipeline import TrainPipeline
from src.entities.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)
from src.entities.config_entities import DataTransformationConfig


class TestTrainingPipelineTransformation:
    """Unit tests for the transformation stage inside the Training Pipeline."""

    @pytest.fixture
    def mock_ingestion_artifact(self, tmp_path):
        """Return a mocked DataIngestionArtifact with sample file paths."""
        return DataIngestionArtifact(
            trained_file_path=tmp_path / "train.csv",
            test_file_path=tmp_path / "test.csv",
        )

    @pytest.fixture
    def mock_validation_artifact(self, tmp_path):
        """Return a successful DataValidationArtifact."""
        return DataValidationArtifact(
            validation_status=True,
            message="Validation OK",
            validation_report_file_path=tmp_path / "validation_report.json",
        )

    @pytest.fixture
    def mock_transformation_config(self, tmp_path):
        """Return a DataTransformationConfig with dummy file paths."""
        t_dir = tmp_path / "data_transformation"
        t_dir.mkdir(exist_ok=True)

        return DataTransformationConfig(
            data_transformation_dir=t_dir,
            transformed_train_file_path=t_dir / "train_transformed.npy",
            transformed_test_file_path=t_dir / "test_transformed.npy",
            transformed_object_file_path=t_dir / "preprocessor.pkl",
        )

    @pytest.fixture
    def mock_pipeline(self, mock_transformation_config):
        """Return a TrainingPipeline instance with mocked transformation config."""
        pipeline = TrainPipeline()
        pipeline.data_transformation_config = mock_transformation_config
        return pipeline

    @pytest.mark.unit
    def test_start_data_transformation_returns_artifact(
        self,
        mock_pipeline,
        mock_ingestion_artifact,
        mock_validation_artifact,
    ):
        """
        Ensure TrainPipeline.start_data_transformation():
        - Instantiates DataTransformation correctly
        - Calls initiate_data_transformation()
        - Returns a DataTransformationArtifact
        """

        # Fake artifact returned by DataTransformation
        fake_artifact = DataTransformationArtifact(
            transformed_object_file_path=Path("x/preprocessor.pkl"),
            transformed_train_file_path=Path("x/train.npy"),
            transformed_test_file_path=Path("x/test.npy"),
        )

        # Patch DataTransformation to avoid running the real component
        with patch(
            "src.pipeline.training_pipeline.DataTransformation"
        ) as mock_transformation_class:

            mock_instance = MagicMock()
            mock_instance.initiate_data_transformation.return_value = fake_artifact

            mock_transformation_class.return_value = mock_instance

            # Execute pipeline step
            output_artifact = mock_pipeline.start_data_transformation(
                data_ingestion_artifact=mock_ingestion_artifact,
                data_validation_artifact=mock_validation_artifact,
            )

        # Assertions
        mock_transformation_class.assert_called_once()
        mock_instance.initiate_data_transformation.assert_called_once()

        assert isinstance(output_artifact, DataTransformationArtifact)
        assert output_artifact.transformed_object_file_path == fake_artifact.transformed_object_file_path
        assert output_artifact.transformed_train_file_path == fake_artifact.transformed_train_file_path
        assert output_artifact.transformed_test_file_path == fake_artifact.transformed_test_file_path

    @pytest.mark.unit
    def test_start_data_transformation_raises_if_validation_failed(
        self,
        mock_pipeline,
        mock_ingestion_artifact,
    ):
        """Ensure the pipeline raises an error when validation_status=False."""

        failed_artifact = DataValidationArtifact(
            validation_status=False,
            message="Schema mismatch",
            validation_report_file_path=Path("dummy/file.json"),
        )

        with pytest.raises(Exception):
            mock_pipeline.start_data_transformation(
                data_ingestion_artifact=mock_ingestion_artifact,
                data_validation_artifact=failed_artifact,
            )
