from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.entities.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from src.entities.config_entities import DataValidationConfig


class TestDataValidationUnit:
    """Unit tests for the DataValidation component."""

    def test_init_loads_schema_success(
        self,
        mock_data_ingestion_artifact: DataIngestionArtifact,
        mock_data_validation_config: DataValidationConfig,
        insurance_schema_config: dict,
    ):
        """Ensure __init__ loads schema configuration successfully via read_yaml_file."""
        with patch("src.components.data_validation.read_yaml_file") as mock_read_yaml:
            mock_read_yaml.return_value = insurance_schema_config
            from src.components.data_validation import DataValidation

            validator = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_validation_config=mock_data_validation_config,
            )

            assert validator._schema_config == insurance_schema_config
            mock_read_yaml.assert_called_once()

    def test_validate_number_of_columns_success(
        self,
        mock_data_ingestion_artifact: DataIngestionArtifact,
        mock_data_validation_config: DataValidationConfig,
        insurance_schema_config: dict,
        insurance_small_df: pd.DataFrame,
    ):
        """validate_number_of_columns should return True when column count matches schema."""
        with patch("src.components.data_validation.read_yaml_file") as mock_read_yaml:
            mock_read_yaml.return_value = insurance_schema_config
            from src.components.data_validation import DataValidation

            validator = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_validation_config=mock_data_validation_config,
            )

            status = validator.validate_number_of_columns(insurance_small_df)
            assert status is True

    def test_validate_number_of_columns_failure_when_missing_cols(
        self,
        mock_data_ingestion_artifact: DataIngestionArtifact,
        mock_data_validation_config: DataValidationConfig,
        insurance_schema_config: dict,
        insurance_small_df: pd.DataFrame,
    ):
        """validate_number_of_columns should return False when some columns are missing."""
        df_missing = insurance_small_df.drop(columns=["Vehicle_Damage"])

        with patch("src.components.data_validation.read_yaml_file") as mock_read_yaml:
            mock_read_yaml.return_value = insurance_schema_config
            from src.components.data_validation import DataValidation

            validator = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_validation_config=mock_data_validation_config,
            )

            status = validator.validate_number_of_columns(df_missing)
            assert status is False

    def test_is_column_exist_success(
        self,
        mock_data_ingestion_artifact: DataIngestionArtifact,
        mock_data_validation_config: DataValidationConfig,
        insurance_schema_config: dict,
        insurance_small_df: pd.DataFrame,
    ):
        """is_column_exist should return True when all numerical and categorical columns exist."""
        with patch("src.components.data_validation.read_yaml_file") as mock_read_yaml:
            mock_read_yaml.return_value = insurance_schema_config
            from src.components.data_validation import DataValidation

            validator = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_validation_config=mock_data_validation_config,
            )

            status = validator.is_column_exist(insurance_small_df)
            assert status is True

    def test_is_column_exist_missing_columns_returns_false(
        self,
        mock_data_ingestion_artifact: DataIngestionArtifact,
        mock_data_validation_config: DataValidationConfig,
        insurance_schema_config: dict,
        insurance_small_df: pd.DataFrame,
    ):
        """is_column_exist should return False when required numerical or categorical columns are missing."""
        df_missing = insurance_small_df.drop(columns=["Vehicle_Damage", "Age"])

        with patch("src.components.data_validation.read_yaml_file") as mock_read_yaml:
            mock_read_yaml.return_value = insurance_schema_config
            from src.components.data_validation import DataValidation

            validator = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_validation_config=mock_data_validation_config,
            )

            status = validator.is_column_exist(df_missing)
            assert status is False

    def test_read_data_file_not_found_raises(self):
        """read_data should raise FileNotFoundError when the CSV file does not exist."""
        from src.components.data_validation import DataValidation

        with pytest.raises(FileNotFoundError):
            DataValidation.read_data("non_existing_file.csv")


class TestDataValidationIntegration:
    """Integration tests for DataValidation using real temporary files."""

    @pytest.mark.validation
    def test_initiate_data_validation_success_creates_artifact_and_report(
        self,
        mock_data_ingestion_artifact: DataIngestionArtifact,
        mock_data_validation_config: DataValidationConfig,
        insurance_schema_config: dict,
    ):
        """
        With valid data, initiate_data_validation should:
        - return a DataValidationArtifact with validation_status=True
        - create a JSON report at the configured path.
        """
        with patch("src.components.data_validation.read_yaml_file") as mock_read_yaml:
            mock_read_yaml.return_value = insurance_schema_config
            from src.components.data_validation import DataValidation

            validator = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_validation_config=mock_data_validation_config,
            )

            artifact: DataValidationArtifact = validator.initiate_data_validation()

            assert isinstance(artifact, DataValidationArtifact)
            assert artifact.validation_status is True
            assert artifact.message == "Data validation completed successfully."

            report_path = artifact.validation_report_file_path
            assert isinstance(report_path, (str, Path))
            report_path = Path(report_path)
            assert report_path.exists()

            import json
            with report_path.open("r") as f:
                data = json.load(f)
            assert data["validation_status"] is True
            assert data["message"] == "Data validation completed successfully."

    @pytest.mark.validation
    def test_initiate_data_validation_with_invalid_data_sets_status_false(
        self,
        mock_data_ingestion_artifact_invalid: DataIngestionArtifact,
        mock_data_validation_config: DataValidationConfig,
        insurance_schema_config: dict,
    ):
        """
        With data that does not match the schema, initiate_data_validation should:
        - return validation_status=False
        - populate a non-empty error message.
        """
        with patch("src.components.data_validation.read_yaml_file") as mock_read_yaml:
            mock_read_yaml.return_value = insurance_schema_config
            from src.components.data_validation import DataValidation

            validator = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact_invalid,
                data_validation_config=mock_data_validation_config,
            )

            artifact: DataValidationArtifact = validator.initiate_data_validation()

            assert isinstance(artifact, DataValidationArtifact)
            assert artifact.validation_status is False
            assert "Columns are missing" in artifact.message or len(artifact.message) > 0

            report_path = Path(artifact.validation_report_file_path)
            assert report_path.exists()
