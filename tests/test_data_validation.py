import pytest
import pandas as pd
from pathlib import Path
import json
from unittest.mock import Mock, patch

from src.components.data_validation import DataValidation
from src.entities.artifact_entity import DataIngestionArtifact, DataValidationArtifact


class TestDataValidation:
    """Test suite for DataValidation class using updated conftest fixtures"""
    
    @pytest.mark.validation
    def test_init_success(self, mock_data_ingestion_artifact, mock_data_validation_config, insurance_schema_config):
        """Test successful initialization of DataValidation"""
        with patch('src.components.data_validation.get_settings') as mock_settings, \
             patch('src.components.data_validation.read_yaml_file') as mock_read_yaml:
            
            mock_settings_instance = Mock()
            mock_settings_instance.paths.schema_file_path = Path("test_schema.yaml")
            mock_settings.return_value = mock_settings_instance
            mock_read_yaml.return_value = insurance_schema_config
            
            dv = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_validation_config=mock_data_validation_config
            )
            
            assert dv.data_ingestion_artifact is not None
            assert dv.data_validation_config is not None
            assert dv._schema_config == insurance_schema_config

    @pytest.mark.validation
    def test_validate_number_of_columns_correct(self, mock_data_ingestion_artifact, mock_data_validation_config, insurance_schema_config):
        """Test column count validation with correct number of columns"""
        with patch('src.components.data_validation.get_settings') as mock_settings, \
             patch('src.components.data_validation.read_yaml_file') as mock_read_yaml:
            
            mock_settings_instance = Mock()
            mock_settings_instance.paths.schema_file_path = Path("test_schema.yaml")
            mock_settings.return_value = mock_settings_instance
            mock_read_yaml.return_value = insurance_schema_config
            
            dv = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_validation_config=mock_data_validation_config
            )
            
            # Use the actual data from mock_data_ingestion_artifact
            train_df = DataValidation.read_data(mock_data_ingestion_artifact.trained_file_path)
            result = dv.validate_number_of_columns(train_df)
            assert result is True

    @pytest.mark.validation
    def test_validate_number_of_columns_incorrect(self, mock_data_ingestion_artifact, mock_data_validation_config, insurance_schema_config):
        """Test column count validation with incorrect number of columns"""
        with patch('src.components.data_validation.get_settings') as mock_settings, \
             patch('src.components.data_validation.read_yaml_file') as mock_read_yaml:
            
            mock_settings_instance = Mock()
            mock_settings_instance.paths.schema_file_path = Path("test_schema.yaml")
            mock_settings.return_value = mock_settings_instance
            mock_read_yaml.return_value = insurance_schema_config
            
            dv = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_validation_config=mock_data_validation_config
            )
            
            # Create dataframe with wrong number of columns
            df_wrong_columns = pd.DataFrame({
                "id": [1, 2, 3],
                "Gender": ["Male", "Female", "Male"]
                # Missing other required columns
            })
            
            result = dv.validate_number_of_columns(df_wrong_columns)
            assert result is False

    @pytest.mark.validation
    def test_is_column_exist_all_present(self, mock_data_ingestion_artifact, mock_data_validation_config, insurance_schema_config):
        """Test column existence check when all columns are present"""
        with patch('src.components.data_validation.get_settings') as mock_settings, \
             patch('src.components.data_validation.read_yaml_file') as mock_read_yaml:
            
            mock_settings_instance = Mock()
            mock_settings_instance.paths.schema_file_path = Path("test_schema.yaml")
            mock_settings.return_value = mock_settings_instance
            mock_read_yaml.return_value = insurance_schema_config
            
            dv = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_validation_config=mock_data_validation_config
            )
            
            # Use the actual data from mock_data_ingestion_artifact
            train_df = DataValidation.read_data(mock_data_ingestion_artifact.trained_file_path)
            result = dv.is_column_exist(train_df)
            assert result is True

    @pytest.mark.validation
    def test_is_column_exist_missing_columns(self, mock_data_ingestion_artifact, mock_data_validation_config, insurance_schema_config):
        """Test column existence check when columns are missing"""
        with patch('src.components.data_validation.get_settings') as mock_settings, \
             patch('src.components.data_validation.read_yaml_file') as mock_read_yaml:
            
            mock_settings_instance = Mock()
            mock_settings_instance.paths.schema_file_path = Path("test_schema.yaml")
            mock_settings.return_value = mock_settings_instance
            mock_read_yaml.return_value = insurance_schema_config
            
            dv = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_validation_config=mock_data_validation_config
            )
            
            # Create dataframe with missing columns
            df_missing_columns = pd.DataFrame({
                "id": [1, 2, 3],
                "Age": [25, 45, 33]
                # Missing Gender, Vehicle_Age, etc.
            })
            
            result = dv.is_column_exist(df_missing_columns)
            assert result is False

    @pytest.mark.validation
    def test_read_data_success(self, mock_data_ingestion_artifact):
        """Test successful data reading from CSV"""
        train_file = mock_data_ingestion_artifact.trained_file_path
        result = DataValidation.read_data(train_file)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "id" in result.columns  # Updated to match actual schema

    @pytest.mark.validation
    def test_read_data_file_not_found(self):
        """Test data reading when file doesn't exist"""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            DataValidation.read_data("nonexistent_file.csv")

    @pytest.mark.validation
    def test_initiate_data_validation_success(self, mock_data_ingestion_artifact, mock_data_validation_config, insurance_schema_config):
        """Test successful data validation process"""
        with patch('src.components.data_validation.get_settings') as mock_settings, \
             patch('src.components.data_validation.read_yaml_file') as mock_read_yaml:
            
            mock_settings_instance = Mock()
            mock_settings_instance.paths.schema_file_path = Path("test_schema.yaml")
            mock_settings.return_value = mock_settings_instance
            mock_read_yaml.return_value = insurance_schema_config
            
            dv = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_validation_config=mock_data_validation_config
            )
            
            result = dv.initiate_data_validation()
            
            assert isinstance(result, DataValidationArtifact)
            assert result.validation_status is True
            assert "successfully" in result.message.lower()
            assert result.validation_report_file_path.exists()

    @pytest.mark.validation
    def test_initiate_data_validation_failure(self, mock_data_ingestion_artifact_invalid, 
                                           mock_data_validation_config, insurance_schema_config):
        """Test data validation failure with invalid data"""
        with patch('src.components.data_validation.get_settings') as mock_settings, \
             patch('src.components.data_validation.read_yaml_file') as mock_read_yaml:
            
            mock_settings_instance = Mock()
            mock_settings_instance.paths.schema_file_path = Path("test_schema.yaml")
            mock_settings.return_value = mock_settings_instance
            mock_read_yaml.return_value = insurance_schema_config
            
            dv = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact_invalid,
                data_validation_config=mock_data_validation_config
            )
            
            result = dv.initiate_data_validation()
            
            assert result.validation_status is False
            assert "columns are missing" in result.message.lower()

    @pytest.mark.validation
    @pytest.mark.parametrize("missing_columns,expected_status", [
        (["Age"], False),
        (["Gender"], False),
        (["Vehicle_Age"], False),
        (["Response"], False),
        ([], True)  # No missing columns
    ])
    def test_column_existence_various_scenarios(self, mock_data_ingestion_artifact, mock_data_validation_config, 
                                              insurance_schema_config, missing_columns, expected_status):
        """Test various column existence scenarios"""
        with patch('src.components.data_validation.get_settings') as mock_settings, \
             patch('src.components.data_validation.read_yaml_file') as mock_read_yaml:
            
            mock_settings_instance = Mock()
            mock_settings_instance.paths.schema_file_path = Path("test_schema.yaml")
            mock_settings.return_value = mock_settings_instance
            mock_read_yaml.return_value = insurance_schema_config
            
            dv = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_validation_config=mock_data_validation_config
            )
            
            # Get the actual valid dataframe
            valid_df = DataValidation.read_data(mock_data_ingestion_artifact.trained_file_path)
            
            # Remove specified columns for testing
            test_df = valid_df.drop(columns=missing_columns, errors='ignore')
            
            result = dv.is_column_exist(test_df)
            assert result == expected_status

    @pytest.mark.validation
    @pytest.mark.slow
    def test_large_dataframe_validation(self, mock_data_ingestion_artifact, mock_data_validation_config, insurance_schema_config):
        """Test validation with large dataframe (marked as slow)"""
        with patch('src.components.data_validation.get_settings') as mock_settings, \
             patch('src.components.data_validation.read_yaml_file') as mock_read_yaml:
            
            mock_settings_instance = Mock()
            mock_settings_instance.paths.schema_file_path = Path("test_schema.yaml")
            mock_settings.return_value = mock_settings_instance
            mock_read_yaml.return_value = insurance_schema_config
            
            dv = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_validation_config=mock_data_validation_config
            )
            
            # Create a larger version of the insurance data
            large_df = pd.DataFrame({
                "id": range(100),
                "Gender": ["Male", "Female"] * 50,
                "Age": [25 + i % 40 for i in range(100)],
                "Driving_License": [1] * 100,
                "Region_Code": [28.0, 8.0, 15.0] * 33 + [28.0],
                "Previously_Insured": [0, 1] * 50,
                "Vehicle_Age": ["1-2 Year", "> 2 Years", "< 1 Year"] * 33 + ["1-2 Year"],
                "Vehicle_Damage": ["Yes", "No"] * 50,
                "Annual_Premium": [2500.0 + i * 10 for i in range(100)],
                "Policy_Sales_Channel": [26.0, 124.0, 152.0] * 33 + [26.0],
                "Vintage": [100 + i for i in range(100)],
                "Response": [0, 1] * 50
            })
            
            # Test column count
            count_result = dv.validate_number_of_columns(large_df)
            assert count_result is True
            
            # Test column existence
            existence_result = dv.is_column_exist(large_df)
            assert existence_result is True

    @pytest.mark.validation
    def test_validation_report_content(self, mock_data_ingestion_artifact, mock_data_validation_config, insurance_schema_config):
        """Test that validation report contains correct information"""
        with patch('src.components.data_validation.get_settings') as mock_settings, \
             patch('src.components.data_validation.read_yaml_file') as mock_read_yaml:
            
            mock_settings_instance = Mock()
            mock_settings_instance.paths.schema_file_path = Path("test_schema.yaml")
            mock_settings.return_value = mock_settings_instance
            mock_read_yaml.return_value = insurance_schema_config
            
            dv = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_validation_config=mock_data_validation_config
            )
            
            result = dv.initiate_data_validation()
            
            # Check report file content
            with open(result.validation_report_file_path, 'r') as f:
                report_content = json.load(f)
            
            assert report_content["validation_status"] == result.validation_status
            assert report_content["message"] == result.message
            assert "validation_status" in report_content
            assert "message" in report_content

    @pytest.mark.validation
    def test_validation_with_empty_dataframe(self, mock_data_validation_config, insurance_schema_config):
        """Test validation with empty dataframe"""
        with patch('src.components.data_validation.get_settings') as mock_settings, \
             patch('src.components.data_validation.read_yaml_file') as mock_read_yaml:
            
            mock_settings_instance = Mock()
            mock_settings_instance.paths.schema_file_path = Path("test_schema.yaml")
            mock_settings.return_value = mock_settings_instance
            mock_read_yaml.return_value = insurance_schema_config
            
            # Create empty ingestion artifact
            empty_artifact = DataIngestionArtifact(
                trained_file_path=Path("empty_train.csv"),
                test_file_path=Path("empty_test.csv")
            )
            
            dv = DataValidation(
                data_ingestion_artifact=empty_artifact,
                data_validation_config=mock_data_validation_config
            )
            
            # Should raise FileNotFoundError when trying to read empty files
            with pytest.raises(FileNotFoundError):
                dv.initiate_data_validation()