# test_data_validation.py
import pytest
import pandas as pd
from pathlib import Path
import json
from unittest.mock import Mock, patch

from src.components.data_validation import DataValidation
from src.entities.artifact_entity import DataValidationArtifact


class TestDataValidation:
    """Test suite for DataValidation class using conftest fixtures"""
    
    @pytest.mark.validation
    def test_init_success(self, data_validation_setup, validation_schema_config):
        """Test successful initialization of DataValidation"""
        dv = data_validation_setup
        assert dv.data_ingestion_artifact is not None
        assert dv.data_validation_config is not None
        assert dv._schema_config == validation_schema_config

    @pytest.mark.validation
    def test_validate_number_of_columns_correct(self, data_validation_setup, valid_dataframe):
        """Test column count validation with correct number of columns"""
        dv = data_validation_setup
        result = dv.validate_number_of_columns(valid_dataframe)
        assert result is True

    @pytest.mark.validation
    def test_validate_number_of_columns_incorrect(self, data_validation_setup, invalid_dataframe_wrong_count):
        """Test column count validation with incorrect number of columns"""
        dv = data_validation_setup
        result = dv.validate_number_of_columns(invalid_dataframe_wrong_count)
        assert result is False

    @pytest.mark.validation
    def test_is_column_exist_all_present(self, data_validation_setup, valid_dataframe):
        """Test column existence check when all columns are present"""
        dv = data_validation_setup
        result = dv.is_column_exist(valid_dataframe)
        assert result is True

    @pytest.mark.validation
    def test_is_column_exist_missing_columns(self, data_validation_setup, invalid_dataframe_missing_columns):
        """Test column existence check when columns are missing"""
        dv = data_validation_setup
        result = dv.is_column_exist(invalid_dataframe_missing_columns)
        assert result is False

    @pytest.mark.validation
    def test_read_data_success(self, mock_data_ingestion_artifact):
        """Test successful data reading from CSV"""
        train_file = mock_data_ingestion_artifact.trained_file_path
        result = DataValidation.read_data(train_file)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "col1" in result.columns

    @pytest.mark.validation
    def test_read_data_file_not_found(self):
        """Test data reading when file doesn't exist"""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            DataValidation.read_data("nonexistent_file.csv")

    @pytest.mark.validation
    def test_initiate_data_validation_success(self, data_validation_setup):
        """Test successful data validation process"""
        dv = data_validation_setup
        result = dv.initiate_data_validation()
        
        assert isinstance(result, DataValidationArtifact)
        assert result.validation_status is True
        assert "successfully" in result.message.lower()
        assert result.validation_report_file_path.exists()

    @pytest.mark.validation
    def test_initiate_data_validation_failure(self, mock_data_ingestion_artifact_invalid, 
                                           mock_data_validation_config, validation_schema_config):
        """Test data validation failure with invalid data"""
        with patch('src.components.data_validation.get_settings') as mock_settings, \
             patch('src.components.data_validation.read_yaml_file') as mock_read_yaml:
            
            mock_settings_instance = Mock()
            mock_settings_instance.paths.schema_file_path = Path("test_schema.yaml")
            mock_settings.return_value = mock_settings_instance
            mock_read_yaml.return_value = validation_schema_config
            
            dv = DataValidation(
                data_ingestion_artifact=mock_data_ingestion_artifact_invalid,
                data_validation_config=mock_data_validation_config
            )
            
            result = dv.initiate_data_validation()
            
            assert result.validation_status is False
            assert "columns are missing" in result.message.lower()

    @pytest.mark.validation
    @pytest.mark.parametrize("missing_columns,expected_status", [
        (["col1"], False),
        (["col2"], False),
        (["col3"], False),
        (["Response"], False),
        ([], True)  # No missing columns
    ])
    def test_column_existence_various_scenarios(self, data_validation_setup, valid_dataframe, 
                                              missing_columns, expected_status):
        """Test various column existence scenarios"""
        dv = data_validation_setup
        
        # Remove specified columns for testing
        test_df = valid_dataframe.drop(columns=missing_columns, errors='ignore')
        
        result = dv.is_column_exist(test_df)
        assert result == expected_status

    @pytest.mark.validation
    @pytest.mark.slow
    def test_large_dataframe_validation(self, data_validation_setup, large_dataframe):
        """Test validation with large dataframe (marked as slow)"""
        dv = data_validation_setup
        
        # Test column count
        count_result = dv.validate_number_of_columns(large_dataframe)
        assert count_result is True
        
        # Test column existence
        existence_result = dv.is_column_exist(large_dataframe)
        assert existence_result is True

    @pytest.mark.validation
    def test_validation_report_content(self, data_validation_setup):
        """Test that validation report contains correct information"""
        dv = data_validation_setup
        result = dv.initiate_data_validation()
        
        # Check report file content
        with open(result.validation_report_file_path, 'r') as f:
            report_content = json.load(f)
        
        assert report_content["validation_status"] == result.validation_status
        assert report_content["message"] == result.message
        assert "validation_status" in report_content
        assert "message" in report_content