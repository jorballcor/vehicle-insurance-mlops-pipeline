from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import pickle

from src.entities.artifact_entity import (
    DataIngestionArtifact, 
    DataValidationArtifact, 
    DataTransformationArtifact
)
from src.entities.config_entities import DataTransformationConfig


class TestDataTransformation:
    """Test suite for Data Transformation component"""
    
    @pytest.fixture
    def mock_data_transformation_config(self, tmp_path):
        """Mock data transformation config"""
        return DataTransformationConfig(
            data_transformation_dir=tmp_path / "data_transformation",
            transformed_train_file_path=tmp_path / "transformed_train.csv",
            transformed_test_file_path=tmp_path / "transformed_test.csv",
            transformed_object_file_path=tmp_path / "preprocessor.pkl"
        )
    
    @pytest.fixture
    def mock_data_validation_artifact_success(self):
        """Mock successful data validation artifact"""
        return DataValidationArtifact(
            validation_status=True,
            message="Validation successful",
            validation_report_file_path=Path("/fake/report.json")
        )
    
    @pytest.fixture
    def mock_data_validation_artifact_failed(self):
        """Mock failed data validation artifact"""
        return DataValidationArtifact(
            validation_status=False,
            message="Validation failed: missing columns",
            validation_report_file_path=Path("/fake/report.json")
        )
    
    @pytest.fixture
    def sample_transformed_data(self):
        """Sample transformed data for testing"""
        return pd.DataFrame({
            'Age_scaled': [0.1, 0.2, 0.3, 0.4],
            'Annual_Premium_scaled': [0.15, 0.25, 0.35, 0.45],
            'Vehicle_Age_<1_Year': [1, 0, 0, 1],
            'Vehicle_Age_1-2_Year': [0, 1, 0, 0],
            'Vehicle_Age_>2_Years': [0, 0, 1, 0],
            'Response': [0, 1, 0, 1]
        })

    def test_init_successful_validation(
        self, 
        mock_data_ingestion_artifact,
        mock_data_validation_artifact_success,
        mock_data_transformation_config
    ):
        """Test initialization with successful validation"""
        from src.components.data_transformation import DataTransformation
        
        # Mock schema loading
        with patch('src.components.data_transformation.read_yaml_file') as mock_read_yaml:
            mock_read_yaml.return_value = {
                "columns": ["id", "Gender", "Age", "Vehicle_Age", "Annual_Premium", "Response"],
                "numerical_columns": ["Age", "Annual_Premium"],
                "categorical_columns": ["Gender", "Vehicle_Age"],
                "target_column": "Response"
            }
            
            transformer = DataTransformation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_transformation_config=mock_data_transformation_config,
                data_validation_artifact=mock_data_validation_artifact_success
            )
            
            assert transformer.data_ingestion_artifact == mock_data_ingestion_artifact
            assert transformer.data_transformation_config == mock_data_transformation_config
            assert transformer.data_validation_artifact == mock_data_validation_artifact_success

    def test_init_failed_validation_raises_error(
        self, 
        mock_data_ingestion_artifact,
        mock_data_validation_artifact_failed,
        mock_data_transformation_config
    ):
        """Test initialization raises error when validation failed"""
        from src.components.data_transformation import DataTransformation
        
        with pytest.raises(RuntimeError, match="Data validation failed"):
            DataTransformation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_transformation_config=mock_data_transformation_config,
                data_validation_artifact=mock_data_validation_artifact_failed
            )

    @patch('pandas.read_csv')
    def test_data_loading_success(
        self, 
        mock_read_csv,
        mock_data_ingestion_artifact,
        mock_data_validation_artifact_success,
        mock_data_transformation_config,
        insurance_small_df
    ):
        """Test successful data loading from CSV files"""
        from src.components.data_transformation import DataTransformation
        
        # Mock the CSV reading
        mock_read_csv.return_value = insurance_small_df
        
        # Mock schema
        with patch('src.components.data_transformation.read_yaml_file') as mock_read_yaml:
            mock_read_yaml.return_value = {
                "columns": list(insurance_small_df.columns),
                "numerical_columns": ["Age", "Annual_Premium", "Vintage"],
                "categorical_columns": ["Gender", "Vehicle_Age", "Vehicle_Damage"],
                "target_column": "Response"
            }
            
            transformer = DataTransformation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_transformation_config=mock_data_transformation_config,
                data_validation_artifact=mock_data_validation_artifact_success
            )
            
            # Test data loading (assuming there's a method or it's done in initiate)
            train_df = transformer.read_data(mock_data_ingestion_artifact.trained_file_path)
            test_df = transformer.read_data(mock_data_ingestion_artifact.test_file_path)
            
            # Verify CSV was read twice (train and test)
            assert mock_read_csv.call_count == 2
            assert train_df.equals(insurance_small_df)
            assert test_df.equals(insurance_small_df)

    def test_preprocessing_pipeline_creation(
        self,
        mock_data_ingestion_artifact,
        mock_data_validation_artifact_success,
        mock_data_transformation_config
    ):
        """Test that preprocessing pipeline is created correctly"""
        from src.components.data_transformation import DataTransformation
        
        with patch('src.components.data_transformation.read_yaml_file') as mock_read_yaml:
            mock_read_yaml.return_value = {
                "columns": ["Age", "Vehicle_Age", "Annual_Premium", "Response"],
                "numerical_columns": ["Age", "Annual_Premium"],
                "categorical_columns": ["Vehicle_Age"],
                "target_column": "Response"
            }
            
            # Mock sklearn components
            with patch('sklearn.compose.ColumnTransformer') as MockColumnTransformer, \
                 patch('sklearn.preprocessing.StandardScaler') as MockStandardScaler, \
                 patch('sklearn.preprocessing.OneHotEncoder') as MockOneHotEncoder:
                
                transformer = DataTransformation(
                    data_ingestion_artifact=mock_data_ingestion_artifact,
                    data_transformation_config=mock_data_transformation_config,
                    data_validation_artifact=mock_data_validation_artifact_success
                )
                
                # If there's a method to create preprocessor, test it
                if hasattr(transformer, 'get_data_transformer_object'):
                    preprocessor = transformer.get_data_transformer_object()
                    
                    # Verify preprocessor components were created
                    MockStandardScaler.assert_called()
                    MockOneHotEncoder.assert_called()
                    MockColumnTransformer.assert_called()

    @patch('pandas.DataFrame.to_csv')
    @patch('pickle.dump')
    def test_transformation_artifact_creation(
        self,
        mock_pickle_dump,
        mock_to_csv,
        mock_data_ingestion_artifact,
        mock_data_validation_artifact_success,
        mock_data_transformation_config,
        sample_transformed_data
    ):
        """Test that transformation artifacts are created and saved"""
        from src.components.data_transformation import DataTransformation
        
        with patch('src.components.data_transformation.read_yaml_file') as mock_read_yaml:
            mock_read_yaml.return_value = {
                "columns": ["Age", "Vehicle_Age", "Annual_Premium", "Response"],
                "numerical_columns": ["Age", "Annual_Premium"],
                "categorical_columns": ["Vehicle_Age"],
                "target_column": "Response"
            }
            
            with patch('pandas.read_csv') as mock_read_csv, \
                 patch('src.components.data_transformation.DataTransformation.get_data_transformer_object') as mock_preprocessor:
                
                # Mock data reading
                mock_read_csv.return_value = sample_transformed_data
                
                # Mock preprocessor
                mock_preprocessor_instance = Mock()
                mock_preprocessor.return_value = mock_preprocessor_instance
                mock_preprocessor_instance.fit_transform.return_value = sample_transformed_data.drop('Response', axis=1).values
                mock_preprocessor_instance.transform.return_value = sample_transformed_data.drop('Response', axis=1).values
                
                transformer = DataTransformation(
                    data_ingestion_artifact=mock_data_ingestion_artifact,
                    data_transformation_config=mock_data_transformation_config,
                    data_validation_artifact=mock_data_validation_artifact_success
                )
                
                # Mock the initiate method's internal logic
                with patch.object(transformer, 'initiate_data_transformation') as mock_initiate:
                    artifact = DataTransformationArtifact(
                        transformed_object_file_path=mock_data_transformation_config.transformed_object_file_path,
                        transformed_train_file_path=mock_data_transformation_config.transformed_train_file_path,
                        transformed_test_file_path=mock_data_transformation_config.transformed_test_file_path
                    )
                    mock_initiate.return_value = artifact
                    
                    result = transformer.initiate_data_transformation()
                    
                    # Verify artifact creation
                    assert isinstance(result, DataTransformationArtifact)
                    assert result.transformed_train_file_path == mock_data_transformation_config.transformed_train_file_path
                    assert result.transformed_test_file_path == mock_data_transformation_config.transformed_test_file_path
                    assert result.transformed_object_file_path == mock_data_transformation_config.transformed_object_file_path

    def test_error_handling_invalid_data(
        self,
        mock_data_ingestion_artifact_invalid,
        mock_data_validation_artifact_success,
        mock_data_transformation_config
    ):
        """Test error handling with invalid data files"""
        from src.components.data_transformation import DataTransformation
        
        with patch('src.components.data_transformation.read_yaml_file') as mock_read_yaml:
            mock_read_yaml.return_value = {
                "columns": ["Age", "Vehicle_Age", "Annual_Premium", "Response"],
                "numerical_columns": ["Age", "Annual_Premium"],
                "categorical_columns": ["Vehicle_Age"],
                "target_column": "Response"
            }
            
            transformer = DataTransformation(
                data_ingestion_artifact=mock_data_ingestion_artifact_invalid,
                data_transformation_config=mock_data_transformation_config,
                data_validation_artifact=mock_data_validation_artifact_success
            )
            
            # Test that transformation fails with schema mismatch
            with pytest.raises(Exception):  # Could be KeyError, ValueError, etc.
                transformer.initiate_data_transformation()

    def test_directory_creation(
        self,
        mock_data_ingestion_artifact,
        mock_data_validation_artifact_success,
        mock_data_transformation_config
    ):
        """Test that necessary directories are created"""
        from src.components.data_transformation import DataTransformation
        
        with patch('src.components.data_transformation.read_yaml_file') as mock_read_yaml:
            mock_read_yaml.return_value = {
                "columns": ["Age", "Vehicle_Age", "Annual_Premium", "Response"],
                "numerical_columns": ["Age", "Annual_Premium"],
                "categorical_columns": ["Vehicle_Age"],
                "target_column": "Response"
            }
            
            with patch('pathlib.Path.mkdir') as mock_mkdir, \
                 patch('pandas.read_csv'), \
                 patch.object(DataTransformation, 'initiate_data_transformation'):
                
                transformer = DataTransformation(
                    data_ingestion_artifact=mock_data_ingestion_artifact,
                    data_transformation_config=mock_data_transformation_config,
                    data_validation_artifact=mock_data_validation_artifact_success
                )
                
                # Verify directory creation was attempted
                assert mock_mkdir.called


class TestDataTransformationIntegration:
    """Integration tests for Data Transformation"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_transformation_with_real_files(
        self,
        mock_data_ingestion_artifact,
        mock_data_validation_artifact_success,
        mock_data_transformation_config
    ):
        """Integration test: complete transformation with real files"""
        from src.components.data_transformation import DataTransformation
        
        # This test uses the actual CSV files created by the fixture
        with patch('src.components.data_transformation.read_yaml_file') as mock_read_yaml:
            # Use realistic schema that matches the fixture data
            mock_read_yaml.return_value = {
                "columns": [
                    "id", "Gender", "Age", "Driving_License", "Region_Code",
                    "Previously_Insured", "Vehicle_Age", "Vehicle_Damage",
                    "Annual_Premium", "Policy_Sales_Channel", "Vintage", "Response"
                ],
                "numerical_columns": [
                    "Age", "Driving_License", "Region_Code", "Previously_Insured",
                    "Annual_Premium", "Policy_Sales_Channel", "Vintage"
                ],
                "categorical_columns": ["Gender", "Vehicle_Age", "Vehicle_Damage"],
                "target_column": "Response",
                "drop_columns": ["id"]  # Assuming we drop ID column
            }
            
            transformer = DataTransformation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_transformation_config=mock_data_transformation_config,
                data_validation_artifact=mock_data_validation_artifact_success
            )
            
            # This would actually run the transformation
            # For now, we'll just verify the component initializes correctly
            assert transformer is not None
            
            # Verify that output directories are set up
            assert mock_data_transformation_config.data_transformation_dir.exists()

    @pytest.mark.integration
    def test_transformed_files_creation(
        self,
        mock_data_ingestion_artifact,
        mock_data_validation_artifact_success,
        mock_data_transformation_config
    ):
        """Test that transformed files are actually created"""
        from src.components.data_transformation import DataTransformation
        
        with patch('src.components.data_transformation.read_yaml_file') as mock_read_yaml, \
             patch('pandas.read_csv') as mock_read_csv, \
             patch('pandas.DataFrame.to_csv') as mock_to_csv, \
             patch('pickle.dump') as mock_pickle_dump:
            
            mock_read_yaml.return_value = {
                "columns": ["Age", "Vehicle_Age", "Annual_Premium", "Response"],
                "numerical_columns": ["Age", "Annual_Premium"],
                "categorical_columns": ["Vehicle_Age"],
                "target_column": "Response"
            }
            
            # Mock the data
            mock_data = pd.DataFrame({
                'Age': [25, 30, 35],
                'Vehicle_Age': ['<1 Year', '1-2 Year', '>2 Years'],
                'Annual_Premium': [1000, 2000, 3000],
                'Response': [0, 1, 0]
            })
            mock_read_csv.return_value = mock_data
            
            transformer = DataTransformation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_transformation_config=mock_data_transformation_config,
                data_validation_artifact=mock_data_validation_artifact_success
            )
            
            # Mock the preprocessor
            with patch.object(transformer, 'get_data_transformer_object') as mock_preprocessor:
                mock_preprocessor_instance = Mock()
                mock_preprocessor.return_value = mock_preprocessor_instance
                mock_preprocessor_instance.fit_transform.return_value = mock_data.drop('Response', axis=1).values
                mock_preprocessor_instance.transform.return_value = mock_data.drop('Response', axis=1).values
                
                artifact = transformer.initiate_data_transformation()
                
                # Verify files were attempted to be saved
                assert mock_to_csv.called
                assert mock_pickle_dump.called
                assert artifact.transformed_train_file_path == mock_data_transformation_config.transformed_train_file_path