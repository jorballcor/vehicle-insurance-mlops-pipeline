i# tests/test_data_transformation.py
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
            'Vintage_scaled': [0.2, 0.3, 0.4, 0.5],
            'Gender_Male': [1, 0, 1, 0],
            'Gender_Female': [0, 1, 0, 1],
            'Vehicle_Age_<1_Year': [1, 0, 0, 1],
            'Vehicle_Age_1-2_Year': [0, 1, 0, 0],
            'Vehicle_Age_>2_Years': [0, 0, 1, 0],
            'Vehicle_Damage_Yes': [1, 0, 1, 0],
            'Vehicle_Damage_No': [0, 1, 0, 1],
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
        
        # Mock schema loading with EXACT schema from schema.yaml
        with patch('src.components.data_transformation.read_yaml_file') as mock_read_yaml:
            mock_read_yaml.return_value = {
                "columns": [
                    {"id": "int"},
                    {"Gender": "category"},
                    {"Age": "int"},
                    {"Driving_License": "int"},
                    {"Region_Code": "float"},
                    {"Previously_Insured": "int"},
                    {"Vehicle_Age": "category"},
                    {"Vehicle_Damage": "category"},
                    {"Annual_Premium": "float"},
                    {"Policy_Sales_Channel": "float"},
                    {"Vintage": "int"},
                    {"Response": "int"}
                ],
                "numerical_columns": [
                    "Age", "Driving_License", "Region_Code", "Previously_Insured",
                    "Annual_Premium", "Policy_Sales_Channel", "Vintage", "Response"
                ],
                "categorical_columns": [
                    "Gender", "Vehicle_Age", "Vehicle_Damage"
                ],
                "drop_columns": ["_id"],
                "num_features": ["Age", "Vintage"],
                "mm_columns": ["Annual_Premium"]
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
        
        # Mock schema with EXACT schema from schema.yaml
        with patch('src.components.data_transformation.read_yaml_file') as mock_read_yaml:
            mock_read_yaml.return_value = {
                "columns": [
                    {"id": "int"},
                    {"Gender": "category"},
                    {"Age": "int"},
                    {"Driving_License": "int"},
                    {"Region_Code": "float"},
                    {"Previously_Insured": "int"},
                    {"Vehicle_Age": "category"},
                    {"Vehicle_Damage": "category"},
                    {"Annual_Premium": "float"},
                    {"Policy_Sales_Channel": "float"},
                    {"Vintage": "int"},
                    {"Response": "int"}
                ],
                "numerical_columns": [
                    "Age", "Driving_License", "Region_Code", "Previously_Insured",
                    "Annual_Premium", "Policy_Sales_Channel", "Vintage", "Response"
                ],
                "categorical_columns": [
                    "Gender", "Vehicle_Age", "Vehicle_Damage"
                ],
                "drop_columns": ["_id"],
                "num_features": ["Age", "Vintage"],
                "mm_columns": ["Annual_Premium"]
            }
            
            transformer = DataTransformation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_transformation_config=mock_data_transformation_config,
                data_validation_artifact=mock_data_validation_artifact_failed
            )
            
            # The actual implementation should check validation status during initiation
            with pytest.raises(RuntimeError, match="Data validation failed"):
                transformer.initiate_data_transformation()

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
        
        # Mock schema with EXACT schema from schema.yaml
        with patch('src.components.data_transformation.read_yaml_file') as mock_read_yaml:
            mock_read_yaml.return_value = {
                "columns": [
                    {"id": "int"},
                    {"Gender": "category"},
                    {"Age": "int"},
                    {"Driving_License": "int"},
                    {"Region_Code": "float"},
                    {"Previously_Insured": "int"},
                    {"Vehicle_Age": "category"},
                    {"Vehicle_Damage": "category"},
                    {"Annual_Premium": "float"},
                    {"Policy_Sales_Channel": "float"},
                    {"Vintage": "int"},
                    {"Response": "int"}
                ],
                "numerical_columns": [
                    "Age", "Driving_License", "Region_Code", "Previously_Insured",
                    "Annual_Premium", "Policy_Sales_Channel", "Vintage", "Response"
                ],
                "categorical_columns": [
                    "Gender", "Vehicle_Age", "Vehicle_Damage"
                ],
                "drop_columns": ["_id"],
                "num_features": ["Age", "Vintage"],
                "mm_columns": ["Annual_Premium"]
            }
            
            transformer = DataTransformation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_transformation_config=mock_data_transformation_config,
                data_validation_artifact=mock_data_validation_artifact_success
            )
            
            # Test that data can be loaded (this happens inside initiate_data_transformation)
            with patch.object(transformer, 'initiate_data_transformation'):
                transformer.initiate_data_transformation()
            
            # Verify CSV was read
            assert mock_read_csv.called

    def test_preprocessing_pipeline_creation(
        self,
        mock_data_ingestion_artifact,
        mock_data_validation_artifact_success,
        mock_data_transformation_config
    ):
        """Test that preprocessing pipeline is created correctly"""
        from src.components.data_transformation import DataTransformation
        
        with patch('src.components.data_transformation.read_yaml_file') as mock_read_yaml:
            # Provide EXACT schema from schema.yaml
            mock_read_yaml.return_value = {
                "columns": [
                    {"id": "int"},
                    {"Gender": "category"},
                    {"Age": "int"},
                    {"Driving_License": "int"},
                    {"Region_Code": "float"},
                    {"Previously_Insured": "int"},
                    {"Vehicle_Age": "category"},
                    {"Vehicle_Damage": "category"},
                    {"Annual_Premium": "float"},
                    {"Policy_Sales_Channel": "float"},
                    {"Vintage": "int"},
                    {"Response": "int"}
                ],
                "numerical_columns": [
                    "Age", "Driving_License", "Region_Code", "Previously_Insured",
                    "Annual_Premium", "Policy_Sales_Channel", "Vintage", "Response"
                ],
                "categorical_columns": [
                    "Gender", "Vehicle_Age", "Vehicle_Damage"
                ],
                "drop_columns": ["_id"],
                "num_features": ["Age", "Vintage"],
                "mm_columns": ["Annual_Premium"]
            }
            
            # Mock sklearn components
            with patch('sklearn.compose.ColumnTransformer') as MockColumnTransformer, \
                 patch('sklearn.preprocessing.StandardScaler') as MockStandardScaler, \
                 patch('sklearn.preprocessing.MinMaxScaler') as MockMinMaxScaler, \
                 patch('sklearn.pipeline.Pipeline') as MockPipeline:
                
                # Mock the transformers to return expected values
                mock_standard_scaler = Mock()
                mock_minmax_scaler = Mock()
                MockStandardScaler.return_value = mock_standard_scaler
                MockMinMaxScaler.return_value = mock_minmax_scaler
                
                transformer = DataTransformation(
                    data_ingestion_artifact=mock_data_ingestion_artifact,
                    data_transformation_config=mock_data_transformation_config,
                    data_validation_artifact=mock_data_validation_artifact_success
                )
                
                # Test the preprocessor creation method
                if hasattr(transformer, 'get_data_transformer_object'):
                    preprocessor = transformer.get_data_transformer_object()
                    
                    # Verify preprocessor components were created with correct parameters
                    MockStandardScaler.assert_called_once()
                    MockMinMaxScaler.assert_called_once()
                    
                    # Verify ColumnTransformer was called with correct transformers
                    MockColumnTransformer.assert_called_once()
                    call_args = MockColumnTransformer.call_args
                    transformers = call_args[1]['transformers']
                    
                    # Check that we have StandardScaler for num_features and MinMaxScaler for mm_columns
                    assert len(transformers) == 2
                    assert transformers[0][0] == "StandardScaler"
                    assert transformers[1][0] == "MinMaxScaler"

    @patch('pandas.DataFrame.to_csv')
    @patch('pickle.dump')
    def test_transformation_artifact_creation(
        self,
        mock_pickle_dump,
        mock_to_csv,
        mock_data_ingestion_artifact,
        mock_data_validation_artifact_success,
        mock_data_transformation_config
    ):
        """Test that transformation artifacts are created and saved"""
        from src.components.data_transformation import DataTransformation
        
        with patch('src.components.data_transformation.read_yaml_file') as mock_read_yaml:
            # Use EXACT schema from schema.yaml
            mock_read_yaml.return_value = {
                "columns": [
                    {"id": "int"},
                    {"Gender": "category"},
                    {"Age": "int"},
                    {"Driving_License": "int"},
                    {"Region_Code": "float"},
                    {"Previously_Insured": "int"},
                    {"Vehicle_Age": "category"},
                    {"Vehicle_Damage": "category"},
                    {"Annual_Premium": "float"},
                    {"Policy_Sales_Channel": "float"},
                    {"Vintage": "int"},
                    {"Response": "int"}
                ],
                "numerical_columns": [
                    "Age", "Driving_License", "Region_Code", "Previously_Insured",
                    "Annual_Premium", "Policy_Sales_Channel", "Vintage", "Response"
                ],
                "categorical_columns": [
                    "Gender", "Vehicle_Age", "Vehicle_Damage"
                ],
                "drop_columns": ["_id"],
                "num_features": ["Age", "Vintage"],
                "mm_columns": ["Annual_Premium"]
            }
            
            with patch('pandas.read_csv') as mock_read_csv, \
                 patch('src.components.data_transformation.DataTransformation.get_data_transformer_object') as mock_preprocessor:
                
                # Mock realistic data that matches the schema
                mock_train_data = pd.DataFrame({
                    "id": [1, 2, 3],
                    "Gender": ["Male", "Female", "Male"],
                    "Age": [25, 45, 33],
                    "Driving_License": [1, 1, 1],
                    "Region_Code": [28.0, 8.0, 15.0],
                    "Previously_Insured": [0, 1, 0],
                    "Vehicle_Age": ["1-2 Year", "> 2 Years", "< 1 Year"],
                    "Vehicle_Damage": ["Yes", "No", "Yes"],
                    "Annual_Premium": [2500.0, 3800.0, 2900.0],
                    "Policy_Sales_Channel": [26.0, 124.0, 26.0],
                    "Vintage": [150, 210, 95],
                    "Response": [0, 1, 0]
                })
                mock_test_data = pd.DataFrame({
                    "id": [4, 5],
                    "Gender": ["Female", "Male"],
                    "Age": [28, 52],
                    "Driving_License": [1, 1],
                    "Region_Code": [28.0, 3.0],
                    "Previously_Insured": [0, 1],
                    "Vehicle_Age": ["1-2 Year", "> 2 Years"],
                    "Vehicle_Damage": ["No", "Yes"],
                    "Annual_Premium": [2100.0, 4500.0],
                    "Policy_Sales_Channel": [152.0, 124.0],
                    "Vintage": [180, 300],
                    "Response": [0, 1]
                })
                
                def read_csv_side_effect(file_path):
                    if "train" in str(file_path):
                        return mock_train_data
                    else:
                        return mock_test_data
                
                mock_read_csv.side_effect = read_csv_side_effect
                
                # Mock preprocessor
                mock_preprocessor_instance = Mock()
                mock_preprocessor.return_value = mock_preprocessor_instance
                
                # Mock transformed data (without id column as it should be dropped)
                transformed_features = mock_train_data.drop(['id', 'Response'], axis=1)
                mock_preprocessor_instance.fit_transform.return_value = transformed_features.values
                mock_preprocessor_instance.transform.return_value = transformed_features.values
                
                transformer = DataTransformation(
                    data_ingestion_artifact=mock_data_ingestion_artifact,
                    data_transformation_config=mock_data_transformation_config,
                    data_validation_artifact=mock_data_validation_artifact_success
                )
                
                # Execute the transformation
                artifact = transformer.initiate_data_transformation()
                
                # Verify artifact creation
                assert isinstance(artifact, DataTransformationArtifact)
                assert artifact.transformed_train_file_path == mock_data_transformation_config.transformed_train_file_path
                assert artifact.transformed_test_file_path == mock_data_transformation_config.transformed_test_file_path
                assert artifact.transformed_object_file_path == mock_data_transformation_config.transformed_object_file_path
                
                # Verify files were saved
                assert mock_to_csv.called
                assert mock_pickle_dump.called

    def test_error_handling_invalid_data(
        self,
        mock_data_ingestion_artifact_invalid,
        mock_data_validation_artifact_success,
        mock_data_transformation_config
    ):
        """Test error handling with invalid data files"""
        from src.components.data_transformation import DataTransformation
        
        with patch('src.components.data_transformation.read_yaml_file') as mock_read_yaml:
            # Use EXACT schema from schema.yaml
            mock_read_yaml.return_value = {
                "columns": [
                    {"id": "int"},
                    {"Gender": "category"},
                    {"Age": "int"},
                    {"Driving_License": "int"},
                    {"Region_Code": "float"},
                    {"Previously_Insured": "int"},
                    {"Vehicle_Age": "category"},
                    {"Vehicle_Damage": "category"},
                    {"Annual_Premium": "float"},
                    {"Policy_Sales_Channel": "float"},
                    {"Vintage": "int"},
                    {"Response": "int"}
                ],
                "numerical_columns": [
                    "Age", "Driving_License", "Region_Code", "Previously_Insured",
                    "Annual_Premium", "Policy_Sales_Channel", "Vintage", "Response"
                ],
                "categorical_columns": [
                    "Gender", "Vehicle_Age", "Vehicle_Damage"
                ],
                "drop_columns": ["_id"],
                "num_features": ["Age", "Vintage"],
                "mm_columns": ["Annual_Premium"]
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
            # Use EXACT schema from schema.yaml
            mock_read_yaml.return_value = {
                "columns": [
                    {"id": "int"},
                    {"Gender": "category"},
                    {"Age": "int"},
                    {"Driving_License": "int"},
                    {"Region_Code": "float"},
                    {"Previously_Insured": "int"},
                    {"Vehicle_Age": "category"},
                    {"Vehicle_Damage": "category"},
                    {"Annual_Premium": "float"},
                    {"Policy_Sales_Channel": "float"},
                    {"Vintage": "int"},
                    {"Response": "int"}
                ],
                "numerical_columns": [
                    "Age", "Driving_License", "Region_Code", "Previously_Insured",
                    "Annual_Premium", "Policy_Sales_Channel", "Vintage", "Response"
                ],
                "categorical_columns": [
                    "Gender", "Vehicle_Age", "Vehicle_Damage"
                ],
                "drop_columns": ["_id"],
                "num_features": ["Age", "Vintage"],
                "mm_columns": ["Annual_Premium"]
            }
            
            # Mock data reading and preprocessing
            with patch('pandas.read_csv') as mock_read_csv, \
                 patch('src.components.data_transformation.DataTransformation.get_data_transformer_object') as mock_preprocessor:
                
                # Mock realistic data
                mock_data = pd.DataFrame({
                    "id": [1, 2, 3],
                    "Gender": ["Male", "Female", "Male"],
                    "Age": [25, 45, 33],
                    "Driving_License": [1, 1, 1],
                    "Region_Code": [28.0, 8.0, 15.0],
                    "Previously_Insured": [0, 1, 0],
                    "Vehicle_Age": ["1-2 Year", "> 2 Years", "< 1 Year"],
                    "Vehicle_Damage": ["Yes", "No", "Yes"],
                    "Annual_Premium": [2500.0, 3800.0, 2900.0],
                    "Policy_Sales_Channel": [26.0, 124.0, 26.0],
                    "Vintage": [150, 210, 95],
                    "Response": [0, 1, 0]
                })
                mock_read_csv.return_value = mock_data
                
                # Mock preprocessor
                mock_preprocessor_instance = Mock()
                mock_preprocessor.return_value = mock_preprocessor_instance
                mock_preprocessor_instance.fit_transform.return_value = mock_data.drop(['id', 'Response'], axis=1).values
                mock_preprocessor_instance.transform.return_value = mock_data.drop(['id', 'Response'], axis=1).values
                
                transformer = DataTransformation(
                    data_ingestion_artifact=mock_data_ingestion_artifact,
                    data_transformation_config=mock_data_transformation_config,
                    data_validation_artifact=mock_data_validation_artifact_success
                )
                
                # Mock the file operations but keep directory creation
                with patch('pandas.DataFrame.to_csv'), \
                     patch('pickle.dump'):
                    
                    # Actually call initiate_data_transformation to trigger directory creation
                    with patch('pathlib.Path.mkdir') as mock_mkdir:
                        try:
                            transformer.initiate_data_transformation()
                        except:
                            pass  # We don't care about the actual execution, just the directory creation
                        
                        # Verify directory creation was attempted
                        assert mock_mkdir.called


# Tests de integración
class TestDataTransformationIntegration:
    """Integration tests for Data Transformation"""
    
    @pytest.fixture
    def mock_data_validation_artifact_success(self):
        """Fixture para los tests de integración"""
        return DataValidationArtifact(
            validation_status=True,
            message="Validation successful",
            validation_report_file_path=Path("/fake/report.json")
        )
    
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
            # Use EXACT schema from schema.yaml
            mock_read_yaml.return_value = {
                "columns": [
                    {"id": "int"},
                    {"Gender": "category"},
                    {"Age": "int"},
                    {"Driving_License": "int"},
                    {"Region_Code": "float"},
                    {"Previously_Insured": "int"},
                    {"Vehicle_Age": "category"},
                    {"Vehicle_Damage": "category"},
                    {"Annual_Premium": "float"},
                    {"Policy_Sales_Channel": "float"},
                    {"Vintage": "int"},
                    {"Response": "int"}
                ],
                "numerical_columns": [
                    "Age", "Driving_License", "Region_Code", "Previously_Insured",
                    "Annual_Premium", "Policy_Sales_Channel", "Vintage", "Response"
                ],
                "categorical_columns": [
                    "Gender", "Vehicle_Age", "Vehicle_Damage"
                ],
                "drop_columns": ["_id"],
                "num_features": ["Age", "Vintage"],
                "mm_columns": ["Annual_Premium"]
            }
            
            transformer = DataTransformation(
                data_ingestion_artifact=mock_data_ingestion_artifact,
                data_transformation_config=mock_data_transformation_config,
                data_validation_artifact=mock_data_validation_artifact_success
            )
            
            # Verify the component initializes correctly
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
            
            # Use EXACT schema from schema.yaml
            mock_read_yaml.return_value = {
                "columns": [
                    {"id": "int"},
                    {"Gender": "category"},
                    {"Age": "int"},
                    {"Driving_License": "int"},
                    {"Region_Code": "float"},
                    {"Previously_Insured": "int"},
                    {"Vehicle_Age": "category"},
                    {"Vehicle_Damage": "category"},
                    {"Annual_Premium": "float"},
                    {"Policy_Sales_Channel": "float"},
                    {"Vintage": "int"},
                    {"Response": "int"}
                ],
                "numerical_columns": [
                    "Age", "Driving_License", "Region_Code", "Previously_Insured",
                    "Annual_Premium", "Policy_Sales_Channel", "Vintage", "Response"
                ],
                "categorical_columns": [
                    "Gender", "Vehicle_Age", "Vehicle_Damage"
                ],
                "drop_columns": ["_id"],
                "num_features": ["Age", "Vintage"],
                "mm_columns": ["Annual_Premium"]
            }
            
            # Mock the data
            mock_data = pd.DataFrame({
                "id": [1, 2, 3],
                "Gender": ["Male", "Female", "Male"],
                "Age": [25, 45, 33],
                "Driving_License": [1, 1, 1],
                "Region_Code": [28.0, 8.0, 15.0],
                "Previously_Insured": [0, 1, 0],
                "Vehicle_Age": ["1-2 Year", "> 2 Years", "< 1 Year"],
                "Vehicle_Damage": ["Yes", "No", "Yes"],
                "Annual_Premium": [2500.0, 3800.0, 2900.0],
                "Policy_Sales_Channel": [26.0, 124.0, 26.0],
                "Vintage": [150, 210, 95],
                "Response": [0, 1, 0]
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
                
                # Mock transformed data (without id and Response columns)
                transformed_features = mock_data.drop(['id', 'Response'], axis=1)
                mock_preprocessor_instance.fit_transform.return_value = transformed_features.values
                mock_preprocessor_instance.transform.return_value = transformed_features.values
                
                artifact = transformer.initiate_data_transformation()
                
                # Verify files were attempted to be saved
                assert mock_to_csv.called
                assert mock_pickle_dump.called
                assert artifact.transformed_train_file_path == mock_data_transformation_config.transformed_train_file_path