from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
from src.entities.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entities.config_entities import DataTransformationConfig, DataValidationConfig
import pytest

# A tiny fake repo that mimics your VehicleInsuranceRepository
class FakeRepo:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def export_collection_as_dataframe(self, collection_name: str) -> pd.DataFrame:
        # collection_name is ignored here; real repo would use it
        return self._df.copy()


@pytest.fixture
def small_df():
    # Small, simple dataframe with an optional target column "Response"
    return pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5, 6],
        "feature_b": [10, 20, 30, 40, 50, 60],
        "Response":  [0, 0, 0, 1, 1, 1],   # balanced 50/50
    })


@pytest.fixture
def imbalanced_df():
    # Imbalanced target to test stratify behavior
    return pd.DataFrame({
        "x": list(range(100)),
        "Response": [1]*10 + [0]*90,  # 10% positive
    })

# Data Ingestion fixtures
@pytest.fixture
def mock_data_ingestion_artifact(tmp_path):
    """Fixture providing mock DataIngestionArtifact with actual CSV files"""
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    
    # Create valid CSV files that match the REAL schema
    train_df = pd.DataFrame({
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
    test_df = pd.DataFrame({
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
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    return DataIngestionArtifact(
        trained_file_path=train_file,
        test_file_path=test_file
    )

@pytest.fixture
def mock_data_ingestion_artifact_invalid(tmp_path):
    """Fixture providing mock DataIngestionArtifact with invalid CSV files"""
    train_file = tmp_path / "train_invalid.csv"
    test_file = tmp_path / "test_invalid.csv"
    
    # Create invalid CSV files that don't match schema
    train_df = pd.DataFrame({
        "wrong_col1": [1, 2, 3],
        "wrong_col2": ["X", "Y", "Z"]
    })
    test_df = pd.DataFrame({
        "wrong_col1": [4, 5],
        "wrong_col2": ["W", "Z"]
    })
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    return DataIngestionArtifact(
        trained_file_path=train_file,
        test_file_path=test_file
    )


# Data Validation fixtures
@pytest.fixture
def insurance_small_df():
    """Fixture providing realistic vehicle insurance data matching the actual schema"""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "Gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
        "Age": [25, 45, 33, 28, 52, 37],
        "Driving_License": [1, 1, 1, 1, 1, 1],
        "Region_Code": [28.0, 8.0, 15.0, 28.0, 3.0, 8.0],
        "Previously_Insured": [0, 1, 0, 0, 1, 0],
        "Vehicle_Age": ["1-2 Year", "> 2 Years", "< 1 Year", "1-2 Year", "> 2 Years", "< 1 Year"],
        "Vehicle_Damage": ["Yes", "No", "Yes", "No", "Yes", "No"],
        "Annual_Premium": [2500.0, 3800.0, 2900.0, 2100.0, 4500.0, 3200.0],
        "Policy_Sales_Channel": [26.0, 124.0, 26.0, 152.0, 124.0, 26.0],
        "Vintage": [150, 210, 95, 180, 300, 120],
        "Response": [0, 1, 0, 0, 1, 0]
    })

@pytest.fixture
def insurance_schema_config():
    """Schema config that matches the actual schema.yaml structure"""
    return {
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
        
@pytest.fixture
def mock_data_validation_config(tmp_path):
    """Fixture providing mock DataValidationConfig"""
    validation_dir = tmp_path / "data_validation"
    validation_dir.mkdir(exist_ok=True)
    
    return DataValidationConfig(
        data_validation_dir=validation_dir,
        validation_report_file_path=validation_dir / "validation_report.yaml"
    )


# Data Transformation fixtures
@pytest.fixture
def mock_data_transformation_config(tmp_path) -> DataTransformationConfig:
    """Create a DataTransformationConfig with temporary output paths."""
    data_transformation_dir = tmp_path / "data_transformation"
    data_transformation_dir.mkdir(exist_ok=True)

    # Aunque uses np.save / dill, la extensión es estética; se puede usar .npy / .pkl
    return DataTransformationConfig(
        data_transformation_dir=data_transformation_dir,
        transformed_train_file_path=data_transformation_dir / "transformed_train.npy",
        transformed_test_file_path=data_transformation_dir / "transformed_test.npy",
        transformed_object_file_path=data_transformation_dir / "preprocessor.pkl",
    )

@pytest.fixture
def sample_transformed_data():
    """Return a small DataFrame representing already-transformed features"""
    return pd.DataFrame({
        "Age_scaled": [0.1, 0.2, 0.3, 0.4],
        "Annual_Premium_scaled": [0.15, 0.25, 0.35, 0.45],
        "Vintage_scaled": [0.2, 0.3, 0.4, 0.5],
        "Gender_Male": [1, 0, 1, 0],
        "Gender_Female": [0, 1, 0, 1],
        "Vehicle_Age_<1_Year": [1, 0, 0, 1],
        "Vehicle_Age_1-2_Year": [0, 1, 0, 0],
        "Vehicle_Age_>2_Years": [0, 0, 1, 0],
        "Vehicle_Damage_Yes": [1, 0, 1, 0],
        "Vehicle_Damage_No": [0, 1, 0, 1],
        "Response": [0, 1, 0, 1],
    })

@pytest.fixture
def schema_config_complete():
    """Return a complete schema configuration used by DataTransformation."""
    return {
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
            {"Response": "int"},
        ],
        "numerical_columns": [
            "Age",
            "Driving_License",
            "Region_Code",
            "Previously_Insured",
            "Annual_Premium",
            "Policy_Sales_Channel",
            "Vintage",
            "Response",
        ],
        "categorical_columns": [
            "Gender",
            "Vehicle_Age",
            "Vehicle_Damage",
        ],
        "drop_columns": ["id"],
        "num_features": ["Age", "Vintage"],
        "mm_columns": ["Annual_Premium"],
    }

@pytest.fixture
def mock_data_transformation_class(schema_config_complete):
    """Return a DataTransformation class patched to use a fixed schema configuration."""
    from unittest.mock import patch

    with patch("src.components.data_transformation.read_yaml_file") as mock_read_yaml:
        mock_read_yaml.return_value = schema_config_complete
        from src.components.data_transformation import DataTransformation
        return DataTransformation

@pytest.fixture
def mock_transformer_instance(
    mock_data_ingestion_artifact,
    mock_data_validation_artifact_success,
    mock_data_transformation_config,
    schema_config_complete,
):
    """Return a DataTransformation instance with mocked schema and artifacts."""
    from unittest.mock import patch

    with patch("src.components.data_transformation.read_yaml_file") as mock_read_yaml:
        mock_read_yaml.return_value = schema_config_complete
        from src.components.data_transformation import DataTransformation

        return DataTransformation(
            data_ingestion_artifact=mock_data_ingestion_artifact,
            data_transformation_config=mock_data_transformation_config,
            data_validation_artifact=mock_data_validation_artifact_success,
        )


# Edge case fixtures
@pytest.fixture
def empty_dataframe():
    """Fixture providing empty DataFrame for edge case testing"""
    return pd.DataFrame()


@pytest.fixture
def corrupted_csv_file(tmp_path):
    """Fixture providing a corrupted CSV file"""
    corrupt_file = tmp_path / "corrupt.csv"
    corrupt_file.write_text("invalid,csv,content\nline1,without,proper,formatting\n")
    return corrupt_file


# Configuración global de pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "validation: mark test as data validation test"
    )
    