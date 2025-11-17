# tests/conftest.py
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
from src.entities.artifact_entity import DataIngestionArtifact
from src.entities.config_entities import DataValidationConfig
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


# New fixtures for Data Validation testing
@pytest.fixture
def sample_schema_config():
    """Fixture providing sample schema configuration for data validation"""
    return {
        "columns": {
            "feature_a": "numerical",
            "feature_b": "numerical", 
            "Response": "categorical"
        },
        "numerical_columns": ["feature_a", "feature_b"],
        "categorical_columns": ["Response"]
    }


@pytest.fixture
def validation_schema_config():
    """Fixture providing comprehensive schema config for validation tests"""
    return {
        "columns": {
            "col1": "int",
            "col2": "float",
            "col3": "string",
            "Response": "int"
        },
        "numerical_columns": ["col1", "col2"],
        "categorical_columns": ["col3", "Response"]
    }


@pytest.fixture
def valid_dataframe():
    """Fixture providing a valid DataFrame matching the schema"""
    return pd.DataFrame({
        "col1": [1, 2, 3, 4, 5],
        "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
        "col3": ["A", "B", "C", "A", "B"],
        "Response": [0, 1, 0, 1, 0]
    })


@pytest.fixture
def invalid_dataframe_missing_columns():
    """Fixture providing DataFrame with missing columns"""
    return pd.DataFrame({
        "col1": [1, 2, 3],  # Missing col2, col3, Response
    })


@pytest.fixture
def invalid_dataframe_wrong_count():
    """Fixture providing DataFrame with wrong number of columns"""
    return pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [1.1, 2.2, 3.3],
        "extra_col": ["X", "Y", "Z"]  # Extra column, missing required ones
    })


@pytest.fixture
def mock_data_ingestion_artifact(tmp_path):
    """Fixture providing mock DataIngestionArtifact with actual CSV files"""
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    
    # Create valid CSV files that match the schema
    train_df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [1.1, 2.2, 3.3],
        "col3": ["A", "B", "C"],
        "Response": [0, 1, 0]
    })
    test_df = pd.DataFrame({
        "col1": [4, 5],
        "col2": [4.4, 5.5],
        "col3": ["A", "B"],
        "Response": [1, 0]
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


@pytest.fixture
def mock_data_validation_config(tmp_path):
    """Fixture providing mock DataValidationConfig"""
    validation_dir = tmp_path / "data_validation"
    validation_dir.mkdir(exist_ok=True)
    
    return DataValidationConfig(
        data_validation_dir=validation_dir,
        validation_report_file_path=validation_dir / "validation_report.yaml"
    )


@pytest.fixture
def data_validation_setup(mock_data_ingestion_artifact, mock_data_validation_config, validation_schema_config):
    """Fixture providing complete setup for DataValidation tests"""
    with pytest.MonkeyPatch().context() as m:
        # Mock get_settings
        mock_settings = Mock()
        mock_settings.paths.schema_file_path = Path("test_schema.yaml")
        m.setattr('src.components.data_validation.get_settings', lambda: mock_settings)
        
        # Mock read_yaml_file to return our test schema
        m.setattr('src.components.data_validation.read_yaml_file', lambda file_path: validation_schema_config)
        
        from src.components.data_validation import DataValidation
        
        return DataValidation(
            data_ingestion_artifact=mock_data_ingestion_artifact,
            data_validation_config=mock_data_validation_config
        )


@pytest.fixture
def schema_yaml_file(tmp_path):
    """Fixture that creates a temporary schema YAML file"""
    schema_content = {
        "columns": {
            "feature_a": "numerical",
            "feature_b": "numerical",
            "Response": "categorical"
        },
        "numerical_columns": ["feature_a", "feature_b"],
        "categorical_columns": ["Response"]
    }
    
    schema_file = tmp_path / "schema.yaml"
    with open(schema_file, 'w') as f:
        json.dump(schema_content, f)
    
    return schema_file


@pytest.fixture
def empty_dataframe():
    """Fixture providing empty DataFrame for edge case testing"""
    return pd.DataFrame()


@pytest.fixture
def large_dataframe():
    """Fixture providing larger DataFrame for performance testing"""
    return pd.DataFrame({
        "col1": range(1000),
        "col2": [i * 1.1 for i in range(1000)],
        "col3": ["A", "B"] * 500,
        "Response": [0, 1] * 500
    })


@pytest.fixture
def dataframe_with_special_characters():
    """Fixture providing DataFrame with special characters in categorical columns"""
    return pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [1.1, 2.2, 3.3],
        "col3": ["A-B", "C_D", "E@F"],  # Special characters
        "Response": [0, 1, 0]
    })


@pytest.fixture
def dataframe_with_mixed_data_types():
    """Fixture providing DataFrame with mixed data types"""
    return pd.DataFrame({
        "col1": [1, "2", 3],  # Mixed types in numerical column
        "col2": [1.1, 2.2, 3.3],
        "col3": ["A", "B", "C"],
        "Response": [0, 1, 0]
    })


# Fixture para testing de errores
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