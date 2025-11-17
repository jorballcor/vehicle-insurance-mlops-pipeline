# settings.py
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    field_validator,
    computed_field,
    conint,
    confloat,
    AliasChoices
)

from pydantic_settings import BaseSettings, SettingsConfigDict

# -------------------------
# Submodelos por dominio
# -------------------------

class MongoSettings(BaseModel):
    url: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("MONGODB_URL", "MONGODB_CONNECTION_URL"),
    )
    database_name: str = "vehicle-insurance"
    collection_name: str = "vehicle-insurance-data"


class PathsSettings(BaseModel):
    pipeline_name: str = ""                      
    artifact_dir: Path = Path("artifact")
    schema_file_path: Path = Path("config") / "schema.yaml"
    model_file_name: str = "model.pkl"
    preprocessing_object_file_name: str = "preprocessing.pkl" 
    file_name: str = "data.csv"
    train_file_name: str = "train.csv"
    test_file_name: str = "test.csv"

    @field_validator(
        "artifact_dir", "schema_file_path", mode="before"
    )
    @classmethod
    def _to_path(cls, v):
        return Path(v)

    @field_validator("schema_file_path")
    @classmethod
    def _exists_or_parent(cls, v: Path):
        # No forzamos que exista, pero normalizamos
        return v


class AWSSettings(BaseSettings):
    # Se tomarán de env si existen; si no, quedan en None
    access_key_id: Optional[SecretStr] = Field(default=None, validation_alias="AWS_ACCESS_KEY_ID")
    secret_access_key: Optional[SecretStr] = Field(default=None, validation_alias="AWS_SECRET_ACCESS_KEY")
    region_name: str = "us-east-1"

    model_config = {
        "env_prefix": "",  # usamos alias explícitos arriba
        "extra": "ignore",
    }


class DataIngestionSettings(BaseModel):
    collection_name: str = "vehicle-insurance-data"
    dir_name: str = "data_ingestion"
    feature_store_dir: str = "feature_store"
    ingested_dir: str = "ingested"
    train_test_split_ratio: confloat(gt=0, lt=1) = 0.25


class DataValidationSettings(BaseModel):
    dir_name: str = "data_validation"
    report_file_name: str = "report.yaml"


class DataTransformationSettings(BaseModel):
    dir_name: str = "data_transformation"
    transformed_data_dir: str = "transformed"
    transformed_object_dir: str = "transformed_object"


class ModelTrainerSettings(BaseModel):
    dir_name: str = "model_trainer"
    trained_model_dir: str = "trained_model"
    trained_model_name: str = "model.pkl"
    expected_score: confloat(ge=0, le=1) = 0.60
    model_config_file_path: Path = Path("config") / "model.yaml"

    # Hiperparámetros (parece RandomForest/árboles)
    n_estimators: conint(ge=1) = 200
    min_samples_split: conint(ge=2) = 7
    min_samples_leaf: conint(ge=1) = 6
    max_depth: conint(ge=1) = 10                     # renombrado desde MIN_SAMPLES_SPLIT_MAX_DEPTH
    criterion: Literal["gini", "entropy", "log_loss", "squared_error"] = "entropy"
    random_state: int = 101

    @field_validator("model_config_file_path", mode="before")
    @classmethod
    def _to_path(cls, v):
        return Path(v)


class ModelEvaluationSettings(BaseModel):
    changed_threshold_score: confloat(gt=0, lt=1) = 0.02
    bucket_name: str = "my-model-mlopsproj"
    pusher_s3_key: str = "model-registry"

    @field_validator("bucket_name")
    @classmethod
    def validate_bucket(cls, v: str):
        # Reglas simplificadas de S3 bucket
        import re
        pat = r"^(?!\d+$)(?!-)(?!.*--)[a-z0-9.-]{3,63}(?<!-)$"
        if not re.fullmatch(pat, v):
            raise ValueError(
                "Nombre de bucket S3 inválido (3–63 chars, [a-z0-9.-], sin guiones dobles/edges y no solo números)."
            )
        return v


class AppSettings(BaseModel):
    host: str = "0.0.0.0"
    port: conint(ge=1, le=65535) = 5000


# -------------------------
# Settings raíz
# -------------------------

class Settings(BaseSettings):
    """
    Central configuration management for the Vehicle Insurance MLOps Pipeline.
    
    This class defines all application settings, constants, and configuration parameters
    using Pydantic for validation and type safety. It organizes settings by domain
    and supports environment variable overrides with nested configuration support.
    
    Attributes:
        target_column (str): The target variable for machine learning predictions.
        mongo (MongoSettings): MongoDB connection and database configuration.
        paths (PathsSettings): File system paths and directory structure.
        aws (AWSSettings): AWS credentials and service configurations.
        ingestion (DataIngestionSettings): Data ingestion pipeline parameters.
        validation (DataValidationSettings): Data validation and quality checks.
        transformation (DataTransformationSettings): Data preprocessing configuration.
        trainer (ModelTrainerSettings): Model training hyperparameters and settings.
        evaluation (ModelEvaluationSettings): Model evaluation and deployment criteria.
        app (AppSettings): Web application host and port configuration.
    
    Computed Fields:
        current_year (int): The current year, computed dynamically.
    
    Configuration:
        Environment variables can override defaults using double underscore notation
        (e.g., MONGO__DATABASE_NAME, TRAINER__N_ESTIMATORS).
    
    Example:
        >>> from src.config.settings import settings
        >>> db_name = settings.mongo.database_name
        >>> model_hyperparams = settings.trainer.n_estimators
    """
    
    # Negocio/datos
    target_column: str = "Response"

    # Agrupados
    mongo: MongoSettings = MongoSettings()
    paths: PathsSettings = PathsSettings()
    aws: AWSSettings = AWSSettings()
    ingestion: DataIngestionSettings = DataIngestionSettings()
    validation: DataValidationSettings = DataValidationSettings()
    transformation: DataTransformationSettings = DataTransformationSettings()
    trainer: ModelTrainerSettings = ModelTrainerSettings()
    evaluation: ModelEvaluationSettings = ModelEvaluationSettings()
    app: AppSettings = AppSettings()

    # Campo computado
    @computed_field
    @property
    def current_year(self) -> int:
        return date.today().year

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings class.
    Even though it is called multiple times, the same instance is returned for efficiency.
    """
    return Settings()