# src/entities/config_entity.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config.settings import get_settings


def new_timestamp() -> str:
    # Use UTC to avoid timezone surprises
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class TrainingPipelineConfig:
    pipeline_name: str
    artifact_dir: Path
    timestamp: str


@dataclass(frozen=True)
class DataIngestionConfig:
    data_ingestion_dir: Path
    feature_store_file_path: Path
    training_file_path: Path
    testing_file_path: Path
    train_test_split_ratio: float
    # Source (Mongo)
    mongo_url: str | None
    mongo_database: str
    collection_name: str


@dataclass(frozen=True)
class DataValidationConfig:
    data_validation_dir: Path
    validation_report_file_path: Path
    

@dataclass(frozen=True)
class DataTransformationConfig:
    data_transformation_dir: Path
    transformed_train_file_path: Path
    transformed_test_file_path: Path
    transformed_object_file_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    model_trainer_dir: Path
    trained_model_file_path: Path
    expected_accuracy: float
    model_config_file_path: Path
    n_estimators: int
    min_samples_split: int
    min_samples_leaf: int
    max_depth: int
    criterion: str
    random_state: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    changed_threshold_score: float
    bucket_name: str
    s3_model_key_path: str


@dataclass(frozen=True)
class RunEntities:
    timestamp: str
    training: TrainingPipelineConfig
    ingestion: DataIngestionConfig
    validation: DataValidationConfig


def build_entities(ts: str | None = None) -> RunEntities:
    """
    Materialize per-run ingestion configs from global settings.
    Mirrors the previous full structure but limited to training + ingestion.
    """
    settings = get_settings()
    
    ts = ts or new_timestamp()

    artifact_root = settings.paths.artifact_dir / ts

    training = TrainingPipelineConfig(
        pipeline_name=settings.paths.pipeline_name,
        artifact_dir=artifact_root,
        timestamp=ts,
    )

    ingestion_root = artifact_root / settings.ingestion.dir_name
    feature_store_file = ingestion_root / settings.ingestion.feature_store_dir / settings.paths.file_name
    train_file = ingestion_root / settings.ingestion.ingested_dir / settings.paths.train_file_name
    test_file = ingestion_root / settings.ingestion.ingested_dir / settings.paths.test_file_name

    ingestion = DataIngestionConfig(
        data_ingestion_dir=ingestion_root,
        feature_store_file_path=feature_store_file,
        training_file_path=train_file,
        testing_file_path=test_file,
        train_test_split_ratio=settings.ingestion.train_test_split_ratio,
        mongo_url=(settings.mongo.url.get_secret_value() if settings.mongo.url else None),
        mongo_database=settings.mongo.database_name,
        collection_name=settings.ingestion.collection_name or settings.mongo.collection_name,
    )
    
    validation_root = artifact_root / settings.validation.dir_name
    validation = DataValidationConfig(
        data_validation_dir=validation_root,
        validation_report_file_path=validation_root / settings.validation.report_file_name,
    )
    
    transf_root = artifact_root / settings.transformation.dir_name
    transformed_train = transf_root / settings.transformation.transformed_data_dir / Path(settings.paths.train_file_name).with_suffix(".npy").name
    transformed_test = transf_root / settings.transformation.transformed_data_dir / Path(settings.paths.test_file_name).with_suffix(".npy").name
    transformed_object = transf_root / settings.transformation.transformed_object_dir / settings.paths.preprocessing_object_file_name

    transformation = DataTransformationConfig(
        data_transformation_dir=transf_root,
        transformed_train_file_path=transformed_train,
        transformed_test_file_path=transformed_test,
        transformed_object_file_path=transformed_object,
    )


    return RunEntities(
        timestamp=ts,
        training=training,
        ingestion=ingestion,
        validation=validation,
    )
