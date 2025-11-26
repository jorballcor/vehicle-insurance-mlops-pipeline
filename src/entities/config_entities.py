from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config.settings import get_settings


def new_timestamp() -> str:
    """UTC timestamp to avoid local timezone issues."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ===================================================================
# CORE PIPELINE CONFIGS
# ===================================================================

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


# ===================================================================
# MODEL TRAINING CONFIGS
# ===================================================================

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
class ModelPusherConfig:
    bucket_name: str
    s3_model_key_path: str


@dataclass(frozen=True)
class VehiclePredictorConfig:
    model_file_path: Path
    model_bucket_name: str


# ===================================================================
# GLOBAL STRUCTURE RETURNED BY build_entities()
# ===================================================================

@dataclass(frozen=True)
class RunEntities:
    timestamp: str
    training: TrainingPipelineConfig
    ingestion: DataIngestionConfig
    validation: DataValidationConfig
    transformation: DataTransformationConfig
    model_trainer: ModelTrainerConfig
    model_evaluation: ModelEvaluationConfig
    model_pusher: ModelPusherConfig
    predictor: VehiclePredictorConfig


# ===================================================================
# FACTORY TO MATERIALIZE CONFIGS FOR A RUN
# ===================================================================

def build_entities(ts: str | None = None) -> RunEntities:
    """
    Compile all configuration entities from global Settings and a timestamp.

    This replaces the old constant-based config file with a dynamic,
    environment-driven setup based on pydantic-settings.
    """
    settings = get_settings()

    ts = ts or new_timestamp()
    artifact_root = settings.paths.artifact_dir / ts

    # --------------------------------------------------------------
    # TRAINING PIPELINE CONFIG
    # --------------------------------------------------------------
    training = TrainingPipelineConfig(
        pipeline_name=settings.paths.pipeline_name,
        artifact_dir=artifact_root,
        timestamp=ts,
    )

    # --------------------------------------------------------------
    # INGESTION CONFIG
    # --------------------------------------------------------------
    ingestion_root = artifact_root / settings.ingestion.dir_name
    feature_store = ingestion_root / settings.ingestion.feature_store_dir / settings.paths.file_name
    train_file = ingestion_root / settings.ingestion.ingested_dir / settings.paths.train_file_name
    test_file = ingestion_root / settings.ingestion.ingested_dir / settings.paths.test_file_name

    ingestion = DataIngestionConfig(
        data_ingestion_dir=ingestion_root,
        feature_store_file_path=feature_store,
        training_file_path=train_file,
        testing_file_path=test_file,
        train_test_split_ratio=settings.ingestion.train_test_split_ratio,
        mongo_url=(settings.mongo.url.get_secret_value() if settings.mongo.url else None),
        mongo_database=settings.mongo.database_name,
        collection_name=settings.ingestion.collection_name or settings.mongo.collection_name,
    )

    # --------------------------------------------------------------
    # VALIDATION CONFIG
    # --------------------------------------------------------------
    validation_root = artifact_root / settings.validation.dir_name

    validation = DataValidationConfig(
        data_validation_dir=validation_root,
        validation_report_file_path=validation_root / settings.validation.report_file_name,
    )

    # --------------------------------------------------------------
    # TRANSFORMATION CONFIG
    # --------------------------------------------------------------
    transf_root = artifact_root / settings.transformation.dir_name

    transformed_train = (
        transf_root
        / settings.transformation.transformed_data_dir
        / Path(settings.paths.train_file_name).with_suffix(".npy").name
    )
    transformed_test = (
        transf_root
        / settings.transformation.transformed_data_dir
        / Path(settings.paths.test_file_name).with_suffix(".npy").name
    )
    transformed_object = (
        transf_root
        / settings.transformation.transformed_object_dir
        / settings.paths.preprocessing_object_file_name
    )

    transformation = DataTransformationConfig(
        data_transformation_dir=transf_root,
        transformed_train_file_path=transformed_train,
        transformed_test_file_path=transformed_test,
        transformed_object_file_path=transformed_object,
    )

    # --------------------------------------------------------------
    # MODEL TRAINER CONFIG
    # --------------------------------------------------------------
    trainer_root = artifact_root / settings.trainer.dir_name
    trained_model_path = (
        trainer_root
        / settings.trainer.trained_model_dir
        / settings.trainer.trained_model_name
    )

    model_trainer = ModelTrainerConfig(
        model_trainer_dir=trainer_root,
        trained_model_file_path=trained_model_path,
        expected_accuracy=settings.trainer.expected_accuracy,
        model_config_file_path=settings.trainer.model_config_file_path,
        n_estimators=settings.trainer.n_estimators,
        min_samples_split=settings.trainer.min_samples_split,
        min_samples_leaf=settings.trainer.min_samples_leaf,
        max_depth=settings.trainer.max_depth,
        criterion=settings.trainer.criterion,
        random_state=settings.trainer.random_state,
    )

    # --------------------------------------------------------------
    # MODEL EVALUATION CONFIG
    # --------------------------------------------------------------
    model_evaluation = ModelEvaluationConfig(
        changed_threshold_score=settings.evaluation.changed_threshold_score,
        bucket_name=settings.evaluation.bucket_name,
        s3_model_key_path=settings.evaluation.pusher_s3_key
    )

    # --------------------------------------------------------------
    # MODEL PUSHER CONFIG
    # --------------------------------------------------------------
    model_pusher = ModelPusherConfig(
        bucket_name=settings.evaluation.bucket_name,
        s3_model_key_path=settings.evaluation.pusher_s3_key,
    )

    # --------------------------------------------------------------
    # VEHICLE PREDICTOR CONFIG (production inference)
    # --------------------------------------------------------------
    predictor = VehiclePredictorConfig(
        model_file_path=settings.evaluation.model_file_path,
        model_bucket_name=settings.evaluation.model_bucket_name,
    )

    # --------------------------------------------------------------
    # RETURN EVERYTHING
    # --------------------------------------------------------------
    return RunEntities(
        timestamp=ts,
        training=training,
        ingestion=ingestion,
        validation=validation,
        transformation=transformation,
        model_trainer=model_trainer,
        model_evaluation=model_evaluation,
        model_pusher=model_pusher,
        predictor=predictor,
    )
