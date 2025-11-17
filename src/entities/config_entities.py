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
class RunEntities:
    timestamp: str
    training: TrainingPipelineConfig
    ingestion: DataIngestionConfig


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

    return RunEntities(
        timestamp=ts,
        training=training,
        ingestion=ingestion,
    )
