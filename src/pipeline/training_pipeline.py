# src/pipeline/training_pipeline.py
from __future__ import annotations

from typing import Optional

from src.logger import log
from src.entities.config_entities import (
    build_entities,
    RunEntities,
    TrainingPipelineConfig,
    DataIngestionConfig,
)
from src.entities.artifact_entity import DataIngestionArtifact
from src.components.data_ingestion import DataIngestion


class TrainPipeline:
    """
    Orchestrates the training pipeline.
    Currently ONLY launches the Data Ingestion stage,
    using:
      - Settings (Pydantic) -> build_entities()
      - Run-specific configs (TrainingPipelineConfig, DataIngestionConfig)
      - Pydantic Artifacts (DataIngestionArtifact)
    """

    def __init__(self, entities: Optional[RunEntities] = None) -> None:
        # If no RunEntities is injected, create a new one (new timestamp, etc.)
        self.entities: RunEntities = entities or build_entities()

        # Shortcuts to the configs we're interested in now
        self.training_cfg: TrainingPipelineConfig = self.entities.training
        self.ingestion_cfg: DataIngestionConfig = self.entities.ingestion

        log.info(
            "TrainPipeline initialized | timestamp=%s | artifact_dir=%s",
            self.entities.timestamp,
            str(self.training_cfg.artifact_dir),
        )

    # ---------------------------
    # Stage 1: Data Ingestion
    # ---------------------------
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Launches the Data Ingestion stage:
          - Export data from source (Mongo through the repository)
          - Save snapshot in feature_store (data.csv)
          - Split train/test (train.csv, test.csv)
          - Return DataIngestionArtifact with the final paths
        """
        log.info("Starting start_data_ingestion in TrainPipeline")
        try:
            data_ingestion = DataIngestion(
                training_cfg=self.training_cfg,
                ingestion_cfg=self.ingestion_cfg,
                # repo: if we don't pass anything, DataIngestion will create the real one (VehicleInsuranceRepository)
            )

            artifact = data_ingestion.initiate_data_ingestion()

            log.info(
                "Data Ingestion completed. train=%s | test=%s",
                str(artifact.trained_file_path),
                str(artifact.test_file_path),
            )
            log.info("End of start_data_ingestion in TrainPipeline")

            return artifact

        except Exception as exc:
            # Here we decide to log and re-raise the generic exception.
            # We don't use MyException nor sys.
            log.error("Error during Data Ingestion in TrainPipeline: %s", exc)
            raise

    # ---------------------------
    # Orchestration (for now: only ingestion)
    # ---------------------------
    def run_pipeline(self) -> None:
        """
        Launches the complete pipeline.
        Currently, only executes Data Ingestion.
        We will add the other stages here later.
        """
        log.info("=== Starting run_pipeline (ingestion only) ===")
        try:
            ingestion_artifact = self.start_data_ingestion()

            # Here you could, for example, log the complete artifact if you want:
            log.info(
                "Ingestion artifact generated: train=%s | test=%s",
                str(ingestion_artifact.trained_file_path),
                str(ingestion_artifact.test_file_path),
            )

            log.info("=== End of run_pipeline (ingestion only) ===")

        except Exception as exc:
            log.error("Failure in run_pipeline (ingestion): %s", exc)
            # re-raise the exception so the CLI/API/whatever is aware
            raise