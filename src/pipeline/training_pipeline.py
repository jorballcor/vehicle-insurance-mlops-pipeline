# src/pipeline/training_pipeline.py
from __future__ import annotations

from typing import Optional

from src.logger import log
from src.entities.config_entities import (
    build_entities,
    RunEntities,
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig
)
from src.entities.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation


class TrainPipeline:
    """
    Orchestrates the training pipeline.
    using:
      - Settings (Pydantic) -> build_entities()
      - Run-specific configs (TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig)
      - Pydantic Artifacts (DataIngestionArtifact, DataValidationArtifact)
    """

    def __init__(self, entities: Optional[RunEntities] = None) -> None:
        # If no RunEntities is injected, create a new one (new timestamp, etc.)
        self.entities: RunEntities = entities or build_entities()

        # Shortcuts to the configs we're interested in now
        self.training_cfg: TrainingPipelineConfig = self.entities.training
        self.ingestion_cfg: DataIngestionConfig = self.entities.ingestion
        self.validation_cfg: DataValidationConfig = self.entities.validation
        

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
    # Stage 2: Data Validation
    # ---------------------------   
    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
    ) -> DataValidationArtifact:
        """
        Lanza el componente de Data Validation.
        """
        log.info("Starting start_data_validation in TrainPipeline")

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.validation_cfg,  # <- viene de tus RunEntities
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            log.info(
                "Data validation completed | status=%s | report=%s",
                data_validation_artifact.validation_status,
                data_validation_artifact.validation_report_file_path,
            )
            log.info("Finished start_data_validation in TrainPipeline")

            return data_validation_artifact

        except Exception as exc:
            log.error("Error during Data Validation in TrainPipeline: %s", exc)
            raise


    # ---------------------------
    # Orchestration (for now: only ingestion & validation)
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

            log.info(
                "Ingestion artifact generated: train=%s | test=%s",
                str(ingestion_artifact.trained_file_path),
                str(ingestion_artifact.test_file_path),
            )
            
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=ingestion_artifact)
            log.info("Completed data validation: %s", str(data_validation_artifact))
            

            log.info("=== End of run_pipeline (ingestion only) ===")

        except Exception as exc:
            log.error("Failure in run_pipeline: %s", exc)
            # re-raise the exception so the CLI/API/whatever is aware
            raise