# src/pipeline/training_pipeline.py
from __future__ import annotations

from typing import Optional

from src.logger import log
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

from src.entities.config_entities import (
    build_entities,
    RunEntities,
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)
from src.entities.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)


class TrainPipeline:
    """
    Orchestrates the full training pipeline:

      1. Data Ingestion
      2. Data Validation
      3. Data Transformation
      4. Model Training
      5. Model Evaluation
      6. Model Pusher (if accepted)
    """

    def __init__(self, entities: Optional[RunEntities] = None) -> None:
        # If no RunEntities is injected, create a new one (new timestamp, etc.)
        self.entities: RunEntities = entities or build_entities()

        # Shortcuts to configuration objects
        self.training_cfg: TrainingPipelineConfig = self.entities.training
        self.ingestion_cfg: DataIngestionConfig = self.entities.ingestion
        self.validation_cfg: DataValidationConfig = self.entities.validation
        self.data_transformation_config: DataTransformationConfig = self.entities.transformation
        self.model_trainer_config: ModelTrainerConfig = self.entities.model_trainer
        self.model_evaluation_config: ModelEvaluationConfig = self.entities.model_evaluation
        self.model_pusher_config: ModelPusherConfig = self.entities.model_pusher

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
        Launch the Data Ingestion stage:
          - Export data from source (Mongo through the repository)
          - Save snapshot in feature_store (data.csv)
          - Split train/test (train.csv, test.csv)
          - Return DataIngestionArtifact with final paths
        """
        log.info("Starting start_data_ingestion in TrainPipeline")

        try:
            data_ingestion = DataIngestion(
                training_cfg=self.training_cfg,
                ingestion_cfg=self.ingestion_cfg,
                # repo: if we don't pass anything, DataIngestion will create the real one
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
        Launch the Data Validation stage.
        """
        log.info("Starting start_data_validation in TrainPipeline")

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.validation_cfg,
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
    # Stage 3: Data Transformation
    # ---------------------------
    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        """
        Launch the Data Transformation stage.
        """
        try:
            log.info("Initializing Data Transformation component...")

            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )

            log.info("Executing data transformation process...")
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            log.info("Data Transformation completed successfully.")
            return data_transformation_artifact

        except Exception as exc:
            msg = f"Error while starting data transformation: {exc}"
            log.error(msg)
            raise RuntimeError(msg) from exc

    # ---------------------------
    # Stage 4: Model Trainer
    # ---------------------------
    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:
        """
        Launch the Model Training stage.
        """
        try:
            log.info("Starting model training stage in TrainPipeline")

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            log.info("Model training completed.")
            return model_trainer_artifact

        except Exception as exc:
            msg = f"Error while starting model training: {exc}"
            log.error(msg)
            raise RuntimeError(msg) from exc

    # ---------------------------
    # Stage 5: Model Evaluation
    # ---------------------------
    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        """
        Launch the Model Evaluation stage.
        Compares the newly trained model with the current production model.
        """
        try:
            log.info("Starting model evaluation stage in TrainPipeline")

            evaluator = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )

            model_evaluation_artifact = evaluator.initiate_model_evaluation()

            log.info("Model evaluation completed.")
            return model_evaluation_artifact

        except Exception as exc:
            msg = f"Error during model evaluation: {exc}"
            log.error(msg)
            raise RuntimeError(msg) from exc

    # ---------------------------
    # Stage 6: Model Pusher
    # ---------------------------
    def start_model_pusher(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
    ) -> ModelPusherArtifact:
        """
        Launch the Model Pusher stage.
        Uploads the accepted model to the production S3 bucket.
        """
        try:
            log.info("Starting model pusher stage in TrainPipeline")

            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config,
            )

            model_pusher_artifact = model_pusher.initiate_model_pusher()

            log.info("Model pusher stage completed.")
            return model_pusher_artifact

        except Exception as exc:
            msg = f"Error during model pusher stage: {exc}"
            log.error(msg)
            raise RuntimeError(msg) from exc

    # ---------------------------
    # Orchestration
    # ---------------------------
    def run_pipeline(self) -> None:
        """
        Run the complete pipeline end-to-end:
          Ingestion → Validation → Transformation → Training → Evaluation → Push (if accepted).
        """
        log.info("=== Starting run_pipeline ===")

        try:
            # 1) Ingestion
            ingestion_artifact = self.start_data_ingestion()

            # 2) Validation
            validation_artifact = self.start_data_validation(
                data_ingestion_artifact=ingestion_artifact
            )

            # 3) Transformation
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=ingestion_artifact,
                data_validation_artifact=validation_artifact,
            )

            # 4) Model Training
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            # 5) Model Evaluation
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )

            if not model_evaluation_artifact.is_model_accepted:
                log.info("Model not accepted by evaluation. Skipping model push.")
                log.info("=== End of run_pipeline (model rejected) ===")
                return

            # 6) Model Pusher
            model_pusher_artifact = self.start_model_pusher(
                model_evaluation_artifact=model_evaluation_artifact
            )
            log.info("Completed model push: %s", str(model_pusher_artifact))

            log.info("=== End of run_pipeline (success) ===")

        except Exception as exc:
            log.error("Failure in run_pipeline: %s", exc)
            raise
