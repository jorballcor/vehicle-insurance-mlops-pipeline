from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.metrics import f1_score

from config.settings import get_settings
from src.entities.config_entities import ModelEvaluationConfig
from src.entities.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact,
)
from src.entities.s3_estimator import VehicleInsuranceEstimator
from src.logger import log
from src.utils.file_utils import load_object


settings = get_settings()
TARGET_COLUMN = settings.target_column


@dataclass
class EvaluateModelResponse:
    """Container for comparison metrics between trained and production models."""

    trained_model_f1_score: float
    best_model_f1_score: float | None
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    """
    Compare the newly trained model against the current production model
    (if any) and decide whether to accept the new model.
    """

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        """
        Args:
            model_eval_config (ModelEvaluationConfig): Evaluation configuration,
                including S3 bucket and key for the production model.
            data_ingestion_artifact (DataIngestionArtifact): Artifact containing
                paths to ingested train/test data.
            model_trainer_artifact (ModelTrainerArtifact): Artifact produced by
                the training step with model path and metrics.
        """
        self.model_eval_config = model_eval_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.model_trainer_artifact = model_trainer_artifact

    def get_best_model(self) -> Optional[VehicleInsuranceEstimator]:
        """
        Try to obtain the current production model from S3.

        Returns:
            Optional[VehicleInsuranceEstimator]: A VehicleInsuranceEstimator if a model exists in S3,
            otherwise None.
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path

            estimator = VehicleInsuranceEstimator(
                bucket_name=bucket_name,
                model_path=model_path,
            )

            if estimator.is_model_present(model_path=model_path):
                log.info(
                    f"Found existing production model in S3: bucket={bucket_name}, key={model_path}"
                )
                return estimator

            log.info("No existing production model found in S3.")
            return None

        except Exception as e:
            msg = f"Error while fetching best (production) model from S3: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    def _map_gender_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map Gender column to 0 for Female and 1 for Male."""
        log.info("Mapping 'Gender' column to binary values.")
        df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).astype(int)
        return df

    def _create_dummy_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dummy variables for categorical features."""
        log.info("Creating dummy variables for categorical features.")
        return pd.get_dummies(df, drop_first=True)

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename specific columns and ensure integer types for dummy columns."""
        log.info("Renaming specific columns and casting to int where needed.")
        df = df.rename(
            columns={
                "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
                "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years",
            }
        )
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype("int")
        return df

    def _drop_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the '_id' column if it exists (Mongo-originated)."""
        log.info("Dropping '_id' column if present.")
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Evaluate the newly trained model against the current production model (if any).

        Returns:
            EvaluateModelResponse: F1 scores for both models and acceptance decision.
        """
        try:
            log.info("Loading test data for model evaluation.")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            log.info("Test data loaded. Applying transformation pipeline used for training...")

            # Apply same custom transformations as in training
            x = self._map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            # Load newly trained model
            trained_model = load_object(
                file_path=self.model_trainer_artifact.trained_model_file_path
            )
            log.info("Trained model loaded from disk.")

            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            log.info(f"F1 score for newly trained model (from training): {trained_model_f1_score}")

            best_model_f1_score: float | None = None
            best_model = self.get_best_model()

            if best_model is not None:
                log.info("Computing F1 score for current production model.")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                log.info(
                    f"F1 score - Production model: {best_model_f1_score}, "
                    f"New trained model: {trained_model_f1_score}"
                )

            tmp_best = 0.0 if best_model_f1_score is None else best_model_f1_score

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best,
                difference=trained_model_f1_score - tmp_best,
            )

            log.info(f"Model evaluation result: {result}")
            return result

        except Exception as e:
            msg = f"Error during model evaluation: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Run the full evaluation flow and create a ModelEvaluationArtifact.

        Returns:
            ModelEvaluationArtifact: Summary of evaluation and model acceptance.
        """
        try:
            log.info("Initializing Model Evaluation component.")
            evaluation_response = self.evaluate_model()

            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluation_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluation_response.difference,
            )

            log.info(f"Model evaluation artifact created: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            msg = f"Error while initiating model evaluation: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e
