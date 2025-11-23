from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.logger import log
from src.utils.file_utils import load_numpy_array_data, load_object, save_object
from src.entities.config_entities import ModelTrainerConfig
from src.entities.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from src.entities.estimator import MyModel


class ModelTrainer:
    """
    Handles model training, evaluation, and creation of the final wrapped model (MyModel),
    which includes both preprocessing and the trained model object.
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        """
        Args:
            data_transformation_artifact (DataTransformationArtifact): 
                Output artifact from the transformation stage.
            model_trainer_config (ModelTrainerConfig):
                Configuration settings required for model training.
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(
        self, train: np.ndarray, test: np.ndarray
    ) -> Tuple[object, ClassificationMetricArtifact]:
        """
        Train a RandomForestClassifier using the transformed train/test arrays.

        Args:
            train (np.ndarray): Training dataset with features + target.
            test (np.ndarray): Test dataset with features + target.

        Returns:
            Tuple containing:
                - trained RandomForest model
                - ClassificationMetricArtifact with F1, precision, recall
        """
        try:
            log.info("Training RandomForestClassifier with configured hyperparameters.")

            # Split into features and target
            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]
            log.info("train/test arrays successfully split into X and y.")

            # Build the model
            model = RandomForestClassifier(
                n_estimators=self.model_trainer_config.n_estimators,
                min_samples_split=self.model_trainer_config.min_samples_split,
                min_samples_leaf=self.model_trainer_config.min_samples_leaf,
                max_depth=self.model_trainer_config.max_depth,
                criterion=self.model_trainer_config.criterion,
                random_state=self.model_trainer_config.random_state,
            )

            # Fit model
            log.info("Fitting RandomForestClassifier...")
            model.fit(x_train, y_train)
            log.info("Model training completed.")

            # Predict & metrics
            y_pred = model.predict(x_test)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Build metric artifact
            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1, precision_score=precision, recall_score=recall
            )

            return model, metric_artifact

        except Exception as e:
            msg = f"Error during model training or metric computation: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Executes the model training pipeline:
        - Loads transformed arrays
        - Trains the model
        - Validates threshold accuracy
        - Wraps it into MyModel
        - Saves to disk
        - Returns ModelTrainerArtifact
        """
        try:
            log.info("Starting model trainer component...")

            # Load transformed arrays
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            if train_arr is None or test_arr is None:
                raise RuntimeError("Transformed train or test array could not be loaded.")

            log.info("Transformed train/test arrays loaded successfully.")

            # Train model & compute metrics
            trained_model, metric_artifact = self.get_model_object_and_report(
                train=train_arr, test=test_arr
            )
            log.info("Model trained and metrics computed.")

            # Load preprocessing pipeline
            preprocessing_obj = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )
            log.info("Preprocessing object loaded successfully.")

            # Check if accuracy meets threshold
            train_accuracy = accuracy_score(
                train_arr[:, -1], trained_model.predict(train_arr[:, :-1])
            )
            log.info(f"Training accuracy: {train_accuracy}")

            if train_accuracy < self.model_trainer_config.expected_accuracy:
                msg = (
                    f"Model accuracy {train_accuracy:.3f} is below the "
                    f"expected threshold {self.model_trainer_config.expected_accuracy:.3f}."
                )
                log.error(msg)
                raise RuntimeError(msg)

            # Wrap model into MyModel class
            log.info("Saving final wrapped model (MyModel).")
            my_model = MyModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=trained_model,
            )

            save_object(self.model_trainer_config.trained_model_file_path, my_model)

            # Return artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )

            log.info(f"ModelTrainerArtifact created: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            msg = f"Error in ModelTrainer: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e
