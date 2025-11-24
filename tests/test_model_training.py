from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.components.model_trainer import ModelTrainer
from src.entities.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from src.entities.config_entities import ModelTrainerConfig
from src.entities.estimator import MyModel


class TestModelTrainerGetModelObjectAndReport:
    """Unit tests for ModelTrainer.get_model_object_and_report."""

    @pytest.mark.unit
    def test_get_model_object_and_report_returns_model_and_metrics(
        self,
        data_transformation_artifact,
        model_trainer_config,
    ):
        """
        Ensure get_model_object_and_report:
        - Instantiates RandomForestClassifier with config parameters
        - Calls fit on train data
        - Computes metrics correctly based on mocked predictions
        """
        # Fake train and test arrays: last column is target
        train = np.array(
            [
                [0.0, 0],
                [1.0, 1],
                [2.0, 1],
            ]
        )
        test = np.array(
            [
                [0.5, 0],
                [1.5, 1],
            ]
        )

        with patch("src.components.model_trainer.RandomForestClassifier") as mock_rf_cls:
            mock_model = MagicMock()
            # Predict perfectly matches y_test = [0, 1]
            mock_model.predict.return_value = np.array([0, 1])
            mock_rf_cls.return_value = mock_model

            trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=model_trainer_config,
            )

            model, metric_artifact = trainer.get_model_object_and_report(
                train=train, test=test
            )

        # Assertions
        mock_rf_cls.assert_called_once()
        mock_model.fit.assert_called_once()
        assert model is mock_model
        assert isinstance(metric_artifact, ClassificationMetricArtifact)
        # Perfect precision/recall/F1 for binary case [0,1] vs [0,1]
        assert metric_artifact.f1_score == 1.0
        assert metric_artifact.precision_score == 1.0
        assert metric_artifact.recall_score == 1.0


class TestModelTrainerInitiateModelTrainer:
    """Unit tests for ModelTrainer.initiate_model_trainer."""

    @pytest.mark.unit
    def test_initiate_model_trainer_success_saves_model_and_returns_artifact(
        self,
        data_transformation_artifact,
        model_trainer_config,
    ):
        """
        Ensure initiate_model_trainer:
        - Loads transformed arrays via load_numpy_array_data
        - Calls get_model_object_and_report
        - Checks accuracy threshold
        - Wraps model into MyModel
        - Saves the model via save_object
        - Returns a ModelTrainerArtifact
        """
        trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=model_trainer_config,
        )

        # Fake transformed arrays: 2 samples, 2 features + 1 target
        train_arr = np.array(
            [
                [0.0, 1.0, 0],
                [1.0, 0.0, 1],
            ]
        )
        test_arr = np.array(
            [
                [0.5, 0.5, 0],
                [1.5, -0.5, 1],
            ]
        )

        # Fake trained model whose predict perfectly matches train targets
        fake_trained_model = MagicMock()
        fake_trained_model.predict.return_value = train_arr[:, -1]

        fake_metric_artifact = ClassificationMetricArtifact(
            f1_score=0.9,
            precision_score=0.9,
            recall_score=0.9,
        )

        fake_preprocessor = MagicMock()

        with patch(
            "src.components.model_trainer.load_numpy_array_data"
        ) as mock_load_array, patch.object(
            ModelTrainer, "get_model_object_and_report"
        ) as mock_get_model_and_metrics, patch(
            "src.components.model_trainer.load_object"
        ) as mock_load_object, patch(
            "src.components.model_trainer.save_object"
        ) as mock_save_object:

            mock_load_array.side_effect = [train_arr, test_arr]
            mock_get_model_and_metrics.return_value = (
                fake_trained_model,
                fake_metric_artifact,
            )
            mock_load_object.return_value = fake_preprocessor

            artifact = trainer.initiate_model_trainer()

        # Assertions on returned artifact
        assert isinstance(artifact, ModelTrainerArtifact)
        assert (
            artifact.trained_model_file_path
            == model_trainer_config.trained_model_file_path
        )
        assert artifact.metric_artifact == fake_metric_artifact

        # save_object should be called once with a MyModel instance
        mock_save_object.assert_called_once()
        args, _ = mock_save_object.call_args
        saved_path, saved_model = args[0], args[1]
        assert saved_path == model_trainer_config.trained_model_file_path
        assert isinstance(saved_model, MyModel)

        # Ensure the accuracy check was evaluated (predict called on train_arr)
        fake_trained_model.predict.assert_called()

    @pytest.mark.unit
    def test_initiate_model_trainer_raises_when_accuracy_below_threshold(
        self,
        data_transformation_artifact,
        tmp_path,
    ):
        """
        Ensure initiate_model_trainer raises RuntimeError when the model
        does not reach the expected accuracy threshold.
        """
        low_acc_config = ModelTrainerConfig(
            model_trainer_dir=tmp_path / "model_trainer_low_acc",
            trained_model_file_path=tmp_path / "model_trainer_low_acc" / "model.pkl",
            expected_accuracy=0.9,  # high threshold
            model_config_file_path=tmp_path / "model_trainer_low_acc" / "config.yaml",
            n_estimators=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            criterion="gini",
            random_state=42,
        )

        trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=low_acc_config,
        )

        # Fake arrays: model will always be wrong -> low accuracy
        train_arr = np.array(
            [
                [0.0, 1.0, 0],
                [1.0, 0.0, 1],
            ]
        )
        test_arr = np.array(
            [
                [0.5, 0.5, 0],
                [1.5, -0.5, 1],
            ]
        )

        fake_trained_model = MagicMock()
        # Always predict the opposite label -> accuracy 0.0
        fake_trained_model.predict.return_value = np.array([1, 0])

        fake_metric_artifact = ClassificationMetricArtifact(
            f1_score=0.1,
            precision_score=0.1,
            recall_score=0.1,
        )

        with patch(
            "src.components.model_trainer.load_numpy_array_data"
        ) as mock_load_array, patch.object(
            ModelTrainer, "get_model_object_and_report"
        ) as mock_get_model_and_metrics, patch(
            "src.components.model_trainer.load_object"
        ) as mock_load_object:

            mock_load_array.side_effect = [train_arr, test_arr]
            mock_get_model_and_metrics.return_value = (
                fake_trained_model,
                fake_metric_artifact,
            )
            mock_load_object.return_value = MagicMock()

            with pytest.raises(RuntimeError):
                trainer.initiate_model_trainer()
