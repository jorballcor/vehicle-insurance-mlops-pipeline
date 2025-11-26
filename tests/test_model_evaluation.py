from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
import pytest

from src.config.settings import get_settings
from src.components.model_evaluation import ModelEvaluation, EvaluateModelResponse
from src.entities.config_entities import ModelEvaluationConfig
from src.entities.artifact_entity import (
    DataIngestionArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
    ModelEvaluationArtifact,
)


settings = get_settings()
TARGET_COLUMN = settings.target_column


@pytest.fixture
def test_csv(tmp_path) -> Path:
    """Create a small CSV matching expected raw schema."""
    df = pd.DataFrame(
        {
            "_id": [1, 2, 3, 4],
            "Gender": ["Male", "Female", "Male", "Female"],
            "Age": [30, 40, 50, 60],
            "Driving_License": [1, 1, 1, 1],
            "Region_Code": [10.0, 20.0, 10.0, 30.0],
            "Previously_Insured": [0, 0, 1, 0],
            "Annual_Premium": [30000.0, 40000.0, 35000.0, 45000.0],
            "Policy_Sales_Channel": [1.0, 2.0, 1.0, 3.0],
            "Vintage": [100, 200, 150, 300],
            "Vehicle_Age_< 1 Year": [1, 0, 0, 1],
            "Vehicle_Age_> 2 Years": [0, 1, 1, 0],
            "Vehicle_Damage": ["Yes", "No", "Yes", "No"],
            TARGET_COLUMN: [0, 1, 0, 1],
        }
    )
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def data_ingestion_artifact(test_csv, tmp_path) -> DataIngestionArtifact:
    train_path = tmp_path / "train.csv"
    train_path.write_text("dummy")  # not used in this test
    return DataIngestionArtifact(
        trained_file_path=train_path,
        test_file_path=test_csv,
    )


@pytest.fixture
def model_trainer_artifact(tmp_path) -> ModelTrainerArtifact:
    model_path = tmp_path / "trained_model.pkl"
    model_path.write_bytes(b"fake-model")

    metric = ClassificationMetricArtifact(
        f1_score=0.9,
        precision_score=0.9,
        recall_score=0.9,
    )

    return ModelTrainerArtifact(
        trained_model_file_path=model_path,
        metric_artifact=metric,
    )


@pytest.fixture
def model_eval_config() -> ModelEvaluationConfig:
    return ModelEvaluationConfig(
        changed_threshold_score=0.01,
        bucket_name="test-bucket",
        s3_model_key_path="models/production_model.pkl",
    )


# -------- Tests --------

@pytest.mark.unit
def test_evaluate_model_no_production_model(
    model_eval_config,
    data_ingestion_artifact,
    model_trainer_artifact,
):
    # Patch load_object (we don't really use the loaded trained model in evaluation logic)
    with patch("src.components.model_evaluation.load_object") as mock_load_obj, \
         patch.object(ModelEvaluation, "get_best_model") as mock_get_best_model:

        mock_load_obj.return_value = MagicMock()
        mock_get_best_model.return_value = None  # no production model

        evaluator = ModelEvaluation(
            model_eval_config=model_eval_config,
            data_ingestion_artifact=data_ingestion_artifact,
            model_trainer_artifact=model_trainer_artifact,
        )

        result: EvaluateModelResponse = evaluator.evaluate_model()

    assert isinstance(result, EvaluateModelResponse)
    assert result.trained_model_f1_score == model_trainer_artifact.metric_artifact.f1_score
    assert result.best_model_f1_score is None
    # If there's no best model, tmp_best = 0 → new model should be accepted
    assert result.is_model_accepted is True
    assert result.difference == result.trained_model_f1_score - 0.0


@pytest.mark.unit
def test_evaluate_model_with_production_model_better_or_worse(
    model_eval_config,
    data_ingestion_artifact,
    model_trainer_artifact,
):
    # Simulate a production model with fixed predictions
    fake_best_estimator = MagicMock()
    # y from CSV: [0,1,0,1] ; if we predict [0,1,0,0] F1 will be < 0.9
    fake_best_estimator.predict.return_value = np.array([0, 1, 0, 0])

    with patch("src.components.model_evaluation.load_object") as mock_load_obj, \
         patch.object(ModelEvaluation, "get_best_model") as mock_get_best_model:

        mock_load_obj.return_value = MagicMock()
        mock_get_best_model.return_value = fake_best_estimator

        evaluator = ModelEvaluation(
            model_eval_config=model_eval_config,
            data_ingestion_artifact=data_ingestion_artifact,
            model_trainer_artifact=model_trainer_artifact,
        )

        result: EvaluateModelResponse = evaluator.evaluate_model()

    assert isinstance(result, EvaluateModelResponse)
    assert result.best_model_f1_score is not None
    # Dado que el F1 del modelo nuevo es 0.9 y el del viejo será menor,
    # el nuevo modelo debe ser aceptado.
    assert result.is_model_accepted is True
    assert result.difference == pytest.approx(
        result.trained_model_f1_score - result.best_model_f1_score
    )


@pytest.mark.unit
def test_initiate_model_evaluation_returns_artifact(
    model_eval_config,
    data_ingestion_artifact,
    model_trainer_artifact,
):
    # Patch evaluate_model to control the result
    fake_response = EvaluateModelResponse(
        trained_model_f1_score=0.9,
        best_model_f1_score=0.8,
        is_model_accepted=True,
        difference=0.1,
    )

    with patch.object(ModelEvaluation, "evaluate_model") as mock_eval:
        mock_eval.return_value = fake_response

        evaluator = ModelEvaluation(
            model_eval_config=model_eval_config,
            data_ingestion_artifact=data_ingestion_artifact,
            model_trainer_artifact=model_trainer_artifact,
        )

        artifact: ModelEvaluationArtifact = evaluator.initiate_model_evaluation()

    assert isinstance(artifact, ModelEvaluationArtifact)
    assert artifact.is_model_accepted is True
    assert str(artifact.s3_model_path) == model_eval_config.s3_model_key_path
    assert artifact.trained_model_path == model_trainer_artifact.trained_model_file_path
    assert artifact.changed_accuracy == fake_response.difference
