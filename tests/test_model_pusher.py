from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.components.model_pusher import ModelPusher
from src.entities.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entities.config_entities import ModelPusherConfig


@pytest.fixture
def model_evaluation_artifact(tmp_path) -> ModelEvaluationArtifact:
    trained_model_path = tmp_path / "trained_model.pkl"
    trained_model_path.write_bytes(b"fake model")

    return ModelEvaluationArtifact(
        is_model_accepted=True,
        s3_model_path="models/old_model.pkl",
        trained_model_path=trained_model_path,
        changed_accuracy=0.1,
    )


@pytest.fixture
def model_pusher_config() -> ModelPusherConfig:
    return ModelPusherConfig(
        bucket_name="test-bucket",
        s3_model_key_path="models/production_model.pkl",
    )


@pytest.mark.unit
def test_model_pusher_uploads_new_model_and_returns_artifact(
    model_evaluation_artifact,
    model_pusher_config,
):
    with patch("src.components.model_pusher.VehicleInsuranceEstimator") as MockEstimator:
        estimator_instance = MockEstimator.return_value
        estimator_instance.save_model = MagicMock()

        pusher = ModelPusher(
            model_evaluation_artifact=model_evaluation_artifact,
            model_pusher_config=model_pusher_config,
        )

        artifact = pusher.initiate_model_pusher()

        estimator_instance.save_model.assert_called_once_with(
            from_file=model_evaluation_artifact.trained_model_path
        )

        assert isinstance(artifact, ModelPusherArtifact)
        assert artifact.bucket_name == model_pusher_config.bucket_name
        assert str(artifact.s3_model_key_path) == model_pusher_config.s3_model_key_path