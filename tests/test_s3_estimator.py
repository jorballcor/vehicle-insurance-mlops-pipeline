from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.entities.s3_estimator import VehicleInsuranceEstimator
from src.entities.estimator import MyModel


BUCKET = "test-bucket"
MODEL_KEY = "models/model.pkl"


@pytest.mark.unit
def test_is_model_present_true():
    with patch("src.entities.s3_estimator.SimpleStorageService") as MockStorage:
        storage_instance = MockStorage.return_value
        storage_instance.s3_key_path_available.return_value = True

        est = VehicleInsuranceEstimator(bucket_name=BUCKET, model_path=MODEL_KEY)
        assert est.is_model_present(MODEL_KEY) is True

        storage_instance.s3_key_path_available.assert_called_once_with(
            bucket_name=BUCKET,
            s3_key=MODEL_KEY,
        )


@pytest.mark.unit
def test_is_model_present_false_on_exception():
    with patch("src.entities.s3_estimator.SimpleStorageService") as MockStorage:
        storage_instance = MockStorage.return_value
        storage_instance.s3_key_path_available.side_effect = Exception("boom")

        est = VehicleInsuranceEstimator(bucket_name=BUCKET, model_path=MODEL_KEY)
        # En nuestro refactor devolvemos False si hay error
        assert est.is_model_present(MODEL_KEY) is False


@pytest.mark.unit
def test_load_model_returns_model():
    fake_model = MyModel(preprocessing_object=MagicMock(), trained_model_object=MagicMock())

    with patch("src.entities.s3_estimator.SimpleStorageService") as MockStorage:
        storage_instance = MockStorage.return_value
        storage_instance.load_model.return_value = fake_model

        est = VehicleInsuranceEstimator(bucket_name=BUCKET, model_path=MODEL_KEY)
        model = est.load_model()

        storage_instance.load_model.assert_called_once_with(
            s3_key=MODEL_KEY,
            bucket_name=BUCKET,
        )
        assert model is fake_model


@pytest.mark.unit
def test_save_model_calls_upload_file_with_correct_args():
    with patch("src.entities.s3_estimator.SimpleStorageService") as MockStorage:
        storage_instance = MockStorage.return_value

        est = VehicleInsuranceEstimator(bucket_name=BUCKET, model_path=MODEL_KEY)
        est.save_model(from_file="local_model.pkl", remove=True)

        storage_instance.upload_file.assert_called_once_with(
            from_file="local_model.pkl",
            to_filename=MODEL_KEY,
            bucket_name=BUCKET,
            remove=True,
        )


@pytest.mark.unit
def test_predict_loads_model_if_not_cached_and_calls_predict():
    with patch("src.entities.s3_estimator.SimpleStorageService") as MockStorage:
        storage_instance = MockStorage.return_value

        fake_model = MagicMock()
        fake_model.predict.return_value = [1]
        storage_instance.load_model.return_value = fake_model

        est = VehicleInsuranceEstimator(bucket_name=BUCKET, model_path=MODEL_KEY)

        df = pd.DataFrame({"x": [1]})
        preds = est.predict(df)

        storage_instance.load_model.assert_called_once()
        fake_model.predict.assert_called_once_with(dataframe=df)
        assert preds == [1]
