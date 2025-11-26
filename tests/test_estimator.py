import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.entities.estimator import TargetValueMapping, MyModel


class TestTargetValueMapping:
    @pytest.mark.unit
    def test_target_value_mapping_basic(self):
        mapping = TargetValueMapping()
        assert mapping.yes == 0
        assert mapping.no == 1

        rev = mapping.reverse_mapping()
        assert rev[0] == "yes"
        assert rev[1] == "no"
        assert set(rev.keys()) == {0, 1}


class TestMyModel:
    @pytest.mark.unit
    def test_predict_calls_transform_and_model_predict(self):
        # Arrange
        df = pd.DataFrame({"feature": [1, 2, 3]})
        transformed = np.array([[0.1], [0.2], [0.3]])

        preprocessing = MagicMock()
        preprocessing.transform.return_value = transformed

        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 0])

        my_model = MyModel(
            preprocessing_object=preprocessing,
            trained_model_object=model,
        )

        # Act
        preds = my_model.predict(df)

        # Assert
        preprocessing.transform.assert_called_once_with(df)
        model.predict.assert_called_once_with(transformed)
        assert (preds == np.array([0, 1, 0])).all()

    @pytest.mark.unit
    def test_predict_raises_error_on_non_dataframe(self):
        preprocessing = MagicMock()
        model = MagicMock()
        my_model = MyModel(preprocessing_object=preprocessing, trained_model_object=model)

        with pytest.raises(RuntimeError) as excinfo:
            my_model.predict([1, 2, 3])  # Not a DataFrame

        assert "Input must be a pandas DataFrame" in str(excinfo.value)

