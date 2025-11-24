import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.logger import log


class TargetValueMapping:
    """Maps categorical target values to integers and provides reverse mapping."""

    def __init__(self):
        self.yes: int = 0
        self.no: int = 1

    def _asdict(self) -> dict:
        """Return internal mapping as a dictionary."""
        return self.__dict__

    def reverse_mapping(self) -> dict:
        """Return a dictionary mapping integer values back to their original labels."""
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class MyModel:
    """
    Wrapper class combining a preprocessing pipeline and a trained model.

    This class ensures:
    - Incoming data is transformed using the same preprocessing pipeline used during training.
    - Predictions are generated consistently using the trained model.
    """

    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        Args:
            preprocessing_object (Pipeline): Preprocessing pipeline used during training.
            trained_model_object (object): Trained machine learning model.
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        """
        Run preprocessing and prediction on the provided dataframe.

        Args:
            dataframe (pd.DataFrame): Input features (already cleaned and with correct structure).

        Returns:
            dataframe (pd.DataFrame): Model predictions.
        """
        try:
            log.info("Starting prediction process.")

            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame.")

            # Apply preprocessing
            log.info("Applying preprocessing transformations.")
            transformed_features = self.preprocessing_object.transform(dataframe)

            # Perform prediction
            log.info("Running model.predict()")
            predictions = self.trained_model_object.predict(transformed_features)

            return predictions

        except Exception as e:
            msg = f"Error during prediction: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
