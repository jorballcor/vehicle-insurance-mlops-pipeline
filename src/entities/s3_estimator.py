from pandas import DataFrame

from src.logger import log
from src.cloud_storage.aws_storage import SimpleStorageService
from src.entities.estimator import MyModel


class VehicleInsuranceEstimator:
    """
    Wrapper class for loading, saving, and running predictions using a model 
    stored in an AWS S3 bucket.
    """

    def __init__(self, bucket_name: str, model_path: str):
        """
        Args:
            bucket_name (str): Name of the S3 bucket where the model is stored.
            model_path (str): Path/key inside the bucket for the model artifact.
        """
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = SimpleStorageService()
        self.loaded_model: MyModel | None = None

    def is_model_present(self, model_path: str) -> bool:
        """
        Check whether the given model exists in the S3 bucket.

        Args:
            model_path (str): S3 key for the model.

        Returns:
            bool: True if the model exists, False otherwise.
        """
        try:
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name,
                s3_key=model_path,
            )
        except Exception as e:
            log.error(f"Error checking model existence in S3: {e}")
            return False

    def load_model(self) -> MyModel:
        """
        Load the model from S3.

        Returns:
            MyModel: The loaded model object.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        try:
            log.info(f"Loading model from S3: bucket={self.bucket_name}, key={self.model_path}")
            return self.s3.load_model(
                s3_key=self.model_path,
                bucket_name=self.bucket_name,
            )
        except Exception as e:
            msg = f"Failed to load model from S3: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    def save_model(self, from_file: str, remove: bool = False) -> None:
        """
        Upload a model file to S3.

        Args:
            from_file (str): Local path to the model file.
            remove (bool): If True, delete the local file after upload.
        """
        try:
            log.info(f"Uploading model to S3: {self.model_path}")
            self.s3.upload_file(
                from_file=from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove,
            )
        except Exception as e:
            msg = f"Failed to upload model to S3: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    def predict(self, dataframe: DataFrame):
        """
        Run prediction using the underlying MyModel wrapper.

        Args:
            dataframe (pd.DataFrame): Input features.

        Returns:
            Any: Prediction results from the underlying model.
        """
        try:
            if self.loaded_model is None:
                log.info("Model not loaded in memory. Loading now from S3...")
                self.loaded_model = self.load_model()

            return self.loaded_model.predict(dataframe=dataframe)

        except Exception as e:
            msg = f"Error during prediction via VehicleInsuranceEstimator: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e
