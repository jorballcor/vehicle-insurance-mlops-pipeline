from pandas import DataFrame

from src.logger import log
from src.entities.config_entities import VehiclePredictorConfig
from src.entities.s3_estimator import Proj1Estimator


class VehicleData:
    """
    Container class for a single vehicle insurance record.

    This class holds all model input features and provides helpers
    to convert them into a pandas DataFrame or a plain dictionary,
    suitable for passing into the prediction pipeline.
    """

    def __init__(
        self,
        Gender,
        Age,
        Driving_License,
        Region_Code,
        Previously_Insured,
        Annual_Premium,
        Policy_Sales_Channel,
        Vintage,
        Vehicle_Age_lt_1_Year,
        Vehicle_Age_gt_2_Years,
        Vehicle_Damage_Yes,
    ):
        """
        Initialize VehicleData with all required model features.

        Args:
            Gender: Encoded gender feature (e.g., 0/1).
            Age: Age of the customer.
            Driving_License: Driving license flag.
            Region_Code: Region code.
            Previously_Insured: Previously insured flag.
            Annual_Premium: Annual premium amount.
            Policy_Sales_Channel: Encoded policy sales channel.
            Vintage: Days since the customer associated with the company.
            Vehicle_Age_lt_1_Year: One-hot or binary flag for vehicle age < 1 year.
            Vehicle_Age_gt_2_Years: One-hot or binary flag for vehicle age > 2 years.
            Vehicle_Damage_Yes: One-hot or binary flag for previous vehicle damage.
        """
        self.Gender = Gender
        self.Age = Age
        self.Driving_License = Driving_License
        self.Region_Code = Region_Code
        self.Previously_Insured = Previously_Insured
        self.Annual_Premium = Annual_Premium
        self.Policy_Sales_Channel = Policy_Sales_Channel
        self.Vintage = Vintage
        self.Vehicle_Age_lt_1_Year = Vehicle_Age_lt_1_Year
        self.Vehicle_Age_gt_2_Years = Vehicle_Age_gt_2_Years
        self.Vehicle_Damage_Yes = Vehicle_Damage_Yes

    def get_vehicle_input_data_frame(self) -> DataFrame:
        """
        Build a pandas DataFrame from the stored vehicle data.

        Returns:
            DataFrame: Single-row DataFrame with all model features.

        Raises:
            RuntimeError: If the DataFrame construction fails for any reason.
        """
        try:
            vehicle_input_dict = self.get_vehicle_data_as_dict()
            return DataFrame(vehicle_input_dict)
        except Exception as e:
            msg = f"Error while converting vehicle data to DataFrame: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    def get_vehicle_data_as_dict(self) -> dict:
        """
        Build a dictionary representation of the stored vehicle data.

        Returns:
            dict: Dictionary mapping feature names to single-element lists.
        """
        log.info("Entered get_vehicle_data_as_dict method of VehicleData class")

        try:
            input_data = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage],
                "Vehicle_Age_lt_1_Year": [self.Vehicle_Age_lt_1_Year],
                "Vehicle_Age_gt_2_Years": [self.Vehicle_Age_gt_2_Years],
                "Vehicle_Damage_Yes": [self.Vehicle_Damage_Yes],
            }

            log.info("Created vehicle data dict successfully")
            log.info("Exited get_vehicle_data_as_dict method of VehicleData class")
            return input_data

        except Exception as e:
            msg = f"Error while creating vehicle data dict: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e


class VehicleDataClassifier:
    """
    Prediction pipeline wrapper that loads the model from S3 (or other storage)
    using VehiclePredictorConfig and exposes a simple predict() method.
    """

    def __init__(
        self,
        prediction_pipeline_config: VehiclePredictorConfig | None = None,
    ) -> None:
        """
        Initialize VehicleDataClassifier with a prediction configuration.

        Args:
            prediction_pipeline_config (VehiclePredictorConfig, optional):
                Configuration that defines where to fetch the model artifact from.
                If None, a default VehiclePredictorConfig instance is used.
        """
        self.prediction_pipeline_config = (
            prediction_pipeline_config or VehiclePredictorConfig()
        )

    def predict(self, dataframe: DataFrame):
        """
        Run prediction on the given DataFrame using the remote/local model.

        Args:
            dataframe (DataFrame): Input features for prediction.

        Returns:
            Any: The prediction output of the underlying estimator (usually array-like or scalar).

        Raises:
            RuntimeError: If prediction fails.
        """
        try:
            log.info("Entered predict method of VehicleDataClassifier class")

            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )

            result = model.predict(dataframe)
            log.info("Prediction successfully generated by VehicleDataClassifier")
            return result

        except Exception as e:
            msg = f"Error during prediction in VehicleDataClassifier: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e
