import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.config.settings import get_settings
from src.entities.config_entities import DataTransformationConfig
from src.entities.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.logger import log
from src.utils.file_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file,
)


settings = get_settings()
TARGET_COLUMN = settings.target_column
SCHEMA_FILE_PATH = settings.paths.schema_file_path


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        :param data_validation_artifact: result of data validation stage
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except FileNotFoundError as e:
            msg = f"Schema file not found at path: {SCHEMA_FILE_PATH}"
            log.error(msg)
            raise FileNotFoundError(msg) from e
        except Exception as e:
            msg = f"Unexpected error loading schema from '{SCHEMA_FILE_PATH}': {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Read a CSV file into a pandas DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError as e:
            msg = f"Data file not found: {file_path}"
            log.error(msg)
            raise FileNotFoundError(msg) from e
        except Exception as e:
            msg = f"Unexpected error reading data from '{file_path}': {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data,
        including feature scaling with StandardScaler and MinMaxScaler.
        """
        log.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            log.info("Transformers initialized: StandardScaler and MinMaxScaler")

            # Load schema configurations
            num_features = self._schema_config.get("num_features")
            mm_columns = self._schema_config.get("mm_columns")

            if not isinstance(num_features, list):
                msg = "Schema config 'num_features' must be a list."
                log.error(msg)
                raise ValueError(msg)

            if not isinstance(mm_columns, list):
                msg = "Schema config 'mm_columns' must be a list."
                log.error(msg)
                raise ValueError(msg)

            log.info("Columns loaded from schema for scaling.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features),
                    ("MinMaxScaler", min_max_scaler, mm_columns),
                ],
                remainder="passthrough",  # Leaves other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            log.info("Final preprocessing pipeline ready.")
            log.info("Exited get_data_transformer_object method of DataTransformation class")

            return final_pipeline

        except Exception as e:
            msg = f"Exception occurred in get_data_transformer_object: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    def _map_gender_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map Gender column to 0 for Female and 1 for Male."""
        log.info("Mapping 'Gender' column to binary values")
        if "Gender" not in df.columns:
            msg = "Column 'Gender' not found in DataFrame during gender mapping."
            log.error(msg)
            raise KeyError(msg)

        df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).astype(int)
        return df

    def _create_dummy_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dummy variables for categorical features."""
        log.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename specific columns and ensure integer types for dummy columns."""
        log.info("Renaming specific columns and casting to int")
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
        """Drop the id column (or the configured drop column) if it exists."""
        log.info("Dropping id column (or configured drop column) if present")
        drop_col = self._schema_config.get("drop_columns")
        if drop_col and drop_col in df.columns:
            df = df.drop(drop_col, axis=1)
        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            log.info("Data transformation started.")
            if not self.data_validation_artifact.validation_status:
                msg = f"Data validation failed: {self.data_validation_artifact.message}"
                log.error(msg)
                raise ValueError(msg)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            log.info("Train and test data loaded.")

            # Separate input and target features
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            log.info("Input and target columns defined for train and test data.")

            # Apply custom transformations in specified sequence
            input_feature_train_df = self._map_gender_column(input_feature_train_df)
            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
            input_feature_train_df = self._rename_columns(input_feature_train_df)

            input_feature_test_df = self._map_gender_column(input_feature_test_df)
            input_feature_test_df = self._drop_id_column(input_feature_test_df)
            input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
            input_feature_test_df = self._rename_columns(input_feature_test_df)
            log.info("Custom transformations applied to train and test data.")

            # Get preprocessor
            log.info("Creating data transformer object (preprocessor).")
            preprocessor = self.get_data_transformer_object()
            log.info("Preprocessor object obtained.")

            # Transform data
            log.info("Initializing transformation for training data.")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

            log.info("Initializing transformation for testing data.")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            log.info("Transformation done end-to-end for train and test data.")

            # Handle class imbalance with SMOTEENN
            log.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority")

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            log.info("SMOTEENN applied to train and test data.")

            # Concatenate features and target
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            log.info("Feature-target concatenation done for train and test arrays.")

            # Save artifacts
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )
            log.info("Saved transformation object and transformed numpy arrays.")

            log.info("Data transformation completed successfully.")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

        except Exception as e:
            msg = f"Error during data transformation: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e
