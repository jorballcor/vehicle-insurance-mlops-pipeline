from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import pandas as pd
from pandas import DataFrame

from src.logger import log
from src.utils.file_utils import read_yaml_file
from src.entities.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entities.config_entities import DataValidationConfig
from src.config.settings import get_settings


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ) -> None:
        """
        :param data_ingestion_artifact: artifact de la etapa de ingestion (rutas train/test)
        :param data_validation_config: configuración por-run de data validation
        """
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        
        settings = get_settings()
        self._schema_path: Path = settings.paths.schema_file_path

        try:
            self._schema_config = read_yaml_file(file_path=self._schema_path)
        except FileNotFoundError as e:
            msg = f"Schema file not found at path: {self._schema_path}"
            log.error(msg)
            raise FileNotFoundError(msg) from e
        except Exception as e:
            msg = f"Unexpected error loading schema from '{self._schema_path}': {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Valida que el número de columnas del dataframe
        coincide con el definido en el schema.
        """
        try:
            required_columns = self._schema_config.get("columns", {})
            if required_columns is None:
                log.error("Schema config is missing 'columns' definition.")
                raise KeyError("Missing 'columns' key in schema config.")

            status = len(dataframe.columns) == len(required_columns)
            log.info("Is required column count correct: [%s]", status)
            return status

        except Exception as e:
            msg = f"Error while validating number of columns: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e


    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Valida la existencia de columnas numéricas y categóricas
        definidas en el schema. Devuelve True si todas existen.
        """
        try:
            dataframe_columns = df.columns

            numerical_columns = self._schema_config.get("numerical_columns", [])
            categorical_columns = self._schema_config.get("categorical_columns", [])

            missing_numerical_columns: list[str] = []
            missing_categorical_columns: list[str] = []

            for column in numerical_columns:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if missing_numerical_columns:
                log.info("Missing numerical columns: %s", missing_numerical_columns)

            for column in categorical_columns:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if missing_categorical_columns:
                log.info("Missing categorical columns: %s", missing_categorical_columns)

            return not (missing_categorical_columns or missing_numerical_columns)

        except Exception as e:
            msg = f"Error while checking column existence: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e


    @staticmethod
    def read_data(file_path: Union[str, Path]) -> DataFrame:
        """
        Lee un CSV en un DataFrame de pandas.
        """
        path = Path(file_path)
        try:
            return pd.read_csv(path)
        except FileNotFoundError as e:
            msg = f"Data file not found: {path}"
            log.error(msg)
            raise FileNotFoundError(msg) from e
        except Exception as e:
            msg = f"Unexpected error reading data from '{path}': {e}"
            log.error(msg)
            raise RuntimeError(msg) from e


    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Lanza el proceso de data validation.

        Devuelve:
            DataValidationArtifact con status, mensaje y ruta del informe.
        """
        try:
            validation_error_msg = ""

            log.info("Starting data validation")

            train_df = DataValidation.read_data(
                file_path=self.data_ingestion_artifact.trained_file_path
            )
            test_df = DataValidation.read_data(
                file_path=self.data_ingestion_artifact.test_file_path
            )

            status = self.validate_number_of_columns(dataframe=train_df)
            if not status:
                validation_error_msg += (
                    "Columns are missing or count mismatch in training dataframe. "
                )
            else:
                log.info(
                    "All required columns present in training dataframe: %s", status
                )

            status = self.validate_number_of_columns(dataframe=test_df)
            if not status:
                validation_error_msg += (
                    "Columns are missing or count mismatch in test dataframe. "
                )
            else:
                log.info(
                    "All required columns present in testing dataframe: %s", status
                )

            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += (
                    "Required numerical/categorical columns are missing in "
                    "training dataframe. "
                )
            else:
                log.info(
                    "All categorical/int columns present in training dataframe: %s",
                    status,
                )

            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += (
                    "Required numerical/categorical columns are missing in "
                    "test dataframe. "
                )
            else:
                log.info(
                    "All categorical/int columns present in testing dataframe: %s",
                    status,
                )

            if validation_error_msg:
                validation_status = False
                final_message = validation_error_msg.strip()
            else:
                validation_status = True
                final_message = "Data validation completed successfully."

            report_path: Path = self.data_validation_config.validation_report_file_path

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=final_message,
                validation_report_file_path=report_path,
            )

            report_path.parent.mkdir(parents=True, exist_ok=True)

            validation_report = {
                "validation_status": validation_status,
                "message": final_message,
            }

            with report_path.open("w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            log.info("Data validation artifact created and saved to JSON file.")
            log.info("Data validation artifact: %s", data_validation_artifact)

            return data_validation_artifact

        except Exception as e:
            msg = f"Error during data validation: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e


