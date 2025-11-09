# src/components/data_ingestion.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entities.config_entities import TrainingPipelineConfig, DataIngestionConfig
from src.entities.artifact_entity import DataIngestionArtifact
from src.data_access.vehicle_insurance_repository import VehicleInsuranceRepository
from src.logger import log


class DataIngestion:
    def __init__(
        self,
        training_cfg: TrainingPipelineConfig,
        ingestion_cfg: DataIngestionConfig,
        repo: Optional[VehicleInsuranceRepository] = None,
        *,
        target_column: Optional[str] = None,
        split_random_state: int = 42,
        shuffle: bool = True,
    ) -> None:
        """
        Data ingestion built around immutable per-run configs.

        Args:
            training_cfg: run context (artifact_dir, timestamp).
            ingestion_cfg: ingestion paths & knobs.
            repo: repository to read data from Mongo (DI-friendly). If None, one is created.
            target_column: if provided and present, will use stratified split.
            split_random_state: deterministic split seed.
            shuffle: whether to shuffle before splitting.
        """
        self.training_cfg = training_cfg
        self.ingestion_cfg = ingestion_cfg
        self.repo = repo or VehicleInsuranceRepository()
        self.target_column = target_column
        self.split_random_state = split_random_state
        self.shuffle = shuffle

        log.info(
            "DataIngestion initialized | artifact_dir=%s | ingestion_dir=%s | collection=%s",
            str(self.training_cfg.artifact_dir),
            str(self.ingestion_cfg.data_ingestion_dir),
            self.ingestion_cfg.collection_name,
        )

    # --------------------------
    # Internal helpers
    # --------------------------
    @staticmethod
    def _ensure_parent_dir(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def _atomic_to_csv(self, df: DataFrame, path: Path) -> None:
        """Write CSV atomically: write to tmp then replace."""
        self._ensure_parent_dir(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, index=False, header=True)
        tmp.replace(path)

    # --------------------------
    # Public API
    # --------------------------
    def export_data_into_feature_store(self) -> DataFrame:
        """
        Export data from MongoDB (via repository) to the feature store CSV and return the DataFrame.
        """
        try:
            log.info("Exporting data from Mongo collection '%s'...", self.ingestion_cfg.collection_name)

            # VehicleInsuranceRepository already uses settings for DB; pass collection explicitly.
            df = self.repo.export_collection_as_dataframe(
                collection_name=self.ingestion_cfg.collection_name
            )

            if df is None or df.empty:
                raise Exception("Exported dataframe is empty.")

            log.info("Exported dataframe shape: %s", df.shape)

            feature_store_path: Path = self.ingestion_cfg.feature_store_file_path
            log.info("Saving feature store file: %s", str(feature_store_path))
            self._atomic_to_csv(df, feature_store_path)

            return df

        except Exception as e:
            log.error("Failed exporting data into feature store: %s", e)
            raise

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Split the given dataframe into train and test sets using configured ratio.
        Saves both CSVs to their configured locations.
        """
        log.info("Entered split_data_as_train_test")

        try:
            if dataframe is None or dataframe.empty:
                raise Exception("Input dataframe for split is empty.")

            stratify = None
            if self.target_column and self.target_column in dataframe.columns:
                stratify = dataframe[self.target_column]
                log.info("Using stratified split on target column: %s", self.target_column)

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.ingestion_cfg.train_test_split_ratio,
                random_state=self.split_random_state,
                shuffle=self.shuffle,
                stratify=stratify,
            )

            log.info(
                "Split done: train=%s, test=%s (ratio=%.3f)",
                train_set.shape,
                test_set.shape,
                self.ingestion_cfg.train_test_split_ratio,
            )

            self._atomic_to_csv(train_set, self.ingestion_cfg.training_file_path)
            self._atomic_to_csv(test_set, self.ingestion_cfg.testing_file_path)

            log.info(
                "Train/Test CSVs written: %s | %s",
                str(self.ingestion_cfg.training_file_path),
                str(self.ingestion_cfg.testing_file_path),
            )

        except Exception as e:
            log.error("Failed splitting data into train/test: %s", e)
            raise

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Full ingestion routine: export to feature store, then split to train/test.
        Returns a Pydantic artifact with final file paths.
        """
        log.info("Initiating data ingestion...")
        try:
            df = self.export_data_into_feature_store()
            self.split_data_as_train_test(df)

            artifact = DataIngestionArtifact(
                trained_file_path=self.ingestion_cfg.training_file_path,
                test_file_path=self.ingestion_cfg.testing_file_path,
            )
            log.info("Data ingestion artifact: %s", artifact.model_dump())
            return artifact

        except Exception as e:
            log.error("Data ingestion failed: %s", e)
            raise

