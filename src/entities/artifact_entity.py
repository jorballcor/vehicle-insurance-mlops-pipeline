# src/entities/artifact_entity.py
from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field, constr, confloat


# ----------------------------
# Base artifact model
# ----------------------------
class Artifact(BaseModel):
    """Base class for all artifact entities."""
    model_config = {
        "frozen": True,       # makes them immutable (like dataclasses)
        "extra": "ignore",    # ignore unexpected keys
        "str_strip_whitespace": True,
    }


# ----------------------------
# Data Ingestion
# ----------------------------
class DataIngestionArtifact(Artifact):
    """Class containing artifacts' paths produced by data ingestion component.

    Args:
        Artifact (BaseModel): Base artifact model.
    """
    trained_file_path: Path = Field(..., description="Path to the training split CSV file")
    test_file_path: Path = Field(..., description="Path to the testing split CSV file")


# ----------------------------
# Data Validation
# ----------------------------
class DataValidationArtifact(Artifact):
    validation_status: bool
    message: constr(strip_whitespace=True, min_length=1)
    validation_report_file_path: Path
    

# ----------------------------
# Data Transformation
# ----------------------------
class DataTransformationArtifact(Artifact):
    transformed_object_file_path: Path
    transformed_train_file_path: Path
    transformed_test_file_path: Path
    

# ----------------------------
# Model Metrics
# ----------------------------
class ClassificationMetricArtifact(Artifact):
    f1_score: confloat(ge=0, le=1)
    precision_score: confloat(ge=0, le=1)
    recall_score: confloat(ge=0, le=1)


# ----------------------------
# Model Trainer
# ----------------------------
class ModelTrainerArtifact(Artifact):
    trained_model_file_path: Path
    metric_artifact: ClassificationMetricArtifact