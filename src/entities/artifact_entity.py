# src/entities/artifact_entity.py
from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field


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

