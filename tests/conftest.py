# tests/conftest.py
from __future__ import annotations

import pandas as pd
import pytest

# A tiny fake repo that mimics your VehicleInsuranceRepository
class FakeRepo:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def export_collection_as_dataframe(self, collection_name: str) -> pd.DataFrame:
        # collection_name is ignored here; real repo would use it
        return self._df.copy()


@pytest.fixture
def small_df():
    # Small, simple dataframe with an optional target column "Response"
    return pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5, 6],
        "feature_b": [10, 20, 30, 40, 50, 60],
        "Response":  [0, 0, 0, 1, 1, 1],   # balanced 50/50
    })


@pytest.fixture
def imbalanced_df():
    # Imbalanced target to test stratify behavior
    return pd.DataFrame({
        "x": list(range(100)),
        "Response": [1]*10 + [0]*90,  # 10% positive
    })
