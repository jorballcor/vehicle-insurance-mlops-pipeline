import os
import dill
import yaml
import numpy as np
from pandas import DataFrame

from src.logger import log


def read_yaml_file(file_path: str) -> dict:
    """
    Read a YAML file and return its contents as a dictionary.
    Raises ValueError or FileNotFoundError when appropriate.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in '{file_path}': {e}")

    except Exception as e:
        raise RuntimeError(f"Unexpected error reading YAML file '{file_path}': {e}")


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Write a dictionary or Python object to a YAML file.
    """
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise RuntimeError(f"Error writing YAML file '{file_path}': {e}")


def load_object(file_path: str) -> object:
    """
    Load a serialized Python object saved with dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except FileNotFoundError:
        raise FileNotFoundError(f"Object file not found: {file_path}")

    except dill.UnpicklingError as e:
        raise ValueError(f"Error unpickling object in '{file_path}': {e}")

    except Exception as e:
        raise RuntimeError(f"Unexpected error loading object from '{file_path}': {e}")


def save_numpy_array_data(file_path: str, array: np.ndarray):
    """
    Save a numpy array to disk.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise RuntimeError(f"Error saving numpy array to '{file_path}': {e}")


def load_numpy_array_data(file_path: str) -> np.ndarray | None:
    """
    Load a numpy array from disk.
    - If file does not exist: logs and returns None.
    - If reading fails: logs and returns None.
    """
    if not os.path.exists(file_path):
        log.error(f"Numpy array file not found: {file_path}")
        return None

    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)

    except Exception as e:
        log.error(f"Error loading numpy array from '{file_path}': {e}")
        return None


def save_object(file_path: str, obj: object) -> None:
    """
    Serialize and save a Python object using dill.
    """
    log.info("Entered save_object")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        log.info("Exited save_object successfully")

    except Exception as e:
        raise RuntimeError(f"Error saving object to '{file_path}': {e}")


def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """
    Drop columns from a DataFrame.
    If an error occurs, logs the error and returns the original DataFrame.
    """
    log.info("Entered drop_columns")

    try:
        df = df.drop(columns=cols, axis=1)
        log.info("Exited drop_columns successfully")
        return df

    except Exception as e:
        log.error(f"Error dropping columns {cols}: {e}")
        return df

