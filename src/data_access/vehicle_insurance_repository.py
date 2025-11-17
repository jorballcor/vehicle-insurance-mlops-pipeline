import pandas as pd
import numpy as np
from typing import Optional, Iterable, Dict, Any

from src.config.mongo_db_connection import MongoDBClient
from src.config.settings import get_settings
from src.logger import log
from src.exceptions import MyException


class VehicleInsuranceRepository:
    """
    A class to export MongoDB records as a pandas DataFrame.
    """

    def __init__(self) -> None:
        """
        Initializes the MongoDB client connection.
        """
        settings = get_settings()
        database_name = settings.mongo.database_name
        
        try:
            self.mongo_client = MongoDBClient(database_name=database_name)
            
        except Exception as e:
            log.error("Error initializing MongoDBClient: %s", e)
            raise

    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Exports an entire MongoDB collection as a pandas DataFrame.

        Parameters:
        ----------
        collection_name : str
            The name of the MongoDB collection to export.
        database_name : Optional[str]
            Name of the database (optional). Defaults to DATABASE_NAME.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the collection data, with '_id' column removed and 'na' values replaced with NaN.
        """
        try:
            # Access specified collection from the default or specified database
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
                
            log.info("Fetching data from MongoDB collection='%s'", collection_name)
            docs: Iterable[Dict[str, Any]] = list(collection.find())
            log.info("Fetched %d documents from MongoDB", len(docs))

            if not docs:
                # Return empty DataFrame if no documents found, DataIngestion will handle this case
                return pd.DataFrame()

            df = pd.DataFrame(docs)
            if "id" in df.columns:
                df = df.drop(columns=["id"], axis=1)
            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            log.error(f"Error exporting collection '{collection_name}' as DataFrame: {e}")
            # I raise the exception, so DataIngestion see the real error
            raise
        
        