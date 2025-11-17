import pymongo
import certifi

from typing import Optional

from src.logger import log
from src.config.settings import get_settings


# Load the certificate authority file to avoid timeout errors when connecting to MongoDB
ca = certifi.where()

class MongoDBClient:
    """
    MongoDBClient is responsible for establishing a connection to the MongoDB database.

    Attributes:
    ----------
    client : MongoClient
        A shared MongoClient instance for the class.
    database : Database
        The specific database instance that MongoDBClient connects to.

    Methods:
    -------
    __init__(database_name: str) -> None
        Initializes the MongoDB connection using the given database name.
    """

    client = None  # Shared MongoClient instance across all MongoDBClient instances

    def __init__(self, database_name: Optional[str] = None) -> None:
        """
        Initializes a connection to the MongoDB database. If no existing connection is found, it establishes a new one.

        Parameters:
        ----------
        database_name : str, optional
            Name of the MongoDB database to connect to. Default is set by DATABASE_NAME constant.

        Raises:
        ------
        Exception
            If there is an issue connecting to MongoDB or if the environment variable for the MongoDB URL is not set.
        """
    
        settings = get_settings()
        
        # Check if a MongoDB client connection has already been established; if not, create a new one
        mongo_url = settings.mongo.url.get_secret_value() if settings.mongo.url else None
        if not mongo_url:
            msg = (
                "Mongo URL not configured. "
                "Set MONGODB_CONNECTION_URL in your environment."
            )
            log.error(msg)
            raise Exception(msg)
                
        db_name = database_name or settings.mongo.database_name

        try:
            if MongoDBClient.client is None:
                log.info("Creating new MongoClient to %s", mongo_url)
                MongoDBClient.client = pymongo.MongoClient(mongo_url, tlsCAFile=ca)

            self.client = MongoDBClient.client
            self.database = self.client[db_name]
            self.database_name = db_name

            log.info("MongoDB connection successful. database=%s", db_name)
            
        except Exception as e:
            log.error(f"Error connecting to MongoDB: {e}")