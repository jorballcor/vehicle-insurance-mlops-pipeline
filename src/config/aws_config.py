import os
import boto3

from src.logger import log
from src.config.settings import get_settings


settings = get_settings()
REGION_NAME = settings.aws.region_name
AWS_ACCESS_KEY_ID = settings.aws.access_key_id
AWS_SECRET_ACCESS_KEY = settings.aws.secret_access_key


class S3Client:
    """
    Wrapper for boto3 S3 client and resource.

    This class:
      - Loads AWS credentials from environment variables.
      - Instantiates boto3 S3 client and resource only once (class-level caching).
      - Raises clear, standard Python errors when credentials are missing.
    """

    s3_client = None
    s3_resource = None

    def __init__(self, region_name: str = REGION_NAME):
        """
        Initialize S3Client and ensure boto3 objects exist.

        Args:
            region_name (str): AWS region for S3 operations.

        Raises:
            EnvironmentError: If AWS credentials are missing.
            RuntimeError: If boto3 fails to initialize.
        """

        # Initialize only once (singleton-like behavior)
        if S3Client.s3_resource is None or S3Client.s3_client is None:
            access_key_id = AWS_ACCESS_KEY_ID
            secret_access_key = AWS_SECRET_ACCESS_KEY

            if not access_key_id:
                msg = f"Missing AWS environment variable: {AWS_ACCESS_KEY_ID}"
                log.error(msg)
                raise EnvironmentError(msg)

            if not secret_access_key:
                msg = f"Missing AWS environment variable: {AWS_SECRET_ACCESS_KEY}"
                log.error(msg)
                raise EnvironmentError(msg)

            try:
                log.info("Initializing boto3 S3 client and resource...")

                S3Client.s3_resource = boto3.resource(
                    "s3",
                    aws_access_key_id=access_key_id,
                    aws_secret_access_key=secret_access_key,
                    region_name=region_name,
                )

                S3Client.s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=access_key_id,
                    aws_secret_access_key=secret_access_key,
                    region_name=region_name,
                )

                log.info("Successfully initialized boto3 S3 client and resource.")

            except Exception as e:
                msg = f"Failed to initialize boto3 S3 client/resource: {e}"
                log.error(msg)
                raise RuntimeError(msg) from e

        # Assign to instance
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client
