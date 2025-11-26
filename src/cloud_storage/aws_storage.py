from io import StringIO
from typing import Union, List
import os
import pickle

from botocore.exceptions import ClientError
from mypy_boto3_s3.service_resource import Bucket
from pandas import DataFrame, read_csv

from src.config.aws_config import S3Client
from src.logger import log


class SimpleStorageService:
    """
    High-level wrapper around AWS S3 operations using boto3.

    This service exposes utilities for:
      - Checking key existence.
      - Reading objects (raw or as CSV/DataFrame).
      - Uploading files and DataFrames.
      - Loading serialized models from S3.
    """

    def __init__(self):
        """
        Initialize SimpleStorageService using shared S3Client (resource + client).
        """
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        """
        Check if a given S3 key exists within a bucket.

        Args:
            bucket_name (str): Name of the S3 bucket.
            s3_key (str): Key path of the file to check.

        Returns:
            bool: True if at least one object with that prefix exists, False otherwise.
        """
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [obj for obj in bucket.objects.filter(Prefix=s3_key)]
            return len(file_objects) > 0
        except Exception as e:
            msg = f"Error while checking S3 key '{s3_key}' in bucket '{bucket_name}': {e}"
            log.error(msg)
            # We treat errors as "not available" to avoid breaking callers that interpret False.
            return False

    @staticmethod
    def read_object(
        object_name: object,
        decode: bool = True,
        make_readable: bool = False,
    ) -> Union[StringIO, str, bytes]:
        """
        Read the content of an S3 object.

        Args:
            object_name (object): Boto3 S3 Object to read.
            decode (bool): If True, decode bytes to string using UTF-8.
            make_readable (bool): If True, wrap the content in a StringIO object
                (useful for pandas.read_csv).

        Returns:
            Union[StringIO, str, bytes]: Content as StringIO, decoded string, or raw bytes.
        """
        try:
            raw_bytes = object_name.get()["Body"].read()

            if decode:
                content = raw_bytes.decode()
            else:
                content = raw_bytes

            if make_readable:
                return StringIO(content)

            return content

        except Exception as e:
            msg = f"Error while reading S3 object: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    def get_bucket(self, bucket_name: str) -> Bucket:
        """
        Retrieve a boto3 Bucket instance for the given bucket name.

        Args:
            bucket_name (str): S3 bucket name.

        Returns:
            Bucket: Boto3 S3 Bucket object.
        """
        log.info("Entered get_bucket method of SimpleStorageService")
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            log.info("Exited get_bucket method of SimpleStorageService")
            return bucket
        except Exception as e:
            msg = f"Error while getting bucket '{bucket_name}': {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object]:
        """
        Retrieve one or more S3 objects whose keys start with `filename`.

        Args:
            filename (str): Key prefix to look for.
            bucket_name (str): Name of the S3 bucket.

        Returns:
            Union[List[object], object]: Single S3 Object or list of Objects.
        """
        log.info("Entered get_file_object method of SimpleStorageService")
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [obj for obj in bucket.objects.filter(Prefix=filename)]

            file_objs = file_objects[0] if len(file_objects) == 1 else file_objects
            log.info("Exited get_file_object method of SimpleStorageService")
            return file_objs
        except Exception as e:
            msg = f"Error while getting file object '{filename}' from bucket '{bucket_name}': {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    def load_model(self, s3_key: str, bucket_name: str) -> object:
        """
        Load a serialized model stored in S3 using pickle.

        Args:
            s3_key (str): Full S3 key/path of the model file.
            bucket_name (str): Name of the S3 bucket.

        Returns:
            object: Deserialized model.
        """
        try:
            file_object = self.get_file_object(s3_key, bucket_name)
            model_bytes = self.read_object(file_object, decode=False)
            model = pickle.loads(model_bytes)
            log.info(f"Model loaded from S3: bucket={bucket_name}, key={s3_key}")
            return model
        except Exception as e:
            msg = f"Error while loading model from S3 (bucket={bucket_name}, key={s3_key}): {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        """
        Create a "folder" (prefix) in the given S3 bucket if it does not exist.

        Args:
            folder_name (str): Name of the folder/prefix.
            bucket_name (str): Name of the S3 bucket.
        """
        log.info("Entered create_folder method of SimpleStorageService")
        try:
            # Check if folder exists by attempting to load it
            self.s3_resource.Object(bucket_name, folder_name).load()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                folder_obj = folder_name.rstrip("/") + "/"
                self.s3_client.put_object(Bucket=bucket_name, Key=folder_obj)
                log.info(f"Created S3 folder '{folder_obj}' in bucket '{bucket_name}'")
        finally:
            log.info("Exited create_folder method of SimpleStorageService")

    def upload_file(
        self,
        from_file: str,
        to_filename: str,
        bucket_name: str,
        remove: bool = True,
    ) -> None:
        """
        Upload a local file to S3 and optionally remove the local file afterwards.

        Args:
            from_file (str): Path to the local file.
            to_filename (str): Target key/path in the S3 bucket.
            bucket_name (str): Name of the S3 bucket.
            remove (bool): Whether to remove the local file after successful upload.
        """
        log.info("Entered upload_file method of SimpleStorageService")
        try:
            log.info(f"Uploading '{from_file}' to 's3://{bucket_name}/{to_filename}'")
            self.s3_resource.meta.client.upload_file(from_file, bucket_name, to_filename)
            log.info(f"Uploaded '{from_file}' to 's3://{bucket_name}/{to_filename}'")

            if remove:
                os.remove(from_file)
                log.info(f"Removed local file '{from_file}' after upload")
        except Exception as e:
            msg = f"Error while uploading file '{from_file}' to S3: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e
        finally:
            log.info("Exited upload_file method of SimpleStorageService")

    def upload_df_as_csv(
        self,
        data_frame: DataFrame,
        local_filename: str,
        bucket_filename: str,
        bucket_name: str,
    ) -> None:
        """
        Save a DataFrame to a local CSV file and upload it to S3.

        Args:
            data_frame (DataFrame): Data to upload.
            local_filename (str): Local temporary CSV filename.
            bucket_filename (str): Target key/path in the S3 bucket.
            bucket_name (str): Name of the S3 bucket.
        """
        log.info("Entered upload_df_as_csv method of SimpleStorageService")
        try:
            data_frame.to_csv(local_filename, index=False, header=True)
            self.upload_file(
                from_file=local_filename,
                to_filename=bucket_filename,
                bucket_name=bucket_name,
            )
        except Exception as e:
            msg = f"Error while uploading DataFrame as CSV to S3: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e
        finally:
            log.info("Exited upload_df_as_csv method of SimpleStorageService")

    def get_df_from_object(self, object_: object) -> DataFrame:
        """
        Convert an S3 object containing CSV data into a pandas DataFrame.

        Args:
            object_ (object): Boto3 S3 Object or similar.

        Returns:
            DataFrame: Parsed DataFrame.
        """
        log.info("Entered get_df_from_object method of SimpleStorageService")
        try:
            content = self.read_object(object_, make_readable=True)
            df = read_csv(content, na_values="na")
            log.info("Exited get_df_from_object method of SimpleStorageService")
            return df
        except Exception as e:
            msg = f"Error while converting S3 object to DataFrame: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:
        """
        Read a CSV file from S3 into a pandas DataFrame.

        Args:
            filename (str): S3 key/path of the CSV file.
            bucket_name (str): Name of the S3 bucket.

        Returns:
            DataFrame: Parsed DataFrame.
        """
        log.info("Entered read_csv method of SimpleStorageService")
        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            df = self.get_df_from_object(csv_obj)
            log.info("Exited read_csv method of SimpleStorageService")
            return df
        except Exception as e:
            msg = f"Error while reading CSV '{filename}' from bucket '{bucket_name}': {e}"
            log.error(msg)
            raise RuntimeError(msg) from e
