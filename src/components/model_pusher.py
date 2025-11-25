from src.cloud_storage.aws_storage import SimpleStorageService
from src.entities.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entities.config_entities import ModelPusherConfig
from src.entities.s3_estimator import VehicleInsuranceEstimator
from src.logger import log


class ModelPusher:
    """
    Upload the newly trained model to the production S3 bucket.

    This component is called **only when the model evaluator has accepted
    the new model** (i.e., it is better than the current production model).
    """

    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        """
        Args:
            model_evaluation_artifact (ModelEvaluationArtifact):
                Contains trained_model_path and acceptance flag.
            model_pusher_config (ModelPusherConfig):
                Contains S3 bucket info and target model key.
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.config = model_pusher_config

        # Reuse your S3 estimator to manage production model
        self.estimator = VehicleInsuranceEstimator(
            bucket_name=self.config.bucket_name,
            model_path=self.config.s3_model_key_path,
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Upload the newly trained model (if accepted) to S3.

        Returns:
            ModelPusherArtifact: contains bucket name + final S3 model path.
        """
        log.info("Starting Model Pusher component...")

        try:
            trained_model_local_path = self.model_evaluation_artifact.trained_model_path

            log.info(
                f"Uploading new production model to S3: "
                f"{trained_model_local_path} → s3://{self.config.bucket_name}/{self.config.s3_model_key_path}"
            )

            # Upload to S3 through VehicleInsuranceEstimator
            self.estimator.save_model(from_file=trained_model_local_path)

            artifact = ModelPusherArtifact(
                bucket_name=self.config.bucket_name,
                s3_model_path=self.config.s3_model_key_path,
            )

            log.info(f"Model successfully pushed to production: {artifact}")
            return artifact

        except Exception as e:
            msg = f"Error during model pusher stage: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e
