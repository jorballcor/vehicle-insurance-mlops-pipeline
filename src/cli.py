from src.pipeline.training_pipeline import TrainPipeline
from src.logger import log

def run_training():
    log.info("=== Starting Training Pipeline ===")
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
    log.info("=== Training Pipeline Completed ===")