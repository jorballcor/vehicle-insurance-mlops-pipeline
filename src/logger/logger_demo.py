# below code is to check the logging config
from src.logger import get_logger

log = get_logger()

log.debug("This is a debug message.")
log.info("This is an info message.")
log.warning("This is a warning message.")
log.error("This is an error message.")
log.critical("This is a critical message.")