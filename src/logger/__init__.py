import logging
import os
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime
import structlog

# Constants for log configuration
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of backup log files to keep

# Construct log file path
log_dir_path = os.path.join(from_root(), LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)

_CONFIGURED = False  # evita doble configuración


def configure_logger():
    """
    Configures structlog with a rotating file handler and console output.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Configure standard library logging (backend)
    logging.basicConfig(
        format="%(message)s",
        handlers=[
            RotatingFileHandler(
                log_file_path,
                maxBytes=MAX_LOG_SIZE,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",  # <-- importante
            ),
            logging.StreamHandler()
        ],
        level=logging.DEBUG,
        # force=True  # <-- descomenta si quieres asegurar limpieza de handlers (Py3.8+)
    )

    # Configure structlog processors
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),  # <-- UTC consistente
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),  # <-- crucial
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str = None):
    """
    Get a structlog logger instance.

    Args:
        name: Optional name for the logger.

    Returns:
        A structlog logger instance
    """
    return structlog.get_logger(name)


# Configure the logger (opcional mover al entrypoint)
configure_logger()
