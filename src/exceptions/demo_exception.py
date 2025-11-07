# # below code is to check the exception config
from src.logger import get_logger
from src.exceptions import MyException

log = get_logger()

try:
    a = 1+'Z'
except Exception as e:
    log.exception(error= str(e))
    #raise MyException(e) from e