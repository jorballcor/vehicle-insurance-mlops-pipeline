from src.logger import get_logger

log = get_logger(__name__)


def _extract_file_line(exc: Exception) -> tuple[str, int]:
    tb = getattr(exc, "__traceback__", None)
    if tb is None:
        return "<unknown>", -1
    last_tb = tb
    while last_tb.tb_next:
        last_tb = last_tb.tb_next
    return last_tb.tb_frame.f_code.co_filename, last_tb.tb_lineno


def error_message_detail(exc: Exception) -> str:
    file_name, line_number = _extract_file_line(exc)
    return (
        f"Error occurred in python script: [{file_name}] "
        f"at line number [{line_number}]: {exc}"
    )


class MyException(Exception):
    """
    Custom exception that preserves original traceback for the message,
    but does NOT log by itself to avoid duplication. Log at the catch site.
    """
    def __init__(self, message_or_exc, original_error: Exception | None = None):
        if isinstance(message_or_exc, Exception) and original_error is None:
            cause = message_or_exc
            message = str(message_or_exc)
        else:
            message = str(message_or_exc)
            cause = original_error

        super().__init__(message)
        self.__cause__ = cause  # keep chaining
        exc_for_detail = cause or self
        self.error_message = error_message_detail(exc_for_detail)

    def __str__(self) -> str:
        return self.error_message
