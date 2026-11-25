import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

from uvicorn.logging import AccessFormatter

from rtp_llm.config.log_config import get_log_path


_ACCESS_LOG_FORMAT = (
    '%(asctime)s.%(msecs)03d %(levelprefix)s %(client_addr)s - '
    '"%(request_line)s" %(status_code)s'
)
_ACCESS_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_uvicorn_logging_config(log_path: Optional[str] = None) -> Dict[str, Any]:
    log_path = get_log_path() if log_path is None else log_path
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": _ACCESS_LOG_FORMAT,
                "datefmt": _ACCESS_LOG_DATE_FORMAT,
            },
        },
        "handlers": {
            "access": {
                "formatter": "access",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": f"{log_path}/uvicorn_access.log",
                "maxBytes": 50 * 1024 * 1024,
                "backupCount": 10,
            },
        },
        "loggers": {
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        },
    }


def configure_uvicorn_logging(log_path: Optional[str] = None) -> None:
    """Configure Uvicorn access logs without closing existing app handlers."""
    log_path = get_log_path() if log_path is None else log_path
    handler = RotatingFileHandler(
        os.path.join(log_path, "uvicorn_access.log"),
        maxBytes=50 * 1024 * 1024,
        backupCount=10,
    )
    handler.setFormatter(
        AccessFormatter(fmt=_ACCESS_LOG_FORMAT, datefmt=_ACCESS_LOG_DATE_FORMAT)
    )
    logger = logging.getLogger("uvicorn.access")
    # dictConfig closes registered handlers, including the app's async sinks.
    for old_handler in logger.handlers[:]:
        logger.removeHandler(old_handler)
        old_handler.close()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.disabled = False
    logger.propagate = False
