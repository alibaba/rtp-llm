import logging
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

from rtp_llm.config.log_config import get_log_path


def get_uvicorn_logging_config(log_path: str = get_log_path()) -> Dict[str, Any]:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(asctime)s.%(msecs)03d %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',  # noqa: E501
                "datefmt": "%Y-%m-%d %H:%M:%S",  # 只包含到秒，毫秒在 fmt 中处理
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
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }


def configure_uvicorn_access_logger(log_path: Optional[str] = None) -> None:
    """Configure HTTP access logging without closing other service handlers."""
    from uvicorn.logging import AccessFormatter

    config = get_uvicorn_logging_config(
        get_log_path() if log_path is None else log_path
    )
    handler_config = config["handlers"]["access"]
    formatter_config = config["formatters"]["access"]
    handler = RotatingFileHandler(
        filename=handler_config["filename"],
        maxBytes=handler_config["maxBytes"],
        backupCount=handler_config["backupCount"],
    )
    handler.setFormatter(
        AccessFormatter(
            fmt=formatter_config["fmt"], datefmt=formatter_config["datefmt"]
        )
    )
    # dictConfig closes every registered handler, including the MM async
    # handlers created before HTTP startup. Only replace Uvicorn's handler.
    logger = logging.getLogger("uvicorn.access")
    for old_handler in logger.handlers[:]:
        logger.removeHandler(old_handler)
        old_handler.close()
    logger.addHandler(handler)
    logger.setLevel(config["loggers"]["uvicorn.access"]["level"])
    logger.propagate = False
    logger.disabled = False
