from typing import Any, Dict

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


def configure_uvicorn_access_logging(log_path: str = get_log_path()) -> None:
    """Configure HTTP access logging without closing unrelated async handlers."""
    import logging
    from logging.handlers import RotatingFileHandler

    from uvicorn.logging import AccessFormatter

    config = get_uvicorn_logging_config(log_path)
    handler_options = dict(config["handlers"]["access"])
    handler_options.pop("class")
    handler_options.pop("formatter")
    formatter_options = dict(config["formatters"]["access"])
    formatter_options.pop("()")
    handler = RotatingFileHandler(**handler_options)
    handler.setFormatter(AccessFormatter(**formatter_options))
    logger = logging.getLogger("uvicorn.access")
    for old_handler in list(logger.handlers):
        logger.removeHandler(old_handler)
        old_handler.close()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.disabled = False
