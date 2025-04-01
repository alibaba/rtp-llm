
# change uvicorn acess logger handler to file
UVICORN_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(asctime)s.%(msecs)03d %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s', # noqa: E501
            "datefmt": "%Y-%m-%d %H:%M:%S",  # 只包含到秒，毫秒在 fmt 中处理
        },
    },
    "handlers": {
        "access": {
            "formatter": "access",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/uvicorn_access.log",
            "maxBytes": 50 * 1024 * 1024,
            "backupCount": 10,
        },
    },
    "loggers": {
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}
