
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(name)s][%(asctime)s][%(process)d][%(threadName)s][%(pathname)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "file_handler": {
            "formatter": "default",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/main.log",
            "maxBytes": 256 * 1024 * 1024,
            "backupCount": 20,
        },
    },
    "loggers": {
        "": {
            "handlers": ["file_handler"],
            "level": "INFO",
            "propagate": True,
        },
    },
}
