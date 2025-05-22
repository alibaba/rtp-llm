import os
world_rank = os.environ.get('WORLD_RANK', '0')
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(name)s][%(asctime)s.%(msecs)03d][%(process)d][%(threadName)s][%(pathname)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",  # 只包含到秒，毫秒在 fmt 中处理
        },
    },
    "handlers": {
        "file_handler": {
            "formatter": "default",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": f"logs/main_{world_rank}.log",
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
