from rtp_llm.config.py_config_modules import StaticConfig

## reserve this env
world_rank = StaticConfig.parallelism_distributed_config.world_rank
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
        "route_file_handler": {
            "formatter": "default",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": f"logs/route.log",
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
        "route_logger": {
            "handlers": ["route_file_handler"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
