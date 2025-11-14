from typing import Dict, Any
import os


def get_logging_config(world_rank: int = 0) -> Dict[str, Any]:
    """Generate logging configuration with specified world_rank.
    
    Args:
        world_rank: The world rank for the log file name.
        
    Returns:
        Dictionary containing logging configuration.
    """
    return {
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

LOGGING_CONFIG = get_logging_config(int(os.environ.get('WORLD_RANK', '0')))
