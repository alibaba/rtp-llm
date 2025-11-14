from typing import Dict, Any
import os
import logging
import logging.config


def get_logging_config(log_path: str, world_rank: int) -> Dict[str, Any]:
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
                "filename": f"{log_path}/main_{world_rank}.log",
                "maxBytes": 256 * 1024 * 1024,
                "backupCount": 20,
            },
            "route_file_handler": {
                "formatter": "default",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": f"{log_path}/route.log",
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

def get_log_path() -> str:
    """Get log path from environment variable.
    
    Returns:
        Log path string, default is "logs".
    """
    log_path = os.environ.get("LOG_PATH", "logs")
    os.makedirs(log_path, exist_ok=True)
    return log_path


def get_log_level() -> str:
    """Get log level from environment variable.
    
    Returns:
        Log level string, default is "INFO". "TRACE" is converted to "DEBUG".
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    if log_level == "TRACE":
        log_level = "DEBUG"
    return log_level


def setup_logging():
    LOGLEVEL = get_log_level()
    logging.basicConfig(
        level=LOGLEVEL,
        format="[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    LOG_PATH = get_log_path()

    # nccl log file path
    if os.environ.get("NCCL_DEBUG_FILE") is None:
        os.environ["NCCL_DEBUG_FILE"] = os.path.join(LOG_PATH, "nccl.log")
        logging.info(f"successfully set NCCL_DEBUG_FILE path to {os.environ['NCCL_DEBUG_FILE']}")

    # FT_SERVER_TEST is used for test, we don't need to setup logging for test
    if os.environ.get("FT_SERVER_TEST") == '1':
        return

    WORLD_RANK = int(os.environ.get('WORLD_RANK', '0'))

    LOGGING_CONFIG = get_logging_config(LOG_PATH, WORLD_RANK)
    LOGGING_CONFIG["loggers"][""]["level"] = LOGLEVEL
    logging.config.dictConfig(LOGGING_CONFIG)


