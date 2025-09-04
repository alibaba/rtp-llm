import logging
from typing import Optional

from concurrent_log_handler import ConcurrentRotatingFileHandler

from rtp_llm.config.py_config_modules import StaticConfig

LOG_PATH_KEY = "LOG_PATH"


def get_handler(file_name: str) -> Optional[logging.Handler]:
    log_path = StaticConfig.profiling_debug_config.log_path
    if log_path == "":
        return None
    else:
        return ConcurrentRotatingFileHandler(
            filename=f"{log_path}/{file_name}",
            mode="a",
            maxBytes=100 * 1024 * 1024,
            backupCount=StaticConfig.profiling_debug_config.log_file_backup_count,
            use_gzip=True,
        )
