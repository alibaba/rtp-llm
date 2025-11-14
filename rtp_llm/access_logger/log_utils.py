import logging
from typing import Optional

from concurrent_log_handler import ConcurrentRotatingFileHandler

def get_handler(file_name: str, log_path: str, backup_count: int) -> Optional[logging.Handler]:
    if log_path == "":
        return None
    else:
        return ConcurrentRotatingFileHandler(
            filename=f"{log_path}/{file_name}",
            mode="a",
            maxBytes=100 * 1024 * 1024,
            backupCount=backup_count,
            use_gzip=True,
        )
