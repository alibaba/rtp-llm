import logging
import os

from concurrent_log_handler import ConcurrentRotatingFileHandler
from typing import Optional

LOG_PATH_KEY='LOG_PATH'

def get_handler(file_name: str) -> Optional[logging.Handler]:
    log_path = os.getenv(LOG_PATH_KEY, None)
    if log_path is None:
        return None
    else:
        return ConcurrentRotatingFileHandler(
            filename=f'{log_path}/{file_name}',
            mode='a',
            maxBytes=10 * 1024 * 1024,
            backupCount=int(os.environ.get('LOG_FILE_BACKUP_COUNT', 16))
        )
