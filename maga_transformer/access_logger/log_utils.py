import logging
import os

from logging.handlers import RotatingFileHandler
from typing import Optional

LOG_PATH_KEY='PY_LOG_PATH'

def get_handler(file_name: str) -> Optional[logging.Handler]:
    log_path = os.getenv(LOG_PATH_KEY, None)
    if log_path is None:
        return None
    else:
        return RotatingFileHandler(
            filename=f'{log_path}/{file_name}',
            mode='a',
            maxBytes=256000000,
            backupCount=int(os.environ.get('LOG_FILE_BACKUP_COUNT', 16))
        )
