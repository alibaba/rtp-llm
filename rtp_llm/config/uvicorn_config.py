import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

from uvicorn.logging import AccessFormatter

from rtp_llm.config.log_config import get_log_path


def configure_uvicorn_logging(log_path: Optional[str] = None) -> None:
    """Configure HTTP access logging without closing application handlers.

    Pass log_config=None to Uvicorn after calling this: dictConfig closes all
    registered handlers, even with disable_existing_loggers=False, permanently
    stopping existing asynchronous MM and frontend access log workers.
    """
    if log_path is None:
        log_path = get_log_path()
    handler = RotatingFileHandler(
        os.path.join(log_path, "uvicorn_access.log"),
        maxBytes=50 * 1024 * 1024,
        backupCount=10,
    )
    handler.setFormatter(
        AccessFormatter(
            fmt='%(asctime)s.%(msecs)03d %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger = logging.getLogger("uvicorn.access")
    for old_handler in logger.handlers[:]:
        logger.removeHandler(old_handler)
        old_handler.close()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.disabled = False
    logger.propagate = False
