import os

import logging.config
from maga_transformer.config.log_config import LOGGING_CONFIG


LOG_PATH = os.environ.get('LOG_PATH', 'logs')
os.makedirs(LOG_PATH, exist_ok=True)
LOGLEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
if LOGLEVEL == "TRACE":
    LOGLEVEL = "DEBUG"

if os.environ.get('FT_SERVER_TEST') is None:
    LOGGING_CONFIG['loggers']['']['level'] = LOGLEVEL
    logging.config.dictConfig(LOGGING_CONFIG)
    if os.environ.get('NCCL_DEBUG_FILE') is None:
        os.environ['NCCL_DEBUG_FILE'] = os.path.join(LOG_PATH, 'nccl.log')
else:
    logging.basicConfig(level=LOGLEVEL,
                    format="[process-%(process)d][%(name)s][%(asctime)s][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
                    datefmt='%m/%d/%Y %H:%M:%S')

logging.info("init logger end")

# load th_transformer.so
from .ops import *
