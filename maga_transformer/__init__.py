import os

import logging.config

if os.environ.get('FT_SERVER_TEST') is not None:
    LOGLEVEL = os.environ.get('PY_LOG_LEVEL', 'INFO').upper()
else:
    LOGLEVEL = os.environ.get('PY_LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(level=LOGLEVEL,
                    format="[%(name)s][%(asctime)s][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
                    datefmt='%m/%d/%Y %H:%M:%S')

os.makedirs('logs', exist_ok=True)
logging.info("init logger end")

# load th_transformer.so
from .ops import *
