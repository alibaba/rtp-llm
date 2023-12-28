import os

import sys
import logging
import logging.config
from maga_transformer.config.log_config import LOGGING_CONFIG
# load th_transformer.so
from .ops import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s:%(lineno)s [%(message)s]',
                    datefmt='%m/%d/%Y %H:%M:%S')

os.makedirs('logs', exist_ok=True)
logging.info("init logger end")
