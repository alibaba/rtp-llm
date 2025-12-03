import importlib.util
import logging
import logging.config
import os
import sys

LOGLEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
if LOGLEVEL == "TRACE":
    LOGLEVEL = "DEBUG"
logging.basicConfig(
    level=LOGLEVEL,
    format="[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from rtp_llm.config.log_config import LOGGING_CONFIG
from rtp_llm.utils.torch_patch import *

## for `__init__.py`, we reserve the envs, don't use StaticConfig.
LOG_PATH = os.environ.get("LOG_PATH", "logs")
os.makedirs(LOG_PATH, exist_ok=True)


file_logger_init_success = False
if os.environ.get("FT_SERVER_TEST") is None:
    LOGGING_CONFIG["loggers"][""]["level"] = LOGLEVEL

    if os.environ.get("NCCL_DEBUG_FILE") is None:
        os.environ["NCCL_DEBUG_FILE"] = os.path.join(LOG_PATH, "nccl.log")
        print(
            f"successfully set NCCL_DEBUG_FILE path to {os.environ['NCCL_DEBUG_FILE']}"
        )
    try:
        logging.config.dictConfig(LOGGING_CONFIG)
        file_logger_init_success = True
        print(f"successfully init logger config {LOGGING_CONFIG}")
    except BaseException as e:
        # for compatible with yitian arm machine in prod env, which lacks hippo infras and envs.
        import traceback

        print(
            f"failed to init logger config {LOGGING_CONFIG}: {e}\n {traceback.format_exc()}"
        )


# try auto set alog config path
if os.environ.get("FT_ALOG_CONF_PATH") is None:

    def find_module_path(module_name: str):
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
            return spec.origin
        else:
            return None

    rtp_llm_path = "rtp_llm"
    try:
        module_path = find_module_path(rtp_llm_path)
        if module_path:
            os.environ["FT_ALOG_CONF_PATH"] = os.path.join(
                os.path.dirname(module_path), "config/alog.conf"
            )
        else:
            print(f"Could not find the module '{rtp_llm_path}'.")
    except ImportError as e:
        print(f"Error importing module: {e}")

logging.info("init logger end")

import time

st = time.time()

import transformers

logging.info(f"transformers version: {transformers.__version__}")

# load th_transformer.so
from .ops import *

# Import internal models to register them
try:
    import internal_source.rtp_llm.models_py
except ImportError:
    logging.warning("Failed to import internal_source models")

consume_s = time.time() - st
print(f"import in __init__ took {consume_s:.2f}s")
