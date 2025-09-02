import logging

from rtp_llm.config.py_config_modules import StaticConfig


def setup_logger(level=None):

    if level is None:
        if StaticConfig.profiling_debug_config.qwen_agent_debug:
            level = logging.DEBUG
        else:
            level = logging.INFO

    logger = logging.getLogger("qwen_agent_logger")
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()
