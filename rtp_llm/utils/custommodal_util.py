import logging
import os
from enum import Enum
from typing import Any, List, Optional

from rtp_llm.utils.import_util import load_module  # Added load_module


class MethodType(Enum):
    Embedding = "embedding"
    Preprocess = "preprocess"


MODEL_PATH_CONFIG = {
    MethodType.Embedding: "embedding_module_path",
    MethodType.Preprocess: "processer_module_path",
}


def load_custom_modal_class(
    custom_modal_config, ckpt_path, type: MethodType
) -> Optional[Any]:
    """Helper to load custom modal class from config."""
    if not isinstance(custom_modal_config, dict):
        return None

    path_str = custom_modal_config.get(MODEL_PATH_CONFIG[type])
    if not path_str:
        return None

    try:
        file_name_part, class_name = path_str.rsplit(".", 1)
        module_file_path = os.path.join(ckpt_path, file_name_part + ".py")

        if not os.path.exists(module_file_path):
            logging.warning(f"Custom modal module file not found: {module_file_path}")
            return None

        logging.debug(f"Loading custom modal class from {module_file_path}")
        module = load_module(module_file_path)
        cls = getattr(module, class_name)
        return cls
    except Exception as e:
        logging.warning(f"Failed to load custom modal class: {e}")
        return None
