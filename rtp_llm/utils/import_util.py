import importlib.util
import logging
import os
from functools import lru_cache


def load_module(module_path: str):
    module_spec = importlib.util.spec_from_file_location(
        "inference_module", module_path
    )
    if module_spec is None:
        raise ModuleNotFoundError(f"failed to load module from [{module_path}]")

    imported_module = importlib.util.module_from_spec(module_spec)

    if module_spec.loader != None:
        module_spec.loader.exec_module(imported_module)
    else:
        raise Exception(f"ModuleSpec [{module_spec}] has no loader.")
    return imported_module


@lru_cache(maxsize=1)
def has_internal_source() -> bool:
    """
    检查项目根目录下是否存在 internal_source 目录。
    结果会被缓存，避免重复检查文件系统。

    Returns:
        bool: 如果 internal_source 目录存在则返回 True，否则返回 False
    """
    # rtp_llm/utils/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # rtp_llm/
    rtp_llm_dir = os.path.dirname(current_dir)
    # root
    project_root = os.path.dirname(rtp_llm_dir)
    internal_source_path = os.path.join(project_root, "internal_source")
    exists = os.path.exists(internal_source_path) and os.path.isdir(
        internal_source_path
    )
    print(f"[CHECK] internal_source directory: {internal_source_path}, found: {exists}")
    return exists
