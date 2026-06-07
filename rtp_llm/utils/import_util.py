import importlib
import importlib.util
import logging
import os
import threading
from functools import lru_cache
from typing import Dict, Iterable, Optional, Set, Union


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


@lru_cache(maxsize=None)
def import_optional_internal_source_entrypoint(relative_module: str) -> bool:
    """Import an internal_source extension entrypoint when it is present.

    Open-source modules should depend only on stable entrypoint names here,
    not on internal model/tokenizer/renderer names or implementation modules.
    """
    if not has_internal_source():
        return False

    module_name = f"internal_source.rtp_llm.{relative_module}"
    try:
        importlib.import_module(module_name)
        return True
    except ModuleNotFoundError as e:
        if e.name is not None and module_name.startswith(f"{e.name}."):
            return False
        if e.name == module_name:
            return False
        raise


class LazyModuleRegistry:
    """Thread-safe name -> module registry for deferred plugin imports."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self.name_to_module: Dict[str, str] = {}
        self.module_paths: Set[str] = set()
        self.loaded_modules: Set[str] = set()
        self._lock = threading.RLock()

    def register(self, names: Union[str, Iterable[str]], module_path: str) -> None:
        registry_names = [names] if isinstance(names, str) else list(names)
        with self._lock:
            for name in registry_names:
                old_module = self.name_to_module.get(name)
                if old_module is not None and old_module != module_path:
                    raise Exception(
                        f"try register lazy {self._kind} {name} with module "
                        f"{old_module} and {module_path}, conflict!"
                    )
                self.name_to_module[name] = module_path
            self.module_paths.add(module_path)

    def get_module_path(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        with self._lock:
            return self.name_to_module.get(name)

    def import_module(self, module_path: str) -> None:
        with self._lock:
            if module_path in self.loaded_modules:
                return
        importlib.import_module(module_path)
        with self._lock:
            self.loaded_modules.add(module_path)

    def import_all_modules(self) -> None:
        with self._lock:
            module_paths = sorted(self.module_paths)
        for module_path in module_paths:
            self.import_module(module_path)


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
    logging.debug(
        "internal_source directory check: %s, found: %s",
        internal_source_path,
        exists,
    )
    return exists
