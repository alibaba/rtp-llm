import importlib.util
import logging
import os
import sys
import pkg_resources

# 确保是从site-packages加载的Python包，避免同名的cpp flashinfer包冲突
def load_flashinfer_python():
    try:
        dist = pkg_resources.get_distribution("flashinfer-python")
        flashinfer_path = dist.location
        logging.info(f"Found flashinfer-python at: {flashinfer_path}")
        spec = importlib.util.spec_from_file_location(
            "flashinfer", os.path.join(flashinfer_path, "flashinfer", "__init__.py")
        )
        if spec and spec.origin and "site-packages" in spec.origin:
            flashinfer_module = importlib.util.module_from_spec(spec)
            sys.modules["flashinfer"] = flashinfer_module
            spec.loader.exec_module(flashinfer_module)
            logging.info(f"load flashinfer-python succeed! spec: {spec}")
            return flashinfer_module
        else:
            logging.warning(f"can't load flashinfer-python, spec: {spec}")
    except Exception as e:
        logging.warning(f"Failed to load flashinfer-python: {e}")
    return None

flashinfer_python = load_flashinfer_python()
