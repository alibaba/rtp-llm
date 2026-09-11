"""Load the actual pure component without unrelated C++ registration imports."""

import importlib.util
import sys
from pathlib import Path


def load_component(name, relative_path):
    path = Path(__file__).resolve().parents[4] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
