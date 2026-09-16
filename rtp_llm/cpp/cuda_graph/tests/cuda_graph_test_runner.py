"""Lazily import the compiled CUDA Graph test binding staged by setup.py."""

import sys
import importlib
from pathlib import Path


_TEST_LIB_DIR = Path(__file__).resolve().parents[3] / "libs" / "test"
if str(_TEST_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_LIB_DIR))

def load_cuda_graph_test_binding():
    try:
        binding = importlib.import_module("libtest_cuda_graph_runner")
    except ImportError as exc:
        raise ImportError(
            f"libtest_cuda_graph_runner.so not found under {_TEST_LIB_DIR}; "
            "run `python setup.py build_ext --inplace` before pytest"
        ) from exc
    return binding


class CudaGraphRunner:
    """Instantiate the native runner only after pytest marker selection.

    The CUDA Graph driver modules must remain importable on non-CUDA workers so
    pytest can deselect their H20-marked tests before execution.
    """

    def __new__(cls, *args, **kwargs):
        return load_cuda_graph_test_binding().CudaGraphRunner(*args, **kwargs)


__all__ = ["CudaGraphRunner", "load_cuda_graph_test_binding"]
