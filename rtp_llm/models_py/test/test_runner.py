"""Run explicit models_py test modules with their declared resource contract."""

import importlib
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


def main():
    module_name, *selectors = sys.argv[1:]
    gpu_count = int(os.environ.get("GPU_COUNT", "0"))
    if not gpu_count:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if gpu_count:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < gpu_count:
            raise RuntimeError(f"{module_name} requires {gpu_count} CUDA device(s)")
        capabilities = {
            "L20_CU13": {(8, 6), (8, 9)},
            "H20_CU13": {(9, 0)},
            "RTX_5000_PRO_CU13": {(12, 0)},
        }[os.environ["MODELS_PY_TEST_GPU_TYPE"]]
        for device in range(gpu_count):
            if torch.cuda.get_device_capability(device) not in capabilities:
                raise RuntimeError(
                    f"{module_name}: wrong GPU architecture for declared pool"
                )
    if module_name.startswith(
        (
            "rtp_llm.models_py.kernels.cuda.qsa.test.",
            "rtp_llm.models_py.modules.qwen4_exp.test.",
        )
    ):
        # A killed Torch extension build can leave a lock in the shared cache.
        with tempfile.TemporaryDirectory() as build_dir, patch.dict(
            os.environ, {"TORCH_EXTENSIONS_DIR": build_dir}
        ):
            result = _run_suite(module_name, selectors)
    else:
        result = _run_suite(module_name, selectors)
    sys.exit(0 if result.wasSuccessful() else 1)


def _run_suite(module_name, selectors):
    module = importlib.import_module(module_name)
    loader = unittest.defaultTestLoader
    suite = (
        loader.loadTestsFromNames(selectors, module)
        if selectors
        else loader.loadTestsFromModule(module)
    )
    if not suite.countTestCases():
        raise RuntimeError(f"{module_name}: no tests selected")
    return unittest.TextTestRunner(verbosity=2).run(suite)
