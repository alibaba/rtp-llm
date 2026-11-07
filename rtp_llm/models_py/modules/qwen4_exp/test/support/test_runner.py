"""Run explicit Qwen4 test modules with their declared resource contract."""

import importlib
import os
import sys
import unittest


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
        }[os.environ["QWEN4_TEST_GPU_TYPE"]]
        for device in range(gpu_count):
            if torch.cuda.get_device_capability(device) not in capabilities:
                raise RuntimeError(
                    f"{module_name}: wrong GPU architecture for declared pool"
                )
    module = importlib.import_module(module_name)
    loader = unittest.defaultTestLoader
    suite = (
        loader.loadTestsFromNames(selectors, module)
        if selectors
        else loader.loadTestsFromModule(module)
    )
    if not suite.countTestCases():
        raise RuntimeError(f"{module_name}: no tests selected")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
