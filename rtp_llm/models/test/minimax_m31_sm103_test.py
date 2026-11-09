"""Explicit SM103 coverage for MiniMax M3.1 NVFP4 kernels."""

import unittest

import torch

from rtp_llm.models.test import minimax_m31_kv_cache_test as kv


def load_tests(loader, tests, pattern):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("MiniMax M3.1 GPU tests require an SM103 worker")
    return loader.loadTestsFromModule(kv)


if __name__ == "__main__":
    unittest.main()
