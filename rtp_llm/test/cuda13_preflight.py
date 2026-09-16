"""Preserve CUDA 13 target requirements before pytest imports test modules."""

import os
import tempfile


def prepare_cuda13_runtime(required_devices):
    import torch

    cache_root = os.environ.get("TEST_TMPDIR") or tempfile.gettempdir()
    os.environ["DG_JIT_CACHE_DIR"] = os.path.abspath(
        os.environ.get("DG_JIT_CACHE_DIR") or os.path.join(cache_root, "deep_gemm")
    )
    if required_devices < 1:
        raise RuntimeError("CUDA 13 tests require at least one device")
    if not torch.version.cuda or int(torch.version.cuda.split(".")[0]) != 13:
        raise RuntimeError(f"CUDA 13 runtime required, got {torch.version.cuda}")
    if torch.cuda.device_count() < required_devices:
        raise RuntimeError(
            f"requires {required_devices} CUDA devices, found {torch.cuda.device_count()}"
        )
    for index in range(required_devices):
        if torch.cuda.get_device_capability(index)[0] != 10:
            raise RuntimeError("DSV4 CUDA13 tests require Blackwell devices")


def pytest_sessionstart(session):
    prepare_cuda13_runtime(int(os.environ.get("GPU_COUNT", "1")))
