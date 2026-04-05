"""Step 1+2: Hello World + GPU verification via REAPI."""
import os
import platform

import pytest


@pytest.mark.gpu(count=0)
def test_hello_remote():
    """Simplest possible remote test - no GPU, no .so, no torch."""
    print(f"hostname: {platform.node()}")
    print(f"python: {platform.python_version()}")
    print(f"cwd: {os.getcwd()}")
    print(f"GPU_COUNT: {os.environ.get('GPU_COUNT', 'unset')}")
    assert 1 + 1 == 2


@pytest.mark.gpu(type="A10", count=1)
@pytest.mark.cuda
def test_gpu_remote():
    """Step 2: Verify GPU available and torch works on remote A10 worker."""
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"torch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    assert "A10" in gpu_name or "NVIDIA" in gpu_name


@pytest.mark.gpu(type="L20", count=1)
@pytest.mark.cuda
@pytest.mark.manual
def test_gpu_remote_l20():
    """Verify GPU available on remote L20 worker."""
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"torch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    assert "L20" in gpu_name or "NVIDIA" in gpu_name


def test_hello_local():
    """Non-GPU test that should always run locally."""
    assert True
