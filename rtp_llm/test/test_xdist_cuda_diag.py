"""Diagnostic test: verify CUDA visibility in xdist worker subprocesses.

This test is designed to run under xdist (-n N) on a GPU machine and report
exactly what each worker process sees. It does NOT import torch at module level
to avoid masking the issue.

Run:
  GPU_COUNT=4 python rtp_llm/test/utils/device_resource.py \
    python -m pytest -p no:remote-gpu -p no:rtp-ci-profile \
    rtp_llm/test/test_xdist_cuda_diag.py -n 4 -v --tb=long

  Or via remote session:
    pytest --remote-session --rtp-ci-profile=py_ut_sm8x \
      -k test_xdist_cuda_diag --timeout=600 -v
"""

import json
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.gpu(count=1)]


class TestXdistCudaDiag:
    """Diagnostic: report what each xdist worker sees regarding CUDA."""

    def test_env_vars(self):
        """Report all GPU-related environment variables."""
        worker = os.environ.get("PYTEST_XDIST_WORKER", "no-xdist")
        info = {
            "worker": worker,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<NOT SET>"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "<NOT SET>"),
            "GPU_COUNT": os.environ.get("GPU_COUNT", "<NOT SET>"),
            "GPU_COUNT_PER_WORKER": os.environ.get("GPU_COUNT_PER_WORKER", "<NOT SET>"),
            "_RTP_TORCH_BEFORE_SLICE": os.environ.get(
                "_RTP_TORCH_BEFORE_SLICE", "<NOT SET>"
            ),
        }
        print(f"\n[DIAG_ENV] {json.dumps(info)}")
        # CVD must be set in an xdist worker
        if worker != "no-xdist":
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            assert cvd is not None, f"CVD not set in xdist worker {worker}"
            assert cvd != "", f"CVD is empty string in xdist worker {worker}"

    def test_nvidia_smi_from_worker(self):
        """Run nvidia-smi from within the xdist worker subprocess."""
        worker = os.environ.get("PYTEST_XDIST_WORKER", "no-xdist")
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,uuid",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            print(f"\n[DIAG_SMI] worker={worker} rc={result.returncode}")
            for line in result.stdout.strip().splitlines():
                print(f"[DIAG_SMI]   {line}")
            if result.stderr.strip():
                print(f"[DIAG_SMI]   stderr: {result.stderr.strip()}")
            assert result.returncode == 0, f"nvidia-smi failed: {result.stderr}"
        except FileNotFoundError:
            pytest.skip("nvidia-smi not found")

    def test_cuda_init(self):
        """Try torch.cuda initialization and report exact result."""
        worker = os.environ.get("PYTEST_XDIST_WORKER", "no-xdist")
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<NOT SET>")

        import torch

        is_avail = torch.cuda.is_available()
        print(f"\n[DIAG_CUDA] worker={worker} CVD={cvd} is_available={is_avail}")

        if not is_avail:
            # Collect extra diagnostics
            print(f"[DIAG_CUDA] torch.version.cuda={torch.version.cuda}")
            print(
                f"[DIAG_CUDA] torch.backends.cuda.is_built={torch.backends.cuda.is_built()}"
            )
            # Try to get the actual error
            try:
                torch.cuda._lazy_init()
            except Exception as e:
                print(f"[DIAG_CUDA] _lazy_init error: {type(e).__name__}: {e}")
            pytest.fail(
                f"torch.cuda.is_available()=False in worker={worker} CVD={cvd}. "
                f"See [DIAG_CUDA] output above for details."
            )

        device_count = torch.cuda.device_count()
        print(f"[DIAG_CUDA] device_count={device_count}")

        for i in range(device_count):
            cap = torch.cuda.get_device_capability(i)
            name = torch.cuda.get_device_name(i)
            print(f"[DIAG_CUDA] device[{i}]: name={name} capability={cap}")

        assert device_count >= 1, f"device_count={device_count} but CVD={cvd}"

    def test_cuda_alloc(self):
        """Try to allocate a tensor on the GPU."""
        worker = os.environ.get("PYTEST_XDIST_WORKER", "no-xdist")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        t = torch.zeros(1, device="cuda:0")
        print(f"\n[DIAG_ALLOC] worker={worker} tensor={t} device={t.device}")
        del t
        torch.cuda.empty_cache()

    def test_subprocess_cuda_visibility(self):
        """Spawn a child process and check if IT can see CUDA too."""
        worker = os.environ.get("PYTEST_XDIST_WORKER", "no-xdist")
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os, torch; "
                'print(f\'CVD={os.environ.get("CUDA_VISIBLE_DEVICES", "<NOT SET>")}\'); '
                "print(f'avail={torch.cuda.is_available()}'); "
                "print(f'count={torch.cuda.device_count() if torch.cuda.is_available() else 0}')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=os.environ.copy(),
        )
        print(f"\n[DIAG_SUBPROCESS] worker={worker} rc={result.returncode}")
        for line in result.stdout.strip().splitlines():
            print(f"[DIAG_SUBPROCESS]   {line}")
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"[DIAG_SUBPROCESS]   stderr: {line}")
        assert (
            result.returncode == 0
        ), f"Subprocess CUDA check failed: {result.stderr[-500:]}"
