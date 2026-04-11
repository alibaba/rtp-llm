"""GPU isolation verification tests.

Validates the two GPU isolation paths:
- Path A (xdist): conftest.py module-level code slices CVD per worker
- Path B (non-xdist): device_resource.py __main__ wrapper sets CVD before pytest

Also verifies that torch was NOT imported before GPU slicing (critical for
ensuring cuInit() sees the correct CUDA_VISIBLE_DEVICES).

Collected by CI profiles (py_ut_sm8x etc.) as a permanent regression guard
for GPU isolation correctness.
"""
import json
import os
import pathlib

import pytest
import torch

_GPU_VERIFY_DIR = pathlib.Path(
    os.environ.get("GPU_VERIFY_DIR", "/tmp/rtp_llm_gpu_verify")
)


def _cvd_gpus():
    """Return list of GPU indices from CUDA_VISIBLE_DEVICES."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return []
    return [g.strip() for g in cvd.split(",") if g.strip()]


@pytest.mark.gpu(type="A10", count=1)
class TestGpuIsolation:
    """Core GPU isolation contract checks (both Path A and Path B)."""

    def test_cvd_is_set(self):
        """CUDA_VISIBLE_DEVICES must be set before tests run."""
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        assert cvd is not None, (
            "CUDA_VISIBLE_DEVICES is not set. "
            "GPU isolation (conftest.py slicing / device_resource wrapper) did not activate."
        )
        gpus = _cvd_gpus()
        assert len(gpus) >= 1, f"CVD is empty: {cvd!r}"

    def test_device_count_matches_cvd(self):
        """torch.cuda.device_count() must equal the number of GPUs in CVD."""
        expected = len(_cvd_gpus())
        actual = torch.cuda.device_count()
        assert actual == expected, (
            f"torch sees {actual} GPU(s) but CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r} implies {expected}. "
            f"cuInit() likely ran before CVD was set."
        )

    def test_can_allocate_on_device_0(self):
        """torch can allocate and compute on the isolated GPU."""
        t = torch.randn(32, device="cuda:0")
        result = t.sum().item()
        assert isinstance(result, float)
        del t
        torch.cuda.empty_cache()

    def test_torch_not_imported_before_gpu_slice(self):
        """Verify torch was NOT in sys.modules when conftest.py sliced CVD.

        In PyTorch 2.x, cuInit() is deferred lazily so import torch alone may
        not trigger it — but some builds or future versions may change this.
        test_device_count_matches_cvd only catches the case where cuInit()
        actually ran early; this test catches the root cause directly.
        """
        worker = os.environ.get("PYTEST_XDIST_WORKER")
        if not worker:
            pytest.skip("Only relevant under xdist (conftest slicing is xdist-only)")

        flag = os.environ.get("_RTP_TORCH_BEFORE_SLICE")
        if flag is None:
            pytest.skip("_RTP_TORCH_BEFORE_SLICE not set (conftest slicing did not run)")
        assert flag == "0", (
            "torch was already imported BEFORE conftest.py GPU slicing ran. "
            "cuInit() may have seen the wrong CUDA_VISIBLE_DEVICES. "
            "Ensure -p no:remote-gpu -p no:rtp-ci-profile are in the pytest command. "
            f"_RTP_TORCH_BEFORE_SLICE={flag}"
        )

    def test_record_worker_gpu(self):
        """Write worker GPU assignment to shared dir for xdist overlap check.

        Non-xdist: writes as 'main.json'.
        xdist: writes as 'gw0.json', 'gw1.json', etc.
        """
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        record = {
            "worker": worker,
            "pid": os.getpid(),
            "cvd": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
        }
        _GPU_VERIFY_DIR.mkdir(parents=True, exist_ok=True)
        (_GPU_VERIFY_DIR / f"{worker}.json").write_text(json.dumps(record))


@pytest.mark.gpu(type="A10", count=1)
class TestXdistDisjoint:
    """Verify xdist workers received non-overlapping GPUs.

    Only meaningful under xdist (-n >= 2). Skipped in single-process mode.
    """

    def test_workers_have_disjoint_gpus(self):
        worker = os.environ.get("PYTEST_XDIST_WORKER")
        if not worker:
            pytest.skip("Not running under xdist")

        files = sorted(_GPU_VERIFY_DIR.glob("gw*.json"))
        if len(files) < 2:
            pytest.skip(f"Only {len(files)} worker record(s) found, need >= 2")

        seen_cvds: dict = {}
        for f in files:
            rec = json.loads(f.read_text())
            cvd = rec["cvd"]
            w = rec["worker"]
            if cvd in seen_cvds:
                pytest.fail(
                    f"GPU OVERLAP: workers {seen_cvds[cvd]} and {w} "
                    f"both assigned CUDA_VISIBLE_DEVICES={cvd}"
                )
            seen_cvds[cvd] = w
