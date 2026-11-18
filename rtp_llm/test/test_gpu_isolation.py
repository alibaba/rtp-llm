"""GPU isolation verification tests.

Validates the two GPU isolation paths:
- Path A (xdist): conftest.py module-level code slices CVD per worker
- Path B (non-xdist): device_resource.py __main__ wrapper sets CVD before pytest

Also verifies that torch was NOT imported before GPU slicing (critical for
ensuring cuInit() sees the correct CUDA_VISIBLE_DEVICES).

Collected by CI profiles (py_ut_sm8x etc.) as a permanent regression guard
for GPU isolation correctness.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from conftest import _xdist_gpu_slot, pytest_configure_node, pytest_testnodedown


@pytest.mark.multi_arch_cuda
def test_xdist_replacement_reuses_logical_gpu_slot():
    assert _xdist_gpu_slot(worker_index=2, worker_count=4, slot_count=4) == 2
    with pytest.raises(ValueError, match="controller-assigned"):
        _xdist_gpu_slot(worker_index=4, worker_count=4, slot_count=4)


@pytest.mark.multi_arch_cuda
def test_xdist_controller_counts_logical_slots_not_replacements():
    from unittest.mock import Mock

    config = SimpleNamespace()

    def node(worker_id):
        channel = Mock()
        channel.receive.side_effect = lambda **kw: channel.send.call_args.args[0]
        gateway = Mock(id=worker_id)
        gateway.spec.popen = True
        gateway.remote_exec.return_value = channel
        return SimpleNamespace(
            config=config,
            gateway=gateway,
            workerinput={"workercount": 4},
            workeroutput={},
        )

    nodes = [node(f"gw{i}") for i in range(4)]
    for worker in nodes:
        pytest_configure_node(worker)
    # gw2 fails first; gw4 must inherit slot 2, while gw0 still owns slot 0.
    nodes[2].workeroutput = {"rtp_gpu_cvd": "2", "rtp_gpu_slot": 2}
    pytest_testnodedown(nodes[2], error=None)
    replacement = node("gw4")
    pytest_configure_node(replacement)
    assert replacement.workerinput["rtp_gpu_slot"] == 2
    assert config._rtp_gpu_slot_allocations["gw0"] == 0
    assert config._rtp_expected_gpu_workers == 4
    replacement.workeroutput = {"rtp_gpu_cvd": "2", "rtp_gpu_slot": 2}
    pytest_testnodedown(replacement, error=None)
    assert config._rtp_worker_gpu_records == {2: {"cvd": "2", "worker": "gw4"}}


def _cvd_gpus():
    """Return list of GPU indices from CUDA_VISIBLE_DEVICES."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return []
    return [g.strip() for g in cvd.split(",") if g.strip()]


@pytest.mark.gpu(type="A10", count=1)
class TestGpuIsolation:
    """Core GPU isolation contract checks (both Path A and Path B).

    These tests verify CI infrastructure (xdist GPU slicing / device_resource wrapper).
    They SKIP in local single-process mode where CUDA_VISIBLE_DEVICES is not set
    (no xdist, no device_resource wrapper). Use pytest -n auto or the device_resource
    wrapper to activate GPU isolation.
    """

    @staticmethod
    def _skip_if_no_isolation():
        """Skip when GPU isolation is not active (local single-process mode)."""
        # CVD is only set by conftest.py xdist slicing or device_resource.py
        # __main__ wrapper. Neither activates in single-process pytest.
        if not os.environ.get("PYTEST_XDIST_WORKER") and not os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ):
            pytest.skip(
                "GPU isolation not active (no xdist, no device_resource wrapper). "
                "Run with -n auto or via device_resource to test isolation."
            )

    def test_cvd_is_set(self):
        """CUDA_VISIBLE_DEVICES must be set before tests run."""
        self._skip_if_no_isolation()
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        assert cvd is not None, (
            "CUDA_VISIBLE_DEVICES is not set. "
            "GPU isolation (conftest.py slicing / device_resource wrapper) did not activate."
        )
        gpus = _cvd_gpus()
        assert len(gpus) >= 1, f"CVD is empty: {cvd!r}"

    def test_device_count_matches_cvd(self):
        """torch.cuda.device_count() must equal the number of GPUs in CVD."""
        self._skip_if_no_isolation()
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
            pytest.skip(
                "_RTP_TORCH_BEFORE_SLICE not set (conftest slicing did not run)"
            )
        assert flag == "0", (
            "torch was already imported BEFORE conftest.py GPU slicing ran. "
            "cuInit() may have seen the wrong CUDA_VISIBLE_DEVICES. "
            "Ensure -p no:remote-gpu -p no:rtp-ci-profile are in the pytest command. "
            f"_RTP_TORCH_BEFORE_SLICE={flag}"
        )


# Cross-worker GPU-disjointness is verified by the controller in conftest.py
# (pytest_configure_node / pytest_testnodedown / pytest_sessionfinish) rather
# than by a test item here. A normal test runs on only ONE xdist worker, so it
# cannot observe the other workers' assignments; the previous file-based
# `test_workers_have_disjoint_gpus` silently pytest.skip()'d when it saw < 2
# shared records, letting a GPU-overlap misconfig pass CI. The controller hook
# sees every worker's reported CVD and FAILS the run on overlap or a missing
# report.
