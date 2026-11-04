#!/usr/bin/env python3
"""Explicit two-GPU Fabric gate; never terminates another GPU process.

Exit 2 means prerequisites/idle resources were unavailable; exit 1 means the
integration tests failed (including skips); only executed passing tests exit 0.
"""

import fcntl
import importlib.util
import os
import subprocess
import sys
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path


def _smi(*args):
    return subprocess.run(
        ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


@contextmanager
def idle_devices(lock_root=Path("/tmp/rtp_llm/smoke/test")):
    """Lock NVML physical indices; pass stable UUID identities to CUDA.

    Numeric visibility entries mean nvidia-smi indices in this test runner.
    CUDA ordinals can have a different order, so never write those indices
    back to CUDA_VISIBLE_DEVICES after checking and locking the devices.
    """
    rows = _smi("--query-gpu=index,uuid").splitlines()
    devices = dict(tuple(part.strip() for part in row.split(",")) for row in rows)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected = list(devices) if visible is None else visible.split(",")
    uuid_to_index = {uuid: index for index, uuid in devices.items()}
    selected = [uuid_to_index.get(value.strip(), value.strip()) for value in selected]
    required = int(os.environ.get("ENGRAM_TEST_GPU_COUNT", "2"))
    if required not in (2, 4):
        raise ValueError("ENGRAM_TEST_GPU_COUNT must be 2 or 4")
    if (
        len(selected) < required
        or len(set(selected)) != len(selected)
        or any(device not in devices for device in selected)
    ):
        raise RuntimeError("two visible physical GPUs are required; MIG is unsupported")
    selected = selected[:required]
    (lock_root / "gpu_status").mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        # FileLock uses flock on Linux. Holding both GPU files interoperates
        # with ordinary RTP tests; a held lock or a live PID is a gate failure.
        with (lock_root / "gpu_status_lock").open("a+") as global_lock:
            fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for device in selected:
                handle = locks.enter_context(
                    (lock_root / "gpu_status" / device).open("a+")
                )
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                pids = _smi(f"--id={device}", "--query-compute-apps=pid")
                if pids:
                    raise RuntimeError(
                        f"GPU {device} is occupied; no process was terminated"
                    )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            devices[index] for index in selected
        )
        os.environ["ENGRAM_TEST_DEVICE"] = "0"
        os.environ["ENGRAM_TEST_PEER_DEVICE"] = "1"
        yield selected


def _wheel_library_environment():
    """Resolve Bazel wheel libraries without the process-cleaning run_under."""
    if os.environ.get("_ENGRAM_TEST_LD_READY") == "1":
        return
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        raise RuntimeError("Torch is required for the Fabric integration gate")
    libraries = [str(Path(spec.origin).parent / "lib")]
    for directory in sys.path:
        libraries.extend(str(path) for path in Path(directory).glob("nvidia/*/lib"))
    env = dict(os.environ, _ENGRAM_TEST_LD_READY="1")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        libraries + [env.get("LD_LIBRARY_PATH", "")]
    )
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:]], env)


def require_fabric(native, numa):
    """Prove HOST_NUMA allocation and cross-GPU Fabric import, not just SM100."""
    before = dict(native.resource_counters())
    granularity = native.host_numa_granularity(numa, 0)
    owner = native.FabricAllocation(granularity, numa, 0)
    mapping = native.FabricMapping(device=1)
    try:
        info = owner.info()
        mapping.attach(
            [(owner.export_handle(), info["allocation_bytes"], info["granularity"])]
        )
    finally:
        mapping.close()
        owner.close()
    if dict(native.resource_counters()) != before:
        raise RuntimeError("Fabric capability probe leaked native resources")


def build_test_suite(native, torch, numa):
    """Bind native tests to the bridge already checked under the GPU locks."""
    from rtp_llm.models_py.modules.dsv41.engram_distributed.test.test_fabric import (
        FabricEmbeddingLookupTest,
        ShardedCudaLifecycleTest,
        native_test_suite,
    )

    loader = unittest.defaultTestLoader
    return unittest.TestSuite(
        [
            native_test_suite(native, torch, device=0, numa=numa),
            loader.loadTestsFromTestCase(FabricEmbeddingLookupTest),
            loader.loadTestsFromTestCase(ShardedCudaLifecycleTest),
        ]
    )


def run_tests(native, torch, numa):
    suite = build_test_suite(native, torch, numa)
    result = unittest.TextTestRunner(verbosity=2, failfast=True).run(suite)
    passed = result.wasSuccessful() and result.testsRun > 0 and not result.skipped
    print(
        f"ENGRAM_FABRIC_TESTS {'PASS' if passed else 'FAIL'} "
        f"run={result.testsRun} skipped={len(result.skipped)}",
        flush=True,
    )
    return 0 if passed else 1


def main():
    try:
        if os.environ.get("ENGRAM_RUN_CUDA_TESTS") != "1":
            raise RuntimeError(
                "set ENGRAM_RUN_CUDA_TESTS=1 on an authorized Fabric test worker"
            )
        numa = int(os.environ["ENGRAM_TEST_NUMA"])
        if numa < 0:
            raise ValueError("ENGRAM_TEST_NUMA must be nonnegative")
        _wheel_library_environment()
        with idle_devices() as devices:
            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() != len(
                devices
            ):
                raise RuntimeError("two usable CUDA GPUs are required")
            from rtp_llm.models_py.modules.dsv41.engram_distributed.runtime import (
                _load_native,
            )

            native = _load_native()
            require_fabric(native, numa)
            print(
                f"ENGRAM_FABRIC_GATE READY devices={devices} "
                f"uuids={os.environ['CUDA_VISIBLE_DEVICES']} numa={numa}",
                flush=True,
            )
            return run_tests(native, torch, numa)
    except Exception as error:
        print(
            f"ENGRAM_FABRIC_GATE FAILED {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
