import ctypes
import datetime
import errno
import mmap
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.qwen4_exp import cpu_target_status as status
from rtp_llm.ops.compute_ops import rtp_llm_ops as ops


def _startup_worker(rank, store):
    dist.init_process_group(
        "gloo",
        init_method="file://" + store,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=60),
    )
    group = dist.group.WORLD
    # This uses the actual native bindings and shared-file bootstrap. Recreated
    # mappings must restart at generation one; close must not poison a peer.
    for cycle in range(2):
        owner = status.create_cpu_target_status(group, ops, timeout_ms=10000)
        for generation in range(1, 10001):
            if cycle and generation % 137 == rank:
                time.sleep(0.0001)
            ok = generation % (7 if rank == 0 else 11) != 0
            assert owner.reduce(ok) == bool(generation % 7 and generation % 11)
        owner.close()
        owner.close()
        dist.barrier()
    # One rank cannot open its mapping. Both must receive startup failure,
    # close any successfully opened mapping, and avoid inference admission.
    seen = []
    original = status._open_mapping

    def fail_open(path, identity):
        seen.append(path)
        if rank == 1:
            raise PermissionError("injected rank1 mapping failure")
        return original(path, identity)

    with patch.object(status, "_open_mapping", fail_open):
        try:
            status.create_cpu_target_status(group, ops)
        except RuntimeError as exc:
            assert "injected rank1 mapping failure" in str(exc)
        else:
            raise AssertionError("asymmetric startup was admitted")
    assert len(seen) == 1 and not Path(seen[0]).exists()
    dist.barrier()
    dist.destroy_process_group()


class CpuTargetStatusTest(unittest.TestCase):
    def mapping(self):
        mapping = mmap.mmap(-1, max(mmap.PAGESIZE, ops.cpu_status_size()))
        pointer = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        self.assertEqual(ops.cpu_status_initialize(pointer, len(mapping)), 0)
        self.addCleanup(mapping.close)
        return mapping, pointer

    def test_default_off_and_unsupported_layouts_do_not_bootstrap(self):
        config = SimpleNamespace(is_mtp=False, enable_qwen4_ple=True)
        pc = SimpleNamespace(
            role_type="DECODE", tp_size=2, dp_size=1, local_world_size=2
        )
        with patch.object(
            status, "create_cpu_target_status", side_effect=AssertionError
        ):
            with patch.dict(os.environ, {"QWEN4_CPU_TARGET_STATUS": "0"}):
                self.assertIsNone(status.prepare_cpu_target_status(config, pc))
            with patch.dict(os.environ, {"QWEN4_CPU_TARGET_STATUS": "1"}):
                for key, value in [
                    ("role_type", "PREFILL"),
                    ("tp_size", 4),
                    ("dp_size", 2),
                    ("local_world_size", 4),
                ]:
                    altered = SimpleNamespace(**{**vars(pc), key: value})
                    self.assertIsNone(status.prepare_cpu_target_status(config, altered))
                self.assertIsNone(
                    status.prepare_cpu_target_status(SimpleNamespace(is_mtp=True), pc)
                )

    def test_native_wait_releases_gil_and_preserves_false(self):
        mapping, pointer = self.mapping()
        results = {}
        threads = [
            threading.Thread(
                target=lambda rank: results.update(
                    {
                        rank: ops.cpu_status_reduce(
                            pointer, rank, 1, int(rank == 0), 1000
                        )
                    }
                ),
                args=(rank,),
            )
            for rank in (0, 1)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(results, {0: 0, 1: 0})

    def test_timeout_poisoning_and_late_peer_are_fail_stop(self):
        mapping, pointer = self.mapping()
        self.assertEqual(ops.cpu_status_reduce(pointer, 0, 1, 1, 20), -errno.ETIMEDOUT)
        self.assertEqual(ops.cpu_status_reduce(pointer, 1, 1, 1, 20), -errno.ECANCELED)
        owner = status.CpuTargetStatus(ops, mapping, 1, timeout_ms=20)
        with self.assertRaisesRegex(RuntimeError, "TP2 CPU status failed"):
            owner.reduce(True)

    def test_bad_generation_and_unaligned_pointer(self):
        mapping, pointer = self.mapping()
        self.assertEqual(ops.cpu_status_reduce(pointer + 1, 0, 1, 1, 20), -errno.EINVAL)
        self.assertEqual(ops.cpu_status_reduce(pointer, 0, 3, 1, 20), -errno.EPROTO)
        self.assertEqual(ops.cpu_status_reduce(pointer, 1, 1, 1, 20), -errno.ECANCELED)

    def test_file_identity_check_rejects_changed_inode(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root) / "a", Path(root) / "b"
            a.touch(mode=0o600)
            b.touch(mode=0o600)
            with self.assertRaisesRegex(RuntimeError, "identity"):
                status._open_mapping(str(a), status._identity(b.stat()))
            self.assertTrue(a.exists())
            with self.assertRaisesRegex(RuntimeError, "identity"):
                status._unlink_owned(str(a), status._identity(b.stat()))
            self.assertTrue(a.exists())

    def test_collective_startup_generations_restart_and_peer_failure(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(
                _startup_worker, args=(str(Path(root) / "store"),), nprocs=2, join=True
            )


if __name__ == "__main__":
    unittest.main()
