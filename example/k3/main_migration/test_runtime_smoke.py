"""Validate replay ownership and published JIT evidence for K3 smoke."""

import io
import os
import pathlib
import tarfile
import tempfile
import time
import unittest

from jit_smoke import check_publication
from runtime_smoke import check_replay_events, check_replay_owners


class RuntimeSmokeChecksTest(unittest.TestCase):
    def event(self, time, generation, epoch, request):
        return {
            "event": "linear_replay_commit",
            "time_ns": time,
            "slot": 0,
            "generation": generation,
            "verify_epoch": epoch,
            "request_id": request,
            "produced_steps": 4,
        }

    def test_slot_reuse_with_new_generation(self):
        self.assertEqual(
            check_replay_events(
                [
                    self.event(1, 1, 1, 10),
                    self.event(2, 1, 2, 10),
                    self.event(3, 2, 1, 20),
                ]
            ),
            {"commits": 3, "slot_reuses": 1},
        )

    def test_cross_request_stale_epoch_and_generation_fail(self):
        for bad in (
            self.event(2, 1, 2, 20),
            self.event(2, 1, 1, 10),
            self.event(2, 0, 2, 10),
        ):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                check_replay_events([self.event(1, 1, 1, 10), bad])

    def test_dp_owners_commit_while_tp_followers_have_no_stream_leases(self):
        events = [self.event(1, 1, 1, 10), self.event(2, 2, 1, 20)]
        ranks = {rank: [] for rank in range(8)}
        ranks[0] = events
        ranks[4] = events
        result = check_replay_owners(ranks, 2)
        self.assertEqual(set(result), {"0", "4"})
        self.assertEqual(result["4"], {"commits": 2, "slot_reuses": 1})

    def test_missing_dp_owner_commits_fail(self):
        ranks = {rank: [] for rank in range(8)}
        ranks[0] = [self.event(1, 1, 1, 10), self.event(2, 2, 1, 20)]
        with self.assertRaisesRegex(AssertionError, "not exercised"):
            check_replay_owners(ranks, 2)


class JitSmokeChecksTest(unittest.TestCase):
    def snapshot(self, root, names, label="sample", version="v2"):
        import zstandard

        path = root / version / "scope" / f"{label}.jit_snapshot.tar.zst"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            for name in names:
                member = tarfile.TarInfo(name)
                member.size = 3
                archive.addfile(member, io.BytesIO(b"bin"))
        path.write_bytes(zstandard.ZstdCompressor().compress(data.getvalue()))
        return path

    def test_startup_publication_can_precede_first_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            startup = time.time_ns()
            path = self.snapshot(
                root, ["deep_gemm/kernel.cubin", "triton/kernel.cubin"]
            )
            # Establish the intended ordering independently of filesystem clock rounding.
            os.utime(path, ns=(startup + 1000, startup + 1000))
            request_started = time.time_ns()
            result = check_publication(root, startup, timeout=1)
            self.assertLess(startup, request_started)
            self.assertTrue(result["passed"])
            self.assertEqual(
                result["compiled_binary_counts"], {"deep_gemm": 1, "triton": 1}
            )

    def test_stale_and_missing_component_snapshots_fail(self):
        for stale in (False, True):
            with self.subTest(stale=stale), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                names = ["deep_gemm/kernel.cubin"]
                if stale:
                    names.append("triton/kernel.cubin")
                path = self.snapshot(root, names)
                startup = time.time_ns()
                if stale:
                    os.utime(path, ns=(startup - 1000, startup - 1000))
                else:
                    os.utime(path, ns=(startup + 1000, startup + 1000))
                with self.assertRaisesRegex(AssertionError, "did not publish"):
                    check_publication(root, startup, timeout=0.01)

    def test_two_partial_snapshots_do_not_make_one_restorable_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            startup = time.time_ns()
            for component in ("deep_gemm", "triton"):
                path = self.snapshot(root, [f"{component}/kernel.cubin"], component)
                os.utime(path, ns=(startup + 1000, startup + 1000))
            with self.assertRaisesRegex(AssertionError, "did not publish"):
                check_publication(root, startup, timeout=0.01)

    def test_previous_cache_version_does_not_count_as_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            startup = time.time_ns()
            path = self.snapshot(
                root,
                ["deep_gemm/kernel.cubin", "triton/kernel.cubin"],
                version="v1",
            )
            os.utime(path, ns=(startup + 1000, startup + 1000))
            with self.assertRaisesRegex(AssertionError, "did not publish"):
                check_publication(root, startup, timeout=0.01)


if __name__ == "__main__":
    unittest.main()
