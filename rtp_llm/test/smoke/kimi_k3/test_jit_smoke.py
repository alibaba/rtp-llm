"""Validate restorable JIT cache publication for K3 smoke."""

import io
import os
import pathlib
import tarfile
import tempfile
import time
import unittest

from rtp_llm.test.smoke.kimi_k3.jit_smoke import check_publication


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
