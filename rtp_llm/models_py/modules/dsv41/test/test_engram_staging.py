import errno
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

_spec = importlib.util.spec_from_file_location(
    "engram_staging_test", Path(__file__).resolve().parents[1] / "engram.py"
)
engram = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engram)


class StopBeforeCuda(Exception):
    pass


class EngramStagingTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path_patch = patch.object(engram, "Path", side_effect=self.staging_path)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def staging_path(self, value) -> Path:
        return self.root if value in ("/dev/shm", "/tmp") else Path(value)

    def test_reserves_shared_pages_before_loading_checkpoint(self) -> None:
        def loader() -> tuple[torch.Tensor, torch.Tensor]:
            partials = list(self.root.glob("*.partial.*"))
            self.assertEqual(len(partials), 1)
            self.assertGreaterEqual(partials[0].stat().st_blocks * 512, 66)
            raise StopBeforeCuda

        with self.assertRaises(StopBeforeCuda):
            engram._PinnedSharedTable(
                None, None, "reservation", "cuda:0", loader=loader, table_shape=(2, 32)
            )
        self.assertFalse(list(self.root.glob("*.partial.*")))

    def test_enospc_prevents_checkpoint_load_and_removes_failed_attempt(self) -> None:
        def loader() -> tuple[torch.Tensor, torch.Tensor]:
            self.fail("Checkpoint loaded after shared-memory reservation failed")

        with patch.object(
            engram.os, "posix_fallocate", side_effect=OSError(errno.ENOSPC, "full")
        ), self.assertRaises(OSError) as caught:
            engram._PinnedSharedTable(
                None, None, "full", "cuda:0", loader=loader, table_shape=(2, 32)
            )
        self.assertEqual(caught.exception.errno, errno.ENOSPC)
        self.assertFalse(list(self.root.glob("*.partial.*")))
        self.assertEqual([path.suffix for path in self.root.iterdir()], [".lock"])

    def test_loader_failure_preserves_other_partial_and_releases_reservation(
        self,
    ) -> None:
        other = self.root / f"rtp_llm_engram_{os.getuid()}_other.partial.{os.getpid()}"
        other.write_bytes(b"live other writer")

        def loader() -> tuple[torch.Tensor, torch.Tensor]:
            raise OSError(errno.EIO, "checkpoint read failed")

        with self.assertRaises(OSError) as caught:
            engram._PinnedSharedTable(
                None, None, "read-error", "cuda:0", loader=loader, table_shape=(2, 32)
            )
        self.assertEqual(caught.exception.errno, errno.EIO)
        self.assertEqual(list(self.root.glob("*.partial.*")), [other])
        self.assertEqual(other.read_bytes(), b"live other writer")

    def test_cache_hit_skips_reservation_and_checkpoint_load(self) -> None:
        weight = torch.arange(64, dtype=torch.uint8).reshape(2, 32)
        scales = torch.tensor([[11], [12]], dtype=torch.uint8)
        with patch.object(engram.torch, "frombuffer", side_effect=StopBeforeCuda):
            with self.assertRaises(StopBeforeCuda):
                engram._PinnedSharedTable(weight, scales, "hit", "cuda:0")

        def loader() -> tuple[torch.Tensor, torch.Tensor]:
            self.fail("Checkpoint loaded on a shared-table cache hit")

        with patch.object(
            engram.os,
            "posix_fallocate",
            side_effect=AssertionError("reserved cache hit"),
        ), patch.object(
            engram.torch, "frombuffer", side_effect=StopBeforeCuda
        ), self.assertRaises(
            StopBeforeCuda
        ):
            engram._PinnedSharedTable(
                None, None, "hit", "cuda:0", loader=loader, table_shape=(2, 32)
            )
        tables = [path for path in self.root.iterdir() if "." not in path.name]
        self.assertEqual(len(tables), 1)
        self.assertEqual(
            tables[0].read_bytes(), weight.numpy().tobytes() + scales.numpy().tobytes()
        )


if __name__ == "__main__":
    unittest.main()
