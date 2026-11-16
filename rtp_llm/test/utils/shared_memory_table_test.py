"""Shared host resource cleanup without Torch imports or CUDA context."""

import fcntl
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.utils import shared_memory_table as shared_table


class SharedTableLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.locks = self.directory / "locks"
        self.locks.mkdir()
        self.prefix = f"rtp_llm_engram_{os.getuid()}_"

    def table(self, index):
        path = self.directory / (self.prefix + f"{index:024x}")
        path.write_bytes(b"data" * 1024)
        return path

    def lease(self, table, pid):
        path = Path(str(table) + f".lease.{pid}." + "a" * 32)
        path.touch()
        return path

    def collect(self):
        return shared_table.reclaim_abandoned_tables(self.directory, self.locks)

    def test_reclaims_orphans_from_multiple_checkpoint_hashes(self):
        first, second = self.table(1), self.table(2)
        lease = self.lease(first, 900001)
        partial = Path(str(second) + ".partial.900002")
        partial.write_bytes(b"partial" * 1024)
        paths = [first, second, partial]
        allocated = sum(path.stat().st_blocks * 512 for path in paths)
        with patch.object(shared_table, "_process_alive", return_value=False):
            self.assertEqual(self.collect(), allocated)
        self.assertFalse(any(path.exists() for path in [*paths, lease]))
        self.assertEqual(len(list(self.locks.glob("*.lock"))), 2)

    def test_preserves_live_reader_and_prunes_only_dead_leases(self):
        table = self.table(1)
        active = self.lease(table, os.getpid())
        dead = self.lease(table, 900001)
        with patch.object(
            shared_table, "_process_alive", side_effect=lambda pid: pid == os.getpid()
        ):
            self.assertEqual(self.collect(), 0)
        self.assertTrue(table.exists())
        self.assertTrue(active.exists())
        self.assertFalse(dead.exists())

    def test_preserves_live_partial_writer(self):
        table = self.table(1)
        partial = Path(str(table) + f".partial.{os.getpid()}")
        table.rename(partial)
        with patch.object(shared_table, "_process_alive", return_value=True):
            self.assertEqual(self.collect(), 0)
        self.assertTrue(partial.exists())

    def test_skips_busy_creator_lock(self):
        table = self.table(1)
        lock_path = self.locks / (table.name + ".lock")
        with lock_path.open("a+b") as writer:
            fcntl.flock(writer, fcntl.LOCK_EX)
            self.assertEqual(self.collect(), 0)
            self.assertTrue(table.exists())
        self.assertGreater(self.collect(), 0)

    def test_preserves_other_uid_unknown_names_and_symlinks(self):
        foreign = self.directory / (f"rtp_llm_engram_{os.getuid() + 1}_" + "a" * 24)
        foreign.write_bytes(b"unrelated")
        unknown = self.directory / (self.prefix + "invalid")
        unknown.write_bytes(b"unrelated")
        link = self.directory / (self.prefix + "b" * 24)
        link.symlink_to(foreign)
        self.assertEqual(self.collect(), 0)
        self.assertTrue(link.is_symlink())
        self.assertEqual(foreign.read_bytes(), b"unrelated")
        self.assertTrue(unknown.exists())

    def test_rejects_foreign_owned_entry(self):
        table = self.table(1)
        original = Path.lstat

        def metadata(path):
            result = original(path)
            if path == table:
                values = list(result)
                values[4] = os.getuid() + 1
                return os.stat_result(values)
            return result

        with patch.object(Path, "lstat", metadata):
            self.assertEqual(self.collect(), 0)
        self.assertTrue(table.exists())

    def test_zombie_does_not_keep_lease_alive(self):
        with patch.object(shared_table.os, "kill"), patch.object(
            Path, "read_text", return_value="42 (worker name) Z 1 2 3"
        ):
            self.assertFalse(shared_table._process_alive(42))
        with patch.object(shared_table.os, "kill", side_effect=PermissionError):
            self.assertTrue(shared_table._process_alive(42))

    def test_current_process_is_alive(self):
        self.assertTrue(shared_table._process_alive(os.getpid()))

    def test_cleanup_is_limited_to_selected_namespace(self):
        original = self.table(1)
        self.prefix = f"rtp_llm_other_{os.getuid()}_"
        other = self.table(2)
        shared_table.reclaim_abandoned_tables(
            self.directory, self.locks, prefix="rtp_llm_other"
        )
        self.assertTrue(original.exists())
        self.assertFalse(other.exists())


if __name__ == "__main__":
    unittest.main()
