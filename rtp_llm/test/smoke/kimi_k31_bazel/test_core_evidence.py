import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm.test.smoke.kimi_k31_bazel.core_evidence import (
    MAX_EVIDENCE_BYTES,
    preserve_core_evidence,
)
from rtp_llm.test.smoke.kimi_k31_bazel.k31_ci import K31Worker


def write_sparse_core(path, machine=62):
    registers = bytearray(336)
    struct.pack_into("<i", registers, 0, 11)
    struct.pack_into("<i", registers, 32, 42)
    struct.pack_into("<Q", registers, 240, 0x456789)
    struct.pack_into("<Q", registers, 264, 0x700010)
    notes = struct.pack("<III", 5, len(registers), 1) + b"CORE\0\0\0\0" + registers
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF\x02\x01\x01" + bytes(9),
        4,
        machine,
        1,
        0,
        64,
        0,
        0,
        64,
        56,
        2,
        0,
        0,
        0,
    )
    note_header = struct.pack("<IIQQQQQQ", 4, 0, 4096, 0, 0, len(notes), len(notes), 4)
    stack_header = struct.pack("<IIQQQQQQ", 1, 6, 2**30, 0x700000, 0, 4096, 4096, 4096)
    with path.open("wb") as stream:
        stream.write(header + note_header + stack_header)
        stream.seek(4096)
        stream.write(notes)
        stream.seek(2**30)
        stream.write(b"STACK" + bytes(4091))


class CoreEvidenceTest(unittest.TestCase):
    def test_arm_core_keeps_notes_without_interpreting_x86_registers(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            write_sparse_core(task / "core-arm-42", machine=183)
            (record,) = preserve_core_evidence(task)["cores"]
            self.assertEqual(record["elf_machine"], 183)
            self.assertEqual(record["threads"], [])
            self.assertTrue(record["core_dump_remains_a_failure"])
            self.assertTrue(
                any(row["file"].startswith("notes-") for row in record["fragments"])
            )

    def test_sparse_core_retains_registers_and_stack_without_reading_the_hole(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            core = task / "core-rank-42"
            write_sparse_core(core)
            manifest = preserve_core_evidence(task)
            self.assertTrue(core.exists())
            (record,) = manifest["cores"]
            self.assertGreater(record["logical_bytes"], 2**30)
            self.assertLess(record["retained_bytes"], MAX_EVIDENCE_BYTES)
            self.assertEqual(record["threads"][0]["signal"], 11)
            self.assertEqual(record["threads"][0]["rip"], 0x456789)
            self.assertTrue(record["core_dump_remains_a_failure"])
            stack = task / "crash-evidence" / core.name / "stack-42.bin"
            self.assertTrue(stack.read_bytes().startswith(b"STACK"))

    def test_cleanup_still_fails_after_the_large_core_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            write_sparse_core(task / "core-rank-42")
            worker = K31Worker(threading.Event())
            worker.task = task
            with mock.patch(
                "rtp_llm.test.smoke.kimi_k31_bazel.k31_ci.Worker.close",
                side_effect=RuntimeError("core detected"),
            ):
                with self.assertRaisesRegex(RuntimeError, "core detected"):
                    worker.close()
            self.assertFalse((task / "core-rank-42").exists())
            self.assertTrue(
                json.loads((task / "crash-evidence.json").read_text())[
                    "core_dump_detected"
                ]
            )
            with mock.patch("rtp_llm.test.smoke.kimi_k31_bazel.k31_ci.Worker.close"):
                with self.assertRaisesRegex(RuntimeError, "core dumps"):
                    worker.close()

    def test_symlink_is_not_collected_or_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            original = task / "unrelated"
            original.write_bytes(b"keep")
            (task / "core-link").symlink_to(original)
            with self.assertRaises(ValueError):
                preserve_core_evidence(task)
            self.assertEqual(original.read_bytes(), b"keep")

    def test_malformed_large_table_preserves_the_original(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            core = task / "core-rank-42"
            write_sparse_core(core)
            with core.open("r+b") as stream:
                stream.seek(32)
                stream.write(struct.pack("<Q", 2**31))
            with self.assertRaises(ValueError):
                preserve_core_evidence(task)
            self.assertTrue(core.exists())


if __name__ == "__main__":
    unittest.main()
