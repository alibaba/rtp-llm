import os
import tempfile
import unittest

from audit_replay_history_test import payload
from audit_replay_history import (HEADER, LEGACY_RECORD_VERSION, MAGIC,
                                  RECORD_VERSION, audit)
from filter_replay_history import crc32c, filter_history


def record(event, data, version=LEGACY_RECORD_VERSION):
    return HEADER.pack(MAGIC, version, event, len(data), crc32c(data)) + data


class FilterHistoryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = os.path.join(self.directory.name, "source.bin")
        self.output = os.path.join(self.directory.name, "text.bin")

    def write_source(self, data):
        with open(self.source, "wb") as stream:
            stream.write(data)

    def test_crc32c_matches_standard_vector_and_chunking(self):
        self.assertEqual(crc32c(b"123456789"), 0xE3069283)
        self.assertEqual(crc32c(b"456789", crc32c(b"123")), 0xE3069283)

    def test_removes_vl_and_preserves_text_records(self):
        first = record(100, payload())
        vl = record(200, payload([{"modality": "image"}], extra_frame=True))
        last = record(300, payload(features=[], version=1), RECORD_VERSION)
        original = first + vl + last
        self.write_source(original)
        result = filter_history(self.source, self.output)
        self.assertEqual(result["counts"], {"text": 2, "vl": 1})
        self.assertEqual(result["earliest_event_time_us"], 100)
        self.assertEqual(result["latest_event_time_us"], 300)
        self.assertEqual(result["output_bytes"], len(first + last))
        with open(self.source, "rb") as stream:
            self.assertEqual(stream.read(), original)
        with open(self.output, "rb") as stream:
            self.assertEqual(stream.read(), first + last)
            stream.seek(0)
            self.assertTrue(audit(stream)["all_selected_text"])

    def test_corrupt_unknown_and_truncated_tail_publish_nothing(self):
        valid = record(100, payload())
        invalid = record(200, payload(version=8))
        corrupted = bytearray(record(200, payload()))
        corrupted[-1] ^= 1
        for suffix in (invalid, bytes(corrupted), b"tail", record(200, payload())[:-1]):
            with self.subTest(suffix=suffix[:4]):
                self.write_source(valid + suffix)
                with self.assertRaises(ValueError):
                    filter_history(self.source, self.output)
                self.assertFalse(os.path.exists(self.output))
                self.assertFalse(os.path.exists(self.output + ".tmp"))
        self.write_source(valid)
        self.assertEqual(filter_history(self.source, self.output)["counts"]["text"], 1)

    def test_limits_existing_output_and_no_text(self):
        first = record(100, payload())
        self.write_source(first + record(200, payload()))
        with self.assertRaisesRegex(ValueError, "exceeds output limits"):
            filter_history(self.source, self.output, max_records=1)
        self.assertFalse(os.path.exists(self.output))
        with self.assertRaisesRegex(ValueError, "exceeds output limits"):
            filter_history(self.source, self.output, max_bytes=len(first) - 1)
        self.assertFalse(os.path.exists(self.output))
        self.write_source(record(100, payload([{"modality": "image"}])))
        with self.assertRaisesRegex(ValueError, "no text requests"):
            filter_history(self.source, self.output)
        self.assertFalse(os.path.exists(self.output))
        self.write_source(first)
        filter_history(self.source, self.output)
        with self.assertRaises(FileExistsError):
            filter_history(self.source, self.output)

    def test_recovers_stale_temporary_output(self):
        valid = record(100, payload())
        self.write_source(valid)
        with open(self.output + ".tmp", "wb") as stream:
            stream.write(b"partial old run")
        filter_history(self.source, self.output)
        self.assertFalse(os.path.exists(self.output + ".tmp"))
        with open(self.output, "rb") as stream:
            self.assertEqual(stream.read(), valid)


if __name__ == "__main__":
    unittest.main()
