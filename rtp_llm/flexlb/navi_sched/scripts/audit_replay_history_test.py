import io
import os
import struct
import tempfile
import unittest

try:
    import msgpack
except ModuleNotFoundError:
    from pip._vendor import msgpack

from audit_replay_history import (HEADER, LEGACY_RECORD_VERSION, MAGIC,
                                  RECORD_VERSION, audit, classify)


def vint(value):
    data = bytearray()
    while value >= 128:
        data.append((value & 127) | 128)
        value >>= 7
    data.append(value)
    return bytes(data)


def payload(features=None, version=9, extra_frame=False):
    request = ["chatcmpl-test", [1, 2, 3], features, {}, None, 1.0]
    frames = [msgpack.packb(request, use_bin_type=True)]
    if extra_frame:
        frames.append(b"tensor")
    forward = msgpack.packb(frames, use_bin_type=True)
    return (struct.pack("<iQ", version, 42) + vint(3) + b"biz"
            + vint(6) + vint(32) + vint(1) + struct.pack("<Q", 17)
            + vint(120000) + vint(len(forward)) + forward)


def record(event, data, version=LEGACY_RECORD_VERSION):
    return HEADER.pack(MAGIC, version, event, len(data), 0) + data


class AuditTest(unittest.TestCase):
    def audit_bytes(self, data, **kwargs):
        with tempfile.TemporaryFile() as stream:
            stream.write(data)
            stream.flush()
            before = os.fstat(stream.fileno()).st_size
            result = audit(stream, **kwargs)
            self.assertEqual(os.fstat(stream.fileno()).st_size, before)
            stream.seek(0)
            self.assertEqual(stream.read(), data)
            return result

    def test_text_and_empty_features(self):
        for features in (None, []):
            data = payload(features)
            self.assertEqual(classify(io.BytesIO(data), len(data)), "text")

    def test_independent_history_payload_version(self):
        data = payload(version=1)
        self.assertEqual(classify(io.BytesIO(data), len(data), RECORD_VERSION), "text")
        result = self.audit_bytes(record(100, data, RECORD_VERSION))
        self.assertTrue(result["all_selected_text"])

    def test_multimodal(self):
        data = payload([{"modality": "image", "data": {"pixel_values": 0}}], extra_frame=True)
        self.assertEqual(classify(io.BytesIO(data), len(data)), "vl")

    def test_mixed_and_time_selection(self):
        data = record(100, payload()) + record(200, payload([{"modality": "image"}]))
        result = self.audit_bytes(data)
        self.assertEqual(result["counts"], {"text": 1, "vl": 1, "unknown": 0})
        self.assertFalse(result["all_selected_text"])
        result = self.audit_bytes(data, start_time_us=100, end_time_us=199)
        self.assertTrue(result["all_selected_text"])
        self.assertEqual(result["ranges"], {"text": {"earliest_us": 100, "latest_us": 100}})
        result = self.audit_bytes(data, start_time_us=100, end_time_us=200)
        self.assertFalse(result["all_selected_text"])
        self.assertEqual(result["counts"], {"text": 1, "vl": 1, "unknown": 0})

    def test_unknown_fails_closed(self):
        for data in (payload(version=8), payload(features={}), payload(extra_frame=True), b"broken"):
            result = self.audit_bytes(record(100, data))
            self.assertFalse(result["all_selected_text"])
            self.assertEqual(result["counts"]["unknown"], 1)
        result = self.audit_bytes(record(100, payload(version=2), RECORD_VERSION))
        self.assertEqual(result["counts"]["unknown"], 1)

    def test_truncated_tail_preserved(self):
        prefix = record(100, payload())
        for suffix in (b"tail", record(200, payload())[:-2]):
            result = self.audit_bytes(prefix + suffix)
            self.assertEqual(result["counts"]["text"], 1)
            self.assertTrue(result["truncated_tail"])
            self.assertFalse(result["all_selected_text"])

    def test_empty_and_invalid_header(self):
        self.assertFalse(self.audit_bytes(b"")["all_selected_text"])
        data = HEADER.pack(0, 1, 100, 0, 0)
        self.assertFalse(self.audit_bytes(data)["all_selected_text"])

    def test_forward_length_and_trailing_data(self):
        data = payload() + b"\x00"
        result = self.audit_bytes(record(100, data))
        self.assertEqual(result["counts"]["unknown"], 1)


if __name__ == "__main__":
    unittest.main()
