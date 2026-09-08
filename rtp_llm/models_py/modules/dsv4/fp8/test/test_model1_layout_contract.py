from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_kv_quant_decode_op import (
    ENTRY_BYTES,
    NOPE_ROPE_STRIDE,
    SCALE_BYTES_PER_TOKEN,
    model1_block_payload_bytes,
    model1_slot_physical_offsets,
    read_model1_kv_slot_bytes,
)


_FIXTURE = Path(__file__).parent / "testdata" / "dsv4_model1_golden_v1.json"


def _sha256(raw: bytes | bytearray) -> str:
    return hashlib.sha256(raw).hexdigest()


def _logical_slot(slot: int) -> bytes:
    """Test-owned MODEL1 logical bytes; does not call production packing code."""
    data = bytes((slot * 37 + offset * 13 + 17) % 256 for offset in range(576))
    scales = bytes(
        [(slot * 11 + index * 7 + 101) % 256 for index in range(7)] + [0]
    )
    return data + scales


def _striped_block(entries: int, stride: int, tail_fill: int) -> bytearray:
    """Test-owned physical oracle: E*576 data, then E*8 scales, then tail."""
    raw = bytearray([tail_fill]) * stride
    for slot in range(entries):
        logical = _logical_slot(slot)
        data_start = slot * NOPE_ROPE_STRIDE
        scale_start = entries * NOPE_ROPE_STRIDE + slot * SCALE_BYTES_PER_TOKEN
        raw[data_start : data_start + NOPE_ROPE_STRIDE] = logical[
            :NOPE_ROPE_STRIDE
        ]
        raw[scale_start : scale_start + SCALE_BYTES_PER_TOKEN] = logical[
            NOPE_ROPE_STRIDE:
        ]
    return raw


class Model1LayoutContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(_FIXTURE.read_text())

    def test_fixture_schema_is_exact_model1_not_v32(self) -> None:
        self.assertEqual(self.fixture["logical_entry_bytes"], 584)
        self.assertEqual(self.fixture["data_bytes_per_entry"], 576)
        self.assertEqual(self.fixture["scale_bytes_per_entry"], 8)
        self.assertNotEqual(self.fixture["logical_entry_bytes"], 656)
        self.assertEqual(
            self.fixture["data_formula"],
            "(slot * 37 + offset * 13 + 17) % 256",
        )

    def test_test_owned_striped_golden_for_e1_e2_e256(self) -> None:
        tail_fill = int(self.fixture["tail_fill_byte"])
        for case in self.fixture["cases"]:
            entries = int(case["entries_per_block"])
            stride = int(case["physical_stride_bytes"])
            with self.subTest(entries=entries):
                raw = _striped_block(entries, stride, tail_fill)
                self.assertEqual(_sha256(raw), case["physical_sha256"])
                self.assertEqual(
                    _sha256(_logical_slot(entries - 1)),
                    case["last_logical_slot_sha256"],
                )
                self.assertEqual(model1_block_payload_bytes(entries), entries * 584)
                self.assertGreaterEqual(stride, entries * 584)
                self.assertEqual(stride % 576, 0)
                for slot in range(entries):
                    self.assertEqual(raw[entries * 576 + slot * 8 + 7], 0)

    def test_last_block_last_slot_materializes_from_padded_storage(self) -> None:
        tail_fill = int(self.fixture["tail_fill_byte"])
        for case in self.fixture["cases"]:
            entries = int(case["entries_per_block"])
            stride = int(case["physical_stride_bytes"])
            physical = _striped_block(entries, stride, tail_fill)
            with self.subTest(entries=entries):
                backing = torch.full((2, stride), tail_fill, dtype=torch.uint8)
                physical_tensor = torch.tensor(list(physical), dtype=torch.uint8)
                backing[0].copy_(physical_tensor)
                backing[1].copy_(physical_tensor)
                cache = backing.as_strided(
                    (2, entries, ENTRY_BYTES),
                    (stride, ENTRY_BYTES, 1),
                )

                logical = read_model1_kv_slot_bytes(
                    cache,
                    block_idx=1,
                    block_offset=entries - 1,
                    block_size=entries,
                )
                self.assertEqual(bytes(logical.tolist()), _logical_slot(entries - 1))
                self.assertEqual(int(logical[-1]), 0)

                data_offset, scale_offset = model1_slot_physical_offsets(
                    entries, entries - 1
                )
                self.assertEqual(data_offset, (entries - 1) * 576)
                self.assertEqual(scale_offset, entries * 576 + (entries - 1) * 8)
                self.assertTrue(torch.all(backing[1, entries * 584 :] == tail_fill))

    def test_rejects_storage_missing_last_block_tail(self) -> None:
        for case in self.fixture["cases"]:
            entries = int(case["entries_per_block"])
            stride = int(case["physical_stride_bytes"])
            payload = entries * ENTRY_BYTES
            with self.subTest(entries=entries):
                # The logical final slot fits, so as_strided itself is legal,
                # but the final physical block-tail padding is absent.
                backing = torch.empty(stride + payload, dtype=torch.uint8)
                cache = backing.as_strided(
                    (2, entries, ENTRY_BYTES),
                    (stride, ENTRY_BYTES, 1),
                )
                with self.assertRaisesRegex(ValueError, "final physical block stride"):
                    read_model1_kv_slot_bytes(cache, 1, entries - 1, entries)

    def test_rejects_nonstandard_logical_stride(self) -> None:
        entries = 2
        stride = 1728
        backing = torch.empty(stride, dtype=torch.uint8)
        cache = backing.as_strided(
            (1, entries, ENTRY_BYTES),
            (stride, ENTRY_BYTES + 1, 1),
        )
        with self.assertRaisesRegex(ValueError, "logical shape"):
            read_model1_kv_slot_bytes(cache, 0, 0, entries)

    def test_rejects_block_slot_oob_and_v32_656(self) -> None:
        backing = torch.empty((1, 1728), dtype=torch.uint8)
        cache = backing.as_strided((1, 2, ENTRY_BYTES), (1728, ENTRY_BYTES, 1))
        with self.assertRaisesRegex(IndexError, "block_idx"):
            read_model1_kv_slot_bytes(cache, 1, 0, 2)
        with self.assertRaisesRegex(IndexError, "block_offset"):
            read_model1_kv_slot_bytes(cache, 0, 2, 2)

        v32_cache = torch.empty((1, 2, 656), dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "must be 584 bytes"):
            read_model1_kv_slot_bytes(v32_cache, 0, 0, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for H2D/D2H checksum")
    def test_host_device_physical_checksum_matches_golden(self) -> None:
        tail_fill = int(self.fixture["tail_fill_byte"])
        for case in self.fixture["cases"]:
            entries = int(case["entries_per_block"])
            stride = int(case["physical_stride_bytes"])
            with self.subTest(entries=entries):
                host_raw = _striped_block(entries, stride, tail_fill)
                device_raw = torch.tensor(
                    list(host_raw), dtype=torch.uint8, device="cuda"
                )
                round_trip = bytes(device_raw.cpu().tolist())
                self.assertEqual(_sha256(round_trip), case["physical_sha256"])


if __name__ == "__main__":
    unittest.main()
