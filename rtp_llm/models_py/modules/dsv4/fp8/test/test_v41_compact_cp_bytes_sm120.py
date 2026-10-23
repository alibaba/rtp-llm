"""Independent wire oracle and planar pool installation across CP ranks."""

import struct
import unittest
from dataclasses import replace

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.fp8._v41_compact_cp_plan import CompactCPPlan
from rtp_llm.models_py.modules.dsv4.fp8._v41_compact_cp_transport import (
    pack,
    prepare_metadata,
    prepare_metadata_batch,
    unpack,
)
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache


def operands(plan, device):
    rows = plan.rows
    main = (
        torch.arange(rows * 288, device=device).reshape(rows, 288) * 7 + plan.rank * 11
    ).to(torch.uint8)
    index = (
        torch.arange(rows * 68, device=device).reshape(rows, 68) * 13 + plan.rank * 19
    ).to(torch.uint8)
    tail = (
        torch.arange(2048, device=device, dtype=torch.float32).reshape(2, 1024) / 32
        if plan.tail_count
        else None
    )
    return main, index, tail


def oracle(plan, main, index, tail):
    result = bytearray(struct.pack("<16q", *plan.header))
    main_bytes = main.cpu().numpy().tobytes()
    index_bytes = index.cpu().numpy().tobytes()
    for i, position in enumerate(plan.boundary_positions):
        result.extend(struct.pack("<q", position))
        result.extend(main_bytes[i * 288 : (i + 1) * 288])
        result.extend(index_bytes[i * 68 : (i + 1) * 68])
    if plan.ratio == 2:
        for i in range(2):
            result.extend(
                struct.pack(
                    "<2q",
                    plan.start + 4094 + i if plan.tail_count else -1,
                    int(bool(plan.tail_count)),
                )
            )
            result.extend(
                tail[i].cpu().numpy().tobytes() if tail is not None else bytes(4096)
            )
    return bytes(result)


class CompactCPBytes(unittest.TestCase):
    def check_device(self, device):
        for layer, ratio in ((2, 2), (20, 1)):
            plans = [
                CompactCPPlan("packed", -9, 2, 7, 3, 11, 4096, rank, layer, ratio)
                for rank in range(4)
            ]
            metadata = prepare_metadata_batch(plans, device)
            for plan, actual in zip(plans, metadata):
                expected = prepare_metadata(plan, device)
                for field in ("header", "positions", "tail_identity"):
                    a, b = getattr(actual, field), getattr(expected, field)
                    if a is None:
                        self.assertIsNone(b)
                    else:
                        self.assertTrue(torch.equal(a, b))
            packets, expected_main, expected_index = [], [], []
            for plan, meta in zip(plans, metadata):
                main, index, tail = operands(plan, device)
                payload = index[:, :64].contiguous().view(torch.int8)
                scales = index[:, 64:].contiguous().view(torch.int32).flatten()
                packet = pack(plan, meta, main, payload, scales, tail)
                self.assertEqual(
                    packet.cpu().numpy().tobytes(), oracle(plan, main, index, tail)
                )
                packets.append(packet)
                expected_main.append(main)
                expected_index.append(index)
            received = torch.stack(packets)
            main, index, tail = unpack(plans, metadata, received)
            torch.testing.assert_close(main, torch.cat(expected_main), rtol=0, atol=0)
            torch.testing.assert_close(index, torch.cat(expected_index), rtol=0, atol=0)
            if ratio == 2:
                torch.testing.assert_close(
                    tail, operands(plans[0], device)[2], rtol=0, atol=0
                )
            else:
                self.assertIsNone(tail)
            positions = [
                position for plan in plans for position in plan.boundary_positions
            ]
            for receiver in range(4):
                for tag, rows, width in (
                    ("csa_kv" if ratio == 2 else "hca_kv", main, 288),
                    ("indexer_kv", index, 68),
                ):
                    layout = dsv41_page_layouts(0)[tag]
                    padding = 37
                    raw = torch.full(
                        (35, layout.stride_bytes + padding),
                        173,
                        dtype=torch.uint8,
                        device=device,
                    )
                    cache = V41PagedCache(raw[:, : layout.stride_bytes], tag, 0)
                    # Main and index tags intentionally use different permutations.
                    shift = receiver + (5 if tag == "indexer_kv" else 0)
                    pages = [1 + ((i * 7 + shift) % 33) for i in range(32)]
                    slots = torch.tensor(
                        [
                            pages[position // 256] * layout.entries
                            + (position % 256) // ratio
                            for position in positions
                        ],
                        device=device,
                        dtype=torch.int64,
                    )
                    expected = bytearray(raw.cpu().numpy().tobytes())
                    row_bytes = rows.cpu().numpy().tobytes()
                    stride = raw.shape[1]
                    for i, position in enumerate(positions):
                        base = pages[position // 256] * stride
                        entry = (position % 256) // ratio
                        a = base + entry * layout.payload_bytes_per_entry
                        b = (
                            base
                            + layout.scale_plane_offset
                            + entry * layout.scale_bytes_per_entry
                        )
                        record = row_bytes[i * width : (i + 1) * width]
                        expected[a : a + layout.payload_bytes_per_entry] = record[
                            : layout.payload_bytes_per_entry
                        ]
                        expected[b : b + layout.scale_bytes_per_entry] = record[
                            layout.payload_bytes_per_entry :
                        ]
                    cache.scatter(
                        rows, slots, backend="triton" if device == "cuda" else "eager"
                    )
                    self.assertEqual(raw.cpu().numpy().tobytes(), bytes(expected))
            if device == "cpu":
                for offset in (
                    0,
                    32,
                    128,
                    128 + 364,
                    plans[0].send_bytes - 4112 if ratio == 2 else 8,
                ):
                    bad = received.clone()
                    bad[1, offset] ^= 1
                    with self.assertRaises(ValueError):
                        unpack(plans, metadata, bad)
                with self.assertRaises(ValueError):
                    unpack([replace(plans[0], rank=1), *plans[1:]], metadata, received)

    def test_cpu_oracle(self):
        self.check_device("cpu")

    def test_sm120_packet_and_planar_guards(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        self.check_device("cuda")


if __name__ == "__main__":
    unittest.main()
