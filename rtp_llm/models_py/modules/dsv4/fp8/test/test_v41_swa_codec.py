"""Byte-level oracle and CUDA graph tests for V4.1's 528B MXFP8 SWA cache."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_swa_triton as codec


def _reference(values):
    """Independent CPU quantization and page-plane bytes, including RoPE."""
    groups = values.detach().cpu().float().reshape(-1, 16, 32)
    maximum = groups.abs().amax(-1).clamp_min(1e-4)
    exponent = (maximum / 448.0).log2().ceil()
    scales = exponent.exp2()
    payload = (groups / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    recovered = (payload.float() * scales[..., None]).reshape(-1, 512)
    return (
        payload.view(torch.uint8).reshape(-1, 512),
        (exponent + 127).to(torch.uint8),
        recovered,
    )


def _pool(blocks, entries, padding=384):
    stride = entries * 528 + padding
    raw = torch.full((blocks, stride), 0x5A, dtype=torch.uint8, device="cuda")
    return raw, raw.as_strided((blocks, entries, 528), (stride, 528, 1))


class V41SwaCodecContractTest(unittest.TestCase):
    def test_cpu_and_v4_pools_are_not_supported(self):
        slots = torch.tensor([0], dtype=torch.int64)
        for size in (528, 584):
            pool = torch.zeros(1, 128, size, dtype=torch.uint8)
            self.assertFalse(codec.is_supported(pool, slots))
            with self.assertRaises(ValueError):
                codec.quantize_and_insert_swa_k_cache(torch.zeros(1, 512), pool, slots)

    def test_reference_zero_and_literal_power_of_two_scales(self):
        groups = (
            torch.tensor([0.0, 448.0, 224.0, 896.0]).repeat_interleave(32).repeat(4)
        )
        payload, scales, recovered = _reference(groups.reshape(1, 512))
        self.assertEqual(scales.tolist(), [[105, 127, 126, 128] * 4])
        # E4M3 positive max 448 is byte 0x7e. Each nonzero group rescales to it.
        self.assertEqual(payload[0, :128].tolist(), [0] * 32 + [0x7E] * 96)
        torch.testing.assert_close(recovered, groups.reshape(1, 512), rtol=0, atol=0)


class V41SwaCodecGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
            raise unittest.SkipTest("MXFP8 codec tests require SM90 or later")

    def test_planar_bytes_roundtrip_padding_and_masked_slots(self):
        torch.manual_seed(410528)
        for entries in (64, 128, 136, 256):
            with self.subTest(entries=entries):
                raw, pool = _pool(5, entries)
                # Non-contiguous input rows, int32 slots, page boundaries and ring tail.
                values = torch.randn(9, 528, dtype=torch.bfloat16, device="cuda")[
                    :, :512
                ]
                values[0].zero_()
                values[1].fill_(1.5)
                values[2, 448:].fill_(448.0)
                slots = torch.tensor(
                    [
                        entries,
                        entries + 1,
                        2 * entries - 1,
                        2 * entries,
                        -1,
                        3 * entries,
                        3 * entries + 5,
                        4 * entries - 1,
                        4 * entries,
                    ],
                    dtype=torch.int32,
                    device="cuda",
                )
                payload, scales, recovered = _reference(values)
                expected = raw.cpu().clone()
                for row, slot in enumerate(slots.cpu().tolist()):
                    if slot < 0:
                        continue
                    page, position = divmod(slot, entries)
                    expected[page, position * 512 : (position + 1) * 512] = payload[row]
                    begin = entries * 512 + position * 16
                    expected[page, begin : begin + 16] = scales[row]
                codec.quantize_and_insert_swa_k_cache(values, pool, slots)
                torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)
                # Deliberately strided output; padding must remain untouched.
                backing = torch.full((9, 544), -73.0, device="cuda")
                output = backing[:, :512]
                self.assertIs(
                    codec.dequantize_swa_k_cache(pool, slots, out=output), output
                )
                recovered[4].zero_()
                torch.testing.assert_close(output.cpu(), recovered, rtol=0, atol=0)
                self.assertTrue(backing[:, 512:].eq(-73).all())

    def test_gather_lengths_offset_strides_and_all_masked(self):
        torch.manual_seed(528)
        _, pool = _pool(2, 136)
        values = torch.randn(6, 512, dtype=torch.bfloat16, device="cuda")
        slots = torch.tensor([136, 137, 138, 139, 140, 141], device="cuda")
        codec.quantize_and_insert_swa_k_cache(values, pool, slots)
        _, _, recovered = _reference(values)
        storage = torch.tensor(
            [[136, 0, 137, 0, -1, 0, 138, 0], [139, 0, 140, 0, 141, 0, -1, 0]],
            device="cuda",
            dtype=torch.int32,
        )
        mapping = storage[:, ::2]
        backing = torch.full((2, 8, 528), -13.0, device="cuda", dtype=torch.bfloat16)
        out = backing[:, :, :512]
        lengths = torch.tensor([4, 2], dtype=torch.int32, device="cuda")
        codec.dequantize_and_gather_k_cache_slots(out, pool, mapping, lengths, 2)
        expected = torch.full((2, 8, 528), -13.0, dtype=torch.bfloat16)
        expected[0, 2:6, :512] = torch.stack(
            (recovered[0], recovered[1], torch.zeros(512), recovered[2])
        )
        expected[1, 2:4, :512] = recovered[3:5]
        torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
        mapping.fill_(-1)
        codec.dequantize_and_gather_k_cache_slots(out, pool, mapping, None, 1)
        self.assertTrue(out[:, 1:5].eq(0).all())

    def test_graph_replay_reads_changed_slots_lengths_and_keys(self):
        _, pool = _pool(2, 136)
        values = torch.ones(4, 512, dtype=torch.bfloat16, device="cuda")
        slots = torch.tensor([136, 137, 138, 139], device="cuda")
        mapping = slots.view(2, 2)
        lengths = torch.tensor([2, 1], dtype=torch.int32, device="cuda")
        out = torch.empty(2, 3, 512, dtype=torch.bfloat16, device="cuda")

        def run():
            codec.quantize_and_insert_swa_k_cache(values, pool, slots)
            out.fill_(-7)
            codec.dequantize_and_gather_k_cache_slots(out, pool, mapping, lengths, 1)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        slots.copy_(torch.tensor([140, 141, -1, 143], device="cuda"))
        values.fill_(2)
        lengths.copy_(torch.tensor([1, 2], dtype=torch.int32, device="cuda"))
        graph.replay()
        expected = torch.full((2, 3, 512), -7.0, dtype=torch.bfloat16)
        expected[0, 1].fill_(2)
        expected[1, 1].zero_()
        expected[1, 2].fill_(2)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    def test_cp_four_byte_slices_restore_payload_and_scale_planes(self):
        torch.manual_seed(41136)
        entries, cp_size, local_bytes = 136, 4, 18048
        slots = torch.tensor(
            [entries + 1, entries + 34, 4 * entries - 1], device="cuda"
        )
        compact = torch.tensor([1, 34, 2 * entries - 1], device="cuda")
        unique = torch.tensor([1, 3], device="cuda")
        metadata = SimpleNamespace(
            unique_blocks=unique, compact_slots=compact, contiguous_block_start=-1
        )
        values = torch.randn(3, 512, dtype=torch.bfloat16, device="cuda")
        pools = [
            torch.full((5, local_bytes), 0x5A, dtype=torch.uint8, device="cuda")
            for _ in range(cp_size)
        ]
        for rank, raw in enumerate(pools):
            codec.quantize_and_insert_k_cache_cp_byte_sliced(
                values, raw, slots, entries, rank, cp_size, metadata
            )
            self.assertTrue(raw[0].eq(0x5A).all())
            self.assertTrue(raw[2].eq(0x5A).all())
            self.assertTrue(raw[4].eq(0x5A).all())
        gathered = torch.cat([raw.index_select(0, unique) for raw in pools], dim=0)
        full = gathered.view(cp_size, 2, local_bytes).permute(1, 0, 2).reshape(2, -1)
        payload, scales, recovered = _reference(values)
        expected = torch.full((2, local_bytes * cp_size), 0x5A, dtype=torch.uint8)
        for row, slot in enumerate(compact.cpu().tolist()):
            page, position = divmod(slot, entries)
            expected[page, position * 512 : (position + 1) * 512] = payload[row]
            start = entries * 512 + position * 16
            expected[page, start : start + 16] = scales[row]
        torch.testing.assert_close(full.cpu(), expected, rtol=0, atol=0)

        mapping = slots.reshape(1, 3)
        read_meta = SimpleNamespace(
            unique_blocks=unique,
            compact_slots=compact.reshape(1, 3),
            contiguous_block_start=-1,
        )
        out = torch.full((1, 5, 512), -3.0, dtype=torch.bfloat16, device="cuda")
        lengths = torch.tensor([2], dtype=torch.int32, device="cuda")
        # Four byte slices are real; only the collective transport is mocked.
        with patch(
            "rtp_llm.models_py.distributed.collective_torch.all_gather",
            return_value=gathered,
        ) as gather:
            codec.dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                out, pools[0], mapping, lengths, 1, entries, 0, cp_size, read_meta
            )
        self.assertEqual(gather.call_count, 1)
        torch.testing.assert_close(
            out[0, 1:3].cpu(), recovered[:2].bfloat16(), rtol=0, atol=0
        )
        self.assertTrue(out[:, 3:].eq(-3).all())

    def test_cp_partial_page_updates_preserve_old_slots_and_padding(self):
        torch.manual_seed(41137)
        entries, cp_size, local_bytes = 136, 4, 18048
        pools = [
            torch.full((5, local_bytes), 0x5A, dtype=torch.uint8, device="cuda")
            for _ in range(cp_size)
        ]
        expected = torch.full((5, local_bytes * cp_size), 0x5A, dtype=torch.uint8)
        updates = (
            (
                [entries + 1, entries + 2, entries + 3, 4 * entries - 1],
                [1, 3],
                [1, 2, 3, 2 * entries - 1],
            ),
            # Update one existing slot, append a new slot and skip a masked
            # row. The previous values at slots 1/3 and page 3 must survive.
            ([entries + 2, entries + 40, -1], [1], [2, 40, -1]),
        )
        for physical, blocks, compact in updates:
            slots = torch.tensor(physical, dtype=torch.int64, device="cuda")
            metadata = SimpleNamespace(
                unique_blocks=torch.tensor(blocks, dtype=torch.int64, device="cuda"),
                compact_slots=torch.tensor(compact, dtype=torch.int64, device="cuda"),
                contiguous_block_start=-1,
            )
            values = torch.randn(
                len(physical), 512, dtype=torch.bfloat16, device="cuda"
            )
            payload, scales, _ = _reference(values)
            for row, slot in enumerate(physical):
                if slot < 0:
                    continue
                page, position = divmod(slot, entries)
                expected[page, position * 512 : (position + 1) * 512] = payload[row]
                start = entries * 512 + position * 16
                expected[page, start : start + 16] = scales[row]
            for rank, raw in enumerate(pools):
                codec.quantize_and_insert_k_cache_cp_byte_sliced(
                    values, raw, slots, entries, rank, cp_size, metadata
                )
            actual = torch.stack(pools).permute(1, 0, 2).reshape(5, -1)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            self.assertTrue(actual[:, entries * 528 :].eq(0x5A).all())

    def test_cp_empty_compaction_zeros_only_valid_gather_rows(self):
        raw = torch.empty((1, 18048), dtype=torch.uint8, device="cuda")
        slots = torch.full((2, 3), -1, dtype=torch.int64, device="cuda")
        metadata = SimpleNamespace(
            unique_blocks=torch.empty(0, dtype=torch.int64, device="cuda"),
            compact_slots=slots,
            contiguous_block_start=-1,
        )
        out = torch.full((2, 4, 512), 7.0, dtype=torch.bfloat16, device="cuda")
        lengths = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
        codec.dequantize_and_gather_k_cache_slots_cp_byte_sliced(
            out, raw, slots, lengths, 1, 136, 0, 4, metadata
        )
        self.assertTrue(out[0, 1:].eq(0).all())
        self.assertTrue(out[1, 1].eq(0).all())
        self.assertTrue(out[1, 2:].eq(7).all())


if __name__ == "__main__":
    unittest.main()
