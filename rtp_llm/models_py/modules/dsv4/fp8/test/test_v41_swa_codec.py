"""Byte-level oracle and CUDA graph tests for V4.1's 528B MXFP8 SWA cache."""

from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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


def _staged_cp_write(values, raw, slots, rank, metadata, entries=136):
    """Original copy-in/quantize/copy-back path as an independent address oracle."""
    local_bytes = raw.shape[1]
    full = torch.empty(
        (metadata.unique_blocks.numel(), local_bytes * 4),
        dtype=torch.uint8,
        device=raw.device,
    )
    local = full[:, rank * local_bytes : (rank + 1) * local_bytes]
    local.copy_(raw.index_select(0, metadata.unique_blocks))
    view = full.as_strided((len(full), entries, 528), (local_bytes * 4, 528, 1))
    codec.quantize_and_insert_swa_k_cache(values, view, metadata.compact_slots)
    raw.index_copy_(0, metadata.unique_blocks, local.contiguous())


class V41SwaCodecContractTest(unittest.TestCase):
    def test_136_entry_byte_owners_split_tokens_and_exclude_padding(self):
        local = 18048
        for token, first, sizes in (
            (35, 0, (128, 384)),
            (70, 1, (256, 256)),
            (105, 2, (384, 128)),
        ):
            owners = [byte // local for byte in range(token * 512, (token + 1) * 512)]
            self.assertEqual(
                [owners.count(first), owners.count(first + 1)], list(sizes)
            )
        self.assertEqual({byte // local for byte in range(136 * 512, 136 * 528)}, {3})
        self.assertEqual(4 * local - 136 * 528, 384)

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

    def test_cp_direct_writer_all_slots_nonfinite_strides_and_padding(self):
        torch.manual_seed(4137)
        entries, local = 136, 18048
        unique = torch.tensor([1, 3], device="cuda")
        compact = torch.tensor(list(range(272)) + [-1, -9], device="cuda")
        slots = torch.tensor(
            list(range(136, 272)) + list(range(408, 544)) + [-1, -9], device="cuda"
        )
        meta = SimpleNamespace(
            unique_blocks=unique, compact_slots=compact, contiguous_block_start=-1
        )
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            values = torch.randn(len(slots), 528, dtype=dtype, device="cuda")[:, :512]
            values[0].zero_()
            values[35].fill_(448)
            values[70].fill_(-896)
            values[105, ::3] = torch.nan
            values[135, ::5] = torch.inf
            values[136, ::7] = -torch.inf
            values[137].fill_(torch.finfo(dtype).max)
            for rank in range(4):
                with self.subTest(dtype=dtype, rank=rank):
                    backing = torch.full(
                        (5, local + 64), 0x5A, dtype=torch.uint8, device="cuda"
                    )
                    expected = backing.clone()
                    _staged_cp_write(values, expected[:, :local], slots, rank, meta)
                    codec.quantize_and_insert_k_cache_cp_byte_sliced(
                        values, backing[:, :local], slots, entries, rank, 4, meta
                    )
                    self.assertTrue(torch.equal(backing, expected))
                    self.assertTrue(backing[:, local:].eq(0x5A).all())
                    if rank == 3:
                        self.assertTrue(backing[:, local - 384 :].eq(0x5A).all())

    def test_cp_rank_major_read_all_entries_lengths_and_strided_output(self):
        torch.manual_seed(4138)
        full, pool = _pool(5, 136)
        values = torch.randn(272, 512, dtype=torch.bfloat16, device="cuda")
        physical = torch.cat(
            (
                torch.arange(136, 272, device="cuda"),
                torch.arange(408, 544, device="cuda"),
            )
        )
        codec.quantize_and_insert_swa_k_cache(values, pool, physical)
        unique = torch.tensor([1, 3], device="cuda")
        gathered = (
            full[unique].view(2, 4, 18048).permute(1, 0, 2).contiguous().view(8, 18048)
        )
        mapping = torch.cat(
            (physical.view(2, 136), torch.full((2, 2), -1, device="cuda")), 1
        )
        compact = torch.cat(
            (
                torch.arange(272, device="cuda").view(2, 136),
                torch.full((2, 2), -1, device="cuda"),
            ),
            1,
        )
        meta = SimpleNamespace(
            unique_blocks=unique, compact_slots=compact, contiguous_block_start=-1
        )
        lengths = torch.tensor([138, 107], dtype=torch.int32, device="cuda")
        expected = torch.full((2, 142, 528), -37, dtype=torch.bfloat16, device="cuda")
        codec.dequantize_and_gather_k_cache_slots(
            expected[:, :, :512], pool, mapping, lengths, 2
        )
        for rank in range(4):
            backing = torch.full_like(expected, -37)
            raw = full[:, rank * 18048 : (rank + 1) * 18048]
            with patch(
                "rtp_llm.models_py.distributed.collective_torch.all_gather",
                return_value=gathered,
            ) as collective:
                codec.dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                    backing[:, :, :512], raw, mapping, lengths, 2, 136, rank, 4, meta
                )
            collective.assert_called_once()
            self.assertTrue(
                torch.equal(backing.view(torch.int16), expected.view(torch.int16))
            )

    def test_cp_changed_maps_graph_and_cross_stream_order(self):
        full, pool = _pool(5, 136)
        raw = [
            torch.full((5, 18048), 0x5A, dtype=torch.uint8, device="cuda")
            for _ in range(4)
        ]
        unique = torch.tensor([1, 3], device="cuda")
        compact = torch.tensor([35, 70, 105, 271, -1], device="cuda")
        physical = torch.tensor([171, 206, 241, 543, -1], device="cuda")
        values = torch.ones(5, 512, dtype=torch.bfloat16, device="cuda")
        lengths = torch.tensor([5], dtype=torch.int32, device="cuda")
        meta = SimpleNamespace(
            unique_blocks=unique, compact_slots=compact, contiguous_block_start=-1
        )
        out = torch.full((1, 7, 512), -37, dtype=torch.bfloat16, device="cuda")
        expected = torch.full_like(out, -37)

        def run():
            for rank in range(4):
                codec.quantize_and_insert_k_cache_cp_byte_sliced(
                    values, raw[rank], physical, 136, rank, 4, meta
                )
            # Four real byte slices; only transport is simulated on one GPU.
            gathered = torch.cat([part.index_select(0, unique) for part in raw])
            codec._gather_swa_rank_major(
                out, gathered, compact.view(1, 5), lengths, 1, 2, 18048, 136
            )

        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
        torch.cuda.current_stream().wait_stream(producer)
        for changed in (False, True):
            if changed:
                unique.copy_(torch.tensor([3, 1], device="cuda"))
                compact.copy_(torch.tensor([135, 171, -1, 241, 206], device="cuda"))
                physical.copy_(torch.tensor([543, 171, -1, 241, 206], device="cuda"))
                values.fill_(2)
                lengths.fill_(3)
            out.fill_(-37)
            expected.fill_(-37)
            producer.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(producer):
                graph.replay()
                ready = producer.record_event()
            consumer.wait_event(ready)
            with torch.cuda.stream(consumer):
                codec.quantize_and_insert_swa_k_cache(values, pool, physical)
                codec.dequantize_and_gather_k_cache_slots(
                    expected, pool, physical.view(1, 5), lengths, 1
                )
                observed = out.clone()
            torch.cuda.current_stream().wait_stream(consumer)
            self.assertTrue(
                torch.equal(observed.view(torch.int16), expected.view(torch.int16))
            )
            combined = torch.stack(raw).permute(1, 0, 2).reshape(5, -1)
            self.assertTrue(torch.equal(combined, full))

    def test_cp_all_negative_writer_and_zero_length_reader_preserve_every_byte(self):
        raw = torch.full((4, 18048), 0x5A, dtype=torch.uint8, device="cuda")
        slots = torch.full((3,), -9, dtype=torch.int64, device="cuda")
        meta = SimpleNamespace(
            unique_blocks=torch.tensor([1, 3], device="cuda"),
            compact_slots=slots,
            contiguous_block_start=-1,
        )
        for rank in range(4):
            codec.quantize_and_insert_k_cache_cp_byte_sliced(
                torch.full((3, 512), torch.nan, device="cuda"),
                raw,
                slots,
                136,
                rank,
                4,
                meta,
            )
        self.assertTrue(raw.eq(0x5A).all())
        out = torch.full((1, 5, 512), -37, dtype=torch.bfloat16, device="cuda")
        codec._gather_swa_rank_major(
            out,
            torch.empty(8, 18048, dtype=torch.uint8, device="cuda"),
            slots.view(1, 3),
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            1,
            2,
            18048,
            136,
        )
        self.assertTrue(out.eq(-37).all())

    def test_cp_prefix_read_completes_before_ring_overwrite(self):
        raw = [
            torch.full((3, 18048), 0x5A, dtype=torch.uint8, device="cuda")
            for _ in range(4)
        ]
        unique = torch.tensor([1, 2], device="cuda")
        compact = torch.tensor([35, 70, 105, 171], device="cuda")
        physical = compact + 136
        meta = SimpleNamespace(
            unique_blocks=unique, compact_slots=compact, contiguous_block_start=-1
        )
        previous = torch.ones(4, 512, dtype=torch.bfloat16, device="cuda")
        fresh = torch.full_like(previous, 2)
        prefix = torch.empty(1, 4, 512, dtype=torch.bfloat16, device="cuda")
        current = torch.empty_like(prefix)
        lengths = torch.tensor([4], dtype=torch.int32, device="cuda")
        for rank in range(4):
            codec.quantize_and_insert_k_cache_cp_byte_sliced(
                previous, raw[rank], physical, 136, rank, 4, meta
            )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gathered = torch.cat([r.index_select(0, unique) for r in raw])
            codec._gather_swa_rank_major(
                prefix, gathered, compact.view(1, 4), lengths, 0, 2, 18048, 136
            )
            for rank in range(4):
                codec.quantize_and_insert_k_cache_cp_byte_sliced(
                    fresh, raw[rank], physical, 136, rank, 4, meta
                )
            gathered = torch.cat([r.index_select(0, unique) for r in raw])
            codec._gather_swa_rank_major(
                current, gathered, compact.view(1, 4), lengths, 0, 2, 18048, 136
            )
        torch.cuda.current_stream().wait_stream(stream)
        self.assertTrue(prefix.eq(1).all())
        self.assertTrue(current.eq(2).all())

    def test_cp_unsupported_address_metadata_uses_staged_fallback(self):
        for entries, strided in ((128, False), (136, True)):
            slots = torch.tensor([1, 35], device="cuda")
            storage = torch.tensor([1, 0, 35, 0], device="cuda")
            compact = storage[::2] if strided else slots
            meta = SimpleNamespace(
                unique_blocks=torch.tensor([1], device="cuda"),
                compact_slots=compact,
                contiguous_block_start=-1,
            )
            values = torch.ones(2, 512, device="cuda")
            raw = torch.full((3, 18048), 0x5A, dtype=torch.uint8, device="cuda")
            expected = raw.clone()
            _staged_cp_write(values, expected, slots, 0, meta, entries)
            with patch.object(
                codec,
                "quantize_and_insert_swa_k_cache",
                wraps=codec.quantize_and_insert_swa_k_cache,
            ) as legacy:
                codec.quantize_and_insert_k_cache_cp_byte_sliced(
                    values, raw, slots, entries, 0, 4, meta
                )
            legacy.assert_called_once()
            self.assertTrue(torch.equal(raw, expected))


class V41SwaGatherOutputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    @contextlib.contextmanager
    def collective(self, expected):
        from rtp_llm.models_py.distributed import collective_torch as collective

        group = object()
        symm = Mock()
        symm.should_torch_symm_mem_allgather.return_value = False

        def gather(out, local, *, group):
            self.assertEqual(out.shape, (4 * local.shape[0], local.shape[1]))
            self.assertEqual(out.dtype, torch.uint8)
            self.assertNotEqual(out.data_ptr(), local.data_ptr())
            out.copy_(expected)

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(torch.distributed, "is_initialized", return_value=True)
            )
            stack.enter_context(
                patch.object(torch.distributed, "get_world_size", return_value=4)
            )
            stack.enter_context(
                patch.object(torch.distributed, "get_backend", return_value="nccl")
            )
            stack.enter_context(
                patch.object(collective, "_get_group", return_value=group)
            )
            stack.enter_context(
                patch.object(
                    collective,
                    "_get_symm_mem",
                    return_value=SimpleNamespace(
                        get_symm_mem_communicator=lambda: symm
                    ),
                )
            )
            native = stack.enter_context(
                patch.object(
                    torch.distributed, "all_gather_into_tensor", side_effect=gather
                )
            )
            legacy = stack.enter_context(
                patch.object(collective, "all_gather", return_value=expected)
            )
            yield native, legacy, symm

    def test_empty_receive_is_fully_overwritten_on_current_stream(self):
        local = torch.arange(256, device="cuda", dtype=torch.uint8).view(2, 128)
        expected = torch.cat([local ^ rank for rank in range(4)])
        original_empty = torch.empty

        def poison(*args, **kwargs):
            return original_empty(*args, **kwargs).fill_(0xA5)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with self.collective(expected) as (native, legacy, _):
            with torch.cuda.stream(side), patch.object(
                codec.torch, "empty", side_effect=poison
            ):
                actual = codec._try_all_gather_cp4_bytes(local)
                observed = actual.clone()
            torch.cuda.current_stream().wait_stream(side)
            self.assertTrue(torch.equal(observed, expected))
            native.assert_called_once()
            legacy.assert_not_called()

    def test_unsupported_policy_returns_before_allocation_or_collective(self):
        local = torch.ones((2, 18048), device="cuda", dtype=torch.uint8)
        with self.collective(local.repeat(4, 1)) as (native, _, symm):
            for target, name, value in (
                (torch.version, "hip", "test-rocm"),
                (torch.cuda, "is_current_stream_capturing", True),
                (torch.distributed, "is_initialized", False),
                (torch.distributed, "get_world_size", 2),
                (torch.distributed, "get_backend", "gloo"),
                (symm, "should_torch_symm_mem_allgather", True),
            ):
                kwargs = {"new": value} if name == "hip" else {"return_value": value}
                with patch.object(target, name, **kwargs), patch.object(
                    codec.torch, "empty"
                ) as empty:
                    self.assertIsNone(codec._try_all_gather_cp4_bytes(local))
                    empty.assert_not_called()
            for unsupported in (local.cpu(), local.float(), local[:, ::2], local[:0]):
                self.assertIsNone(codec._try_all_gather_cp4_bytes(unsupported))
            native.assert_not_called()

    def test_collective_failure_is_not_retried(self):
        local = torch.zeros((1, 18048), device="cuda", dtype=torch.uint8)
        with self.collective(local.repeat(4, 1)) as (native, legacy, _):
            native.side_effect = RuntimeError("collective failure")
            with self.assertRaisesRegex(RuntimeError, "collective failure"):
                codec._try_all_gather_cp4_bytes(local)
            native.assert_called_once()
            legacy.assert_not_called()

    def test_actual_readback_wrapper_and_capture_fallback(self):
        full, pool = _pool(5, 136)
        values = torch.randn((272, 512), device="cuda", dtype=torch.bfloat16)
        physical = torch.cat(
            (
                torch.arange(136, 272, device="cuda"),
                torch.arange(408, 544, device="cuda"),
            )
        )
        codec.quantize_and_insert_swa_k_cache(values, pool, physical)
        unique = torch.tensor([1, 3], device="cuda")
        gathered = (
            full[unique].view(2, 4, 18048).permute(1, 0, 2).contiguous().view(8, 18048)
        )
        compact = torch.arange(272, device="cuda").view(2, 136)
        meta = SimpleNamespace(
            unique_blocks=unique, compact_slots=compact, contiguous_block_start=-1
        )
        lens = torch.tensor([136, 107], device="cuda", dtype=torch.int32)
        expected = torch.full((2, 140, 528), -37, device="cuda", dtype=torch.bfloat16)
        codec.dequantize_and_gather_k_cache_slots(
            expected[:, :, :512], pool, physical.view(2, 136), lens, 1
        )
        for rank in range(4):
            actual = torch.full_like(expected, -37)
            with self.collective(gathered) as (native, legacy, _):
                codec.dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                    actual[:, :, :512],
                    full[:, rank * 18048 : (rank + 1) * 18048],
                    physical.view(2, 136),
                    lens,
                    1,
                    136,
                    rank,
                    4,
                    meta,
                )
                native.assert_called_once()
                legacy.assert_not_called()
            self.assertTrue(
                torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            )
        actual.fill_(-37)
        with self.collective(gathered) as (native, legacy, _):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                codec.dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                    actual[:, :, :512],
                    full[:, :18048],
                    physical.view(2, 136),
                    lens,
                    1,
                    136,
                    0,
                    4,
                    meta,
                )
            torch.cuda.current_stream().wait_stream(side)
            native.reset_mock()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                codec.dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                    actual[:, :, :512],
                    full[:, :18048],
                    physical.view(2, 136),
                    lens,
                    1,
                    136,
                    0,
                    4,
                    meta,
                )
            graph.replay()
            torch.cuda.synchronize()
            native.assert_not_called()
            legacy.assert_called_once()
            self.assertTrue(
                torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            )


if __name__ == "__main__":
    unittest.main()
