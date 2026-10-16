"""Host page plans preserve variable-length target/draft slot dependencies."""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_swa_metadata as metadata
from rtp_llm.models_py.modules.dsv41.fp8._v41_attention_jit_warmup import (
    PoolLayout,
    _warm_host_swa_metadata,
)


class HostCompactTableTest(unittest.TestCase):
    def test_ragged_aliases_padding_strides_and_negative_view(self):
        prefixes = (0, 17, 511, 1023, 3, 2047, 8191)
        lengths = (0, 1, 129, 513, 7, 257, 17)
        for dtype, columns, read, negative in itertools.product(
            (torch.int32, torch.int64), (13, 18, 24), (False, True), (False, True)
        ):
            host = (torch.arange(7 * 36).reshape(7, 36) % 23 + 1).to(dtype)[:, ::2]
            host[:, ::7] = 0
            host[:, 1::11] = -3
            if negative:
                host = torch._neg_view(-host)
            active = []
            for req, (prefix, length) in enumerate(zip(prefixes, lengths)):
                if not read and not length:
                    continue
                start, end = (
                    (max(0, prefix - 127), prefix)
                    if read
                    else (prefix, prefix + length)
                )
                for col in range(
                    start // 512, min((end + 511) // 512, columns, host.shape[1])
                ):
                    if int(host[req, col]) > 0:
                        active.append((req, col, int(host[req, col])))
            unique = sorted({block for _, _, block in active})
            expected = torch.full((len(lengths), columns), -1, dtype=torch.int32)
            for req, col, block in active:
                expected[req, col] = unique.index(block)
            actual_unique, actual = metadata._host_compact_table(
                host, prefixes, lengths, columns, 512, read, pin_memory=False
            )
            self.assertEqual(actual_unique, unique)
            self.assertTrue(torch.equal(actual, expected))

    def test_host_plan_has_no_per_request_tensor_indexing(self):
        host = torch.ones(64, 70, dtype=torch.int32)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            metadata._host_compact_table(
                host, (31744,) * 64, (1024,) * 64, 70, 512, False, pin_memory=False
            )
        names = [event.name for event in prof.events()]
        for name in ("aten::select", "aten::slice", "aten::index_put_", "aten::item"):
            self.assertNotIn(name, names)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class HostSwaMetadataTest(unittest.TestCase):
    @staticmethod
    def fixture(prefix_dtype=torch.int64, length_dtype=torch.int32, offset=0):
        prefixes = (0, 17, 511, 1023, 3, 2047, 8191)
        lengths = (0, 1, 129, 513, 7, 257, 17)
        host = (torch.arange(7 * 18).reshape(7, 18) % 23 + 1).int()
        host[:, ::7] = 0
        host[:, 1::11] = -3

        def device(values, dtype):
            return torch.tensor(
                [0] * offset + list(values), dtype=dtype, device="cuda"
            )[offset:]

        table = device(host.flatten().tolist(), torch.int32).view_as(host)
        cp = SimpleNamespace(
            cp_size=4,
            kv_cache_sharded=True,
            prefix_lengths_host=prefixes,
            input_lengths_global_host=lengths,
            prefix_lengths=device(prefixes, prefix_dtype),
            input_lengths_global=device(lengths, length_dtype),
            cu_seqlens_global=device(
                (0,) + tuple(itertools.accumulate(lengths)), torch.int32
            ),
        )
        # All-gathered CP rank order differs from packed request order. Padding
        # is interleaved, and the zero-length request contributes no rows.
        req, pos = [], []
        for rank in range(4):
            for request, (prefix, length) in enumerate(zip(prefixes, lengths)):
                pair = (length + 7) // 8
                for start in (rank * pair, (7 - rank) * pair):
                    for relative in range(start, start + pair):
                        valid = relative < length
                        req.append(request if valid else -1)
                        pos.append(prefix + relative if valid else -1)
        return host, table, cp, req, pos, device

    @staticmethod
    def reference(host, cp, requests, positions, entries=136, span=512):
        slots = []
        for request, position in zip(requests, positions):
            slot = -1
            if 0 <= request < len(cp.prefix_lengths_host) and position >= 0:
                end = (
                    cp.prefix_lengths_host[request]
                    + cp.input_lengths_global_host[request]
                )
                column = position // span
                if position < end and column < host.shape[1]:
                    block = int(host[request, column])
                    if block > 0 and position + entries >= min(
                        (column + 1) * span, end
                    ):
                        slot = block * entries + position % entries
            slots.append(slot)
        unique = sorted({slot // entries for slot in slots if slot >= 0})
        lookup = {block: row for row, block in enumerate(unique)}
        compact = [
            lookup[slot // entries] * entries + slot % entries if slot >= 0 else -1
            for slot in slots
        ]
        return slots, unique, compact

    def test_cp_padding_aliases_and_runtime_shapes_do_not_compile_after_warmup(self):
        _warm_host_swa_metadata(
            PoolLayout("swa_kv", 136, 136 * 528, 512, 512, 0), "cuda"
        )
        with patch.object(
            metadata._planned_position_slots,
            "_do_compile",
            side_effect=AssertionError("draft slot metadata JIT in forward"),
        ), patch.object(
            metadata._planned_slots,
            "_do_compile",
            side_effect=AssertionError("target slot metadata JIT in forward"),
        ):
            for pd, ld, rd, xd, offset in itertools.product(
                (torch.int32, torch.int64),
                (torch.int32, torch.int64),
                (torch.int32, torch.int64),
                (torch.int32, torch.int64),
                (0, 1),
            ):
                host, table, cp, requests, positions, device = self.fixture(
                    pd, ld, offset
                )
                expected = self.reference(host, cp, requests, positions)
                actual = metadata.try_host_slot_metadata(
                    table,
                    host,
                    cp,
                    entries=136,
                    span=512,
                    num_blocks=25,
                    request_ids=device(requests, rd),
                    positions=device(positions, xd),
                )
                self.assertIsNotNone(actual)
                self.assertEqual(actual[0].tolist(), expected[0])
                self.assertEqual(actual[1].unique_blocks.tolist(), expected[1])
                self.assertEqual(actual[1].compact_slots.tolist(), expected[2])
                for read in (False, True):
                    planned = metadata.try_host_slot_metadata(
                        table, host, cp, entries=136, span=512, num_blocks=25, read=read
                    )
                    self.assertIsNotNone(planned)

    def test_native_table_padding_widths_preserve_physical_page_ids(self):
        host, table, cp, requests, positions, device = self.fixture()
        expected = self.reference(host, cp, requests, positions)
        table = torch.nn.functional.pad(table, (0, 4), value=-1)
        for delta in (0, 4, 7):
            cpu = torch.nn.functional.pad(host, (0, delta), value=-1)
            actual = metadata.try_host_slot_metadata(
                table,
                cpu,
                cp,
                entries=136,
                span=512,
                num_blocks=25,
                request_ids=device(requests, torch.int32),
                positions=device(positions, torch.int32),
            )
            self.assertEqual(actual[0].tolist(), expected[0])
            self.assertEqual(actual[1].compact_slots.tolist(), expected[2])

    def test_no_device_host_sync_or_dynamic_size_operations(self):
        host, table, cp, requests, positions, device = self.fixture()
        req = device(requests, torch.int32)
        pos = device(positions, torch.int32)
        for _ in range(2):
            metadata.try_host_slot_metadata(
                table,
                host,
                cp,
                entries=136,
                span=512,
                num_blocks=25,
                request_ids=req,
                positions=pos,
            )
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            metadata.try_host_slot_metadata(
                table,
                host,
                cp,
                entries=136,
                span=512,
                num_blocks=25,
                request_ids=req,
                positions=pos,
            )
        names = [event.name for event in prof.events()]
        for forbidden in (
            "aten::nonzero",
            "aten::_unique2",
            "aten::item",
            "cudaStreamSynchronize",
        ):
            self.assertNotIn(forbidden, names)

    def test_empty_pages_empty_rows_and_capture_fallback(self):
        host, table, cp, requests, positions, device = self.fixture()
        host.fill_(-1)
        table.fill_(-1)
        for empty in (False, True):
            req = device([] if empty else requests, torch.int32)
            pos = device([] if empty else positions, torch.int32)
            actual = metadata.try_host_slot_metadata(
                table,
                host,
                cp,
                entries=136,
                span=512,
                num_blocks=25,
                request_ids=req,
                positions=pos,
            )
            self.assertEqual(actual[0].numel(), req.numel())
            self.assertEqual(actual[1].unique_blocks.numel(), 0)
            self.assertTrue(torch.all(actual[0] == -1))
            self.assertTrue(torch.all(actual[1].compact_slots == -1))
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=True
        ), patch.object(
            metadata, "_upload", side_effect=AssertionError("host copy inside graph")
        ):
            self.assertIsNone(
                metadata.try_host_slot_metadata(
                    table,
                    host,
                    cp,
                    entries=136,
                    span=512,
                    num_blocks=25,
                    request_ids=req,
                    positions=pos,
                )
            )


if __name__ == "__main__":
    unittest.main()
