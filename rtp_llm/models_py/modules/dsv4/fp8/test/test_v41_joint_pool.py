"""Independent Torch byte oracle and warm/collective-schedule contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.attn_type import HCA_KV, INDEXER_KV
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_fp4_triton as codec
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_joint_pool as joint
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_pools as pools
from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import cp_kv_slot_mapping


class JointPoolHostTest(unittest.TestCase):
    def test_common_small_batch_has_no_device_work(self):
        with patch.object(torch, "empty", side_effect=AssertionError("allocation")):
            for batch in (0, 1, 6, 31, 129):
                self.assertIsNone(
                    joint.try_gather(None, None, None, [1] * batch, None, groups=[])
                )

    def test_unsupported_topology_and_ratio_are_common_fallbacks(self):
        for size, ratio, sharded in ((1, 1, False), (2, 1, True), (4, 4, True)):
            attn = SimpleNamespace(
                _cp_ctx=SimpleNamespace(cp_size=size, kv_cache_sharded=sharded),
                compress_ratio=ratio,
                _kv_cache=object(),
            )
            self.assertIsNone(
                joint.try_gather(attn, None, None, [1] * 32, None, groups=[])
            )

    def test_cold_qualified_rank_cannot_choose_old_collective_schedule(self):
        seq = Mock(spec=torch.Tensor)
        seq.device, seq.dtype, seq.layout, seq.shape = (
            torch.device("cuda:0"),
            torch.int64,
            torch.strided,
            (32,),
        )
        seq.stride.return_value = 1
        table = torch.zeros((32, 1), dtype=torch.int32)
        attn = SimpleNamespace(
            _cp_ctx=SimpleNamespace(cp_size=4, cp_rank=0, kv_cache_sharded=True),
            _kv_cache=SimpleNamespace(
                seq_size_per_block=128, kernel_seq_size_per_block=128
            ),
            _block_tables_by_type={HCA_KV: table, INDEXER_KV: table},
            _global_region=lambda: HCA_KV,
            _source_entries=lambda region, pool: 128,
            compress_ratio=1,
            _gather_shards=Mock(),
        )
        with patch.dict(joint._READY, {}, clear=True), patch.object(
            torch, "empty", side_effect=AssertionError("allocation")
        ):
            with self.assertRaisesRegex(RuntimeError, "cold"):
                joint.try_gather(
                    attn, None, None, [1] * 32, seq, groups=[(0, 32, 8192)]
                )
        attn._gather_shards.assert_not_called()

    def test_warmup_rejects_common_unsupported_layout_without_cuda(self):
        layout = joint.PoolLayout(128, 128, 128, torch.int32, 1)
        with patch.object(
            torch.cuda, "current_device", side_effect=AssertionError("CUDA")
        ):
            self.assertFalse(
                joint.warmup(layout, layout, cp_size=1, cp_rank=0, device="cuda:0")
            )
            self.assertFalse(
                joint.warmup(layout, layout, cp_size=4, cp_rank=0, device="cpu")
            )
            self.assertFalse(
                joint.warmup(
                    layout,
                    joint.PoolLayout(128, 64, 128, torch.int32, 2),
                    cp_size=4,
                    cp_rank=0,
                    device="cuda:0",
                )
            )

    def test_padding_is_common_layout_fallback_not_ratio_inference(self):
        plain = joint.PoolLayout(128, 128, 128, torch.int32, 1)
        for entries in (64, 129, 256):
            padded = joint.PoolLayout(128, entries, 128, torch.int32, 1)
            self.assertFalse(joint.is_supported_layout(padded, plain, cp_size=4))
            self.assertFalse(joint.is_supported_layout(plain, padded, cp_size=4))

    def test_native_schema_has_independent_index_capacity(self):
        for ratio in (1, 2):
            main = joint.PoolLayout(128, 128 // ratio, 128, torch.int32, ratio)
            index = joint.PoolLayout(128, 128, 128, torch.int64, ratio)
            self.assertTrue(joint.is_supported_layout(main, index, cp_size=4))
        compact_index = joint.PoolLayout(128, 64, 128, torch.int64, 2)
        self.assertFalse(joint.is_supported_layout(main, compact_index, cp_size=4))
        with patch.object(
            torch.cuda, "current_device", side_effect=AssertionError("CUDA")
        ):
            self.assertFalse(
                joint.warmup(main, compact_index, cp_size=4, cp_rank=0, device="cuda:0")
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class JointPoolCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("SM100 required")

    def assert_bits(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def fixture(
        self, counts, ratio, rank, dtype=torch.int32, index_dtype=None, stride_padding=0
    ):
        host = [n * ratio + ratio - 1 for n in counts]
        backing = torch.zeros(len(host) * 2 + 1, dtype=torch.int64, device="cuda")
        backing[1::2] = torch.tensor(host, dtype=torch.int64, device="cuda")
        ends = backing[1::2]
        tpb = 128
        entries = {HCA_KV: tpb // ratio, INDEXER_KV: tpb}
        columns = max(1, (max(host, default=0) + 511) // 512)
        pages = 1 + len(host) * columns
        tables, data = {}, []
        for region, width in (
            (HCA_KV, codec.FP4_GLOBAL_ENTRY_BYTES),
            (INDEXER_KV, codec.FP4_INDEXER_ENTRY_BYTES),
        ):
            eb = entries[region]
            storage = torch.randint(
                0,
                256,
                (pages, eb * width + stride_padding),
                dtype=torch.uint8,
                device="cuda",
            )
            pool = storage.as_strided((pages, eb, width), (storage.stride(0), width, 1))
            if ratio == 2 and region == INDEXER_KV:
                # The unused native half must never leak into gathered bytes.
                storage[:, 64 * 64 : 128 * 64] = 0xA5
                storage[:, 128 * 64 + 64 * 4 : 128 * 68] = 0x5A
            table_dtype = (index_dtype or dtype) if region == INDEXER_KV else dtype
            table = torch.zeros(
                (len(host), columns + 3), dtype=table_dtype, device="cuda"
            )[:, 1 : columns + 1]
            table.copy_(
                torch.arange(1, pages, dtype=table_dtype, device="cuda").view(
                    len(host), columns
                )
            )
            table[::3, ::2] = 0
            table[1::3, 1::2] = -1
            tables[region] = table
            data.append(pool)
        attn = SimpleNamespace(
            _cp_ctx=SimpleNamespace(cp_size=4, cp_rank=rank, kv_cache_sharded=True),
            _kv_cache=SimpleNamespace(
                seq_size_per_block=128, kernel_seq_size_per_block=tpb
            ),
            _block_tables_by_type=tables,
            _global_region=lambda: HCA_KV,
            _source_entries=lambda region, pool: pool.shape[1],
            compress_ratio=ratio,
            _gather_shards=lambda value: None,
        )

        def slots(region, pos, requests):
            selected = cp_kv_slot_mapping(
                pos.clamp_min(0),
                tables[region],
                requests,
                tpb,
                entries[region],
                ratio,
                4,
                rank,
                owner_tokens_per_block=128,
            )
            return torch.where(pos >= 0, selected, -1)

        attn._slots = slots
        return (attn, *data, host, ends)

    def warm(self, ratio, rank, dtype, index_dtype=None):
        layout = joint.PoolLayout(128, 128 // ratio, 128, dtype, ratio)
        index_layout = joint.PoolLayout(128, 128, 128, index_dtype or dtype, ratio)
        self.assertTrue(
            joint.warmup(
                layout,
                index_layout,
                cp_size=4,
                cp_rank=rank,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
        )

    def oracle(self, attn, main, index, host):
        ratio = attn.compress_ratio
        result = []
        packets = []
        for first, stop, rows in pools._request_groups(tuple(x // ratio for x in host)):
            counts = torch.tensor(
                [x // ratio for x in host[first:stop]], dtype=torch.int64, device="cuda"
            )
            sizes = (counts + 255) // 256 * 256
            prefix = sizes.cumsum(0)
            row = torch.arange(rows, dtype=torch.int64, device="cuda")
            req = torch.bucketize(row, prefix[:-1], right=True)
            local = row - (prefix - sizes)[req]
            pos = torch.where(local < counts[req], (local + 1) * ratio - 1, -1)
            gathered = []
            for region, pool, payload in ((HCA_KV, main, 256), (INDEXER_KV, index, 64)):
                slots = attn._slots(region, pos, req + first)
                eb, width = pool.shape[1:]
                pages = pool.view(pool.shape[0], eb * width)
                packed = torch.cat(
                    (
                        pages[:, : eb * payload].reshape(-1, payload),
                        pages[:, eb * payload :].reshape(-1, width - payload),
                    ),
                    dim=1,
                )
                raw = packed[slots.clamp_min(0)].clone()
                raw[slots < 0] = 0
                gathered.append(raw)
            raw, idx = gathered
            quant, scale = (
                idx[:, :64].contiguous().view(torch.int8),
                idx[:, 64:].contiguous().view(torch.int32).flatten(),
            )
            packets.append(
                torch.cat(
                    (
                        quant.view(torch.uint8).flatten(),
                        scale.view(torch.uint8),
                        raw.flatten(),
                    )
                )
            )
            g = codec.dequantize_k_cache_bytes_fp4(raw)
            padded = [(x // ratio + 255) // 256 * 256 for x in host[first:stop]]
            result.extend(
                (gg[: n // ratio], qq[: n // ratio], ss[: n // ratio])
                for n, gg, qq, ss in zip(
                    host[first:stop],
                    g.split(padded),
                    quant.split(padded),
                    scale.split(padded),
                )
            )
        return result, packets

    def test_exact_torch_bytes_all_owners_and_table_types(self):
        for ratio in (1, 2):
            for dtype, index_dtype in (
                (torch.int32, torch.int32),
                (torch.int32, torch.int64),
                (torch.int64, torch.int32),
                (torch.int64, torch.int64),
            ):
                for rank in range(4):
                    with self.subTest(
                        ratio=ratio, dtype=dtype, index_dtype=index_dtype, rank=rank
                    ):
                        fixture = self.fixture(
                            tuple((i * 73) % 1100 for i in range(32)),
                            ratio,
                            rank,
                            dtype,
                            index_dtype,
                        )
                        attn, main, index, host, ends = fixture
                        self.warm(ratio, rank, dtype, index_dtype)
                        expected, packets = self.oracle(attn, main, index, host)
                        observed = []
                        attn._gather_shards = lambda packet: observed.append(
                            packet.clone()
                        )
                        # Forward must use compiled handles, not a JIT lookup/compile.
                        with patch.object(
                            joint._metadata,
                            "run",
                            side_effect=AssertionError("forward JIT"),
                        ), patch.object(
                            joint._gather,
                            "run",
                            side_effect=AssertionError("forward JIT"),
                        ), patch.object(
                            joint._dequant,
                            "run",
                            side_effect=AssertionError("forward JIT"),
                        ):
                            actual = pools.try_gather_prefill_pools(
                                attn, main, index, host, ends
                            )
                        self.assertEqual(len(observed), len(packets))
                        for a, b in zip(observed, packets):
                            self.assert_bits(a, b)
                        for (g, keys), refs in zip(actual, expected):
                            for a, b in zip((g, keys.quant, keys.scale), refs):
                                self.assert_bits(a, b)
                                self.assertTrue(a.is_contiguous())

    def test_cold_invalid_alignment_and_transport_error_never_fallback(self):
        attn, main, index, host, ends = self.fixture([7] * 32, 1, 0)
        attn._gather_shards = Mock()
        with patch.dict(joint._READY, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "cold"):
                pools.try_gather_prefill_pools(attn, main, index, host, ends)
        attn._gather_shards.assert_not_called()
        self.warm(1, 0, torch.int32)
        unaligned = torch.empty(main.numel() + 1, dtype=torch.uint8, device="cuda")[
            1:
        ].view(main.shape)
        with self.assertRaisesRegex(RuntimeError, "local storage"):
            pools.try_gather_prefill_pools(attn, unaligned, index, host, ends)
        attn._gather_shards.assert_not_called()
        attn._gather_shards.side_effect = RuntimeError("transport witness")
        with self.assertRaisesRegex(RuntimeError, "transport witness"):
            pools.try_gather_prefill_pools(attn, main, index, host, ends)
        self.assertEqual(attn._gather_shards.call_count, 1)

    def test_native_capacity_with_padded_physical_stride(self):
        attn, main, index, host, ends = self.fixture(
            [129] * 32, 2, 0, stride_padding=16
        )
        self.warm(2, 0, torch.int32)
        expected, packets = self.oracle(attn, main, index, host)
        observed = []
        attn._gather_shards = lambda value: observed.append(value.clone())
        actual = pools.try_gather_prefill_pools(attn, main, index, host, ends)
        self.assertEqual(len(observed), len(packets))
        for left, right in zip(observed, packets):
            self.assert_bits(left, right)
        for (g, keys), refs in zip(actual, expected):
            for left, right in zip((g, keys.quant, keys.scale), refs):
                self.assert_bits(left, right)

    def test_actual_entries_key_rejects_stale_compact_specialization(self):
        attn, main, index, host, ends = self.fixture([65] * 32, 2, 0)
        self.warm(2, 0, torch.int32)
        key = (
            torch.cuda.current_device(),
            2,
            128,
            128,
            128,
            0,
            torch.int32,
            torch.int32,
            64,
            128,
        )
        kernels = joint._READY[key]
        attn._gather_shards = Mock()
        for stale_key in (key[:-2], (*key[:-1], 64)):
            with patch.dict(
                joint._READY, {stale_key: kernels}, clear=True
            ), patch.object(
                torch,
                "empty",
                side_effect=AssertionError("allocation before readiness"),
            ):
                with self.assertRaisesRegex(RuntimeError, "cold"):
                    pools.try_gather_prefill_pools(attn, main, index, host, ends)
        attn._gather_shards.assert_not_called()

    def test_native_slots_match_torch_at_owner_and_page_boundaries(self):
        counts = (0, 1, 63, 64, 65, 127, 128, 129) * 4
        for rank in range(4):
            attn, main, index, host, ends = self.fixture(counts, 2, rank)
            self.warm(2, rank, torch.int32)
            key = (
                torch.cuda.current_device(),
                2,
                128,
                128,
                128,
                rank,
                torch.int32,
                torch.int32,
                64,
                128,
            )
            rows = sum((n + 255) // 256 * 256 for n in counts)
            slots = torch.empty((2, rows), dtype=torch.int64, device="cuda")
            mt, it = (attn._block_tables_by_type[x] for x in (HCA_KV, INDEXER_KV))
            joint._READY[key][0][((rows + 127) // 128, 1, 1)](
                *joint._metadata_args(ends, mt, it, slots, rows, 0, 32, key[1:])
            )
            sizes = [(n + 255) // 256 * 256 for n in counts]
            positions = torch.cat(
                [
                    torch.where(
                        torch.arange(size, device="cuda") < n,
                        (torch.arange(size, device="cuda") + 1) * 2 - 1,
                        -1,
                    )
                    for n, size in zip(counts, sizes)
                ]
            ).long()
            requests = torch.repeat_interleave(
                torch.arange(32, device="cuda"), torch.tensor(sizes, device="cuda")
            )
            for actual, region in zip(slots, (HCA_KV, INDEXER_KV)):
                self.assertTrue(
                    torch.equal(actual, attn._slots(region, positions, requests))
                )

    def test_compact_index_capacity_rejected_before_allocation_or_collective(self):
        for rank in range(4):
            attn, main, index, host, ends = self.fixture([65] * 32, 2, rank)
            compact = torch.empty(
                (index.shape[0], 64, 68), dtype=torch.uint8, device="cuda"
            )
            attn._gather_shards = Mock()
            with patch.object(
                torch, "empty", side_effect=AssertionError("joint allocation")
            ):
                self.assertIsNone(
                    joint.try_gather(
                        attn,
                        main,
                        compact,
                        host,
                        ends,
                        groups=pools._request_groups(tuple(n // 2 for n in host)),
                    )
                )
            attn._gather_shards.assert_not_called()

    def test_small_batch_retains_original_three_transports(self):
        attn, main, index, host, ends = self.fixture((0, 1, 255, 256, 257, 1025), 1, 0)
        self.warm(1, 0, torch.int32)
        expected, _ = self.oracle(attn, main, index, host)
        attn._gather_shards = Mock()
        actual = pools.try_gather_prefill_pools(attn, main, index, host, ends)
        self.assertEqual(attn._gather_shards.call_count, 3)
        for (g, keys), refs in zip(actual, expected):
            for a, b in zip((g, keys.quant, keys.scale), refs):
                self.assert_bits(a, b)

    def test_padded_region_layout_falls_back_for_every_rank(self):
        for rank in range(4):
            attn, main, index, host, ends = self.fixture([7] * 32, 1, rank)
            padded = torch.empty(
                (main.shape[0], 129, 288), dtype=torch.uint8, device="cuda"
            )
            attn._gather_shards = Mock()
            with patch.object(
                torch, "empty", side_effect=AssertionError("joint allocation")
            ):
                self.assertIsNone(
                    joint.try_gather(
                        attn,
                        padded,
                        index,
                        host,
                        ends,
                        groups=pools._request_groups(tuple(host)),
                    )
                )
            attn._gather_shards.assert_not_called()

    def test_graph_side_stream_and_no_packet_alias(self):
        attn, main, index, host, ends = self.fixture([257] * 32, 2, 0)
        self.warm(2, 0, torch.int32)
        packet_pointers = []
        attn._gather_shards = lambda x: packet_pointers.append(
            x.untyped_storage().data_ptr()
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            pools.try_gather_prefill_pools(attn, main, index, host, ends)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = pools.try_gather_prefill_pools(attn, main, index, host, ends)
        for g, keys in actual:
            for value in (g, keys.quant, keys.scale):
                self.assertNotEqual(
                    value.untyped_storage().data_ptr(), packet_pointers[-1]
                )
        main.random_(0, 256)
        index.random_(0, 256)
        graph.replay()
        expected, _ = self.oracle(attn, main, index, host)
        for (g, keys), refs in zip(actual, expected):
            for a, b in zip((g, keys.quant, keys.scale), refs):
                self.assert_bits(a, b)


if __name__ == "__main__":
    unittest.main()
