"""Logical ownership, bounded scratch, and byte-exact owned-row transport."""

import functools
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton

from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import cp_kv_slot_mapping
from rtp_llm.models_py.modules.dsv41.attn_type import HCA_KV, INDEXER_KV
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_fp4_triton as codec
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_joint_pool as joint
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_owned_pool as owned
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_pools as pools


class OwnedPoolHostTest(unittest.TestCase):
    def test_unsupported_plan_does_not_allocate_consumer_output(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_kv_workspace

        with patch.object(
            _v41_prefill_kv_workspace,
            "prepare_global_workspace",
            side_effect=AssertionError("fallback allocated an unused full KV output"),
        ):
            for cp, ends in ((8, (129,)), (4, (1_000_000,))):
                layout = (1, 256, 128, 128, 0, torch.int32, torch.int32, 128, 128, cp)
                self.assertIsNone(
                    owned.try_gather(
                        None,
                        None,
                        ends,
                        None,
                        None,
                        None,
                        layout,
                        None,
                        swa=object(),
                        output_stream=object(),
                    )
                )

    def test_owned_counts_and_inverse_cover_each_logical_row_once(self):
        rng = random.Random(481)
        for cp in (1, 2, 4):
            for unit in (64, 128, 256):
                for count in (
                    0,
                    1,
                    63,
                    64,
                    127,
                    128,
                    129,
                    255,
                    256,
                    257,
                    1025,
                    rng.randrange(4096),
                ):
                    actual = []
                    for rank in range(cp):
                        local_count = owned.owned_count(count, cp, unit, rank)
                        rows = [j for j in range(count) if j // unit % cp == rank]
                        self.assertEqual(local_count, len(rows))
                        for k, row in enumerate(rows):
                            self.assertEqual(k, row // (unit * cp) * unit + row % unit)
                        actual.extend(rows)
                    self.assertEqual(sorted(actual), list(range(count)))

    def test_global_scratch_bound_and_tail_only_padding(self):
        rng = random.Random(1903)
        for cp in (1, 2, 4):
            for batch in (1, 2, 31, 32, 64, 128):
                for counts in (
                    (32768,) * batch,
                    tuple(rng.randrange(32769) for _ in range(batch)),
                    (0,) * batch,
                ):
                    plan = owned.plan_groups(counts, cp_size=cp, owner_rows=128)
                    self.assertIsNotNone(plan)
                    self.assertLessEqual(plan.temporary_bytes, owned._LIMIT)
                    cursor = 0
                    for group in plan.groups:
                        self.assertEqual(group.first, cursor)
                        ns = counts[group.first : group.stop]
                        self.assertEqual(
                            group.rows, sum(triton.cdiv(n, 256) * 256 for n in ns)
                        )
                        actual_owned = [
                            sum(owned.owned_count(n, cp, 128, rank) for n in ns)
                            for rank in range(cp)
                        ]
                        self.assertEqual(group.local_rows, max(actual_owned))
                        cursor = group.stop
                    self.assertEqual(cursor, batch)
                    coefficient = 356 * (cp + 1 if cp > 1 else 1) + 16
                    self.assertEqual(
                        plan.temporary_bytes,
                        coefficient * plan.max_local_rows + 16 * plan.max_output_rows,
                    )
                    self.assertEqual(
                        plan.index_output_bytes,
                        68 * sum(group.rows for group in plan.groups),
                    )

    def test_scratch_includes_inverse_and_independent_group_maxima(self):
        plan = owned.plan_groups(
            (1,) * 10 + (2048,), cp_size=4, owner_rows=128, limit=1_200_000
        )
        self.assertEqual(len(plan.groups), 2)
        self.assertGreater(plan.groups[0].rows, plan.groups[1].rows)
        self.assertLess(plan.groups[0].local_rows, plan.groups[1].local_rows)
        self.assertEqual(
            plan.temporary_bytes,
            1796 * plan.groups[1].local_rows + 16 * plan.groups[0].rows,
        )
        self.assertLessEqual(plan.temporary_bytes, 1_200_000)
        for batch, ratio, groups in (
            (32, 1, (17, 15)),
            (32, 2, (32,)),
            (64, 1, (17, 17, 17, 13)),
            (64, 2, (35, 29)),
        ):
            plan = owned.plan_groups(
                (32768 // ratio,) * batch, cp_size=4, owner_rows=256 // ratio
            )
            self.assertEqual(tuple(g.stop - g.first for g in plan.groups), groups)
            self.assertLessEqual(plan.temporary_bytes, owned._LIMIT)

    def test_padded_tiles_remain_in_one_request_for_ragged_and_empty_rows(self):
        rng = random.Random(137)
        for counts in (
            (0, 1, 0, 127, 128, 129, 0, 257, 0),
            (0,) * 128,
            (0,) * 127 + (1,),
            tuple(rng.randrange(1025) for _ in range(128)),
        ):
            stops, cursor = [], 0
            for count in counts:
                cursor += triton.cdiv(count, 256) * 256
                stops.append(cursor)
            for base in range(0, cursor, 128):
                request = sum(stop <= base for stop in stops)
                self.assertLess(request, len(counts))
                self.assertGreater(counts[request], 0)
                self.assertGreaterEqual(base, stops[request - 1] if request else 0)
                self.assertLess(base + 127, stops[request])

    def test_common_unsupported_and_oversized_geometry(self):
        for counts, cp, unit in (
            ((), 4, 128),
            ((1,) * 129, 4, 128),
            ((-1,), 4, 128),
            ((1,), 3, 128),
            ((1,), 4, 0),
            ((10**9,), 4, 128),
        ):
            self.assertIsNone(owned.plan_groups(counts, cp_size=cp, owner_rows=unit))

    def test_cached_plan_uses_complete_geometry_and_copies_list_keys(self):
        owned._plan_groups_cached.cache_clear()
        self.addCleanup(owned._plan_groups_cached.cache_clear)
        counts = [127, 513]
        plan = owned.plan_groups(counts, cp_size=4, owner_rows=128)
        self.assertIs(plan, owned.plan_groups(tuple(counts), cp_size=4, owner_rows=128))
        counts[1] = 514
        cases = (
            (tuple(counts), 4, 128, owned._LIMIT),
            ((127, 513), 2, 128, owned._LIMIT),
            ((127, 513), 4, 64, owned._LIMIT),
            ((127, 513), 4, 128, 400000),
        )
        for values, cp, unit, limit in cases:
            actual = owned.plan_groups(values, cp_size=cp, owner_rows=unit, limit=limit)
            expected = owned._plan_groups_cached.__wrapped__(values, cp, unit, limit)
            self.assertEqual(actual, expected)
        self.assertEqual(owned._plan_groups_cached.cache_info().misses, 5)
        self.assertIs(plan, owned.plan_groups((127, 513), cp_size=4, owner_rows=128))

    def test_cached_plan_rejects_equal_but_invalid_count_types(self):
        owned._plan_groups_cached.cache_clear()
        self.addCleanup(owned._plan_groups_cached.cache_clear)
        owned.plan_groups((1,), cp_size=4, owner_rows=128)
        before = owned._plan_groups_cached.cache_info()
        for counts in ((True,), (1.0,), ([],), (-1,), (), (1,) * 129):
            self.assertIsNone(owned.plan_groups(counts, cp_size=4, owner_rows=128))
        self.assertEqual(owned._plan_groups_cached.cache_info(), before)
        # Preserve existing unusual scalar behavior without retaining it.
        for cp, limit in ((True, owned._LIMIT), (1, float(owned._LIMIT))):
            self.assertEqual(
                owned.plan_groups((1,), cp_size=cp, owner_rows=128, limit=limit),
                owned._plan_groups_cached.__wrapped__((1,), cp, 128, limit),
            )
        self.assertEqual(owned._plan_groups_cached.cache_info(), before)

    def test_cached_plan_is_bounded_and_contains_only_immutable_host_values(self):
        from dataclasses import FrozenInstanceError, fields

        owned._plan_groups_cached.cache_clear()
        self.addCleanup(owned._plan_groups_cached.cache_clear)
        for i in range(160):
            plan = owned.plan_groups((i, 129), cp_size=4, owner_rows=128)
        info = owned._plan_groups_cached.cache_info()
        self.assertEqual((info.maxsize, info.currsize), (128, 128))
        owned.plan_groups((0, 129), cp_size=4, owner_rows=128)
        self.assertEqual(owned._plan_groups_cached.cache_info().misses, info.misses + 1)

        def check(value):
            if value is None or type(value) is int:
                return
            if type(value) is tuple:
                for item in value:
                    check(item)
                return
            self.assertIn(type(value), (owned.OwnedPlan, owned.OwnedGroup))
            self.assertTrue(value.__dataclass_params__.frozen)
            for field in fields(value):
                check(getattr(value, field.name))

        check(plan)
        with self.assertRaises(FrozenInstanceError):
            plan.groups[0].rows = 0


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class OwnedPoolCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("Blackwell required")

    def fixture(
        self,
        counts,
        cp,
        ratio,
        dtypes=(torch.int32, torch.int64),
        token_blocks=(128, 128),
    ):
        ends_host = [n * ratio + ratio - 1 for n in counts]
        ends_backing = torch.zeros(
            2 * len(counts) + 1, dtype=torch.int64, device="cuda"
        )
        ends = ends_backing[1::2]
        ends.copy_(torch.tensor(ends_host, dtype=torch.int64, device="cuda"))
        mtpb, itpb = token_blocks
        owner = 256
        ranks = []
        for rank in range(cp):
            tables, values = {}, []
            for tag, entries, width, dtype in (
                (HCA_KV, mtpb // ratio, 288, dtypes[0]),
                (INDEXER_KV, itpb, 68, dtypes[1]),
            ):
                tpb = mtpb if tag == HCA_KV else itpb
                columns = max(
                    2,
                    triton.cdiv(max(ends_host, default=0), owner * cp) * (owner // tpb),
                )
                pages = 1 + len(counts) * columns
                offset, padding = (4, 12) if tag == INDEXER_KV else (1, 11)
                backing = torch.randint(
                    0,
                    256,
                    (pages * (entries * width + padding) + offset,),
                    dtype=torch.uint8,
                    device="cuda",
                )
                pool = backing[offset:].as_strided(
                    (pages, entries, width), (entries * width + padding, width, 1)
                )
                table = torch.zeros(
                    (len(counts), columns + 2), dtype=dtype, device="cuda"
                )[:, 1:-1]
                table.copy_(
                    torch.arange(1, pages, dtype=dtype, device="cuda").reshape(
                        len(counts), columns
                    )
                )
                table[::3, ::2] = 0
                table[1::3, 1::2] = -1
                tables[tag] = table
                values.append(pool)
            attn = SimpleNamespace(
                _cp_ctx=SimpleNamespace(
                    cp_size=cp, cp_rank=rank, kv_cache_sharded=cp > 1
                ),
                _kv_cache=SimpleNamespace(
                    get_seq_size_per_block=lambda tag: owner,
                    get_kernel_seq_size_per_block=lambda tag: (
                        mtpb if tag == HCA_KV else itpb
                    ),
                ),
                _block_tables_by_type=tables,
                _global_region=lambda: HCA_KV,
                _source_entries=lambda tag, pool: pool.shape[1],
                compress_ratio=ratio,
                _gather_shards=lambda value: self.fail(
                    "owned transport used all_reduce"
                ),
            )
            main, index = values
            self.assertTrue(
                joint.warmup(
                    joint.PoolLayout(mtpb, mtpb // ratio, owner, dtypes[0], ratio),
                    joint.PoolLayout(itpb, itpb, owner, dtypes[1], ratio),
                    cp_size=cp,
                    cp_rank=rank,
                    device=ends.device,
                )
            )
            ranks.append((attn, main, index))
        return ranks, ends_host, ends

    def slots(self, attn, tag, pool, positions, requests):
        cp = attn._cp_ctx
        return cp_kv_slot_mapping(
            positions,
            attn._block_tables_by_type[tag],
            requests,
            attn._kv_cache.get_kernel_seq_size_per_block(tag),
            pool.shape[1],
            attn.compress_ratio,
            cp.cp_size,
            cp.cp_rank,
            owner_tokens_per_block=256,
        )

    def expected_packet(self, ranks, counts, group):
        packets = []
        for attn, main, index in ranks:
            cp = attn._cp_ctx
            ratio = attn.compress_ratio
            u = 256 // ratio
            logical, requests = [], []
            for request in range(group.first, group.stop):
                selected = [
                    j
                    for j in range(counts[request])
                    if j // u % cp.cp_size == cp.cp_rank
                ]
                logical.extend((j + 1) * ratio - 1 for j in selected)
                requests.extend([request] * len(selected))
            pos = torch.tensor(logical, dtype=torch.int64, device="cuda")
            req = torch.tensor(requests, dtype=torch.int64, device="cuda")
            q, s = codec.gather_indexer_k_fp4(
                index, self.slots(attn, INDEXER_KV, index, pos, req)
            )
            raw = codec.gather_k_cache_bytes_fp4(
                main, self.slots(attn, HCA_KV, main, pos, req)
            )
            packet = torch.zeros(
                group.local_rows * 356, dtype=torch.uint8, device="cuda"
            )
            pq, ps, pr = joint._views(packet, group.local_rows)
            pq[: len(logical)].copy_(q)
            ps[: len(logical)].copy_(s)
            pr[: len(logical)].copy_(raw)
            packets.append(packet)
        return packets

    def expected_outputs(self, ranks, counts):
        expected = []
        for request, count in enumerate(counts):
            ratio = ranks[0][0].compress_ratio
            pos = (
                torch.arange(count, dtype=torch.int64, device="cuda") + 1
            ) * ratio - 1
            req = torch.full_like(pos, request)
            qs = torch.zeros((count, 68), dtype=torch.uint8, device="cuda")
            raw = torch.zeros((count, 288), dtype=torch.uint8, device="cuda")
            for attn, main, index in ranks:
                q, s = codec.gather_indexer_k_fp4(
                    index, self.slots(attn, INDEXER_KV, index, pos, req)
                )
                qs[:, :64] += q.view(torch.uint8)
                qs[:, 64:] += s.view(torch.uint8).reshape(count, 4)
                raw += codec.gather_k_cache_bytes_fp4(
                    main, self.slots(attn, HCA_KV, main, pos, req)
                )
            expected.append(
                (
                    codec.dequantize_k_cache_bytes_fp4(raw),
                    qs[:, :64].contiguous().view(torch.int8),
                    qs[:, 64:].contiguous().view(torch.int32).flatten(),
                )
            )
        return expected

    def assert_bits(self, left, right):
        self.assertEqual(left.dtype, right.dtype)
        self.assertEqual(left.shape, right.shape)
        self.assertTrue(
            torch.equal(
                left.contiguous().view(torch.uint8),
                right.contiguous().view(torch.uint8),
            )
        )

    def check(
        self,
        counts,
        cp,
        ratio,
        *,
        direct=False,
        limit=None,
        dtypes=(torch.int32, torch.int64),
        token_blocks=(128, 128),
    ):
        ranks, host, ends = self.fixture(counts, cp, ratio, dtypes, token_blocks)
        plan_fn = functools.partial(owned.plan_groups, limit=limit or owned._LIMIT)
        plan = plan_fn(counts, cp_size=cp, owner_rows=256 // ratio)
        self.assertIsNotNone(plan)
        expected_packets = [
            self.expected_packet(ranks, counts, group) for group in plan.groups
        ]
        expected = self.expected_outputs(ranks, counts)
        for rank, (attn, main, index) in enumerate(ranks):
            calls, pointers = [], []
            workspace = None
            if direct:
                starts, cursor = [], 3
                for i, count in enumerate(counts):
                    starts.append(cursor)
                    cursor += count + i % 3 + 1
                workspace = SimpleNamespace(
                    output=torch.full(
                        (cursor + 3, 512), 7.25, dtype=torch.bfloat16, device="cuda"
                    ),
                    global_sizes=tuple(counts),
                    global_starts=tuple(starts),
                    device_offsets=torch.tensor(
                        starts, dtype=torch.int64, device="cuda"
                    ),
                    publish=lambda result: result,
                )
                before = workspace.output.clone()

            def collect(receive, send, size, owner_rank):
                self.assertEqual(owner_rank, rank)
                group_index = len(calls)
                self.assert_bits(send, expected_packets[group_index][rank])
                receive.copy_(torch.cat(expected_packets[group_index]))
                calls.append(group_index)
                pointers.append(
                    (
                        send.untyped_storage().data_ptr(),
                        receive.untyped_storage().data_ptr(),
                    )
                )

            with patch.object(owned, "plan_groups", plan_fn), patch.object(
                owned, "_collect", collect
            ), patch.object(
                triton.runtime.jit.JITFunction,
                "_do_compile",
                side_effect=AssertionError("hot-path JIT"),
            ):
                actual = pools.try_gather_prefill_pools(
                    attn, main, index, host, ends, workspace=workspace
                )
            self.assertEqual(len(calls), sum(g.local_rows > 0 for g in plan.groups))
            self.assertLessEqual(len(set(pointers)), 1)
            for (g, keys), refs in zip(actual, expected):
                for value, reference in zip((g, keys.quant, keys.scale), refs):
                    self.assert_bits(value, reference)
                self.assertEqual(keys.quant.stride(), (64, 1))
                self.assertEqual(keys.scale.stride(), (1,))
            if direct:
                for start, reference in zip(workspace.global_starts, expected):
                    before[start : start + reference[0].shape[0]].copy_(reference[0])
                self.assert_bits(workspace.output, before)
        return ranks, host, ends

    def test_all_owners_ratios_and_ragged_layouts(self):
        for cp in (1, 2, 4):
            for ratio in (1, 2):
                with self.subTest(cp=cp, ratio=ratio):
                    self.check(
                        (0, 1, 63, 64, 127, 128, 129, 255, 256, 257, 1025),
                        cp,
                        ratio,
                        direct=True,
                    )

    def test_small_large_batch_and_forced_groups_reuse_scratch(self):
        for counts in ((257,), (0, 0), tuple(129 + i for i in range(32)), (1,) * 128):
            with self.subTest(batch=len(counts)):
                self.check(counts, 4, 2, limit=400000)

    def test_table_dtypes_and_non_direct_output(self):
        for dtype in (torch.int32, torch.int64):
            self.check((1, 257), 2, 1, dtypes=(dtype, dtype))

    def test_independent_main_index_kernel_pages(self):
        self.check(
            (0, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1027),
            4,
            2,
            direct=True,
            token_blocks=(64, 128),
        )

    def test_inverse_metadata_matches_owner_rows_padding_and_direct_destinations(self):
        counts = (0, 1, 0, 127, 128, 129, 0, 257, 1027, 0)
        for cp in (1, 2, 4):
            for ratio in (1, 2):
                ranks, _, ends = self.fixture(counts, cp, ratio)
                attn, main, index = ranks[0]
                mt, it = (
                    attn._block_tables_by_type[tag] for tag in (HCA_KV, INDEXER_KV)
                )
                layout = (
                    ratio,
                    256,
                    128,
                    128,
                    0,
                    mt.dtype,
                    it.dtype,
                    main.shape[1],
                    index.shape[1],
                    cp,
                )
                kernels = joint._READY[(ends.device.index, *layout)][5]
                plan = owned.plan_groups(
                    counts, cp_size=cp, owner_rows=256 // ratio, limit=1_200_000
                )
                self.assertIsNotNone(plan)
                slots = torch.empty(
                    2 * plan.max_local_rows + 1, dtype=torch.int64, device="cuda"
                )[1:]
                backing = torch.full(
                    (2 * plan.max_output_rows + 2,),
                    -777,
                    dtype=torch.int64,
                    device="cuda",
                )
                inverse = backing[1:-1]
                starts, cursor = [], 11
                for count in counts:
                    starts.append(cursor)
                    cursor += count + 7
                offsets = torch.tensor([0, *starts], dtype=torch.int64, device="cuda")[
                    1:
                ]
                for direct in (False, True):
                    for group in plan.groups:
                        inverse.fill_(-777)
                        codes, destinations, before = [], [], [0] * cp
                        canonical = 0
                        unit = 256 // ratio
                        for request in range(group.first, group.stop):
                            count = counts[request]
                            for row in range(triton.cdiv(count, 256) * 256):
                                rank = row // unit % cp
                                source = (
                                    before[rank]
                                    + row // (cp * unit) * unit
                                    + row % unit
                                )
                                codes.append(
                                    (source << 2) | rank if row < count else -1
                                )
                                destinations.append(
                                    starts[request] + row if direct else canonical + row
                                )
                            canonical += triton.cdiv(count, 256) * 256
                            before = [
                                value + owned.owned_count(count, cp, unit, rank)
                                for rank, value in enumerate(before)
                            ]
                        with patch.object(
                            triton.runtime.jit.JITFunction,
                            "_do_compile",
                            side_effect=AssertionError("hot-path JIT"),
                        ):
                            kernels[4 if direct else 0][
                                (
                                    triton.cdiv(max(group.rows, group.local_rows), 128),
                                    1,
                                    1,
                                )
                            ](
                                *owned._metadata_args(
                                    ends,
                                    mt,
                                    it,
                                    slots,
                                    inverse,
                                    offsets,
                                    group.local_rows,
                                    group.rows,
                                    group.first,
                                    group.stop,
                                    layout,
                                    direct,
                                )
                            )
                        actual = inverse.cpu().tolist()
                        self.assertEqual(actual[: group.rows], codes)
                        self.assertEqual(
                            actual[group.rows : 2 * group.rows], destinations
                        )
                        self.assertEqual(
                            actual[2 * group.rows :],
                            [-777] * (len(actual) - 2 * group.rows),
                        )
                        self.assertEqual(backing[[0, -1]].cpu().tolist(), [-777, -777])

    def test_cp1_graph_replays_changed_pool_values(self):
        counts = (1, 129, 257)
        ranks, host, ends = self.fixture(counts, 1, 2)
        attn, main, index = ranks[0]
        # Capture retains fixed host geometry but re-reads current pool bytes.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = pools.try_gather_prefill_pools(attn, main, index, host, ends)
        main.random_(0, 256)
        index.random_(0, 256)
        graph.replay()
        expected = self.expected_outputs(ranks, counts)
        for (g, keys), refs in zip(actual, expected):
            for value, reference in zip((g, keys.quant, keys.scale), refs):
                self.assert_bits(value, reference)

    def test_cold_or_invalid_local_state_raises_before_collective(self):
        ranks, host, ends = self.fixture((3, 5), 4, 1)
        attn, main, index = ranks[0]
        with patch.object(owned, "_collect", side_effect=AssertionError("collective")):
            with patch.dict(joint._READY, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "startup warmup"):
                    pools.try_gather_prefill_pools(attn, main, index, host, ends)
            malformed = index.as_strided(
                index.shape, index.stride(), index.storage_offset() + 1
            )
            with patch.object(torch, "empty", side_effect=AssertionError("allocation")):
                with self.assertRaisesRegex(ValueError, "4-byte aligned"):
                    pools.try_gather_prefill_pools(attn, main, malformed, host, ends)

    def test_collective_failure_propagates_without_fallback(self):
        ranks, host, ends = self.fixture((3, 5), 4, 1)
        with patch.object(
            owned, "_collect", side_effect=RuntimeError("transport witness")
        ):
            with self.assertRaisesRegex(RuntimeError, "transport witness"):
                pools.try_gather_prefill_pools(*ranks[0], host, ends)


if __name__ == "__main__":
    unittest.main()
