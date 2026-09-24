"""CPU planning/proof tests and explicitly opt-in SM100 byte comparisons.

CPU: python3 test_v41_batched_producer.py
GPU (only on an assigned device): DSV41_RUN_PRODUCER_GPU_TESTS=1 with the
same invocation in the model's CUDA13 environment. GPU tests compare the
unchanged segment kernels, not an algebraically equivalent float reference.
"""

from __future__ import annotations

import ast
import math
import os
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def _cpu_module():
    # CPU tests must not import the model package, Triton, or initialize CUDA.
    path = ROOT / "_v41_batched_producer.py"
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        or (isinstance(node, ast.FunctionDef) and not node.name.startswith("_"))
    ]
    module = ModuleType("v41_batched_producer_cpu")
    sys.modules[module.__name__] = module
    module.__dict__.update(
        torch=torch,
        math=math,
        os=os,
        dataclass=dataclass,
        _MAX_GROUP_ROWS=65536,
        _MAX_PACK_PROJECTIONS=512,
        legacy=SimpleNamespace(_enabled=lambda tensor: False),
        CSA_STATE=5,
        INDEXER_KV=3,
        require_pool_tokens_per_block=lambda cache, region: (
            cache.group_seq_size_per_block[cache.group_region_names.index(region)]
            if region == 5
            else cache.kernel_seq_size_per_block
        ),
    )
    validator_tree = ast.parse((ROOT / "_v41_prefill_global.py").read_text())
    validator_names = {"_matrix", "_vector", "_pool", "_frequencies"}
    validators = [
        n
        for n in validator_tree.body
        if isinstance(n, ast.FunctionDef) and n.name in validator_names
    ]
    scope = dict(torch=torch)
    exec(
        compile(ast.Module(body=validators, type_ignores=[]), "validators", "exec"),
        scope,
    )
    for name in validator_names:
        setattr(module.legacy, name, scope[name])
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
        module.__dict__,
    )
    return module


CPU = _cpu_module()


def _metadata(prefixes, request_segments, ratio):
    segments, cursor, compact = {}, 0, 0
    for prefix, sizes in zip(prefixes, request_segments):
        position = prefix
        for size in sizes:
            indices = [i for i in range(size) if (position + i + 1) % ratio == 0]
            phase = (ratio - 1 - position) % ratio
            segments[cursor] = (cursor + size, compact, compact + len(indices), phase)
            cursor += size
            compact += len(indices)
            position += size
    return SimpleNamespace(segments=segments)


def _state_mapping():
    path = ROOT / "_cp_slot_mapping.py"
    tree = ast.parse(path.read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "cp_state_slot_mapping"
    )
    namespace = dict(torch=torch)
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[node.name]


class BatchedProducerCPU(unittest.TestCase):
    def test_transport_cpu_and_host_validation_fallback(self):
        send = torch.zeros((8, 512))
        self.assertFalse(CPU.pack_projected_group(send, [], [], []))
        self.assertIsNone(
            CPU.restore_projected_group(send, torch.arange(8), 0, 2, 8, 0, 8)
        )
        with patch.object(CPU, "is_supported", return_value=True):
            self.assertTrue(CPU.pack_projected_group(send, [], [], []))
            self.assertTrue(
                CPU.pack_projected_group(send, [torch.empty((0, 512))], [8], [0])
            )
            source = torch.ones((2, 512))
            for inputs, offsets, columns in (
                ([source, source], [1, 2], [0, 0]),
                ([source], [7], [0]),
                ([source], [0], [512]),
                ([send[:1]], [0], [0]),
                ([source], [0], []),
                ([source[:, ::2]], [0], [0]),
                ([source], [torch.empty((), device="meta")], [0]),
            ):
                self.assertFalse(
                    CPU.pack_projected_group(send, inputs, offsets, columns)
                )
            self.assertEqual(torch.count_nonzero(send).item(), 0)
            self.assertIsNone(
                CPU.restore_projected_group(send, torch.arange(8), 0, 2, 0, 0, 8)
            )
            self.assertIsNone(
                CPU.restore_projected_group(send, torch.arange(8), 0, 2, 8, 1, 8)
            )
            self.assertIsNone(
                CPU.restore_projected_group(
                    send, torch.arange(8), 0, 2, 8, 0, 8, out=send
                )
            )

    def _preflight_case(self, ratio=2):
        regions = (1 if ratio == 2 else 2, 3, 5)
        host = torch.arange(1, 7, dtype=torch.int32).reshape(1, 2, 3).repeat(3, 1, 1)
        cache = SimpleNamespace(
            group_region_names=regions,
            group_seq_size_per_block=(64, 64, 512),
            seq_size_per_block=128,
            kernel_seq_size_per_block=64,
        )
        pools = {
            regions[0]: torch.zeros((14, 64 // ratio, 288), dtype=torch.uint8),
            3: torch.zeros((14, 64, 68), dtype=torch.uint8),
            5: torch.zeros((14, 1024)),
        }
        owner = SimpleNamespace(
            global_norm=torch.ones(512),
            index_k_norm=torch.ones(128).bfloat16(),
            index_wk=torch.ones((128, 512)).bfloat16(),
        )
        attention = SimpleNamespace(
            compress_ratio=ratio,
            _kv_cache=cache,
            _shared_attention={"prefill_producer_host_tables": (regions, host)},
            _owner=lambda: owner,
            _global_region=lambda: regions[0],
            _source_pool=lambda region: pools.get(region),
            _source_entries=lambda region, pool: 2 if region == 5 else pool.shape[1],
            _block_tables_by_type={
                r: torch.zeros((2, 6), dtype=torch.int32) for r in regions
            },
            freqs_cis=torch.ones((2048, 32), dtype=torch.complex64),
            eps=1e-6,
        )
        cp = SimpleNamespace(
            cp_size=4,
            cp_rank=0,
            kv_cache_sharded=True,
            prefix_lengths_host=(511, 1023),
            input_lengths_global_host=(17, 33),
        )
        return attention, torch.zeros((32, 5120), dtype=torch.bfloat16), cp, pools

    def test_preflight_bound_pools_and_metadata_gates(self):
        with patch.object(CPU, "is_supported", return_value=True):
            for ratio in (1, 2):
                attention, x, cp, pools = self._preflight_case(ratio)
                self.assertTrue(CPU.can_batch_groups(attention, x, cp))
                attention._shared_attention.clear()
                self.assertFalse(CPU.can_batch_groups(attention, x, cp))
            attention, x, cp, pools = self._preflight_case()
            pools[5] = None
            self.assertFalse(CPU.can_batch_groups(attention, x, cp))
            for n in (1, 129):
                attention, x, cp, pools = self._preflight_case()
                cp.input_lengths_global_host = (17,) * n
                cp.prefix_lengths_host = (0,) * n
                self.assertFalse(CPU.can_batch_groups(attention, x, cp))
            cp.input_lengths_global_host = (1000000, 1)
            cp.prefix_lengths_host = (0, 0)
            self.assertFalse(CPU.can_batch_groups(attention, x, cp))
            cp.input_lengths_global_host = (1, 1)
            self.assertFalse(CPU.can_batch_groups(attention, x, cp))

    def test_preflight_layout_and_host_table_rejections(self):
        with patch.object(CPU, "is_supported", return_value=True):
            for key in (1, 3):
                attention, x, cp, pools = self._preflight_case()
                pools[key] = None
                self.assertFalse(CPU.can_batch_groups(attention, x, cp))
            attention, x, cp, pools = self._preflight_case()
            attention._owner().index_k_norm = torch.ones(128, dtype=torch.float16)
            self.assertFalse(CPU.can_batch_groups(attention, x, cp))
            attention, x, cp, pools = self._preflight_case()
            regions, host = attention._shared_attention["prefill_producer_host_tables"]
            for bad_host in (host[:1], host[:, :1], host.to("meta"), host.float()):
                attention._shared_attention["prefill_producer_host_tables"] = (
                    regions,
                    bad_host,
                )
                self.assertFalse(CPU.can_batch_groups(attention, x, cp))

    def test_preflight_rejects_shared_writable_pages_and_state(self):
        with patch.object(CPU, "is_supported", return_value=True):
            for region_idx in (0, 1, 2):
                attention, x, cp, pools = self._preflight_case()
                cp.prefix_lengths_host = (0, 0)
                cp.input_lengths_global_host = (512, 512)
                _, host = attention._shared_attention["prefill_producer_host_tables"]
                host[region_idx, 1] = host[region_idx, 0]
                self.assertFalse(CPU.can_batch_groups(attention, x, cp))
                for pool in pools.values():
                    self.assertEqual(torch.count_nonzero(pool).item(), 0)

    def test_preflight_unsharded_uses_single_rank_mapping(self):
        attention, x, cp, pools = self._preflight_case()
        cp.kv_cache_sharded = False
        cp.prefix_lengths_host = (0, 128)
        cp.input_lengths_global_host = (17, 33)
        with patch.object(CPU, "is_supported", return_value=True), patch.object(
            CPU, "state_slots_are_unique", return_value=True
        ) as proof:
            self.assertTrue(CPU.can_batch_groups(attention, x, cp))
            self.assertEqual(proof.call_args.args[-2:], (1, 0))

    def test_original_reduction_boundaries(self):
        self.assertEqual(
            [CPU.reduction_width(n) for n in (0, 1, 7, 8, 15, 16, 256)],
            [0, 128, 128, 64, 64, 32, 32],
        )
        with self.assertRaises(ValueError):
            CPU.reduction_width(-1)

    def test_group_offsets_and_mixed_segment_phases(self):
        prefixes = [0, 127, 2048]
        sizes = [[1, 2, 14, 16, 30, 32, 3], [1, 1, 13, 16, 30, 32, 4], [1, 17]]
        lengths = list(map(sum, sizes))
        for ratio in (1, 2):
            metadata = _metadata(prefixes, sizes, ratio)
            for start, end in ((0, lengths[0]), (lengths[0], sum(lengths))):
                plan = CPU.make_plan(
                    metadata, start, end, ratio=ratio, request_lengths=lengths
                )
                self.assertIsNotNone(plan)
                actual, widths = [], []
                for begin, _, first, stop, phase in plan.segments:
                    actual.extend(
                        range(
                            begin + phase, begin + phase + (stop - first) * ratio, ratio
                        )
                    )
                    widths.extend([CPU.reduction_width(stop - first)] * (stop - first))
                positions = [
                    p
                    for prefix, length in zip(prefixes, lengths)
                    for p in range(prefix, prefix + length)
                ][start:end]
                expected = [i for i, p in enumerate(positions) if (p + 1) % ratio == 0]
                self.assertEqual(actual, expected)
                self.assertEqual(len(widths), plan.count)
                if ratio == 2 and start == 0:
                    self.assertEqual(set(widths), {32, 64, 128})

    def test_empty_compact_segment_and_group(self):
        metadata = _metadata([0], [[1]], 2)
        plan = CPU.make_plan(metadata, 0, 1, ratio=2, request_lengths=[1])
        self.assertEqual(plan.count, 0)
        self.assertEqual(plan.segments, ((0, 1, 0, 0, 1),))

    def test_reject_partial_requests_and_bad_metadata(self):
        metadata = _metadata([0, 1], [[8, 8], [8, 8]], 2)
        for start, end in ((0, 8), (8, 16), (1, 16), (0, 33), (0, 0), (0, 65537)):
            self.assertIsNone(
                CPU.make_plan(metadata, start, end, ratio=2, request_lengths=[16, 16])
            )
        for segment in ((9, 0, 4, 0), (8, 1, 5, 1), (8, 0, 3, 1), (8, 0, 4, 2)):
            bad = SimpleNamespace(segments=dict(metadata.segments))
            bad.segments[0] = segment
            self.assertIsNone(
                CPU.make_plan(bad, 0, 16, ratio=2, request_lengths=[16, 16])
            )

    def test_suffix_proof_matches_actual_mapping(self):
        mapping = _state_mapping()
        prefixes, lengths = [0, 511, 4097, 8191], [1025, 577, 2048, 3]
        table = torch.arange(1, 4 * 20 + 1).reshape(4, 20)
        table[0, 0] = -1
        table[1, 1] = 0
        pos = torch.tensor(
            [
                p
                for prefix, length in zip(prefixes, lengths)
                for p in range(prefix, prefix + length)
            ]
        )
        req = torch.tensor(
            [r for r, length in enumerate(lengths) for _ in range(length)]
        )
        ends = torch.tensor(prefixes) + torch.tensor(lengths)
        for eb in (1, 2, 3, 16):
            for rank in range(4):
                self.assertTrue(
                    CPU.state_slots_are_unique(
                        table, prefixes, lengths, eb, 512, 4, rank
                    )
                )
                slots = mapping(pos, table, req, eb, 512, 4, rank, ends)
                valid = slots[slots >= 0].tolist()
                self.assertEqual(len(valid), len(set(valid)))

    def test_alias_and_wrapping_tables_are_unsupported(self):
        for rank in range(4):
            self.assertFalse(
                CPU.state_slots_are_unique([[1, 1]], [0], [1024], 2, 512, 4, rank)
            )
            self.assertFalse(
                CPU.state_slots_are_unique([[1]], [0], [1024], 2, 512, 4, rank)
            )
            self.assertFalse(
                CPU.state_slots_are_unique(
                    [[1], [1]], [0, 0], [512, 512], 2, 512, 4, rank
                )
            )

    def test_shared_readonly_prefix_is_allowed(self):
        for rank in range(4):
            self.assertTrue(
                CPU.state_slots_are_unique(
                    [[1, 2], [1, 3]], [512, 513], [7, 511], 2, 512, 4, rank
                )
            )

    def test_shared_partial_prefix_write_requires_proof(self):
        mapping = _state_mapping()
        prefixes, lengths = [400, 409], [21, 16]
        positions = torch.tensor(
            [p for start, n in zip(prefixes, lengths) for p in range(start, start + n)]
        )
        requests = torch.tensor([r for r, n in enumerate(lengths) for _ in range(n)])
        table = torch.tensor([[7], [7]], dtype=torch.int32)
        for rank in range(4):
            self.assertFalse(
                CPU.state_slots_are_unique(table, prefixes, lengths, 2, 512, 4, rank)
            )
            slots = mapping(
                positions,
                table,
                requests,
                2,
                512,
                4,
                rank,
                torch.tensor(prefixes) + torch.tensor(lengths),
            )
            valid = slots[slots >= 0].tolist()
            self.assertLess(len(set(valid)), len(valid))

    def test_proof_rejects_nonhost_metadata_without_readback(self):
        # Meta tensors have no data to read: rejection must precede tolist/item.
        unavailable = torch.empty((1,), device="meta", dtype=torch.int64)
        self.assertFalse(CPU.state_slots_are_unique([[1]], unavailable, [1], 2, 512))
        self.assertFalse(CPU.state_slots_are_unique([[1]], [0], unavailable, 2, 512))
        self.assertFalse(
            CPU.state_slots_are_unique(unavailable.view(1, 1), [0], [1], 2, 512)
        )

    def test_disabled_and_unproven_state_never_launch(self):
        tensor = torch.empty((1, 512))
        with patch.dict(os.environ, {"RTP_V41_BATCHED_PRODUCER": "0"}):
            self.assertFalse(CPU.is_supported(tensor))
        self.assertFalse(CPU.store_states(tensor, tensor, None, None))
        with patch.dict(os.environ, {"RTP_V41_BATCHED_PRODUCER": "1"}):
            self.assertIsNone(CPU.prepare(CPU.GroupPlan(0, 1, 0, 0, 2, ()), "cpu"))


@unittest.skipUnless(
    os.environ.get("DSV41_RUN_PRODUCER_GPU_TESTS") == "1",
    "GPU execution requires explicit assignment/opt-in",
)
class BatchedProducerGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rtp_llm.models_py.modules.dsv4.fp8 import (
            _v41_batched_producer,
            _v41_prefill_global,
        )

        cls.batched, cls.legacy = _v41_batched_producer, _v41_prefill_global
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("Blackwell CUDA13 environment required")

    def _assert_bytes(self, actual, expected):
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def _compare(self, ratio, norm_dtype, rank, strided=True, empty=False):
        batched, legacy = self.batched, self.legacy
        prefixes = [0, 127, 1023]
        sizes = [[1, 2, 14, 16, 30, 32, 3], [1, 1, 13, 16, 30, 32, 4], [1, 17]]
        if empty:
            prefixes, sizes = [0], [[1]]
        lengths = list(map(sum, sizes))
        metadata = _metadata(prefixes, sizes, ratio)
        rows = sum(lengths)
        device = "cuda"
        generator = torch.Generator().manual_seed(913)
        raw = torch.randn(rows, 1024 if strided else 512, generator=generator).to(
            device
        )
        values = raw[:, :512]
        scores = (
            raw[:, 512:]
            if strided
            else torch.randn(rows, 512, generator=generator).to(device)
        )
        values[0].fill_(-0.0)
        scores[0].fill_(80.0)
        positions_cpu = torch.tensor(
            [
                p
                for prefix, n in zip(prefixes, lengths)
                for p in range(prefix, prefix + n)
            ]
        )
        requests_cpu = torch.tensor(
            [r for r, n in enumerate(lengths) for _ in range(n)]
        )
        positions, requests = positions_cpu.to(device), requests_cpu.to(device)
        starts = torch.tensor(prefixes, device=device)
        previous = torch.randn(len(lengths), 1024, generator=generator).to(device)
        previous_before = previous.clone()
        norm = (torch.rand(512, generator=generator) + 0.5).to(
            device=device, dtype=norm_dtype
        )
        index_norm = (torch.rand(128, generator=generator) + 0.5).to(
            device=device, dtype=norm_dtype
        )
        projection = (torch.randn(128, 512, generator=generator) / math.sqrt(512)).to(
            device=device, dtype=torch.bfloat16
        )
        angles = (
            torch.arange(max(p + n for p, n in zip(prefixes, lengths))).float()[:, None]
            * torch.linspace(0.001, 0.1, 32)[None, :]
        )
        freqs = torch.polar(torch.ones_like(angles), angles).to(device)
        boundaries_cpu = torch.tensor(
            [i for i, p in enumerate(positions_cpu.tolist()) if (p + 1) % ratio == 0],
            dtype=torch.long,
        )
        boundaries = boundaries_cpu.to(device)
        compact_positions = positions[boundaries]
        count, entries = len(boundaries), 64
        pages = (count + entries - 1) // entries + 2

        def pool(width):
            return torch.full(
                (pages, entries * width + 128), 0x5A, dtype=torch.uint8, device=device
            )[:, : entries * width].view(pages, entries, width)

        main_ref, main_got = pool(288), pool(288)
        index_ref, index_got = pool(68), pool(68)
        main_slots = torch.arange(count, device=device) + entries
        main_slots[::4] = -1
        index_slots = main_slots.flip(0).contiguous()
        table = torch.arange(1, len(lengths) * 24 + 1).reshape(len(lengths), 24)
        table[:, 1::3] = -1
        state_slots = _state_mapping()(
            positions_cpu,
            table,
            requests_cpu,
            2,
            64,
            4,
            rank,
            torch.tensor(prefixes) + torch.tensor(lengths),
        ).to(device)
        self.assertTrue(
            batched.state_slots_are_unique(table, prefixes, lengths, 2, 64, 4, rank)
        )
        state_rows = (len(lengths) * 24 + 1) * 2
        state_ref = torch.full((state_rows, 1056), -123.0, device=device)[:, :1024]
        state_got = torch.full((state_rows, 1056), -123.0, device=device)[:, :1024]
        expected_latent = torch.empty((count, 512), dtype=torch.bfloat16, device=device)
        expected_index = torch.empty((count, 128), dtype=torch.bfloat16, device=device)
        carry = None
        for start, (end, first, stop, phase) in metadata.segments.items():
            idx = torch.arange(phase, end - start, ratio, device=device)
            latent = legacy.compress_main(
                values[start:end],
                scores[start:end] if ratio == 2 else None,
                norm,
                1e-6,
                positions[start:end],
                requests[start:end],
                starts,
                previous,
                idx,
                freqs,
                main_ref,
                main_slots[first:stop],
                ratio,
                carry,
            )
            self.assertIsNotNone(latent)
            expected_latent[first:stop] = latent
            if ratio == 2:
                self.assertTrue(
                    legacy.store_states(
                        values[start:end],
                        scores[start:end],
                        state_slots[start:end],
                        state_ref,
                    )
                )
                carry = (values[end - 1 : end].clone(), scores[end - 1 : end].clone())
            projected = torch.nn.functional.linear(latent, projection)
            expected_index[first:stop] = projected
            self.assertTrue(
                legacy.store_index(
                    projected,
                    index_norm,
                    1e-6,
                    compact_positions[first:stop],
                    freqs,
                    index_ref,
                    index_slots[first:stop],
                    ratio,
                )
            )

        groups = ((0, lengths[0]),) if empty else ((0, lengths[0]), (lengths[0], rows))
        actual_latent = torch.empty_like(expected_latent)
        with patch.dict(os.environ, {"RTP_V41_BATCHED_PRODUCER": "1"}):
            for start, end in groups:
                plan = batched.make_plan(
                    metadata, start, end, ratio=ratio, request_lengths=lengths
                )
                prepared = batched.prepare(plan, device)
                self.assertIsNotNone(prepared)
                first, stop = plan.first, plan.stop
                latent = batched.compress_main(
                    values[start:end],
                    scores[start:end] if ratio == 2 else None,
                    norm,
                    1e-6,
                    positions[start:end],
                    requests[start:end],
                    starts,
                    previous,
                    prepared,
                    freqs,
                    main_got,
                    main_slots[first:stop],
                )
                self.assertIsNotNone(latent)
                actual_latent[first:stop] = latent
                if ratio == 2:
                    self.assertTrue(
                        batched.store_states(
                            values[start:end],
                            scores[start:end],
                            state_slots[start:end],
                            state_got,
                            slots_are_unique=True,
                        )
                    )
                projected_group = torch.empty(
                    (plan.count, 128), dtype=torch.bfloat16, device=device
                )
                # This preserves the original GEMM M, layout, and launch order.
                for _, _, begin, finish, _ in plan.segments:
                    projected_group[begin:finish] = torch.nn.functional.linear(
                        latent[begin:finish], projection
                    )
                self._assert_bytes(projected_group, expected_index[first:stop])
                self.assertTrue(
                    batched.store_index(
                        projected_group,
                        index_norm,
                        1e-6,
                        compact_positions[first:stop],
                        freqs,
                        index_got,
                        index_slots[first:stop],
                        ratio,
                    )
                )
        for actual, expected in (
            (actual_latent, expected_latent),
            (main_got, main_ref),
            (index_got, index_ref),
            (state_got, state_ref),
            (previous, previous_before),
        ):
            self._assert_bytes(actual, expected)

    def test_mixed_widths_request_and_group_boundaries(self):
        for ratio in (1, 2):
            for dtype in (torch.float32, torch.bfloat16):
                for rank in range(4):
                    with self.subTest(ratio=ratio, dtype=dtype, rank=rank):
                        self._compare(ratio, dtype, rank)

    def test_contiguous_projection_and_empty_compact_group(self):
        self._compare(2, torch.bfloat16, 0, strided=False)
        self._compare(2, torch.float32, 0, empty=True)


@unittest.skipUnless(
    os.environ.get("DSV41_RUN_PRODUCER_GPU_TESTS") == "1",
    "GPU execution requires explicit assignment/opt-in",
)
class ProjectedTransportGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_batched_producer

        cls.batched = _v41_batched_producer
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("Blackwell required")

    def setUp(self):
        self.env = patch.dict(os.environ, {"RTP_V41_BATCHED_PRODUCER": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def _bits(self, count):
        patterns = torch.tensor(
            [
                0,
                -2147483648,
                0x7FC12345,
                0x7F800001,
                0x7F800000,
                -8388608,
                1,
                -1,
                0x3F800000,
            ],
            dtype=torch.int32,
        )
        return patterns.repeat((count + len(patterns) - 1) // len(patterns))[:count].to(
            "cuda"
        )

    def _same(self, actual, expected):
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def test_pack_mixed_segments_columns_bits_and_canaries(self):
        for width in (512, 1024):
            with self.subTest(width=width):
                backing = torch.full(
                    (66, width + 17), 0x12345678, dtype=torch.int32, device="cuda"
                )
                expected = backing.clone()
                send = backing.view(torch.float32)[1:-1, 2 : 2 + width]
                target = expected.view(torch.float32)[1:-1, 2 : 2 + width]
                layout = [
                    (40, 15, 0),
                    (1, 1, 0),
                    (26, 3, 0),
                    (4, 7, 0),
                    (15, 8, 0),
                    (64, 0, 0),
                ]
                if width == 1024:
                    layout += [(1, 2, 512), (17, 9, 512)]
                sources = [
                    self._bits(n * 512 + 1)[1:].view(n, 512).view(torch.float32)
                    for _, n, _ in layout
                ]
                originals = [s.clone() for s in sources]
                for (offset, n, column), source in zip(layout, sources):
                    target[offset : offset + n, column : column + 512].copy_(source)
                self.assertTrue(
                    self.batched.pack_projected_group(
                        send, sources, [t[0] for t in layout], [t[2] for t in layout]
                    )
                )
                self._same(backing, expected)
                for source, original in zip(sources, originals):
                    self._same(source, original)

    def test_pack_empty_invalid_and_table_bound(self):
        send = torch.zeros((1024, 512), dtype=torch.float32, device="cuda")
        before = send.clone()
        source = self._bits(512).view(1, 512).view(torch.float32)
        self.assertTrue(self.batched.pack_projected_group(send, [], [], []))
        self.assertTrue(
            self.batched.pack_projected_group(send, [source[:0]], [1024], [0])
        )
        self.assertFalse(
            self.batched.pack_projected_group(
                send, [source] * 513, list(range(513)), [0] * 513
            )
        )
        self.assertFalse(
            self.batched.pack_projected_group(send, [source, source], [0, 0], [0, 0])
        )
        self.assertFalse(self.batched.pack_projected_group(send, [send[:1]], [0], [0]))
        self._same(send, before)
        self.assertTrue(
            self.batched.pack_projected_group(
                send, [source] * 512, list(range(512)), [0] * 512
            )
        )
        self._same(send[:512], source.expand(512, 512))
        self._same(send[512:], before[512:])

    def test_restore_integer_mapping_strides_bits_and_canaries(self):
        for width in (512, 1024):
            for dtype in (torch.int32, torch.int64):
                for map_stride in (1, 3):
                    with self.subTest(width=width, dtype=dtype, map_stride=map_stride):
                        local_start, local_rows, full_chunk = 17, 13, 83
                        real_start, real_rows = 3, 37
                        gathered_backing = self._bits(54 * (width + 17)).reshape(
                            54, width + 17
                        )
                        gathered = gathered_backing.view(torch.float32)[
                            1:-1, 1 : 1 + width
                        ]
                        original = gathered_backing.clone()
                        selected = [
                            (i * 3 % 4, i * 7 % local_rows) for i in range(real_rows)
                        ]
                        indices = torch.tensor(
                            [rank * local_rows + local for rank, local in selected],
                            dtype=torch.long,
                            device="cuda",
                        )
                        mapping_backing = torch.full(
                            ((real_start + real_rows + 2) * map_stride + 1,),
                            -1,
                            dtype=dtype,
                            device="cuda",
                        )
                        mapping = mapping_backing[1::map_stride]
                        mapping[real_start : real_start + real_rows] = torch.tensor(
                            [
                                rank * full_chunk + local_start + local
                                for rank, local in selected
                            ],
                            dtype=dtype,
                            device="cuda",
                        )
                        output_backing = torch.full(
                            (real_rows + 2, width + 11),
                            0x12345678,
                            dtype=torch.int32,
                            device="cuda",
                        )
                        expected = output_backing.clone()
                        expected.view(torch.float32)[1:-1, 2 : 2 + width].copy_(
                            gathered.index_select(0, indices)
                        )
                        out = output_backing.view(torch.float32)[1:-1, 2 : 2 + width]
                        actual = self.batched.restore_projected_group(
                            gathered,
                            mapping,
                            local_start,
                            local_rows,
                            full_chunk,
                            real_start,
                            real_rows,
                            out=out,
                        )
                        self.assertIs(actual, out)
                        self._same(output_backing, expected)
                        self._same(gathered_backing, original)

    def test_restore_empty_and_large_int64_offsets(self):
        empty = torch.empty((0, 512), dtype=torch.float32, device="cuda")
        restored = self.batched.restore_projected_group(
            empty, torch.empty(0, dtype=torch.long, device="cuda"), 0, 0, 1, 0, 0
        )
        self.assertEqual(restored.shape, (0, 512))
        gathered = self._bits(12 * 512).view(12, 512).view(torch.float32)
        chunk, start = 2**31 + 83, 2**31 + 17
        mapping = torch.tensor(
            [3 * chunk + start + 2, start, 2 * chunk + start + 1],
            device="cuda",
            dtype=torch.int64,
        )
        actual = self.batched.restore_projected_group(
            gathered, mapping, start, 3, chunk, 0, 3
        )
        self._same(actual, gathered[torch.tensor([11, 0, 7], device="cuda")])

    def test_cross_stream_pack_source_lifetime_and_private_warmup(self):
        self.assertTrue(self.batched.warmup_projected_groups(torch.device("cuda", 0)))
        stream = torch.cuda.Stream()
        source = self._bits(17 * 512).view(17, 512).view(torch.float32)
        expected = source.clone()
        send = torch.zeros((17, 512), device="cuda")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.assertTrue(self.batched.pack_projected_group(send, [source], [0], [0]))
        del source
        pressure = [torch.zeros((17, 512), device="cuda") for _ in range(16)]
        torch.cuda.current_stream().wait_stream(stream)
        self._same(send, expected)
        del pressure


if __name__ == "__main__":
    unittest.main()
