"""Existing local Prefill row kernels against unchanged codec/workspace paths."""

import json
import math
import os
from pathlib import Path
from contextlib import nullcontext
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.cp import CPContext
from rtp_llm.models_py.modules.dsv41.ppu import _quant, graph_cache_kernels, prefill
from rtp_llm.models_py.modules.dsv41.ppu.ced import _bounded_query_layout
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import decode_kv, encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache, _StagedSwaSegments
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy
from rtp_llm.models_py.modules.dsv41.prefill.ced import _bounded_replay_selected


LOCAL_ENV = "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE"
FUSED_ENV = "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE_FUSED"
CP_INDEX_ENV = "RTP_V41_PPU_CP8_FUSED_WORKSPACE_INDICES"
CP_DEQUANT_ENV = "RTP_V41_PPU_CP8_FUSED_WORKSPACE_DEQUANT"


def assert_bytes(test, actual, expected):
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.shape, expected.shape)
    test.assertTrue(torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8),
    ))


def _bounded_layout(prefixes, lengths, starts, device="cpu", *, selected=None):
    """Build the actual CP1 replay query, including its odd-count padding."""
    device = torch.device(device)
    rows = sum(lengths)
    indices = torch.arange(rows, device=device)
    requests = torch.tensor([rid for rid, n in enumerate(lengths) for _ in range(n)],
                            device=device, dtype=torch.int32)
    positions = torch.cat([torch.arange(p, p + n, device=device) for p, n in zip(prefixes, lengths)])
    valid = torch.ones(rows, device=device, dtype=torch.bool)
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    original = CPContext(
        cp_size=1, cp_rank=0, chunk_length=rows, padded_seq_len=rows, seq_len_full=rows,
        relative_positions=indices, prefix_length=prefixes[0], global_positions=positions,
        local_is_real=valid, unpad_restore=indices, seq_len_total=prefixes[0] + lengths[0],
        cp_info=NS(prefill_qkv_restore_indice_cpu=torch.arange(rows),
                   prefill_qkv_padding_mask_cpu=torch.ones(rows, dtype=torch.bool)),
        req_id_per_token=requests, prefix_lengths=torch.tensor(prefixes, device=device),
        input_lengths_global=torch.tensor(lengths, device=device),
        cu_seqlens_global=torch.tensor(cumulative, device=device),
        chunk_lengths_per_req=tuple(lengths), input_lengths_global_host=tuple(lengths),
        prefix_lengths_host=tuple(prefixes),
    )
    bounded = selected is None
    query, _, _ = _bounded_query_layout(
        original, _bounded_replay_selected(lengths, starts) if bounded else selected,
    )
    if bounded:
        if len(starts) == 1:
            query.swa_replay_start = starts[0]
        else:
            query.swa_replay_starts_host = tuple(starts)
    cumulative = [0]
    for chunk in query.chunk_lengths_per_req:
        cumulative.append(cumulative[-1] + chunk)
    context = prefill.build_step_context(
        query.global_positions, query.req_id_per_token.long(),
        torch.tensor(cumulative, device=device, dtype=torch.int32), valid_mask=query.local_is_real,
    )
    return query, context


def _tensor_bounds(context, chunks):
    return [(rid, int(part.min()), int(part.max()))
            for rid, part in enumerate(context.positions.split(chunks)) if part.numel()]


class _NativePools:
    """Physical tag tables and padded native rows, with no collective backend."""

    def __init__(self, geometry, totals, device):
        self.entries = {"hca_kv": 128, "csa_kv": 64, "indexer_kv": 128,
                        "swa_kv": 256, "csa_state": 2}
        self.spans = {"hca_kv": 128, "csa_kv": 128, "indexer_kv": 128,
                      "swa_kv": 512, "csa_state": 512}
        keys = [(layer, "swa_kv") for layer in range(geometry.num_layers)]
        for source in geometry.kv_source_layer_ids:
            keys.extend(((source, "csa_kv" if geometry.ratio(source) == 2 else "hca_kv"),
                         (source, "indexer_kv")))
            if geometry.ratio(source) == 2:
                keys.append((source, "csa_state"))
        self.handles, self.inputs = {}, {}
        self.group_tags = sorted({tag for _, tag in keys})
        for layer, tag in keys:
            columns = math.ceil(max(totals, default=0) / self.spans[tag]) + 1
            blocks = len(totals) * columns
            if tag not in self.inputs:
                host = torch.arange(blocks, dtype=torch.int32).reshape(len(totals), columns).flip(1)
                table = host.to(device)
                self.inputs[tag] = NS(
                    kv_cache_block_id=host, kv_cache_block_id_device=table,
                    kv_cache_kernel_block_id=host, kv_cache_kernel_block_id_device=table,
                )
            width = {"hca_kv": 288, "csa_kv": 288, "indexer_kv": 68,
                     "swa_kv": 528, "csa_state": 1024}[tag]
            dtype = torch.float32 if tag == "csa_state" else torch.uint8
            self.handles[layer, tag] = NS(
                kv_cache_base=torch.full((blocks, self.entries[tag] * width + 16),
                                         173, device=device, dtype=dtype),
                layer_id=layer, tag=tag,
            )

    def get_layer_cache(self, layer, tag):
        return self.handles[layer, tag]

    def get_entries_per_block(self, tag):
        return self.entries[tag]

    def get_seq_size_per_block(self, tag):
        return self.spans[tag]

    def get_kernel_seq_size_per_block(self, tag):
        return self.spans[tag]


class _NativeFixture:
    def __init__(self, device, *, ratio=2, prefixes=(129,), lengths=(9,), queries=None):
        self.device, self.ratio = device, ratio
        self.prefixes, self.lengths = tuple(prefixes), tuple(lengths)
        totals = tuple(p + n for p, n in zip(prefixes, lengths))
        self.geometry = NS(
            num_layers=4, kv_source_layer_ids=(0, 2) if ratio else (),
            head_dim=512, window=128, ratio=lambda layer: ratio,
            kv_source_for=lambda layer: (0 if layer < 2 else 2) if ratio else None,
        )
        self.native = _NativePools(self.geometry, totals, device)
        self.cache = SharedNativeCache(
            PrefillPolicy().freeze({}), self.geometry, self.native, self.native.inputs,
            totals, device=device, prefill=True,
        )
        self.ops = NS(decode_kv=decode_kv)
        for source in self.geometry.kv_source_layer_ids:
            tag = "global_kv_2" if ratio == 2 else "global_kv_1"
            for request, total in enumerate(totals):
                positions = torch.arange(total // ratio, device=device) * ratio
                packed = encode_kv(self.values(positions, source + request), pool_tag=tag)
                self.cache._copy_rows((source, tag), request, positions, packed)
        for layer in range(4):
            self.stage_layer(layer)
        self.full_layout = queries is None
        if queries is None:
            queries = [list(range(p, p + n)) for p, n in zip(prefixes, lengths)]
        self.chunks = tuple(map(len, queries))
        positions = torch.tensor([p for query in queries for p in query], dtype=torch.int64, device=device)
        requests = torch.tensor([rid for rid, query in enumerate(queries) for _ in query],
                                dtype=torch.int64, device=device)
        offsets = [0]
        for chunk in self.chunks:
            offsets.append(offsets[-1] + chunk)
        self.context = prefill.build_step_context(
            positions, requests, torch.tensor(offsets, device=device, dtype=torch.int32),
        )
        # The original local token deliberately has no CP-specific attributes.
        self.layout = NS(chunk_lengths_per_req=self.chunks, prefix_lengths_host=self.prefixes)
        self.selected = None
        if ratio:
            backing = torch.full((positions.numel(), 10), -1, device=device, dtype=torch.int32)
            self.selected = backing[:, ::2]
            self.selected.copy_(torch.tensor([0, -1, 0, 3, 999], device=device, dtype=torch.int32))

    def values(self, positions, shift=0):
        row = (positions[:, None] % 29 - 14).float() * 0.125
        column = (torch.arange(512, device=self.device)[None] % 17).float() * 0.015625
        return (row + column + shift * 0.03125).to(torch.bfloat16)

    def stage_layer(self, layer, shift=0):
        all_positions, all_requests, all_fresh = [], [], []
        for request, (prefix, length) in enumerate(zip(self.prefixes, self.lengths)):
            old = torch.arange(max(0, prefix - 127), prefix, device=self.device)
            old_packed = encode_kv(self.values(old, layer + request + shift), pool_tag="swa_kv")
            self.cache._copy_rows((layer, "swa_kv"), request, old, old_packed)
            fresh = torch.arange(prefix, prefix + length, device=self.device)
            all_positions.append(fresh)
            all_requests.append(torch.full_like(fresh, request))
            all_fresh.append(encode_kv(self.values(fresh, layer + request + shift), pool_tag="swa_kv"))
        self.cache.stage_swa(
            layer, torch.cat(all_positions), torch.cat(all_requests), torch.cat(all_fresh), self.prefixes,
        )

    def args(self, layer=1, starts=None):
        return dict(
            ops=self.ops, cache=self.cache, layer_id=layer, geometry=self.geometry,
            context=self.context, selected_blocks=self.selected, window=128, swa_starts=starts,
        )

    def build(self, *, fused, layer=1, shared=None, starts=None):
        shared = {} if shared is None else shared
        if self.full_layout:
            shared["local_prefill_workspace_layout"] = self.layout
        # Keep a test-only unfused oracle after production switches retire.
        # The fused arm uses the actual automatic metadata admission.
        unfused = nullcontext() if fused else patch.object(
            prefill, "_local_prefill_workspace_fusion_supported", return_value=False,
        )
        with unfused:
            output = prefill.build_cp_prefill_workspace(
                **self.args(layer, starts), cp=self.layout, shared=shared, q8=False,
            )
        return output, shared["prefill_kv_workspace"][2]


class LocalPrefillFusedCPU(unittest.TestCase):
    def test_host_bounds_equal_actual_full_and_bounded_cp1_layout(self):
        for prefixes, lengths, starts in (
            ((126976,), (4096,), (0,)),
            ((126976,), (4096,), (3956,)),
            ((0,), (1,), (0,)),
            ((19,), (139,), (0,)),
            ((129, 0, 31, 7), (4096, 7, 0, 11), (3956, 0, 0, 8)),
            ((0, 17, 129), (0, 1, 5), (0, 0, 2)),
        ):
            with self.subTest(prefixes=prefixes, lengths=lengths, starts=starts):
                query, context = _bounded_layout(prefixes, lengths, starts)
                cache = NS(lengths=tuple(p + n for p, n in zip(prefixes, lengths)))
                actual = prefill._local_prefill_workspace_bounds(cache, context, query)
                self.assertEqual(actual, _tensor_bounds(context, query.chunk_lengths_per_req))
                # The adapter's unpadded full layout has the same extrema as
                # a replay selection retaining every fresh row, even if odd.
                full_query, full_context = _bounded_layout(prefixes, lengths, (0,) * len(lengths))
                full = NS(chunk_lengths_per_req=lengths, prefix_lengths_host=prefixes)
                unpadded = NS(positions=torch.cat([
                    torch.arange(p, p + n) for p, n in zip(prefixes, lengths)
                ]), request_ids=torch.empty(sum(lengths), dtype=torch.long))
                self.assertEqual(
                    prefill._local_prefill_workspace_bounds(cache, unpadded, full, full_layout=True),
                    _tensor_bounds(full_context, full_query.chunk_lengths_per_req),
                )
        empty = NS(chunk_lengths_per_req=(0, 0), prefix_lengths_host=(0, 19))
        context = NS(positions=torch.empty(0, dtype=torch.long), request_ids=torch.empty(0, dtype=torch.long))
        self.assertEqual(prefill._local_prefill_workspace_bounds(
            NS(lengths=(0, 19)), context, empty, full_layout=True,
        ), [])

    def test_host_bounds_reject_unowned_exact_or_mismatched_layouts(self):
        query, context = _bounded_layout((257,), (4096,), (3956,))
        cache = NS(lengths=(4353,))
        full = NS(chunk_lengths_per_req=(140,), prefix_lengths_host=(257,))
        self.assertIsNone(prefill._local_prefill_workspace_bounds(cache, context, full))
        self.assertIsNone(prefill._local_prefill_workspace_bounds(cache, context, full, full_layout=True))
        for field, value in (
            ("swa_replay_start", None), ("swa_replay_start", 3955),
            ("swa_replay_start", 4097), ("cp_size", 8), ("cp_rank", 1),
            ("kv_cache_sharded", True), ("chunk_lengths_per_req", (138,)),
            ("global_positions", query.global_positions.clone()),
            ("local_is_real", query.local_is_real.clone()),
            ("gather_restore_positions", query.gather_restore_positions[:-1]),
            ("gather_restore_positions", query.gather_restore_positions.int()),
        ):
            with self.subTest(field=field), patch.object(query, field, value):
                self.assertIsNone(prefill._local_prefill_workspace_bounds(cache, context, query))
        self.assertIsNone(prefill._local_prefill_workspace_bounds(NS(lengths=(4354,)), context, query))
        # Exact selections can have holes and duplicates. Neither the CP1
        # shape nor a same-sized selection proves a bounded suffix.
        exact, exact_context = _bounded_layout((17,), (9,), (0,), selected=torch.tensor([0, 0, 3, 8]))
        self.assertIsNone(prefill._local_prefill_workspace_bounds(NS(lengths=(26,)), exact_context, exact))

    def test_admission_is_local_ppu_prefill_and_ignores_retired_selectors(self):
        cache = SharedNativeCache.__new__(SharedNativeCache)
        cache._is_prefill, cache.sharded, cache.lengths = True, False, (17,)
        device = torch.device("cuda", 0)
        positions = NS(is_cuda=True, ndim=1, dtype=torch.int64, shape=(7,), device=device, numel=lambda: 7)
        context = NS(positions=positions, request_ids=NS(shape=(7,), device=device))
        layout, geometry = NS(chunk_lengths_per_req=(7,)), NS(head_dim=512)
        with patch.object(_quant, "is_ppu", return_value=True):
            self.assertTrue(prefill._local_prefill_workspace_fusion_supported(cache, geometry, context, layout))
            for flag in ("0", "1"):
                with patch.dict(os.environ, {FUSED_ENV: flag, CP_INDEX_ENV: flag, CP_DEQUANT_ENV: flag}):
                    self.assertTrue(prefill._local_prefill_workspace_fusion_supported(cache, geometry, context, layout))
            for name, value in (("is_cuda", False), ("ndim", 2), ("dtype", torch.int32)):
                old = getattr(positions, name)
                setattr(positions, name, value)
                self.assertFalse(prefill._local_prefill_workspace_fusion_supported(cache, geometry, context, layout))
                setattr(positions, name, old)
            for name, value in (("_is_prefill", False), ("sharded", True)):
                old = getattr(cache, name)
                setattr(cache, name, value)
                self.assertFalse(prefill._local_prefill_workspace_fusion_supported(cache, geometry, context, layout))
                setattr(cache, name, old)
            for invalid in (NS(cp_size=8, chunk_lengths_per_req=(7,)),
                            NS(cp_size=1, kv_cache_sharded=True, chunk_lengths_per_req=(7,)),
                            NS(chunk_lengths_per_req=(6,)), NS(chunk_lengths_per_req=(7, 0))):
                self.assertFalse(prefill._local_prefill_workspace_fusion_supported(cache, geometry, context, invalid))
            geometry.head_dim = 8
            self.assertFalse(prefill._local_prefill_workspace_fusion_supported(cache, geometry, context, layout))
            geometry.head_dim = 512
            with patch.object(_quant, "is_ppu", return_value=False):
                self.assertFalse(prefill._local_prefill_workspace_fusion_supported(cache, geometry, context, layout))

    @torch.inference_mode()
    def test_cpu_rows_keep_reference_fallback_under_auto_selection(self):
        for ratio in (0, 1, 2):
            with self.subTest(ratio=ratio):
                fixture = _NativeFixture("cpu", ratio=ratio, prefixes=(3, 19, 7), lengths=(4, 0, 3))
                with patch.object(prefill, "_full_cp_workspace_bounds", side_effect=AssertionError("local token is not CP")), patch.object(
                    prefill, "_local_prefill_workspace_bounds", side_effect=AssertionError("CPU must retain the existing oracle"),
                ):
                    for fused in (False, True):
                        actual, workspace = fixture.build(fused=fused)
                        self.assertFalse(workspace.fused_dequant or workspace.fused_indices)
                        expected = prefill._build_reference_workspace(**fixture.args())
                        for left, right in zip(actual, expected):
                            assert_bytes(self, left, right)

    @torch.inference_mode()
    def test_cp8_auto_fusion_ignores_retired_selectors_and_preserves_row_prep_gate(self):
        fixture = _NativeFixture("cpu", ratio=0, prefixes=(0,), lengths=(4,))
        with patch.object(fixture.cache, "cp8_row_prep_enabled", return_value=True), patch.object(
            prefill, "_full_cp_workspace_bounds", return_value=None,
        ) as bounds:
            for flag in ("0", "1"):
                with patch.dict(os.environ, {LOCAL_ENV: "1", FUSED_ENV: "0", CP_INDEX_ENV: flag, CP_DEQUANT_ENV: flag}):
                    output = prefill.build_cp_prefill_workspace(
                        **fixture.args(), cp=fixture.layout, shared=(shared := {}), q8=False,
                    )
                workspace = shared["prefill_kv_workspace"][2]
                self.assertTrue(workspace.fused_dequant and workspace.fused_indices)
                expected = prefill._build_reference_workspace(**fixture.args())
                for left, right in zip(output, expected):
                    assert_bytes(self, left, right)
            self.assertEqual(bounds.call_count, 2)
        # CP8 placement/row preparation remains an independent prerequisite.
        with patch.object(fixture.cache, "cp8_row_prep_enabled", return_value=False):
            prefill.build_cp_prefill_workspace(
                **fixture.args(), cp=fixture.layout, shared=(shared := {}), q8=False,
            )
        workspace = shared["prefill_kv_workspace"][2]
        self.assertFalse(workspace.fused_dequant or workspace.fused_indices)


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test",
)
class LocalPrefillFusedDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(_quant.is_ppu(), "local fusion qualification requires a real ZW PPU")

    @torch.inference_mode()
    def test_indices_and_dequant_decline_independently(self):
        fixture = _NativeFixture("cuda", ratio=2, prefixes=(129,), lengths=(9,))
        expected, _ = fixture.build(fused=False)
        indices_op = graph_cache_kernels.try_assemble_cp_prefill_indices
        dequant_op = _quant.try_dequantize_prefill_rows
        for decline_indices, decline_dequant in (
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(
                indices=decline_indices, dequant=decline_dequant
            ), patch.object(
                graph_cache_kernels,
                "try_assemble_cp_prefill_indices",
                wraps=None if decline_indices else indices_op,
                **({"return_value": None} if decline_indices else {}),
            ) as indices, patch.object(
                _quant,
                "try_dequantize_prefill_rows",
                wraps=None if decline_dequant else dequant_op,
                **({"return_value": False} if decline_dequant else {}),
            ) as dequant, patch.object(
                fixture.cache,
                "try_decode_swa_range",
                return_value=False,
            ), patch.dict(
                os.environ, {FUSED_ENV: "0", CP_INDEX_ENV: "0", CP_DEQUANT_ENV: "0"}
            ):
                actual, _ = fixture.build(fused=True)
            self.assertEqual(indices.call_count, 1)
            self.assertGreater(dequant.call_count, 0)
            for left, right in zip(actual, expected):
                assert_bytes(self, left, right)

    @torch.inference_mode()
    def test_cp8_query_ranks_use_auto_indices_with_padding(self):
        # All eight zigzag query layouts against one complete synthetic bank;
        # this is not a distributed gather/placement or full-model test.
        for rank in range(8):
            first, last = rank * 2, (16 - rank) * 2
            query = list(range(first, first + 2)) + list(range(last - 2, last))
            query = [min(p, 30) for p in query]
            fixture = _NativeFixture(
                "cuda", ratio=2, prefixes=(0,), lengths=(31,), queries=[query]
            )
            fixture.layout = NS(
                cp_size=8,
                cp_rank=rank,
                chunk_lengths_per_req=(4,),
                input_lengths_global_host=(31,),
                prefix_lengths_host=(0,),
                gather_restore_positions=None,
                global_positions=fixture.context.positions,
            )
            expected = prefill._build_reference_workspace(**fixture.args())
            with patch.object(
                fixture.cache, "cp8_row_prep_enabled", return_value=True
            ), patch.dict(
                os.environ,
                {FUSED_ENV: "0", CP_INDEX_ENV: "0", CP_DEQUANT_ENV: "0"},
            ), patch.object(
                graph_cache_kernels,
                "try_assemble_cp_prefill_indices",
                wraps=(graph_cache_kernels.try_assemble_cp_prefill_indices),
            ) as indices:
                actual, workspace = fixture.build(fused=True)
            self.assertTrue(workspace.fused_indices and workspace.fused_dequant)
            self.assertEqual(indices.call_count, 1)
            for left, right in zip(actual, expected):
                assert_bytes(self, left, right)

    @torch.inference_mode()
    def test_cp8_swa_segments_ignore_retired_dequant_selector(self):
        from rtp_llm.models_py.modules.dsv41.ppu import prefill_swa
        from rtp_llm.models_py.modules.dsv41.ppu.cp_sharding import PoolPlacement

        fixture = _NativeFixture("cuda", ratio=0, prefixes=(129,), lengths=(9,))
        cache = fixture.cache
        pool = cache.pools[1, "swa_kv"]
        # Only staging/readback is under test. Communication and native writes
        # below are doubles; real CP metadata and the decoder remain in use.
        cache.placements[1, "swa_kv"] = PoolPlacement.create(
            {"size": 8, "rank": 0},
            (1, "swa_kv"),
            pool,
            fixture.native,
            cache.inputs[pool.framework_tag],
            fixture.geometry,
        )
        old_positions = torch.arange(2, 129, device="cuda")
        positions = torch.arange(129, 138, device="cuda")
        tail = encode_kv(fixture.values(old_positions), pool_tag="swa_kv")
        fresh = encode_kv(fixture.values(positions, 2), pool_tag="swa_kv")
        expected = torch.empty((136, 512), dtype=torch.bfloat16, device="cuda")
        decode_kv(torch.cat((tail, fresh)), pool_tag="swa_kv", out=expected)
        cache.bind_swa_fresh_lengths((9,))
        for flag in ("0", "1"):
            with patch.object(
                cache, "_batched_swa_enabled", return_value=True
            ), patch.object(
                cache,
                "cp8_row_prep_enabled",
                return_value=True,
            ), patch.object(
                prefill_swa, "gather_tails", return_value=(tail,)
            ), patch.object(
                prefill_swa,
                "write_prefill_range",
                return_value=list(range(129, 138)),
            ), patch.dict(
                os.environ,
                {CP_DEQUANT_ENV: flag, "RTP_V41_PPU_CP8_DIRECT_SWA_WORKSPACE": "1",
                 "RTP_V41_PPU_CP8_FUSED_SWA_WRITE": "0"},
            ):
                cache.stage_swa(
                    1, positions, torch.zeros_like(positions), fresh, (129,)
                )
            saved = cache._fresh_swa[1, 0]
            self.assertIsInstance(saved, _StagedSwaSegments)
            output = torch.empty_like(expected)
            self.assertTrue(cache.try_decode_swa_range(1, 0, 2, output))
            assert_bytes(self, output, expected)
            assert_bytes(
                self,
                cache.read_swa(1, 0, torch.arange(2, 138, device="cuda")),
                torch.cat((tail, fresh)),
            )

    @torch.inference_mode()
    def test_supported_host_bounds_avoid_min_max_download_and_preserve_bytes(self):
        for ratio in (0, 1, 2):
            for prefixes, lengths, starts in (
                ((257,), (4096,), None),
                ((257,), (4096,), (3956,)),
                ((129, 31, 77), (9, 0, 7), (0, 0, 4)),
            ):
                with self.subTest(ratio=ratio, lengths=lengths, starts=starts):
                    fixture = _NativeFixture("cuda", ratio=ratio, prefixes=prefixes, lengths=lengths)
                    replay = None
                    if starts is not None:
                        fixture.layout, fixture.context = _bounded_layout(prefixes, lengths, starts, "cuda")
                        fixture.chunks = fixture.layout.chunk_lengths_per_req
                        fixture.full_layout = False
                        if ratio:
                            fixture.selected = fixture.selected[:1].expand(sum(fixture.chunks), -1).clone()
                        replay = {rid: p + s for rid, (p, s) in enumerate(zip(prefixes, starts))}
                    expected, _ = fixture.build(fused=False, starts=replay)
                    positions = fixture.context.positions
                    query_storage = positions.untyped_storage()._cdata

                    def query_only(operation, name):
                        def checked(tensor, *args, **kwargs):
                            # split() changes the view's data_ptr/storage_offset.
                            # Storage identity covers those query views while
                            # permitting native pool coordinate validation.
                            if (tensor.device == positions.device
                                    and tensor.untyped_storage()._cdata == query_storage):
                                raise AssertionError(f"constructor query-position {name}")
                            return operation(tensor, *args, **kwargs)
                        return checked

                    guarded = {name: query_only(getattr(torch.Tensor, name), name)
                               for name in ("min", "max", "tolist")}
                    # The old limits.tolist() is downstream of forbidden query
                    # reductions. Unrelated native-cache tolist() is permitted.
                    with patch.object(torch.Tensor, "min", new=guarded["min"]), patch.object(
                        torch.Tensor, "max", new=guarded["max"],
                    ), patch.object(torch.Tensor, "tolist", new=guarded["tolist"]):
                        for name in guarded:
                            with self.assertRaisesRegex(AssertionError, "constructor query-position"):
                                getattr(positions.split(fixture.chunks)[0][1:], name)()
                        actual, workspace = fixture.build(fused=True, starts=replay)
                    self.assertTrue(workspace.fused_dequant and workspace.fused_indices)
                    for left, right in zip(actual, expected):
                        assert_bytes(self, left, right)

    @torch.inference_mode()
    def test_native_strides_multirequest_topk_and_source_refresh_are_exact(self):
        for ratio in (0, 1, 2):
            with self.subTest(ratio=ratio):
                fixture = _NativeFixture("cuda", ratio=ratio, prefixes=(129, 257, 31), lengths=(9, 0, 7))
                self.assertFalse(fixture.selected is not None and fixture.selected.is_contiguous())
                for layer in (1, 3):
                    old, _ = fixture.build(fused=False, layer=layer)
                    with patch.object(prefill, "_full_cp_workspace_bounds", side_effect=AssertionError("local token is not CP")), patch.object(
                        fixture.ops, "decode_kv", side_effect=AssertionError("eligible rows must use fused dequant"),
                    ):
                        actual, workspace = fixture.build(fused=True, layer=layer)
                    self.assertTrue(workspace.fused_dequant and workspace.fused_indices)
                    expected = prefill._build_reference_workspace(**fixture.args(layer))
                    for left, previous, reference in zip(actual, old, expected):
                        assert_bytes(self, left, previous)
                        assert_bytes(self, left, reference)
                shared = {}
                first, _ = fixture.build(fused=True, shared=shared)
                saved = first[0].clone()
                fixture.stage_layer(1, shift=2)
                second, _ = fixture.build(fused=True, shared=shared)
                self.assertEqual(first[0].data_ptr(), second[0].data_ptr())
                self.assertFalse(torch.equal(second[0], saved))
                expected = prefill._build_reference_workspace(**fixture.args())
                for left, right in zip(second, expected):
                    assert_bytes(self, left, right)
                if fixture.selected is not None:
                    fixture.selected = fixture.selected.flip(1).contiguous()
                    third, _ = fixture.build(fused=True, shared=shared)
                    self.assertIsNot(second[1], third[1])
                    for left, right in zip(third, prefill._build_reference_workspace(**fixture.args())):
                        assert_bytes(self, left, right)
                previous_workspace = shared["prefill_kv_workspace"][2]
                switched, replacement = fixture.build(fused=True, layer=3, shared=shared)
                if ratio:
                    self.assertIsNot(previous_workspace, replacement)
                    self.assertNotEqual(second[0].data_ptr(), switched[0].data_ptr())
                for left, right in zip(switched, prefill._build_reference_workspace(**fixture.args(3))):
                    assert_bytes(self, left, right)
                for handle in fixture.native.handles.values():
                    self.assertTrue(bool((handle.kv_cache_base[:, -16:] == 173).all()))

    @torch.inference_mode()
    def test_full4096_and_compact140_use_current_vectorized_oracle(self):
        for compact in (False, True):
            with self.subTest(compact=compact):
                prefix, fresh = 257, 4096
                queries = ([list(range(prefix + fresh - 140, prefix + fresh))] if compact else None)
                fixture = _NativeFixture("cuda", ratio=2, prefixes=(prefix,), lengths=(fresh,), queries=queries)
                if compact:
                    fixture.layout.cp_size = 1
                    fixture.layout.gather_restore_positions = torch.arange(140, device="cuda")
                starts = {0: prefix + fresh - 140} if compact else None
                expected, _ = fixture.build(fused=False, starts=starts)
                hits = []
                operation = graph_cache_kernels.try_assemble_cp_prefill_indices

                def assemble(*args):
                    output = operation(*args)
                    hits.append(output is not None)
                    return output

                with patch.object(graph_cache_kernels, "try_assemble_cp_prefill_indices", new=assemble):
                    actual, workspace = fixture.build(fused=True, starts=starts)
                self.assertEqual(hits, [True], "existing Triton indices must actually admit the layout")
                self.assertTrue(workspace.fused_dequant and workspace.fused_indices)
                self.assertEqual(actual[1].shape[0], 140 if compact else 4096)
                for left, right in zip(actual, expected):
                    assert_bytes(self, left, right)

    @torch.inference_mode()
    def test_encoded_row_dequant_matches_codec_in_strided_bf16_output(self):
        positions = torch.arange(33, device="cuda")
        values = (positions[:, None].float() / 16 + torch.arange(512, device="cuda")[None] % 9 / 8).bfloat16()
        values[0].zero_()
        values[1].fill_(-0.0)
        for global_pool, tag in ((True, "global_kv_2"), (False, "swa_kv")):
            with self.subTest(tag=tag):
                packed = encode_kv(values, pool_tag=tag)
                backing = torch.full((33, 528), 17, device="cuda", dtype=torch.bfloat16)
                output = backing[:, :512]
                expected = torch.empty_like(output)
                decode_kv(packed, pool_tag=tag, out=expected)
                self.assertTrue(_quant.try_dequantize_prefill_rows(packed, output, global_pool=global_pool))
                assert_bytes(self, output, expected)
                assert_bytes(self, backing[:, 512:], torch.full_like(backing[:, 512:], 17))
                padded = torch.zeros((33, packed.shape[1] + 16), device="cuda", dtype=torch.uint8)
                padded[:, :packed.shape[1]].copy_(packed)
                sentinel = output.clone()
                self.assertFalse(_quant.try_dequantize_prefill_rows(
                    padded[:, :packed.shape[1]], output, global_pool=global_pool,
                ))
                assert_bytes(self, output, sentinel)

    @torch.inference_mode()
    def test_actual_profile_contains_existing_fused_kernels(self):
        fixture = _NativeFixture("cuda", ratio=2, prefixes=(129,), lengths=(32,))
        fixture.build(fused=True)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                              torch.profiler.ProfilerActivity.CUDA]) as profile:
            actual, _ = fixture.build(fused=True)
            torch.cuda.synchronize()
        expected, _ = fixture.build(fused=False)
        for left, right in zip(actual, expected):
            assert_bytes(self, left, right)
        artifact_root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        with tempfile.TemporaryDirectory(prefix="local-prefill-fused-") as temporary:
            directory = Path(artifact_root) if artifact_root else Path(temporary)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "local-prefill-fused.trace.json"
            profile.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
        names = [event["name"] for event in events if event.get("cat") == "kernel"]
        for kernel in ("assemble_cp_prefill_indices", "dequantize_k_cache_paged_v41_fp4",
                       "dequantize_k_cache_paged_v41_fp8"):
            self.assertEqual(sum(kernel in name for name in names), 1,
                             f"exactly one existing PPU kernel must execute: {kernel}")


if __name__ == "__main__":
    unittest.main()
