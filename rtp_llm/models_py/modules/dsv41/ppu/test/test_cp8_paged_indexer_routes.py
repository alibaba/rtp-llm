"""Actual _select CP paged routing with independent CPU scorer boundaries.

The real CPIndexerRows publication/slicing, dense candidate publication,
Torch fallback and RTP sparse remap wrapper run here. Only native scoring,
query quantization, disposable token selection and the remap JIT kernel are
CPU doubles. These tests do not qualify native scoring or WORLD8 transport.
"""

from contextlib import ExitStack
from itertools import accumulate
import math
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_topk as topk_native
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sparse_prefill_indexer as sparse
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_prefill_topk import assert_equivalent, reference
from rtp_llm.models_py.modules.dsv41.ppu import attention_layer as attention
from rtp_llm.models_py.modules.dsv41.ppu import cp_prefill_paged_indexer as paged
from rtp_llm.models_py.modules.dsv41.ppu.prefill import build_step_context
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_l20_query_projection import _context
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_paged_indexer import _fixture as _bank_fixture
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_paged_indexer import UPLOAD, _cpu_upload
from rtp_llm.models_py.modules.dsv41.prefill.ced import _bounded_replay_selected, _host_query_layout


class _CudaBranchTensor(torch.Tensor):
    """CPU storage; only the native scorer/remap admission predicate is doubled."""

    @property
    def is_cuda(self):
        return True


def _cuda_branch(value):
    return value if isinstance(value, _CudaBranchTensor) else value.as_subclass(_CudaBranchTensor)


def _fixture(*, compact, count=2304, prefix=256, ratio=2):
    # Retain the actual4096-fresh-row CP8 geometry while bounding CPU K width
    # through a shorter cached prefix. Compact queries use the actual140 plan.
    fresh = (count - prefix) * ratio
    if fresh != 4096:
        raise AssertionError("route fixture must retain actual4096-fresh-token requests")
    original, _ = _context(0, (fresh,) * 8, (prefix * ratio,) * 8)
    if compact:
        selected = _bounded_replay_selected((fresh,) * 8, (fresh - 140,) * 8)
        context, _, _ = _host_query_layout(original, selected, None, keep_candidate_rows=True)
    else:
        context = original
    state, bank, _ = _bank_fixture((count,) * 8, prefix, context.chunk_lengths_per_req, ratio=ratio,
                                   source=20 if compact else 0)
    state.context, state.compacted = context, compact
    state.use_compact_cp8_ops = lambda value: compact and value is state.context
    slices = attention._indexer_request_slices(context.chunk_lengths_per_req, context.chunk_length, 8)
    rows = paged.prepare_rows(state, bank, request_slices=slices, positions=context.global_positions,
                              request_ids=context.req_id_per_token, ratio=ratio)
    if rows is None:
        raise AssertionError("real CP paged row publication was not admitted")
    for name in ("read_indexer_fp4", "read_source", "descriptors"):
        setattr(state.cache, name, Mock(side_effect=AssertionError(f"paged route must not call {name}")))
    size = context.chunk_length
    step = build_step_context(context.global_positions, context.req_id_per_token,
                              torch.tensor((0, *accumulate(context.chunk_lengths_per_req)), dtype=torch.int32))
    return NS(state=state, bank=bank, rows=rows, context=step, slices=slices,
              query=torch.arange(size, dtype=torch.float32).view(size, 1, 1),
              weights=torch.ones(size, 32), x=_cuda_branch(torch.zeros(size, 8)), size=size, ratio=ratio)


def _scores(fixture, width, *, sparse_columns=False):
    rows = fixture.size
    ids = torch.arange(rows)
    requests = fixture.state.context.req_id_per_token.long()
    # A request-specific winner range catches cross-request table/row swaps.
    logits = torch.full((rows, width), -4, dtype=torch.bfloat16)
    winners = (torch.arange(512)[None] + requests[:, None] * 67 + ids[:, None] % 23 * 3) % width
    values = (torch.arange(512, dtype=torch.int16) + 0x3F00).view(torch.bfloat16).expand(rows, -1)
    logits.scatter_(1, winners, values)
    logits[0, :2] = torch.tensor((torch.nan, torch.inf), dtype=logits.dtype)
    logits[1].fill_(1)  # cutoff ties keep the established exact-value contract
    logits[2].fill_(-torch.inf)
    if not sparse_columns:
        visible = fixture.rows.visible
        logits.masked_fill_(torch.arange(width)[None] >= visible[:, None], -torch.inf)
    return logits


def _candidate_reference(logits, visible, block, topk):
    # Independent per-request max-pool/newest-block publication authority.
    width = logits.shape[1]
    blocks = (width + block - 1) // block
    padded = torch.full((logits.shape[0], blocks * block), -torch.inf, dtype=logits.dtype)
    padded[:, :width] = logits
    pooled = padded.reshape(logits.shape[0], blocks, block).amax(-1).float()
    newest = ((visible - 1).clamp_min(0) // block).long()
    pooled[torch.arange(logits.shape[0]), newest] = torch.where(visible > 0, torch.inf, -torch.inf)
    scores, chosen = pooled.topk(min(topk, blocks), dim=-1)
    return torch.where(scores > -torch.inf, chosen, -1).int()


def _plan(candidates, visible, count, block):
    indices = torch.zeros_like(candidates)
    ends = torch.zeros(candidates.shape[0], dtype=torch.int32)
    for row in range(candidates.shape[0]):
        valid = candidates[row][(candidates[row] >= 0) & (candidates[row] * block < visible[row].clamp_max(count))]
        unique = valid.unique(sorted=True)
        if unique.numel():
            indices[row].fill_(int(unique[-1]))
            indices[row, :unique.numel()] = unique
            ends[row] = unique.numel() * block
    return sparse.SparsePrefillPlan(torch.zeros_like(ends), visible.clamp(0, count).int(), indices, ends,
                                    torch.empty(0, dtype=torch.uint8), count, block)


def _remap_reference(columns, plan, logits):
    safe = columns.long().clamp(0, plan.sparse_columns - 1)
    blocks = plan.sparse_indices.gather(1, safe // plan.block_size)
    logical = blocks * plan.block_size + safe % plan.block_size
    valid = ((columns >= 0) & (columns < plan.end[:, None]) & (columns < plan.sparse_columns)
             & (logical >= 0) & (logical < plan.row_ke[:, None])
             & logits.gather(1, safe).isfinite())
    return torch.where(valid, logical, -1).int()


def _legacy_request_sparse_reference(logits, plan):
    """Independent scalar RTP bounds/finite rule for one legacy request tile."""
    columns = logits.topk(512, dim=-1).indices.tolist()
    result = []
    for row, selected in enumerate(columns):
        mapped = []
        for column in selected:
            if not 0 <= column < min(int(plan.end[row]), plan.sparse_columns):
                mapped.append(-1)
                continue
            position = int(plan.sparse_indices[row, column // 8]) * 8 + column % 8
            mapped.append(position if 0 <= position < int(plan.row_ke[row])
                          and math.isfinite(float(logits[row, column])) else -1)
        result.append(mapped)
    return torch.tensor(result, dtype=torch.int32)


class _RemapKernelDouble:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(columns, indices, end, visible, logits, output, *metadata, **kwargs):
            self.calls.append((columns.shape[0], columns.shape[1]))
            block = metadata[6]
            plan = sparse.SparsePrefillPlan(torch.zeros_like(end), visible, indices, end,
                                            torch.empty(0, dtype=torch.uint8), int(visible.max()), block)
            output.copy_(_remap_reference(columns, plan, logits))
        return launch


class CP8PagedIndexerSelectContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(patch.dict(os.environ, {
            attention.PREFILL_DENSE_FP4_SCORE_ENV: "1", attention.PREFILL_SPARSE_FP4_SCORE_ENV: "1",
            attention.CP8_INDEXER_TILE_PREP_ENV: "0",
        }))
        self.stack.enter_context(patch.object(paged, "_device_supported", return_value=True))
        self.stack.enter_context(patch(UPLOAD, side_effect=_cpu_upload))
        self.stack.enter_context(patch.object(paged, "_deep_gemm", return_value=NS()))
        self.stack.enter_context(patch("rtp_llm.models_py.modules.dsv41.ppu.layers.cp8_yoco_operator_eligible",
                                       return_value=True))
        self.stack.enter_context(patch.object(attention, "is_ppu", return_value=True))
        self.stack.enter_context(patch.object(attention, "has_fp8_fp4_mqa_logits", return_value=True))
        self.stack.enter_context(patch.object(sparse, "_get_deep_gemm", return_value=NS()))
        self.stack.enter_context(patch.object(attention, "_quantized_indexer_query",
                                               side_effect=lambda value: NS(values=value, scales=None)))
        self.stack.enter_context(patch.object(attention, "write_torch_topk_results", return_value=False))
        self.stack.enter_context(patch.object(topk_native, "write_torch_topk_results", return_value=False))
        # Keep the actual1024-row early/144-row late grouping while bounding
        # CPU scorer widths. This is the score-budget boundary, not a sorter.
        self.stack.enter_context(patch.object(attention, "logits_chunk_rows", return_value=1024))
        self.kernel = _RemapKernelDouble()
        self.stack.enter_context(patch.object(sparse, "_remap_sparse_prefill_topk_kernel", self.kernel))

    def _run(self, fixture, layer, source_logits, *, shared=None, mode="torch", candidate_k=2048):
        shared = {} if shared is None else shared
        unit = attention.SharedPPUAttention.__new__(attention.SharedPPUAttention)
        nn.Module.__init__(unit)
        unit.layer_id, unit.kv_source_layer_id, unit.index_source_layer_id = layer, 20 if layer >= 20 else 0, layer
        self.assertEqual(unit.kv_source_layer_id, fixture.bank.source)
        unit.is_index_source = True
        unit.geometry = NS(index_topk=512, index_n_heads=32, index_head_dim=128,
                           ratio=lambda source: fixture.ratio)
        unit.v41_config = dict(candidate_source_layer_id=20, candidate_block_size=8,
                               candidate_topk_blocks=candidate_k)
        unit._project_indexer = Mock(side_effect=AssertionError("explicit projection inputs must be consumed"))
        unit.ops = NS(index_score=Mock(side_effect=AssertionError("paged route must not unpack K")))
        calls, plans, selection = [], [], []
        active = {}

        def check_rows(query, rows, kind):
            ids = query[:, 0, 0].long()
            expected_ids = fixture.state.context.req_id_per_token.index_select(0, ids)
            self.assertTrue(torch.equal(rows.request_ids, expected_ids))
            self.assertEqual(rows.request_ids_host, tuple(expected_ids.tolist()))
            self.assertTrue(torch.equal(rows.tables, fixture.bank.request_table.index_select(0, expected_ids.long())))
            self.assertTrue(torch.equal(rows.visible, fixture.rows.visible.index_select(0, ids)))
            calls.append((kind, ids.clone(), tuple(expected_ids.unique().tolist())))
            active["ids"] = ids
            return ids

        def dense_score(query, scales, weights, rows):
            ids = check_rows(query, rows, "dense")
            return _cuda_branch(source_logits.index_select(0, ids).clone())

        def make_plan(current_shared, key, candidates, visible, count, block, **kwargs):
            result = _plan(candidates, visible, count, block)
            plans.append((key, result))
            return result

        def sparse_score(query, scales, weights, rows, plan):
            ids = check_rows(query, rows, "sparse")
            self.assertEqual(plan.rows, rows.count)
            self.assertTrue(torch.equal(plan.row_ke, rows.visible))
            return _cuda_branch(source_logits.index_select(0, ids).clone())

        def select(logits, visible, **kwargs):
            self.assertEqual(kwargs, {"contract": "local", "disposable": True})
            selection.append((active["ids"].clone(), visible.clone()))
            if mode == "decline":
                return None
            ids = active["ids"]
            if layer == 20 and shared.get("candidates") is not None:
                original = source_logits.index_select(0, ids)
                expected = _candidate_reference(original, visible, 8, candidate_k)
                self.assertTrue(torch.equal(shared["candidates"].index_select(0, ids), expected),
                                "candidate publication must precede destructive token selection")
            chosen = reference(logits, visible).flip(-1)
            logits.fill_(-17)  # private disposable tiles may be destroyed now
            return _cuda_branch(chosen)

        before = (fixture.rows.tables.clone(), fixture.rows.visible.clone(), source_logits.view(torch.uint8).clone())
        with patch.object(paged, "score_dense", side_effect=dense_score), patch.object(
            paged, "score_sparse", side_effect=sparse_score
        ), patch.object(attention, "_prefill_sparse_plan", side_effect=make_plan), patch.object(
            attention, "select_tokens", side_effect=select
        ):
            output = unit._select(fixture.x, torch.zeros(fixture.size, 8), fixture.context, fixture.state.cache,
                                  shared, prefill=True, request_slices=fixture.slices, yoco_small_ops=True,
                                  allow_tile_prep=False, indexer_inputs=(fixture.query, fixture.weights),
                                  cp_token_topk=mode != "torch", cp_indexer_rows=fixture.rows)
        self.assertTrue(torch.equal(fixture.rows.tables, before[0]))
        self.assertTrue(torch.equal(fixture.rows.visible, before[1]))
        self.assertTrue(torch.equal(source_logits.view(torch.uint8), before[2]))
        for name in ("read_indexer_fp4", "read_source", "descriptors"):
            getattr(fixture.state.cache, name).assert_not_called()
        unit.ops.index_score.assert_not_called()
        self.assertEqual(len(selection), 0 if mode == "torch" else len(calls))
        return output, shared, calls, plans

    def test_body_handshake_uses_native_int32_owner_after_int64_row_consumer_cast(self):
        from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_split_l20 import _component

        block, state, _, _, _ = _component("cpu", 512)
        cp, _ = _context(0, (512,) * 8, (126976,) * 8)
        state.original_context = state.context = cp
        state.lengths, state.query_prefixes = (512,) * 8, (126976,) * 8
        state.cache.sharded = True
        state.cache.lengths = (126976 + 512,) * 8
        state.cache.enable_shared_indexer = Mock(side_effect=AssertionError("handshake must stop before publication"))
        block.attn._cp_ctx = cp
        block.attn.geometry.index_n_heads = 32
        block.attn.geometry.index_head_dim = 128
        block.attn.geometry.index_topk = 512
        block.attn.project_query_kv = Mock(side_effect=AssertionError("handshake must precede projection"))
        block.attn._get_mega_attention = Mock(side_effect=AssertionError("handshake must precede Mega preparation"))
        rows, consumers, calls = [], [], []
        original_rows, original_builder = state.rows, attention.build_step_context

        def read_rows(context):
            value = original_rows(context)
            rows.append(value)
            self.assertEqual(value[1].dtype, torch.int64)
            self.assertIsNot(value[1], cp.req_id_per_token)
            self.assertFalse(torch._C._is_alias_of(value[1], cp.req_id_per_token))
            return value

        def build_context(positions, request_ids, cu, *args, **kwargs):
            self.assertEqual(request_ids.dtype, torch.int64)
            self.assertIs(request_ids, rows[-1][1])
            result = original_builder(positions, request_ids, cu, *args, **kwargs)
            self.assertEqual(result.request_ids.dtype, torch.int64)
            consumers.append(result)
            return result

        class HandshakeReached(RuntimeError):
            pass

        def handshake(actual_state, *, request_slices, positions, request_ids):
            calls.append(request_ids)
            self.assertIs(actual_state, state)
            self.assertIs(positions, cp.global_positions)
            self.assertIs(request_ids, cp.req_id_per_token)
            self.assertEqual(request_ids.dtype, torch.int32)
            self.assertIsNot(request_ids, rows[-1][1], "derived int64 rows cannot certify the native owner")
            self.assertEqual(request_slices, tuple(slice(start, start + 64) for start in range(0, 512, 64)))
            raise HandshakeReached("native CP paged handshake reached before projection")

        state.rows = read_rows
        with patch.dict(os.environ, {"RTP_V41_PPU_CP8_REUSE_LOCAL_ROWS": "0"}), patch.object(
            attention, "optimized_diagnostics_enabled", return_value=False
        ), patch.object(attention, "build_step_context", side_effect=build_context), patch.object(
            paged, "can_prepare_rows", side_effect=handshake
        ):
            os.environ.pop(attention.CP8_INDEXER_REQUEST_SLICES_ENV, None)
            with self.assertRaisesRegex(HandshakeReached, "before projection"):
                block.attn._body(torch.zeros(512, block.attn.geometry.hidden_size, dtype=torch.bfloat16))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(consumers), 1)
        self.assertEqual(cp.req_id_per_token.dtype, torch.int32)
        self.assertTrue(torch.equal(consumers[0].request_ids, cp.req_id_per_token.long()))
        state.cache.enable_shared_indexer.assert_not_called()
        block.attn.project_query_kv.assert_not_called()
        block.attn._get_mega_attention.assert_not_called()

    def test_actual_full_and_compact_shapes_use13_dense_and4_sparse_tiles(self):
        full = _fixture(compact=False)
        source = _scores(full, full.rows.width)
        shared, dense_calls = {}, []
        for layer in (2, 8, 14):
            _, shared, calls, _ = self._run(full, layer, source, shared=shared)
            dense_calls.extend(calls)
        late = _fixture(compact=True, count=32768, prefix=28672, ratio=1)
        late_scores = _scores(late, late.rows.width)
        _, shared, calls, _ = self._run(late, 20, late_scores, shared=shared)
        dense_calls.extend(calls)
        sparse_scores = _scores(late, 16384, sparse_columns=True)
        sparse_calls = []
        for layer in (24, 28, 32, 36):
            _, shared, calls, plans = self._run(late, layer, sparse_scores, shared=shared)
            sparse_calls.extend(calls)
            self.assertEqual(len(plans), 1)
            self.assertEqual(plans[0][0][:3], ("cp_paged", 0, 144))
        self.assertEqual([ids.numel() for _, ids, _ in dense_calls], [1024] * 12 + [144])
        self.assertEqual([ids.numel() for _, ids, _ in sparse_calls], [144] * 4)
        self.assertEqual(len(shared["cp_paged_indexer_calls"]), 17)
        self.assertTrue(all(len(requests) == 2 for _, _, requests in dense_calls[:-1]))
        self.assertTrue(all(requests == tuple(range(8)) for _, _, requests in sparse_calls))
        self.assertEqual(len(self.kernel.calls), 4, "one direct RTP remap per sparse tile")

    def test_dense_publication_finite_ties_and_selector_fallback_match_per_request_legacy(self):
        fixture = _fixture(compact=True)
        source = _scores(fixture, fixture.rows.width)
        expected = torch.cat([reference(source[rows], fixture.rows.visible[rows]) for rows in fixture.slices])
        candidates = torch.cat([_candidate_reference(source[rows], fixture.rows.visible[rows], 8, 16)
                                for rows in fixture.slices])
        for mode in ("torch", "decline", "native"):
            with self.subTest(mode=mode):
                output, shared, calls, _ = self._run(fixture, 20, source, mode=mode, candidate_k=16)
                assert_equivalent(self, output, expected, source, fixture.rows.visible)
                self.assertTrue(torch.equal(shared["candidates"], candidates))
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][2], tuple(range(8)))
                self.assertEqual(len(shared.get("cp_token_topk_calls", [])), int(mode == "native"))
        self.assertEqual(int(fixture.state.context.local_is_real.sum()), 112)
        self.assertEqual(fixture.size, 144)

    def test_global_sparse_plan_matches_request_local_remap_padding_and_nonfinite(self):
        fixture = _fixture(compact=True, count=32768, prefix=28672, ratio=1)
        source = _scores(fixture, 16384, sparse_columns=True)
        request_ids = fixture.state.context.req_id_per_token
        candidates = torch.arange(2048, dtype=torch.int32)[None].repeat(144, 1)
        candidates += request_ids[:, None] * 64
        candidates[~fixture.state.context.local_is_real] = -1
        candidates[3, 1:] = candidates[3, 0]  # duplicate compaction leaves a short8-token sparse prefix
        expected_rows = []
        for rows in fixture.slices:
            plan = _plan(candidates[rows], fixture.rows.visible[rows], 32768, 8)
            logits = source[rows].clone()
            logits.masked_fill_(torch.arange(16384)[None] >= plan.end[:, None], -torch.inf)
            expected_rows.append(_legacy_request_sparse_reference(logits, plan))
        expected = torch.cat(expected_rows)
        for mode in ("torch", "decline", "native"):
            with self.subTest(mode=mode):
                before = len(self.kernel.calls)
                output, shared, calls, plans = self._run(fixture, 24, source,
                    shared={"candidates": candidates.clone()}, mode=mode)
                self.assertTrue(torch.equal(output.sort(-1).values, expected.sort(-1).values))
                self.assertTrue((output[~fixture.state.context.local_is_real] == -1).all())
                self.assertTrue((output[2] == -1).all(), "nonfinite score rows must remain empty")
                self.assertLessEqual(int((output[3] >= 0).sum()), 8)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][2], tuple(range(8)))
                self.assertEqual(plans[0][1].rows, 144)
                self.assertEqual(len(self.kernel.calls) - before, 1)
                self.assertIs(shared["topk"][24], output)


if __name__ == "__main__":
    unittest.main()
