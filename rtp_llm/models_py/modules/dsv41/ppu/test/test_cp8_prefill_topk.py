"""CP8 uses the existing disposable score-tile selector, not a new sorter.

CPU contracts use an explicit native double. Device cases run the unchanged
PPU selector on CP-shaped tiles; they do not prove WORLD8 cache transport.
"""

import os
import unittest
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.cp import CPContext
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_prefill_topk import (
    assert_equivalent,
    reference,
)
from rtp_llm.models_py.modules.dsv41.ppu import attention_layer as attention
from rtp_llm.models_py.modules.dsv41.ppu import prefill_token_topk as token
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.prefill import build_step_context
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_token_topk import (
    _BoundsDouble,
    _finish_double,
    _native_double,
)


class _CudaBranchTensor(torch.Tensor):
    """CPU storage with only the scorer branch predicate doubled."""

    @property
    def is_cuda(self):
        return True


def _state(*, compact=False, rank=0, device="cpu"):
    count, fresh, prefix = 8, 4096, 126976
    keep = 140 if compact else fresh
    chunk = ((keep + 15) // 16) * 2
    half = chunk // 2
    local = torch.cat(
        (
            torch.arange(rank * half, (rank + 1) * half, device=device),
            torch.arange((15 - rank) * half, (16 - rank) * half, device=device),
        )
    )
    valid = local < keep
    position = prefix + fresh - keep + local.clamp_max(keep - 1)
    chunks = (chunk,) * count
    rows = chunk * count
    context = CPContext(
        8,
        rank,
        rows,
        rows * 8,
        count * fresh,
        torch.arange(rows, device=device),
        prefix,
        position.repeat(count),
        valid.repeat(count),
        torch.arange(count * fresh, device=device),
        prefix + fresh,
        NS(),
        req_id_per_token=torch.repeat_interleave(
            torch.arange(count, device=device, dtype=torch.int32), chunk
        ),
        prefix_lengths=torch.full((count,), prefix, device=device),
        chunk_lengths_per_req=chunks,
        kv_cache_sharded=True,
        input_lengths_global_host=(fresh,) * count,
        prefix_lengths_host=(prefix,) * count,
        gather_restore_positions=(
            torch.arange(count * keep, device=device) if compact else None
        ),
    )
    cache = SharedNativeCache.__new__(SharedNativeCache)
    cache._is_prefill, cache.sharded, cache.lengths = (
        True,
        True,
        (prefix + fresh,) * count,
    )
    state = NS(
        prefill=True,
        idle=False,
        compacted=compact,
        context=context,
        cache=cache,
        lengths=(fresh,) * count,
        query_prefixes=(prefix,) * count,
        attn=NS(is_cuda_graph=False),
        runtime=NS(
            policy=NS(
                mode="compact",
                keep_tokens=140,
                window=128,
                max_rollback_tokens=0,
                encoder_replay=False,
            ),
            model=NS(
                _is_speculative=False,
                _is_decode_role=False,
                _prefill_cp_size=8,
                parallelism_config=NS(tp_size=8),
            ),
        ),
    )
    state.use_compact_cp8_ops = lambda value: compact and value is state.context
    return state, attention._indexer_request_slices(chunks, rows, count)


def _score_rows(rows, width, *, device="cpu", dtype=torch.bfloat16):
    # Distinct request/row winners expose a remap or cross-request mix-up.
    scores = torch.full((rows, width), -4, dtype=dtype, device=device)
    values = (
        (torch.arange(512, dtype=torch.int16, device=device) + 0x3F00)
        .view(torch.bfloat16)
        .to(dtype)
    )
    for row in range(rows):
        columns = torch.arange(512, device=device) + row % 19 * 3
        scores[row, columns] = values
    scores[0, :2] = torch.tensor([torch.nan, torch.inf], dtype=dtype, device=device)
    if rows > 1:
        scores[1].fill_(1)  # exact cutoff ties retain the established contract
    return scores


def _run_dense(
    *,
    native,
    rows_per_request=2,
    width=1024,
    publish=False,
    device="cpu",
    dtype=torch.bfloat16,
    selector=None
):
    rows = 8 * rows_per_request
    unit = attention.SharedPPUAttention.__new__(attention.SharedPPUAttention)
    nn.Module.__init__(unit)
    unit.layer_id, unit.kv_source_layer_id, unit.index_source_layer_id = 2, 0, 2
    unit.is_index_source = True
    unit.geometry = NS(
        index_topk=512, index_n_heads=32, index_head_dim=128, ratio=lambda source: 1
    )
    unit.v41_config = dict(
        candidate_source_layer_id=2 if publish else -1,
        candidate_block_size=8,
        candidate_topk_blocks=4,
    )
    positions = torch.arange(rows, device=device) % 73 + width - 100
    requests = torch.repeat_interleave(
        torch.arange(8, dtype=torch.int32, device=device), rows_per_request
    )
    offsets = torch.tensor(
        tuple(range(0, rows + 1, rows_per_request)), dtype=torch.int32, device=device
    )
    context = build_step_context(positions, requests, offsets)
    source = _score_rows(rows, width, device=device, dtype=dtype)
    columns = torch.arange(width, device=device)
    source.masked_fill_(columns[None] >= (positions + 1)[:, None], -torch.inf)
    # Scoring remains a test-only producer. Selection is either Torch or the
    # real wrapper/native provider; every returned logits tile is private.
    seen = []

    def score(query, keys, weights, **kwargs):
        start = sum(value.shape[0] for value in seen)
        tile = source[start : start + query.values.shape[0]].clone()
        seen.append(tile)
        return tile

    unit._project_indexer = lambda *args, **kwargs: (
        torch.zeros(rows, 32, 128, device=device),
        torch.ones(rows, 32, device=device),
    )
    unit.ops = NS(index_score=score)
    cache = NS(
        lengths=(width,) * 8,
        read_source=lambda *args, global_entries: (
            None,
            torch.zeros(global_entries, 68, dtype=torch.uint8, device=device),
        ),
    )
    shared = {}
    x = torch.zeros(rows, 8, device=device)
    if device == "cpu":
        x = x.as_subclass(_CudaBranchTensor)
    with ExitStack() as stack:
        stack.enter_context(
            patch.dict(os.environ, {attention.PREFILL_DENSE_FP4_SCORE_ENV: "1"})
        )
        stack.enter_context(
            patch.object(attention, "has_fp8_fp4_mqa_logits", return_value=True)
        )
        stack.enter_context(
            patch.object(
                attention,
                "_quantized_indexer_query",
                side_effect=lambda q: NS(values=q, scales=None),
            )
        )
        if selector is not None:
            stack.enter_context(
                patch.object(
                    token,
                    "try_select_tokens",
                    side_effect=lambda *args, **kwargs: selector(
                        shared, *args, **kwargs
                    ),
                )
            )
        output = unit._select(
            x,
            torch.zeros(rows, 8, device=device),
            context,
            cache,
            shared,
            prefill=True,
            request_slices=attention._indexer_request_slices(
                (rows_per_request,) * 8, rows, 8
            ),
            yoco_small_ops=True,
            cp_token_topk=native,
        )
    return output, source, positions + 1, shared, seen


def _run_sparse(*, native, rows_per_request=2, device="cpu"):
    rows, width, count = rows_per_request * 8, 16384, 32768
    unit = attention.SharedPPUAttention.__new__(attention.SharedPPUAttention)
    nn.Module.__init__(unit)
    unit.layer_id, unit.kv_source_layer_id, unit.index_source_layer_id = 24, 20, 24
    unit.is_index_source = True
    unit.geometry = NS(
        index_topk=512, index_n_heads=32, index_head_dim=128, ratio=lambda source: 1
    )
    unit.v41_config = dict(
        candidate_source_layer_id=20, candidate_block_size=8, candidate_topk_blocks=2048
    )
    positions = torch.arange(rows, device=device) % 73 + count - 100
    requests = torch.repeat_interleave(
        torch.arange(8, dtype=torch.int32, device=device), rows_per_request
    )
    offsets = torch.tensor(
        tuple(range(0, rows + 1, rows_per_request)), dtype=torch.int32, device=device
    )
    context = build_step_context(positions, requests, offsets)
    candidates = torch.arange(2048, device=device, dtype=torch.int32).repeat(rows, 1)
    candidates.add_(requests[:, None] * 64)
    shared = {"candidates": candidates}
    source = _score_rows(rows, width, device=device)
    plans = []
    scored = []

    def plan(shared, key, candidate_rows, visible, key_count, block_size, **kwargs):
        size = candidate_rows.shape[0]
        ends = torch.full((size,), width, dtype=torch.int32, device=device)
        # Include a short sparse tail and an empty padded score row in each
        # request. Both selectors retain the same existing score-row mask.
        ends[0] = 513
        if size > 1:
            ends[-1] = 0
        result = attention.sparse_prefill_indexer.SparsePrefillPlan(
            row_ks=torch.zeros(size, dtype=torch.int32, device=device),
            row_ke=visible.clamp(0, key_count).int().contiguous(),
            sparse_indices=candidate_rows.contiguous(),
            end=ends,
            metadata=torch.empty(0, dtype=torch.uint8, device=device),
            key_count=key_count,
            block_size=block_size,
        )
        plans.append(result)
        return result

    def score(q, scales, keys, weights, plan):
        start = sum(value.shape[0] for value in scored)
        tile = source[start : start + plan.rows].clone()
        scored.append(tile)
        return tile

    def remap(columns, plan, *, logits=None, out=None):
        valid = (columns >= 0) & (columns < plan.end[:, None])
        if logits is not None:
            valid &= logits.gather(1, columns.long().clamp(0, width - 1)).isfinite()
        result = torch.where(valid, columns + plan.sparse_indices[:, :1] * 8, -1).int()
        if out is not None:
            out.copy_(result)
            return out
        return result

    unit._project_indexer = lambda *args, **kwargs: (
        torch.zeros(rows, 32, 128, device=device),
        torch.ones(rows, 32, device=device),
    )
    cache = NS(
        lengths=(count,) * 8,
        read_source=lambda *args, global_entries: (
            None,
            torch.zeros(global_entries, 68, dtype=torch.uint8, device=device),
        ),
    )
    x = torch.zeros(rows, 8, device=device)
    if device == "cpu":
        x = x.as_subclass(_CudaBranchTensor)
    with ExitStack() as stack:
        stack.enter_context(
            patch.dict(os.environ, {attention.PREFILL_SPARSE_FP4_SCORE_ENV: "1"})
        )
        stack.enter_context(patch.object(attention, "is_ppu", return_value=True))
        stack.enter_context(
            patch.object(
                attention.sparse_prefill_indexer, "_get_deep_gemm", return_value=NS()
            )
        )
        stack.enter_context(
            patch.object(attention, "_prefill_sparse_plan", side_effect=plan)
        )
        stack.enter_context(
            patch.object(attention.sparse_prefill_indexer, "score", side_effect=score)
        )
        stack.enter_context(
            patch.object(
                attention,
                "_quantized_indexer_query",
                side_effect=lambda q: NS(values=q, scales=None),
            )
        )
        if device == "cpu":
            stack.enter_context(
                patch.object(
                    attention.sparse_prefill_indexer, "remap", side_effect=remap
                )
            )
        output = unit._select(
            x,
            torch.zeros(rows, 8, device=device),
            context,
            cache,
            shared,
            prefill=True,
            request_slices=attention._indexer_request_slices(
                (rows_per_request,) * 8, rows, 8
            ),
            yoco_small_ops=True,
            cp_token_topk=native,
        )
    # In this fixture each request uses a distinct contiguous block range;
    # invert the actual native remap to validate the exact old score contract.
    columns = torch.where(output >= 0, output - requests[:, None] * 512, -1).int()
    return columns, source, torch.cat([p.end for p in plans]), shared


class CP8PrefillTopKContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(patch.dict(os.environ, {token.ENV: "1"}))
        self.stack.enter_context(patch.object(attention, "is_ppu", return_value=True))
        self.stack.enter_context(
            patch("torch.cuda.is_current_stream_capturing", return_value=False)
        )

    def _native_double(self):
        self.stack.enter_context(
            patch.object(token, "_device_supported", return_value=True)
        )
        self.stack.enter_context(
            patch.object(token, "_native_ops", return_value=(_native_double, None))
        )
        self.stack.enter_context(
            patch.object(token, "_hierarchical_ops", return_value=None)
        )
        self.stack.enter_context(
            patch.object(token._topk, "_prefill_topk_bounds_kernel", _BoundsDouble())
        )
        self.stack.enter_context(
            patch.object(token._topk, "finish_tokens", side_effect=_finish_double)
        )

    def test_full_and_compact_all_ranks_keep_request_and_padding_contract(self):
        guard = attention._cp8_prefill_token_topk_enabled
        for compact in (False, True):
            for rank in range(8):
                with self.subTest(compact=compact, rank=rank):
                    state, slices = _state(compact=compact, rank=rank)
                    before = (
                        state.context.global_positions.clone(),
                        state.context.local_is_real.clone(),
                    )
                    self.assertTrue(
                        guard(state, state.context, slices, state.context.chunk_length)
                    )
                    self.assertTrue(
                        torch.equal(before[0], state.context.global_positions)
                    )
                    self.assertTrue(torch.equal(before[1], state.context.local_is_real))
        state, slices = _state(compact=True)
        self.assertFalse(state.context.local_is_real.all())
        self.assertTrue(guard(state, state.context, slices, 144))

    def test_cp_guard_declines_unbound_layouts_and_other_phases(self):
        guard = attention._cp8_prefill_token_topk_enabled
        mutations = (
            lambda s: setattr(s, "idle", True),
            lambda s: setattr(s, "prefill", False),
            lambda s: setattr(s, "verify", True),
            lambda s: setattr(s, "proposal", True),
            lambda s: setattr(s, "is_graph", True),
            lambda s: setattr(s, "is_graph_commit", True),
            lambda s: setattr(s.attn, "is_cuda_graph", True),
            lambda s: setattr(s.cache, "sharded", False),
            lambda s: setattr(s.cache, "_is_prefill", False),
            lambda s: setattr(s.runtime.model, "_is_decode_role", True),
            lambda s: setattr(s.runtime.model, "_is_speculative", True),
            lambda s: setattr(s.runtime.model.parallelism_config, "tp_size", 1),
            lambda s: setattr(s.runtime.model, "_prefill_cp_size", 1),
            lambda s: setattr(s.runtime.policy, "keep_tokens", 128),
            lambda s: setattr(s.runtime.policy, "max_rollback_tokens", 1),
            lambda s: setattr(s.runtime.policy, "encoder_replay", True),
            lambda s: setattr(s.context, "kv_cache_sharded", False),
            lambda s: setattr(s.context, "cp_size", 1),
            lambda s: setattr(s.context, "cp_rank", 8),
            lambda s: setattr(s.context, "padded_seq_len", 32767),
            lambda s: setattr(s.context, "chunk_lengths_per_req", (4096,)),
            lambda s: setattr(s.context, "prefix_lengths_host", (0,) * 8),
            lambda s: setattr(s.cache, "lengths", (131071,) * 8),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                state, slices = _state()
                mutate(state)
                self.assertFalse(guard(state, state.context, slices, 4096))
        state, slices = _state()
        self.assertFalse(guard(state, replace(state.context), slices, 4096))
        self.assertFalse(guard(state, state.context, None, 4096))
        self.assertFalse(guard(state, state.context, slices[::-1], 4096))
        self.assertFalse(guard(state, state.context, slices, 4095))
        self.assertFalse(guard(state, state.context, slices, 4096, diagnostic={}))
        self.assertFalse(
            guard(state, state.context, slices, 4096, packed_plan=object())
        )
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            self.assertFalse(guard(state, state.context, slices, 4096))

    def test_batched_request_tiles_preserve_finite_and_tie_contract(self):
        self._native_double()
        actual, original, visible, shared, _ = _run_dense(native=True)
        assert_equivalent(self, actual, reference(original, visible), original, visible)
        self.assertEqual(shared["cp_token_topk_calls"], [(2, "dense", 16, 1024)])
        self.assertEqual(int((actual[0] >= 0).sum()), 510)
        self.assertFalse(torch.equal(actual[2], actual[4]))

    def test_publication_precedes_destructive_native_selection(self):
        selected_rows = 0

        def select(shared, logits, visible, **kwargs):
            nonlocal selected_rows
            self.assertTrue(kwargs["disposable"])
            rows = slice(selected_rows, selected_rows + logits.shape[0])
            # The bounded slab being selected is already published; later
            # request slabs may still contain their initial -1 sentinels.
            self.assertTrue(
                torch.equal(shared["candidates"][rows], old_shared["candidates"][rows])
            )
            selected_rows = rows.stop
            output = reference(logits, visible)
            logits.fill_(-torch.inf)
            return output

        control, _, _, old_shared, _ = _run_dense(native=False, publish=True)
        actual, _, _, shared, _ = _run_dense(native=True, publish=True, selector=select)
        self.assertEqual(selected_rows, actual.shape[0])
        self.assertTrue(torch.equal(shared["candidates"], old_shared["candidates"]))
        self.assertTrue(torch.equal(actual.sort(-1).values, control.sort(-1).values))

    def test_unsupported_dtype_falls_back_before_native_mutation(self):
        with patch.object(
            token, "_native_ops", side_effect=AssertionError("FP32 native launch")
        ):
            actual, original, visible, shared, seen = _run_dense(
                native=True, dtype=torch.float32
            )
        self.assertNotIn("cp_token_topk_calls", shared)
        assert_equivalent(self, actual, reference(original, visible), original, visible)
        self.assertTrue(
            torch.equal(torch.cat(seen).view(torch.uint8), original.view(torch.uint8))
        )

    def test_sparse_request_remap_short_and_padding_rows_keep_the_original_contract(
        self,
    ):
        self._native_double()
        control, original, visible, old_shared = _run_sparse(native=False)
        actual, _, _, shared = _run_sparse(native=True)
        expected = reference(original, visible)
        assert_equivalent(self, control, expected, original, visible)
        assert_equivalent(self, actual, expected, original, visible)
        self.assertNotIn("cp_token_topk_calls", old_shared)
        self.assertEqual(shared["cp_token_topk_calls"], [(24, "sparse", 2, 16384)] * 8)
        self.assertTrue((actual[1::2] == -1).all())


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1",
    "explicit PPU device test",
)
class CP8PrefillTopKDevice(unittest.TestCase):
    @torch.inference_mode()
    def test_actual_native_selector_full_and_compact_cp_shapes(self):
        self.assertTrue(torch.cuda.is_available())
        with patch.dict(os.environ, {token.ENV: "1"}):
            for rows, width in ((512, 65536), (18, 131072)):
                with self.subTest(rows_per_request=rows, width=width):
                    actual, original, visible, shared, _ = _run_dense(
                        native=True,
                        rows_per_request=rows,
                        width=width,
                        device="cuda",
                    )
                    self.assertTrue(shared["cp_token_topk_calls"])
                    assert_equivalent(
                        self, actual, reference(original, visible), original, visible
                    )

    @torch.inference_mode()
    def test_actual_native_selector_and_remap_keep_eight_request_domains(self):
        self.assertTrue(torch.cuda.is_available())
        with patch.dict(os.environ, {token.ENV: "1"}):
            actual, original, visible, shared = _run_sparse(
                native=True,
                rows_per_request=18,
                device="cuda",
            )
        self.assertEqual(shared["cp_token_topk_calls"], [(24, "sparse", 18, 16384)] * 8)
        assert_equivalent(self, actual, reference(original, visible), original, visible)


if __name__ == "__main__":
    unittest.main()
