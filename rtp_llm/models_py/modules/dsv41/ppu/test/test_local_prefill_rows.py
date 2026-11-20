"""Per-forward local Prefill metadata, with the legacy path as a value oracle.

Native input/CP metadata is read-only during one forward. Ordinary tensor
versions additionally protect changed fixtures; inference-mode buffers may be
updated between forwards, each of which owns a fresh metadata cache.
"""

import os
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from itertools import accumulate
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.cp import CPContext
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import (
    LOCAL_PREFILL_FULL_ROWS_ENV,
    LOCAL_PREFILL_ROWS_ENV,
    SharedForwardState,
)


ENV = LOCAL_PREFILL_ROWS_ENV
LOOKUP_ENV = {
    ENV: "1",
    "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE": "1",
    "RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA": "1",
}


@contextmanager
def public_ced_cpu_scope(state):
    """Mock native admission only; execute the real decorated create/layout."""
    from rtp_llm.models_py.modules.dsv41.prefill import ced
    from rtp_llm.models_py.modules.dsv41.fp8 import _kv_cache_utils

    tag = ced.DECODER_SWA_KV
    base = torch.empty((1, 140 * 528), dtype=torch.uint8)
    state.cache.native = NS(
        group_tags=(tag,), get_seq_size_per_block=lambda _: 140,
        get_layer_cache=lambda *args: NS(kv_cache_base=base),
    )
    state.attn.is_prefill = True
    state.inputs.need_all_logits = state.inputs.need_all_hidden_states = False
    state.v4.capture_aux_hidden_layer_ids = ()
    with ExitStack() as stack:
        for replacement in (
            patch.object(torch.Tensor, "is_cuda", new=property(lambda _: True)),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU contract must not initialize GPU")),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(ced.collective_torch, "_get_group", return_value=None),
            patch.object(ced, "_supported_model", return_value=True),
            patch.object(ced, "as_attention_inputs_by_tag", return_value={
                str(tag): NS(kv_cache_block_id=torch.ones((len(state.lengths), 1), dtype=torch.int32)),
            }),
            patch.object(_kv_cache_utils, "pool_entry_count", return_value=140),
        ):
            stack.enter_context(replacement)
        yield ced


def state_for(device, prefixes=(17, 99, 64), lengths=(3, 0, 5), *, inputs=None, cu=None):
    if inputs is None:
        inputs = NS(input_ids=torch.arange(sum(lengths), device=device, dtype=torch.int64))
    if cu is None:
        cu = torch.tensor((0, *accumulate(lengths)), dtype=torch.int32)
    runtime = NS(model=NS(v4=NS(layers=[])), policy=NS(mode="compact"))
    attn = NS(cu_seqlens=cu, is_cuda_graph=False)
    return SharedForwardState(
        runtime, inputs, attn, NS(sharded=False), prefill=True, idle=False,
        prefixes=prefixes, lengths=lengths,
    )


def cp1_context(state, *, compact=False):
    device = state.inputs.input_ids.device
    if compact:
        # Requests (4,3) retain rows (1,3) and (4), plus one padded copy.
        positions = (18, 20, 64, 64)
        requests = (0, 0, 1, 1)
        real = (True, True, True, False)
        chunks = (2, 2)
        selected = torch.tensor((1, 3, 4), dtype=torch.int64, device=device)
    else:
        positions = tuple(p + i for p, n in zip(state.query_prefixes, state.lengths) for i in range(n))
        requests = tuple(rid for rid, n in enumerate(state.lengths) for _ in range(n))
        real = (True,) * len(positions)
        chunks, selected = state.lengths, None
    rows = len(positions)
    indices = torch.arange(rows, dtype=torch.int64, device=device)
    return CPContext(
        cp_size=1, cp_rank=0, chunk_length=rows, padded_seq_len=rows,
        seq_len_full=sum(state.lengths), relative_positions=indices,
        prefix_length=state.query_prefixes[0],
        global_positions=torch.tensor(positions, dtype=torch.int64, device=device),
        local_is_real=torch.tensor(real, dtype=torch.bool, device=device),
        unpad_restore=torch.tensor([i for i, valid in enumerate(real) if valid], device=device),
        seq_len_total=max(p + n for p, n in zip(state.query_prefixes, state.lengths)),
        cp_info=None, req_id_per_token=torch.tensor(requests, dtype=torch.int32, device=device),
        chunk_lengths_per_req=chunks, kv_cache_sharded=False,
        input_lengths_global_host=state.lengths, prefix_lengths_host=state.query_prefixes,
        gather_restore_positions=selected,
    )


def assert_rows(test, actual, expected):
    test.assertEqual(len(actual), len(expected))
    for left, right in zip(actual, expected):
        if isinstance(right, slice):
            test.assertIsInstance(left, slice)
            test.assertEqual((left.start, left.stop, left.step), (right.start, right.stop, right.step))
        else:
            test.assertEqual(left.dtype, right.dtype)
            test.assertEqual(left.shape, right.shape)
            test.assertEqual(left.device, right.device)
            test.assertTrue(torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def compact_lookup_state(test):
    from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy

    state = state_for("cpu", prefixes=(126976,), lengths=(4096,))
    state.runtime.policy = SharedPrefillPolicy(mode="compact")
    state.v4._propagate_cp_ctx = lambda context: None
    with public_ced_cpu_scope(state):
        state.prepare_plan()
    test.assertIsNotNone(state._indexer_lookup_compact_birth)
    return state


def legacy_rows(state, context=None, *, full=False, selected_only=False):
    with patch.dict(os.environ, {ENV: "0"}):
        # A new state also prevents the old selected-only cache from hiding
        # an intentional in-place update in an invalidation test.
        oracle = state_for(
            state.inputs.input_ids.device, state.query_prefixes, state.lengths,
            inputs=state.inputs, cu=state.attn.cu_seqlens,
        )
        oracle.prefill, oracle.idle = state.prefill, state.idle
        return oracle.full_rows(context, selected_only=selected_only) if full else oracle.rows(context)


class _LocalRowsCases:
    def test_row_views_preserve_cached_tuple_and_selector_identity(self):
        from rtp_llm.models_py.modules.dsv41.ppu.row_metadata import RowLayout, SelectorKind

        with patch.dict(os.environ, {ENV: "1", LOCAL_PREFILL_FULL_ROWS_ENV: "1"}):
            state = state_for(self.device)
            rows = state.rows()
            view = state.row_view().validate()
            self.assertIs(view.as_rows(), rows)
            self.assertIs(view.layout, RowLayout.LOCAL_QUERY)
            self.assertIs(view.selector.value, rows[3])
            self.assertIs(view.selector.kind, SelectorKind.MASK)
            full = state.full_row_view(None).validate()
            self.assertIs(full.positions, rows[0])
            self.assertIs(full.request_ids, rows[1])
            self.assertIs(full.selector.value, full.as_full_rows()[2])
            self.assertIs(full.selector.kind, SelectorKind.ALL)
            self.assertFalse(view.selector.belongs_to(full))
            context = cp1_context(state)
            contextual = state.full_row_view(context)
            self.assertIs(contextual.as_full_rows(), state.full_rows(context))
            self.assertIs(contextual.layout, RowLayout.FULL)
            self.assertIs(contextual.scope.context, context)

    def test_fresh_description_rejects_rebound_sources_and_new_forward(self):
        with patch.dict(os.environ, {ENV: "1"}):
            state = state_for(self.device)
            context = cp1_context(state)
            view = state.row_view(context)
            selector = view.selector
            context.global_positions = context.global_positions.clone()
            current = state.row_view(context)
            self.assertFalse(selector.belongs_to(current))
            self.assertIs(current.positions, context.global_positions)
            next_state = state_for(self.device, inputs=state.inputs, cu=state.attn.cu_seqlens)
            self.assertFalse(current.selector.belongs_to(next_state.row_view(context)))

    def test_full_row_slice_preserves_kv_producer_and_metadata_bytes(self):
        for prefixes, lengths in (
            ((17, 99, 64), (3, 0, 5)), ((4, 10), (0, 0)), ((126976,), (4096,)),
        ):
            with self.subTest(lengths=lengths), patch.dict(os.environ, {ENV: "1"}):
                state = state_for(self.device, prefixes, lengths)
                with patch.dict(os.environ, {LOCAL_PREFILL_FULL_ROWS_ENV: "0"}):
                    positions, requests, mask = state.full_rows(None)
                with patch.dict(os.environ, {LOCAL_PREFILL_FULL_ROWS_ENV: "1"}):
                    current_pos, current_req, selector = state.full_rows(None)
                self.assertIs(current_pos, positions)
                self.assertIs(current_req, requests)
                self.assertEqual(selector, slice(None))
                self.assertIs(state.rows()[3], mask, "public local row masks stay tensors")
                for width, dtype in ((512, torch.bfloat16), (528, torch.uint8), (512, torch.float32)):
                    values = (torch.arange(sum(lengths) * width, device=self.device) % 127).to(dtype)
                    values = values.reshape(sum(lengths), width)
                    expected, actual = values[mask], values[selector]
                    self.assertTrue(torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)))
                    if values.numel():
                        self.assertEqual(actual.data_ptr(), values.data_ptr())
                self.assertTrue(torch.equal(positions[mask], current_pos[selector]))
                self.assertTrue(torch.equal(requests[mask], current_req[selector]))

    def test_ragged_zero_and_target_length_rows_match_legacy(self):
        for prefixes, lengths in (
            ((17, 99, 64), (3, 0, 5)), ((4, 10), (0, 0)), ((126976,), (4096,)),
        ):
            with self.subTest(prefixes=prefixes, lengths=lengths):
                state = state_for(self.device, prefixes, lengths)
                expected = legacy_rows(state)
                with patch.dict(os.environ, {ENV: "1"}):
                    actual = state.rows()
                    repeated = state.rows()
                assert_rows(self, actual, expected)
                self.assertTrue(all(left is right for left, right in zip(actual, repeated)))
                self.assertEqual(tuple(value.dtype for value in actual),
                                 (torch.int64, torch.int64, torch.int32, torch.bool))
                self.assertEqual(actual[0].tolist(), [p + i for p, n in zip(prefixes, lengths) for i in range(n)])
                self.assertEqual(actual[1].tolist(), [rid for rid, n in enumerate(lengths) for _ in range(n)])
                self.assertEqual(actual[2].tolist(), [0, *accumulate(lengths)])

    def test_rows_and_full_rows_none_share_one_native_cu_upload(self):
        from rtp_llm.ops import PyAttentionInputs

        state = state_for(self.device, prefixes=(126976,), lengths=(4096,))
        native = PyAttentionInputs()
        self.assertEqual(type(native).__name__, "PyAttentionInputs", "native binding must not be an EmptyClass fallback")
        native.cu_seqlens = state.attn.cu_seqlens
        native.is_cuda_graph = False
        state.attn = native
        source = native.cu_seqlens
        self.assertTrue(hasattr(source, "_cdata"), "exercise the actual native Tensor getter")
        self.assertTrue(all(native.cu_seqlens._cdata == source._cdata for _ in range(10)))
        original_to = torch.Tensor.to
        uploads = []

        def to(tensor, *args, **kwargs):
            if tensor._cdata == source._cdata:
                uploads.append((args, kwargs))
            return original_to(tensor, *args, **kwargs)

        with patch.dict(os.environ, {ENV: "1"}), patch.object(torch.Tensor, "to", new=to):
            first = state.rows()
            for _ in range(20):
                current = state.rows()
                full = state.full_rows(None)
                self.assertTrue(all(left is right for left, right in zip(first, current)))
                self.assertIs(full[0], first[0])
                self.assertIs(full[1], first[1])
                self.assertIs(full[2], first[3])
        self.assertEqual(len(uploads), 1)

    def test_full_to_compact_cp1_views_preserve_padding_and_selection(self):
        state = state_for(self.device, prefixes=(17, 64), lengths=(4, 3))
        original, compact = cp1_context(state), cp1_context(state, compact=True)
        with patch.dict(os.environ, {ENV: "1"}):
            full_before = state.rows()
            state.bind_context(original)
            assert_rows(self, state.rows(original), legacy_rows(state, original))
            state.bind_context(compact)
            state.compacted = True
            first = state.rows(compact)
            second = state.rows(compact)
            assert_rows(self, first, legacy_rows(state, compact))
            self.assertTrue(all(left is right for left, right in zip(first, second)))
            self.assertEqual(first[0].tolist(), [18, 20, 64, 64])
            self.assertEqual(first[2].tolist(), [0, 2, 4])
            self.assertEqual(first[3].tolist(), [True, True, True, False])
            for selected_only in (False, True):
                left = state.full_rows(compact, selected_only=selected_only)
                right = legacy_rows(state, compact, full=True, selected_only=selected_only)
                assert_rows(self, left, right)
                repeated = state.full_rows(compact, selected_only=selected_only)
                self.assertTrue(all(a is b for a, b in zip(left, repeated)))
            assert_rows(self, state.rows(), full_before)
            self.assertIs(state.context, compact)
            self.assertIs(state.original_context, original)

    def test_cu_inplace_change_and_rebind_refresh_without_new_state(self):
        state = state_for(self.device, prefixes=(17, 64), lengths=(3, 2))
        with patch.dict(os.environ, {ENV: "1"}):
            first = state.rows()
            state.attn.cu_seqlens.copy_(torch.tensor((0, 2, 5), dtype=torch.int32))
            updated = state.rows()
            assert_rows(self, updated, legacy_rows(state))
            self.assertEqual(updated[1].tolist(), [0, 0, 1, 1, 1])
            self.assertIsNot(first[2], updated[2])
            state.attn.cu_seqlens = state.attn.cu_seqlens.clone()
            rebound = state.rows()
            assert_rows(self, rebound, updated)
            self.assertIsNot(rebound[2], updated[2])

    def test_cp1_tensor_and_selected_updates_invalidate_cached_views(self):
        state = state_for(self.device, prefixes=(17, 64), lengths=(4, 3))
        compact = cp1_context(state, compact=True)
        with patch.dict(os.environ, {ENV: "1"}):
            first = state.rows(compact)
            old_selected = state.full_rows(compact, selected_only=True)
            old_mask = state.full_rows(compact)
            compact.req_id_per_token[-1] = 0
            compact.global_positions[-1] += 1
            compact.local_is_real[-1] = True
            updated = state.rows(compact)
            assert_rows(self, updated, legacy_rows(state, compact))
            self.assertIsNot(first[2], updated[2])
            compact.gather_restore_positions[0] = 0
            selected = state.full_rows(compact, selected_only=True)
            full = state.full_rows(compact)
            assert_rows(self, selected, legacy_rows(state, compact, full=True, selected_only=True))
            assert_rows(self, full, legacy_rows(state, compact, full=True))
            self.assertIsNot(selected[0], old_selected[0])
            self.assertIsNot(full[2], old_mask[2])
            self.assertEqual(selected[0].tolist(), [17, 20, 64])
            self.assertEqual(full[2].tolist(), [True, False, False, True, True, False, False])

    def test_new_forward_with_reused_input_and_metadata_storage_is_fresh(self):
        for inference in (False, True):
            with self.subTest(inference=inference), torch.inference_mode(inference):
                first = state_for(self.device)
                with patch.dict(os.environ, {ENV: "1"}):
                    old = first.rows()
                    old_positions = old[0].clone()
                    input_pointer = first.inputs.input_ids.data_ptr()
                    cu_pointer = first.attn.cu_seqlens.data_ptr()
                    first.inputs.input_ids.add_(50)
                    first.attn.cu_seqlens.copy_(torch.tensor((0, 1, 5, 8), dtype=torch.int32))
                    second = state_for(
                        self.device, prefixes=(1024, 8, 128), lengths=(1, 4, 3),
                        inputs=first.inputs, cu=first.attn.cu_seqlens,
                    )
                    current = second.rows()
                    assert_rows(self, current, legacy_rows(second))
                    self.assertEqual(second.inputs.input_ids.data_ptr(), input_pointer)
                    self.assertEqual(second.attn.cu_seqlens.data_ptr(), cu_pointer)
                    self.assertIsNot(old[0], current[0])
                    self.assertIsNot(old[2], current[2])
                    self.assertFalse(torch.equal(current[0], old_positions))
                    self.assertEqual(current[0].tolist(), [1024, 8, 9, 10, 11, 128, 129, 130])

    def test_layout_host_lengths_and_prefixes_are_part_of_the_cache_key(self):
        state = state_for(self.device, prefixes=(17, 64), lengths=(3, 2))
        with patch.dict(os.environ, {ENV: "1"}):
            first = state.rows()
            state.lengths, state.query_prefixes = (1, 4), (90, 140)
            state.attn.cu_seqlens.copy_(torch.tensor((0, 1, 5), dtype=torch.int32))
            current = state.rows()
            assert_rows(self, current, legacy_rows(state))
            self.assertIsNot(current[0], first[0])
            self.assertEqual(current[0].tolist(), [90, 140, 141, 142, 143])
            original = cp1_context(state)
            first_context = state.rows(original)
            rebound = replace(original, global_positions=original.global_positions.clone())
            current_context = state.rows(rebound)
            assert_rows(self, current_context, first_context)
            self.assertIsNot(current_context[2], first_context[2])


class LocalPrefillRowsCPU(_LocalRowsCases, unittest.TestCase):
    device = "cpu"

    def test_engram_l0_unmasked_compare_preserves_long_values_and_rejection_without_index(self):
        for ids_dtype, window_dtype in ((torch.int32, torch.int64), (torch.int64, torch.int32)):
            with self.subTest(ids_dtype=ids_dtype, window_dtype=window_dtype):
                ids = torch.tensor((3, 5, 7), dtype=ids_dtype)
                state = state_for("cpu", prefixes=(0,), lengths=(3,))
                storage = torch.tensor((3, 99, 5, 99, 7, 99), dtype=window_dtype)
                state.window_current_tokens = storage[::2]
                before_ids, before_storage = ids.clone(), storage.clone()
                args = object(), ids, object(), object()
                with patch.object(torch, "ones_like", side_effect=AssertionError("unexpected all-true mask")), \
                        patch.object(torch.Tensor, "__getitem__", side_effect=AssertionError("unexpected indexing")), \
                        patch.object(torch, "nonzero", side_effect=AssertionError("unexpected nonzero")), \
                        patch.object(torch, "equal", wraps=torch.equal) as equal:
                    result = state.before_block(NS(layer_id=0), *args)
                self.assertTrue(all(got is original for got, original in zip(result, args)))
                equal.assert_called_once()
                self.assertTrue(all(value.dtype == torch.int64 for value in equal.call_args.args))
                self.assertTrue(torch.equal(ids, before_ids))
                self.assertTrue(torch.equal(storage, before_storage))
                state.window_current_tokens.add_(1)
                with patch.object(torch, "ones_like", side_effect=AssertionError("unexpected mask")), \
                        patch.object(torch.Tensor, "__getitem__", side_effect=AssertionError("unexpected indexing")):
                    with self.assertRaisesRegex(ValueError, "real CP Engram windows disagree with input tokens"):
                        state.before_block(NS(layer_id=0), *args)

    def test_engram_l0_context_padding_and_malformed_inputs_keep_legacy_indexing(self):
        state = state_for("cpu", prefixes=(0,), lengths=(4,))
        state.context = cp1_context(state)
        state.context.local_is_real = torch.tensor((True, False, True, False))
        ids = torch.tensor((3, -4, 7, -5), dtype=torch.int32)
        state.window_current_tokens = torch.tensor((3, 99, 7, 99), dtype=torch.int64)
        args = object(), ids, object(), object()
        original_getitem, selected = torch.Tensor.__getitem__, []

        def getitem(value, selector):
            if selector is state.context.local_is_real:
                selected.append(value)
            return original_getitem(value, selector)

        with patch.object(torch.Tensor, "__getitem__", new=getitem):
            result = state.before_block(NS(layer_id=0), *args)
        self.assertTrue(all(got is original for got, original in zip(result, args)))
        self.assertEqual(len(selected), 2)
        self.assertIs(selected[0], state.window_current_tokens)
        self.assertIs(selected[1], ids)
        state.window_current_tokens[0] = 8
        with self.assertRaisesRegex(ValueError, "real CP Engram windows disagree"):
            state.before_block(NS(layer_id=0), *args)

        # An unknown context may supply None as its selector. Preserve its
        # original Tensor[None] semantics rather than inventing an all-true mask.
        state.context.local_is_real = None
        state.window_current_tokens = ids.long()
        selected.clear()
        with patch.object(torch, "ones_like", side_effect=AssertionError("changed context selector")), \
                patch.object(torch.Tensor, "__getitem__", new=getitem):
            state.before_block(NS(layer_id=0), *args)
        self.assertEqual(len(selected), 2)

        state.context = None
        for malformed_ids, window in (
            (ids, torch.tensor((3, 5, 7))),
            (ids.reshape(2, 2), torch.tensor((3, 5, 7, 9))),
        ):
            with self.subTest(shape=malformed_ids.shape, window_shape=window.shape):
                state.window_current_tokens = window
                with patch.object(torch, "ones_like", wraps=torch.ones_like) as mask:
                    with self.assertRaises(IndexError):
                        state.before_block(NS(layer_id=0), object(), malformed_ids, None, None)
                mask.assert_called_once_with(malformed_ids, dtype=torch.bool)
        state.window_current_tokens = ids.float()
        with patch.object(torch, "ones_like", wraps=torch.ones_like) as mask:
            state.before_block(NS(layer_id=0), *args)
        mask.assert_called_once_with(ids, dtype=torch.bool)
        state.window_current_tokens = torch.empty(ids.shape, dtype=torch.int64, device="meta")
        with patch.object(torch, "ones_like", side_effect=RuntimeError("legacy device fallback")):
            with self.assertRaisesRegex(RuntimeError, "legacy device fallback"):
                state.before_block(NS(layer_id=0), *args)

    def test_engram_l0_real_visual_canonical_mapping_and_errors_are_preserved(self):
        from types import MethodType
        from rtp_llm.models_py.modules.dsv41.ppu.model import DeepSeekV41PPUModel

        state = state_for("cpu", prefixes=(0,), lengths=(4,))
        model = state.runtime.model
        model._image_plan = object()
        model.shared_weight_views = NS(geometry=NS(image_token_id=13))
        model._engram_input_ids = MethodType(DeepSeekV41PPUModel._engram_input_ids, model)
        state.v4._image_token_mask = torch.tensor((False, True, False, False))
        ids = torch.tensor((2, 16, 5, -1), dtype=torch.int32)
        before = ids.clone()
        state.window_current_tokens = torch.tensor((2, 13, 5, -1), dtype=torch.int64)
        args = object(), ids, object(), object()
        with patch.object(model, "_engram_input_ids", wraps=model._engram_input_ids) as canonical, \
                patch.object(torch, "ones_like", side_effect=AssertionError("unexpected mask")), \
                patch.object(torch.Tensor, "__getitem__", side_effect=AssertionError("unexpected indexing")):
            result = state.before_block(NS(layer_id=0), *args)
        canonical.assert_called_once()
        self.assertIs(canonical.call_args.args[0], ids)
        self.assertIs(canonical.call_args.args[1], state.v4._image_token_mask)
        self.assertTrue(all(got is original for got, original in zip(result, args)))
        self.assertTrue(torch.equal(ids, before))
        for values in ((2, 16, 5, -1), (2, 13, 6, -1)):
            state.window_current_tokens = torch.tensor(values)
            with self.assertRaisesRegex(ValueError, "real CP Engram windows disagree"):
                state.before_block(NS(layer_id=0), *args)
        state.v4._image_token_mask = torch.tensor((False, True))
        with self.assertRaisesRegex(ValueError, "Engram visual row mask disagrees with input tokens"):
            state.before_block(NS(layer_id=0), *args)

    def _public_meta_forward(self, change=None):
        from types import MethodType
        from rtp_llm.models_py.modules.dsv41.ppu.attention_layer import SharedPPUAttention
        from rtp_llm.models_py.modules.dsv41.prefill import forward as public_forward

        state = state_for("cpu", prefixes=(17,), lengths=(3,))
        runtime, v4 = state.runtime, state.runtime.model.v4
        runtime.current = state
        v4._ppu_runtime = runtime
        v4.fp8_kv_cache, v4.hc_mult = True, 1
        v4.args = NS(v41_config=object())
        v4._cp_info, v4._cp_size = None, 1
        v4.capture_aux_hidden_layer_ids = ()
        v4._mtp_hidden_buffer = None
        v4._prefill_ws_q_dim = v4._prefill_ws_main_w = v4._prefill_ws_idx_w = 2
        v4._prefill_ws_full_rows = 3
        v4._dsv4_prefill_fast_layer_calls = None
        v4.embed = lambda ids: torch.ones((ids.numel(), 2))
        v4._hc_head_reduce = lambda h: h.squeeze(1)
        v4.norm = lambda h: h
        state.attn.input_lengths = torch.tensor((3,), dtype=torch.int32)
        state.attn.prefix_lengths = torch.tensor((17,), dtype=torch.int32)
        positions = torch.arange(17, 20, dtype=torch.int64)
        workspace, seen = object(), []

        class Layer:
            def __init__(self, index):
                self.attn = SharedPPUAttention.__new__(SharedPPUAttention)
                torch.nn.Module.__init__(self.attn)
                self.attn.runtime = runtime
                self.attn.layer_id = self.attn.compress_ratio = index
                self.attn.window_size = 128
                self.attn._kv_cache = self.attn._block_tables_by_type = None
                self.attn._cp_ctx = self.attn._prefill_meta_shared = None
                self.attn._shared_attention = None

            def __call__(self, h, *args, **kwargs):
                seen.append(self.attn._prefill_meta_shared)
                return h

        v4.layers = [Layer(0), Layer(1)]
        v4._propagate_cp_ctx = lambda cp: [setattr(layer.attn, "_cp_ctx", cp) for layer in v4.layers]
        original_builder = SharedPPUAttention._build_shared_prefill_meta

        def replacement_builder(attn, *args, **kwargs):
            return original_builder(attn, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {
                "RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA": "0" if change == "host_off" else "1",
            }))
            stack.enter_context(patch.object(public_forward._rt, "ENABLED", False))
            stack.enter_context(patch.object(public_forward._fwd_dbg, "enabled", return_value=False))
            allocate = stack.enter_context(patch.object(public_forward, "PrefillWorkspace", return_value=workspace))
            build = stack.enter_context(patch.object(
                public_forward, "build_and_propagate_prefill_meta_fp8",
                wraps=public_forward.build_and_propagate_prefill_meta_fp8,
            ))
            if change == "instance_builder":
                v4.layers[1].attn._build_shared_prefill_meta = MethodType(replacement_builder, v4.layers[1].attn)
            elif change == "class_builder":
                stack.enter_context(patch.object(SharedPPUAttention, "_build_shared_prefill_meta", replacement_builder))
            elif change == "unknown_consumer":
                v4.layers[1].attn._shared_prefill_meta_inputs_required = None
            elif change == "cp8":
                v4._cp_size = 8
            elif change == "graph":
                state.is_graph = True
            elif change == "verify":
                state.verify = True
            elif change == "stale_runtime":
                runtime.current = None
            if change is None:
                for owner, name in ((torch.Tensor, "item"), (torch.Tensor, "index_select"), (torch, "searchsorted")):
                    stack.enter_context(patch.object(owner, name, side_effect=AssertionError("unused generic metadata")))
            result = public_forward.forward_layers(
                v4, None, state.inputs.input_ids, positions, state.attn.cu_seqlens,
                None, None, attn_inputs=state.attn,
            )
        allocate.assert_called_once()
        build.assert_called_once()
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(meta.workspace is workspace for meta in seen))
        self.assertTrue(all(layer.attn._prefill_meta_shared is None for layer in v4.layers))
        self.assertTrue(torch.equal(result, torch.ones((3, 2))))
        self.assertIs(build.call_args.kwargs["workspace"], workspace)
        # The public forward already normalizes positions via reshape(-1).
        # Preserve that view and its values, rather than requiring TensorImpl
        # identity that neither the original nor candidate forward provides.
        actual_positions = build.call_args.kwargs["position_ids"]
        self.assertTrue(torch._C._is_alias_of(actual_positions, positions))
        self.assertTrue(torch.equal(actual_positions, positions))
        self.assertEqual(actual_positions.shape, positions.shape)
        self.assertEqual(actual_positions.dtype, positions.dtype)
        self.assertEqual(actual_positions.stride(), positions.stride())
        self.assertEqual(actual_positions.storage_offset(), positions.storage_offset())
        return build.call_args

    @torch.inference_mode()
    def test_public_prefill_omits_only_unconsumed_metadata_and_keeps_builder_lifetime(self):
        call = self._public_meta_forward()
        self.assertEqual(call.args[2], 0)
        for name in ("sp_per_req", "req_id_per_token", "input_lengths", "prefix_lengths"):
            self.assertIsNone(call.kwargs[name])
        self.assertEqual(call.kwargs["max_seqlen_q"], 0)

    @torch.inference_mode()
    def test_public_prefill_metadata_falls_back_for_changed_consumers_and_domain(self):
        for change in ("instance_builder", "class_builder", "unknown_consumer", "cp8", "graph", "verify", "host_off", "stale_runtime"):
            with self.subTest(change=change):
                call = self._public_meta_forward(change)
                self.assertEqual(call.args[2], 17)
                self.assertEqual(call.kwargs["max_seqlen_q"], 3)
                for name, values in (("sp_per_req", (17,)), ("req_id_per_token", (0, 0, 0)),
                                     ("input_lengths", (3,)), ("prefix_lengths", (17,))):
                    self.assertEqual(tuple(call.kwargs[name].tolist()), values)

    def _compact_lookup_state(self):
        return compact_lookup_state(self)


    @torch.inference_mode()
    def test_compact_cu_upload_preserves_values_independent_storage_version_and_fallback(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_swa_metadata

        def cpu_upload(values, device, dtype):
            self.assertEqual(device.type, "cpu", "this fixture must never upload to an accelerator")
            return torch.tensor(values, device=device, dtype=dtype)

        for change in (None, "cpu", "missing_birth", "graph", "cp8", "mutation"):
            with self.subTest(change=change), patch.dict(os.environ, {
                **LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1",
            }):
                state = self._compact_lookup_state()
                state._activate_compact(torch.empty((140, 1)))
                context = state.context
                if change == "missing_birth":
                    state._indexer_lookup_compact_birth = None
                elif change == "graph":
                    state.is_graph = True
                elif change == "cp8":
                    context.cp_size = 8
                elif change == "mutation":
                    context.req_id_per_token.add_(0)
                # The real decorated planner/birth proof above stays intact;
                # only upload admission and transfer are CPU stand-ins here.
                with patch.object(torch.Tensor, "is_cuda", new=property(
                    lambda tensor: change != "cpu" and tensor is state.inputs.input_ids,
                )), patch.object(torch.cuda, "is_current_stream_capturing", return_value=False), \
                        patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("unexpected GPU initialization")), \
                        patch.object(_v41_swa_metadata, "_upload", side_effect=cpu_upload) as upload:
                    rows = state.rows(context)
                    cu = rows[2]
                    self.assertEqual(cu.shape, (2,))
                    self.assertEqual(cu.dtype, torch.int32)
                    self.assertEqual(cu.device, state.inputs.input_ids.device)
                    self.assertTrue(torch.equal(cu, torch.tensor((0, 140), dtype=torch.int32)))
                    self.assertNotEqual(cu.untyped_storage().data_ptr(), state.plan.cu_seqlens.untyped_storage().data_ptr())
                    if change is None:
                        upload.assert_called_once_with((0, 140), state.inputs.input_ids.device, torch.int32)
                    else:
                        upload.assert_not_called()
                    if change in (None, "cpu"):
                        self.assertFalse(cu.is_inference())
                        self.assertIs(state.rows(context)[2], cu)
                        self.assertEqual(upload.call_count, int(change is None))
                        version = cu._version
                        cu.add_(0)
                        self.assertGreater(cu._version, version)
                        self.assertIsNone(state.local_indexer_lookup_slices(context, rows, 140))
                    else:
                        self.assertTrue(cu.is_inference(), "fallback retains its original allocation scope")

    @torch.inference_mode()
    def test_compact_host_validation_and_evidence_match_device_oracle_without_readback(self):
        reports, outputs = [], []
        with patch.dict(os.environ, {**LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1"}):
            for host in (False, True):
                state = self._compact_lookup_state()
                self.assertTrue(all(not value.is_inference() for value in state._indexer_lookup_stamp.rows))
                self.assertIsNotNone(state._compact_validation_original)
                if not host:
                    # Keep the original device check/evidence as the value
                    # oracle while using exactly the same real planner.
                    state._compact_validation_original = None
                hidden = torch.empty((140, 1))
                with ExitStack() as stack:
                    if host:
                        for name in ("item", "__bool__", "__int__", "cpu", "tolist", "sum", "all"):
                            stack.enter_context(patch.object(
                                torch.Tensor, name, side_effect=AssertionError("unexpected device value read"),
                            ))
                        stack.enter_context(patch.object(
                            torch, "tensor", side_effect=AssertionError("unexpected terminal-position H2D"),
                        ))
                    state._activate_compact(hidden)
                    reports.append(state.evidence())
                    if host:
                        self.assertIsNotNone(state._compact_host_evidence())
                        self.assertFalse(state.valid_output_rows.is_inference())
                outputs.append(state.valid_output_rows.clone())
            self.assertEqual(reports[0], reports[1])
            self.assertEqual(reports[1]["local_positions"], list(range(130932, 131072)))
            self.assertEqual(reports[1]["local_valid_output_rows"], 140)
            self.assertTrue(torch.equal(outputs[0], outputs[1]))
            expected = torch.zeros(4096, dtype=torch.bool)
            expected[-140:] = True
            self.assertTrue(torch.equal(outputs[1], expected))

    @torch.inference_mode()
    def test_compact_host_validation_changed_or_unknown_sources_keep_device_check(self):
        changes = ("positions", "identity_requests", "plan_positions", "send_indices", "plan_rebind",
                   "mask", "versionless_positions", "missing_original", "missing_birth", "cache_clear",
                   "graph", "verify", "sg_align")
        original_bool = torch.Tensor.__bool__
        for change in changes:
            with self.subTest(change=change), patch.dict(os.environ, {
                **LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1",
            }):
                state = self._compact_lookup_state()
                if change == "positions":
                    state.original_context.global_positions.add_(0)
                elif change == "identity_requests":
                    state.original_context.req_id_per_token.add_(0)
                elif change == "plan_positions":
                    state.plan.context.global_positions.add_(0)
                elif change == "send_indices":
                    state.plan.exchange.send_indices.add_(0)
                elif change == "plan_rebind":
                    state.plan = replace(state.plan)
                elif change == "mask":
                    state.original_context.local_is_real.logical_not_().logical_not_()
                elif change == "versionless_positions":
                    state.original_context.global_positions = state.original_context.global_positions.clone()
                    self.assertTrue(state.original_context.global_positions.is_inference())
                elif change == "missing_original":
                    state._compact_validation_original = None
                elif change == "missing_birth":
                    state._indexer_lookup_compact_birth = None
                elif change == "cache_clear":
                    state._local_prefill_rows.pop(id(None))
                elif change in ("graph", "verify"):
                    setattr(state, "is_graph" if change == "graph" else change, True)
                else:
                    state.local_prefill_sg_align = True
                checks = []

                def checked_bool(value):
                    checks.append(value)
                    return original_bool(value)

                with patch.object(torch.Tensor, "__bool__", new=checked_bool):
                    state._activate_compact(torch.empty((140, 1)))
                self.assertEqual(len(checks), 1, "unknown or changed geometry must run the original all() check")
                self.assertIsNone(state._compact_validation_output)
                self.assertIsNone(state._compact_host_evidence())
                self.assertEqual(state.evidence()["local_valid_output_rows"], 140)

    @torch.inference_mode()
    def test_compact_host_evidence_rechecks_output_mutation_and_replacement(self):
        for change in ("write", "write_restore", "replace"):
            with self.subTest(change=change), patch.dict(os.environ, {
                **LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1",
            }):
                state = self._compact_lookup_state()
                state._activate_compact(torch.empty((140, 1)))
                stamp = state._compact_validation_output
                self.assertIsNotNone(state._compact_host_evidence())
                if change == "replace":
                    state.valid_output_rows = state.valid_output_rows.clone()
                else:
                    state.valid_output_rows[-1] = False
                    if change == "write_restore":
                        state.valid_output_rows[-1] = True
                self.assertIsNone(state._compact_host_evidence())
                self.assertEqual(state.evidence()["local_valid_output_rows"], 139 if change == "write" else 140)
                self.assertIs(state._compact_validation_output, stamp, "a consumer must not renew the output stamp")

    @torch.inference_mode()
    def test_compact_lookup_real_plan_preserves_advanced_indices_without_readback(self):
        from rtp_llm.models_py.modules.dsv41.ppu.attention_layer import _indexer_lookup_rows

        with patch.dict(os.environ, {**LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1"}):
            state = self._compact_lookup_state()
            state._activate_compact(torch.empty((140, 1)))
            cp = state.context
            rows = state.rows(cp)
            self.assertTrue(all(not value.is_inference() for value in rows))
            legacy = _indexer_lookup_rows(rows[1], 0, None)
            with patch.object(torch.Tensor, "item", side_effect=AssertionError("device read")), \
                    patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("device read")), \
                    patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host copy")), \
                    patch.object(torch, "where", side_effect=AssertionError("dynamic lookup")):
                for layer in (20, 24, 28, 32, 36):
                    with self.subTest(layer=layer):
                        lookup = state.local_indexer_lookup_slices(cp, state.rows(cp), 140)
                        actual = _indexer_lookup_rows(rows[1], 0, lookup)
                        self.assertEqual(lookup, (slice(0, 140),))
            self.assertTrue(torch.equal(actual, legacy))
            values = torch.arange(420).reshape(140, 3)
            self.assertTrue(torch.equal(values[actual], values[legacy]))
            self.assertNotEqual(values[actual].data_ptr(), values.data_ptr())
            self.assertFalse(state.local_prefill_sg_align)

    @torch.inference_mode()
    def test_compact_lookup_plan_birth_and_derived_row_mutations_are_sticky(self):
        for change in ("before_activation", "plan_request", "derived_request", "derived_cu", "mask", "rebind"):
            with self.subTest(change=change), patch.dict(os.environ, {
                **LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1",
            }):
                state = self._compact_lookup_state()
                birth = state._indexer_lookup_compact_birth
                if change == "before_activation":
                    state.plan.context.req_id_per_token.add_(0)
                state._activate_compact(torch.empty((140, 1)))
                cp, rows = state.context, state.rows(state.context)
                if change != "before_activation":
                    self.assertEqual(state.local_indexer_lookup_slices(cp, rows, 140), (slice(0, 140),))
                if change == "plan_request":
                    cp.req_id_per_token.add_(0)
                elif change == "derived_request":
                    rows[1].add_(1).sub_(1)
                elif change == "derived_cu":
                    rows[2].add_(0)
                elif change == "mask":
                    rows[3].logical_not_().logical_not_()
                elif change == "rebind":
                    rows = (rows[0], rows[1].clone(), rows[2], rows[3])
                self.assertIsNone(state.local_indexer_lookup_slices(cp, rows, 140))
                self.assertIsNone(state.local_indexer_lookup_slices(cp, state.rows(cp), 140))
                self.assertIs(state._indexer_lookup_compact_birth, birth)

    @torch.inference_mode()
    def test_compact_lookup_preserves_phase_plan_and_unknown_row_fallbacks(self):
        for change in ("graph", "verify", "proposal", "decode", "sharded", "sg_align",
                       "cp8", "wrong_context", "plan_rebind", "cache_clear", "short", "missing_birth"):
            with self.subTest(change=change), patch.dict(os.environ, {
                **LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1",
            }):
                state = self._compact_lookup_state()
                state._activate_compact(torch.empty((140, 1)))
                cp, rows, count = state.context, state.rows(state.context), 140
                if change in ("graph", "verify", "proposal"):
                    setattr(state, "is_graph" if change == "graph" else change, True)
                elif change == "decode":
                    state.prefill = False
                elif change == "sharded":
                    state.cache.sharded = True
                elif change == "sg_align":
                    state.local_prefill_sg_align = True
                elif change == "cp8":
                    cp.cp_size = 8
                elif change == "wrong_context":
                    cp = object()
                elif change == "plan_rebind":
                    state.plan = replace(state.plan)
                elif change == "cache_clear":
                    state._local_prefill_rows.pop(id(cp))
                elif change == "short":
                    count = 139
                else:
                    state._indexer_lookup_compact_birth = None
                self.assertIsNone(state.local_indexer_lookup_slices(cp, rows, count))

    def test_indexer_lookup_birth_keeps_legacy_indices_and_copy_semantics(self):
        from rtp_llm.models_py.modules.dsv41.ppu.attention_layer import _indexer_lookup_rows

        for count in (4, 140, 4096):
            with self.subTest(count=count), torch.inference_mode(), patch.dict(os.environ, LOOKUP_ENV):
                state = state_for("cpu", prefixes=(126976,), lengths=(count,))
                rows = state.rows()
                self.assertFalse(rows[1].is_inference())
                legacy = _indexer_lookup_rows(rows[1], 0, None)
                with patch.object(torch.Tensor, "item", side_effect=AssertionError("device read")), \
                        patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("device read")), \
                        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host copy")), \
                        patch.object(torch, "where", side_effect=AssertionError("dynamic lookup")):
                    lookup = state.local_indexer_lookup_slices(None, rows, count)
                    actual = _indexer_lookup_rows(rows[1], 0, lookup)
                    self.assertEqual(lookup, (slice(0, count),))
                self.assertEqual(actual.dtype, legacy.dtype)
                self.assertTrue(torch.equal(actual, legacy))
                values = torch.arange(count * 3).reshape(count, 3)
                selected = values[actual]
                self.assertTrue(torch.equal(selected, values[legacy]))
                self.assertNotEqual(selected.data_ptr(), values.data_ptr(), "keep advanced-indexing copies")
                output = torch.full_like(values, -1)
                output[actual] = selected
                self.assertTrue(torch.equal(output, values))
        mixed = torch.tensor((1, 0, 1, 0), dtype=torch.int64)
        self.assertTrue(torch.equal(_indexer_lookup_rows(mixed, 0, None), torch.tensor((1, 3))))

    def test_indexer_lookup_mutated_or_rebound_rows_and_sources_decline(self):
        changes = (
            "request_write", "request_rebind", "mask_write", "cu_write", "cu_rebind",
            "input_write", "input_rebind", "input_missing", "cu_missing", "cu_sparse",
            "lengths", "prefixes", "short_queries", "cache_rebind",
            "cache_clear", "cached_rows_rebind",
        )
        for change in changes:
            with self.subTest(change=change), torch.no_grad(), patch.dict(os.environ, LOOKUP_ENV):
                state = state_for("cpu", prefixes=(17,), lengths=(4,))
                rows = state.rows()
                self.assertEqual(state.local_indexer_lookup_slices(None, rows, 4), (slice(0, 4),))
                count = 4
                if change == "request_write":
                    rows[1][0] = 1
                elif change == "request_rebind":
                    rows = (rows[0], rows[1].clone(), rows[2], rows[3])
                elif change == "mask_write":
                    rows[3][0] = False
                elif change == "cu_write":
                    state.attn.cu_seqlens.add_(0)
                elif change == "cu_rebind":
                    state.attn.cu_seqlens = state.attn.cu_seqlens.clone()
                elif change == "input_write":
                    state.inputs.input_ids.add_(0)
                elif change == "input_rebind":
                    state.inputs.input_ids = state.inputs.input_ids.clone()
                elif change == "input_missing":
                    state.inputs.input_ids = None
                elif change == "cu_missing":
                    state.attn.cu_seqlens = None
                elif change == "cu_sparse":
                    state.attn.cu_seqlens = state.attn.cu_seqlens.to_sparse()
                elif change == "lengths":
                    state.lengths = (3,)
                elif change == "prefixes":
                    state.query_prefixes = (18,)
                elif change == "short_queries":
                    count = 3
                elif change == "cache_rebind":
                    state.cache = NS(sharded=False)
                elif change == "cache_clear":
                    state._local_prefill_rows.clear()
                else:
                    cached = state._local_prefill_rows[id(None)]
                    state._local_prefill_rows[id(None)] = (cached[0], tuple(value.clone() for value in rows), *cached[2:])
                self.assertIsNone(state.local_indexer_lookup_slices(None, rows, count))
                if change == "request_write":
                    rows[1][0] = 0
                    self.assertIsNone(state.local_indexer_lookup_slices(None, rows, 4))
                    self.assertIs(state.rows()[1], rows[1], "cache hits must not renew the proof")

    def test_indexer_lookup_unknown_phases_and_unproven_birth_decline(self):
        changes = (
            "workspace_off", "host_metadata_off", "rows_off", "graph", "verify", "proposal",
            "attn_graph", "attn_verify", "decode_role", "speculative", "idle", "sharded",
            "decode", "sg_align", "compacted", "cp1", "cp8", "padded", "external_rows",
        )
        for change in changes:
            with self.subTest(change=change), torch.no_grad(), patch.dict(os.environ, LOOKUP_ENV):
                state = state_for("cpu", prefixes=(17,), lengths=(4,))
                rows = state.rows()
                context = None
                if change in ("workspace_off", "host_metadata_off", "rows_off"):
                    key = {"workspace_off": "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE",
                           "host_metadata_off": "RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA",
                           "rows_off": ENV}[change]
                    os.environ[key] = "0"
                elif change in ("graph", "verify", "proposal"):
                    setattr(state, "is_graph" if change == "graph" else change, True)
                elif change in ("attn_graph", "attn_verify"):
                    setattr(state.attn, "is_cuda_graph" if change == "attn_graph" else "is_target_verify", True)
                elif change in ("decode_role", "speculative"):
                    setattr(state.runtime.model, "_is_decode_role" if change == "decode_role" else "_is_speculative", True)
                elif change == "idle":
                    state.idle = True
                elif change == "sharded":
                    state.cache.sharded = True
                elif change == "decode":
                    state.prefill = False
                elif change == "sg_align":
                    state.local_prefill_sg_align = True
                elif change == "compacted":
                    state.compacted = True
                elif change in ("cp1", "cp8", "padded"):
                    context = cp1_context(state)
                    if change == "cp8":
                        context = replace(context, cp_size=8, kv_cache_sharded=True)
                    elif change == "padded":
                        context.local_is_real[-1] = False
                else:
                    rows = tuple(value.clone() for value in rows)
                self.assertIsNone(state.local_indexer_lookup_slices(context, rows, 4))
        with torch.inference_mode(), patch.dict(os.environ, LOOKUP_ENV):
            state = state_for("cpu", prefixes=(17,), lengths=(4,))
            with patch.dict(os.environ, {"RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA": "0"}):
                rows = state.rows()
            self.assertTrue(rows[1].is_inference())
            self.assertIsNone(state.local_indexer_lookup_slices(None, rows, 4))
        with torch.no_grad(), patch.dict(os.environ, LOOKUP_ENV):
            state = state_for("cpu", prefixes=(17, 20), lengths=(2, 2), cu=torch.tensor([0, 1, 4]))
            self.assertIsNone(state.local_indexer_lookup_slices(None, state.rows(), 4))

    def test_batched_indexer_lookup_matches_request_ids_and_checks_host_cu(self):
        from rtp_llm.models_py.modules.dsv41.ppu.attention_layer import _indexer_lookup_rows

        with torch.no_grad(), patch.dict(os.environ, LOOKUP_ENV):
            state = state_for("cpu", prefixes=(17, 20, 31, 90), lengths=(4, 2, 6, 5))
            rows = state.rows()
            lookup = state.local_indexer_lookup_slices(None, rows, 17)
            self.assertEqual(lookup, (slice(0, 4), slice(4, 6), slice(6, 12), slice(12, 17)))
            for rank in range(4):
                self.assertTrue(torch.equal(_indexer_lookup_rows(rows[1], rank, lookup),
                                            _indexer_lookup_rows(rows[1], rank, None)))
            state.attn.cu_seqlens.add_(0)
            self.assertIsNone(state.local_indexer_lookup_slices(None, rows, 17))

    @torch.inference_mode()
    def test_batched_compact_output_proof_keeps_every_requests_tail(self):
        from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy

        with patch.dict(os.environ, {**LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1"}):
            state = state_for("cpu", prefixes=(101, 4097), lengths=(272, 512))
            state.runtime.policy = SharedPrefillPolicy(mode="compact")
            state.v4._propagate_cp_ctx = lambda context: None
            with public_ced_cpu_scope(state):
                state.prepare_plan()
            state._activate_compact(torch.empty((280, 1)))
            positions = (*range(233, 373), *range(4469, 4609))
            self.assertEqual(state._compact_host_evidence(), (positions, 280, True))
            rows = state.rows(state.context)
            self.assertEqual(state.local_indexer_lookup_slices(state.context, rows, 280),
                             (slice(0, 140), slice(140, 280)))
            state.plan.exchange.send_indices.add_(0)
            self.assertIsNone(state._compact_host_evidence())

    def test_all_live_mask_birth_is_versioned_inside_inference_mode(self):
        with torch.inference_mode(), patch.dict(os.environ, {ENV: "1"}):
            state = state_for("cpu", prefixes=(126976,), lengths=(4096,))
            rows = state.rows()
            mask = rows[3]
            self.assertFalse(mask.is_inference())
            with patch.object(torch.Tensor, "item", side_effect=AssertionError("device read")), \
                    patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("device read")), \
                    patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host copy")):
                for _ in range(3):
                    self.assertIsNone(state.mega_attention_valid_mask(None, mask, 4096))
            version = mask._version
            mask[0] = False
            self.assertGreater(mask._version, version)
            self.assertIs(state.mega_attention_valid_mask(None, mask, 4096), mask)
            mask[0] = True
            self.assertIs(state.mega_attention_valid_mask(None, mask, 4096), mask,
                          "restoring values must not renew the construction stamp")
            self.assertIs(state.rows()[3], mask)

    def test_all_live_mask_replacement_sources_and_unknown_phases_fail_closed(self):
        with torch.no_grad(), patch.dict(os.environ, {ENV: "1"}):
            for change in ("mask", "source_rebind", "source_write", "lengths", "graph", "verify",
                           "proposal", "attn_graph", "attn_verify", "decode", "idle", "sharded"):
                with self.subTest(change=change):
                    state = state_for("cpu", prefixes=(17,), lengths=(4,))
                    mask = state.rows()[3]
                    self.assertIsNone(state.mega_attention_valid_mask(None, mask, 4))
                    if change == "mask":
                        mask = mask.clone()
                    elif change == "source_rebind":
                        state.attn.cu_seqlens = state.attn.cu_seqlens.clone()
                    elif change == "source_write":
                        state.attn.cu_seqlens.add_(0)
                    elif change == "lengths":
                        state.lengths = (3, 1)
                    elif change in ("graph", "verify", "proposal"):
                        setattr(state, "is_graph" if change == "graph" else change, True)
                    elif change in ("attn_graph", "attn_verify"):
                        setattr(state.attn, "is_cuda_graph" if change == "attn_graph" else "is_target_verify", True)
                    elif change == "decode":
                        state.prefill = False
                    elif change == "idle":
                        state.idle = True
                    else:
                        state.cache.sharded = True
                    self.assertIs(state.mega_attention_valid_mask(None, mask, 4), mask)
            state = state_for("cpu", prefixes=(17,), lengths=(4,))
            for cp_size in (1, 8):
                unknown = replace(cp1_context(state), cp_size=cp_size)
                mask = state.rows(unknown)[3]
                self.assertIs(state.mega_attention_valid_mask(unknown, mask, 4), mask)
            with torch.inference_mode():
                external = torch.ones(4, dtype=torch.bool)
            self.assertIs(state.mega_attention_valid_mask(None, external, 4), external)

    def test_identity_context_builds_equal_cpu_mirrors_without_copy_when_proven(self):
        with torch.inference_mode(), patch.dict(os.environ, {ENV: "1"}):
            state = state_for("cpu", prefixes=(17,), lengths=(4,))
            rows = state.rows()
            with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected host copy")):
                context = state.identity_context()
            self.assertIs(context.global_positions, rows[0])
            self.assertIs(context.local_is_real, rows[3])
            self.assertTrue(torch.equal(context.cp_info.prefill_qkv_restore_indice_cpu, torch.arange(4)))
            self.assertTrue(torch.equal(context.cp_info.prefill_qkv_padding_mask_cpu, rows[3]))
            self.assertIsNone(state.mega_attention_valid_mask(context, rows[3], 4))
            rows[3][1] = False
            original_cpu, copies = torch.Tensor.cpu, []

            def counted_cpu(value, *args, **kwargs):
                copies.append(value)
                return original_cpu(value, *args, **kwargs)

            with patch.object(torch.Tensor, "cpu", new=counted_cpu):
                fallback = state.identity_context()
            self.assertEqual(len(copies), 1)
            self.assertIs(copies[0], rows[3])
            self.assertTrue(torch.equal(fallback.cp_info.prefill_qkv_padding_mask_cpu, rows[3]))
            self.assertIs(state.mega_attention_valid_mask(fallback, rows[3], 4), rows[3])

    @torch.inference_mode()
    def test_real_public_create_versions_original_slab_without_changing_aliases(self):
        state = state_for("cpu", prefixes=(126976,), lengths=(4096,))
        plans = []
        with patch.dict(os.environ, {ENV: "1", "DSV41_CED_VECTOR_LAYOUT": "1"}), public_ced_cpu_scope(state) as ced:
            original = state.identity_context()
            for versioned in (False, True):
                with self.subTest(versioned=versioned), patch.object(
                    ced, "_upload_layout_arrays", wraps=ced._upload_layout_arrays,
                ) as upload:
                    # This deliberately calls the real inference-decorated
                    # public entry, including selected-row and slab creation.
                    plan = ced.CEDPlan.create(
                        state.v4, original, state.attn, state.cache.native,
                        bounded_replay=True, bounded_replay_keep_tokens=140,
                        versioned_metadata=versioned,
                    )
                self.assertIsNotNone(plan)
                self.assertEqual(upload.call_count, 1)
                context, exchange = plan.context, plan.exchange
                fields = (context.global_positions, context.req_id_per_token, context.local_is_real,
                          context.unpad_restore, context.gather_restore_positions,
                          exchange.send_indices, exchange.receive_positions)
                self.assertEqual({value.is_inference() for value in fields}, {not versioned})
                self.assertEqual(plan.cu_seqlens.is_inference(), not versioned)
                self.assertEqual(len({value.untyped_storage().data_ptr() for value in fields}), 1,
                                 "all original typed layout views must retain the single slab")
                self.assertIsNotNone(exchange._layout_host_buffer)
                self.assertEqual(exchange.candidate_rows_host, tuple(range(3956, 4096)))
                self.assertEqual(exchange.projection_groups_host, ((tuple(range(140)), tuple(range(3956, 4096))),))
                plans.append((plan, fields))
            for control, candidate in zip(plans[0][1], plans[1][1]):
                self.assertTrue(torch.equal(control, candidate))
            self.assertTrue(torch.equal(plans[0][0].cu_seqlens, plans[1][0].cu_seqlens))
            mask = plans[1][0].context.local_is_real
            version = mask._version
            mask[0] = False
            self.assertGreater(mask._version, version)

    @torch.inference_mode()
    def test_real_single_layout_certifies_compact_mask_and_rejects_mutation(self):
        from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy

        with patch.dict(os.environ, {ENV: "1", "RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA": "1"}):
            state = state_for("cpu", prefixes=(126976,), lengths=(4096,))
            state.runtime.policy = SharedPrefillPolicy(mode="compact")
            state.v4._propagate_cp_ctx = lambda context: None
            with public_ced_cpu_scope(state):
                state.prepare_plan()
                mask = state.plan.context.local_is_real
                self.assertFalse(mask.is_inference())
                state._activate_compact(torch.empty((140, 1)))
                self.assertIsNone(state.mega_attention_valid_mask(state.context, mask, 140))
                mask[-1] = False
                self.assertIs(state.mega_attention_valid_mask(state.context, mask, 140), mask)

    @torch.inference_mode()
    def test_entry_plan_uses_real_create_once_and_l19_checks_same_metadata(self):
        from rtp_llm.models_py.modules.dsv41.ppu import prefill_runtime as module
        from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
        from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import DENSE_PROJECTIONS

        with patch.dict(os.environ, {ENV: "1", "RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA": "1",
                                     DENSE_PROJECTIONS: "1", "DSV41_CED_VECTOR_LAYOUT": "1"}):
            state = state_for("cpu", prefixes=(126976,), lengths=(4096,))
            state.runtime.policy = SharedPrefillPolicy(mode="compact")
            state.runtime.model._prefill_cp_size = 1
            state.runtime.model.parallelism_config = NS(tp_size=1)
            with public_ced_cpu_scope(state), patch.object(
                module, "bounded_plan", wraps=module.bounded_plan,
            ) as build, patch.object(state, "identity_context", wraps=state.identity_context) as identity:
                state.prepare_local_plan_at_entry()
                plan, original = state.plan, state.original_context
                self.assertIsNotNone(plan)
                self.assertIsNone(state.context, "early preparation must not activate compact rows")
                self.assertEqual(state.executed_layers, [])
                self.assertEqual(state.local_l20_query_rows(4096), slice(3956, 4096))
                # The ordinary CP1 executor propagates None before layers.
                state.bind_context(None)
                state.prepare_local_plan_at_entry()
                args = (object(), object(), object(), object())
                self.assertEqual(state.before_block(NS(layer_id=19), *args), args)
                self.assertIs(state.plan, plan)
                self.assertIs(state.original_context, original)
                self.assertEqual(build.call_count, 1)
                self.assertEqual(identity.call_count, 1)
                state.plan.exchange.send_indices.add_(0)
                with self.assertRaisesRegex(RuntimeError, "metadata changed"):
                    state.prepare_plan()
                self.assertEqual(build.call_count, 1, "changed metadata must not be recertified")

    @torch.no_grad()
    def test_batched_entry_plan_reuses_real_request_metadata_until_l19(self):
        from rtp_llm.models_py.modules.dsv41.ppu import prefill_runtime as module
        from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy

        with patch.dict(os.environ, {**LOOKUP_ENV, "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1"}):
            state = state_for("cpu", prefixes=(126976, 126976), lengths=(4096, 4096))
            state.runtime.policy = SharedPrefillPolicy(mode="compact")
            state.runtime.model._prefill_cp_size = 1
            state.runtime.model.parallelism_config = NS(tp_size=1)
            with public_ced_cpu_scope(state), patch.object(
                module, "bounded_plan", wraps=module.bounded_plan,
            ) as build:
                state.prepare_local_plan_at_entry()
                plan = state.plan
                self.assertIsNotNone(plan)
                self.assertIsNone(state.context)
                self.assertEqual(plan.context.chunk_lengths_per_req, (140, 140))
                state.before_block(NS(layer_id=19), None, None, None, None)
                self.assertIs(state.plan, plan)
                build.assert_called_once()

    @torch.no_grad()
    def test_entry_plan_preserves_other_phase_and_layout_entrypoints(self):
        from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy

        for change in ("flag", "cp", "tp", "ragged", "short", "graph", "verify", "idle",
                       "decode", "sharded", "sg", "full", "encoder_replay", "rollback", "after_layer"):
            with self.subTest(change=change), patch.dict(os.environ, {
                "RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA": "0" if change == "flag" else "1",
            }):
                state = state_for("cpu", prefixes=(126976,), lengths=(4096,))
                state.runtime.policy = SharedPrefillPolicy(mode="compact")
                state.runtime.model._prefill_cp_size = 8 if change == "cp" else 1
                state.runtime.model.parallelism_config = NS(tp_size=8 if change == "tp" else 1)
                if change == "ragged":
                    state.lengths = (2048, 2048)
                elif change == "short":
                    state.lengths = (140,)
                elif change in ("graph", "verify", "idle"):
                    setattr(state, "is_graph" if change == "graph" else change, True)
                elif change == "decode":
                    state.prefill = False
                elif change == "sharded":
                    state.cache.sharded = True
                elif change == "sg":
                    state.local_prefill_sg_align = True
                elif change == "full":
                    state.runtime.policy = SharedPrefillPolicy(mode="full")
                elif change == "encoder_replay":
                    state.runtime.policy = SharedPrefillPolicy(mode="compact", encoder_replay=True)
                elif change == "rollback":
                    state.runtime.policy = SharedPrefillPolicy(mode="compact", max_rollback_tokens=4)
                elif change == "after_layer":
                    state.executed_layers.append(0)
                with patch.object(state, "prepare_plan") as prepare:
                    state.prepare_local_plan_at_entry()
                prepare.assert_not_called()
                self.assertIsNone(state._entry_plan_stamp)

    def test_bounded_plan_birth_certifies_only_unpadded_versioned_compact_rows(self):
        from rtp_llm.models_py.modules.dsv41.ppu import prefill_runtime as module
        from rtp_llm.models_py.modules.dsv41.ppu.ced import _bounded_query_layout
        from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
        from rtp_llm.models_py.modules.dsv41.prefill.ced import CEDPlan, _bounded_replay_selected

        for length in (4096, 129):
            with self.subTest(length=length), torch.inference_mode(), patch.dict(os.environ, {
                ENV: "1", "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1",
                "RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA": "1",
            }):
                state = state_for("cpu", prefixes=(126976,), lengths=(length,))
                state.runtime.policy = SharedPrefillPolicy(mode="compact")
                state.cache.native = NS()
                state.v4._propagate_cp_ctx = lambda context: None
                constructed = []

                def construct_plan(v4, original, attn, native, policy):
                    self.assertFalse(torch.is_inference_mode_enabled())
                    self.assertFalse(torch.is_grad_enabled())
                    starts = (max(length - 140, 0),)
                    selected = _bounded_replay_selected(state.lengths, starts)
                    query, exchange, groups = _bounded_query_layout(original, selected)
                    query.swa_replay_start = starts[0]
                    cu = torch.tensor((0, query.chunk_length), dtype=torch.int32)
                    constructed.append(query.local_is_real)
                    return CEDPlan(original, query, cu, exchange, groups, ()), "compact140"

                with patch.object(module, "permits_ced", return_value=True), \
                        patch.object(module, "bounded_plan", side_effect=construct_plan):
                    state.prepare_plan()
                mask = state.plan.context.local_is_real
                self.assertIs(mask, constructed[0])
                self.assertFalse(mask.is_inference())
                # Before activation, a valid plan alone cannot authorize rows.
                self.assertIs(state.mega_attention_valid_mask(state.plan.context, mask, mask.numel()), mask)
                state._activate_compact(torch.empty(state.plan.context.chunk_length, 1))
                rows = state.rows(state.context)
                self.assertIs(rows[3], mask)
                if length == 129:
                    self.assertIs(state.mega_attention_valid_mask(state.context, mask, mask.numel()), mask)
                    continue
                self.assertIsNone(state.mega_attention_valid_mask(state.context, mask, 140))
                rebound = mask.clone()
                state.context.local_is_real = rebound
                self.assertIs(state.mega_attention_valid_mask(state.context, rebound, 140), rebound)
                state.context.local_is_real = mask
                mask[0] = False
                self.assertIs(state.mega_attention_valid_mask(state.context, mask, 140), mask)

    def test_full_row_slice_keeps_unsupported_and_compact_selectors(self):
        for reason in ("disabled", "rows_disabled", "decode", "idle", "graph", "attn_graph", "sharded", "cp1", "cp8"):
            with self.subTest(reason=reason):
                state = state_for("cpu", prefixes=(17, 64), lengths=(4, 3))
                context = None
                if reason == "decode":
                    state.prefill = False
                elif reason == "idle":
                    state.idle = True
                elif reason == "graph":
                    state.is_graph = True
                elif reason == "attn_graph":
                    state.attn.is_cuda_graph = True
                elif reason == "sharded":
                    state.cache.sharded = True
                elif reason in ("cp1", "cp8"):
                    context = cp1_context(state, compact=True)
                    if reason == "cp8":
                        context = replace(context, cp_size=8, kv_cache_sharded=True)
                with patch.dict(os.environ, {
                    ENV: "0" if reason == "rows_disabled" else "1",
                    LOCAL_PREFILL_FULL_ROWS_ENV: "0" if reason == "disabled" else "1",
                }):
                    actual = state.full_rows(context)
                    expected = legacy_rows(state, context, full=True)
                assert_rows(self, actual, expected)

    def test_optin_does_not_cache_decode_idle_graph_or_cp8(self):
        for reason in ("disabled", "decode", "idle", "graph", "attn_graph", "sharded", "cp8"):
            with self.subTest(reason=reason):
                state = state_for("cpu", prefixes=(17,), lengths=(4,))
                context = None
                if reason == "decode":
                    state.prefill = False
                elif reason == "idle":
                    state.idle = True
                elif reason == "graph":
                    state.is_graph = True
                elif reason == "attn_graph":
                    state.attn.is_cuda_graph = True
                elif reason == "sharded":
                    state.cache.sharded = True
                elif reason == "cp8":
                    context = replace(cp1_context(state), cp_size=8, kv_cache_sharded=True)
                with patch.dict(os.environ, {
                    ENV: "0" if reason == "disabled" else "1", "RTP_V41_PPU_CP8_REUSE_LOCAL_ROWS": "0",
                }):
                    expected = legacy_rows(state, context)
                    assert_rows(self, state.rows(context), expected)
                    assert_rows(self, state.rows(context), expected)
                    state.full_rows(context)
                self.assertEqual(state._local_prefill_rows, {})
                self.assertEqual(state._local_prefill_full_rows, {})


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test",
)
class LocalPrefillRowsDevice(_LocalRowsCases, unittest.TestCase):
    device = "cuda"

    def setUp(self):
        self.assertTrue(is_ppu(), "device qualification requires PPU")

    def test_actual_kv_selection_removes_dynamic_boolean_gather(self):
        import json
        import tempfile
        from pathlib import Path

        state = state_for("cuda", prefixes=(126976,), lengths=(4096,))
        values = torch.arange(4096 * 512, device="cuda").reshape(4096, 512).bfloat16()
        with patch.dict(os.environ, {ENV: "1", LOCAL_PREFILL_FULL_ROWS_ENV: "0"}):
            old = state.full_rows(None)[2]
        with patch.dict(os.environ, {ENV: "1", LOCAL_PREFILL_FULL_ROWS_ENV: "1"}):
            new = state.full_rows(None)[2]
        expected = values[old]
        torch.cuda.synchronize()
        with tempfile.TemporaryDirectory(prefix="local-prefill-full-rows-") as temporary:
            directory = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", temporary))
            for name, selector in (("mask", old), ("slice", new)):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profile:
                    actual = values[selector]
                    torch.cuda.synchronize()
                self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
                path = directory / f"local-prefill-full-rows-{name}.trace.json"
                profile.export_chrome_trace(str(path))
                names = [event["name"] for event in json.loads(path.read_text())["traceEvents"]
                         if event.get("cat") == "cpu_op"]
                for operation in ("aten::index", "aten::nonzero"):
                    self.assertEqual(operation in names, name == "mask")


if __name__ == "__main__":
    unittest.main()
