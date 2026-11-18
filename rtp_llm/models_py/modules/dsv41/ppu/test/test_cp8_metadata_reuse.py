"""CP-native row births and same-stream context/int32-position reuse.

CPU contracts run actual CP row construction, bounded transport metadata,
StepContext construction and the real block0 window check. Only physical
accelerator/stream admission is doubled. The explicit PPU case checks real
device aliases and MegaAttention's unchanged int32 consumer preparation.
"""

from contextlib import ExitStack
import gc
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import weakref

import torch

from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.mega_attention import PPUMegaAttention
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.prefill import build_step_context
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_cross_layer_mhc import (
    CP_ENV, _activate, _plan, _state,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_l20_query_projection import _context
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_metadata import local_metadata_cpu_stream


ENV = dict(CP_ENV, RTP_V41_PPU_CP8_REUSE_LOCAL_ROWS="1",
           RTP_V41_PPU_OPTIMIZED_DIAGNOSTICS="0", RTP_V41_PPU_OPTIMIZED_LATE_FOCUS="0",
           RTP_V41_PPU_PACKED_ATTENTION="0", DSV4_FORWARD_TENSOR_DEBUG="0",
           DSV4_DEBUG_HIDDEN_COMPARE="0")


def _fixture(*, rank=1, lengths=(4096,) * 8, prefixes=(126976,) * 8,
             device="cpu", native_inference=False):
    # inference_mode(False) restores GradMode even inside the caller's
    # no_grad scope. Both native source kinds are prefill inference inputs;
    # preserve no_grad at the actual bind_context birth boundary.
    with torch.inference_mode(native_inference), torch.no_grad():
        cp, ids = _context(rank, lengths, prefixes, device)
        state = _state(cp, ids)
    return state, cp


def _step(state, cp, rows):
    return state.cp_prefill_step_context(cp, tuple(value for value in rows), builder=build_step_context)


def _bytes(value):
    return value.contiguous().view(torch.uint8)


class CP8MetadataReuseContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(patch.dict(os.environ, ENV))
        self.stack.enter_context(patch("rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", return_value=True))
        self.stack.enter_context(patch("torch.cuda.is_current_stream_capturing", return_value=False))
        # Keep the closed-consumer function identities intact. Diagnostics
        # admission is controlled through the real environment above.
        self.stack.enter_context(patch(
            "rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata._upload",
            side_effect=lambda values, device, dtype: torch.tensor(values, device=device, dtype=dtype)))
        # Authenticate the simulated source stream at bind_context itself,
        # before any row/result consumer has a chance to construct metadata.
        self.stack.enter_context(local_metadata_cpu_stream())

    def test_real_full_and_compact_native_rows_context_cast_and_private_versions(self):
        for native_inference in (False, True):
            state, cp = _fixture(native_inference=native_inference)
            with local_metadata_cpu_stream():
                for compact in (False, True):
                    if compact:
                        _activate(state, _plan(state))
                        cp = state.context
                    rows = state.rows(cp)
                    self.assertEqual(rows[1].dtype, torch.int64)
                    self.assertEqual(cp.req_id_per_token.dtype, torch.int32)
                    self.assertFalse(rows[1].is_inference())
                    self.assertFalse(rows[2].is_inference())
                    self.assertFalse(torch._C._is_alias_of(rows[1], cp.req_id_per_token))
                    expected = build_step_context(*rows[:3], valid_mask=rows[3])
                    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("no readback")), patch.object(
                        torch.Tensor, "item", side_effect=AssertionError("no scalar synchronization")
                    ):
                        actual = _step(state, cp, rows)
                        cast = state.cp_prefill_positions32(cp, rows)
                        self.assertIsNotNone(actual)
                        self.assertIsNotNone(cast)
                        self.assertIs(state.rows(cp), rows)
                        self.assertIs(_step(state, cp, tuple(rows)), actual)
                        self.assertIs(state.cp_prefill_positions32(cp, tuple(rows)), cast)
                    for field in ("positions", "valid_mask", "request_ids", "owner_ranks", "request_offsets"):
                        self.assertTrue(torch.equal(_bytes(getattr(actual, field)), _bytes(getattr(expected, field))), field)
                    self.assertTrue(torch._C._is_alias_of(actual.positions, rows[0]))
                    self.assertIs(actual.valid_mask, rows[3])
                    self.assertTrue(torch._C._is_alias_of(actual.request_ids, rows[1]))
                    self.assertFalse(actual.owner_ranks.is_inference())
                    self.assertFalse(actual.request_offsets.is_inference())
                    self.assertFalse(cast.is_inference())
                    self.assertEqual(cast.dtype, torch.int32)
                    self.assertTrue(torch.equal(cast, rows[0].int()))
                    self.assertFalse(any(torch._C._is_alias_of(cast, value) for value in rows))
                stats = state._cp_metadata_evidence
                self.assertEqual((stats["row_constructs"], stats["step_constructs"], stats["positions32_constructs"]),
                                 (2, 2, 2))
                self.assertEqual((stats["step_hits"], stats["positions32_hits"]), (2, 2))
                self.assertIs(state.runtime._cp_metadata_evidence, stats)
                self.assertFalse(state._mega_all_live_phase(), "the original DP/B1 guard stays local")

    def test_l19_future_plan_does_not_change_full_epoch_rank0_compact_keeps_mask(self):
        state, cp = _fixture(rank=0)
        with local_metadata_cpu_stream():
            rows = state.rows(cp)
            full = _step(state, cp, rows)
            cast = state.cp_prefill_positions32(cp, rows)
            plan = _plan(state)
            state.plan, state.plan_reason = plan, "compact140"
            self.assertIs(_step(state, cp, state.rows(cp)), full)
            self.assertIs(state.cp_prefill_positions32(cp, rows), cast)
            _activate(state, plan)
            compact = state.context
            query_rows = state.rows(compact)
            current = _step(state, compact, query_rows)
            self.assertIsNotNone(current)
            self.assertIsNot(current, full)
            self.assertIs(current.valid_mask, compact.local_is_real)
            self.assertEqual(sum(plan.exchange.receive_sizes), 112)
            self.assertEqual(compact.chunk_length, 144)
            self.assertIsNone(state.cp_prefill_positions32(compact, query_rows))
            self.assertIs(state.mega_attention_valid_mask(compact, query_rows[3], 144), query_rows[3])
            self.assertIs(_step(state, compact, state.rows(compact)), current)
            self.assertFalse(state._cp_metadata_invalid)
            self.assertIsNone(_step(state, cp, rows), "full metadata cannot authorize a compact consumer")

    def test_local_invalid_does_not_poison_closed_cp_consumer_and_unknown_row_cache_is_not_a_birth(self):
        from rtp_llm.models_py.modules.dsv41.ppu import attention_layer

        state, cp = _fixture()
        attn = attention_layer.SharedPPUAttention.__new__(attention_layer.SharedPPUAttention)
        torch.nn.Module.__init__(attn)
        attn.ops, attn.runtime = PPUOps.__new__(PPUOps), state.runtime
        state._local_metadata_invalid = True
        self.assertTrue(attention_layer._local_step_context_consumer(attn))
        with local_metadata_cpu_stream():
            rows = state.rows(cp)
            first = attention_layer._local_step_context(attn, state, cp, rows)
            self.assertIs(attention_layer._local_step_context(attn, state, cp, rows), first)
            self.assertEqual(state._cp_metadata_evidence["step_constructs"], 1)
            self.assertTrue(state._local_metadata_invalid)
        other, context = _fixture()
        with local_metadata_cpu_stream():
            values = other.rows(context)
            other._cp_row_records.clear()
            # This is the old CP cache hit, not a real long()/cu construction.
            self.assertIs(other.rows(context), values)
            self.assertIsNone(_step(other, context, values))
            self.assertIsNone(other.cp_prefill_positions32(context, values))
            self.assertFalse(other._cp_row_records)

    def test_native_and_derived_source_mutation_rebind_and_unknown_private_owners_fail_closed(self):
        for change in ("native", "long", "cu", "row_clone", "inference_clone", "compact_map", "compact_plan"):
            state, cp = _fixture(native_inference=False)
            with local_metadata_cpu_stream():
                if change.startswith("compact"):
                    _activate(state, _plan(state))
                    cp = state.context
                rows = state.rows(cp)
                self.assertIsNotNone(_step(state, cp, rows))
                actual = rows
                if change == "native":
                    cp.req_id_per_token.add_(0)
                elif change == "long":
                    rows[1].add_(0)
                elif change == "cu":
                    rows[2].add_(0)
                elif change in ("row_clone", "inference_clone"):
                    with torch.inference_mode(change == "inference_clone"):
                        replacement = rows[1].clone()
                    actual = (rows[0], replacement, *rows[2:])
                elif change == "compact_map":
                    state.plan.exchange.receive_positions.add_(0)
                else:
                    state.plan.context.local_is_real = state.plan.context.local_is_real.clone()
                self.assertIsNone(_step(state, cp, actual))
                self.assertIsNone(state.cp_prefill_positions32(cp, rows))
                self.assertTrue(state._cp_metadata_invalid)
                self.assertEqual(state._cp_metadata_evidence["invalidations"], 1)

    def test_unknown_native_source_stream_is_not_certified_by_first_row_consumer(self):
        state, cp = _fixture()
        original = state._cp_metadata_stream
        self.assertIsNotNone(original)
        state._cp_metadata_stream = None
        rows = state.rows(cp)
        self.assertIsNone(_step(state, cp, rows))
        self.assertFalse(state._cp_row_records)
        state._cp_metadata_stream = original
        self.assertIsNone(state.cp_prefill_positions32(cp, rows))
        self.assertTrue(state._cp_metadata_invalid)

    def test_result_mutation_rebind_private_cast_and_stream_changes_are_sticky(self):
        for change in ("owner_write", "offset_write", "cast_write", "position_clone", "mask_clone",
                       "inference_owner", "parallel", "stream", "device", "capture", "current"):
            state, cp = _fixture()
            with local_metadata_cpu_stream() as stream:
                rows = state.rows(cp)
                context = _step(state, cp, rows)
                cast = state.cp_prefill_positions32(cp, rows)
                constructs = dict(state._cp_metadata_evidence)
                if change == "owner_write":
                    context.owner_ranks.add_(0)
                elif change == "offset_write":
                    context.request_offsets.add_(0)
                elif change == "cast_write":
                    cast.add_(0)
                elif change in ("position_clone", "mask_clone"):
                    field = "positions" if change == "position_clone" else "valid_mask"
                    object.__setattr__(context, field, getattr(context, field).clone())
                elif change == "inference_owner":
                    with torch.inference_mode():
                        replacement = context.owner_ranks.clone()
                    object.__setattr__(context, "owner_ranks", replacement)
                elif change == "parallel":
                    object.__setattr__(context, "parallel_context", object())
                elif change == "stream":
                    stream.handle += 1
                elif change == "device":
                    stream.device = torch.device("cuda:1")
                elif change == "capture":
                    stream.capture = True
                else:
                    state.runtime.current = None
                self.assertIsNone(state.cp_prefill_positions32(cp, rows))
                self.assertIsNone(_step(state, cp, rows))
                self.assertEqual(state._cp_metadata_evidence["step_constructs"], constructs["step_constructs"])
                self.assertEqual(state._cp_metadata_evidence["positions32_constructs"], constructs["positions32_constructs"])
                stream.handle, stream.device, stream.capture = 19, None, False
                state.runtime.current = state
                self.assertIsNone(_step(state, cp, rows))

    def test_builder_changes_errors_and_midconstruction_mutation_never_retry(self):
        for change in ("identity", "error", "source", "stream"):
            state, cp = _fixture()
            with local_metadata_cpu_stream() as stream:
                rows = state.rows(cp)
                calls = []

                def builder(*args, **kwargs):
                    calls.append(1)
                    if change == "error":
                        raise RuntimeError("CP constructor submitted once")
                    value = build_step_context(*args, **kwargs)
                    if change == "source":
                        rows[2].add_(0)
                    elif change == "stream":
                        stream.handle += 1
                    return value

                if change == "error":
                    with self.assertRaisesRegex(RuntimeError, "submitted once"):
                        state.cp_prefill_step_context(cp, rows, builder=builder)
                else:
                    actual = state.cp_prefill_step_context(cp, rows, builder=builder)
                    self.assertIsNotNone(actual, "completed construction returns once, without fallback replay")
                    if change == "identity":
                        self.assertIsNone(_step(state, cp, rows))
                self.assertIsNone(state.cp_prefill_step_context(cp, rows, builder=builder))
                self.assertEqual(calls, [1])

    def test_entries_keep_inflight_result_buffers_without_retaining_the_forward(self):
        state, cp = _fixture()
        runtime = state.runtime
        with local_metadata_cpu_stream():
            rows = state.rows(cp)
            context = _step(state, cp, rows)
            cast = state.cp_prefill_positions32(cp, rows)
            original = context.owner_ranks
            old = weakref.ref(original)
            object.__setattr__(context, "owner_ranks", original.clone())
            del original
            gc.collect()
            self.assertIsNotNone(old(), "the in-flight original owner must survive a field rebind")
            self.assertIsNone(_step(state, cp, rows))
        evidence = runtime._cp_metadata_evidence
        owner = weakref.ref(state)
        runtime.current = None
        del state
        gc.collect()
        self.assertIsNone(owner(), "metadata result/receipt must not introduce a strong state cycle")
        self.assertIsInstance(evidence, dict)
        self.assertIsNotNone(context)
        self.assertIsNotNone(cast)
        self.assertIsNone(old(), "old private owner releases with the forward")

    def test_block0_all_live_uses_exact_native_mask_and_preserves_unknown_layout_fallback(self):
        for change in ("none", "stride", "mutated_mask", "wrong_shape", "image", "mismatch"):
            state, cp = _fixture(rank=0, lengths=(256,) * 8, prefixes=(0,) * 8)
            ids = state.inputs.input_ids
            state.window_current_tokens = ids.clone()
            if change == "stride":
                backing = torch.zeros(ids.numel() * 2, dtype=ids.dtype)
                backing[::2] = ids
                ids = backing[::2]
            elif change == "mutated_mask":
                cp.local_is_real.logical_not_().logical_not_()
            elif change == "wrong_shape":
                ids = ids[:, None]
            elif change == "image":
                state.v4._image_token_mask = torch.ones_like(ids, dtype=torch.bool)
                state.runtime.model._image_plan = object()
                state.runtime.model._engram_input_ids = Mock(side_effect=lambda value, mask: value + 7)
                state.window_current_tokens = ids + 7
            elif change == "mismatch":
                state.window_current_tokens = ids + 1
            indexing = []
            original_getitem = torch.Tensor.__getitem__

            def getitem(value, key):
                if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
                    indexing.append(1)
                return original_getitem(value, key)

            with patch.object(torch.Tensor, "__getitem__", new=getitem):
                if change in ("wrong_shape", "mismatch"):
                    with self.assertRaisesRegex(ValueError, "real CP Engram windows disagree"):
                        state.before_block(NS(layer_id=0), None, ids, cp.global_positions, state.attn.cu_seqlens)
                else:
                    state.before_block(NS(layer_id=0), None, ids, cp.global_positions, state.attn.cu_seqlens)
            self.assertEqual(len(indexing), 0 if change in ("none", "image", "mismatch") else 2)
            self.assertEqual(state._cp_metadata_evidence["window_all_live_checks"],
                             int(change in ("none", "image", "mismatch")))
            if change == "image":
                state.runtime.model._engram_input_ids.assert_called_once()

    def test_native_window_columns_preserve_legacy_values_offsets_and_image_comparison_without_gathers(self):
        for layout in ("native_column", "column_offset", "row_stride", "int32_column"):
            for change in ("equal", "mismatch", "image"):
                with self.subTest(layout=layout, change=change):
                    state, cp = _fixture(rank=0, lengths=(256,) * 8, prefixes=(0,) * 8)
                    ids = state.inputs.input_ids
                    count = ids.numel()
                    dtype = torch.int32 if layout == "int32_column" else torch.int64
                    backing = torch.full((count * 2 + 1, 5) if layout == "row_stride" else (count, 3),
                                         -17, dtype=dtype)
                    window = (backing[1::2, 2] if layout == "row_stride" else
                              backing[:, 1 if layout == "column_offset" else 0])
                    self.assertEqual(window.stride(), (10,) if layout == "row_stride" else (3,))
                    self.assertFalse(window.is_contiguous())
                    if layout in ("column_offset", "row_stride"):
                        self.assertGreater(window.storage_offset(), 0)
                    comparison = ids + 7 if change == "image" else ids
                    window.copy_(comparison)
                    if change == "mismatch":
                        window[count // 2] += 1
                    if change == "image":
                        state.v4._image_token_mask = torch.ones_like(ids, dtype=torch.bool)
                        state.runtime.model._image_plan = object()
                        state.runtime.model._engram_input_ids = Mock(side_effect=lambda value, mask: value + 7)
                    state.window_current_tokens = window
                    # This is the exact old ATen expression, evaluated before
                    # counting gathers in the actual Block0 boundary below.
                    expected = torch.equal(window[cp.local_is_real].long(), comparison[cp.local_is_real].long())
                    self.assertEqual(expected, change != "mismatch")
                    original = backing.clone()
                    original_window_key = (window.data_ptr(), window.stride(), window.storage_offset(), window._version)
                    original_getitem = torch.Tensor.__getitem__
                    gathers = []

                    def getitem(value, key):
                        if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
                            gathers.append(1)
                        return original_getitem(value, key)

                    with patch.object(torch.Tensor, "__getitem__", new=getitem):
                        if expected:
                            result = state.before_block(NS(layer_id=0), None, ids,
                                                        cp.global_positions, state.attn.cu_seqlens)
                            self.assertIs(result[1], ids)
                        else:
                            with self.assertRaisesRegex(ValueError, "real CP Engram windows disagree"):
                                state.before_block(NS(layer_id=0), None, ids,
                                                   cp.global_positions, state.attn.cu_seqlens)
                    self.assertEqual(gathers, [])
                    self.assertEqual(state._cp_metadata_evidence["window_all_live_checks"], 1)
                    self.assertIs(state.window_current_tokens, window)
                    self.assertEqual((window.data_ptr(), window.stride(), window.storage_offset(), window._version),
                                     original_window_key)
                    self.assertTrue(torch.equal(backing, original))
                    if change == "image":
                        state.runtime.model._engram_input_ids.assert_called_once_with(ids, state.v4._image_token_mask)

    def test_unknown_window_layouts_and_native_cp_padding_keep_legacy_masked_checks(self):
        for change in ("broadcast", "float", "wrong_shape", "cp_padding"):
            with self.subTest(change=change):
                state, cp = _fixture(rank=0, lengths=((17,) * 8 if change == "cp_padding" else (256,) * 8),
                                     prefixes=(0,) * 8)
                ids = state.inputs.input_ids
                backing = torch.full((ids.numel(), 3), -17, dtype=torch.int64)
                window = backing[:, 0]
                window.copy_(ids)
                if change == "broadcast":
                    window = ids[:1].expand_as(ids)
                elif change == "float":
                    window = backing.float()[:, 0]
                elif change == "wrong_shape":
                    window = window[:, None]
                else:
                    window[~cp.local_is_real] = -999
                    self.assertFalse(state._cp_full_rows_birth[2])
                state.window_current_tokens = window
                expected = torch.equal(window[cp.local_is_real].long(), ids[cp.local_is_real].long())
                original_getitem = torch.Tensor.__getitem__
                gathers = []

                def getitem(value, key):
                    if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
                        gathers.append(1)
                    return original_getitem(value, key)

                with patch.object(torch.Tensor, "__getitem__", new=getitem):
                    if expected:
                        state.before_block(NS(layer_id=0), None, ids, cp.global_positions, state.attn.cu_seqlens)
                    else:
                        with self.assertRaisesRegex(ValueError, "real CP Engram windows disagree"):
                            state.before_block(NS(layer_id=0), None, ids, cp.global_positions, state.attn.cu_seqlens)
                self.assertEqual(len(gathers), 2)
                self.assertEqual(state._cp_metadata_evidence["window_all_live_checks"], 0)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device alias test")
class CP8MetadataReuseDeviceAliases(unittest.TestCase):
    @torch.inference_mode()
    def test_native_stride3_window_column_matches_the_original_masked_value_check(self):
        self.assertTrue(is_ppu())
        with patch.dict(os.environ, ENV):
            for column in (0, 1):
                state, cp = _fixture(rank=0, device="cuda:0", native_inference=True)
                ids = state.inputs.input_ids
                windows = torch.full((ids.numel(), 3), -17, dtype=torch.int64, device=ids.device)
                window = windows[:, column]
                window.copy_(ids)
                state.window_current_tokens = window
                self.assertEqual(window.stride(), (3,))
                self.assertEqual(window.storage_offset(), column)
                self.assertTrue(torch.equal(window[cp.local_is_real].long(), ids[cp.local_is_real].long()))
                state.before_block(NS(layer_id=0), None, ids, cp.global_positions, state.attn.cu_seqlens)
                self.assertEqual(state._cp_metadata_evidence["window_all_live_checks"], 1)
                self.assertTrue(torch.equal(window, ids))
                window[ids.numel() // 2] += 1
                self.assertFalse(torch.equal(window[cp.local_is_real].long(), ids[cp.local_is_real].long()))
                with self.assertRaisesRegex(ValueError, "real CP Engram windows disagree"):
                    state.before_block(NS(layer_id=0), None, ids, cp.global_positions, state.attn.cu_seqlens)
                self.assertEqual(state._cp_metadata_evidence["window_all_live_checks"], 2)

    @torch.inference_mode()
    def test_actual_device_native_aliases_private_cast_and_unchanged_mega_int32_consumer(self):
        self.assertTrue(is_ppu())
        with patch.dict(os.environ, ENV):
            for rank in (0, 1):
                state, cp = _fixture(rank=rank, device="cuda:0", native_inference=True)
                for compact in (False, True):
                    if compact:
                        _activate(state, _plan(state))
                        cp = state.context
                    rows = state.rows(cp)
                    context = _step(state, cp, rows)
                    self.assertIsNotNone(context)
                    self.assertTrue(torch._C._is_alias_of(context.positions, cp.global_positions))
                    self.assertIs(context.valid_mask, cp.local_is_real)
                    self.assertTrue(torch._C._is_alias_of(context.request_ids, rows[1]))
                    self.assertFalse(context.owner_ranks.is_inference())
                    cast = state.cp_prefill_positions32(cp, rows)
                    if compact and rank == 0:
                        self.assertIsNone(cast)
                        self.assertIs(state.mega_attention_valid_mask(cp, rows[3], 144), rows[3])
                        continue
                    self.assertIsNotNone(cast)
                    self.assertFalse(cast.is_inference())
                    self.assertFalse(torch._C._is_alias_of(cast, cp.global_positions))
                    native = NS(cos_sin_cache=torch.empty(0, device="cuda:0"), groups=8,
                                softmax_scale=1.0, sink=torch.empty(0, device="cuda:0"))
                    common = PPUMegaAttention._common(native, torch.empty(0, device="cuda:0"), cast, None)
                    self.assertIs(common["token_positions"], cast, "the unchanged Mega ABI must borrow int32 storage")
                    self.assertTrue(torch.equal(cast, cp.global_positions.to(torch.int32)))
                    self.assertIs(state.cp_prefill_positions32(cp, state.rows(cp)), cast)
                    self.assertIs(_step(state, cp, rows), context)
                torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
