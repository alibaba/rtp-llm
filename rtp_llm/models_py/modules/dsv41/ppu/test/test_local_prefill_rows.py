"""Per-forward local Prefill metadata, with the legacy path as a value oracle.

Native input/CP metadata is read-only during one forward. Ordinary tensor
versions additionally protect changed fixtures; inference-mode buffers may be
updated between forwards, each of which owns a fresh metadata cache.
"""

import os
import unittest
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
