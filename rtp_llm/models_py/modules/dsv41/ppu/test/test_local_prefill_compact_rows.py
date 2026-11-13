"""CP1 selected rows against the frozen scatter-then-mask route.

Register this file plus test_local_prefill_rows.py, test_local_prefill_fused.py
and the existing PTPC fixture data. PPU assertions require the explicit local
workspace device switch and an actual PPU; these tests do not run a model.
"""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import PropertyMock, patch

import torch

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.modules.dsv41 import cp as cp_module
from rtp_llm.models_py.modules.dsv41.ppu import attention_layer
from rtp_llm.models_py.modules.dsv41.ppu import ced as ppu_ced
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.ced import _bounded_query_layout
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import (
    LOCAL_PREFILL_COMPACT_ROWS_ENV,
    LOCAL_PREFILL_ROWS_ENV,
)
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import _NativeFixture, assert_bytes
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_rows import state_for
from rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_prefill_reuse import _ProjectionBoundary, original_norm, prefill_component
from rtp_llm.models_py.modules.dsv41.prefill.ced import CEDPlan, _bounded_replay_selected
from rtp_llm.models_py.modules.dsv41.prefill import ced as public_ced


ENV = LOCAL_PREFILL_COMPACT_ROWS_ENV


def compact_state(device, prefixes=(17,), lengths=(4096,)):
    state = state_for(device, prefixes, lengths)
    state.runtime.policy = SharedPrefillPolicy(mode="compact")
    state.v4._propagate_cp_ctx = lambda context: None
    original = state.identity_context()
    starts = tuple(max(n - 140, 0) for n in lengths)
    selected = _bounded_replay_selected(lengths, starts)
    query, exchange, groups = _bounded_query_layout(original, selected)
    if len(starts) == 1:
        query.swa_replay_start = starts[0]
    else:
        query.swa_replay_starts_host = starts
    cu = [0]
    for size in query.chunk_lengths_per_req:
        cu.append(cu[-1] + size)
    state.original_context, state.plan_reason = original, "compact140"
    state.plan = CEDPlan(original, query, torch.tensor(cu, device=device, dtype=torch.int32), exchange, groups, ())
    # Exercise the actual trusted activation, not a test-created proof token.
    with patch.dict(os.environ, {ENV: "1"}):
        state._activate_compact(torch.empty(query.chunk_length, 1, device=device))
    return state


def rows_payload(state, width=512, dtype=torch.bfloat16, shift=0):
    rows = state.context.chunk_length
    values = (torch.arange(rows * width, device=state.inputs.input_ids.device) + shift).remainder(127)
    return values.to(dtype).reshape(rows, width)


def old_rows(state, values):
    with patch.dict(os.environ, {ENV: "0", LOCAL_PREFILL_ROWS_ENV: "0"}):
        positions, requests, valid = state.full_rows(state.context)
        restored = cp_module._cp_restore_gathered_full_2d(values, state.context)
    return restored[valid], positions[valid], requests[valid]


def selected_rows(state, values):
    with patch.dict(os.environ, {ENV: "1"}):
        positions, requests, valid = state.full_rows(state.context, selected_only=True)
        return values.index_select(0, state.context.unpad_restore)[valid], positions[valid], requests[valid]


class _CompactRowsCases:
    def test_descriptions_distinguish_compact_query_selected_and_full_domains(self):
        from rtp_llm.models_py.modules.dsv41.ppu.row_metadata import RowLayout, SelectorKind

        with patch.dict(os.environ, {ENV: "1", LOCAL_PREFILL_ROWS_ENV: "1"}):
            state = compact_state(self.device)
            proof = state._local_compact_row_proof
            query = state.row_view(state.context).validate()
            selected = state.full_row_view(state.context, selected_only=True).validate()
            full = state.full_row_view(state.context).validate()
            self.assertIs(query.layout, RowLayout.COMPACT_QUERY)
            self.assertIs(selected.layout, RowLayout.SELECTED)
            self.assertIs(full.layout, RowLayout.FULL)
            self.assertIs(selected.as_full_rows(), state.full_rows(state.context, selected_only=True))
            self.assertIs(selected.selector.kind, SelectorKind.ALL)
            self.assertIs(full.selector.kind, SelectorKind.MASK)
            self.assertFalse(selected.selector.belongs_to(full))
            state.context.global_positions.add_(1)
            self.assertFalse(state.use_local_compact_rows(state.context))
            # Description construction cannot replace the activation-born proof.
            state.row_view(state.context)
            state.full_row_view(state.context, selected_only=True)
            self.assertIs(state._local_compact_row_proof, proof)
            self.assertFalse(state.use_local_compact_rows(state.context))

    def test_single_layout_matches_old_admission_transport_and_restore(self):
        for prefix, length in ((0, 129), (17, 140), (126976, 4096), (31, 4097)):
            with self.subTest(prefix=prefix, length=length):
                state = state_for(self.device, (prefix,), (length,))
                context = state.identity_context()
                table = torch.zeros((1, 32), dtype=torch.int32, device=self.device)
                base = torch.empty((2, 136 * 528), dtype=torch.uint8, device=self.device)
                native = NS(group_tags=(public_ced.DECODER_SWA_KV,),
                            get_seq_size_per_block=lambda tag: 256,
                            get_entries_per_block=lambda tag: 136,
                            get_layer_cache=lambda layer, tag: NS(kv_cache_base=base))
                model = NS(capture_aux_hidden_layer_ids=())
                attn = NS(is_prefill=True)
                with ExitStack() as stack:
                    stack.enter_context(patch.object(public_ced, "_supported_model", return_value=True))
                    stack.enter_context(patch.object(public_ced, "as_attention_inputs_by_tag", return_value={
                        str(public_ced.DECODER_SWA_KV): NS(kv_cache_block_id=table)}))
                    stack.enter_context(patch.object(collective_torch, "_get_group", return_value=None))
                    stack.enter_context(patch.object(torch.distributed, "is_initialized", return_value=True))
                    stack.enter_context(patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))
                    if self.device == "cpu":
                        stack.enter_context(patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True))
                    plans = []
                    for enabled in ("0", "1"):
                        with patch.dict(os.environ, {"RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA": enabled}), patch.object(
                            public_ced, "_upload_layout_arrays", wraps=public_ced._upload_layout_arrays,
                        ) as uploads:
                            plan, reason = ppu_ced.bounded_plan(model, context, attn, native, SharedPrefillPolicy(mode="compact"))
                        plans.append((plan, reason))
                        self.assertEqual(uploads.call_count, 2 if enabled == "0" and length > 140 else 1)
                    self.assertEqual(plans[0][1], plans[1][1])
                    old, new = plans[0][0], plans[1][0]
                    if old is None:
                        self.assertIsNone(new)
                        continue
                    self.assertEqual(new.context.swa_replay_start, length - 140)
                    self.assertIsNone(new.exchange.group)
                    for name in ("global_positions", "relative_positions", "req_id_per_token", "local_is_real", "unpad_restore", "gather_restore_positions"):
                        assert_bytes(self, getattr(new.context, name), getattr(old.context, name))
                    assert_bytes(self, new.cu_seqlens, old.cu_seqlens)
                    for name in ("send_indices", "receive_positions"):
                        assert_bytes(self, getattr(new.exchange, name), getattr(old.exchange, name))
                    for name in ("send_sizes", "receive_sizes", "candidate_rows_host", "projection_groups_host", "receive_is_identity"):
                        self.assertEqual(getattr(new.exchange, name), getattr(old.exchange, name))
                    value = torch.arange(length * 3, device=self.device).reshape(length, 3)
                    assert_bytes(self, new.exchange.compact(value), old.exchange.compact(value))
                    assert_bytes(self, new.restore_rows(new.exchange.compact(value)), old.restore_rows(old.exchange.compact(value)))
                    with patch.dict(os.environ, {"RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA": "1"}), patch.object(
                        public_ced, "_supported_model", return_value=False,
                    ):
                        self.assertIsNone(ppu_ced.bounded_plan(model, context, attn, native, SharedPrefillPolicy(mode="compact"))[0])

    def test_host_geometry_reuse_keeps_live_descriptor_invalidation(self):
        with torch.inference_mode(False), patch.dict(os.environ, {
            ENV: "1", "RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA": "1",
        }):
            state = compact_state(self.device)
            self.assertTrue(state.use_local_compact_rows(state.context))
            saved = state._local_compact_host_geometry
            self.assertTrue(state.use_local_compact_rows(state.context))
            self.assertIs(saved, state._local_compact_host_geometry)
            state.context.global_positions.copy_(state.context.global_positions.clone())
            self.assertFalse(state.use_local_compact_rows(state.context))

    def test_real_bounded_order_ragged_odd_padding_empty_and_last_rows_match_old(self):
        for prefixes, lengths in (
            ((126976,), (4096,)), ((0, 31, 129), (4096, 139, 0)),
            ((17, 64, 9), (4096, 7, 1)), ((5, 0), (141, 3)),
        ):
            with self.subTest(lengths=lengths):
                state = compact_state(self.device, prefixes, lengths)
                with patch.dict(os.environ, {ENV: "1"}):
                    self.assertTrue(state.use_local_compact_rows(state.context))
                for width, dtype in ((512, torch.bfloat16), (528, torch.uint8), (512, torch.float32)):
                    values = rows_payload(state, width, dtype)
                    values[~state.context.local_is_real] = 255 if dtype == torch.uint8 else -999
                    for actual, expected in zip(selected_rows(state, values), old_rows(state, values)):
                        assert_bytes(self, actual, expected)
                _, positions, requests = selected_rows(state, rows_payload(state))
                for rid, (prefix, length) in enumerate(zip(prefixes, lengths)):
                    if length:
                        self.assertEqual(int(positions[requests == rid][-1]), prefix + length - 1)
                with patch.dict(os.environ, {ENV: "1"}):
                    first = state.full_rows(state.context, selected_only=True)
                    repeated = state.full_rows(state.context, selected_only=True)
                self.assertTrue(all(left is right for left, right in zip(first, repeated)))
                self.assertEqual(first[2], slice(None))

    def test_activation_versions_invalidate_before_and_after_first_consumer(self):
        for name in ("global_positions", "req_id_per_token", "local_is_real", "unpad_restore", "gather_restore_positions"):
            for consume in (False, True):
                with self.subTest(name=name, consume=consume), torch.inference_mode(False), patch.dict(os.environ, {ENV: "1"}):
                    state = compact_state(self.device)
                    self.assertTrue(state.use_local_compact_rows(state.context))
                    if consume:
                        state.full_rows(state.context, selected_only=True)
                    tensor = getattr(state.context, name)
                    tensor.copy_(tensor.clone())  # Version change without changing numerical values.
                    self.assertFalse(state.use_local_compact_rows(state.context))
        with torch.inference_mode(False), patch.dict(os.environ, {ENV: "1"}):
            state = compact_state(self.device)
            state.context.gather_restore_positions[:] = state.context.gather_restore_positions.flip(0)
            self.assertFalse(state.use_local_compact_rows(state.context), "an unsorted mutated selection cannot use compact order")
            fresh = compact_state(self.device)
            self.assertTrue(fresh.use_local_compact_rows(fresh.context))

    def test_fresh_inference_forward_and_mutable_swa_values_never_reuse_payload(self):
        # Inference metadata is immutable inside a native forward. New forwards
        # may reuse source addresses/contents and still own distinct proofs.
        with torch.inference_mode(), patch.dict(os.environ, {ENV: "1"}):
            first = compact_state(self.device)
            first_meta = first.full_rows(first.context, selected_only=True)
            values = rows_payload(first)
            before = selected_rows(first, values)[0]
            values.add_(1)
            after = selected_rows(first, values)[0]
            self.assertFalse(torch.equal(before, after))
            for actual, expected in zip(selected_rows(first, values), old_rows(first, values)):
                assert_bytes(self, actual, expected)
            second = compact_state(self.device, prefixes=(99,), lengths=(4096,))
            self.assertTrue(second.use_local_compact_rows(second.context))
            second_meta = second.full_rows(second.context, selected_only=True)
            self.assertIsNot(first_meta[0], second_meta[0])
            self.assertEqual(int(second_meta[0][-1]), 99 + 4096 - 1)
            self.assertIsNot(first._local_compact_row_proof, second._local_compact_row_proof)

    def test_native_swa_cache_and_global_sources_keep_identical_bytes(self):
        prefixes, lengths = (129, 31, 77), (4096, 7, 1)
        state = compact_state(self.device, prefixes, lengths)
        payload = rows_payload(state)
        legacy, current = old_rows(state, payload), selected_rows(state, payload)
        for ratio in (0, 1, 2):
            with self.subTest(ratio=ratio):
                left = _NativeFixture(self.device, ratio=ratio, prefixes=prefixes, lengths=lengths)
                right = _NativeFixture(self.device, ratio=ratio, prefixes=prefixes, lengths=lengths)
                for fixture, (values, positions, requests) in ((left, legacy), (right, current)):
                    fixture.cache.stage_swa(
                        1, positions, requests, encode_kv(values, pool_tag="swa_kv"), prefixes,
                        retain_prefix_tail=False,
                        bounded_starts=tuple(p + max(n - 140, 0) for p, n in zip(prefixes, lengths)),
                    )
                for key in left.native.handles:
                    assert_bytes(self, left.native.handles[key].kv_cache_base, right.native.handles[key].kv_cache_base)
                for rid in range(len(lengths)):
                    positions = current[1][current[2] == rid]
                    assert_bytes(self, left.cache.read_swa(1, rid, positions), right.cache.read_swa(1, rid, positions))
                for source in left.geometry.kv_source_layer_ids:
                    for rid, count in enumerate(left.cache.lengths):
                        assert_bytes(self, left.cache.read_global(source, rid, global_entries=count // ratio),
                                     right.cache.read_global(source, rid, global_entries=count // ratio))

    def _body_staging(self, flag, *, state, value=None, source=False):
        with prefill_component(state.context.chunk_length, indexed=False, device=self.device) as (attn, _, run, _, hidden):
            attn.runtime.current, attn.runtime.policy = state, state.runtime.policy
            attn.runtime.freqs_for = lambda layer: torch.ones(
                max(p + n for p, n in zip(state.query_prefixes, state.lengths)), 1,
                device=self.device, dtype=torch.complex64,
            )
            attn._cp_ctx, attn.layer_id, attn.is_kv_source = state.context, 21, source
            captured = []

            def stage(layer, positions, requests, packed, prefixes, **kwargs):
                captured.append((positions.clone(), requests.clone(), packed.clone(), prefixes, kwargs))
                if source:
                    raise _ProjectionBoundary()

            state.cache.stage_swa = stage
            if value is None:
                value = torch.randn(state.context.chunk_length, hidden, device=self.device, dtype=torch.bfloat16)
            with ExitStack() as stack:
                if self.device == "cpu":
                    # The production norm provider requires CUDA. This CPU row
                    # fixture uses its existing reference; PPU keeps real norms.
                    for norm in (attn.q_norm, attn.kv_norm):
                        stack.enter_context(patch.object(
                            norm, "forward", side_effect=lambda value, norm=norm: original_norm(norm, value),
                        ))
                stack.enter_context(patch.dict(os.environ, {ENV: str(int(flag)), attention_layer.LOCAL_PREFILL_WORKSPACE_ENV: "0"}))
                stack.enter_context(patch.object(attention_layer, "prepare_packed_attention", return_value=(None, "disabled")))
                stack.enter_context(patch.object(attention_layer, "_build_reference_workspace", side_effect=_ProjectionBoundary))
                # No DP/TP/distributed operation may be called for this CP1 route.
                for owner, name in ((attention_layer, "cp_all_gather_full_varlen"),
                                    (attention_layer, "cp_all_gather_full"), (collective_torch, "all_gather"),
                                    (torch.distributed, "all_gather"), (torch.distributed, "all_gather_into_tensor")):
                    stack.enter_context(patch.object(owner, name, side_effect=AssertionError("local CP1 collective")))
                run(value)
            return captured, value, state.shared["n5_routes"][-1]

    def test_actual_attention_stages_same_rows_without_any_collective(self):
        state = compact_state(self.device)
        # Keep the exact same modules/weights between flag variants by seeding
        # fixture construction, as well as reusing the same input tensor.
        torch.manual_seed(851)
        old, value, old_route = self._body_staging(False, state=state)
        torch.manual_seed(851)
        current, _, route = self._body_staging(True, state=state, value=value)
        self.assertEqual(len(old), 1)
        self.assertEqual(len(current), 1)
        for left, right in zip(current[0][:3], old[0][:3]):
            assert_bytes(self, left, right)
        self.assertEqual(current[0][3:], old[0][3:])
        self.assertNotIn("swa_rows_backend", old_route)
        self.assertEqual(route["swa_rows_backend"], "local_compact_selected")
        torch.manual_seed(851)
        producer, _, route = self._body_staging(True, state=state, value=value, source=True)
        self.assertNotIn("swa_rows_backend", route, "KV producer must keep its full scatter route")
        for left, right in zip(producer[0][:3], old[0][:3]):
            assert_bytes(self, left, right)


class LocalPrefillCompactRowsCPU(_CompactRowsCases, unittest.TestCase):
    device = "cpu"

    def test_flag_decode_idle_graph_cp8_exact_unknown_and_host_order_fallback(self):
        with patch.dict(os.environ, {ENV: "1"}):
            for field, value in (("prefill", False), ("idle", True), ("is_graph", True), ("compacted", False)):
                state = compact_state(self.device)
                setattr(state, field, value)
                self.assertFalse(state.use_local_compact_rows(state.context), field)
            for field, value in (("cp_size", 8), ("kv_cache_sharded", True), ("swa_replay_start", None)):
                state = compact_state(self.device)
                setattr(state.context, field, value)
                self.assertFalse(state.use_local_compact_rows(state.context), field)
            state = compact_state(self.device)
            for owner, field, value in ((state.attn, "is_cuda_graph", True), (state.cache, "sharded", True),
                                        (state.runtime, "policy", SharedPrefillPolicy(mode="exact")),
                                        (state.runtime.model, "_is_speculative", True),
                                        (state.plan.exchange, "projection_groups_host", ())):
                with patch.object(owner, field, value, create=True):
                    self.assertFalse(state.use_local_compact_rows(state.context), field)
            self.assertFalse(state.use_local_compact_rows(None))
            with patch.dict(os.environ, {ENV: "0"}):
                self.assertFalse(state.use_local_compact_rows(state.context))


class LocalPrefillHostMetadataCPU(LocalPrefillCompactRowsCPU):
    def setUp(self):
        flags = patch.dict(os.environ, {"RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA": "1"})
        flags.start()
        self.addCleanup(flags.stop)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalPrefillCompactRowsDevice(_CompactRowsCases, unittest.TestCase):
    device = "cuda"

    def setUp(self):
        self.assertTrue(is_ppu(), "compact row qualification requires an actual ZW PPU")

    @torch.inference_mode()
    def test_actual_profile_has_no_full4096_scatter_or_boolean_reselection(self):
        state = compact_state(self.device)
        torch.manual_seed(851)
        _, value, _ = self._body_staging(True, state=state)
        torch.cuda.synchronize()
        torch.manual_seed(851)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                              torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as control:
            old, _, old_route = self._body_staging(False, state=state, value=value)
            torch.cuda.synchronize()
        torch.manual_seed(851)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                              torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            current, _, route = self._body_staging(True, state=state, value=value)
            torch.cuda.synchronize()
        self.assertNotIn("swa_rows_backend", old_route)
        self.assertEqual(route["swa_rows_backend"], "local_compact_selected")
        for staged in (old, current):
            self.assertEqual(len(staged), 1)
            self.assertEqual(staged[0][2].shape, (140, 512), "actual device fixture must stage BF16 KV512")
            self.assertEqual(staged[0][2].dtype, torch.bfloat16)
        for left, right in zip(current[0][:3], old[0][:3]):
            assert_bytes(self, left, right)
        root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        with tempfile.TemporaryDirectory(prefix="compact-rows-") as temporary:
            path = Path(root or temporary) / "local-prefill-compact-rows.trace.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            profile.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
            control_path = path.with_name("local-prefill-compact-rows-control.trace.json")
            control.export_chrome_trace(str(control_path))
            control_events = json.loads(control_path.read_text())["traceEvents"]
        self.assertTrue(any(event.get("cat") == "cpu_op" and event["name"] == "aten::index_copy_"
                            and event.get("args", {}).get("Input Dims", [None])[0] == [4096, 512]
                            for event in control_events), "shape probe must see the control's actual full KV512 scatter")
        self.assertTrue(any(event.get("cat") == "cpu_op" and event["name"] == "aten::index"
                            and event.get("args", {}).get("Input Dims", [None])[0] == [4096, 512]
                            for event in control_events), "shape probe must see the control's actual full KV512 reselection")
        forbidden = [(event["name"], event.get("args", {}).get("Input Dims")) for event in events
                     if event.get("cat") == "cpu_op" and event["name"] in ("aten::index", "aten::index_copy_")
                     and event.get("args", {}).get("Input Dims", [None])[0] == [4096, 512]]
        self.assertEqual(forbidden, [])
        self.assertTrue(any(event.get("cat") == "cpu_op" and event["name"] == "aten::index_select"
                            and event.get("args", {}).get("Input Dims", [None])[0] == [140, 512]
                            for event in events), "actual compact KV restore must execute")


class LocalPrefillHostMetadataDevice(LocalPrefillCompactRowsDevice):
    def setUp(self):
        super().setUp()
        flags = patch.dict(os.environ, {"RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA": "1"})
        flags.start()
        self.addCleanup(flags.stop)


if __name__ == "__main__":
    unittest.main()
