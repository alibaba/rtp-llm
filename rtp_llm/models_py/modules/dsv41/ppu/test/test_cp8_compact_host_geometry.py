"""CP140 upload-source bounds and original-source terminal coverage.

CPU cases use actual native zigzag metadata and the public packed planner.
Only physical accelerator/stream admission and the upload boundary are CPU
doubles. Device/WORLD8 cases retain real versioned uploads and row exchange;
they are component contracts, not full-model timing or service evidence.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import timedelta
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import weakref

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_swa_metadata as metadata
from rtp_llm.models_py.modules.dsv41.ppu import _quant, local_prefill_qkv as qkv
from rtp_llm.models_py.modules.dsv41.ppu import prefill, prefill_runtime
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_cross_layer_mhc import _plan
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_metadata_reuse import ENV, _fixture
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_metadata import local_metadata_cpu_stream


FLAG = qkv.CP8_COMPACT_HOST_GEOMETRY_ENV
FLAGS = dict(ENV, **{FLAG: "1"}, RTP_V41_PPU_CP8_BATCH_SWA_DECODE="0",
             RTP_V41_SHARED_VALIDATE_STATE="0")


@contextmanager
def _environment(values):
    # Never clear libc environ while native ProcessGroup threads are alive.
    absent = object()
    previous = {name: os.environ.get(name, absent) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is absent:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _plan_inputs_fixture(**kwargs):
    state, original = _fixture(**kwargs)
    state.inputs.need_all_logits = state.inputs.need_all_hidden_states = False
    # The inherited SharedNativeCache.__new__ fixture has no cache payload.
    # Real prepare_plan evaluates this argument before the admission double;
    # retain an explicit owner rather than bypassing that production boundary.
    state.cache.native = object()
    return state, original


def _prepared(*, group=None, **kwargs):
    state, original = _plan_inputs_fixture(**kwargs)

    def create(v4, context, attn, native, policy):
        # The state boundary is real prepare_plan; only public model/cache
        # admission is outside this metadata-only component fixture.
        if native is not state.cache.native:
            raise AssertionError("prepare_plan must pass the fixture's native cache owner")
        return _plan(state, group=group), "compact140"

    with patch.object(prefill_runtime, "bounded_plan", side_effect=create):
        state.prepare_plan()
    return state, original


def _activate(state):
    compact = state.plan.context
    state.compact_ids = torch.zeros(compact.chunk_length, dtype=torch.int64,
                                    device=compact.global_positions.device)
    state._activate_compact(torch.empty((compact.chunk_length, 1), dtype=torch.bfloat16,
                                      device=compact.global_positions.device))


def _legacy_oracles(state):
    """The prior ATen consumers, independent of the new host derivation."""
    original, compact, exchange = state.original_context, state.plan.context, state.plan.exchange
    output = torch.zeros(exchange.original_rows, dtype=torch.bool, device=original.global_positions.device)
    output.index_fill_(0, exchange.send_indices, True)
    ends = torch.tensor([p + n - 1 for p, n in zip(state.query_prefixes, state.lengths)],
                        dtype=torch.int64, device=original.global_positions.device)
    last = original.local_is_real & (original.global_positions == ends[original.req_id_per_token.long()])
    coverage = bool(output[last].all())
    terminal = tuple(last.nonzero().flatten().tolist())
    bounds = tuple((rid, int(chunk.min()), int(chunk.max())) for rid, chunk in enumerate(
        compact.global_positions.split(compact.chunk_lengths_per_req)) if chunk.numel())
    return coverage, terminal, bounds, output


@contextmanager
def _no_readback():
    fail = AssertionError("compact host consumer must not read device values")
    with (patch.object(torch.Tensor, "cpu", side_effect=fail),
          patch.object(torch.Tensor, "tolist", side_effect=fail),
          patch.object(torch.Tensor, "item", side_effect=fail),
          patch.object(torch.Tensor, "__bool__", side_effect=fail),
          patch.object(torch.Tensor, "nonzero", side_effect=fail)):
        yield


class CP8CompactHostGeometryContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.inference_mode(False))
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(_environment(FLAGS))
        self.stack.enter_context(patch.object(_quant, "is_ppu", return_value=True))
        self.stream = self.stack.enter_context(local_metadata_cpu_stream())
        self.stack.enter_context(patch.object(metadata, "_upload", side_effect=lambda values, device, dtype:
            torch.tensor(values, device=device, dtype=dtype)))
        self.stack.enter_context(patch.object(logging.getLogger(), "isEnabledFor", return_value=False))

    def test_CP8_all_ranks_formal_prime_ragged_match_actual_legacy_consumers_without_readback(self):
        for lengths, prefixes in (
            ((4096,) * 8, (126976,) * 8),
            ((4097,) * 8, (126977,) * 8),
            ((17, 140, 141, 257, 528), (0, 3, 91, 8192, 7)),
        ):
            for rank in range(8):
                with self.subTest(lengths=lengths, rank=rank):
                    state, _ = _prepared(rank=rank, lengths=lengths, prefixes=prefixes)
                    birth = state.plan._ppu_cp8_query_birth
                    self.assertIsNotNone(state._cp_compact_host_birth)
                    expected = _legacy_oracles(state)
                    host = birth.host_geometry
                    self.assertEqual((host.output_covered, tuple(sorted(host.terminal_source_rows)), host.bounds),
                                     expected[:3])
                    with _no_readback():
                        _activate(state)
                        positions = state.context.global_positions.view(-1)
                        bounds = state.cp_compact_workspace_bounds(state.context, positions)
                        current = state.cp_compact_workspace_bounds_current(state.context, positions)
                    self.assertTrue(current)
                    self.assertEqual(bounds, expected[2])
                    self.assertTrue(torch.equal(state.valid_output_rows, expected[3]))
                    stats = state._cp_compact_host_geometry_evidence
                    self.assertIs(state.shared["cp_compact_host_geometry"], stats)
                    self.assertEqual((stats["plan_constructs"], stats["output_coverage"],
                                      stats["workspace_bounds"], stats["workspace_checks"]), (1, 1, 1, 1))
                    if lengths == (4096,) * 8:
                        self.assertEqual(state.context.chunk_length, 144)
                        self.assertEqual(int(state.context.local_is_real.sum()), 112 if rank == 0 else 144)
                        # Output coverage counts original source rows, not
                        # compact receive rows (rank0 receives only112).
                        self.assertEqual(int(state.valid_output_rows.sum()), 1120 if rank == 0 else 0)

    def test_CP8_versions_storage_alias_host_tuple_plan_exchange_mask_append_and_stream_are_current(self):
        def add_zero(value):
            value.add_(0)

        changes = {
            "private_positions_version": lambda s: add_zero(s.context.global_positions),
            "private_positions_alias_version": lambda s: add_zero(s.context.global_positions.view(-1)),
            "private_positions_clone": lambda s: setattr(s.context, "global_positions", s.context.global_positions.clone()),
            "private_positions_storage": lambda s: s.context.global_positions.set_(s.context.global_positions.clone()),
            "private_send_version": lambda s: add_zero(s.plan.exchange.send_indices),
            "private_receive_version": lambda s: add_zero(s.plan.exchange.receive_positions),
            "private_slab_version": lambda s: add_zero(s.plan.exchange._layout_host_buffer),
            "private_slab_clone": lambda s: setattr(s.plan.exchange, "_layout_host_buffer", s.plan.exchange._layout_host_buffer.clone()),
            "private_mask": lambda s: s.context.local_is_real.logical_not_(),
            "private_cu": lambda s: add_zero(s.plan.cu_seqlens),
            "private_indexer_group": lambda s: add_zero(s.plan.indexer_groups[0][0]),
            "native_restore_version": lambda s: add_zero(s.original_context.cp_info.prefill_qkv_restore_indice_cpu),
            "native_restore_clone": lambda s: setattr(s.original_context.cp_info, "prefill_qkv_restore_indice_cpu",
                s.original_context.cp_info.prefill_qkv_restore_indice_cpu.clone()),
            "native_mask": lambda s: s.original_context.cp_info.prefill_qkv_padding_mask_cpu.logical_not_(),
            "native_positions": lambda s: add_zero(s.original_context.global_positions),
            "native_mask_alias": lambda s: s.original_context.local_is_real.view(-1).logical_not_(),
            "native_owner": lambda s: setattr(s, "inputs", NS(**vars(s.inputs))),
            "host_send_tuple": lambda s: setattr(s.plan.exchange, "send_sizes", tuple(list(s.plan.exchange.send_sizes))),
            "host_projection_tuple": lambda s: setattr(s.plan.exchange, "projection_groups_host",
                tuple(list(s.plan.exchange.projection_groups_host))),
            "host_lengths_tuple": lambda s: setattr(s.context, "input_lengths_global_host", tuple(list(s.lengths))),
            "replay_tuple": lambda s: setattr(s.context, "swa_replay_starts_host",
                tuple(list(s.context.swa_replay_starts_host))),
            "replay_changed": lambda s: setattr(s.context, "swa_replay_starts_host", (0,) * len(s.lengths)),
            "append": lambda s: setattr(s, "append_starts", (0,) * len(s.lengths)),
            "plan": lambda s: setattr(s, "plan", replace(s.plan)),
            "exchange": lambda s: setattr(s.plan, "exchange", replace(s.plan.exchange)),
            "exchange_hook": lambda s: setattr(s.plan.exchange, "compact", lambda value: value),
            "plan_hook": lambda s: setattr(s.plan, "restore_rows", lambda value: value),
            "output_mask_version": lambda s: s.valid_output_rows.logical_not_(),
            "output_mask_rebind": lambda s: setattr(s, "valid_output_rows", s.valid_output_rows.clone()),
            "compact_ids_version": lambda s: add_zero(s.compact_ids),
            "compact_ids_rebind": lambda s: setattr(s, "compact_ids", s.compact_ids.clone()),
            "capture": lambda s: setattr(s.attn, "is_cuda_graph", True),
            "all_logits": lambda s: setattr(s.inputs, "need_all_logits", True),
            "current_forward": lambda s: setattr(s.runtime, "current", object()),
        }
        for name, change in changes.items():
            with self.subTest(change=name):
                state, _ = _prepared(rank=1, lengths=(256, 272), prefixes=(17, 49))
                _activate(state)
                context, birth = state.context, state._cp_compact_host_birth
                self.assertTrue(state.cp_compact_workspace_bounds_current(context, context.global_positions))
                change(state)
                self.assertFalse(state.cp_compact_workspace_bounds_current(context, context.global_positions))
                self.assertIsNone(state.cp_compact_workspace_bounds(context, context.global_positions))
                self.assertIs(state._cp_compact_host_birth, birth, "consumers cannot remint changed metadata")
        state, _ = _prepared()
        _activate(state)
        self.stream.handle += 1
        self.assertFalse(state.cp_compact_workspace_bounds_current(state.context, state.context.global_positions))

    def test_CP8_positions_require_exact_source_alias_extent_stride_and_dtype(self):
        state, _ = _prepared()
        _activate(state)
        cp, source = state.context, state.context.global_positions
        self.assertIsNotNone(state.cp_compact_workspace_bounds(cp, source.view(-1)))
        for positions in (source.clone(), source.int(), source[1:], source[::2],
                          torch.empty(source.numel() * 2, dtype=source.dtype)[::2]):
            self.assertIsNone(state.cp_compact_workspace_bounds(cp, positions))
        self.assertIsNone(state.cp_compact_workspace_bounds(state.original_context, source))

    def test_CP8_native_inference_sources_keep_readonly_boundary_private_transport_is_pristine_versioned(self):
        for inference in (False, True):
            with self.subTest(native_inference=inference):
                state, _ = _prepared(native_inference=inference)
                birth = state.plan._ppu_cp8_query_birth
                self.assertEqual(state.original_context.global_positions.is_inference(), inference)
                for value in birth.transport:
                    self.assertFalse(value.is_inference())
                    self.assertEqual(value._version, 0)
                with _no_readback():
                    _activate(state)
                    result = state.cp_compact_workspace_bounds(state.context, state.context.global_positions)
                self.assertIsNotNone(result)
                # Even inference native owners may never be rebound to a clone.
                state.original_context.cp_info.prefill_qkv_padding_mask_cpu = \
                    state.original_context.cp_info.prefill_qkv_padding_mask_cpu.clone()
                self.assertFalse(state.cp_compact_workspace_bounds_current(state.context, state.context.global_positions))

    def test_CP8_native_host_birth_precedes_plan_and_unknown_layouts_keep_legacy_fallback(self):
        for change in ("restore_version", "mask_version", "no_slab", "wrong_slab", "nonpacked_view",
                       "mutable_host", "prebirth_slab_write", "prebirth_device_write"):
            with self.subTest(change=change):
                state, _ = _plan_inputs_fixture(lengths=(256, 272), prefixes=(17, 49))
                if change == "restore_version":
                    state.original_context.cp_info.prefill_qkv_restore_indice_cpu.add_(0)
                elif change == "mask_version":
                    state.original_context.cp_info.prefill_qkv_padding_mask_cpu.logical_not_().logical_not_()

                def create(v4, context, attn, native, policy):
                    if native is not state.cache.native:
                        raise AssertionError("prepare_plan must pass the fixture's native cache owner")
                    plan = _plan(state)
                    if change == "no_slab":
                        plan.exchange._layout_host_buffer = None
                    elif change == "wrong_slab":
                        plan.exchange._layout_host_buffer = torch.empty(16, dtype=torch.uint8)
                    elif change == "nonpacked_view":
                        plan.context.global_positions = plan.context.global_positions.clone()
                    elif change == "mutable_host":
                        plan.exchange.send_sizes = list(plan.exchange.send_sizes)
                    elif change == "prebirth_slab_write":
                        plan.exchange._layout_host_buffer.add_(0)
                    elif change == "prebirth_device_write":
                        plan.context.global_positions.add_(0)
                    # This test models unsupported metadata at the actual plan
                    # creation boundary, rather than refreshing it at a consumer.
                    plan._ppu_cp8_query_birth = qkv.CP8QueryPlanBirth(plan)
                    return plan, "compact140"

                with patch.object(prefill_runtime, "bounded_plan", side_effect=create):
                    state.prepare_plan()
                self.assertIsNone(state._cp_compact_host_birth)
                _activate(state)
                self.assertIsNone(state.cp_compact_workspace_bounds(state.context, state.context.global_positions))
                self.assertEqual(state._cp_compact_host_geometry_evidence["output_coverage"], 0)
                self.assertTrue(_legacy_oracles(state)[0])

    def test_CP8_default_off_and_late_opt_in_cannot_create_a_native_birth(self):
        with _environment({FLAG: "0"}):
            os.environ.pop(FLAG)
            self.assertEqual(qkv.env_value(FLAG, "0"), "0", "the independent switch defaults off")
            state, _ = _prepared()
            self.assertIsNone(state._cp_compact_host_native_birth)
            self.assertIsNone(state.plan._ppu_cp8_query_birth.host_geometry)
            _activate(state)
            self.assertIsNone(state.cp_compact_workspace_bounds(state.context, state.context.global_positions))
        self.assertIsNone(state.cp_compact_workspace_bounds(state.context, state.context.global_positions))
        self.assertEqual(state._cp_compact_host_geometry_evidence["plan_constructs"], 0)
        with _environment({FLAG: "0"}):
            self.assertEqual(qkv.env_value(FLAG, "0"), "0")

    def test_CP8_same_stream_hook_mutation_is_checked_after_stream_read(self):
        for name in ("positions", "send", "mask", "native_mask", "output", "plan"):
            for activation in (False, True):
                with self.subTest(source=name, activation=activation):
                    state, _ = _prepared(rank=1, lengths=(256, 272), prefixes=(17, 49))
                    if not activation:
                        _activate(state)
                    compact = state.plan.context
                    values = dict(positions=compact.global_positions, send=state.plan.exchange.send_indices,
                                  mask=compact.local_is_real,
                                  native_mask=state.original_context.cp_info.prefill_qkv_padding_mask_cpu)
                    if name == "output" and activation:
                        continue
                    if name == "output":
                        values[name] = state.valid_output_rows
                    changed = False

                    def stream(device):
                        nonlocal changed
                        if not changed:
                            changed = True
                            if name == "plan":
                                state.plan = replace(state.plan)
                            elif values[name].dtype == torch.bool:
                                values[name].logical_not_().logical_not_()
                            else:
                                values[name].add_(0)
                        return NS(device=device, cuda_stream=self.stream.handle)

                    with patch.object(torch.cuda, "current_stream", side_effect=stream):
                        if activation:
                            # Value-preserving mutations must select the old
                            # exact validation, never certify a fresh fastpath.
                            _activate(state)
                            self.assertEqual(state._cp_compact_host_geometry_evidence["output_coverage"], 0)
                        else:
                            self.assertIsNone(state.cp_compact_workspace_bounds(compact, compact.global_positions))
                    self.assertTrue(changed)

    def test_CP8_missing_terminal_send_keeps_the_actual_validation_error(self):
        state, _ = _prepared(rank=0, lengths=(256, 272), prefixes=(17, 49))
        terminal = state.plan._ppu_cp8_query_birth.host_geometry.terminal_source_rows[0]
        send = state.plan.exchange.send_indices
        send[send == terminal] = 0
        with self.assertRaisesRegex(RuntimeError, "YOCO removed a framework first-token output row"):
            _activate(state)
        self.assertEqual(state._cp_compact_host_geometry_evidence["output_coverage"], 0)

    def test_CP8_actual_workspace_constructor_uses_exact_host_bounds_without_tensor_minmax(self):
        state, _ = _prepared(rank=0)
        expected = _legacy_oracles(state)[2]
        _activate(state)
        state.cache.cp = {"size": 8}
        cp = state.context
        context = NS(positions=cp.global_positions, request_ids=cp.req_id_per_token.long())
        with _no_readback(), patch.object(torch.Tensor, "min", side_effect=AssertionError("no device bounds min")), \
             patch.object(torch.Tensor, "max", side_effect=AssertionError("no device bounds max")):
            workspace = prefill._CPPrefillWorkspace(NS(), state.cache, NS(head_dim=512), context, cp,
                                                   None, 0, 128, {}, state=state)
        self.assertTrue(workspace._cp_compact_host_bounds)
        self.assertIs(workspace._cp_compact_host_state(), state)
        self.assertEqual(tuple((rid, first, first + count - 1) for rid, _, first, count, _ in workspace.swa_ranges),
                         tuple((rid, max(0, first - 127), last) for rid, first, last in expected))

    def test_CP8_explicit_evidence_keeps_original_readback_semantics_separate_from_proof(self):
        state, _ = _prepared(rank=0)
        _activate(state)
        calls, original = [], torch.Tensor.tolist

        def tolist(value):
            calls.append(value)
            return original(value)

        with patch.object(torch.Tensor, "tolist", new=tolist):
            evidence = state.evidence()
        self.assertTrue(any(value is state.context.global_positions for value in calls))
        self.assertEqual(evidence["local_positions"], original(state.context.global_positions))
        self.assertEqual(evidence["local_valid_output_rows"], int(state.valid_output_rows.sum()))

    def test_CP8_actual_cached_workspace_consumer_rejects_stale_before_refresh(self):
        state, _ = _prepared()
        _activate(state)
        cp, positions = state.context, state.context.global_positions
        bounds = state.cp_compact_workspace_bounds(cp, positions)
        self.assertIsNotNone(bounds)
        refresh = Mock()
        workspace = NS(_cp_compact_host_bounds=True, _cp_compact_host_state=weakref.ref(state),
                       _shared_global=None, refresh_swa=refresh,
                       indices=lambda *args: (torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.int32)),
                       kv=torch.empty((1, 4), dtype=torch.bfloat16))
        state.shared["prefill_kv_workspace"] = cp, (None, 0, 128, {}, False), workspace
        kwargs = dict(ops=NS(), cache=state.cache, layer_id=21,
                      geometry=NS(ratio=lambda layer: 0, kv_source_for=lambda layer: None),
                      context=NS(positions=positions), selected_blocks=None, window=128,
                      swa_starts=None, cp=cp, shared=state.shared, state=state)
        with _no_readback():
            prefill.build_cp_prefill_workspace(**kwargs)
        self.assertEqual(refresh.call_count, 1)
        other, _ = _prepared()
        with self.assertRaisesRegex(RuntimeError, "stale CP compact host workspace bounds"):
            prefill.build_cp_prefill_workspace(**dict(kwargs, state=other))
        self.assertEqual(refresh.call_count, 1, "no refresh for a different forward owner")
        cp.local_is_real.logical_not_()
        with self.assertRaisesRegex(RuntimeError, "stale CP compact host workspace bounds"):
            prefill.build_cp_prefill_workspace(**kwargs)
        self.assertEqual(refresh.call_count, 1, "no refresh/index work after stale proof")


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1",
                     "explicit native PPU lease required")
class CP8CompactHostGeometryDevice(unittest.TestCase):
    def test_CP8_actual_upload_values_private_aliases_and_native_stream(self):
        self.assertTrue(_quant.is_ppu())
        with _environment(FLAGS), torch.inference_mode(False), torch.no_grad(), patch.object(
                logging.getLogger(), "isEnabledFor", return_value=False):
            for rank in range(8):
                state, _ = _prepared(rank=rank, device="cuda:0", lengths=(257, 528), prefixes=(17, 8192))
                expected = _legacy_oracles(state)
                with _no_readback():
                    _activate(state)
                    bounds = state.cp_compact_workspace_bounds(state.context, state.context.global_positions.view(-1))
                self.assertEqual(bounds, expected[2])
                self.assertTrue(torch.equal(state.valid_output_rows, expected[3]))
                with torch.cuda.stream(torch.cuda.Stream()):
                    self.assertFalse(state.cp_compact_workspace_bounds_current(state.context, state.context.global_positions))
                state.plan.exchange.send_indices.add_(0)
                self.assertFalse(state.cp_compact_workspace_bounds_current(state.context, state.context.global_positions))


def _CP8_world_worker(rank, store):
    import torch.distributed as dist

    with _environment(FLAGS):
        torch.cuda.set_device(rank)
        try:
            dist.init_process_group("nccl", init_method="file://" + store, rank=rank, world_size=8,
                                    timeout=timedelta(seconds=300))
            with torch.inference_mode(False), torch.no_grad(), patch.object(
                    logging.getLogger(), "isEnabledFor", return_value=False):
                for lengths, prefixes in (((4096,) * 8, (126976,) * 8),
                                          ((4097,) * 8, (126977,) * 8),
                                          ((17, 140, 141, 257), (0, 3, 8192, 7))):
                    state, _ = _prepared(rank=rank, lengths=lengths, prefixes=prefixes,
                                         device=f"cuda:{rank}", group=dist.group.WORLD)
                    expected = _legacy_oracles(state)
                    _activate(state)
                    positions = state.context.global_positions
                    with _no_readback():
                        bounds = state.cp_compact_workspace_bounds(state.context, positions)
                    if bounds != expected[2] or not torch.equal(state.valid_output_rows, expected[3]):
                        raise AssertionError((rank, "CP8 host proof differs from actual legacy metadata"))
                    original = state.original_context
                    # Actual collective transport and restore preserve every
                    # certified terminal source row with request/position tags.
                    value = (original.req_id_per_token.long() * 1000000 + original.global_positions).unsqueeze(1)
                    compact = state.plan.exchange.compact(value)
                    restored = state.plan.exchange.restore(compact)
                    ends = torch.tensor([p + n - 1 for p, n in zip(prefixes, lengths)], device=value.device)
                    last = original.local_is_real & (original.global_positions == ends[original.req_id_per_token.long()])
                    if not torch.equal(restored[last], value[last]):
                        raise AssertionError((rank, "CP8 actual WORLD8 terminal restoration differs"))
                    if state._cp_compact_host_geometry_evidence["output_coverage"] != 1:
                        raise AssertionError((rank, "CP8 native plan was not admitted"))
                    dist.barrier()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_CP8_WORLD8") == "1", "explicit eight-PPU lease required")
class CP8CompactHostGeometryWorld8(unittest.TestCase):
    def test_CP8_native_all_rank_formal_prime_ragged_transport(self):
        self.assertTrue(_quant.is_ppu())
        with tempfile.TemporaryDirectory(prefix="cp8-host-geometry-") as directory:
            torch.multiprocessing.spawn(_CP8_world_worker, args=(str(Path(directory) / "store"),),
                                        nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
