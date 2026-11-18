"""CP-only Engram entry/publication/native-birth contracts.

CPU cases explicitly double native device/pinning admission and transport.
Device and WORLD8 cases use pinned owner gathers; WORLD8 retains the real
uniform ID-gather/reduce-scatter protocol and owns all spawned processes.
"""

from contextlib import ExitStack
from datetime import timedelta
from dataclasses import replace
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv41.ppu import cp_prefill_engram_rounds as cp_rounds
from rtp_llm.models_py.modules.dsv41.ppu import engram, model_runtime
from rtp_llm.models_py.modules.dsv41.ppu.model_runtime import SharedNativeRuntime
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_cross_layer_mhc import CP_ENV, _state
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_l20_query_projection import _context
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_engram_rounds import (
    _CPUWorld, _bindings, _ids, _same_bytes, _table,
)


def _fixture(rank, group, device="cpu", *, lengths=(1376,), prefixes=(256,), native=False):
    cp, tokens = _context(rank, lengths, prefixes, device)
    state = _state(cp, tokens)
    runtime = state.runtime
    # Recreate the actual entry ordering: the public Context has not bound
    # yet when Runtime negotiates rounds and produces hash tensors.
    state.original_context = state.context = None
    state._cp_full_rows_bound, state._cp_full_rows_birth, state._cp_full_rows_invalid = False, None, False
    info = NS(prefill_actual_input_lengths_cpu=torch.tensor(lengths, dtype=torch.int32),
              prefill_prefix_lengths_cpu=torch.tensor(prefixes, dtype=torch.int32),
              prefill_cp_chunk_lengths=torch.tensor(cp.chunk_lengths_per_req, dtype=torch.int32))
    state.attn.context_parallel_info, cp.cp_info = info, info
    windows = torch.full((tokens.numel(), 3), -1, dtype=torch.int64, device=device)
    windows[:, 0] = tokens
    windows[~cp.local_is_real] = -1
    state.inputs.engram_token_windows = windows
    runtime.device = torch.device(device)
    runtime.geometry = NS(engram_layer_ids=(1, 14), engram_max_ngram_size=3, image_token_id=-200)
    runtime.hash_params = object()
    runtime._engram_epoch, runtime._engram_incomplete = 0, False
    runtime._engram_requested, runtime._engram_flag_valid = False, True
    runtime._cp_engram_requested, runtime._cp_engram_flag_valid = True, True
    runtime._engram_evidence = {}
    runtime.model._v4_args = NS(tp_size=1, dp_size=1, ep_size=8, dp_rank=0, ep_rank=rank)
    runtime.model.parallelism_config.tp_rank = rank
    bindings = _bindings(rank, group, native=native)
    if not native:
        for shard, _, _ in bindings.values():
            shard.pinned = True
    runtime.model._shared_ops = NS(_engram=bindings, _enpm_group=bindings[1][2])
    runtime.model.v4.layers = [NS(engram=None, engram_hashes=None, engram_token_mask=None) for _ in range(15)]
    for layer in (1, 14):
        runtime.model.v4.layers[layer].engram = object()
    return runtime, state, cp, bindings


def _prepare(runtime, state):
    schedule = SharedNativeRuntime._prepare_engram_rounds(runtime, state)
    state.engram_round_schedule = schedule
    return schedule


def _publish(runtime, state, cp, *, all_dead=False):
    first = _ids(cp.chunk_length, cp.cp_rank, runtime.device)
    second = _ids(cp.chunk_length, (cp.cp_rank + 3) % 8, runtime.device)
    first[~cp.local_is_real], second[~cp.local_is_real] = -1, -1
    if all_dead:
        first.fill_(-1)
        second.fill_(-1)
    hashes = torch.stack((first, second), dim=1)
    with patch.object(model_runtime, "compute_engram_hash_ids", return_value=hashes):
        SharedNativeRuntime.prepare_hashes(runtime, state.inputs, False)
    state.bind_context(cp)
    return tuple(runtime.model.v4.layers[layer].engram_hashes for layer in (1, 14))


def _expected(rows):
    payload, scale = _table()
    flat = rows.cpu().reshape(-1)
    valid = (flat >= 0) & (flat < payload.shape[0])
    result = torch.zeros((flat.numel(), 256), dtype=torch.bfloat16)
    result[valid] = engram.dequant_rows(payload.view(torch.uint8)[flat[valid]].view(torch.float8_e4m3fn), scale[flat[valid]])
    return result.reshape(*rows.shape, 256)


class CP8EngramRoundsContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(patch.dict(os.environ, CP_ENV))
        self.stack.enter_context(patch("rtp_llm.models_py.modules.dsv41.ppu.layers.is_ppu", return_value=True))
        self.stack.enter_context(patch("rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", return_value=True))
        self.stack.enter_context(patch("torch.cuda.is_current_stream_capturing", return_value=False))
        self.stack.enter_context(patch(
            "rtp_llm.models_py.modules.dsv41.ppu.attention_layer.optimized_diagnostics_enabled", return_value=False))
        self.stack.enter_context(patch.object(cp_rounds, "_device_supported", return_value=True))
        self.stack.enter_context(patch.object(cp_rounds, "_device_rows_supported", side_effect=lambda rows, device: rows.device == device))
        self.stack.enter_context(patch.object(torch.Tensor, "is_pinned", return_value=True))
        self.stack.enter_context(patch.object(engram, "_triton_kernels", return_value=object()))

    def test_entry_has_no_native_birth_and_the_actual_hash_producer_publishes_before_claim(self):
        with _CPUWorld() as world:
            runtime, state, cp, bindings = _fixture(0, world.group)
            schedule = _prepare(runtime, state)
            self.assertIsNotNone(schedule)
            self.assertIsNone(state.original_context)
            self.assertEqual(len(world.packets), 1)
            self.assertEqual(world.packets[0].numel(), 29)
            self.assertEqual(world.packets[0][-3].item(), cp_rounds.CP8_MODE_TAG + cp.chunk_length * 24)
            self.assertFalse(runtime._engram_evidence["native_birth_claimed"])
            shard, layout, adapter = bindings[1]
            with self.assertRaisesRegex(RuntimeError, "binding/lifecycle"):
                schedule.claim(adapter, shard, layout, torch.zeros((cp.chunk_length, 24), dtype=torch.int64), 1)
            rows = _publish(runtime, state, cp)
            self.assertTrue(state._cp_full_rows_valid())
            self.assertEqual(runtime._engram_evidence["published_layers"], [1, 14])
            for layer, hashes in zip((1, 14), rows):
                shard, layout, adapter = bindings[layer]
                record = schedule.claim(adapter, shard, layout, hashes, layer)
                self.assertTrue(record["uniform_rank_rows"])
                record["complete"] = True
            self.assertTrue(runtime._engram_evidence["native_birth_claimed"])
            self.assertEqual(runtime._engram_evidence["native_birth_layers"], [1, 14])
            self.assertTrue(schedule.close())
            self.assertFalse(schedule.close())
            self.assertEqual(schedule._lease.publications, {})

    def test_mixed_mode_count_header_and_binding_disagreements_reject_before_lookup(self):
        for field in (0, 2, 4, 7, 8, 26, 28):
            with self.subTest(field=field), _CPUWorld() as world:
                runtime, state, cp, _ = _fixture(0, world.group)
                def mutate(packet):
                    packet[field] += 1
                world.mutate = mutate
                with self.assertRaisesRegex(RuntimeError, "configuration/lifecycle disagreement"):
                    _prepare(runtime, state)
                self.assertEqual(len(world.packets), 1)
        # The legacy DP control shape/last dimension pair stays unchanged.
        with _CPUWorld() as world:
            _, evidence = world.prepare(requested=False, supported=False)
            self.assertEqual(len(evidence["packet"]), 29)
            self.assertEqual(evidence["packet"][26:28], [256, -256])

    def test_disabled_unsupported_and_invalid_flags_keep_joint_admission(self):
        for change, decision in (("off", "disabled"), ("unsupported", "fallback"), ("empty", "fallback")):
            with self.subTest(change=change), _CPUWorld() as world:
                runtime, state, cp, _ = _fixture(0, world.group)
                if change == "off":
                    runtime._cp_engram_requested = False
                elif change == "unsupported":
                    runtime.model._is_speculative = True
                else:
                    state.inputs.input_ids = torch.empty(0, dtype=torch.int64)
                    state.inputs.engram_token_windows = torch.empty((0, 3), dtype=torch.int64)
                self.assertIsNone(_prepare(runtime, state))
                self.assertEqual(runtime._engram_evidence["decision"], decision)
                self.assertEqual(len(world.packets), 1)
        for change in ("cp_flag", "dp_flag", "incomplete", "group"):
            with self.subTest(change=change), _CPUWorld() as world:
                runtime, state, cp, bindings = _fixture(0, world.group)
                if change == "cp_flag": runtime._cp_engram_flag_valid = False
                elif change == "dp_flag": runtime._engram_requested = True
                elif change == "incomplete": runtime._engram_incomplete = True
                else: bindings[1][2].pg = object()
                with self.assertRaisesRegex(RuntimeError, "configuration/lifecycle disagreement"):
                    _prepare(runtime, state)

    def test_hash_input_binding_and_native_birth_changes_cannot_remint_a_claim(self):
        for change in ("hash", "hash_alias", "window", "cpu_metadata", "native", "epoch", "lease"):
            with self.subTest(change=change), _CPUWorld() as world:
                runtime, state, cp, bindings = _fixture(0, world.group)
                schedule = _prepare(runtime, state)
                rows = list(_publish(runtime, state, cp))
                if change == "hash": rows[0].add_(0)
                elif change == "hash_alias": rows[0] = rows[0].view_as(rows[0])
                elif change == "window": state.inputs.engram_token_windows.add_(0)
                elif change == "cpu_metadata": state.attn.context_parallel_info.prefill_prefix_lengths_cpu.add_(0)
                elif change == "native": cp.global_positions.add_(0)
                elif change == "epoch": runtime._engram_epoch += 1
                else: schedule = replace(schedule)
                shard, layout, adapter = bindings[1]
                with self.assertRaisesRegex(RuntimeError, "CP8 Engram"):
                    schedule.claim(adapter, shard, layout, rows[0], 1)
        with _CPUWorld() as world:
            runtime, state, cp, bindings = _fixture(0, world.group)
            schedule = _prepare(runtime, state)
            rows = _publish(runtime, state, cp)
            shard, layout, adapter = bindings[1]
            shard.payload = shard.payload.clone()
            with self.assertRaisesRegex(RuntimeError, "host binding changed"):
                schedule.claim(adapter, shard, layout, rows[0], 1)

    def test_uniform_certificate_reuses_owner_gather_and_rs_without_a_layer_count_vote(self):
        with _CPUWorld() as world:
            runtime, state, cp, bindings = _fixture(0, world.group, lengths=(1377, 17), prefixes=(256, 9))
            schedule = _prepare(runtime, state)
            rows = _publish(runtime, state, cp)
            self.assertFalse(state._cp_full_rows_birth[2])  # Real native padding is retained.
            old_lookup = engram._lookup_round
            outputs = []
            for layer, hashes in zip((1, 14), rows):
                shard, layout, adapter = bindings[layer]
                expected = _expected(hashes)
                with patch.object(adapter, "all_max_count", side_effect=AssertionError("no layer count vote")), \
                        patch.object(adapter, "all_reduce_sum", side_effect=AssertionError("CP keeps reduce_scatter")), \
                        patch.object(adapter, "all_gather_ids", side_effect=lambda ids: ids.repeat(8)) as gather, \
                        patch.object(adapter, "reduce_scatter_sum", side_effect=lambda contribution, length: expected.reshape(-1, 256)[:length].clone()) as scatter, \
                        patch.object(engram, "_lookup_round", wraps=old_lookup) as owner:
                    got = engram.engram_lookup(hashes, layer_id=layer, shard=shard, layout=layout,
                                               group=adapter, uniform_rank_rows=True, round_schedule=schedule)
                self.assertTrue(_same_bytes(got, expected))
                self.assertEqual(gather.call_count, 1)
                self.assertEqual(scatter.call_count, 1)
                self.assertEqual(owner.call_count, 1)
                outputs.append(got)
                self.assertFalse(torch._C._is_alias_of(got, hashes))
            before = outputs[0].clone()
            self.assertTrue(schedule.close())
            rows[0].fill_(-1)
            self.assertTrue(_same_bytes(outputs[0], before))
            self.assertEqual([record["rounds"] for record in schedule._lease.records], [[cp.chunk_length * 24]] * 2)
            self.assertEqual(len(world.packets), 1)

    def test_cp_and_dp_schedule_uniform_arguments_remain_separate(self):
        with _CPUWorld() as world:
            runtime, state, cp, bindings = _fixture(0, world.group)
            schedule = _prepare(runtime, state)
            rows = _publish(runtime, state, cp)
            shard, layout, adapter = bindings[1]
            with self.assertRaisesRegex(RuntimeError, "independent uniform schedule"):
                engram.engram_lookup(rows[0], layer_id=1, shard=shard, layout=layout,
                                     group=adapter, round_schedule=schedule)
            legacy, _ = world.prepare(local_flat_rows=rows[0].numel())
            with self.assertRaisesRegex(RuntimeError, "independent padded schedule"):
                engram.engram_lookup(rows[0], layer_id=1, shard=shard, layout=layout,
                                     group=adapter, uniform_rank_rows=True, round_schedule=legacy)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU lease required")
class CP8EngramRoundsDevice(unittest.TestCase):
    @torch.no_grad()
    def test_native_pinned_owner_gather_accepts_the_full_98304_row_query(self):
        self.assertTrue(engram.is_ppu())
        payload, scales = _table()
        layout = engram.EngramShardLayout.from_table(payload.shape[0], 1)
        shard = engram.HostShard.from_tensors(payload, scales, layout, 0)
        rows = _ids(4096, 0, "cuda:0")
        native = engram._triton_kernels()
        with patch.object(native, "engram_gather", wraps=native.engram_gather) as gather:
            output = engram.engram_lookup(rows, layer_id=1, shard=shard, layout=layout)
        self.assertEqual(gather.call_count, 1)
        self.assertTrue(_same_bytes(output.cpu(), _expected(rows)))
        self.assertFalse(torch._C._is_alias_of(output, rows))


@patch.dict(os.environ, CP_ENV)
def _world8_worker(rank, store):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    dist.init_process_group("nccl", init_method="file://" + store, rank=rank, world_size=8,
                            timeout=timedelta(seconds=240))
    try:
        device = torch.device("cuda", rank)
        with torch.no_grad():
            for lengths, prefixes, all_dead in (((126977,), (0,), False),
                                                ((4096,) * 8, (126976,) * 8, False),
                                                ((1377, 17), (256, 9), False), ((16,), (9,), True)):
                runtime, state, cp, bindings = _fixture(rank, dist.group.WORLD, device,
                                                       lengths=lengths, prefixes=prefixes, native=True)
                if all_dead:
                    state.inputs.engram_token_windows.fill_(-1)
                schedule = _prepare(runtime, state)
                assert schedule is not None
                rows = _publish(runtime, state, cp, all_dead=all_dead)
                previous = []
                for layer, hashes in zip((1, 14), rows):
                    shard, layout, adapter = bindings[layer]
                    with patch.object(adapter, "all_gather_ids", wraps=adapter.all_gather_ids) as gather, \
                            patch.object(adapter, "reduce_scatter_sum", wraps=adapter.reduce_scatter_sum) as scatter, \
                            patch.object(adapter, "all_max_count", side_effect=AssertionError("uniform CP has no MAX")):
                        old = engram.engram_lookup(hashes, layer_id=layer, shard=shard, layout=layout,
                                                   group=adapter, uniform_rank_rows=True)
                        old_rounds = len(engram._round_lengths(hashes.numel(), 4096))
                        assert gather.call_count == scatter.call_count == old_rounds
                        if lengths == (126977,):
                            assert cp.chunk_length == 15874 and hashes.numel() == 380976
                            assert old_rounds == 94
                        gather.reset_mock(); scatter.reset_mock()
                        actual = engram.engram_lookup(hashes, layer_id=layer, shard=shard, layout=layout,
                                                      group=adapter, uniform_rank_rows=True, round_schedule=schedule)
                        rounds = len(engram._round_lengths(hashes.numel(), 98304))
                        assert gather.call_count == scatter.call_count == rounds
                        if lengths == (126977,):
                            assert rounds == gather.call_count == scatter.call_count == 4
                            assert schedule._lease.records[-1]["rounds"] == [98304, 98304, 98304, 86064]
                    assert _same_bytes(actual, old) and _same_bytes(actual.cpu(), _expected(hashes))
                    assert not torch._C._is_alias_of(actual, hashes)
                    previous.append((actual, actual.clone()))
                assert schedule.close() and schedule._lease.publications == {}
                rows[0].fill_(-1)
                assert all(_same_bytes(output, saved) for output, saved in previous)
                assert runtime._engram_evidence["native_birth_layers"] == [1, 14]
            for disagreement in ("mode", "count", "empty_owner", "flag"):
                lengths = (1376 if disagreement != "count" or rank != 7 else 1392,)
                runtime, state, cp, bindings = _fixture(rank, dist.group.WORLD, device,
                                                       lengths=lengths, prefixes=(256,), native=True)
                if disagreement == "empty_owner" and rank == 7:
                    state.inputs.input_ids = torch.empty(0, dtype=torch.int64, device=device)
                    state.inputs.engram_token_windows = torch.empty((0, 3), dtype=torch.int64, device=device)
                if disagreement == "flag" and rank == 7:
                    runtime._cp_engram_requested = False
                rejected = False
                try:
                    if disagreement == "mode" and rank == 7:
                        engram.prepare_local_prefill_engram_schedule(device, requested=True, supported=True,
                            epoch=1, process_group=dist.group.WORLD, bindings=bindings,
                            local_flat_rows=state.inputs.input_ids.numel() * 24)
                    else:
                        _prepare(runtime, state)
                except RuntimeError as error:
                    rejected = "disagreement" in str(error)
                assert rejected
        torch.cuda.synchronize(device)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_CP8_WORLD8") == "1", "explicit eight-PPU lease required")
class CP8EngramRoundsWorld8(unittest.TestCase):
    def test_uniform_owner_protocol_both_layers_and_joint_rejections(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        with tempfile.TemporaryDirectory(prefix="cp8-engram-rounds-") as directory:
            torch.multiprocessing.spawn(_world8_worker, args=(str(Path(directory) / "store"),), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
