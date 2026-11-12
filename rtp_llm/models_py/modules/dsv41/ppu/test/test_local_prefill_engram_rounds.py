"""P eager Engram schedule contracts; native WORLD8 only via --distributed.

Unittest never spawns workers. CPU transport doubles prove protocol/routing,
not PCCL or pinned-host arithmetic. Root owns GPU locking and execution.
"""

import argparse
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import traceback
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
from torch import nn

from rtp_llm.models_py.modules.dsv41.ppu import block, engram, native_model
from rtp_llm.models_py.modules.dsv41.ppu.layers import PPUSharedEngram
from rtp_llm.models_py.modules.dsv41.ppu.model_runtime import SharedNativeRuntime
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps


DEVICE = torch.device("cuda:0")


def _table(nonfinite=False, count=2048):
    pattern = [0, 56, 184, 64, 192, 1, 129, 126, 254]
    if nonfinite:
        pattern += [128, 127, 255]
    payload = torch.tensor(pattern, dtype=torch.uint8).repeat((count * 256 + len(pattern) - 1) // len(pattern))[:count * 256]
    payload = payload.reshape(count, 256).view(torch.float8_e4m3fn)
    scale = torch.tensor([0, 1, 126, 127, 128, 253, 254, 255] if nonfinite else [126, 127, 128, 129] * 2,
                         dtype=torch.uint8)[None].repeat(count, 1)
    return payload, scale


def _bindings(rank, group, *, native=False, nonfinite=False, count=2048):
    payload, scale = _table(nonfinite, count)
    layout = engram.EngramShardLayout.from_table(count, 8)
    adapter = engram.DistributedEnpmGroup(group)
    result = {}
    for layer in (1, 14):
        begin, size = layout.row_start(rank), layout.valid_rows(rank)
        p = payload.view(torch.uint8)[begin:begin + size].clone().view(torch.float8_e4m3fn)
        s = scale[begin:begin + size].clone()
        if native:
            p, s = p.pin_memory(), s.pin_memory()
        shard = engram.HostShard(p, s, layout, rank, 256, pinned=native)
        result[layer] = shard, layout, adapter
    return result


def _ids(tokens, rank, device, *, idle=False, width=2048):
    values = (torch.arange(tokens * 24, device=device, dtype=torch.int64) * 37 + rank * 251) % width
    values[::31], values[1::47] = -1, width + 3
    values[2::53] = 7  # repeated IDs, including ranks that do not own them
    if idle:
        values.fill_(-1)
    return values.reshape(tokens, 24)


def _same_bytes(actual, expected):
    return torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


class _CPUWorld:
    """Explicit host-only control-collective double, never native evidence."""

    def __enter__(self):
        self.stack, self.packets, self.mutate = ExitStack(), [], None
        self.group = object()
        enter = self.stack.enter_context
        original_tensor = torch.tensor

        def tensor(*args, **kwargs):
            if torch.device(kwargs.get("device", "cpu")).type == "cuda":
                kwargs["device"] = "cpu"
            return original_tensor(*args, **kwargs)

        def reduce(packet, **kwargs):
            self.packets.append(packet.clone())
            if self.mutate is not None:
                self.mutate(packet)

        for name, value in (("is_initialized", True), ("get_world_size", 8), ("get_rank", 0)):
            enter(patch.object(dist, name, return_value=value))
        enter(patch.object(dist, "group", NS(WORLD=self.group)))
        enter(patch.object(dist, "all_reduce", side_effect=reduce))
        enter(patch.object(torch, "tensor", side_effect=tensor))
        enter(patch("torch.cuda.is_current_stream_capturing", return_value=False))
        self.bindings = _bindings(0, self.group)
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def prepare(self, *, requested=True, supported=True, epoch=1, **kwargs):
        evidence = {}
        schedule = engram.prepare_local_prefill_engram_schedule(
            DEVICE, requested=requested, supported=supported, epoch=epoch,
            process_group=self.group, evidence=evidence, bindings=self.bindings, **kwargs,
        )
        return schedule, evidence


class LocalPrefillEngramRoundsCPU(unittest.TestCase):
    def test_real_reference_receipt_finalizes_its_forward_dict_without_cross_forward_alias(self):
        first = dict(decision="adopted", epoch=1, round_cap=98304, packet=[], consumed_layers=[])
        runtime = NS(model=NS(_shared_ops=NS(moe_engine=NS(_mega_capacity=4096))), _engram_evidence=first)
        state = NS(runtime=runtime, lengths=(4096,), query_prefixes=(126976,), plan_reason="compact140",
                   compacted=True, shared={}, normalization={"mega_transitions": 40})
        with patch.object(native_model, "_REFERENCE_PREFILLS", []) as records, patch(
            "torch.cuda.max_memory_allocated", side_effect=AssertionError("receipt must not query a device"),
        ), patch("torch.cuda.max_memory_reserved", side_effect=AssertionError("receipt must not query a device")):
            native_model.record_reference_prefill(state)
            self.assertIs(records[0]["engram_rounds"], first)
            # Simulate the Runtime finally phase, after report/record already
            # ran, then a completely new forward's host-only evidence owner.
            first["consumed_layers"].append(dict(layer=1, complete=True))
            first.update(complete=True, torch_allocator_peak_allocated_bytes=1234,
                         torch_allocator_peak_reserved_bytes=2345)
            second = dict(decision="disabled", epoch=2, consumed_layers=[])
            runtime._engram_evidence = second
            native_model.record_reference_prefill(state)
            second.update(complete=False, torch_allocator_peak_allocated_bytes=3456)
            self.assertIs(records[1]["engram_rounds"], second)
            self.assertEqual(records[0]["engram_rounds"]["epoch"], 1)
            self.assertTrue(records[0]["engram_rounds"]["complete"])
            with tempfile.TemporaryDirectory(prefix="a10-engram-receipt-") as directory:
                native_model._flush_reference_prefills(directory, 0)
                saved = [json.loads(line) for line in (Path(directory) / "prefill-rank0.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 0)
            self.assertEqual(saved[0]["engram_rounds"]["torch_allocator_peak_allocated_bytes"], 1234)
            self.assertEqual(saved[0]["engram_rounds"]["consumed_layers"], [dict(layer=1, complete=True)])
            self.assertEqual(saved[1]["engram_rounds"]["epoch"], 2)
            self.assertFalse(saved[1]["engram_rounds"]["complete"])

    def test_disabled_and_unsupported_still_enter_the_same_control_collective(self):
        with _CPUWorld() as world:
            for requested, supported, decision in ((False, True, "disabled"), (True, False, "fallback")):
                schedule, evidence = world.prepare(requested=requested, supported=supported)
                self.assertIsNone(schedule)
                self.assertEqual(evidence["decision"], decision)
                self.assertEqual(evidence["round_cap"], 98304)
                self.assertEqual(len(evidence["packet"]), 28)
            self.assertEqual(len(world.packets), 2)
            schedule, evidence = world.prepare()
            self.assertEqual(schedule.round_cap, 98304)
            self.assertEqual(schedule.layers, (1, 14))
            self.assertIs(evidence["consumed_layers"], schedule._lease.records)
            with self.assertRaises(FrozenInstanceError):
                schedule.epoch = 2
            forged = replace(schedule, round_cap=4096)
            with self.assertRaisesRegex(RuntimeError, "another schedule"):
                forged.close()
            self.assertTrue(schedule._lease.active)
            self.assertFalse(schedule.close(), "an unused lease is incomplete, not successful")
            self.assertFalse(schedule._lease.active)

    def test_control_mismatch_or_fatal_is_jointly_rejected(self):
        for position, value in ((0, 1), (2, 98305), (4, 2), (7, 1), (8, 4096)):
            with self.subTest(position=position), _CPUWorld() as world:
                world.mutate = lambda packet, position=position, value=value: packet.__setitem__(position, value)
                # Position0 changes a flag0 proposal to max1 with min0.
                with self.assertRaisesRegex(RuntimeError, "configuration/lifecycle disagreement"):
                    world.prepare(requested=position != 0)
                self.assertEqual(len(world.packets), 1)
        with _CPUWorld() as world:
            for arguments in ({"round_cap": 98305}, {"layers": (1, 15)}, {"epoch": 0}):
                with self.assertRaises(RuntimeError):
                    world.prepare(**arguments)

    def test_local_simulator_large_round_preserves_ragged_invalid_and_empty_results(self):
        payload, scale = _table()
        layout = engram.EngramShardLayout.from_table(2048, 8)
        shards = []
        for rank in range(8):
            start, size = layout.row_start(rank), layout.valid_rows(rank)
            shards.append(engram.HostShard(payload.view(torch.uint8)[start:start + size].clone().view(torch.float8_e4m3fn),
                                           scale[start:start + size].clone(), layout, rank, 256))
        rows = [_ids(tokens, rank, "cpu", idle=rank == 1)
                for rank, tokens in enumerate((0, 1, 170, 171, 5, 140, 172, 3))]
        simulator = engram.LocalEnpmGroup(shards, layout)
        old = simulator.lookup(rows)
        new = simulator.lookup(rows, max_rows_per_round=98304)
        self.assertEqual(simulator.round_lengths, [172 * 24])
        for a, b in zip(old, new):
            self.assertTrue(_same_bytes(a, b))
        self.assertTrue((new[1] == 0).all())

    @torch.no_grad()
    def test_common_runtime_entry_keeps_idle_and_disabled_proposals_in_packet(self):
        with _CPUWorld() as world:
            bindings = world.bindings
            model = NS(_is_decode_role=False, _is_speculative=False, _prefill_cp_size=1,
                       _v4_args=NS(tp_size=1, dp_size=8, ep_size=8, dp_rank=0, ep_rank=0),
                       parallelism_config=NS(tp_size=1),
                       _shared_ops=NS(_engram=bindings, _enpm_group=bindings[1][2]))
            runtime = NS(model=model, geometry=NS(engram_layer_ids=(1, 14)), device=DEVICE,
                         _engram_epoch=0, _engram_requested=False, _engram_flag_valid=True,
                         _engram_incomplete=False, _engram_evidence={})
            for idle in (False, True):
                state = NS(prefill=not idle, idle=idle, attn=NS(is_cuda_graph=False),
                           cache=None if idle else NS(sharded=False))
                with patch.object(engram, "prepare_local_prefill_engram_schedule", return_value=None) as agree:
                    SharedNativeRuntime._prepare_engram_rounds(runtime, state)
                    agree.assert_called_once()
                    self.assertFalse(agree.call_args.kwargs["requested"])
            for graph_flag in ("is_graph", "is_graph_commit"):
                setattr(state, graph_flag, True)
                with patch.object(engram, "prepare_local_prefill_engram_schedule") as agree:
                    self.assertIsNone(SharedNativeRuntime._prepare_engram_rounds(runtime, state))
                    agree.assert_not_called()
                setattr(state, graph_flag, False)
            state.attn.is_cuda_graph = True
            with patch.object(engram, "prepare_local_prefill_engram_schedule") as agree:
                self.assertIsNone(SharedNativeRuntime._prepare_engram_rounds(runtime, state))
                agree.assert_not_called()
            state.attn.is_cuda_graph = False
            with patch("torch.cuda.is_current_stream_capturing", return_value=True), patch.object(
                engram, "prepare_local_prefill_engram_schedule",
            ) as agree:
                self.assertIsNone(SharedNativeRuntime._prepare_engram_rounds(runtime, state))
                agree.assert_not_called()
            model.parallelism_config.tp_size = 8
            runtime._engram_requested = True
            with patch.object(engram, "prepare_local_prefill_engram_schedule", return_value=None) as agree:
                SharedNativeRuntime._prepare_engram_rounds(runtime, state)
                self.assertTrue(agree.call_args.kwargs["fatal"])
            model.parallelism_config.tp_size = 1
            runtime._engram_incomplete = True
            with patch.object(engram, "prepare_local_prefill_engram_schedule", return_value=None) as agree:
                SharedNativeRuntime._prepare_engram_rounds(runtime, state)
                self.assertTrue(agree.call_args.kwargs["fatal"])
            model._is_decode_role = True
            with patch.object(engram, "prepare_local_prefill_engram_schedule") as agree:
                self.assertIsNone(SharedNativeRuntime._prepare_engram_rounds(runtime, state))
                agree.assert_not_called()

    @torch.no_grad()
    def test_active_and_idle_block_prefetch_and_old_cp_decode_defaults(self):
        for phase, idle, schedule in (("prefill", False, object()), ("decode", True, object()),
                                      ("decode", False, None), ("graph", False, None)):
            state = NS(prefill=phase == "prefill", idle=idle, is_graph=phase == "graph", context=None,
                       cache=NS(sharded=False), plan=None, executed_layers=[],
                       layer_calls={1: torch.zeros(1, dtype=torch.int64)},
                       v4=NS(capture_aux_hidden_layer_ids=()), engram_round_schedule=schedule,
                       runtime=NS(policy=NS(mode="full"), model=NS(_is_speculative=False)))
            state.before_block = lambda unit, *args: args
            state.publication = lambda layer: nullcontext()
            unit = block.SharedPPUBlock.__new__(block.SharedPPUBlock)
            nn.Module.__init__(unit)
            unit.layer_id, unit._mhc_phase = 1, "unbound"
            unit.attn = NS(runtime=NS(current=state, model=state.runtime.model))
            unit.engram, unit.engram_hashes = NS(prefetch=Mock()), torch.zeros(1, 24, dtype=torch.int64)
            with patch.object(block, "cp8_yoco_fast_prefill_enabled", return_value=False), patch.object(
                block, "_local_prefill_mega_mhc_enabled", return_value=False,
            ), patch.object(block.Block, "forward", return_value="prefill"), patch.object(
                block.Block, "forward_decode", return_value="decode",
            ):
                if phase == "prefill":
                    unit.forward(None, None, None, None)
                else:
                    unit.forward_decode(None, None, None)
            if phase == "graph":
                unit.engram.prefetch.assert_not_called()
            else:
                unit.engram.prefetch.assert_called_once()
                self.assertIs(unit.engram.prefetch.call_args.kwargs.get("round_schedule"), schedule)

    def test_layers_and_ops_forward_only_explicit_schedule_and_keep_simulator_default(self):
        unit = PPUSharedEngram.__new__(PPUSharedEngram)
        nn.Module.__init__(unit)
        unit.layer_id, unit._prefetched = 1, None
        unit.ops = NS(engram_lookup=Mock(return_value=torch.ones(1, 24, 256)))
        hashes, token = torch.zeros(1, 24, dtype=torch.int64), object()
        unit.prefetch(hashes, round_schedule=token)
        self.assertIs(unit.ops.engram_lookup.call_args.kwargs["round_schedule"], token)
        unit.clear()
        unit.prefetch(hashes)
        self.assertNotIn("round_schedule", unit.ops.engram_lookup.call_args.kwargs)
        ops = PPUOps.__new__(PPUOps)
        with _CPUWorld() as world:
            ops._engram = world.bindings
            with patch("rtp_llm.models_py.modules.dsv41.ppu.ops._engram_lookup", return_value="native") as lookup:
                self.assertEqual(ops.engram_lookup(hashes, layer_id=1, round_schedule=token), "native")
                self.assertIs(lookup.call_args.kwargs["round_schedule"], token)
                ops.engram_lookup(hashes, layer_id=1)
                self.assertNotIn("round_schedule", lookup.call_args.kwargs)

    def test_cp8_uniform_argument_keeps_its_old_fixed_rows_meaning(self):
        class CudaBranchTensor(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        with _CPUWorld() as world:
            shard, layout, adapter = world.bindings[1]
            rows = torch.zeros(1, 24, dtype=torch.int64).as_subclass(CudaBranchTensor)
            with patch.object(adapter, "all_max_count", side_effect=AssertionError("old uniform must skip MAX")), patch.object(
                adapter, "all_gather_ids", side_effect=lambda ids: ids.repeat(8),
            ), patch.object(engram, "_lookup_round", return_value=torch.ones(8 * 24, 256, dtype=torch.bfloat16)), patch.object(
                adapter, "reduce_scatter_sum", side_effect=lambda values, length: values[:length],
            ) as scatter:
                result = engram.engram_lookup(rows, layer_id=1, shard=shard, layout=layout,
                                               group=adapter, uniform_rank_rows=True)
            self.assertEqual(result.shape, (1, 24, 256))
            scatter.assert_called_once()
            schedule, _ = world.prepare()
            with self.assertRaisesRegex(RuntimeError, "independent padded schedule"):
                engram.engram_lookup(rows, layer_id=1, shard=shard, layout=layout, group=adapter,
                                     uniform_rank_rows=True, round_schedule=schedule)
            schedule.close()


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalPrefillEngramRoundsDevice(unittest.TestCase):
    @torch.no_grad()
    def test_single_ppu_native_pinned_gather_and_world8_precondition(self):
        self.assertTrue(engram.is_ppu(), "requires actual ZW PPU")
        payload, scale = _table()
        layout = engram.EngramShardLayout.from_table(2048, 1)
        shard = engram.HostShard.from_tensors(payload, scale, layout, 0)
        self.assertTrue(shard.pinned and shard.payload.is_pinned() and shard.scale.is_pinned())
        ids = _ids(171, 0, "cuda")
        before = shard.payload.view(torch.uint8).clone(), shard.scale.clone(), ids.clone()
        native = engram._triton_kernels()
        self.assertIsNotNone(native)
        with patch.object(native, "engram_gather", wraps=native.engram_gather) as gather:
            got = engram.engram_lookup(ids, layer_id=1, shard=shard, layout=layout)
        self.assertEqual(gather.call_count, 1)
        flat = ids.cpu().reshape(-1)
        expected = torch.zeros(flat.numel(), 256, dtype=torch.bfloat16)
        valid = (flat >= 0) & (flat < 2048)
        expected[valid] = engram.dequant_rows(payload.view(torch.uint8)[flat[valid]].view(torch.float8_e4m3fn), scale[flat[valid]])
        self.assertTrue(_same_bytes(got.cpu(), expected.reshape(*ids.shape, 256)))
        self.assertTrue(_same_bytes(shard.payload.view(torch.uint8), before[0]))
        self.assertTrue(_same_bytes(shard.scale, before[1]) and _same_bytes(ids, before[2]))
        if not dist.is_initialized():
            with self.assertRaisesRegex(RuntimeError, "WORLD8"):
                engram.prepare_local_prefill_engram_schedule(torch.device("cuda:0"), requested=True,
                    supported=True, epoch=1, process_group=None)


def _distributed_worker(rank, port, directory, pg_timeout):
    report = dict(rank=rank, world_size=8, normal_teardown=False, cases=[], rejections=[])
    path = Path(directory) / f"a10-engram-rank{rank}.json"

    def save():
        path.write_text(json.dumps(report, indent=2))

    try:
        if not engram.is_ppu() or torch.cuda.device_count() != 8:
            raise RuntimeError("WORLD8 contract requires exactly eight visible ZW PPU devices")
        torch.cuda.set_device(rank)
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        device = torch.device("cuda", rank)
        if torch.cuda.get_device_capability(device) != (8, 9):
            raise RuntimeError("WORLD8 contract requires PPU89")
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                                world_size=8, timeout=timedelta(seconds=pg_timeout))
        bindings = _bindings(rank, dist.group.WORLD, native=True)
        nonfinite_bindings = _bindings(rank, dist.group.WORLD, native=True, nonfinite=True)
        native = engram._triton_kernels()
        if native is None or not all(shard.pinned and shard.payload.is_pinned() and shard.scale.is_pinned()
                                     for shard, _, _ in bindings.values()):
            raise RuntimeError("real pinned HostShard/native gather unavailable")
        epoch = 0
        scenarios = (
            ("uniform98304", [4096] * 8, False),
            ("ragged_old_and_new_boundaries", [0, 1, 170, 171, 4095, 4096, 4097, 140], False),
            ("idle_24_dead", [4096 if r % 2 == 0 else 1 for r in range(8)], False),
            ("all_empty", [0] * 8, False),
            ("signed_zero_nan_inf_scales", [171, 170, 1, 140, 3, 7, 0, 171], True),
        )
        for name, counts, nonfinite in scenarios:
            current = nonfinite_bindings if nonfinite else bindings
            raw_ids = _ids(counts[rank], rank, device, idle=name == "idle_24_dead" and rank % 2 == 1)
            id_storage = torch.full((raw_ids.numel() + 16,), -777, device=device, dtype=torch.int64)
            ids = id_storage[8:-8].view(raw_ids.shape)
            ids.copy_(raw_ids)
            ids_before = id_storage.clone()
            owners_before = [(shard.payload.view(torch.uint8).clone(), shard.scale.clone()) for shard, _, _ in current.values()]
            outputs, arms = [], []
            for enabled in (False, True):
                epoch += 1
                evidence = {}
                schedule = engram.prepare_local_prefill_engram_schedule(device, requested=enabled,
                    supported=True, epoch=epoch, process_group=dist.group.WORLD, evidence=evidence, bindings=current)
                observations = dict(native_gather=[], all_reduce=[], reduce_scatter=[])
                original_gather, original_reduce, original_rs = native.engram_gather, dist.all_reduce, dist.reduce_scatter_tensor

                def gather(*args, **kwargs):
                    observations["native_gather"].append(dict(ids=args[2].numel(), output=list(args[3].shape)))
                    return original_gather(*args, **kwargs)

                def reduce(tensor, *args, **kwargs):
                    observations["all_reduce"].append(dict(shape=list(tensor.shape), op=str(kwargs.get("op", dist.ReduceOp.SUM))))
                    return original_reduce(tensor, *args, **kwargs)

                def scatter(out, values, *args, **kwargs):
                    observations["reduce_scatter"].append(dict(input=list(values.shape), output=list(out.shape)))
                    with torch.profiler.record_function("a10.engram.actual_padded_reduce_scatter"):
                        return original_rs(out, values, *args, **kwargs)

                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                             torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
                with patch.object(native, "engram_gather", side_effect=gather), patch.object(
                    dist, "all_reduce", side_effect=reduce,
                ), patch.object(dist, "reduce_scatter_tensor", side_effect=scatter), torch.no_grad(), profile:
                    values = []
                    for layer in (1, 14):
                        shard, layout, adapter = current[layer]
                        values.append(engram.engram_lookup(ids, layer_id=layer, shard=shard, layout=layout,
                                                          group=adapter, **({"round_schedule": schedule} if schedule is not None else {})))
                    torch.cuda.synchronize(device)
                if schedule is not None and not schedule.close():
                    raise AssertionError("incomplete native schedule lease")
                if not _same_bytes(id_storage, ids_before):
                    raise AssertionError("request IDs or neighboring storage changed")
                if any(value.shape != (*ids.shape, 256) or torch._C._is_alias_of(value, ids) for value in values):
                    raise AssertionError("output shape/ownership changed")
                group_max = max(counts) * 24
                rounds = engram._round_lengths(group_max, 98304 if enabled else 4096)
                expected_rounds = 2 * len(rounds)
                if len(observations["native_gather"]) != expected_rounds:
                    raise AssertionError((name, enabled, observations))
                if len(observations["reduce_scatter"]) != (expected_rounds if enabled else 0):
                    raise AssertionError("wrong collective backend/count")
                if len(observations["all_reduce"]) != (2 if enabled else 2 + expected_rounds):
                    raise AssertionError("MAX-count must remain and old SUM must be replaced only when admitted")
                trace = Path(directory) / f"a10-engram-{name}-flag{int(enabled)}-rank{rank}.trace.json"
                profile.export_chrome_trace(str(trace))
                events = json.loads(trace.read_text())["traceEvents"]
                kernels = [event["name"] for event in events if event.get("cat") == "kernel"]
                if sum("_engram_gather_kernel" in value for value in kernels) != expected_rounds:
                    raise AssertionError("actual native gather kernel not proven")
                if expected_rounds and not any("pccl" in value.lower() for value in kernels):
                    raise AssertionError("actual PCCL device kernels missing")
                if enabled and sum(event.get("name") == "a10.engram.actual_padded_reduce_scatter" for event in events) != expected_rounds:
                    raise AssertionError("real reduce-scatter marker missing")
                arms.append(dict(evidence=evidence, observations=observations,
                                 torch_allocator_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                 torch_allocator_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                                 memory_scope="per-arm torch allocator peak; excludes C++/PCCL/host pinned"))
                outputs.append(values)
            for old, new in zip(outputs[0], outputs[1]):
                if not _same_bytes(old, new):
                    raise AssertionError(f"{name}: BF16 bytes differ; no tolerance is permitted")
            for (shard, _, _), before in zip(current.values(), owners_before):
                if not _same_bytes(shard.payload.view(torch.uint8), before[0]) or not _same_bytes(shard.scale, before[1]):
                    raise AssertionError("host shard mutated")
            report["cases"].append(dict(name=name, local_tokens=counts[rank], arms=arms, exact_bf16=True))
            save()
            del outputs, values, old, new
        # All ranks call every proposal, including deliberately unsupported
        # and mismatched owners. No local fallback changes the wire schedule.
        for name in ("flag", "cap", "epoch", "group", "geometry", "fatal"):
            epoch += 1
            evidence = {}
            proposal_bindings = _bindings(rank, dist.group.WORLD, native=True, count=4096) if name == "geometry" and rank == 4 else bindings
            try:
                engram.prepare_local_prefill_engram_schedule(device,
                    requested=not (name == "flag" and rank == 0), supported=True,
                    epoch=epoch + int(name == "epoch" and rank == 2),
                    process_group=object() if name == "group" and rank == 3 else dist.group.WORLD,
                    round_cap=98305 if name == "cap" and rank == 1 else 98304,
                    fatal=name == "fatal" and rank == 4, evidence=evidence, bindings=proposal_bindings)
            except RuntimeError:
                if evidence.get("decision") != "rejected" or len(evidence.get("packet", ())) != 28:
                    raise AssertionError("transport/runtime error is not a coherent protocol rejection")
                report["rejections"].append(dict(name=name, evidence=evidence))
            else:
                raise AssertionError("configuration mismatch was not rejected on every owner")
        epoch += 1
        evidence = {}
        result = engram.prepare_local_prefill_engram_schedule(device, requested=True,
            supported=rank != 5, epoch=epoch, process_group=dist.group.WORLD, evidence=evidence, bindings=bindings)
        if result is not None or evidence["decision"] != "fallback":
            raise AssertionError("one unsupported owner must make all owners choose the old schedule")
        report["unsupported"] = evidence
        fallback_rows = _ids(3, rank, device)
        for layer in (1, 14):
            shard, layout, adapter = bindings[layer]
            engram.engram_lookup(fallback_rows, layer_id=layer, shard=shard, layout=layout, group=adapter)
        # Every owner deliberately invalidates the same lease boundary. The
        # admitted schedule must raise, never silently switch to old SUM.
        for failure in ("closed", "order", "host_version", "native_exception"):
            epoch += 1
            evidence = {}
            schedule = engram.prepare_local_prefill_engram_schedule(device, requested=True,
                supported=True, epoch=epoch, process_group=dist.group.WORLD, evidence=evidence, bindings=bindings)
            if failure == "closed":
                schedule.close()
            elif failure == "host_version":
                scale = bindings[1][0].scale
                scale.copy_(scale.clone())
            shard, layout, adapter = bindings[14 if failure == "order" else 1]
            observed = []
            original_rs = dist.reduce_scatter_tensor

            def unexpected_rs(*args, **kwargs):
                observed.append(True)
                return original_rs(*args, **kwargs)

            try:
                with patch.object(dist, "reduce_scatter_tensor", side_effect=unexpected_rs), patch.object(
                    native, "engram_gather", side_effect=RuntimeError("injected native gather failure")
                    if failure == "native_exception" else native.engram_gather,
                ):
                    engram.engram_lookup(_ids(1, rank, device), layer_id=14 if failure == "order" else 1,
                                         shard=shard, layout=layout, group=adapter, round_schedule=schedule)
            except RuntimeError as error:
                expected = {"closed": "binding/lifecycle", "order": "binding/lifecycle",
                            "host_version": "host binding", "native_exception": "injected native gather failure"}[failure]
                if expected not in str(error):
                    raise
                if observed:
                    raise AssertionError("failed schedule reached reduce-scatter or retried")
                report["rejections"].append(dict(name=failure, evidence=evidence))
            else:
                raise AssertionError("invalid lease/native failure silently fell back")
            finally:
                schedule.close()
        dist.barrier()
        dist.destroy_process_group()
        report["normal_teardown"] = True
        save()
    except BaseException:
        report["error"] = traceback.format_exc()
        save()
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            raise


def _distributed_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--output-dir", default=os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"))
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--pg-timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    if not args.output_dir or args.timeout_seconds <= 0 or args.pg_timeout_seconds <= 0:
        parser.error("output directory and positive bounded timeouts are required")
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    workers = [context.Process(target=_distributed_worker, args=(rank, port, str(directory), args.pg_timeout_seconds)) for rank in range(8)]
    for worker in workers:
        worker.start()
    deadline, failed = time.monotonic() + args.timeout_seconds, False
    try:
        while any(worker.is_alive() for worker in workers):
            if time.monotonic() >= deadline or any(worker.exitcode not in (None, 0) for worker in workers):
                failed = True
                break
            for worker in workers:
                worker.join(timeout=0.1)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=2)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=2)
    teardown = []
    for rank in range(8):
        try:
            report = json.loads((directory / f"a10-engram-rank{rank}.json").read_text())
            teardown.append(report.get("normal_teardown") is True and "error" not in report)
        except (OSError, ValueError):
            teardown.append(False)
    failed = failed or not all(teardown) or any(worker.exitcode != 0 for worker in workers)
    (directory / "a10-engram-parent.json").write_text(json.dumps(dict(
        failed=failed, exitcodes=[worker.exitcode for worker in workers], normal_teardown=teardown,
    ), indent=2))
    if failed:
        raise SystemExit("WORLD8 Engram contract failed; inspect owned worker artifacts")


if __name__ == "__main__":
    if "--distributed" in sys.argv:
        _distributed_main()
    else:
        unittest.main()
