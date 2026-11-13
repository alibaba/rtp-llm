"""DP capacity protocol contracts; no checkpoint or expert weights required.

CPU: python -m rtp_llm.models_py.modules.dsv41.ppu.test.test_dp_prefill_mega_capacity
Real WORLD8: same command plus --distributed --output-dir DIR
The distributed CLI alone spawns eight PPU workers. Unittest never spawns it.
Native SymmBuffer/pre-dispatch evidence proves lifecycle, not expert GEMM output.
"""

import argparse
import copy
import json
import os
import socket
import sys
import time
import traceback
import unittest
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from rtp_llm.eplb.runtime import EplbLayer
from rtp_llm.models_py.modules.dsv41.ppu import moe
from rtp_llm.models_py.modules.dsv41.ppu.ced import _bounded_query_layout
from rtp_llm.models_py.modules.dsv41.ppu.model_runtime import (
    DP_PREFILL_FORWARD_MEGA_CAPACITY_ENV,
    SharedNativeRuntime,
    _dp_prefill_row_bound,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_moe_routes import (
    _state,
    _unit,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_rows import state_for
from rtp_llm.models_py.modules.dsv41.prefill.ced import _bounded_replay_selected


DEVICE = torch.device("cuda:0")
SELECTION = {"RTP_V41_PPU_MOE_BACKEND": "megamoe", "RTP_V41_PPU_MOE_PRECISION": "w4a4"}


def _engine(group, rank=0, *, inter_dim=2304):
    # A07d actual model process.log V4Args (all eight ranks): moe_inter=2304.
    # PPUMoE's generic 2048 default is not the Flash1017 fixture geometry.
    with patch.dict(os.environ, SELECTION):
        return moe.PPUMoE(ep_size=8, ep_rank=rank, group=group, inter_dim=inter_dim)


class _Group:
    def size(self):
        return 8

    def rank(self):
        return 0


class _Buffer:
    def __init__(self, events):
        self.events = events

    def destroy(self):
        self.events.append(("destroy",))


class _CPUWorld:
    def __enter__(self):
        self.stack, self.events, self.packets = ExitStack(), [], []
        self.group, self.mutate = _Group(), None
        enter = self.stack.enter_context
        tensor, zeros = torch.tensor, torch.zeros

        def cpu(factory):
            def call(*args, **kwargs):
                if kwargs.get("device") is not None and torch.device(kwargs["device"]).type == "cuda":
                    kwargs["device"] = "cpu"
                return factory(*args, **kwargs)
            return call

        for target, value in (
            ("group", NS(WORLD=self.group)), ("is_initialized", True),
            ("get_world_size", 8), ("get_rank", 0),
        ):
            enter(patch.object(dist, target, **({"new": value} if target == "group" else {"return_value": value})))
        enter(patch.object(dist, "all_reduce", side_effect=self.reduce))
        enter(patch.object(dist, "barrier", side_effect=lambda **kw: self.events.append(("barrier",))))
        enter(patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))
        enter(patch.object(torch.cuda, "synchronize", side_effect=lambda **kw: self.events.append(("sync",))))
        enter(patch.object(torch, "tensor", side_effect=cpu(tensor)))
        enter(patch.object(torch, "zeros", side_effect=cpu(zeros)))
        self.engine = _engine(self.group)
        self.engine._mega_module = NS(get_symm_buffer_for_mega_moe=self.create)
        return self

    def create(self, *args, **kwargs):
        self.events.append(("create", args[2]))
        return _Buffer(self.events)

    def reduce(self, packet, op, group):
        assert op == dist.ReduceOp.MAX and group is self.group
        self.packets.append(packet.tolist())
        if packet.numel() == 9 and self.mutate is not None:
            self.mutate(packet)

    def __exit__(self, *args):
        try:
            self.engine.finish_dp_prefill_forward(self.engine._dp_active_preparation)
            self.engine.close(group=self.group)
        finally:
            self.stack.close()


def _query(lengths, device):
    prefixes = tuple(17 + i for i in range(len(lengths)))
    state = state_for(torch.device(device), prefixes, lengths)
    context = state.identity_context()
    starts = tuple(max(n - 140, 0) for n in lengths)
    selected = _bounded_replay_selected(lengths, starts)
    return _bounded_query_layout(context, selected)[0]


class DPPrefillCapacityCPU(unittest.TestCase):
    def test_flash1017_geometry_is_explicit_and_wrong2048_fails_before_allocation(self):
        with _CPUWorld() as world:
            self.assertEqual(world.engine.inter_dim, 2304)
            wrong = _engine(world.group, inter_dim=2048)
            wrong._mega_module = world.engine._mega_module
            with self.assertRaisesRegex(RuntimeError, "configuration/lifecycle disagreement") as caught:
                wrong.prepare_dp_prefill_forward(DEVICE, 1, supported=True)
            self.assertEqual(world.events, [])
            self.assertIn("'inter_dim': 2048", str(caught.exception))
            self.assertIn("'configuration_bad': True", str(caught.exception))
            self.assertIn("'runtime_fatal': False", str(caught.exception))
            self.assertIn("reduced_packet=[1, 0, 1, 0, 0, 0, 0, 0, 0]", str(caught.exception))

    def test_bounds_cover_actual_public_compact_padding(self):
        for lengths in ((0, 0), (1,), (4096,), (7, 0, 3), (141, *([1] * 300))):
            bound = _dp_prefill_row_bound(lengths, sum(lengths))
            expected = sum((min(n, 140) + 1) // 2 * 2 for n in lengths)
            self.assertGreaterEqual(bound, sum(lengths))
            self.assertGreaterEqual(bound, expected)
            if sum(lengths):
                query = _query(lengths, "cpu")
                self.assertEqual(query.chunk_length, expected)
                self.assertLessEqual(query.chunk_length, bound)
        self.assertEqual(_dp_prefill_row_bound((4096,), 4096), 4096)
        self.assertEqual(_dp_prefill_row_bound((141, *([1] * 300)), 441), 742)
        self.assertEqual(_dp_prefill_row_bound((1,), 1, idle=True), 1)
        for lengths, rows in (((2,), 1), ((-1,), 0), ((1.5,), 1)):
            with self.assertRaises(ValueError):
                _dp_prefill_row_bound(lengths, rows)

    def test_initialized_legacy_engine_activation_reuse_and_coordinated_growth(self):
        with _CPUWorld() as world:
            engine = world.engine
            old = engine._ensure_mega_capacity(DEVICE, 65)
            self.assertEqual((engine._mega_capacity, engine._mega_generation), (128, 1))
            receipt = engine.prepare_dp_prefill_forward(DEVICE, 66, supported=True)
            self.assertIs(receipt.buffer, old)
            self.assertEqual(world.packets[-1], [66, 0, 0, 1, -1, 128, -128, 1, -1])
            engine.finish_dp_prefill_forward(receipt)
            with self.assertRaises(ValueError):
                engine._validate_dp_preparation(receipt, 1)
            world.mutate = lambda packet: packet[0].fill_(130)
            current = engine.prepare_dp_prefill_forward(DEVICE, 1, supported=True)
            self.assertEqual((current.bound, current.capacity, current.generation), (130, 256, 2))
            self.assertEqual(world.events, [("create", 128), ("sync",), ("barrier",),
                                            ("destroy",), ("create", 256)])
            with self.assertRaises(ValueError):
                engine._validate_dp_preparation(receipt, 1)
            with self.assertRaises(ValueError):
                engine._validate_dp_preparation(current, 131)

    def test_global_unsupported_keeps_all_forty_legacy_max_calls(self):
        with _CPUWorld() as world:
            world.mutate = lambda packet: packet[1].fill_(1)
            self.assertIsNone(world.engine.prepare_dp_prefill_forward(DEVICE, 1, supported=True))
            self.assertEqual(world.events, [])
            for _ in range(40):
                world.engine._ensure_mega_buffer(NS(device=DEVICE, shape=(1, 5120)))
            self.assertEqual([len(packet) for packet in world.packets], [9] + [1] * 40)
            self.assertEqual(world.engine._mega_generation, 1)

    def test_mismatch_and_fatal_precede_fallback_or_allocator(self):
        for field, value in ((4, 0), (5, 256), (7, 2), (2, 1)):
            with self.subTest(field=field), _CPUWorld() as world:
                engine = world.engine
                engine.prepare_dp_prefill_forward(DEVICE, 1, supported=True)
                before = list(world.events)

                def disagree(packet):
                    packet[field] = value
                    packet[1] = 1

                world.mutate = disagree
                with self.assertRaisesRegex(RuntimeError, "disagreement"):
                    engine.prepare_dp_prefill_forward(DEVICE, 1, supported=False)
                self.assertEqual(world.events, before)
                self.assertIsNone(engine._dp_active_preparation)

    @torch.inference_mode()
    def test_all_forty_business_and_idle_consumers_receive_same_lease(self):
        for idle in (False, True):
            with _CPUWorld() as world:
                receipt = world.engine.prepare_dp_prefill_forward(DEVICE, 4, supported=True)
                state = _state()
                state.idle, state.prefill = idle, not idle
                state.dp_prefill_mega_buffer = receipt
                unit = _unit(state, torch.device("cpu"))
                calls = []

                def consume(hidden, ids, weights, **kwargs):
                    calls.append(kwargs["prepared_buffer"])
                    return hidden

                unit.ops.moe = consume
                with patch.dict(os.environ, {"RTP_V41_PPU_LOCAL_PREFILL_DIRECT_MOE_ROUTES": "0"}):
                    for layer in range(40):
                        unit.layer_id = layer
                        unit(torch.ones(3, 32, dtype=torch.bfloat16))
                self.assertEqual(len(calls), 40)
                self.assertTrue(all(value is receipt for value in calls))
                self.assertEqual(len(world.packets), 1)
                if idle:
                    self.assertEqual(list(state.routed_rows.values()), [0] * 40)

    @torch.inference_mode()
    def test_mapping_is_independent_of_typed_receipt_and_raw_cp8(self):
        ids = torch.tensor([[0, 1, 2, -1, 3, 0, -1, -1], [3, 2, 1, 0, -1, -1, -1, -1]])
        weights = (ids >= 0).float()
        for mapping in (torch.tensor([[2], [0], [3], [1]]),
                        torch.tensor([[5, 0], [3, 1], [4, 7], [2, 6]])):
            with _CPUWorld() as world:
                engine = world.engine
                receipt = engine.prepare_dp_prefill_forward(DEVICE, 2, supported=True)
                layer = NS(layer_id=0, logical_to_physical=mapping,
                           logical_count=torch.full((4,), mapping.shape[1], dtype=torch.int64),
                           replica_offset=3)
                layer.map_ids = MethodType(EplbLayer.map_ids, layer)
                engine._weights[0], engine._eplb_layers[0] = (), layer
                hidden = torch.ones(2, 5120, dtype=torch.bfloat16)
                expected = layer.map_ids(ids)
                with patch.dict(os.environ, {"RTP_V41_PPU_CP8_YOCO_FAST_PREFILL": "1"}), patch.object(
                    engine, "_forward_megamoe", return_value=hidden,
                ) as native, patch.object(engine, "_map_expert_ids", wraps=engine._map_expert_ids) as mapped:
                    engine.forward(hidden, ids, weights, layer_id=0)
                    self.assertTrue(torch.equal(native.call_args.args[1], expected))
                    engine.forward(hidden, ids, weights, layer_id=0, prepared_buffer=receipt)
                    self.assertEqual(mapped.call_args.kwargs, {})
                    self.assertTrue(torch.equal(native.call_args.args[1], expected))
                    self.assertIs(native.call_args.kwargs["prepared_buffer"], receipt.buffer)
                    engine.forward(hidden, ids, weights, layer_id=0, prepared_buffer=receipt.buffer)
                    self.assertEqual(mapped.call_args.kwargs, {})
                    self.assertTrue(torch.equal(native.call_args.args[1], expected))
                engine.finish_dp_prefill_forward(receipt)
                with self.assertRaises(ValueError):
                    engine.forward(hidden, ids, weights, layer_id=0, prepared_buffer=receipt)

    @torch.inference_mode()
    def test_runtime_mode_bound_fallback_and_frozen_activation(self):
        engine = NS(prepare_dp_prefill_forward=Mock(return_value=None))
        model = NS(_v4_args=NS(tp_size=1, dp_size=8, ep_size=8, dp_rank=0, ep_rank=0),
                   parallelism_config=NS(tp_size=1), _prefill_cp_size=1,
                   _is_decode_role=False, _is_speculative=False,
                   shared_weight_views=NS(quant_mode="mxfp4_fp8_ptpc"),
                   _shared_ops=NS(moe_engine=engine))
        runtime = NS(dp_prefill_capacity_enabled=False, model=model, device=DEVICE,
                     policy=NS(mode="compact", keep_tokens=140, window=128,
                               max_rollback_tokens=0, encoder_replay=False))
        state = NS(lengths=(141, *([1] * 300)), query_prefixes=(0,) * 301,
                   inputs=NS(input_ids=torch.arange(441)), attn=NS(is_cuda_graph=False,
                   context_parallel_info=object()), idle=False, prefill=True, cache=NS(sharded=False))
        prepare = lambda: SharedNativeRuntime._prepare_dp_prefill_mega_capacity(runtime, state)
        prepare()
        engine.prepare_dp_prefill_forward.assert_not_called()
        runtime.dp_prefill_capacity_enabled = True
        with patch("rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", return_value=True), patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 9),
        ), patch.dict(os.environ, {DP_PREFILL_FORWARD_MEGA_CAPACITY_ENV: "0"}):
            prepare()  # Instance setting, frozen at startup, not a hot env read.
            self.assertEqual(engine.prepare_dp_prefill_forward.call_args.args[1], 742)
            self.assertEqual(engine.prepare_dp_prefill_forward.call_args.kwargs,
                             dict(supported=True, fatal=False))
            state.prefill = False
            prepare()
            self.assertFalse(engine.prepare_dp_prefill_forward.call_args.kwargs["supported"])
            state.idle, state.cache = True, None
            prepare()
            self.assertTrue(engine.prepare_dp_prefill_forward.call_args.kwargs["supported"])
            self.assertEqual(engine.prepare_dp_prefill_forward.call_args.args[1], 441)
            model._is_decode_role = True
            prepare()
            self.assertTrue(engine.prepare_dp_prefill_forward.call_args.kwargs["fatal"])


class _SyntheticGate(torch.nn.Module):
    def forward(self, hidden):
        ids = torch.arange(8, device=hidden.device).expand(hidden.shape[0], 8).clone().long()
        ids[:, 6:] = -1
        weights = (ids >= 0).float() / 6
        return ids, weights


def _vote_runtime(device, stop, observed):
    runtime = NS(_stopped=False, _pending_verify_host_vote=None, device=device,
                 _requested=stop, _observed=observed)
    runtime._finish_vote = MethodType(SharedNativeRuntime._finish_vote, runtime)
    runtime.vote = MethodType(SharedNativeRuntime.vote, runtime)
    return runtime


def _distributed_worker(rank, port, directory, pg_timeout):
    report = {"rank": rank, "world_size": 8, "normal_teardown": False, "scenarios": []}
    path = Path(directory) / f"a07-capacity-rank{rank}.json"
    engine, profile = None, None

    def save():
        path.write_text(json.dumps(report, indent=2))

    try:
        from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu

        if not is_ppu() or torch.cuda.device_count() != 8:
            raise RuntimeError("distributed contract requires exactly eight visible PPU devices")
        torch.cuda.set_device(rank)
        torch.set_num_threads(1)
        device = torch.device("cuda", rank)
        if torch.cuda.get_device_capability(device) != (8, 9):
            raise RuntimeError("distributed contract requires PPU890")
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                                rank=rank, world_size=8, timeout=timedelta(seconds=pg_timeout))
        engine = _engine(dist.group.WORLD, rank)
        report["engine_geometry"] = {
            name: getattr(engine, name) for name in (
                "hidden_size", "inter_dim", "num_routed_experts",
                "num_physical_experts", "num_local_experts",
            )
        }
        save()
        native = engine._get_mega_module()
        vote = _vote_runtime(device, lambda: False, lambda: None)
        profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                     torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
        reductions = []
        actual_reduce = dist.all_reduce

        def observed_reduce(tensor, *args, **kwargs):
            reductions.append(dict(elements=tensor.numel(), op=str(kwargs.get("op", dist.ReduceOp.SUM))))
            return actual_reduce(tensor, *args, **kwargs)

        def frame(name, lengths, *, idle=False, unsupported=False):
            reduction_start = len(reductions)
            actual = 1 if idle else sum(lengths)
            if not vote.vote(0 if idle else actual):
                raise AssertionError("scenario needs at least one business owner")
            bound = _dp_prefill_row_bound(lengths, actual, idle=idle)
            receipt = engine.prepare_dp_prefill_forward(device, bound, supported=not unsupported)
            fallback = name == "mixed-unsupported"
            if (receipt is None) != fallback:
                raise AssertionError("partial or missing prepared peers")
            state = _state()
            state.prefill, state.idle = not idle, idle
            state.dp_prefill_mega_buffer = receipt
            unit = _unit(state, device, width=5120)
            unit.gate = _SyntheticGate()
            calls, last_event = [], None

            def consume(hidden, ids, weights, *, prepared_buffer=None, **kwargs):
                nonlocal last_event
                if fallback:
                    if prepared_buffer is not None:
                        raise AssertionError("fallback skipped a peer's legacy MAX")
                    buffer = engine._ensure_mega_buffer(hidden)
                else:
                    if prepared_buffer is not receipt:
                        raise AssertionError("layer did not consume the common receipt")
                    engine._validate_dp_preparation(receipt, hidden.shape[0])
                    buffer = receipt.buffer
                native.mega_moe_pre_dispatch(hidden.contiguous(), ids.contiguous(), weights.contiguous(), buffer)
                last_event = torch.cuda.Event()
                last_event.record()
                calls.append(hidden.shape[0])
                return hidden

            unit.ops.moe = consume
            query = _query(lengths, device) if not idle else None
            full = torch.ones(actual, 5120, dtype=torch.bfloat16, device=device)
            compact = torch.ones(query.chunk_length, 5120, dtype=torch.bfloat16, device=device) if query else full
            for layer in range(40):
                unit.layer_id = layer
                if layer == 20 and query is not None:
                    state.context, state.compacted = query, True
                unit(compact if layer >= 20 else full)
            if len(calls) != 40 or (idle and any(state.routed_rows.values())):
                raise AssertionError("missing collective consumer or fake business route")
            if receipt is not None:
                engine.finish_dp_prefill_forward(receipt)
                try:
                    engine._validate_dp_preparation(receipt, 1)
                except ValueError:
                    pass
                else:
                    raise AssertionError("finished receipt remained active")
            entry = dict(name=name, idle=idle, local_bound=bound, rows=calls,
                         control=copy.deepcopy(engine._dp_capacity_evidence),
                         actual_capacity=engine._mega_capacity, generation=engine._mega_generation,
                         actual_reductions=reductions[reduction_start:])
            capacity_max = [call for call in entry["actual_reductions"] if call["op"] == str(dist.ReduceOp.MAX)]
            if sum(call["elements"] == 9 for call in capacity_max) != 1:
                raise AssertionError("expected exactly one coordinated capacity vector")
            if sum(call["elements"] == 1 for call in capacity_max) != (40 if fallback else 0):
                raise AssertionError("wrong actual per-layer capacity MAX participation")
            report["scenarios"].append(entry)
            save()
            return last_event

        with patch.dict(os.environ, {"RTP_V41_PPU_LOCAL_PREFILL_DIRECT_MOE_ROUTES": "0",
                                     "RTP_V41_SHARED_EVIDENCE_DIR": ""}), patch.object(
            dist, "all_reduce", side_effect=observed_reduce,
        ), torch.inference_mode(), profile:
            frame("fresh-create", (1,), idle=rank != 0)
            if (engine._mega_capacity, engine._mega_generation) != (128, 1):
                raise AssertionError("new DP API did not create native capacity128/gen1")
            engine.close(group=dist.group.WORLD)
            # Dynamic activation after the old allocator has already warmed.
            engine._ensure_mega_capacity(device, 1)
            old = engine._mega_buffer
            frame("activate-existing", (1,), idle=rank != 0)
            if engine._mega_buffer is not old or engine._mega_capacity != 128:
                raise AssertionError("activation rebuilt an adequate legacy buffer")
            event = frame("reuse-ragged", (3, 0, 5), idle=rank in (1, 6))
            frame("grow-one-owner", (129,) if rank == 3 else (7,), idle=rank in (1, 6))
            if event is not None and not event.query():
                raise AssertionError("growth failed to drain old native users")
            frame("padding-over-full", (141, *([1] * 300)) if rank == 5 else (9,), idle=rank == 1)
            frame("target4096", (4096,) if rank == 0 else (17, 3), idle=rank in (2, 4, 7))
            if engine._mega_capacity != 4096:
                raise AssertionError("actual target capacity differs from4096")
            frame("reuse4096", (13, 1), idle=rank in (0, 3))
            frame("mixed-unsupported", (5, 1), idle=rank == 2, unsupported=rank == 7)
            for kind in ("presence", "capacity", "generation", "fatal"):
                saved = engine._mega_buffer, engine._mega_capacity, engine._mega_generation
                try:
                    if rank == 2:
                        if kind == "presence":
                            engine._mega_buffer, engine._mega_capacity = None, 0
                        elif kind == "capacity":
                            engine._mega_capacity *= 2
                        elif kind == "generation":
                            engine._mega_generation += 1
                    try:
                        engine.prepare_dp_prefill_forward(device, 1, supported=False,
                                                         fatal=kind == "fatal" and rank == 2)
                    except RuntimeError as error:
                        report["scenarios"].append(dict(name=f"mismatch-{kind}", error=str(error)))
                    else:
                        raise AssertionError("mismatch did not fail uniformly")
                finally:
                    engine._mega_buffer, engine._mega_capacity, engine._mega_generation = saved
                dist.barrier()
                save()
            before = engine._dp_prepare_epoch
            if vote.vote(0):
                raise AssertionError("all-idle business vote should skip allocation")
            stopping = []
            stop_vote = _vote_runtime(device, lambda: rank == 0, lambda: stopping.append(True))
            if stop_vote.vote(1 if rank == 0 else 0) or not stopping:
                raise AssertionError("original stop vote semantics changed")
            if engine._dp_prepare_epoch != before:
                raise AssertionError("all-idle/stop entered the capacity protocol")
            report["scenarios"].append(dict(name="all-idle-and-stop", no_capacity_packet=True))
            dist.barrier()
            engine.close(group=dist.group.WORLD)
            report["normal_teardown"] = engine._mega_buffer is None
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(Path(directory) / f"a07-capacity-rank{rank}.trace.json"))
        report["backend"] = dist.get_backend()
        report["scope"] = "PCCL+native SymmBuffer/pre-dispatch; no expert GEMM/model-output claim"
        save()
        dist.destroy_process_group()
    except BaseException:
        report["error"] = traceback.format_exc()
        save()  # Save before any possibly blocking native cleanup.
        raise


def _distributed_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--output-dir", default=os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--pg-timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    if not args.output_dir or args.timeout_seconds <= 0 or args.pg_timeout_seconds <= 0:
        parser.error("an output directory and positive bounded timeouts are required")
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    children = [context.Process(target=_distributed_worker,
                                args=(rank, port, str(directory), args.pg_timeout_seconds))
                for rank in range(8)]
    for child in children:
        child.start()
    deadline, failed = time.monotonic() + args.timeout_seconds, False
    try:
        while any(child.is_alive() for child in children):
            if time.monotonic() >= deadline or any(child.exitcode not in (None, 0) for child in children):
                failed = True
                break
            for child in children:
                child.join(timeout=0.1)
    finally:
        # Terminate only this invocation's own children, never another GPU job.
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join(timeout=2)
            if child.is_alive():
                child.kill()
                child.join(timeout=2)
    completed = []
    for rank in range(8):
        try:
            receipt = json.loads((directory / f"a07-capacity-rank{rank}.json").read_text())
            completed.append(receipt.get("normal_teardown") is True and "error" not in receipt)
        except (OSError, ValueError):
            completed.append(False)
    failed = failed or any(child.exitcode != 0 for child in children) or not all(completed)
    result = dict(failed=failed, exitcodes=[child.exitcode for child in children],
                  normal_teardown=completed, timeout_seconds=args.timeout_seconds)
    (directory / "a07-capacity-parent.json").write_text(json.dumps(result, indent=2))
    if failed:
        raise SystemExit("DP capacity WORLD8 contract failed; inspect per-rank artifacts")


if __name__ == "__main__":
    if "--distributed" in sys.argv:
        _distributed_main()
    else:
        unittest.main()
