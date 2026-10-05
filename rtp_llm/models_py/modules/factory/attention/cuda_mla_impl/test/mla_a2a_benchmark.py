"""Communication/merge diagnostic using the production O/LSE buffer owner.

Alternating inputs are checked at every consumption inside a Graph. Optional
cold timing includes source preparation and is not full-MLA performance evidence.
"""
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time
import traceback
import unittest

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.mla_topology_benchmark import (
    _mla_bench_evict_l2, _resource_snapshot,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.page_rr_mla_decode_test import (
    _layer_collective,
)
from rtp_llm.models_py.distributed.collective_torch import (
    Group, all_to_all_single, destroy_distributed_environment,
    get_process_group, init_distributed_environment,
)
# The factory import above initializes the declared CuTe wheel path.
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl._fia2a.ops import COUNT, EPOCH, READY
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl._fia2a.workspace import _AllToAllBuffers
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_dcp_comm import _pack_a2a, _combine_a2a
from rtp_llm.ops import NcclCommConfig, ParallelismConfig, RoleType
from rtp_llm.test.utils.port_util import PortManager


@triton.jit
def _a2a_check(Actual, Expected, Errors, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    actual = tl.load(Actual + index, index < N, other=0).to(tl.float32)
    expected = tl.load(Expected + index, index < N, other=0).to(tl.float32)
    valid = tl.abs(actual - expected) <= 2e-3 + .015 * tl.abs(expected)
    count = tl.sum((~valid).to(tl.int32), 0)
    tl.atomic_add(Errors, count)


@triton.jit
def _a2a_arrival_delay(NANOSECONDS: tl.constexpr):
    start = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [],
                                     dtype=tl.int64, is_pure=False, pack=1)
    now = start
    while now - start < NANOSECONDS:
        now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [],
                                       dtype=tl.int64, is_pure=False, pack=1)


def _worker(rank, world, tp, port, cases, directory, failed):
    torch.cuda.set_device(rank)
    parallel = ParallelismConfig()
    parallel.world_size = parallel.local_world_size = world
    parallel.world_rank = parallel.local_rank = rank
    parallel.tp_size, parallel.dp_size = tp, world // tp
    parallel.tp_rank, parallel.dp_rank = rank % tp, rank // tp
    parallel.role_type = RoleType.DECODE
    init_distributed_environment(parallel, NcclCommConfig(nccl_ip="127.0.0.1"), port, timeout=300)
    group = get_process_group(Group.TP)
    device = torch.device("cuda", rank)
    graph = None
    try:
        prop = torch.cuda.get_device_properties(rank)
        flush_bytes = max(256 * 1024 * 1024, 2 * prop.L2_cache_size)
        flush = torch.empty(flush_bytes, dtype=torch.uint8, device=device)
        def evict():
            _mla_bench_evict_l2[(triton.cdiv(flush_bytes, 1024),)](flush, flush_bytes, 1024)
        evict()
        inner, tail = 33, 16
        for case in cases:
            tokens, heads = case["T"], case.get("H", 96)
            assert heads % tp == 0 and tokens > 0
            local_heads = heads // tp
            buffers = _AllToAllBuffers(heads, max(16, 1 << (tokens - 1).bit_length()), device, group)
            source_o, source_lse = buffers.output[0, :tokens], buffers.lse[0, :tokens]
            packed = torch.empty((tp, tokens, local_heads, 514), device=device, dtype=torch.bfloat16)
            torch.manual_seed(20261002 + 100 * parallel.dp_rank + parallel.tp_rank)
            partials = [torch.randn_like(source_o) for _ in range(2)]
            lses = [torch.randn_like(source_lse) for _ in range(2)]
            lengths = torch.ones(tokens, dtype=torch.int32, device=device)
            lengths[::7] = 0
            for values, logs in zip(partials, lses):
                # An empty shard's O is deliberately unusable: a zero weight
                # alone cannot suppress NaN if the implementation still reads it.
                values[::7] = float("nan")
                logs[::7] = -float("inf")

            def prepare(generation):
                # The model's TP collective orders source writes after peers'
                # reads; this direct harness has none, so it waits explicitly.
                _layer_collective()
                source_o.copy_(partials[generation])
                source_lse.copy_(lses[generation])

            def operation(mode):
                if mode == "NCCL":
                    _pack_a2a[(tokens, heads)](source_o, source_lse, lengths,
                        packed, packed.view(torch.float32), tokens,
                        heads=heads, local_heads=local_heads, dim=512, block=512)
                    received = all_to_all_single(packed, Group.TP)
                    output = source_o.new_empty(local_heads, tokens, 512)
                    _combine_a2a[(tokens, local_heads)](received, received.view(torch.float32), output,
                        tokens, local_heads=local_heads, dim=512, cp_size=tp, block=512)
                    return output
                else:
                    return buffers.combine(tokens)

            expected = []
            for generation in range(2):
                prepare(generation)
                expected.append(operation("NCCL").clone())
            torch.cuda.synchronize()
            dist.barrier(group=group)
            modes = case.get("modes", ["NCCL", "PULL_MERGE"])
            for mode in modes:
                assert mode in ("NCCL", "PULL_MERGE")
                case_id = f"tp{tp}_h{heads}_t{tokens}"
                path = Path(directory) / case_id / mode / f"rank{rank}"
                path.mkdir(parents=True, exist_ok=True)
                control_before = buffers.control.cpu().tolist()
                errors = torch.zeros((), dtype=torch.int64, device=device)
                # Rotate the delayed producer between generations/ranks. This
                # delay is a correctness stressor, never a main MLA timing input.
                skew_ns = int(case.get("skew_ns", 200000))
                def checked(generation):
                    prepare(generation)
                    if parallel.tp_rank == generation % tp and skew_ns:
                        _a2a_arrival_delay[(1,)](skew_ns, num_warps=1)
                    actual = operation(mode)
                    elements = actual.numel()
                    _a2a_check[(triton.cdiv(elements, 1024),)](
                        actual, expected[generation], errors, elements, 1024)
                    return actual
                for generation in range(2):
                    actual = checked(generation)
                    torch.testing.assert_close(actual, expected[generation], atol=2e-3, rtol=.015)
                graph = torch.cuda.CUDAGraph()
                dist.barrier(group=group)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    for iteration in range(8):
                        checked(iteration % 2)
                for _ in range(32):
                    graph.replay()
                torch.cuda.synchronize()
                assert errors.item() == 0, (rank, case, mode, errors.item())
                control_after = buffers.control.cpu().tolist()
                if mode == "PULL_MERGE":
                    # Two eager calls and 32 replays of eight calls. Capture
                    # itself must not advance any publication/consumption epoch.
                    epoch = control_before[EPOCH] + 2 + 32 * 8
                    expected_control = list(control_before)
                    expected_control[EPOCH], expected_control[COUNT] = epoch, 0
                    for slot in range(READY, READY + tp):
                        expected_control[slot] = epoch
                else:
                    expected_control = control_before
                assert control_after == expected_control, (rank, case, mode, control_after, expected_control)
                (path / "correctness.json").write_text(json.dumps(dict(
                    rank=rank, tp=tp, T=tokens, H=heads, mode=mode,
                    control_before=control_before, control_after=control_after,
                    eager_calls=2, graph_calls=32 * 8, errors=0, skew_ns=skew_ns,
                ), indent=2) + "\n")
                graph.reset()
                graph = None
                print(f"A2A CORRECT rank={rank} tp={tp} T={tokens} H={heads} {mode}", flush=True)
                if case.get("correctness_only", False):
                    continue

                if rank == 0:
                    (path.parent / "resource-before.json").write_text(json.dumps(_resource_snapshot()))
                # Both arms include the same copies from the synthetic immutable
                # inputs; CUSTOM then enters the production Pull+Merge kernel.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(inner):
                        evict()
                        prepare(0)
                        actual = operation(mode)
                for _ in range(2):
                    graph.replay()
                torch.cuda.synchronize()
                def events(count):
                    result = []
                    for _ in range(count):
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record(); graph.replay(); end.record(); end.synchronize()
                        result.append(begin.elapsed_time(end)*1000)
                    return result
                planning = events(3)
                count = torch.tensor(math.ceil(500000 / statistics.median(planning)), device=device, dtype=torch.int32)
                dist.all_reduce(count, op=dist.ReduceOp.MAX)
                settle = int(count.item())
                for _ in range(settle):
                    graph.replay()
                bare = events(5)
                dist.barrier(group=group)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    dist.barrier(group=group)
                    torch.cuda.synchronize()
                    with torch.profiler.record_function("MLA_A2A_SETTLE"):
                        for _ in range(settle):
                            graph.replay()
                    with torch.profiler.record_function("MLA_A2A_MEASURE"):
                        graph.replay()
                    torch.cuda.synchronize()
                profiler.export_chrome_trace(str(path / "trace.json"))
                torch.testing.assert_close(actual, expected[0], atol=2e-3, rtol=.015)
                manifest = dict(case_id=case_id, rank=rank, world=world, tp=tp, owner=parallel.dp_rank,
                    mode=mode, T=tokens, H=heads, cache="cold", flush_bytes=flush_bytes,
                    L2_bytes=prop.L2_cache_size, inner=inner, tail=tail,
                    selected_iterations=list(range(inner-1-tail, inner-1)),
                    scope="synthetic source copies through communication and LSE merge; not full MLA",
                    protocol="direct source Pull+Merge with embedded ready/done" if mode == "PULL_MERGE"
                             else "production pack -> all_to_all_single -> combine",
                    correctness=dict(alternating_generations=2, graph_bodies=8, graph_replays=32,
                                     per_consumption_errors=0, empty_source_o_nan=True, skew_ns=skew_ns),
                    bare_outer_event_us=bare, settling_replays=settle,
                    torch=torch.__version__, capability=torch.cuda.get_device_capability(rank))
                (path / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
                if rank == 0:
                    (path.parent / "resource-after.json").write_text(json.dumps(_resource_snapshot()))
                print(f"A2A BENCH PASS rank={rank} {case_id} {mode}", flush=True)
                graph.reset()
                graph = None
                dist.barrier(group=group)
            buffers = expected = partials = lses = None
    except BaseException:
        error = traceback.format_exc()
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / f"failure-rank{rank}.json").write_text(json.dumps(dict(error=error, rank=rank)))
        print(error, flush=True)
        failed.set()
        raise
    finally:
        if graph is not None:
            graph.reset()
        destroy_distributed_environment()


class MlaAllToAllBenchmark(unittest.TestCase):
    def test_communication_and_merge(self):
        world = int(os.environ.get("MLA_A2A_WORLD", "8"))
        tp = int(os.environ.get("MLA_A2A_TP", str(world)))
        self.assertIn(tp, (4, 8))
        self.assertEqual(world % tp, 0)
        self.assertGreaterEqual(torch.cuda.device_count(), world)
        cases = json.loads(os.environ["MLA_A2A_CASES"])
        directory = os.environ["MLA_A2A_OUTPUT"]
        mp.set_start_method("spawn", force=True)
        failed = mp.Event()
        ports, locks = PortManager().get_consecutive_ports(1)
        workers = [mp.Process(target=_worker, args=(rank, world, tp, ports[0], cases, directory, failed),
                              name=f"mla-a2a-rank-{rank}") for rank in range(world)]
        try:
            for worker in workers:
                worker.start()
            pending = list(workers)
            deadline = time.monotonic() + 4800
            while pending:
                self.assertFalse(failed.is_set(), "A2A worker failed; see failure-rank*.json")
                for worker in pending[:]:
                    if worker.exitcode is not None:
                        worker.join()
                        self.assertEqual(worker.exitcode, 0, worker.name)
                        pending.remove(worker)
                self.assertLess(time.monotonic(), deadline, "A2A workers timed out")
                if pending:
                    time.sleep(.1)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            for worker in workers:
                if worker.pid is not None:
                    worker.join(timeout=10)
                    if worker.is_alive():
                        worker.kill()
                        worker.join(timeout=10)
            for lock in locks:
                lock.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
