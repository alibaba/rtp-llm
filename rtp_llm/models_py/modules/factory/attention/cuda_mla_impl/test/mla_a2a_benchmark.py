"""Collective diagnostic: production packed MLA payload, cold CUDA Graphs.

The pull path uses the production buffer owner and kernel. This target does
not measure a full attention implementation.
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
from rtp_llm.models_py.distributed.collective_torch import (
    Group, all_to_all_single, destroy_distributed_environment,
    get_process_group, init_distributed_environment,
)
# The factory import above initializes the declared CuTe wheel path.
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl._fia2a.workspace import _AllToAllBuffers
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_dcp_comm import _pack_a2a, _combine_a2a
from rtp_llm.ops import NcclCommConfig, ParallelismConfig, RoleType
from rtp_llm.test.utils.port_util import PortManager


@triton.jit
def _a2a_check(Actual, Expected, Errors, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    actual = tl.load(Actual + index, index < N, other=0)
    expected = tl.load(Expected + index, index < N, other=0)
    count = tl.sum((actual != expected).to(tl.int32), 0)
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
            shape = (tp, tokens, local_heads, 514)
            buffers = _AllToAllBuffers(heads, max(16, 1 << (tokens - 1).bit_length()), device, group)
            packed = buffers.packed(tokens)
            torch.manual_seed(20261002 + 100 * parallel.dp_rank + parallel.tp_rank)
            partials = [torch.randn(tokens, heads, 512, device=device, dtype=torch.bfloat16) for _ in range(2)]
            lses = [torch.randn(tokens, heads, device=device) for _ in range(2)]
            lengths = torch.ones(tokens, dtype=torch.int32, device=device)
            lengths[::7] = 0
            # Normal finite input for the production combine; a separate raw-bit
            # check below includes infinities and NaN payloads in the LSE slots.
            def pack(generation=0):
                _pack_a2a[(tokens, heads)](partials[generation], lses[generation], lengths,
                    packed, packed.view(torch.float32), tokens,
                    heads=heads, local_heads=local_heads, dim=512, block=512)
            def combine(received):
                _combine_a2a[(tokens, local_heads)](received, received.view(torch.float32), output,
                    tokens, local_heads=local_heads, dim=512, cp_size=tp, block=512)
            output = torch.empty(local_heads, tokens, 512, dtype=torch.bfloat16, device=device)
            nccl_output = torch.empty_like(packed)
            expected, expected_combined = [], []
            for generation in range(2):
                pack(generation)
                # Reference source order is constructed independently of the copy
                # kernel, outside all correctness captures and measurements.
                gathered = [torch.empty_like(packed) for _ in range(tp)]
                dist.all_gather(gathered, packed, group=group)
                received = torch.stack([x[parallel.tp_rank] for x in gathered])
                expected.append(received)
                combine(received)
                expected_combined.append(output.clone())
                del gathered
            words = packed.numel() // 2
            errors = torch.zeros((), dtype=torch.int64, device=device)
            modes = case.get("modes", ["NCCL", "PULL4096"])
            for mode in modes:
                assert mode in ("NCCL", "PULL4096")
                block = 4096 if mode == "PULL4096" else None
                packed = torch.empty(shape, dtype=torch.bfloat16, device=device) if mode == "NCCL" else buffers.packed(tokens)
                copy_kernel = buffers.warmup(tokens) if block else None
                torch.cuda.synchronize()
                dist.barrier(group=group)
                def exchange():
                    if mode == "NCCL":
                        return all_to_all_single(packed, Group.TP, output=nccl_output)
                    else:
                        return buffers.exchange(tokens)
                def checked(generation):
                    pack(generation)
                    actual = exchange()
                    _a2a_check[(triton.cdiv(words, 1024),)](
                        actual.view(torch.int32), expected[generation].view(torch.int32), errors, words, 1024)
                    combine(actual)
                    output_words = output.numel() // 2
                    _a2a_check[(triton.cdiv(output_words, 1024),)](
                        output.view(torch.int32), expected_combined[generation].view(torch.int32),
                        errors, output_words, 1024)
                for generation in range(2):
                    checked(generation)
                    torch.testing.assert_close(output, expected_combined[generation], atol=0, rtol=0)
                # Each consumption is checked inside the graph, so a later good
                # exchange cannot hide an earlier overwrite/epoch error.
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    for iteration in range(8):
                        checked(iteration % 2)
                for _ in range(16):
                    graph.replay()
                torch.cuda.synchronize()
                assert errors.item() == 0, (rank, case, mode, errors.item())
                graph.reset()
                graph = None
                # Bit-exact transfer must preserve exceptional FP32 codes too.
                pack()
                lse_bits = packed.view(torch.int32).reshape(-1, 257)[:, -1]
                codes = torch.tensor([0x7f800000, -8388608, 0x7fc12345, -2147483648], dtype=torch.int32, device=device)
                lse_bits.copy_(codes.repeat(triton.cdiv(lse_bits.numel(), 4))[:lse_bits.numel()])
                gathered = [torch.empty_like(packed) for _ in range(tp)]
                dist.all_gather(gathered, packed, group=group)
                exceptional = torch.stack([x[parallel.tp_rank] for x in gathered])
                actual = exchange()
                assert torch.equal(actual.view(torch.int32), exceptional.view(torch.int32))
                del gathered, exceptional
                if block and rank == 0:
                    assembly = Path(directory) / "assembly"
                    assembly.mkdir(parents=True, exist_ok=True)
                    prefix = f"tp{tp}_h{heads}_t{tokens}_{mode}"
                    for suffix in ("ptx", "ttir"):
                        (assembly / f"{prefix}.{suffix}").write_text(copy_kernel.asm[suffix])
                print(f"A2A CORRECT rank={rank} tp={tp} T={tokens} H={heads} {mode}", flush=True)
                if case.get("correctness_only", False):
                    continue
                for delays in case.get("delay_ns", [[0] * tp]):
                    assert len(delays) == tp
                    delay_ns = int(delays[parallel.tp_rank])
                    delay_id = "aligned" if not any(delays) else "skew" + str(max(delays))
                    arm = f"{mode}_{delay_id}"
                    case_id = f"tp{tp}_h{heads}_t{tokens}"
                    path = Path(directory) / case_id / arm / f"rank{rank}"
                    path.mkdir(parents=True, exist_ok=True)
                    if rank == 0:
                        (path.parent / "resource-before.json").write_text(json.dumps(_resource_snapshot()))
                    if delay_ns:
                        _a2a_arrival_delay[(1,)](delay_ns, num_warps=1)
                    # Finish host resource queries and per-rank JIT before any
                    # eager peer barrier. Neither belongs to the timed interval.
                    torch.cuda.synchronize()
                    dist.barrier()
                    def body():
                        pack()
                        # Common experimental arrival control, excluded from the
                        # collective interval for BOTH arms. The candidate still
                        # pays its own complete two-barrier protocol afterwards.
                        buffers.synchronize()
                        if delay_ns:
                            _a2a_arrival_delay[(1,)](delay_ns, num_warps=1)
                        received = exchange()
                        combine(received)
                    body()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    dist.barrier()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(graph):
                        for _ in range(inner):
                            evict()
                            body()
                    for _ in range(2):
                        graph.replay()
                    torch.cuda.synchronize()
                    manifest = dict(case_id=case_id, rank=rank, world=world, tp=tp,
                        owner=parallel.dp_rank, tp_rank=parallel.tp_rank, mode=mode,
                        source_allocation="ordinary torch" if mode == "NCCL" else "symmetric VMM",
                        T=tokens, H=heads, shape=shape, dtype="BF16 wire / int32 bit copy",
                        remote_bytes_per_rank=packed.nbytes*(tp-1)//tp,
                        self_bytes_per_rank=packed.nbytes//tp, block_words=block,
                        cache="cold", flush_bytes=flush_bytes, L2_bytes=prop.L2_cache_size,
                        inner=inner, tail=tail, selected_iterations=list(range(inner-1-tail, inner-1)),
                        delay_ns=delays, scope="pack -> common alignment -> arrival delay -> collective -> combine",
                        alignment="one common GPU barrier outside the collective interval; diagnostic only",
                        protocol="publish barrier + integer pull + completion barrier" if block else "production all_to_all_single",
                        correctness=dict(alternating_generations=2, graph_bodies=8, graph_replays=16,
                                         per_consumption_bit_errors=0, per_combine_bit_errors=0,
                                         exceptional_lse_bits=True),
                        torch=torch.__version__, nccl=torch.cuda.nccl.version(),
                        capability=torch.cuda.get_device_capability(rank))
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
                    manifest["bare_outer_event_us"] = events(5)
                    manifest["settling_replays"] = settle
                    dist.barrier()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
                        dist.barrier()
                        torch.cuda.synchronize()
                        with torch.profiler.record_function("MLA_A2A_SETTLE"):
                            for _ in range(settle):
                                graph.replay()
                        with torch.profiler.record_function("MLA_A2A_MEASURE"):
                            graph.replay()
                        torch.cuda.synchronize()
                    profiler.export_chrome_trace(str(path / "trace.json"))
                    torch.testing.assert_close(output, expected_combined[0], atol=0, rtol=0)
                    (path / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
                    if rank == 0:
                        (path.parent / "resource-after.json").write_text(json.dumps(_resource_snapshot()))
                    print(f"A2A BENCH PASS rank={rank} {case_id} {arm}", flush=True)
                    graph.reset()
                    graph = None
                    dist.barrier()
            # Release case storage after resetting every graph. Keep closure
            # cells bound until their functions are replaced by the next case.
            buffers = expected = expected_combined = partials = lses = None
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
    def test_packed_collective(self):
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
