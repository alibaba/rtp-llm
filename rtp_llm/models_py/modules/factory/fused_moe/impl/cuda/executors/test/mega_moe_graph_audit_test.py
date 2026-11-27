"""Empirical CUDA Graph node & profiler audit for MegaMoE & MegaMoE-SE."""

from __future__ import annotations

import ctypes
import os
import tempfile
import time
import unittest
from collections import Counter, defaultdict
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
    _mega_moe_se_available,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test import (
    mega_moe_routed_only_execution_test as moe_test,
)

_NODE_TYPE_NAMES = {
    0: "kernel",
    1: "memcpy",
    2: "memset",
    3: "host",
    4: "graph",
    5: "empty",
    6: "event_wait",
    7: "event_record",
}


class _CudaMemsetParams(ctypes.Structure):
    _fields_ = [
        ("dst", ctypes.c_void_p),
        ("pitch", ctypes.c_size_t),
        ("value", ctypes.c_uint),
        ("elementSize", ctypes.c_uint),
        ("width", ctypes.c_size_t),
        ("height", ctypes.c_size_t),
    ]


_CUDART = ctypes.CDLL("libcudart.so")
_CUDART.cudaGraphGetNodes.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
]
_CUDART.cudaGraphNodeGetType.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
]
_CUDART.cudaGraphMemsetNodeGetParams.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_CudaMemsetParams),
]


def _inspect_cuda_graph(graph: torch.cuda.CUDAGraph) -> dict:
    raw = ctypes.c_void_p(graph.raw_cuda_graph())
    num_nodes = ctypes.c_size_t(0)
    rc = _CUDART.cudaGraphGetNodes(raw, None, ctypes.byref(num_nodes))
    if rc != 0:
        raise RuntimeError(f"cudaGraphGetNodes(count) failed rc={rc}")
    n = num_nodes.value
    nodes = (ctypes.c_void_p * n)()
    rc = _CUDART.cudaGraphGetNodes(raw, nodes, ctypes.byref(num_nodes))
    if rc != 0:
        raise RuntimeError(f"cudaGraphGetNodes(list) failed rc={rc}")

    type_counts: Counter[str] = Counter()
    memset_details: Counter[tuple[int, int, int, int, int]] = Counter()
    memset_total_bytes = 0
    for i in range(n):
        ntype = ctypes.c_int(-1)
        _CUDART.cudaGraphNodeGetType(nodes[i], ctypes.byref(ntype))
        tname = _NODE_TYPE_NAMES.get(ntype.value, f"other_{ntype.value}")
        type_counts[tname] += 1
        if ntype.value == 2:
            p = _CudaMemsetParams()
            _CUDART.cudaGraphMemsetNodeGetParams(nodes[i], ctypes.byref(p))
            nbytes = int(p.elementSize) * int(p.width) * int(p.height)
            memset_total_bytes += nbytes
            memset_details[
                (int(p.elementSize), int(p.value), int(p.width), int(p.height), nbytes)
            ] += 1

    return {
        "total_nodes": n,
        "type_counts": dict(type_counts),
        "memset_nodes": type_counts.get("memset", 0),
        "memcpy_nodes": type_counts.get("memcpy", 0),
        "kernel_nodes": type_counts.get("kernel", 0),
        "memset_total_bytes": memset_total_bytes,
        "memset_details": dict(memset_details),
    }


def _extract_profiler_audit(prof: torch.profiler.profile) -> dict:
    interesting = (
        "memset",
        "memcpy",
        "zero_",
        "fill_",
        "copy_",
        "index_select",
        "indexselect",
        "eventsynchronize",
    )
    hits: dict[str, Counter[str]] = defaultdict(Counter)
    cuda_kernels: Counter[str] = Counter()
    for evt in prof.events():
        name_l = evt.name.lower()
        if any(tok in name_l for tok in interesting):
            stack = evt.stack[:4] if evt.stack else []
            stack_str = " <- ".join(str(s) for s in stack) or "<cpp>"
            hits[evt.name][stack_str] += 1
        if evt.device_type == torch.profiler.DeviceType.CUDA:
            cuda_kernels[evt.name] += 1
    return {
        "cpu_hits": {k: dict(v.most_common(8)) for k, v in hits.items()},
        "cuda_kernels": dict(cuda_kernels.most_common(25)),
    }


def _audit_distributed(
    rank: int,
    label: str,
    fn,
    device: torch.device,
    *,
    warmup_iters: int = 3,
    replay_iters: int = 50,
    profile_capture: bool = True,
) -> dict:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize(device)
    capture_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(capture_stream):
        moe_test.prepare_capture_stream_barrier(device)
    moe_test._sync_ranks_for_mega(device)

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    prof_res = None
    t0 = time.perf_counter()
    if profile_capture and rank == 0:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
        ) as prof:
            with torch.inference_mode(), torch.cuda.graph(graph, stream=capture_stream):
                out = fn()
        prof_res = _extract_profiler_audit(prof)
    else:
        with torch.inference_mode(), torch.cuda.graph(graph, stream=capture_stream):
            out = fn()
    torch.cuda.synchronize(device)
    dist.barrier()
    capture_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize(device)
    dist.barrier()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    latencies_us: list[float] = []
    for _ in range(replay_iters):
        start_ev.record()
        graph.replay()
        end_ev.record()
        end_ev.synchronize()
        latencies_us.append(start_ev.elapsed_time(end_ev) * 1000.0)
    dist.barrier()
    latencies_us.sort()
    info = _inspect_cuda_graph(graph)
    if rank == 0:
        print(
            f"[MOE_AUDIT] {label}: nodes={info['total_nodes']} "
            f"(kernel={info['kernel_nodes']}, memset={info['memset_nodes']}, "
            f"memcpy={info['memcpy_nodes']}) memset_bytes={info['memset_total_bytes']} "
            f"capture={capture_ms:.2f}ms replay_med={latencies_us[len(latencies_us)//2]:.2f}us",
            flush=True,
        )
        if info["memset_details"]:
            print(
                f"  memset_details[(elem,val,w,h,bytes)]: {info['memset_details']}",
                flush=True,
            )
        if prof_res:
            if prof_res["cpu_hits"]:
                print(f"  cpu_hits: {prof_res['cpu_hits']}", flush=True)
            print(f"  cuda_kernels: {prof_res['cuda_kernels']}", flush=True)

    graph.reset()
    del out, graph
    return info


def _audit_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    os.environ["MODEL_WARM_UP"] = "0"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=timedelta(seconds=180),
    )
    try:
        for include_shared, hash_routing, tag in (
            (False, False, "routed"),
            (True, False, "se"),
            (False, True, "hash"),
        ):
            torch.manual_seed(20260901)
            global_weights = moe_test._make_global_weights(
                device, include_shared=include_shared, hash_routing=hash_routing
            )
            local_start = rank * (moe_test._EXPERTS // world_size)
            local_end = local_start + (moe_test._EXPERTS // world_size)
            local_weights = {
                moe_test.W.moe_gate: global_weights[moe_test.W.moe_gate],
                moe_test.W.moe_gate_bias: global_weights[moe_test.W.moe_gate_bias],
            }
            if hash_routing:
                local_weights[moe_test.W.moe_gate_tid2eid] = global_weights[
                    moe_test.W.moe_gate_tid2eid
                ]
            for key in (
                moe_test.W.moe_w1,
                moe_test.W.moe_s1,
                moe_test.W.moe_w2,
                moe_test.W.moe_s2,
            ):
                local_weights[key] = global_weights[key][local_start:local_end].clone()
            if include_shared:
                for key in (
                    moe_test.W.ffn_w13,
                    moe_test.W.ffn_s13,
                    moe_test.W.ffn_w2,
                    moe_test.W.ffn_s2,
                ):
                    local_weights[key] = global_weights[key].clone()
            del global_weights
            layers = [
                moe_test._make_moe(
                    dict(local_weights),
                    ep_size=world_size,
                    ep_rank=rank,
                    strategy=None,
                    n_shared_experts=int(include_shared),
                    hash_routing=hash_routing,
                    layer_id=7 + i,
                    max_tokens_per_rank=128,
                    chunking_enabled=False,
                )
                for i in range(4)
            ]
            if rank == 0:
                print(
                    f"[MOE_SETUP] tag={tag} executor={type(layers[0].fused_moe.fused_experts).__name__} "
                    f"se_avail={_mega_moe_se_available()}",
                    flush=True,
                )
            for bs in (1, 8, 32, 64, 96):
                x = torch.randn(bs, moe_test._DIM, dtype=torch.bfloat16, device=device)
                input_ids = torch.arange(bs, dtype=torch.long, device=device) % 64

                def run_1():
                    with torch.inference_mode():
                        return layers[0](x, input_ids)

                res1 = _audit_distributed(
                    rank,
                    f"moe_{tag}_1layer_bs{bs}",
                    run_1,
                    device,
                    profile_capture=(bs in (1, 32, 96)),
                )
                assert res1["memset_nodes"] == 0, (tag, bs, res1)
                assert res1["memcpy_nodes"] == 0, (tag, bs, res1)

                if bs in (1, 32, 96):

                    def run_4():
                        with torch.inference_mode():
                            y = x
                            for layer in layers:
                                y = layer(y, input_ids)
                            return y

                    res4 = _audit_distributed(
                        rank,
                        f"moe_{tag}_4layer_bs{bs}",
                        run_4,
                        device,
                        profile_capture=(bs == 32),
                    )
                    assert res4["memset_nodes"] == 0, (tag, bs, res4)
                    assert res4["memcpy_nodes"] == 0, (tag, bs, res4)
            del layers
            torch.cuda.empty_cache()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class MegaMoeGraphAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise unittest.SkipTest("2 GPUs required")
        cap = torch.cuda.get_device_capability()
        if cap not in ((10, 0), (10, 3)):
            raise unittest.SkipTest(f"Requires sm_100a/sm_103a, got {cap}")

    def test_audit_mega_moe_and_se_graphs(self) -> None:
        with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as tmpdir:
            rendezvous = os.path.join(tmpdir, "audit_rendezvous")
            mp.spawn(
                _audit_worker,
                args=(2, rendezvous),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
