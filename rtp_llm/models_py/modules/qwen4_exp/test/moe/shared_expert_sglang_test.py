"""TP1/TP2 shared expert reduction, fused gate/add and CUDA Graph regression."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.distributed as dist
import torch.nn.functional as F

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.modules.qwen4_exp.moe.shared_gemm import (
    prepare_bf16_shared_weights,
)
from rtp_llm.models_py.modules.qwen4_exp.test.moe.shared_expert_overlap_test import (
    make_layer,
)
from rtp_llm.utils.model_weight import W


def main():
    if "LOCAL_RANK" not in os.environ:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                __file__,
            ],
            check=True,
        )
        return
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if rank == 0:
        assert not dist.is_initialized()
        run(rank, device, tp_size=1)
    dist.init_process_group("nccl", device_id=device)
    try:
        run(rank, device, tp_size=2)
    finally:
        dist.destroy_process_group()


def run(rank, device, tp_size):
    torch.manual_seed(216)
    hidden, intermediate = 2560, 640
    full_up = torch.empty(
        hidden, 2 * intermediate, device=device, dtype=torch.bfloat16
    ).normal_(0, 0.01)
    full_down = torch.empty(
        intermediate, hidden, device=device, dtype=torch.bfloat16
    ).normal_(0, 0.01)
    gate = torch.empty(hidden, 1, device=device, dtype=torch.bfloat16).normal_(0, 0.01)
    source = {W.ffn_w13: full_up.clone(), W.ffn_w2: full_down.clone()}
    config = SimpleNamespace(
        _qwen4_dense_storage="source_bf16", hidden_size=hidden, inter_size=intermediate
    )
    parallelism = SimpleNamespace(get_attn_tp_size=lambda: tp_size, tp_rank=rank)
    prepare_bf16_shared_weights([source], config, parallelism)
    pointers = [value.data_ptr() for value in source.values()]
    prepare_bf16_shared_weights([source], config, parallelism)
    assert pointers == [value.data_ptr() for value in source.values()]
    assert source[W.ffn_w13].shape == (hidden, 2 * intermediate // tp_size)
    assert source[W.ffn_w2].shape == (intermediate // tp_size, hidden)
    layer = make_layer()
    layer._shared_stream = None
    layer.fused_moe = SimpleNamespace(
        router=SimpleNamespace(tp_collective_size=tp_size)
    )
    layer._bf16_shared_tp = True
    layer._bf16_shared_stream = torch.cuda.Stream(device=device)
    layer.shared_expert.up_proj = CudaF16Linear(source[W.ffn_w13])
    layer.shared_expert.down_proj = CudaF16Linear(source[W.ffn_w2])
    layer.shared_expert_gate = CudaF16Linear(gate)
    calls = []

    def routed(value, *, skip_tp_allreduce=False):
        assert skip_tp_allreduce
        calls.append("routed_partial")
        return value * ((rank + 1) / 4)

    def reduce(value, *, group):
        assert tp_size == 2 and group == Group.TP
        calls.append("combined_reduce")
        dist.all_reduce(value)
        return value

    layer._forward_routed = routed
    layer.tp_all_reduce = reduce
    layer.shared_expert_gate.forward = Mock(
        side_effect=AssertionError("separate gate GEMV")
    )
    results = []
    out = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    for rows in (1, 2, 4, 8, 16, 32, 33, 64):
        torch.manual_seed(1000 + rows)
        x = torch.empty(rows, hidden, device=device, dtype=torch.bfloat16).normal_(
            0, 0.1
        )
        calls.clear()
        serial = layer(x)
        dual = layer(x, is_cuda_graph=True)
        assert torch.equal(serial, dual), (rank, rows, "eager dual")
        per_forward = ["routed_partial"] + (["combined_reduce"] if tp_size == 2 else [])
        assert calls == per_forward * 2
        graph = torch.cuda.CUDAGraph()
        if tp_size == 2:
            dist.barrier()
        with torch.cuda.graph(graph):
            captured = layer(x, is_cuda_graph=True)
        assert calls == per_forward * 3
        max_error = 0.0
        for _ in range(3):
            x.normal_(0, 0.1)
            up = F.linear(x, full_up.t().contiguous())
            g, v = up.chunk(2, dim=-1)
            shared = F.linear(
                (F.silu(g.float()) * v.float()).to(torch.bfloat16),
                full_down.t().contiguous(),
            )
            gate_scale = torch.sigmoid(x.float().matmul(gate.float()))
            expected = (
                (0.75 if tp_size == 2 else 0.25) * x.float()
                + gate_scale * shared.float()
            ).to(torch.bfloat16)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, expected, rtol=0.02, atol=0.002)
            max_error = max(
                max_error, (captured.float() - expected.float()).abs().max().item()
            )
            if tp_size == 2:
                peer = captured.clone()
                dist.broadcast(peer, 0)
                assert torch.equal(peer, captured)
        if rows == 8 and tp_size == 2:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                graph.replay()
                torch.cuda.synchronize()
            trace = out / f"shared-tp-rank{rank}.json"
            profile.export_chrome_trace(str(trace))
            kernels = [
                event
                for event in json.loads(trace.read_text())["traceEvents"]
                if event.get("cat") == "kernel"
            ]
            assert sum("act_and_mul" in event["name"] for event in kernels) == 1
            assert sum("_tn_align" in event["name"] for event in kernels) == 2
            assert sum("splitKreduce_kernel" in event["name"] for event in kernels) == 1
            assert (
                sum(
                    "_fused_gate_sigmoid_mul_add_kernel" in event["name"]
                    for event in kernels
                )
                == 1
            )
            assert any("ncclDevKernel_AllReduce" in event["name"] for event in kernels)
        results.append({"m": rows, "graph_replays": 3, "max_abs_error": max_error})
        graph.reset()
    receipt = {
        "rank": rank,
        "tp_size": tp_size,
        "status": "PASSED",
        "cases": results,
        "up_shape": list(layer.shared_expert.up_proj.weight.shape),
        "down_shape": list(layer.shared_expert.down_proj.weight.shape),
    }
    (out / (f"rank{rank}.json" if tp_size == 2 else "tp1.json")).write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print("SHARED_TP_SGLANG_REGRESSION_PASS", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
