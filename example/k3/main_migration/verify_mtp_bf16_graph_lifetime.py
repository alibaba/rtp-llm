"""Check MTP workspace ownership with real BF16 collective/GEMM graphs.

The inherited cache initializer is mocked to isolate workspace ownership.
Retained Q1/Q4 graphs must survive repeated MTP initialization and replay
across different batch sizes. Full-model smoke remains a separate gate.
"""

import argparse
import json
from math import gcd
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, tp, port, output):
    torch.cuda.set_device(rank)
    from rtp_llm.models_py.distributed.collective_torch import (
        Group,
        _get_group,
        destroy_distributed_environment,
        init_distributed_environment,
    )
    from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
    from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpModel
    from rtp_llm.ops import NcclCommConfig, ParallelismConfig
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    parallel = ParallelismConfig()
    for key, value in dict(
        world_size=8,
        world_rank=rank,
        local_rank=rank,
        local_world_size=8,
        tp_size=tp,
        dp_size=8 // tp,
        ep_size=8,
    ).items():
        setattr(parallel, key, value)
    nccl = NcclCommConfig()
    nccl.nccl_ip = "127.0.0.1"
    init_distributed_environment(parallel, nccl, port, disable_custom_all_reduce=True)
    group = _get_group(Group.TP)
    device = torch.device("cuda", rank)
    model = KimiK3MtpModel.__new__(KimiK3MtpModel)
    torch.nn.Module.__init__(model)
    model.tp_size = tp
    model._max_generate_batch_size = 64
    model.config = SimpleNamespace(gen_num_per_cycle=3, hidden_size=7168)
    attention = SimpleNamespace(
        input=SimpleNamespace(weight=torch.empty((1,), device=device)),
        _mtp_bf16_collectives=None,
    )
    model.layers = [SimpleNamespace(attention=attention)]
    resource = SimpleNamespace(is_decode_role=True, max_decode_graph_batch_size=64)
    with patch.object(KimiK3Model, "initialize", return_value=True):
        assert model.initialize(resource)
    ops = attention._mtp_bf16_collectives
    assert ops.gather is not None and ops.scatter is not None
    # Replicated synthetic weights make each rank's partial identical, so
    # summation across TP4/TP8 is an exact power-of-two scale. This avoids
    # making FP32 summation association part of the lifetime assertion.
    torch.manual_seed(61009)
    w1 = (torch.randn((1024, 7168), device=device) * 0.01).to(torch.bfloat16)
    w2 = (torch.randn((7168, 1024), device=device) * 0.01).to(torch.bfloat16)
    head = (torch.randn((1024, 7168), device=device) * 0.01).to(torch.bfloat16)
    torch.manual_seed(61009 + rank)
    retained = []

    def compute(x):
        # Consume the collective's output directly, without a test-side clone.
        gathered = ops.all_gather(x)
        projected = rtp_llm_ops.cublas_gemm_bf16_fp32_accum(gathered, w1)
        partial = rtp_llm_ops.cublas_gemm_bf16_fp32_accum(projected, w2)
        return ops.reduce_scatter(partial).clone()

    def reference(x):
        full = torch.empty((x.shape[0] * tp, 7168), device=device, dtype=torch.bfloat16)
        dist.all_gather_into_tensor(full, x, group=group)
        projected = rtp_llm_ops.cublas_gemm_bf16_fp32_accum(full, w1)
        partial = rtp_llm_ops.cublas_gemm_bf16_fp32_accum(projected, w2)
        parts = torch.empty(
            (tp * partial.shape[0], 7168), device=device, dtype=torch.bfloat16
        )
        dist.all_gather_into_tensor(parts, partial, group=group)
        reduced = parts.reshape(tp, partial.shape[0], 7168).float().sum(0)
        begin = group.rank() * x.shape[0]
        return reduced[begin : begin + x.shape[0]].to(torch.bfloat16)

    for batch in (1, 31, 32, 63, 64):
        for q in (1, 4):
            alignment = tp // gcd(tp, q)
            physical_batch = (batch + alignment - 1) // alignment * alignment
            count = physical_batch * q // tp
            x = torch.randn((count, 7168), device=device, dtype=torch.bfloat16)
            rows = torch.arange(count, device=device) + group.rank() * count
            x[rows >= batch * q] = 0
            expected = reference(x)
            for _ in range(10):
                compute(x)
            torch.cuda.synchronize()
            torch.testing.assert_close(compute(x), expected, rtol=0, atol=0)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = compute(x)
            retained.append(dict(batch=batch, q=q, x=x, result=result, graph=graph))
    records = []
    pointers = {name: tensor.data_ptr() for name, tensor in ops.gather.storage.items()}
    with patch.object(KimiK3Model, "initialize", return_value=True):
        assert model.initialize(resource)
        assert attention._mtp_bf16_collectives is ops
        assert {
            name: tensor.data_ptr() for name, tensor in ops.gather.storage.items()
        } == pointers
        try:
            model.initialize(
                SimpleNamespace(is_decode_role=True, max_decode_graph_batch_size=128)
            )
        except ValueError as error:
            assert "Cannot replace MTP workspaces" in str(error)
        else:
            raise AssertionError("Unsafe workspace growth was allowed after capture")
    # All captures remain alive. Exercise descending then irregular buckets.
    for order in (list(reversed(range(10))), [0, 9, 1, 8, 2, 7, 3, 6, 4, 5]):
        for index in order:
            case = retained[index]
            case["x"].mul_(-0.5)
            expected = reference(case["x"])
            expected_head = rtp_llm_ops.cublas_gemm_bf16_bf16_fp32(expected, head)
            dist.barrier()
            case["graph"].replay()
            actual_head = rtp_llm_ops.cublas_gemm_bf16_bf16_fp32(case["result"], head)
            torch.cuda.synchronize()
            torch.testing.assert_close(case["result"], expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_head, expected_head, rtol=0, atol=0)
            records.append(
                dict(
                    batch=case["batch"],
                    q=case["q"],
                    graph_gemm_exact=True,
                    repeated_initialization_workspace_identity=True,
                    captured_workspace_growth_rejected=True,
                    post_graph_fp32_head_exact=True,
                )
            )
    Path(output + f".rank{rank}.json").write_text(
        json.dumps(dict(rank=rank, cases=records))
    )
    dist.barrier()
    del retained
    destroy_distributed_environment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=(4, 8), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mp.spawn(worker, args=(args.tp, args.port, args.output), nprocs=8, join=True)
    ranks = [
        json.loads(Path(args.output + f".rank{r}.json").read_text()) for r in range(8)
    ]
    report = dict(
        passed=True,
        diagnostic=True,
        performance=False,
        tp=args.tp,
        cases=sum(len(r["cases"]) for r in ranks),
        ranks=ranks,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "tp", "cases")}), flush=True)
