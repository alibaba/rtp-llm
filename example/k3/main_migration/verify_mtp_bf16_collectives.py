"""Check Native MTP's BF16 projection workspaces on every TP4/TP8 rank.

This is a correctness test with synthetic tensors, not a checkpoint test or a
performance measurement. AG is checked exactly, and Push RS uses the BF16 cast
of the FP32 sum as its reference. Eager and Graph must return identical values.
"""

import argparse
import json
import logging
from math import gcd
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, tp, port, output):
    logging.basicConfig(level=logging.INFO)
    torch.cuda.set_device(rank)
    from rtp_llm.models_py.distributed.collective_torch import (
        Group,
        _get_group,
        destroy_distributed_environment,
        init_distributed_environment,
    )
    from rtp_llm.models_py.modules.kimi_k3.mtp_collectives import (
        KimiK3MtpBf16Collectives,
    )
    from rtp_llm.ops import NcclCommConfig, ParallelismConfig

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
    ops = KimiK3MtpBf16Collectives(device, max_tokens=256, hidden_size=7168)
    assert (
        ops.gather is not None and ops.scatter is not None
    ), "Must exercise custom BF16 paths"
    torch.manual_seed(61009 + rank)
    rows = []
    for batch in (1, 31, 32, 63, 64):
        for q in (1, 4):
            alignment = tp // gcd(tp, q)
            physical_batch = (batch + alignment - 1) // alignment * alignment
            physical_tokens = physical_batch * q
            local_tokens = physical_tokens // tp
            local = torch.randn(
                (local_tokens, 7168), device=device, dtype=torch.bfloat16
            )
            global_rows = (
                torch.arange(local_tokens, device=device) + group.rank() * local_tokens
            )
            local[global_rows >= batch * q] = 0
            partial = torch.randn(
                (physical_tokens, 7168), device=device, dtype=torch.bfloat16
            )
            partial[batch * q :] = 0

            def reference():
                expected_ag = torch.empty(
                    (physical_tokens, 7168), device=device, dtype=torch.bfloat16
                )
                dist.all_gather_into_tensor(expected_ag, local, group=group)
                all_partial = torch.empty(
                    (tp * physical_tokens, 7168), device=device, dtype=torch.bfloat16
                )
                dist.all_gather_into_tensor(all_partial, partial, group=group)
                expected_rs = (
                    all_partial.reshape(tp, physical_tokens, 7168).float().sum(0)
                )
                expected_rs = expected_rs[
                    group.rank() * local_tokens : (group.rank() + 1) * local_tokens
                ].to(torch.bfloat16)
                return expected_ag, expected_rs

            expected_ag, expected_rs = reference()
            for _ in range(10):
                ops.all_gather(local)
                ops.reduce_scatter(partial)
            torch.cuda.synchronize()
            eager_ag = ops.all_gather(local).clone()
            eager_rs = ops.reduce_scatter(partial).clone()
            torch.testing.assert_close(eager_ag, expected_ag, rtol=0, atol=0)
            torch.testing.assert_close(eager_rs, expected_rs, rtol=0, atol=0)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_ag = ops.all_gather(local).clone()
                graph_rs = ops.reduce_scatter(partial).clone()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_ag, eager_ag, rtol=0, atol=0)
            torch.testing.assert_close(graph_rs, eager_rs, rtol=0, atol=0)
            # Change real values in-place after capture. Replay must consume the
            # new inputs, including the reserved dummy-token zeros.
            local.mul_(-0.5)
            partial.mul_(0.5)
            expected_ag, expected_rs = reference()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_ag, expected_ag, rtol=0, atol=0)
            torch.testing.assert_close(graph_rs, expected_rs, rtol=0, atol=0)
            rows.append(
                dict(
                    batch=batch,
                    physical_batch=physical_batch,
                    q=q,
                    actual_backend="bf16_custom_ag_push_rs",
                    all_gather_exact=True,
                    reduce_scatter_fp32_reference_exact=True,
                    graph_eager_exact=True,
                    replay_refreshed_input_exact=True,
                    dummy_tokens_zero=True,
                )
            )
            del graph
            dist.barrier()
    Path(output + f".rank{rank}.json").write_text(
        json.dumps(dict(rank=rank, tp=tp, cases=rows), indent=2) + "\n"
    )
    dist.barrier()
    destroy_distributed_environment()


def main():
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
        performance=False,
        tp=args.tp,
        dp=8 // args.tp,
        case_count=sum(len(r["cases"]) for r in ranks),
        ranks=ranks,
        boundary="Synthetic BF16 AG/RS and Graph/eager/input-refresh correctness; no model checkpoint outputs or performance acceptance",
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(dict(passed=True, cases=report["case_count"], tp=args.tp)),
        flush=True,
    )


if __name__ == "__main__":
    main()
