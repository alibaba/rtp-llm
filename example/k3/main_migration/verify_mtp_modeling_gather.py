"""Exact synthetic MTP embedding/recurrent gather checks on eight ranks.

This checks native embedding, TP ordering, DP owner separation, shared Graph
workspace lifetime, input refresh and fallback. It is not a model accuracy or
performance result.
"""

import argparse
import json
from math import gcd
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F


def worker(rank, tp, port, output):
    torch.cuda.set_device(rank)
    from rtp_llm.models_py.distributed.collective_torch import (
        Group,
        _get_group,
        destroy_distributed_environment,
        init_distributed_environment,
    )
    from rtp_llm.models_py.distributed.symm_mem import get_symm_mem_communicator
    from rtp_llm.models_py.modules.base.common.embedding import Embedding
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
    assert get_symm_mem_communicator() is None
    group = _get_group(Group.TP)
    device = torch.device("cuda", rank)
    hidden, vocabulary = 7168, 97
    ops = KimiK3MtpBf16Collectives(device, max_tokens=256, hidden_size=hidden)
    assert not ops.modeling_gather.disabled, "Must exercise actual BF16 multimem"
    assert ops.gather is not None and ops.scatter is not None
    workspace_pointer = ops.modeling_gather.buffer.data_ptr()
    torch.manual_seed(71009 + rank)
    weight = torch.randn(
        (vocabulary, hidden // tp), device=device, dtype=torch.bfloat16
    )
    embedding = Embedding(None, parallel, weight)

    def embedding_reference(ids):
        shards = torch.empty(
            (tp * vocabulary, hidden // tp), device=device, dtype=weight.dtype
        )
        dist.all_gather_into_tensor(shards, weight, group=group)
        full = (
            shards.view(tp, vocabulary, hidden // tp)
            .permute(1, 0, 2)
            .reshape(vocabulary, hidden)
        )
        return F.embedding(ids.long(), full)

    rows = []
    for batch in (1, 31, 32, 63, 64):
        for q in (1, 4):
            alignment = tp // gcd(tp, q)
            physical_batch = (batch + alignment - 1) // alignment * alignment
            tokens = physical_batch * q
            ids = torch.arange(tokens, device=device, dtype=torch.int32) % vocabulary
            ids[batch * q :] = 0
            local = torch.randn(
                (tokens // tp, hidden), device=device, dtype=torch.bfloat16
            )
            local_rows = torch.arange(tokens // tp, device=device) + group.rank() * (
                tokens // tp
            )
            local[local_rows >= batch * q] = 0

            def recurrent_reference():
                result = torch.empty((tokens, hidden), device=device, dtype=local.dtype)
                dist.all_gather_into_tensor(result, local, group=group)
                return result

            expected_embedding = embedding_reference(ids)
            expected_recurrent = recurrent_reference()
            for _ in range(10):
                embedding(ids, tp_gather=ops.all_gather_modeling_boundary)
                ops.all_gather_modeling_boundary(local)
            torch.cuda.synchronize()
            dist.barrier(group=group)
            eager_embedding = embedding(ids, tp_gather=ops.all_gather_modeling_boundary)
            eager_recurrent = ops.all_gather_modeling_boundary(local)
            torch.testing.assert_close(
                eager_embedding, expected_embedding, rtol=0, atol=0
            )
            torch.testing.assert_close(
                eager_recurrent, expected_recurrent, rtol=0, atol=0
            )
            # A boundary output owns its storage while subsequent gathers reuse
            # the symmetric buffer. Test this before capture as well.
            saved = eager_embedding.clone()
            ops.all_gather_modeling_boundary(local)
            torch.testing.assert_close(eager_embedding, saved, rtol=0, atol=0)
            torch.cuda.synchronize()
            dist.barrier(group=group)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_embedding = embedding(
                    ids, tp_gather=ops.all_gather_modeling_boundary
                )
                projection_input = ops.all_gather(local).clone()
                graph_recurrent = ops.all_gather_modeling_boundary(local)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                graph_embedding, expected_embedding, rtol=0, atol=0
            )
            torch.testing.assert_close(
                projection_input, expected_recurrent, rtol=0, atol=0
            )
            torch.testing.assert_close(
                graph_recurrent, expected_recurrent, rtol=0, atol=0
            )
            ids[: batch * q].add_(1).remainder_(vocabulary)
            local.mul_(-0.5)
            weight.mul_(-0.5)
            expected_embedding = embedding_reference(ids)
            expected_recurrent = recurrent_reference()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                graph_embedding, expected_embedding, rtol=0, atol=0
            )
            torch.testing.assert_close(
                projection_input, expected_recurrent, rtol=0, atol=0
            )
            torch.testing.assert_close(
                graph_recurrent, expected_recurrent, rtol=0, atol=0
            )
            assert ops.modeling_gather.buffer.data_ptr() == workspace_pointer
            rows.append(
                dict(
                    batch=batch,
                    physical_batch=physical_batch,
                    q=q,
                    embedding_tp_order_exact=True,
                    recurrent_tp_order_exact=True,
                    output_survives_buffer_reuse_exact=True,
                    graph_eager_exact=True,
                    replay_input_refresh_exact=True,
                    private_workspace_pointer_stable=True,
                )
            )
            del graph
            dist.barrier(group=group)

    # The actual long-prompt eligibility rule must retain the existing gather.
    # A smaller hidden dimension keeps this synthetic test's storage bounded.
    prompt_local = (
        torch.arange(1024 * 32, device=device).reshape(1024, 32).to(torch.bfloat16)
        + rank
    )
    assert not ops.modeling_gather.should_torch_symm_mem_allgather(prompt_local)
    expected_prompt = torch.empty((tp * 1024, 32), device=device, dtype=torch.bfloat16)
    dist.all_gather_into_tensor(expected_prompt, prompt_local, group=group)
    torch.testing.assert_close(
        ops.all_gather_modeling_boundary(prompt_local), expected_prompt, rtol=0, atol=0
    )
    # Simulate unavailable multicast unanimously after the Graph cases; verify
    # the supported fallback's values without claiming unavailable-hardware QA.
    ops.modeling_gather.disabled = True
    fallback = ops.all_gather_modeling_boundary(prompt_local[:4].contiguous())
    expected_fallback = torch.empty((tp * 4, 32), device=device, dtype=torch.bfloat16)
    dist.all_gather_into_tensor(
        expected_fallback, prompt_local[:4].contiguous(), group=group
    )
    torch.testing.assert_close(fallback, expected_fallback, rtol=0, atol=0)
    Path(output + f".rank{rank}.json").write_text(
        json.dumps(
            dict(
                rank=rank,
                tp_rank=group.rank(),
                dp_owner=rank // tp,
                tp=tp,
                cases=rows,
                long_prompt_fallback_exact=True,
                simulated_unavailable_multicast_fallback_exact=True,
                global_custom_allreduce_stayed_disabled=get_symm_mem_communicator()
                is None,
            ),
            indent=2,
        )
        + "\n"
    )
    dist.barrier()
    destroy_distributed_environment()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(4, 8), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mp.spawn(worker, args=(args.tp, args.port, args.output), nprocs=8, join=True)
    ranks = [
        json.loads(Path(args.output + f".rank{rank}.json").read_text())
        for rank in range(8)
    ]
    report = dict(
        passed=True,
        performance=False,
        tp=args.tp,
        dp=8 // args.tp,
        cases=sum(len(rank["cases"]) for rank in ranks),
        ranks=ranks,
        boundary="Synthetic exact BF16 embedding/recurrent layout and Graph workspace tests; checkpoint smoke and modeling performance remain separate gates",
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=True, tp=args.tp, cases=report["cases"])), flush=True)


if __name__ == "__main__":
    main()
