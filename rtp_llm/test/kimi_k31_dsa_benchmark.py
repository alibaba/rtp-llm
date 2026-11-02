"""Measure the shared Indexer/SFA pipeline with real K3.1 dimensions.

This is an operator workload, including PageRR cache writes and communication.
It excludes projection GEMMs, model layers, MoE and full MTP cycles. Serving
measurements and 93-layer timelines are separate acceptance evidence.
"""

import argparse
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.modules.base.cuda.bf16_indexer_op import Bf16DraftIndexerOp
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_workspace import (
    DsaWorkspaceOwner,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_page_rr_params import (
    DsaPageRRParams,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_page_rr_impl import (
    PageRRSparseMlaOp,
)
from rtp_llm.ops import NcclCommConfig, ParallelismConfig, compute_ops


def workload(parallel, phase, *, component="pipeline", is_graph=False):
    tp, page, kernel, history = parallel.tp_size, 1024, 128, 65536
    prefill = phase == "prefill"
    batch, width = (1, history) if prefill else (32, 4)
    rows = batch * width
    extent = history if prefill else history + width
    columns = math.ceil(extent / (tp * page)) * page // kernel
    table = torch.arange(1, batch * columns + 1, dtype=torch.int32, device="cuda").view(
        batch, columns
    )
    lengths = torch.full((batch,), width, dtype=torch.int32)
    prefixes = torch.full((batch,), 0 if prefill else history, dtype=torch.int32)
    inputs = SimpleNamespace(
        is_prefill=prefill,
        is_target_verify=not prefill,
        is_mtp_draft_update=False,
        is_cuda_graph=False,
        input_lengths=lengths,
        prefix_lengths=prefixes,
        input_lengths_device=lengths.cuda(),
        prefix_lengths_device=prefixes.cuda(),
        sequence_lengths_plus_1_device=torch.full(
            (batch,), extent, dtype=torch.int32, device="cuda"
        ),
        cu_seqlens_device=torch.arange(
            0, rows + 1, width, dtype=torch.int32, device="cuda"
        ),
        padding_offset=torch.zeros(rows, dtype=torch.int32, device="cuda"),
        logical_token_count=rows,
        physical_token_count=rows,
        kv_cache_kernel_block_id_device=table,
    )
    owner = DsaWorkspaceOwner(tp_size=tp)
    torch.manual_seed(3101)
    owner.value_weights[0] = (
        torch.randn(96, 512, 128, device="cuda") * 0.02
    ).bfloat16()
    params = DsaPageRRParams(
        SimpleNamespace(tokens_per_block=page, kernel_tokens_per_block=kernel),
        parallel,
        workspace_owner=owner,
        is_cuda_graph=is_graph,
    )
    params.prepare(inputs)
    params.prepare_execution("indexer")
    params.prepare_execution("sfa")
    op = Bf16DraftIndexerOp(
        SimpleNamespace(
            indexer_use_fp8=False, index_rope_dim=0, index_head_dim=128, index_topk=2048
        )
    )
    torch.manual_seed(3101)
    query = (torch.randn(rows, 32, 128, device="cuda") * 0.1).bfloat16()
    keys = (torch.randn(rows, 128, device="cuda") * 0.1).bfloat16()
    gates = torch.rand(rows, 32, 1, device="cuda") * 0.01
    latent = (torch.randn(rows, 576, device="cuda") * 0.1).bfloat16()
    sparse_query = (torch.randn(rows, 96 // tp, 576, device="cuda") * 0.1).bfloat16()
    # Each owner stores its own history. Query/weights are identical on all
    # owners, so Decode scores still include all 32 Indexer heads.
    torch.manual_seed(3201 + parallel.tp_rank)
    index_data = (
        torch.randn(batch * columns + 1, kernel, 128, device="cuda") * 0.1
    ).bfloat16()
    index_cache = SimpleNamespace(kv_cache_base=index_data.view(torch.uint8))
    sfa_cache = (
        torch.randn(batch * columns + 1, kernel, 576, device="cuda") * 0.1
    ).bfloat16()
    sparse = PageRRSparseMlaOp(
        96 // tp,
        512,
        64,
        128,
        kernel,
        1.0,
        2048,
        parallelism_config=parallel,
        workspace_owner=owner,
    )
    sparse.plan(params, table, inputs)

    def execute():
        op.encode_q_k(query, keys, index_cache, params, inputs, params)
        if component == "indexer":
            selected = op.compute_topk(
                query, gates, index_cache, params, inputs, params
            )
            return selected, selected
        compute_ops.concat_and_cache_mla(
            latent[:, :512],
            latent[:, 512:],
            sfa_cache,
            params.slot_mapping,
            "auto",
            op.scale,
        )
        selected = op.compute_topk(query, gates, index_cache, params, inputs, params)
        output = sparse.forward(sparse_query, sfa_cache.view(-1, 1, 576), selected)
        return selected, output

    # Keep constant tensors alive for replay; only execution results may be
    # released between samples. No allocator flush or process restart.
    return execute


def synchronize():
    dist.barrier()
    torch.cuda.synchronize()


def measure(execute):
    synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start_time_ns = time.time_ns()
    begin.record()
    result = execute()
    end.record()
    end.synchronize()
    sample = dict(
        cuda_pipeline_ms=begin.elapsed_time(end),
        start_time_ns=start_time_ns,
        end_time_ns=time.time_ns(),
        allocated_baseline_bytes=baseline,
        allocated_peak_bytes=torch.cuda.max_memory_allocated(),
        temporary_peak_bytes=torch.cuda.max_memory_allocated() - baseline,
        reserved_peak_bytes=torch.cuda.max_memory_reserved(),
    )
    del result
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument(
        "--component", choices=("pipeline", "indexer"), default="pipeline"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.graph and args.phase != "decode":
        parser.error("The accepted Prefill path is eager")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    parallel = ParallelismConfig()
    parallel.tp_size, parallel.dp_size, parallel.ep_size = 8, 1, 8
    parallel.world_size = parallel.local_world_size = 8
    parallel.world_rank = parallel.local_rank = rank
    parallel.ffn_sp_size = 8
    parallel.prefill_cp_config.kv_cache_sharded = True
    nccl = NcclCommConfig()
    nccl.nccl_ip = os.environ["MASTER_ADDR"]
    collective.init_distributed_environment(
        parallel, nccl, int(os.environ["MASTER_PORT"]), disable_custom_all_reduce=True
    )
    execute = workload(
        parallel, args.phase, component=args.component, is_graph=args.graph
    )
    # Materialize JIT kernels, collective plans and allocation before the one
    # representative warmup group requested by the user.
    result = execute()
    synchronize()
    assert torch.isfinite(result[1]).all().item()
    del result
    initialized_allocated = torch.cuda.memory_allocated()
    runner = execute
    graph_pool = None
    capture_peak = None
    if args.graph:
        before_capture = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        graph = torch.cuda.CUDAGraph()
        synchronize()
        with torch.cuda.graph(graph):
            captured = execute()
        synchronize()
        graph_pool = torch.cuda.memory_allocated() - before_capture
        capture_peak = torch.cuda.max_memory_allocated() - before_capture

        def replay():
            graph.replay()
            return captured

        runner = replay

    # The original closure owns every static input/cache. Retain it even when
    # timings invoke only graph.replay; a graph does not own its input tensors.
    gc.collect()
    assert torch.cuda.memory_allocated() >= initialized_allocated

    result = runner()  # exactly one representative warmup group
    synchronize()
    assert torch.isfinite(result[1]).all().item()
    del result
    samples = [measure(runner) for _ in range(5)]
    row = dict(
        rank=rank,
        samples=samples,
        graph_retained_allocated_bytes=graph_pool,
        graph_capture_temporary_peak_bytes=capture_peak,
    )
    ranks = [None] * 8
    dist.all_gather_object(ranks, row)
    if rank == 0:
        if args.output.exists():
            raise RuntimeError("Refusing to replace benchmark evidence")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = dict(
            phase=args.phase,
            graph=args.graph,
            history_tokens=65536,
            component=args.component,
            batch_size=1 if args.phase == "prefill" else 32,
            query_tokens=65536 if args.phase == "prefill" else 4,
            precision="BF16 Indexer score/cache and absorbed SFA",
            complete_indexer_heads=32,
            topk=2048,
            page_tokens=1024,
            kernel_page_tokens=128,
            effective_tp_cp_dcp=8,
            initialization_groups=1,
            warmup_groups=1,
            independent_samples=5,
            includes=(
                "Indexer cache writes, reads, communication, score and TopK"
                if args.component == "indexer"
                else "cache writes, Indexer, communication, sparse SFA"
            ),
            excludes="projection GEMMs, decoder, MoE, full MTP cycle",
            ranks=ranks,
            allocation_accounting="PyTorch allocator; vendor/NCCL allocations excluded",
            runtime_shutdown_completed=False,
            max_rank_sample_ms=[
                max(r["samples"][i]["cuda_pipeline_ms"] for r in ranks)
                for i in range(5)
            ],
            maximum_temporary_peak_bytes=max(
                s["temporary_peak_bytes"] for r in ranks for s in r["samples"]
            ),
        )
        report["median_max_rank_cuda_pipeline_ms"] = statistics.median(
            report["max_rank_sample_ms"]
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "phase",
                        "graph",
                        "median_max_rank_cuda_pipeline_ms",
                        "maximum_temporary_peak_bytes",
                    )
                }
            )
        )
    dist.barrier()
    # NCCL graph registrations must be released before destroying the process
    # group. Captured outputs and the replay closure keep the graph pool alive.
    del execute
    del runner
    if args.graph:
        del graph, captured, replay
    gc.collect()
    dist.destroy_process_group()
    if rank == 0:
        report["runtime_shutdown_completed"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
