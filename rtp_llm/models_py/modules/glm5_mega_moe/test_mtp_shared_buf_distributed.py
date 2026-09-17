"""Real shared-buffer EP replay regression (SM100+, synthetic weights).

Run with a timeout to catch collective hangs:
  timeout 600s torchrun --standalone --nproc_per_node=4 -m \\
    rtp_llm.models_py.modules.glm5_mega_moe.test_mtp_shared_buf_distributed

Tests MegaMoE communication/graph ownership, not MTP indexer acceptance accuracy.
"""

import argparse
import json
import os
import statistics
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--equal-batches", action="store_true")
    parser.add_argument("--rounds", type=int, default=64)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--variant", choices=("fp4", "fp8", "fp4_se", "fp8_se", "fused"), default="fp4"
    )
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError(
            "This regression requires SM100+; do not count it as passed on older GPUs"
        )
    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=120),
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2 or 256 % world:
        raise RuntimeError("Requires at least two ranks and world_size dividing 256")
    from .mega_moe import GLM5MegaMoE
    from .mega_moe_fp8 import GLM5MegaMoEFP8
    from .mega_moe_fp8_se import GLM5MegaMoEFP8SE
    from .mega_moe_fused import GLM5MegaMoEFused
    from .mega_moe_se import GLM5MegaMoESE

    model_cls = {
        "fp4": GLM5MegaMoE,
        "fp8": GLM5MegaMoEFP8,
        "fp4_se": GLM5MegaMoESE,
        "fp8_se": GLM5MegaMoEFP8SE,
        "fused": GLM5MegaMoEFused,
    }[args.variant]

    torch.manual_seed(42 + rank)
    dim, inter, experts, topk = 6144, 2048, 256, 8
    source = model_cls.from_params(
        layer_id=0,
        dim=dim,
        moe_inter_dim=inter,
        n_routed_experts=experts,
        n_activated_experts=topk,
        ep_size=world,
        ep_rank=rank,
        max_tokens_per_rank=128,
    )

    def weight(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device="cuda") * 0.01

    def fp8_weight(*shape):
        data = torch.randn(*shape, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.full(
            (*shape[:-2], shape[-2] // 128, shape[-1] // 128),
            2**-7,
            dtype=torch.float8_e8m0fnu,
            device="cuda",
        )
        return data, scales

    if args.variant.startswith("fp8"):
        w1, s1 = fp8_weight(experts // world, inter, dim)
        w2, s2 = fp8_weight(experts // world, dim, inter)
        w3, s3 = fp8_weight(experts // world, inter, dim)
        source.setup_weights_from_fp8(w1, s1, w2, s2, w3, s3)
        del w1, s1, w2, s2, w3, s3
    else:
        w1 = weight(experts // world, inter, dim)
        w2 = weight(experts // world, dim, inter)
        w3 = weight(experts // world, inter, dim)
        source.setup_weights_from_bf16(w1, w2, w3)
        del w1, w2, w3
    if args.variant in ("fp4_se", "fp8_se", "fused"):
        sw1, ss1 = fp8_weight(2 * inter, dim)
        sw2, ss2 = fp8_weight(dim, inter)
        source.setup_shared_expert_from_fp8(sw1, ss1, sw2, ss2)
        del sw1, ss1, sw2, ss2
    clone = source.clone_for_cuda_graph(share_mega_buf=True)
    prefill = source.clone_for_cuda_graph()
    assert prefill._mega_buf is not source._mega_buf
    assert clone._mega_buf is source._mega_buf
    assert clone._mega_y.data_ptr() != source._mega_y.data_ptr()

    cases = []
    for size in (1, 8, 32):
        # Different local batch sizes, but exactly one EP call per graph.
        tokens = size if args.equal_batches or rank % 2 else max(1, size // 2)
        x = weight(tokens, dim)
        weights = torch.full((tokens, topk), 1 / topk, device="cuda")
        ids = torch.stack(
            [torch.randperm(experts, device="cuda")[:topk] for _ in range(tokens)]
        )
        graphs = []
        for model in (source, clone, prefill):
            for _ in range(3):
                model(x, weights, ids)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                if model is clone:
                    # Model local cold-indexer delay before entering EP; peers
                    # choosing reuse must tolerate different arrival times.
                    torch.cuda._sleep(100000)
                output = model(x, weights, ids)
            torch.cuda.synchronize()
            graphs.append((graph, output))
        # CUDA Graph retains pointer arguments, not Python ownership of external
        # routing tensors. Keep every captured input alive across size switches.
        cases.append((x, weights, ids, graphs))

    for step in range(args.rounds):
        x, weights, ids, graphs = cases[step % len(cases)]
        x.normal_(std=0.01)
        # All ranks run the source graph to form the reference for these inputs.
        if args.verbose:
            print(
                f"rank={rank} step={step} source replay tokens={x.size(0)}", flush=True
            )
        graphs[0][0].replay()
        torch.cuda.synchronize()
        expected = graphs[0][1].clone()
        # No host collective decides which clone to replay. Exercise all-source,
        # all-clone and opposite mixed choices repeatedly.
        choice = (step // 4) % 2 if step % 4 < 2 else (rank + step) % 2
        if args.source_only:
            choice = 0
        if args.verbose:
            print(f"rank={rank} step={step} selected replay clone={choice}", flush=True)
        graphs[choice][0].replay()
        torch.cuda.synchronize()
        actual = graphs[choice][1].clone()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # All peers enter the separate draft-prefill arena, then return to the
        # shared decode arena in the next round. Neither arena may lose phase.
        graphs[2][0].replay()
        torch.testing.assert_close(graphs[2][1], expected, rtol=0, atol=0)
    # Also queue mixed graphs without per-step host waits. Retain snapshots for
    # checking every replay after completion, including separate prefill output.
    references = []
    for _, _, _, graphs in cases:
        graphs[0][0].replay()
        references.append(graphs[0][1].clone())
    snapshots = []
    for step in range(args.rounds):
        case_id = step % len(cases)
        graphs = cases[case_id][-1]
        choice = 0 if args.source_only else (rank + step) % 2
        graphs[choice][0].replay()
        snapshots.append((graphs[choice][1].clone(), references[case_id]))
        graphs[2][0].replay()
        snapshots.append((graphs[2][1].clone(), references[case_id]))
    torch.cuda.synchronize()
    for actual, expected in snapshots:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    dist.barrier()
    if rank == 0:
        print(
            f"PASS: {args.variant} {args.rounds} checked and {args.rounds} queued mixed EP graph replays with separate prefill arena switches",
            flush=True,
        )
    if args.benchmark:
        benchmark_control_overhead(cases[0][-1][0][0])
    dist.destroy_process_group()


def benchmark_control_overhead(graph):
    """Isolate control overhead with identical warm-path MegaMoE GPU work.

    This is not a whole-model throughput or cold-indexer benchmark.
    """
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    measurements = {"gloo": [], "local": []}
    iterations = 100
    for mode in ("gloo", "local", "local", "gloo") * 2:
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        for _ in range(iterations):
            if mode == "gloo":
                flag = torch.tensor([0], dtype=torch.int32)
                dist.all_reduce(flag, group=control)
                assert flag.item() == 0
            graph.replay()
        torch.cuda.synchronize()
        measurements[mode].append((time.perf_counter() - start) * 1e6 / iterations)
    result = {
        mode + "_us_per_step": statistics.median(samples)
        for mode, samples in measurements.items()
    }
    results = [None] * dist.get_world_size()
    dist.all_gather_object(results, result, group=control)
    if dist.get_rank() == 0:
        print("CONTROL_BENCHMARK " + json.dumps(results), flush=True)
    dist.destroy_process_group(control)


if __name__ == "__main__":
    main()
