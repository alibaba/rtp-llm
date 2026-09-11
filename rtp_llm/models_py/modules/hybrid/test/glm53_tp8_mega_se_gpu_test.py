"""Real MegaMoE SE TP-config parity; GPU7 EP1 or explicitly enabled EP8."""

import json
import os
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist


def worker(rank, world_size, store):
    from indexer_paged_score_partition_test import load_locked_runtime

    load_locked_runtime()
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.glm53_cmp_full_attention_benchmark import (
        quantize_weight,
    )
    from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_se_wrapper import (
        MegaMoeSEWrapper,
    )
    from rtp_llm.utils.model_weight import W

    os.environ["GLM5_MEGA_MOE_JIT_WARMUP"] = "0"
    os.environ["GLM5_MEGA_MOE_SE_JIT_WARMUP"] = "0"
    torch.cuda.set_device(rank)
    config = SimpleNamespace(
        hidden_size=6144,
        inter_size=2048,
        moe_inter_size=2048,
        expert_num=256,
        moe_k=8,
        max_seq_len=256,
        gen_num_per_cycle=3,
        swiglu_limit=10.0,
    )
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method="file://" + store,
        device_id=torch.device("cuda", rank),
    )
    try:
        with torch.inference_mode():
            modules = []
            for tp in (1, 8):
                torch.manual_seed(20260911 + rank)
                weights = {}
                for wk, sk, n, k in (
                    (W.moe_w1, W.moe_s1, 4096, 6144),
                    (W.moe_w2, W.moe_s2, 6144, 2048),
                ):
                    weights[wk] = torch.randint(
                        -128,
                        128,
                        (256 // world_size, n, k // 2),
                        dtype=torch.int8,
                        device="cuda",
                    )
                    weights[sk] = torch.full(
                        (256 // world_size, n, k // 32),
                        2**-8,
                        dtype=torch.float8_e8m0fnu,
                        device="cuda",
                    )
                torch.manual_seed(
                    20260912
                )  # Full shared weights replicated across EP ranks.
                for wk, sk, n, k in (
                    (W.ffn_w13, W.ffn_s13, 4096, 6144),
                    (W.ffn_w2, W.ffn_s2, 6144, 2048),
                ):
                    source = (torch.randn((n, k), device="cuda") / k**0.5).bfloat16()
                    weights[wk], weights[sk] = quantize_weight(source)
                parallel = SimpleNamespace(
                    tp_size=tp,
                    ep_size=world_size,
                    ep_rank=rank,
                    role_type=None,
                    get_ffn_tp_size=lambda tp=tp: tp,
                )
                modules.append(
                    MegaMoeSEWrapper(
                        config,
                        parallel,
                        weights,
                        layer_idx=917 + tp,
                        max_generate_batch_size=64,
                    )
                )
                print(
                    json.dumps(
                        dict(
                            stage="constructed",
                            logical_tp=tp,
                            physical_ep=world_size,
                            rank=rank,
                        )
                    ),
                    flush=True,
                )
            modules.append(modules[1].clone_for_cuda_graph())
            for rows in (4, 16, 64, 256):
                x = torch.randn((rows, 6144), device="cuda", dtype=torch.bfloat16)
                ids = (
                    torch.arange(rows * 8, device="cuda").reshape(rows, 8) % 256
                ).long()
                gates = torch.full(
                    (rows, 8), 2.5 / 8, device="cuda", dtype=torch.float32
                )

                def run(module):
                    if world_size > 1 and not torch.cuda.is_current_stream_capturing():
                        torch.cuda.synchronize()
                        dist.barrier()
                    return module(x, gates, ids).clone()

                eager = [run(m) for m in modules]
                for out in eager[1:]:
                    torch.testing.assert_close(
                        out.view(torch.uint8),
                        eager[0].view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                graphs, outputs = [], []
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for module in modules:
                        run(module)
                stream.synchronize()
                for module in modules:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = run(module)
                    graphs.append(graph)
                    outputs.append(output)
                for state in range(3):
                    torch.manual_seed(20260913 + state)
                    x.normal_()  # Replicated TP hidden states after attention reduction.
                    ids.add_(8).remainder_(256)
                    for graph in graphs:
                        graph.replay()
                    torch.cuda.synchronize()
                    reference = run(modules[0])
                    for output in outputs:
                        assert torch.isfinite(output).all()
                        torch.testing.assert_close(
                            output.view(torch.uint8),
                            reference.view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                print(
                    json.dumps(
                        dict(
                            rows=rows,
                            changed_states=3,
                            tp1_tp8_clone_byte_equal=True,
                            physical_ep=world_size,
                            rank=rank,
                            distributed_tp8_verified=world_size == 8,
                        )
                    ),
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


def main():
    distributed = "--distributed" in sys.argv
    if distributed:
        if (
            os.environ.get("GLM53_SE_ALLOW_EIGHT_GPUS") != "1"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3,4,5,6,7"
        ):
            raise RuntimeError("EP8 requires explicit eight-GPU authorization/env")
    elif os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise RuntimeError("Default test uses physical GPU7 only")
    with tempfile.TemporaryDirectory(prefix="glm53-se-parity-") as directory:
        store = directory + "/store"
        if distributed:
            torch.multiprocessing.spawn(worker, args=(8, store), nprocs=8, join=True)
        else:
            worker(0, 1, store)


if __name__ == "__main__":
    main()
