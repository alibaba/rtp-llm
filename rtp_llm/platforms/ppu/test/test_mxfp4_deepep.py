"""Real PPU DeepEP dispatch/MLP/combine with an analytic MXFP4 oracle.

Run with four or eight exclusively leased visible GPUs. Four-rank diagnostics
do not qualify the eight-rank model or its SGLang throughput target.
"""

import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@torch.inference_mode()
def run_rank(rank, world, rendezvous):
    from rtp_llm.models_py.distributed.deepep_wrapper import (
        DeepEPWrapper,
        DeepepWrapperConfig,
    )
    from rtp_llm.platforms.ppu.models.dsv4.ppu_moe_config import PpuMoeConfig as MoeCfg
    from rtp_llm.platforms.ppu.models.dsv4.ppu_deepep_fp4 import PpuDeepEPFP4Strategy
    from rtp_llm.utils.model_weight import W

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=90),
    )
    batch, capacity, dim, inter, experts, topk = 80, 128, 4096, 2048, 256, 6
    local = experts // world
    device = torch.device("cuda", rank)
    config = DeepepWrapperConfig(
        ep_rank=rank,
        ep_size=world,
        tp_size=1,
        local_rank=rank,
        world_size=world,
        hidden_size=dim,
        expert_num=experts,
        moe_k=topk,
        deep_ep_num_sm=24,
        use_deepep_low_latency=True,
        use_deepep_internode=False,
        ll_num_max_token=capacity,
        ll_num_max_token_per_rank=capacity,
    )
    DeepEPWrapper.create(config)
    strategy = PpuDeepEPFP4Strategy(
        MoeCfg(
            layer_id=0,
            dim=dim,
            moe_inter_dim=inter,
            n_routed_experts=experts,
            n_activated_experts=topk,
            swiglu_limit=0.0,
            ep_size=world,
            ep_rank=rank,
            n_local_experts=local,
            local_expert_start=rank * local,
            local_expert_end=(rank + 1) * local,
            max_tokens_per_rank=capacity,
            tp_size=1,
        )
    )
    # Each nibble denotes exactly 1. W1/W3 use 2**-8; W2 uses an
    # expert-specific power of two, making route/EP-shard mistakes observable.
    w1 = torch.full((local, inter, dim // 2), 0x22, dtype=torch.uint8, device=device)
    s1 = torch.full((local, inter, dim // 32), 119, dtype=torch.uint8, device=device)
    w2 = torch.full((local, dim, inter // 2), 0x22, dtype=torch.uint8, device=device)
    s2 = torch.empty((local, dim, inter // 32), dtype=torch.uint8, device=device)
    for expert in range(local):
        s2[expert].fill_(121 + (rank * local + expert) % 4)
    strategy.setup_weights(
        {
            W.v4_routed_w1_w: w1,
            W.v4_routed_w3_w: w1,
            W.v4_routed_w2_w: w2,
            W.v4_routed_w1_s: s1.view(torch.float8_e8m0fnu),
            W.v4_routed_w3_s: s1.view(torch.float8_e8m0fnu),
            W.v4_routed_w2_s: s2.view(torch.float8_e8m0fnu),
        }
    )
    del w1, w2, s1, s2
    x = torch.empty((batch, dim), dtype=torch.bfloat16, device=device)
    indices = torch.empty((batch, topk), dtype=torch.int64, device=device)
    weights = torch.empty((batch, topk), dtype=torch.float32, device=device)

    def inputs(step, hot, active):
        # W13 outputs 0.25, 0.75, 1 or 3. Masked SwiGLU/MXFP4 outputs
        # 1/32, 3/8, 3/4 or 8; unscaled W2 sums give 1, 12, 24 or 256.
        case = (rank + step) % 4
        x.fill_((1, 3, 4, 12)[case] / 64)
        ids = torch.arange(topk).expand(batch, -1).clone()
        ids += step * 64
        if not hot:
            ids += (torch.arange(batch) + rank * batch)[:, None] * topk
        ids %= experts
        ids[active:] = -1
        route_weights = torch.full((batch, topk), 0.125)
        route_weights[active:] = 0
        indices.copy_(ids)
        weights.copy_(route_weights)
        multipliers = (2.0 ** (ids.clamp_min(0) % 4)) * route_weights
        reference = (1, 12, 24, 256)[case] * multipliers.sum(-1)
        return reference.to(torch.bfloat16).float().to(device)

    def check(output, expected):
        assert output.shape == (batch, dim)
        assert output.dtype == torch.float32
        torch.testing.assert_close(
            output, expected[:, None].expand_as(output), rtol=0, atol=0
        )

    expected = inputs(0, True, batch)
    check(strategy(x, weights, indices), expected)
    # All four/eight ranks route their 80 tokens to the same six experts:
    # 320/640 valid rows per expert must survive dispatch and combine.
    assert batch * world > 128
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            strategy(x, weights, indices)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = strategy(x, weights, indices)
    torch.cuda.current_stream().wait_stream(stream)
    for step, hot, active in (
        (0, True, batch),
        (1, False, max(0, batch - 27 * rank)),
        (2, True, batch),
        (3, True, 0),
        (4, False, batch),
    ):
        expected = inputs(step, hot, active)
        graph.replay()
        check(output, expected)
        dist.barrier()
    torch.cuda.synchronize()
    print(
        f"PASS rank={rank} EP{world}: eager + 5 Graph replays, hot rows={batch*world}",
        flush=True,
    )
    del graph, output, strategy
    DeepEPWrapper.reset()
    dist.destroy_process_group()


class Mxfp4DeepEPTest(unittest.TestCase):
    def test_actual_dispatch_mlp_combine_and_graph(self):
        world = int(os.environ.get("RTP_PPU_EP_TEST_WORLD_SIZE", "4"))
        if world not in (4, 8):
            raise ValueError("The model-shape diagnostic requires EP4 or EP8")
        if not torch.cuda.is_available() or torch.cuda.device_count() < world:
            self.skipTest(f"requires {world} exclusively leased PPU GPUs")
        if any(torch.cuda.get_device_name(i) != "ZW-M890P" for i in range(world)):
            self.skipTest("requires PPU M890P")
        with tempfile.TemporaryDirectory(prefix="rtp-ppu-deepep-") as directory:
            mp.spawn(
                run_rank,
                args=(world, directory + "/rendezvous"),
                nprocs=world,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
