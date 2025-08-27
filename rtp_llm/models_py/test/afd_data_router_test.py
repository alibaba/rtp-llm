import itertools
import math
import os
from unittest import TestCase, main

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.moe.afd_data_router import (
    AfdDataRouterAttn,
    AfdDataRouterFfn,
)


def _generate_mock_routing_data(
    seed: int,
    num_tokens: int,
    num_experts: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)

    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device="cuda"), dim=1
    )
    gating_logits = torch.randn(num_tokens, num_experts, device="cuda")
    _, topk_ids = torch.topk(gating_logits, top_k, dim=1)

    return topk_ids, topk_weights


def _run_attn_rank_logic(
    rank: int,
    world_size: int,
    num_attn_ranks: int,
    num_experts: int,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
    group: dist.ProcessGroup,
):
    """Handles all logic for an Attention rank."""

    # Initialize the Attention data router
    num_ffn_ranks = world_size - num_attn_ranks

    router = AfdDataRouterAttn(
        rank,
        world_size,
        num_attn_ranks,
        num_experts,
        num_tokens,  # num_max_dispatch_tokens_per_rank
        hidden_dim,
        group,
    )

    # --- Create Mock Input Data ---
    torch.manual_seed(rank)
    hidden_states = torch.randn(
        num_tokens, hidden_dim, device=f"cuda:{rank}", dtype=torch.bfloat16
    )

    topk_ids, topk_weights = _generate_mock_routing_data(
        seed=rank,  # Use the rank as the seed for this rank's data
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
    )

    # --- 1. Prepare Phase: Send data ---
    router.prepare(hidden_states, topk_weights, topk_ids, num_experts, None)
    torch.cuda.synchronize(device=None)

    # --- 2. Finalize Phase: Receive combined result ---
    final_output = router.finalize()
    torch.cuda.synchronize(device=None)

    # --- Verification on Attn Rank ---
    if final_output.shape != hidden_states.shape:
        print(
            f"*** ERROR *** Shape mismatch: final_output.shape={final_output.shape}, hidden_states.shape={hidden_states.shape}"
        )

    expected_output = torch.zeros_like(hidden_states)
    for token_idx in range(num_tokens):
        for k_idx in range(top_k):
            expert_id = topk_ids[token_idx, k_idx].item()
            weight = topk_weights[token_idx, k_idx]

            ffn_rank_id = expert_id % num_ffn_ranks
            ffn_rank = ffn_rank_id + num_attn_ranks

            contribution = hidden_states[token_idx] * (ffn_rank + 2.0)
            expected_output[token_idx] += weight * contribution

    if not torch.allclose(final_output, expected_output, atol=1.0):
        print(
            f"*** ERROR *** Output mismatch: final_output={final_output}, expected_output={expected_output}"
        )


def _run_ffn_rank_logic(
    rank: int,
    world_size: int,
    num_attn_ranks: int,
    num_experts: int,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
    group: dist.ProcessGroup,
):
    """Handles all logic for an FFN/Expert rank."""
    # Initialize the FFN data router
    num_ffn_ranks = world_size - num_attn_ranks
    router = AfdDataRouterFfn(
        rank,
        world_size,
        num_attn_ranks,
        num_experts,
        num_tokens,  # num_max_dispatch_tokens_per_rank
        hidden_dim,
        group,
    )

    # --- 1. Prepare Phase: Receive data ---
    dummy_a1 = torch.empty(0, hidden_dim, device=f"cuda:{rank}")
    dummy_topk_ids = torch.empty(0, top_k, dtype=torch.int64, device=f"cuda:{rank}")
    payload = router.prepare(dummy_a1, None, None, dummy_topk_ids, num_experts, None)
    torch.cuda.synchronize(device=None)

    # --- Verification on FFN Rank ---
    expected_tokens_count = 0
    ffn_rank_id = rank - num_attn_ranks

    for attn_rank_idx in range(num_attn_ranks):
        topk_ids, _ = _generate_mock_routing_data(
            seed=attn_rank_idx,
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
        )
        for token_idx in range(num_tokens):
            for k_idx in range(top_k):
                expert_id = topk_ids[token_idx, k_idx].item()
                if (expert_id % num_ffn_ranks) == ffn_rank_id:
                    expected_tokens_count += 1

    recved_tokens_count = payload.expert_tokens_meta.expert_num_tokens.sum().item()
    if recved_tokens_count != expected_tokens_count:
        print(
            f"*** ERROR *** Token count mismatch on FFN rank {rank}: recved_tokens_count={recved_tokens_count}, expected_tokens_count={expected_tokens_count}"
        )

    # Simulate expert computation
    processed_data = payload.expert_x * (rank + 2.0)

    # --- 2. Finalize Phase: Send processed data back ---
    dummy_topk_weights = torch.empty(0, top_k, device=f"cuda:{rank}")
    dummy_topk_ids = torch.empty(0, top_k, device=f"cuda:{rank}")

    output = router.finalize(
        processed_data, dummy_topk_weights, dummy_topk_ids, False, None, None
    )
    torch.cuda.synchronize(device=None)

    if output.numel() != 0:
        print(f"*** ERROR *** Output tensor not empty: output.numel()={output.numel()}")


def router_test_runner(
    rank: int,
    world_size: int,
    num_attn_ranks: int,
    num_experts: int,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
):
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{rank}"),
    )
    torch.set_default_device("cuda")
    torch.cuda.set_device(rank)
    group = dist.new_group(list(range(world_size)))

    try:
        assert isinstance(group, dist.ProcessGroup)

        if rank < num_attn_ranks:
            _run_attn_rank_logic(
                rank,
                world_size,
                num_attn_ranks,
                num_experts,
                num_tokens,
                hidden_dim,
                top_k,
                group,
            )
        else:
            _run_ffn_rank_logic(
                rank,
                world_size,
                num_attn_ranks,
                num_experts,
                num_tokens,
                hidden_dim,
                top_k,
                group,
            )
    finally:
        dist.barrier()
        dist.destroy_process_group()


# --- The Main Test Class ---
class AfdDataRouterTest(TestCase):
    # Test parameters
    DTYPES = [torch.bfloat16]
    NUM_TOKENS = [128]
    HIDDEN_SIZES = [2048]
    NUM_EXPERTS = [128]
    TOP_K = [8]
    INTER_SIZES = [1024]

    # Test configurations: (world_size, num_attn_ranks)
    TEST_CONFIGS = [
        # (2, 1),
        (4, 2),  # 4 ranks, 2 attn ranks, 2 expert ranks
        # (4, 3),  # 4 ranks, 3 attn ranks, 1 expert ranks
        # (8, 7),  # 8 ranks, 7 attn ranks, 1 expert rank
    ]

    def test_afd_data_router(self):
        """
        Main test runner that spawns distributed processes.
        """

        os.environ["ACCL_LOW_LATENCY_OPTIMIZE"] = "1"
        os.environ["ACCL_TOPO_FIX"] = "1"
        os.environ["ACCL_LOAD_BALANCE"] = "1"
        os.environ["NCCL_TOPO_DUMP_FILE"] = "/tmp/nccl_topo.xml"
        os.environ["MASTER_PORT"] = str(torch.randint(10000, 65535, (1,)).item())
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["USE_DEEPEP_LOW_LATENCY"] = "1"

        for world_size, num_attn_ranks in self.TEST_CONFIGS:
            for params in itertools.product(
                self.NUM_EXPERTS,
                self.NUM_TOKENS,
                self.HIDDEN_SIZES,
                self.TOP_K,
            ):
                num_ffn_ranks = world_size - num_attn_ranks
                assert (
                    num_ffn_ranks > 0 and params[0] % num_ffn_ranks == 0
                ), "Invalid FFN rank configuration"

                device_properties = torch.cuda.get_device_properties(0)
                accl_num_warp_groups: int = math.ceil(
                    params[0]
                    * world_size
                    / (world_size - num_attn_ranks)
                    / device_properties.multi_processor_count
                )  # assume tp_size = 1
                os.environ["ACCL_DISPATCH_NUM_WARP_GROUPS"] = str(accl_num_warp_groups)
                os.environ["ACCL_COMBINE_NUM_WARP_GROUPS"] = str(accl_num_warp_groups)
                with self.subTest(
                    world_size=world_size, num_attn_ranks=num_attn_ranks, params=params
                ):
                    # The arguments passed to the worker function.
                    # The first argument must be world_size, followed by the test parameters.
                    spawn_args = (world_size, num_attn_ranks, *params)

                    torch.multiprocessing.spawn(
                        router_test_runner,
                        args=spawn_args,
                        nprocs=world_size,
                    )


if __name__ == "__main__":
    main()
