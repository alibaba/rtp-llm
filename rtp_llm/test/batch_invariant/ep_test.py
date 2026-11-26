"""Verify routed expert shards and their combine on a four-GPU NCCL group."""

import argparse
import os

import torch

from rtp_llm.config.collective_config import CollectiveConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    _transform_scale_ue8m0,
    quant_weight_ue8m0,
)
from rtp_llm.models_py.modules import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops import MoeConfig, NcclCommConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


def expert_weights(rank, fp8):
    # For x[0] equal to 1 or 2, the rounded SwiGLU is 64*x[0]**2. The
    # down projection puts (global expert ID + 1)*x[0]**2 in channel 0.
    up = torch.zeros((2, 256, 512), dtype=torch.bfloat16, device="cuda")
    down = torch.zeros((2, 512, 128), dtype=torch.bfloat16, device="cuda")
    up[:, 0, 0], up[:, 128, 0] = 8, 8
    down[0, 0, 0], down[1, 0, 0] = (2 * rank + 1) / 64, (2 * rank + 2) / 64
    if not fp8:
        return {W.moe_w1: up, W.moe_w2: down}
    weights = {}
    for weight_key, scale_key, matrix in (
        (W.moe_w1, W.moe_s1, up),
        (W.moe_w2, W.moe_s2, down),
    ):
        weights[weight_key], scale = quant_weight_ue8m0(matrix, [128, 128])
        weights[scale_key] = _transform_scale_ue8m0(scale, mn=matrix.shape[-2])
    return weights


def check_experts(parallel, rank, tp, fp8):
    model = ModelConfig()
    model.model_type = "qwen35_moe"
    model.data_type = "bf16"
    model.hidden_size, model.expert_num, model.moe_k, model.moe_inter_size = (
        512,
        8,
        4,
        128,
    )
    moe = MoeConfig()
    moe.use_all_gather = True
    moe.use_deepep_moe = False
    if fp8:
        model.quant_config = Fp8BlockWiseQuantConfig()
        moe.moe_strategy = "fp8_per_block_no_dp" if tp else "fp8_per_block_pure_dp"
    else:
        moe.moe_strategy = "no_quant_cpp" if tp else "no_quant_pure_dp"
    config = MoEConfigAdapter(model, parallel, moe, batch_invariant=True)
    layer = FusedMoeFactory().create_fused_moe(config, expert_weights(rank, fp8))
    assert (config.local_expert_start, config.local_expert_end) == (
        2 * rank,
        2 * rank + 2,
    )

    cases = (
        (
            ((1, 1, 1, 1), 0, 0),
            ((256, 256, 256, 256), 0, 255),
            ((257, 257, 257, 257), 0, 0),
        )
        if tp
        else (
            ((1, 0, 3, 7), 0, 0),
            ((256, 2, 0, 255), 0, 255),
            ((256, 2, 0, 257), 3, 0),
        )
    )
    selected = [1, 2, 5, 6]
    for lengths, owner, target_row in cases:
        n = lengths[rank]
        x = torch.zeros((n, model.hidden_size), device="cuda", dtype=torch.bfloat16)
        x[:, 0] = 2
        if tp or rank == owner:
            x[target_row, 0] = 1
        ids = (
            torch.tensor(selected, device=x.device, dtype=layer.topk_ids_dtype)
            .expand(n, -1)
            .contiguous()
        )
        probabilities = torch.full(
            ids.shape, 0.25, device=x.device, dtype=torch.float32
        )
        payload = layer.router.prepare(x, None, None, probabilities, ids)
        global_row = target_row if tp else owner * max(lengths) + target_row
        if fp8:
            expected_ids = [e - 2 * rank if e // 2 == rank else -1 for e in selected]
            assert payload.expert_topk_ids[global_row].tolist() == expected_ids
        else:
            assert payload.expert_topk_ids[global_row].tolist() == selected
        output = layer.fused_experts.execute(payload, "SiGLU", None, None, False, None)
        partial = output.fused_expert_output[global_row]
        assert partial[0].item() == (selected[rank] + 1) / 4
        assert not partial[1:].count_nonzero().item()
        result = layer.router.finalize(
            output, probabilities, ids, False, {"original_num_tokens": n}
        )
        expected = torch.zeros_like(x)
        expected[:, 0] = x[:, 0].square() * (sum(selected) + len(selected)) / 4
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        if rank == 0:
            print(
                f"{layer.strategy_name} {lengths=} {owner=} {target_row=} differences=0",
                flush=True,
            )


def check_rank_order(rank, legacy):
    for rows in (1, 11, 193, 256, 257, 525):
        # Different summation orders produce 0, 1, or 2. Rank order yields 1.
        contribution = (2**28, 1, -(2**28), 1)[rank]
        x = torch.full(
            (4 * rows, 128), contribution, device="cuda", dtype=torch.bfloat16
        )
        out = collective.reduce_scatter(
            x, collective.Group.STAGE, batch_invariant=not legacy
        )
        gathered = collective.all_gather(
            out[:1, :1].contiguous(), collective.Group.STAGE
        ).flatten()
        if rank == 0:
            print(
                f"reduce-scatter {rows=} values_by_destination={gathered.tolist()}",
                flush=True,
            )
        if not legacy:
            torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tp", action="store_true", help="Use TP4/DP1 instead of TP1/DP4"
    )
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--legacy-reduce", action="store_true")
    args = parser.parse_args()
    assert int(os.environ["WORLD_SIZE"]) == 4
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    parallel = ParallelismConfig()
    parallel.world_size = parallel.local_world_size = parallel.ep_size = 4
    parallel.tp_size = parallel.ffn_tp_size = 4 if args.tp else 1
    parallel.dp_size = 1 if args.tp else 4
    parallel.world_rank = parallel.local_rank = parallel.ep_rank = rank
    parallel.tp_rank = rank if args.tp else 0
    parallel.dp_rank = 0 if args.tp else rank
    collective.init_distributed_environment(
        parallel,
        NcclCommConfig(nccl_ip=os.environ["MASTER_ADDR"]),
        int(os.environ["MASTER_PORT"]),
        collective_config=CollectiveConfig(batch_invariant=True),
    )
    try:
        if not args.tp:
            check_rank_order(rank, args.legacy_reduce)
        if not args.legacy_reduce:
            check_experts(parallel, rank, args.tp, args.fp8)
    finally:
        collective.destroy_distributed_environment()


if __name__ == "__main__":
    main()
