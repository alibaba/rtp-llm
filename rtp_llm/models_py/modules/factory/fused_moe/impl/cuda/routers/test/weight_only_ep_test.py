"""Two CUDA ranks: checkpoint -> local experts -> routed SwiGLU -> collective."""

import multiprocessing as mp
import os

import torch

from rtp_llm.config.quant_config import (
    AWQConfig,
    GPTQConfig,
    WeightOnlyInt8PerChannelQuantConfig,
)
from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
)
from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPWrapper
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.test.deepep_normal_router_test import (
    init_router,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.test.deepep_normal_router_test import (
    test_single as run_ranks,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


def make_layer(config, method, dtype):
    model = config.model_config
    hidden, inter, group = model.hidden_size, 32, 32
    source = (
        WeightOnlyInt8PerChannelQuantConfig()
        if method == "INT8"
        else (AWQConfig if method == "awq" else GPTQConfig)(4, group, True)
    )
    model.quant_config = source
    model.data_type = "bf16" if dtype == torch.bfloat16 else "fp16"
    model.activation_type = "SiGLU"
    model.moe_inter_size = inter
    layer = BaseMoEExperts(
        num_experts=model.expert_num,
        top_k=model.moe_k,
        hidden_size=hidden,
        moe_intermediate_size=inter,
        tp_size=config.tp_size,
        tp_rank=config.tp_rank,
        ep_size=config.ep_size,
        ep_rank=config.ep_rank,
        params_dtype=dtype,
        model_config=model,
        parallelism_config=config.parallelism_config,
        moe_config=config.moe_config,
        quant_config=QuantizationConfig(method, source),
        layer_idx=0,
    )
    # All lanes carry code 4 and stored zero 1. No production packer or
    # dequantizer is used to construct the test checkpoint or the reference.
    for expert in range(model.expert_num):
        for proj, k, n, numerator in (
            ("up_proj", hidden, inter, 1),
            ("gate_proj", hidden, inter, 2),
            ("down_proj", inter, hidden, expert + 1),
        ):
            value = numerator / k
            prefix = f"{expert}.{proj}."
            if method == "INT8":
                pieces = {"weight": torch.full((n, k), value, dtype=dtype)}
            else:
                shape = (k, n // 8) if method == "awq" else (k // 8, n)
                pieces = {
                    "qweight": torch.full(shape, 0x44444444, dtype=torch.int32),
                    "qzeros": torch.full(
                        (k // group, n // 8), 0x11111111, dtype=torch.int32
                    ),
                    "scales": torch.full((k // group, n), value, dtype=dtype),
                }
            layer.load_weights({prefix + key: tensor for key, tensor in pieces.items()})
    layer.process_weights_after_loading()
    return layer


def run_worker(rank, use_fp8, token_counts, parallelism, port, topk_ids_dtype):
    del use_fp8, token_counts
    tp_mode = parallelism.tp_size == parallelism.ep_size
    config, _ = init_router(
        rank, False, parallelism, port, all_gather_only=tp_mode, top_k=2
    )
    try:
        for dtype in (torch.float16, torch.bfloat16):
            for method in ("awq", "gptq", "INT8"):
                layer = make_layer(config, method, dtype)
                if tp_mode:
                    expected_router = "PureTpRouterWeightOnly"
                elif dtype == torch.float16:
                    expected_router = "StageRouterWeightOnlyFp16"
                else:
                    expected_router = "DeepepNormalRouterWeightOnly"
                assert type(layer.fused_moe.router).__name__ == expected_router
                # Different DP origins submit different batch sizes and values.
                n = 3 if tp_mode else 3 + rank
                t = ((torch.arange(n) + 1 + config.dp_rank) / 8 + 0.00037).to(dtype)
                x = t[:, None].expand(n, config.hidden_size).contiguous()
                last = layer.num_experts - 1
                for pair in ((0, last), (last - 1, last), (0, 1)):
                    # Second/third cases leave an entire EP rank with no tokens.
                    ids = (
                        torch.tensor(pair, dtype=topk_ids_dtype)
                        .expand(n, 2)
                        .contiguous()
                    )
                    wei = torch.tensor([0.25, 0.75]).expand(n, 2).contiguous()
                    actual = layer(x, wei, ids)
                    # AWQ: q-z=3; GPTQ: q-(z+1)=2. Online INT8 uses
                    # max/128 and clips positive-max code 128 to 127.
                    coefficient = {"awq": 3, "gptq": 2, "INT8": 127 / 128}[method]
                    up = (t.float() * coefficient).to(dtype)
                    gate = (t.float() * (2 * coefficient)).to(dtype)
                    act = torch.nn.functional.silu(gate) * up
                    partial = (
                        act.float()[:, None] * (ids + 1).float() * coefficient * wei
                    )
                    expected = partial.to(dtype).sum(dim=1)[:, None].expand_as(x)
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=0.012 if dtype == torch.bfloat16 else 0.002,
                        atol=0,
                    )
    finally:
        DeepEPWrapper.reset()
        destroy_distributed_environment()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    # Permit the default TP=EP test to run without a DeepEP installation.
    transport = os.environ.get("WEIGHT_ONLY_EP_TRANSPORT", "both")
    if transport not in ("tp", "normal", "both"):
        raise ValueError(f"Unknown WEIGHT_ONLY_EP_TRANSPORT={transport!r}")
    if transport in ("tp", "both"):
        run_ranks(2, 2, False, worker=run_worker)
    if transport in ("normal", "both"):
        run_ranks(2, 1, False, worker=run_worker)
