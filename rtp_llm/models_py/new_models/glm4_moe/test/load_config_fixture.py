"""Small CPU topology for direct GQA/MoE checkpoint tests."""

import types

import torch

from rtp_llm.models_py.model_loader import NewLoaderConfig
from rtp_llm.models_py.quant_methods import QuantizationConfig


def make_load_config(tp=1, rank=0, ep=1, ep_rank=0):
    world = max(tp, ep)
    parallel = types.SimpleNamespace(
        tp_size=tp,
        tp_rank=rank,
        ep_size=ep,
        ep_rank=ep_rank,
        dp_size=1,
        dp_rank=0,
        pp_size=1,
        pp_rank=0,
        pp_stage_layer_counts=[],
        world_size=world,
        world_rank=max(rank, ep_rank),
        local_world_size=world,
        local_rank=rank,
        get_attn_tp_size=lambda: tp,
        get_attn_tp_rank=lambda: rank,
        get_ffn_tp_size=lambda: tp,
        get_ffn_tp_rank=lambda: rank,
        prefill_cp_config=types.SimpleNamespace(
            is_enabled=lambda: False, is_prefill_enabled=lambda: False
        ),
        ffn_disaggregate_config=types.SimpleNamespace(enable_ffn_disaggregate=False),
    )
    moe = types.SimpleNamespace(
        ll_num_max_token=1,
        masked_max_token_num=1,
        moe_strategy=0,
        use_mori_ep=False,
        use_deepep_moe=False,
        use_deepep_low_latency=False,
        use_all_gather=True,
        fake_balance_expert=False,
    )
    return NewLoaderConfig(
        tp_size=tp,
        tp_rank=rank,
        ep_size=ep,
        ep_rank=ep_rank,
        device="cpu",
        compute_dtype=torch.float32,
        quant_config=QuantizationConfig("none"),
        parallelism_config=parallel,
        moe_config=moe,
    )
