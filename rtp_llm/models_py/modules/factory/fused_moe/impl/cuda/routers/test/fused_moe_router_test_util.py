import os
from typing import Tuple

import torch
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.ops import ParallelismConfig
from rtp_llm.test.utils.numeric_util import per_token_cast_back


def set_start_method() -> None:
    # Unit tests may be run under different runners; avoid crashing if already set.
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass


def setup_low_latency_env(use_nvlink_for_low_latency_mode: bool) -> None:
    """
    Follow DeepEP tests' convention: if user didn't set ACCL_LOW_LATENCY_OPTIMIZE,
    set it based on whether we want NVLink-only mode or allow RDMA mode.
    """
    if use_nvlink_for_low_latency_mode:
        os.environ.setdefault("ACCL_LOW_LATENCY_OPTIMIZE", "1")
    else:
        os.environ["ACCL_TOPO_FIX"] = "1"
        os.environ["ACCL_LOAD_BALANCE"] = "1"
        os.environ["NVSHMEM_IB_GID_INDEX"] = "3"
        os.environ.setdefault("ACCL_LOW_LATENCY_OPTIMIZE", "0")
    os.environ["ACCL_DISPATCH_NUM_WARP_GROUPS"] = "4"
    os.environ["ACCL_COMBINE_NUM_WARP_GROUPS"] = "4"


def build_quant_config(mode: str) -> FusedMoEQuantConfig:
    if mode == "bf16":
        return FusedMoEQuantConfig(quant_dtype=None)
    if mode == "fp8_per_block":
        return FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=[128, 128],
        )
    if mode == "fp8_per_token":
        # Match DeepEP per-token-quant test setup.
        os.environ["ACCL_FP8_CAST_LEVEL"] = "2"
        return FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
            per_out_ch_quant=False,
            block_shape=None,
        )
    raise ValueError(f"unknown quant mode: {mode}")


def dequant_per_expert(
    expert_x: torch.Tensor,
    expert_x_scale: torch.Tensor,
    per_token_quant: bool,
) -> torch.Tensor:
    # expert_x: [E_local, M, K]
    # expert_x_scale:
    # - fp8_per_block: [E_local, M, K/128]
    # - fp8_per_token: [E_local, M, 1]
    out = torch.empty(expert_x.size(), dtype=torch.bfloat16, device=expert_x.device)
    for i in range(expert_x.size(0)):
        out[i] = per_token_cast_back(
            expert_x[i],
            expert_x_scale[i],
            pertoken_quant=per_token_quant,
        )
    return out


def wait_dispatch_events(payload) -> None:
    # PEO overlap mode returns events that guard when recv buffers become visible.
    evs = getattr(payload, "dispatch_recv_events", None)
    if not evs:
        return
    for ev in evs:
        torch.cuda.current_stream().wait_event(ev)
    # Make sure we can safely read expert_x/expert_x_scale.
    torch.cuda.synchronize()


def build_parallelism_config(
    rank: int,
    world_size: int,
    tp_size: int,
    nccl_port: int,
) -> Tuple[ParallelismConfig, int]:
    dp_size = world_size // tp_size
    ep_size = world_size

    parallelism_config = ParallelismConfig()
    parallelism_config.tp_size = tp_size
    parallelism_config.tp_rank = rank % tp_size
    parallelism_config.dp_size = dp_size
    parallelism_config.dp_rank = rank // tp_size
    parallelism_config.ep_size = ep_size
    parallelism_config.ep_rank = rank % ep_size
    parallelism_config.local_rank = rank
    parallelism_config.world_size = world_size
    parallelism_config.world_rank = rank
    parallelism_config.local_world_size = world_size
    parallelism_config.nccl_ip = "127.0.0.1"
    parallelism_config.th_nccl_port = nccl_port
    return parallelism_config, dp_size


def create_test_data(
    dp_size: int,
    max_generate_batch_size: int,
    hidden_size: int,
    num_experts: int,
    num_topk: int,
    quant_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create deterministic test data across all ranks.

    Quant-error control:
    - fp8_per_block: only require the LAST 128 elements to be identical (one fp8 block),
      other dimensions can vary and are not validated.
    - fp8_per_token: the whole hidden_size has identical values (stronger constraint),
      because per-token scales depend on all hidden dims.
    """
    # Keep values in a small range to reduce FP8 quantization error. The scale factor
    # does not need to match block size (128); it's purely for numeric stability.
    token_vals = (
        torch.arange(
            dp_size * max_generate_batch_size, device="cuda", dtype=torch.float32
        ).view(dp_size, max_generate_batch_size, 1)
        * 0.01
    )

    if quant_mode == "fp8_per_token":
        # Entire hidden vector shares one value -> smaller dynamic range for per-token quant.
        hidden_states = token_vals.repeat(1, 1, hidden_size).to(torch.bfloat16)
    else:
        # bf16 / fp8_per_block: allow other dims to vary; only enforce LAST 128 dims identical.
        assert (
            hidden_size >= 128
        ), f"hidden_size must be >= 128 to validate tail block, got {hidden_size}"
        hidden_states = (
            torch.randn(
                (dp_size, max_generate_batch_size, hidden_size),
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.01
        )
        hidden_states[:, :, -128:] = token_vals.repeat(1, 1, 128).to(torch.bfloat16)

    topk_ids = torch.rand(
        (dp_size, max_generate_batch_size, num_experts), device="cuda"
    ).topk(num_topk, dim=-1, largest=True)[1]
    topk_weights = (
        torch.ones((dp_size, max_generate_batch_size, num_topk), device="cuda").to(
            torch.float32
        )
        / num_topk
    )
    return hidden_states, topk_ids, topk_weights


def build_reference_recv(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    num_experts: int,
    num_max_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build reference expert recv buffers for this ep_rank."""
    assert num_experts % ep_size == 0
    num_local_experts = num_experts // ep_size
    hidden_size = hidden_states.size(-1)
    ref_recv_x = torch.zeros(
        (num_local_experts, num_max_tokens, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    ref_recv_count = torch.zeros((num_local_experts,), dtype=torch.int32, device="cuda")
    for local_expert_id in range(num_local_experts):
        expert_id = ep_rank * num_local_experts + local_expert_id
        expert_mask = (topk_ids == expert_id).any(dim=-1)  # [dp_size, T]
        num_selected_tokens = expert_mask.sum()
        ref_recv_x[local_expert_id, :num_selected_tokens] = hidden_states[expert_mask]
        ref_recv_count[local_expert_id] = num_selected_tokens
    return ref_recv_x, ref_recv_count
