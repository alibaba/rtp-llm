# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/kernels.py
# but make some modifications for RTP-LLM
# Licensed under the Apache License, Version 2.0
import logging
import math
import os
from contextlib import nullcontext
from functools import cache
from typing import Any, Dict, Optional

import torch
import triton.language as tl

logger = logging.getLogger(__name__)

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    configure_deep_gemm_mk_alignment,
    configure_deep_gemm_num_sms,
    get_theoretical_mk_alignment_for_contiguous_layout,
    is_deep_gemm_e8m0_used,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
    GlobalExpertIdsContext,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.qwen4_exp.moe.compact_fp8_decode import (
    _STABLE_ALIGNMENT_MAX_ROUTES,
    compact_fp8_decode,
    h20_compact_moe_phase,
)
from rtp_llm.models_py.triton_kernels.common.activation import (
    MaskedSiluInputLayout,
    create_packed_scale_tensor,
    silu_and_mul,
    silu_and_mul_masked_post_quant_packed_fwd,
    silu_mul_masked_fp8_post_quant_fwd,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    ep_scatter_v2,
    tma_align_input_scale,
)
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    get_default_config as get_triton_moe_config,
)
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    invoke_fused_moe_kernel as invoke_triton_moe_kernel,
)
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    moe_align_block_size_compiled,
)
from rtp_llm.models_py.triton_kernels.moe.sm90_prefill_silu_fp8 import (
    silu_mul_native_fp8,
)
from rtp_llm.models_py.utils.arch import get_num_device_sms, get_sm
from rtp_llm.models_py.utils.math import align, ceil_div
from rtp_llm.models_py.utils.memory import dispose_tensor
from rtp_llm.ops import compute_ops
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128
from rtp_llm.utils.model_weight import W

_native_contiguous_silu_fp8 = getattr(compute_ops, "sm90_contiguous_silu_fp8", None)

_SM120_TRITON_MIN_TOKENS = 1
_SM120_TRITON_MAX_TOKENS = 32
_SM120_TRITON_MAX_TOKENS_ENV = "RTP_LLM_SM120_TRITON_FP8_MAX_TOKENS"
_CUDA_GRAPH_WARMUP_FORWARD_ENV = "RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD"
_SM120_TUNED_FP8_CONFIGS = {
    # Qwen3-30B-A3B, TP=1. Exhaustive search over BM={8,16,32},
    # BN={64,128,256}, BK={64,128}, warps={4,8}, and stages={2,3,4,5}.
    (128, 1536, 2048, 8): {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    },
    (128, 2048, 768, 8): {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
    },
}


def _get_sm120_triton_max_tokens() -> int:
    raw_max_tokens = os.environ.get(
        _SM120_TRITON_MAX_TOKENS_ENV,
        str(_SM120_TRITON_MAX_TOKENS),
    )
    try:
        max_tokens = int(raw_max_tokens)
    except ValueError as error:
        raise ValueError(
            f"{_SM120_TRITON_MAX_TOKENS_ENV} must be an integer, "
            f"got {raw_max_tokens!r}"
        ) from error
    if not 0 <= max_tokens <= _SM120_TRITON_MAX_TOKENS:
        raise ValueError(
            f"{_SM120_TRITON_MAX_TOKENS_ENV} must be in "
            f"[0, {_SM120_TRITON_MAX_TOKENS}], got {max_tokens}"
        )
    return max_tokens


def _is_cuda_graph_warmup_or_capture() -> bool:
    """Return whether this forward is preparing or capturing a CUDA graph.

    ``enable_cuda_graph`` is an engine-level capability and is also true for
    ordinary eager prefill forwards.  The C++ graph runner marks its eager
    warmup with an environment flag, while PyTorch exposes the subsequent
    capture directly.  Requiring either signal keeps short eager prefill out
    of the static ``torch.compile(dynamic=False)`` routing path.
    """
    return (
        os.environ.get(_CUDA_GRAPH_WARMUP_FORWARD_ENV) == "1"
        or torch.cuda.is_current_stream_capturing()
    )


@cache
def _log_sm120_triton_fp8_path(
    min_tokens: int, max_tokens: int, config_source: str
) -> None:
    if max_tokens == 0:
        logger.info(
            "SM120 CUDA Graph Triton FP8 MoE is disabled by %s=0",
            _SM120_TRITON_MAX_TOKENS_ENV,
        )
        return
    logger.info(
        "SM120 CUDA Graph MoE uses Triton FP8 for %d-%d tokens; "
        "config source=%s; other shapes use DeepGEMM (set %s=0 to disable)",
        min_tokens,
        max_tokens,
        config_source,
        _SM120_TRITON_MAX_TOKENS_ENV,
    )


def align_up_math(n: int, alignment: int = 128) -> int:
    return int(math.ceil(n / alignment)) * alignment


def get_sm120_triton_fp8_config(
    M: int,
    E: int,
    N: int,
    K: int,
    top_k: int,
) -> Dict[str, Any]:
    """Return SM120 FP8 MoE configs without changing the SM90 defaults."""
    tuned_config = _SM120_TUNED_FP8_CONFIGS.get((E, N, K, top_k))
    if tuned_config is not None and M <= _SM120_TRITON_MAX_TOKENS:
        return dict(tuned_config)

    # Keep the SM120 override pure even if the generic selector starts caching
    # configs in the future.
    config = dict(get_triton_moe_config(M, E, N, K, top_k))
    avg_tokens_per_expert = M * top_k / max(E, 1)
    if avg_tokens_per_expert <= 2:
        # Generic SM120 small-batch fallback from the original optimization.
        config.update(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 4,
            }
        )
    return config


def _empty_fp32_moe_scale(shape, device, *, mn_major=False):
    if not mn_major:
        return torch.empty(shape, device=device, dtype=torch.float32)
    # Expert rows are padded to 128, satisfying the 16-byte TMA alignment.
    # DeepGEMM accepts these strides directly without its transpose kernel.
    m, groups = shape[-2:]
    assert m % 4 == 0
    strides = (1, m) if len(shape) == 2 else (m * groups, 1, m)
    return torch.empty_strided(shape, strides, device=device, dtype=torch.float32)


_CONTIGUOUS_MOE_BUCKET_ROWS = 73728


def _acquire_contiguous_moe_workspace(
    device: torch.device,
    all_tokens: int,
    K: int,
    N: int,
    in_dtype: torch.dtype,
    *,
    graph_layout: bool,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if (
        graph_layout
        or all_tokens <= 4096
        or torch.is_grad_enabled()
        or _is_cuda_graph_warmup_or_capture()
    ):
        return None
    alloc_tokens = (
        _CONTIGUOUS_MOE_BUCKET_ROWS
        if all_tokens <= _CONTIGUOUS_MOE_BUCKET_ROWS
        else align_up_math(all_tokens, 8192)
    )
    in_bytes = align_up_math(alloc_tokens * K * in_dtype.itemsize, 256)
    gateup_bytes = align_up_math(alloc_tokens * N * 2, 256)
    down_bytes = align_up_math(alloc_tokens * K * 2, 256)
    needed_bytes = max(in_bytes + gateup_bytes, down_bytes)
    buf = torch.empty(needed_bytes, device=device, dtype=torch.uint8)
    in_end = all_tokens * K * in_dtype.itemsize
    gateup_off = align_up_math(in_end, 256)
    gateup_end = gateup_off + all_tokens * N * 2
    down_end = all_tokens * K * 2
    recv_x = buf[:in_end].view(in_dtype).view(all_tokens, K)
    gateup_out = buf[gateup_off:gateup_end].view(torch.bfloat16).view(all_tokens, N)
    down_out = buf[:down_end].view(torch.bfloat16).view(all_tokens, K)
    return recv_x, gateup_out, down_out


class DeepGemmHybridExecutor(FusedMoeExpertExecutor):
    _qwen4_sm90_mixed_graph: bool = False
    BLOCK_SIZE = 128
    EXPERT_ALIGNMENT = 128
    DEEPGEMM_BLOCK_SHAPE: list[int] = [128, 128]

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.DEEPGEMM_CONTINUOUS

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepGemmHybridExecutor can handle the configuration"""
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(resolver.is_bf16(config))
        checker.check(has_deep_gemm())
        checker.check(get_sm()[0] >= 9)
        checker.check(not config.enable_cuda_graph or get_sm()[0] == 12)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.num_experts = config.expert_num

        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.start_expert_id = self.ep_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1

        self.top_k = config.moe_k
        self.activation = config.activation_type
        self.renormalize = True
        self.use_fp8_w8a8 = True
        self.use_block_quant = True
        # Per-process pinned staging buffers for the eager expert-count path:
        # one for the device->host routed counts, one for the host->device
        # padded counts.  A plain ``.cpu()``/``torch.tensor(pin_memory=True)``
        # pays a pageable staging copy and a fresh cudaHostAlloc every layer;
        # these buffers are reallocated only when the expert count changes.
        self._expert_counts_pinned = None
        self._padded_counts_pinned = None

        self.masked_max_token_num = config.masked_max_token_num
        self.enable_cuda_graph = config.enable_cuda_graph
        self.is_sm120 = get_sm()[0] == 12
        if self.is_sm120 and self.enable_cuda_graph and self.ep_size == 1:
            self.sm120_triton_max_tokens = _get_sm120_triton_max_tokens()
        else:
            # Do not let an SM120-only rollback knob affect H20/SM100 or an
            # executor configuration that can never enter the Triton path.
            self.sm120_triton_max_tokens = 0

        # 权重初始化
        self.w13_weight = weights[W.moe_w1]
        self.w2_weight = weights[W.moe_w2]
        self.w13_weight_scale_inv = weights[W.moe_s1]
        self.w2_weight_scale_inv = weights[W.moe_s2]
        self.w13_weight_scale = None
        self.w2_weight_scale = None

        self.E, self.N, self.K = self.w13_weight.size()
        assert self.N % 2 == 0
        assert self.w2_weight.size(0) == self.E
        assert self.w2_weight.size(1) == self.K
        assert self.w2_weight.size(2) == self.N // 2

        if self.is_sm120 and self.enable_cuda_graph and self.ep_size == 1:
            gate_shape = (self.E, self.N, self.K, self.top_k)
            down_shape = (self.E, self.K, self.N // 2, self.top_k)
            config_source = (
                "tuned"
                if gate_shape in _SM120_TUNED_FP8_CONFIGS
                and down_shape in _SM120_TUNED_FP8_CONFIGS
                else "generic"
            )
            _log_sm120_triton_fp8_path(
                _SM120_TRITON_MIN_TOKENS,
                self.sm120_triton_max_tokens,
                config_source,
            )

        self.w13_weight_fp8 = (
            self.w13_weight,
            self.w13_weight_scale_inv,
        )
        self.w2_weight_fp8 = (
            self.w2_weight,
            self.w2_weight_scale_inv,
        )

        self.num_gemm_sms = get_num_device_sms()
        self._sm90_mn_major_scales = (
            torch.cuda.get_device_capability(self.w13_weight.device) == (9, 0)
            and (self.E, self.N, self.K) == (256, 1280, 2560)
            and self.top_k == 10
            and self.ep_size == 2
            and not is_deep_gemm_e8m0_used()
        )
        self._sm90_prefill_silu_fp8 = self._sm90_mn_major_scales
        self._sm90_cuda_contiguous_silu = (
            self._sm90_mn_major_scales
            and _native_contiguous_silu_fp8 is not None
        )
        self._h20_compact_fp8 = (
            os.environ.get("RTP_LLM_H20_COMPACT_FP8_MOE", "1") == "1"
            and self._sm90_mn_major_scales
            and config.tp_size == config.ep_size == 2
            and getattr(config, "dp_size", 0) == 1
            and getattr(getattr(config, "model_config", None), "_qwen4_full_fp8", False)
            and not getattr(config.model_config, "_qwen4_full_fp8_nvfp4", False)
            and self.w13_weight.dtype == self.w2_weight.dtype == torch.float8_e4m3fn
            and self.w13_weight_scale_inv.dtype
            == self.w2_weight_scale_inv.dtype
            == torch.float32
        )

    def _can_consume_h20_compact_global_ids(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> bool:
        if not self._h20_compact_fp8:
            return False
        phase = h20_compact_moe_phase.get()
        if _is_cuda_graph_warmup_or_capture():
            use_compact = self._qwen4_sm90_mixed_graph and phase in (
                "decode",
                "target_verify",
            )
        else:
            use_compact = not self.enable_cuda_graph or phase == "eager"
        if not use_compact or hidden_states.ndim != 2 or not hidden_states.is_cuda:
            return False
        rows = hidden_states.shape[0]
        max_rows = min(
            self.masked_max_token_num, _STABLE_ALIGNMENT_MAX_ROUTES // self.top_k
        )
        return (
            0 < rows <= max_rows
            and hidden_states.shape[1] == self.K
            and hidden_states.is_contiguous()
            and topk_ids.is_cuda
            and topk_ids.device == hidden_states.device
            and topk_ids.dtype in (torch.int32, torch.int64)
            and topk_ids.shape == (rows, self.top_k)
            and activation == "SiGLU"
            and expert_map is None
            and a1_scale is None
            and a2_scale is None
            and not apply_router_weight_on_input
            and not extra_expert_args
        )

    def can_consume_global_expert_ids(
        self,
        hidden_states,
        topk_ids,
        activation,
        expert_map,
        a1_scale,
        a2_scale,
        apply_router_weight_on_input,
        extra_expert_args,
    ) -> bool:
        return not self.enable_cuda_graph and self._can_consume_h20_compact_global_ids(
            hidden_states,
            topk_ids,
            activation,
            expert_map,
            a1_scale,
            a2_scale,
            apply_router_weight_on_input,
            extra_expert_args,
        )

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "hidden_states_fp8 is not initialized"
        token_num = payload.expert_x.shape[0]
        # This local routed path neither remaps partitioned expert ids nor
        # participates in an EP dispatch/combine collective.  expert_map=None
        # alone does not imply that the payload follows the non-EP contract.
        if (
            self.is_sm120
            and self.enable_cuda_graph
            and self.ep_size == 1
            and _is_cuda_graph_warmup_or_capture()
            and token_num >= _SM120_TRITON_MIN_TOKENS
            and token_num <= self.sm120_triton_max_tokens
            and expert_map is None
            and activation == "SiGLU"
            and not apply_router_weight_on_input
        ):
            return self.execute_triton_fp8(
                payload,
                activation,
                apply_router_weight_on_input,
            )
        # The contiguous path uses a shape-derived fixed workspace and performs
        # all routing metadata work on GPU, so it is safe to capture/replay.
        # It avoids the E * padded_M masked layout that dominates small decode
        # batches on SM120.
        if token_num <= self.masked_max_token_num and not self.enable_cuda_graph:
            return self.execute_masked(
                payload,
                activation,
                expert_map,
                a2_scale,
                apply_router_weight_on_input,
                extra_expert_args,
            )
        else:
            return self.execute_contiguous(
                payload,
                activation,
                expert_map,
                a2_scale,
                apply_router_weight_on_input,
                extra_expert_args,
            )

    def execute_triton_fp8(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> CombineForwardPayload:
        """Run small SM120 decode batches without per-expert 64-row padding."""
        if not self.is_sm120:
            raise RuntimeError("Triton FP8 MoE fast path requires SM120")
        if self.ep_size != 1:
            raise ValueError("Triton FP8 MoE fast path requires ep_size == 1")
        if activation != "SiGLU":
            raise ValueError("Triton FP8 MoE fast path only supports SiGLU")
        if apply_router_weight_on_input:
            raise ValueError(
                "Triton FP8 MoE fast path does not support router weight on input"
            )
        if payload.expert_x is None:
            raise ValueError("Triton FP8 MoE fast path requires expert_x")
        if payload.expert_x_scale is None:
            raise ValueError("Triton FP8 MoE fast path requires expert_x_scale")
        if payload.expert_topk_ids is None:
            raise ValueError("Triton FP8 MoE fast path requires expert_topk_ids")
        if payload.expert_topk_weights is None:
            raise ValueError("Triton FP8 MoE fast path requires expert_topk_weights")

        hidden_states_fp8 = payload.expert_x
        hidden_states_scale = payload.expert_x_scale
        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        token_num, hidden_size = hidden_states_fp8.shape
        topk = topk_ids.shape[1]
        expert_num, gate_up_size, _ = self.w13_weight.shape
        intermediate_size = gate_up_size // 2

        config1 = get_sm120_triton_fp8_config(
            token_num, expert_num, gate_up_size, hidden_size, topk
        )
        config2 = get_sm120_triton_fp8_config(
            token_num, expert_num, hidden_size, intermediate_size, topk
        )
        block_m = min(config1["BLOCK_SIZE_M"], config2["BLOCK_SIZE_M"])
        config1["BLOCK_SIZE_M"] = block_m
        config2["BLOCK_SIZE_M"] = block_m
        # The execute() gate restricts this path to EP=1, where SelectTopk
        # produces global expert ids in [0, expert_num).  EP padding sentinels
        # must stay on the contiguous path, which owns their remapping logic.
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size_compiled(topk_ids, block_m, expert_num)
        )

        route_num = token_num * topk
        gate_up_output = torch.empty(
            (route_num, gate_up_size),
            device=hidden_states_fp8.device,
            dtype=torch.bfloat16,
        )
        invoke_triton_moe_kernel(
            hidden_states_fp8,
            self.w13_weight,
            gate_up_output,
            topk_weights.reshape(-1),
            topk_ids.reshape(-1),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            topk,
            config1,
            tl.bfloat16,
            A_scale=hidden_states_scale,
            B_scale=self.w13_weight_scale_inv,
            block_shape=self.DEEPGEMM_BLOCK_SHAPE,
            scale_ue8m0=True,
        )

        down_input = torch.empty(
            (route_num, intermediate_size),
            device=hidden_states_fp8.device,
            dtype=torch.bfloat16,
        )
        silu_and_mul(down_input, gate_up_output)
        down_input_fp8, down_input_scale = sgl_per_token_group_quant_fp8(
            down_input,
            group_size=self.BLOCK_SIZE,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        down_output = torch.empty(
            (route_num, hidden_size),
            device=hidden_states_fp8.device,
            dtype=torch.bfloat16,
        )
        invoke_triton_moe_kernel(
            down_input_fp8,
            self.w2_weight,
            down_output,
            topk_weights.reshape(-1),
            topk_ids.reshape(-1),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            1,
            config2,
            tl.bfloat16,
            A_scale=down_input_scale,
            B_scale=self.w2_weight_scale_inv,
            block_shape=self.DEEPGEMM_BLOCK_SHAPE,
            scale_ue8m0=True,
        )
        output = down_output.view(token_num, topk, hidden_size).sum(dim=1)
        return CombineForwardPayload(fused_expert_output=output)

    def execute_masked(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "hidden_states_fp8 is not initialized"
        assert (
            payload.expert_x_scale is not None
        ), "hidden_states_scale is not initialized"
        assert payload.expert_topk_ids is not None, "expert_topk_ids is not initialized"
        assert (
            payload.expert_topk_weights is not None
        ), "expert_topk_weights is not initialized"
        assert (
            payload.expert_tokens_meta is not None
        ), "expert_tokens_meta is not initialized"

        hidden_states_fp8 = payload.expert_x
        hidden_states_scale = payload.expert_x_scale
        topk_idx = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        global_context = getattr(payload, "router_context", None)
        if isinstance(global_context, GlobalExpertIdsContext):
            phase = h20_compact_moe_phase.get()
            if _is_cuda_graph_warmup_or_capture():
                use_compact = self._qwen4_sm90_mixed_graph and phase in (
                    "decode",
                    "target_verify",
                )
            else:
                use_compact = not self.enable_cuda_graph or phase == "eager"
            rows = hidden_states_fp8.shape[0] if hidden_states_fp8.ndim == 2 else 0
            max_rows = min(
                self.masked_max_token_num, _STABLE_ALIGNMENT_MAX_ROUTES // self.top_k
            )
            if not (
                self._h20_compact_fp8
                and use_compact
                and 0 < rows <= max_rows
                and hidden_states_fp8.shape[1] == self.K
                and hidden_states_fp8.is_contiguous()
                and hidden_states_fp8.dtype == torch.float8_e4m3fn
                and hidden_states_scale.dtype == torch.float32
                and hidden_states_scale.shape == (rows, self.K // self.BLOCK_SIZE)
                and topk_idx.dtype in (torch.int32, torch.int64)
                and topk_idx.shape == (rows, self.top_k)
                and topk_weights.dtype == torch.float32
                and topk_weights.shape == topk_idx.shape
                and global_context.num_local == self.E
                and global_context.start == self.start_expert_id
                and activation == "SiGLU"
                and expert_map is None
                and a2_scale is None
                and not apply_router_weight_on_input
                and not extra_expert_args
            ):
                raise RuntimeError(
                    "Global expert ID payload cannot fall through to a legacy consumer"
                )
            output = compact_fp8_decode(
                hidden_states_fp8,
                hidden_states_scale,
                topk_idx,
                topk_weights,
                None,
                self.w13_weight_fp8,
                self.w2_weight_fp8,
                global_start=global_context.start,
            )
            if not self.enable_cuda_graph:
                dispose_tensor(hidden_states_fp8)
            return CombineForwardPayload(fused_expert_output=output)

        assert payload.expert_tokens_meta.expert_num_tokens is not None
        num_recv_tokens_per_expert = payload.expert_tokens_meta.expert_num_tokens

        if self._h20_compact_fp8:
            phase = h20_compact_moe_phase.get()
            if _is_cuda_graph_warmup_or_capture():
                use_compact = self._qwen4_sm90_mixed_graph and phase in (
                    "decode",
                    "target_verify",
                )
            else:
                use_compact = payload.expert_ids_are_local and (
                    not self.enable_cuda_graph or phase == "eager"
                )
            # Verify can pack several rows per request. Routed offsets are int32.
            max_rows = 2048
            if phase == "target_verify":
                max_rows = (2**31 - 1) // (self.top_k * max(self.K, self.N))
            if use_compact and hidden_states_fp8.ndim == 2:
                rows = hidden_states_fp8.shape[0]
            else:
                rows = 0
            if (
                0 < rows <= max_rows
                and hidden_states_fp8.shape[1] == self.K
                and hidden_states_fp8.is_contiguous()
                and hidden_states_fp8.dtype == torch.float8_e4m3fn
                and hidden_states_scale.dtype == torch.float32
                and hidden_states_scale.shape == (rows, self.K // self.BLOCK_SIZE)
                and topk_idx.dtype in (torch.int32, torch.int64)
                and topk_idx.shape == (rows, self.top_k)
                and topk_weights.dtype == torch.float32
                and topk_weights.shape == topk_idx.shape
                and num_recv_tokens_per_expert.shape == (self.E,)
                and num_recv_tokens_per_expert.device == hidden_states_fp8.device
                and num_recv_tokens_per_expert.dtype == torch.int32
                and num_recv_tokens_per_expert.is_contiguous()
                and activation == "SiGLU"
                and expert_map is None
                and a2_scale is None
                and not apply_router_weight_on_input
                and not extra_expert_args
            ):
                output = compact_fp8_decode(
                    hidden_states_fp8,
                    hidden_states_scale,
                    topk_idx,
                    topk_weights,
                    num_recv_tokens_per_expert,
                    self.w13_weight_fp8,
                    self.w2_weight_fp8,
                )
                if not self.enable_cuda_graph:
                    dispose_tensor(hidden_states_fp8)
                return CombineForwardPayload(fused_expert_output=output)

        with configure_deep_gemm_num_sms(self.num_gemm_sms):
            token_num = hidden_states_fp8.shape[0]
            mn_major_scales = self._sm90_mn_major_scales and token_num in (16, 32, 64)
            num_experts = num_recv_tokens_per_expert.shape[0]
            max_token_num = token_num * self.top_k
            token_num_mean_per_expert = ceil_div(max_token_num, num_experts)
            alignment = align(token_num, self.EXPERT_ALIGNMENT)
            expected_m = min(alignment, token_num_mean_per_expert)

            _, K = hidden_states_fp8.size()
            assert K == self.K, f"K: {K} != self.K: {self.K}"
            hidden_states_fp8_shape = hidden_states_fp8.shape
            hidden_states_fp8_device = hidden_states_fp8.device
            input_tensor = [
                torch.empty(
                    (self.num_experts_per_partition, alignment, self.K),
                    device=hidden_states_fp8_device,
                    dtype=hidden_states_fp8.dtype,
                ),
                (
                    torch.zeros(
                        [
                            self.num_experts_per_partition,
                            ceil_div(self.K // self.BLOCK_SIZE, 4),
                            alignment,
                        ],
                        device=hidden_states_fp8_device,
                        dtype=torch.int,
                    ).transpose(1, 2)
                    if is_deep_gemm_e8m0_used()
                    else _empty_fp32_moe_scale(
                        (
                            self.num_experts_per_partition,
                            alignment,
                            self.K // self.BLOCK_SIZE,
                        ),
                        device=hidden_states_fp8_device,
                        mn_major=mn_major_scales,
                    )
                ),
            ]
            output_index = torch.empty_like(topk_idx)
            expert_start_loc = torch.empty_like(num_recv_tokens_per_expert)
            ep_scatter_v2(
                hidden_states_fp8,
                hidden_states_scale,
                topk_idx,
                alignment,
                expert_start_loc,
                input_tensor[0].view(
                    self.num_experts_per_partition * alignment, self.K
                ),
                input_tensor[1],
                output_index,
                scale_ue8m0=is_deep_gemm_e8m0_used(),
            )
            # Eager execution consumes the routed input as before; graph
            # warmup/capture must retain the same storage for later replays.
            if not self.enable_cuda_graph:
                dispose_tensor(hidden_states_fp8)
            upgate_output = torch.empty(
                (self.num_experts_per_partition, alignment, self.N),
                device=hidden_states_fp8_device,
                dtype=torch.bfloat16,
            )
            # Gate and Up GroupGEMM-0
            m_grouped_fp8_gemm_nt_masked(
                (input_tensor[0], input_tensor[1]),
                self.w13_weight_fp8,
                upgate_output,
                num_recv_tokens_per_expert,
                expected_m,
                disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
            )

            del input_tensor
            # Allocate down_input
            down_input = torch.empty(
                (self.num_experts_per_partition, alignment, self.N // 2),
                device=hidden_states_fp8_device,
                dtype=torch.float8_e4m3fn,
            )

            # SM100 (compute capability 10.x) uses fused packed kernel for better performance
            # when UE8M0 scale format is enabled
            sm_major = torch.cuda.get_device_capability()[0]
            if (
                sm_major == 10
                and is_deep_gemm_e8m0_used()
                and self.N % (self.DEEPGEMM_BLOCK_SHAPE[0] * 2 * 4) == 0
            ):
                # Create packed scale tensor with proper layout for deep_gemm
                # Shape: (E, T, G // 4) where G = hidden_dim // 2 // group_size
                down_input_scale = create_packed_scale_tensor(
                    expert_num=self.num_experts_per_partition,
                    token_num_padded=alignment,
                    hidden_dim=self.N,
                    quant_group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                    device=hidden_states_fp8_device,
                )
                # Fused SiLU-and-mul + FP8 quantization with UE8M0 scale packing
                silu_and_mul_masked_post_quant_packed_fwd(
                    upgate_output,
                    down_input,
                    down_input_scale,
                    self.DEEPGEMM_BLOCK_SHAPE[0],
                    num_recv_tokens_per_expert,
                )
            else:
                # Standard path for other SM versions
                down_input_scale = _empty_fp32_moe_scale(
                    (
                        self.num_experts_per_partition,
                        alignment,
                        self.N // 2 // self.DEEPGEMM_BLOCK_SHAPE[0],
                    ),
                    device=hidden_states_fp8_device,
                    mn_major=mn_major_scales,
                )
                silu_mul_masked_fp8_post_quant_fwd(
                    input=upgate_output,
                    output=down_input,
                    output_scale=down_input_scale,
                    quant_group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                    masked_m=num_recv_tokens_per_expert,
                    expected_m=expected_m,
                    input_layout=MaskedSiluInputLayout.BATCH_TOKEN_ALIGNMENT,
                    scale_ue8m0=is_deep_gemm_e8m0_used(),
                )

            # Free upgate_output
            dispose_tensor(upgate_output)
            down_output = torch.empty(
                (self.num_experts_per_partition, alignment, self.K),
                device=hidden_states_fp8_device,
                dtype=torch.bfloat16,
            )

            # Down GroupGEMM-1
            m_grouped_fp8_gemm_nt_masked(
                (
                    down_input,
                    down_input_scale,
                ),
                self.w2_weight_fp8,
                down_output,
                num_recv_tokens_per_expert,
                expected_m,
                disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
            )

            # Free down_input and down_input_scale
            dispose_tensor(down_input)
            dispose_tensor(down_input_scale)

            gather_out = torch.empty(
                hidden_states_fp8_shape,
                device=hidden_states_fp8_device,
                dtype=torch.bfloat16,
            )
            ep_gather(
                down_output.view(self.num_experts_per_partition * alignment, self.K),
                topk_idx,
                topk_weights,
                output_index,
                gather_out,
            )
            return CombineForwardPayload(fused_expert_output=gather_out)

    def _read_expert_counts_host(
        self, num_recv_tokens_per_expert: torch.Tensor
    ) -> list:
        """Blocking D2H of the routed expert counts over a cached pinned buffer.

        The eager path reads the counts once per layer.  A plain ``.cpu()``
        pays a pageable staging copy on top of the unavoidable stream sync;
        one non-blocking copy into a persistent pinned buffer keeps the
        transfer itself async, and the sync below waits for the producing
        router/recompute kernels queued on the current stream (any earlier
        H2D from the padded-counts buffer has also drained by then, so
        overwriting that buffer next layer stays ordered).
        """
        if (
            self._expert_counts_pinned is None
            or self._expert_counts_pinned.shape != num_recv_tokens_per_expert.shape
            or self._expert_counts_pinned.dtype != num_recv_tokens_per_expert.dtype
        ):
            self._expert_counts_pinned = torch.empty(
                num_recv_tokens_per_expert.shape,
                dtype=num_recv_tokens_per_expert.dtype,
                pin_memory=True,
                device="cpu",
            )
        self._expert_counts_pinned.copy_(num_recv_tokens_per_expert, non_blocking=True)
        torch.cuda.current_stream(num_recv_tokens_per_expert.device).synchronize()
        return self._expert_counts_pinned.tolist()

    def execute_contiguous(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
        *,
        use_eager_layout: bool = False,
    ) -> CombineForwardPayload:
        # An engine with decode graphs can still run an ordinary eager
        # prefill. Use compact host-padded metadata without mutating its mode.
        if use_eager_layout and _is_cuda_graph_warmup_or_capture():
            raise ValueError(
                "eager grouped layout is not valid during graph capture/warmup"
            )
        graph_layout = self.enable_cuda_graph and not use_eager_layout
        assert payload.expert_x is not None, "hidden_states_fp8 is not initialized"
        assert (
            payload.expert_x_scale is not None
        ), "hidden_states_scale is not initialized"
        assert payload.expert_topk_ids is not None, "expert_topk_ids is not initialized"
        assert (
            payload.expert_topk_weights is not None
        ), "expert_topk_weights is not initialized"
        assert (
            payload.expert_tokens_meta is not None
        ), "expert_tokens_meta is not initialized"
        hidden_states_fp8 = payload.expert_x
        hidden_states_scale = payload.expert_x_scale
        topk_idx = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        num_recv_tokens_per_expert = payload.expert_tokens_meta.expert_num_tokens
        if num_recv_tokens_per_expert is None:
            num_recv_tokens_per_expert_cpu = (
                payload.expert_tokens_meta.expert_num_tokens_cpu
            )
            if graph_layout:
                raise ValueError(
                    "expert_num_tokens GPU tensor is required for CUDA graph"
                )
            if num_recv_tokens_per_expert_cpu is None:
                raise ValueError(
                    "expert_num_tokens or expert_num_tokens_cpu is required"
                )
            num_recv_tokens_per_expert = torch.as_tensor(
                num_recv_tokens_per_expert_cpu,
                dtype=torch.int32,
                device=hidden_states_fp8.device,
            )

        num_experts_local = num_recv_tokens_per_expert.shape[0]
        token_num = hidden_states_fp8.shape[0]
        use_cuda_silu = (
            self._sm90_cuda_contiguous_silu
            and self._sm90_mn_major_scales
            and self._sm90_prefill_silu_fp8
            and not graph_layout
        )
        mn_major_scales = self._sm90_mn_major_scales and (
            use_cuda_silu or token_num in (8192, 32768)
        )
        routed_tokens = token_num * topk_idx.shape[1]
        if routed_tokens == 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(
                    hidden_states_fp8.shape,
                    device=hidden_states_fp8.device,
                    dtype=torch.bfloat16,
                )
            )
        if self.is_sm120:
            expert_alignment = min(
                self.EXPERT_ALIGNMENT,
                get_theoretical_mk_alignment_for_contiguous_layout(
                    routed_tokens, num_experts_local
                ),
            )
            max_active_experts = min(routed_tokens, num_experts_local)
            all_tokens = align_up_math(
                routed_tokens + max_active_experts * (expert_alignment - 1),
                expert_alignment,
            )
        else:
            # Preserve the pre-upgrade SM9x/SM100x layout: fixed 128-token
            # expert alignment and CPU-padded metadata.
            expert_alignment = self.EXPERT_ALIGNMENT
            max_active_experts = min(routed_tokens, num_experts_local)
            all_tokens = align_up_math(
                routed_tokens + max_active_experts * (expert_alignment - 1),
                expert_alignment,
            )

        async_device_counts = (
            not graph_layout
            and use_cuda_silu
            and payload.expert_tokens_meta.expert_num_tokens_cpu is None
            and token_num <= 2048
        )
        if not graph_layout and not async_device_counts:
            num_recv_tokens_per_expert_cpu = (
                payload.expert_tokens_meta.expert_num_tokens_cpu
            )
            if num_recv_tokens_per_expert_cpu is None:
                num_recv_tokens_per_expert_cpu = self._read_expert_counts_host(
                    num_recv_tokens_per_expert
                )
            elif isinstance(num_recv_tokens_per_expert_cpu, torch.Tensor):
                num_recv_tokens_per_expert_cpu = num_recv_tokens_per_expert_cpu.tolist()
            padded_num_tokens_per_expert = [
                align_up_math(int(x), expert_alignment)
                for x in num_recv_tokens_per_expert_cpu
            ]
            all_tokens = sum(padded_num_tokens_per_expert)
            if all_tokens == 0:
                return CombineForwardPayload(
                    fused_expert_output=torch.zeros(
                        hidden_states_fp8.shape,
                        device=hidden_states_fp8.device,
                        dtype=torch.bfloat16,
                    )
                )
        _, K = hidden_states_fp8.size()
        N = self.w13_weight.size(1)
        hidden_states_fp8_shape = hidden_states_fp8.shape
        hidden_states_fp8_device = hidden_states_fp8.device
        ws_views = _acquire_contiguous_moe_workspace(
            hidden_states_fp8_device,
            all_tokens,
            K,
            N,
            hidden_states_fp8.dtype,
            graph_layout=graph_layout,
        )
        input_tensor = [
            (
                ws_views[0]
                if ws_views is not None
                else torch.empty(
                    (all_tokens, K),
                    device=hidden_states_fp8.device,
                    dtype=hidden_states_fp8.dtype,
                )
            ),
            (
                torch.zeros(
                    [ceil_div(K // self.BLOCK_SIZE, 4), all_tokens],
                    device=hidden_states_fp8.device,
                    dtype=torch.int,
                ).transpose(0, 1)
                if is_deep_gemm_e8m0_used()
                else _empty_fp32_moe_scale(
                    (all_tokens, K // self.BLOCK_SIZE),
                    device=hidden_states_fp8.device,
                    mn_major=mn_major_scales,
                )
            ),
        ]
        # DeepGEMM's SM120 grouped kernel skips tiles whose group ID is -1;
        # keep unused graph workspace tiles inactive instead of routing to 0.
        # This relies on the pinned SM120 wheel (8bcfcab); non-SM120 scatter
        # below fills all padded rows with valid expert IDs for older wheels.
        m_indices = (
            torch.zeros(
                (all_tokens,), device=hidden_states_fp8.device, dtype=torch.int32
            )
            if async_device_counts
            else torch.full(
                (all_tokens,), -1, device=hidden_states_fp8.device, dtype=torch.int32
            )
        )
        output_index = torch.full_like(topk_idx, -1)
        scatter_num_tokens_per_expert = num_recv_tokens_per_expert
        scatter_alignment = expert_alignment
        if not self.is_sm120 and not graph_layout and not async_device_counts:
            # Reuse one persistent pinned buffer instead of a per-layer
            # cudaHostAlloc; the counts read above synchronized the stream,
            # so the previous layer's H2D from this buffer has drained
            # before the host rewrites it.
            if (
                self._padded_counts_pinned is None
                or self._padded_counts_pinned.numel()
                != len(padded_num_tokens_per_expert)
            ):
                self._padded_counts_pinned = torch.empty(
                    len(padded_num_tokens_per_expert),
                    dtype=torch.int32,
                    pin_memory=True,
                    device="cpu",
                )
            self._padded_counts_pinned.copy_(
                torch.tensor(
                    padded_num_tokens_per_expert,
                    dtype=torch.int32,
                ),
                non_blocking=False,
            )
            scatter_num_tokens_per_expert = self._padded_counts_pinned.cuda(
                non_blocking=True
            )
            scatter_alignment = 1
        # DeepGEMM on SM90 skips WGMMA when a 64-row group's first index is
        # negative. The activation needs the expert owner of padded rows.
        gemm_m_indices = (
            torch.full(
                (all_tokens,), -1, device=hidden_states_fp8.device, dtype=torch.int32
            )
            if async_device_counts
            else (torch.empty_like(m_indices) if use_cuda_silu else m_indices)
        )
        expert_start_loc = torch.empty_like(scatter_num_tokens_per_expert)
        ep_scatter(
            hidden_states_fp8,
            hidden_states_scale,
            topk_idx,
            scatter_num_tokens_per_expert,
            expert_start_loc,
            input_tensor[0],
            input_tensor[1],
            m_indices,
            output_index,
            scale_ue8m0=is_deep_gemm_e8m0_used(),
            align_m=scatter_alignment,
            derive_counts_from_topk=self.is_sm120 and graph_layout,
            gemm_padding_indices=(
                (gemm_m_indices, num_recv_tokens_per_expert)
                if use_cuda_silu else None
            ),
        )
        # Preserve the eager input lifetime without invalidating graph buffers.
        if not graph_layout:
            dispose_tensor(hidden_states_fp8)
        gateup_output = (
            ws_views[1]
            if ws_views is not None
            else torch.empty(
                (all_tokens, N),
                device=hidden_states_fp8_device,
                dtype=torch.bfloat16,
            )
        )
        if not is_deep_gemm_e8m0_used() and not mn_major_scales:
            input_tensor[1] = tma_align_input_scale(input_tensor[1])
        if self.is_sm120:
            configure_deep_gemm_mk_alignment_context = configure_deep_gemm_mk_alignment(
                expert_alignment
            )
        else:
            configure_deep_gemm_mk_alignment_context = nullcontext()
        with configure_deep_gemm_mk_alignment_context:
            m_grouped_fp8_gemm_nt_contiguous(
                (input_tensor[0], input_tensor[1]),
                self.w13_weight_fp8,
                gateup_output,
                gemm_m_indices,
                disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
            )
        del input_tensor
        if mn_major_scales and self._sm90_prefill_silu_fp8:
            if use_cuda_silu:
                if async_device_counts:
                    start, end = 0, all_tokens
                else:
                    counts = num_recv_tokens_per_expert_cpu
                    padded = padded_num_tokens_per_expert
                    start, end = 0, all_tokens
                    first = 0
                    while first < len(padded) and counts[first] == padded[first]:
                        start += padded[first]
                        first += 1
                    if first == len(padded):
                        start = end = 0
                    else:
                        last = len(padded) - 1
                        while counts[last] == padded[last]:
                            end -= padded[last]
                            last -= 1
                        start += counts[first]
                down_input_fp8, down_input_scale = _native_contiguous_silu_fp8(
                    gateup_output,
                    m_indices if start < end else None,
                    expert_start_loc if start < end else None,
                    start,
                    end,
                )
            else:
                down_input_fp8, down_input_scale = silu_mul_native_fp8(gateup_output)
            del gateup_output
            down_output = (
                ws_views[2]
                if ws_views is not None
                else torch.empty(
                    (all_tokens, K),
                    device=hidden_states_fp8_device,
                    dtype=torch.bfloat16,
                )
            )
            del ws_views
        else:
            down_input = torch.empty(
                (
                    all_tokens,
                    N // 2,
                ),
                device=gateup_output.device,
                dtype=torch.bfloat16,
            )
            gateup_output = gateup_output.view(-1, N)
            silu_and_mul(down_input, gateup_output)
            del gateup_output
            down_output = (
                ws_views[2]
                if ws_views is not None
                else torch.empty(
                    (all_tokens, K),
                    device=hidden_states_fp8_device,
                    dtype=torch.bfloat16,
                )
            )
            del ws_views
            if is_deep_gemm_e8m0_used():
                down_input_fp8, down_input_scale = sgl_per_token_group_quant_fp8(
                    down_input,
                    group_size=self.BLOCK_SIZE,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=is_deep_gemm_e8m0_used(),
                )
            else:
                down_input_fp8, down_input_scale = trt_fp8_quantize_128(
                    down_input, mn_major_scales
                )
                if mn_major_scales:
                    # Native col_major_scale returns [groups, padded_rows].
                    down_input_scale = down_input_scale.t()
            del down_input
        if not is_deep_gemm_e8m0_used() and not mn_major_scales:
            down_input_scale = tma_align_input_scale(down_input_scale)
        if self.is_sm120:
            configure_deep_gemm_mk_alignment_context = configure_deep_gemm_mk_alignment(
                expert_alignment
            )
        else:
            configure_deep_gemm_mk_alignment_context = nullcontext()
        with configure_deep_gemm_mk_alignment_context:
            m_grouped_fp8_gemm_nt_contiguous(
                (down_input_fp8, down_input_scale),
                self.w2_weight_fp8,
                down_output,
                gemm_m_indices,
                disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
            )
        del down_input_fp8, down_input_scale
        gather_out = torch.empty(
            hidden_states_fp8_shape,
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)
        return CombineForwardPayload(fused_expert_output=gather_out)
