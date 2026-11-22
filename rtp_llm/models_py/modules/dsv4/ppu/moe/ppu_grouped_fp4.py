"""M890P grouped-MXFP4 routed experts for DSV4.

This strategy is deliberately separate from the CUDA ``grouped_fp4`` backend.
It consumes the checkpoint's packed MXFP4 tensors, executes two PPU nopad
grouped GEMMs, and returns the normalized routed partial expected by ``MoE``'s
post-W2 route-scale/TP-reduce contract.

There is no fallback in this module.  A wrong topology, storage geometry,
device, or external symbol fails closed.
"""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from typing import Dict, Tuple

import torch
from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_moe_config import PpuMoeConfig as MoeCfg

logger = logging.getLogger(__name__)

_ROUTE_WEIGHT_CONTRACT = "post_w2_normalized_then_scale_v1"
_GROUPED_GEMM_SYMBOL = "deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad"
_ACTIVATION_MODE = "eager_swiglu_bf16_then_mxfp4"
_EXACT_GATHER_SYMBOL = (
    "rtp_llm.models_py.kernels.ppu.moe.ppu_moe_exact_gather."
    "gather_local_loop_compatible"
)


@lru_cache(maxsize=1)
def _log_operator_path_once(grouped_module: str) -> None:
    """Emit one process-local marker for the exact callables bound by setup."""

    logger.info(
        "DSV4_PPU_GROUPED_OPERATOR_PATH strategy=ppu_grouped_fp4 "
        "operator=%s operator_module=%s activation_mode=%s exact_gather=%s",
        _GROUPED_GEMM_SYMBOL,
        grouped_module,
        (
            "sglang_fused_swiglu_mxfp4"
            if os.environ.get("DSV4_PPU_SGLANG_MOE", "0") == "1"
            else _ACTIVATION_MODE
        ),
        _EXACT_GATHER_SYMBOL,
    )


def _module_path(module: object) -> str:
    """Return a stable module identity without inspecting device state."""

    return str(
        getattr(module, "__file__", None)
        or getattr(module, "__name__", None)
        or type(module).__module__
    )


def _supports_topology(cfg: MoeCfg) -> bool:
    """Weight geometry determines compatibility after validating group sizes."""

    return int(cfg.tp_size) > 0 and int(cfg.ep_size) > 0


def _runtime_eligible() -> bool:
    """Probe the PPU-only leaf without making generic GPU TP4 select it."""

    if not torch.cuda.is_available():
        return False
    try:
        import deep_gemm
    except (ImportError, RuntimeError):
        return False
    return callable(getattr(deep_gemm, "m_grouped_gemm_fp4_fp4_bf16_nt_nopad", None))


def _derive_inter_local_and_tp(
    cfg: MoeCfg,
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    w3_shape: Tuple[int, ...],
    s1_shape: Tuple[int, ...],
    s2_shape: Tuple[int, ...],
    s3_shape: Tuple[int, ...],
) -> Tuple[int, int]:
    """Validate packed MXFP4 geometry and derive the routed TP contract."""

    if not _supports_topology(cfg):
        raise RuntimeError("ppu_grouped_fp4 requires positive TP and EP sizes")

    experts = int(cfg.n_local_experts)
    dim = int(cfg.dim)
    if experts != int(cfg.n_routed_experts):
        raise ValueError(
            "Local grouped MXFP4 must bind every routed expert on each TP rank"
        )
    if len(w1_shape) != 3:
        raise ValueError(f"packed MXFP4 w1 must be rank 3, got {w1_shape}")
    inter_local = int(w1_shape[1])
    if dim <= 0 or dim % 512:
        raise ValueError(
            "PPU grouped MXFP4 requires positive dim aligned to 512 for "
            f"MXFP4 scale blocks and ep_gather, got dim={dim}"
        )
    if inter_local <= 0 or inter_local % 64:
        raise ValueError(
            "PPU grouped MXFP4 requires positive inter_local aligned to 64, "
            f"got inter_local={inter_local}"
        )

    expected_w1 = (experts, inter_local, dim // 2)
    expected_w2 = (experts, dim, inter_local // 2)
    expected_s1 = (experts, inter_local, dim // 32)
    expected_s2 = (experts, dim, inter_local // 32)
    if w1_shape != expected_w1 or w3_shape != expected_w1:
        raise ValueError(
            "packed MXFP4 w1/w3 geometry mismatch: "
            f"got {w1_shape}/{w3_shape}, expected {expected_w1}"
        )
    if w2_shape != expected_w2:
        raise ValueError(
            f"packed MXFP4 w2 geometry mismatch: got {w2_shape}, expected {expected_w2}"
        )
    if s1_shape != expected_s1 or s3_shape != expected_s1:
        raise ValueError(
            "MXFP4 w1/w3 scale geometry mismatch: "
            f"got {s1_shape}/{s3_shape}, expected {expected_s1}"
        )
    if s2_shape != expected_s2:
        raise ValueError(
            f"MXFP4 w2 scale geometry mismatch: got {s2_shape}, expected {expected_s2}"
        )

    full_inter = int(cfg.moe_inter_dim)
    if inter_local == full_inter:
        routed_tp_size = 1
    elif inter_local * int(cfg.tp_size) == full_inter:
        routed_tp_size = int(cfg.tp_size)
    else:
        raise ValueError(
            "routed intermediate is neither full nor a pure TP preshard: "
            f"inter_local={inter_local}, moe_inter_dim={full_inter}, "
            f"tp_size={cfg.tp_size}"
        )
    return inter_local, routed_tp_size


class PpuGroupedFP4Strategy(torch.nn.Module):
    """Strict M890P packed-MXFP4 strategy for DSV4."""

    name = "ppu_grouped_fp4"
    route_weight_contract = _ROUTE_WEIGHT_CONTRACT
    routed_tp_size = 1

    def __init__(
        self,
        cfg: MoeCfg,
        *,
        sglang_moe=None,
        fused_gather=None,
        fused_scale_gather=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.sglang_moe = (
            (os.environ.get("DSV4_PPU_SGLANG_MOE", "0") == "1")
            if sglang_moe is None
            else bool(sglang_moe)
        )
        self.inter_local = 0
        self._grouped_gemm = None
        self._fused_gather = fused_gather
        self._fused_scale_gather = fused_scale_gather

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return _supports_topology(cfg) and _runtime_eligible()

    def setup_weights(self, layer_weights: Dict) -> None:
        """Bind packed routed tensors and prepare their PPU scale layout once."""

        if not _supports_topology(self.cfg):
            raise RuntimeError(
                "ppu_grouped_fp4 setup requires positive TP and EP sizes"
            )

        from rtp_llm.utils.model_weight import W

        keys = (
            W.v4_routed_w1_w,
            W.v4_routed_w1_s,
            W.v4_routed_w2_w,
            W.v4_routed_w2_s,
            W.v4_routed_w3_w,
            W.v4_routed_w3_s,
        )
        try:
            w1, s1, w2, s2, w3, s3 = (layer_weights[key] for key in keys)
        except KeyError as error:
            raise KeyError(
                f"ppu_grouped_fp4 requires all six packed routed tensors; missing {error}"
            ) from error

        inter_local, routed_tp_size = _derive_inter_local_and_tp(
            self.cfg,
            tuple(w1.shape),
            tuple(w2.shape),
            tuple(w3.shape),
            tuple(s1.shape),
            tuple(s2.shape),
            tuple(s3.shape),
        )
        if any(
            weight.dtype not in (torch.int8, torch.uint8) for weight in (w1, w2, w3)
        ):
            raise TypeError("ppu_grouped_fp4 requires packed int8/uint8 MXFP4 weights")
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if e8m0_dtype is None or any(
            scale.dtype != e8m0_dtype for scale in (s1, s2, s3)
        ):
            raise TypeError("ppu_grouped_fp4 requires float8_e8m0fnu checkpoint scales")
        tensors = (w1, s1, w2, s2, w3, s3)
        if any(not tensor.is_cuda for tensor in tensors):
            raise ValueError("ppu_grouped_fp4 requires CUDA-compatible PPU tensors")
        if any(tensor.device != w1.device for tensor in tensors):
            raise ValueError("ppu_grouped_fp4 weights/scales must share one device")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("ppu_grouped_fp4 weights/scales must be contiguous")

        import deep_gemm

        grouped_gemm = getattr(deep_gemm, "m_grouped_gemm_fp4_fp4_bf16_nt_nopad", None)
        if not callable(grouped_gemm):
            raise RuntimeError(
                "PPU deep_gemm lacks m_grouped_gemm_fp4_fp4_bf16_nt_nopad"
            )
        from rtp_llm.models_py.kernels.ppu.moe.ppu_mxfp4 import (
            prepare_fp4_weight_scale_mxfp4,
        )

        w13 = torch.cat((w1, w3), dim=1).view(torch.uint8).contiguous()
        s13 = prepare_fp4_weight_scale_mxfp4(torch.cat((s1, s3), dim=1).contiguous())
        w2_packed = w2.view(torch.uint8).contiguous()
        s2_prepared = prepare_fp4_weight_scale_mxfp4(s2)

        self.register_buffer("_ppu_w13", w13, persistent=False)
        self.register_buffer("_ppu_s13", s13, persistent=False)
        self.register_buffer("_ppu_w2", w2_packed, persistent=False)
        self.register_buffer("_ppu_s2", s2_prepared, persistent=False)
        self.inter_local = inter_local
        self.routed_tp_size = routed_tp_size
        self._grouped_gemm = grouped_gemm
        for key in keys:
            layer_weights.pop(key)
        _log_operator_path_once(_module_path(deep_gemm))

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        *,
        shared: torch.Tensor | None = None,
        route_scale: float | None = None,
        shared_schedule=None,
    ) -> torch.Tensor:
        """Route EP1 tokens, run two compact nopad PPU grouped GEMMs, and gather."""

        if self.inter_local <= 0:
            raise RuntimeError("ppu_grouped_fp4 weights were not bound")
        if self._grouped_gemm is None:
            raise RuntimeError("ppu_grouped_fp4 grouped GEMM was not bound by setup")
        if x.ndim != 2 or x.dtype != torch.bfloat16:
            raise TypeError(
                f"x must be BF16 [N,D], got dtype={x.dtype}, shape={x.shape}"
            )
        if weights.ndim != 2 or weights.dtype != torch.float32:
            raise TypeError(
                "route weights must be FP32 [N,K], "
                f"got dtype={weights.dtype}, shape={weights.shape}"
            )
        if indices.ndim != 2 or indices.dtype != torch.int64:
            raise TypeError(
                "expert indices must be int64 [N,K], "
                f"got dtype={indices.dtype}, shape={indices.shape}"
            )
        if x.size(0) != weights.size(0) or weights.shape != indices.shape:
            raise ValueError(
                f"incompatible routed shapes x={x.shape}, weights={weights.shape}, "
                f"indices={indices.shape}"
            )
        if x.size(1) != int(self.cfg.dim):
            raise ValueError(f"x hidden dim must be {self.cfg.dim}, got {x.size(1)}")
        if any(tensor.device != x.device for tensor in (weights, indices)):
            raise ValueError(
                "x, route weights, and expert indices must share one device"
            )

        if (shared is None) != (route_scale is None):
            raise ValueError("shared and route_scale must be provided together")
        if shared is not None:
            if (
                shared.dtype != torch.bfloat16
                or shared.shape != x.shape
                or not shared.is_contiguous()
                or shared.device != x.device
            ):
                raise TypeError(
                    "shared must be contiguous BF16 with the same shape/device as x"
                )
            if not math.isfinite(float(route_scale)):
                raise ValueError("route_scale must be finite")

        token_count, dim = x.shape
        if token_count == 0:
            dtype = torch.bfloat16 if shared is not None else torch.float32
            return torch.zeros((0, dim), dtype=dtype, device=x.device)

        from rtp_llm.models_py.kernels.ppu.moe.ppu_moe_exact_gather import (
            gather_and_combine_local_loop_compatible,
            gather_local_loop_compatible,
        )
        from rtp_llm.models_py.kernels.ppu.moe.ppu_moe_nopad import (
            compact_mxfp4_routes_nopad,
        )
        from rtp_llm.models_py.kernels.ppu.moe.ppu_mxfp4 import downcast_to_mxfp4
        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.expert import (
            require_silu_mul_split,
        )

        experts = int(self.cfg.n_local_experts)
        adjusted_ids = indices.contiguous()
        x_fp4, x_scale = downcast_to_mxfp4(x.contiguous())
        compact_fp4, compact_scale, expert_ids, output_index, expert_counts = (
            compact_mxfp4_routes_nopad(
                x_fp4,
                x_scale,
                adjusted_ids,
                experts,
                fused_scale_gather=self._fused_scale_gather,
            )
        )

        total = token_count * indices.size(1)
        gate_up = torch.empty(
            (total, 2 * self.inter_local),
            dtype=torch.bfloat16,
            device=x.device,
        )
        if shared_schedule is not None:
            shared_schedule.before_up()
        self._grouped_gemm(
            (compact_fp4, compact_scale),
            (self._ppu_w13, self._ppu_s13),
            None,
            gate_up,
            expert_ids,
            expert_counts,
        )

        if shared_schedule is not None:
            shared_schedule.after_up()

        if self.sglang_moe:
            from rtp_llm.ops.compute_ops import rtp_llm_ops

            # The SG clamped branch quantizes its FP32 product directly.
            # Zero in RTP's config means no clamp; SG expresses that as None.
            limit = float(self.cfg.swiglu_limit)
            hidden_fp4, hidden_scale = rtp_llm_ops.ppu_silu_and_mul_post_quant_mxfp4(
                gate_up, limit if limit > 0 else None
            )
        else:
            hidden = (
                require_silu_mul_split()(
                    gate_up[:, : self.inter_local].float().contiguous(),
                    gate_up[:, self.inter_local :].float().contiguous(),
                    clamp_limit=self.cfg.swiglu_limit,
                )
                .to(torch.bfloat16)
                .contiguous()
            )
            hidden_fp4, hidden_scale = downcast_to_mxfp4(hidden)
        down = torch.empty((total, dim), dtype=torch.bfloat16, device=x.device)
        if shared_schedule is not None:
            shared_schedule.before_down()
        self._grouped_gemm(
            (hidden_fp4, hidden_scale),
            (self._ppu_w2, self._ppu_s2),
            None,
            down,
            expert_ids,
            expert_counts,
        )

        if shared is not None:
            local = torch.empty_like(shared)
            gather_and_combine_local_loop_compatible(
                down,
                adjusted_ids,
                weights.contiguous(),
                output_index,
                shared,
                local,
                float(route_scale),
            )
            return local

        gathered = torch.empty((token_count, dim), dtype=torch.float32, device=x.device)
        gather_local_loop_compatible(
            down,
            adjusted_ids,
            weights.contiguous(),
            output_index,
            gathered,
            fused=self._fused_gather,
        )
        return gathered


def _quantize_per_row_fp8(
    values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match the ordinary DeepGEMM W4A8 per-row E4M3 contract."""

    values = values.float()
    maximum = values.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-10)
    scale = maximum * (1.0 / 448.0)
    payload = (values * (448.0 / maximum)).clamp(-448.0, 448.0)
    return payload.to(torch.float8_e4m3fn).contiguous(), scale.contiguous()


def _quantize_block128_ue8m0_fp8(
    values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match official block128 UE8M0 QAT with RTP's verified v2 kernel."""

    from rtp_llm.models_py.modules.factory.linear.ppu_impl.fp8_linear import (
        _resolve_ppu_quant_symbols,
    )

    if values.ndim != 2 or values.dtype != torch.bfloat16 or not values.is_contiguous():
        raise TypeError("W4A8 activation must be contiguous BF16 [M,K]")
    if values.shape[1] % 128:
        raise ValueError("W4A8 activation K must be divisible by 128")
    payload = torch.empty_like(values, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        (values.shape[0], values.shape[1] // 128),
        dtype=torch.float32,
        device=values.device,
    )
    if values.shape[0]:
        _, quantize_v2 = _resolve_ppu_quant_symbols()
        fp8_max = torch.finfo(payload.dtype).max
        quantize_v2(
            values,
            payload,
            scale,
            128,
            1.0e-4,
            -fp8_max,
            fp8_max,
            True,
            False,
            None,
        )
    return payload, scale


def _prepare_mxfp4_as_nvfp4(
    packed: torch.Tensor, scale: torch.Tensor, deep_gemm
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Represent checkpoint K32 MXFP4 scales exactly in the NVFP4 ABI."""

    experts, n, packed_k = packed.shape
    k = packed_k * 2
    if n % 128 or k % 128 or scale.shape != (experts, n, k // 32):
        raise ValueError("W4A8 NVFP4 requires E,N,K/2 weights and E,N,K/32 scales")
    magnitude = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 2,
        device=packed.device,
        dtype=torch.float32,
    )
    byte = torch.arange(256, device=packed.device)
    byte_lut = torch.maximum(magnitude[byte & 15], magnitude[byte >> 4])
    code_max = (
        byte_lut[packed.view(torch.uint8).long()]
        .reshape(experts, n, k // 32, 16)
        .amax(-1)
    )
    scale_fp32 = scale.float()
    value_bound = (
        (code_max * scale_fp32)
        .reshape(experts, n // 128, 128, k // 128, 4)
        .amax((2, 4))
    )
    scale_bound = scale_fp32.reshape(experts, n // 128, 128, k // 128, 4).amax((2, 4))
    bound = torch.maximum(value_bound, scale_bound).clamp_min(1.0e-30)
    outer = torch.exp2(torch.ceil(torch.log2(bound / 448.0))).contiguous()
    divisor = outer.repeat_interleave(128, 1).repeat_interleave(8, 2)
    native = (
        (scale_fp32.repeat_interleave(2, -1) / divisor)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    restored = native.float() * divisor
    expected = scale_fp32.repeat_interleave(2, -1)
    if not torch.equal(restored, expected):
        raise RuntimeError("checkpoint MXFP4 scales are not exactly representable")
    prepared = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_NVFP4,
        False,
        packed.view(torch.uint8).contiguous(),
        native,
    )
    return prepared[0], prepared[1], outer


def _compact_fp8_routes_nopad(
    payload: torch.Tensor,
    scale: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Group per-token FP8 rows by expert and retain the inverse route map."""

    flat_ids = topk_ids.reshape(-1)
    valid = (flat_ids >= 0) & (flat_ids < num_experts)
    safe_ids = torch.where(valid, flat_ids, torch.zeros_like(flat_ids))
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    expert_counts.scatter_add_(0, safe_ids.long(), valid.to(torch.int32))
    sort_key = torch.where(valid, flat_ids, torch.full_like(flat_ids, num_experts))
    permutation = torch.argsort(sort_key, stable=True)
    source_rows = torch.div(
        permutation, topk_ids.shape[1], rounding_mode="floor"
    ).long()
    compact_payload = payload.index_select(0, source_rows).contiguous()
    compact_scale = scale.index_select(0, source_rows).contiguous()
    expert_ids = torch.where(
        valid.index_select(0, permutation),
        flat_ids.index_select(0, permutation),
        torch.zeros_like(flat_ids),
    ).to(torch.int32)
    compact_positions = torch.arange(
        flat_ids.numel(), dtype=torch.int64, device=topk_ids.device
    )
    output_index_flat = torch.empty_like(compact_positions)
    output_index_flat.scatter_(0, permutation.long(), compact_positions)
    output_index_flat = torch.where(
        valid, output_index_flat, torch.full_like(output_index_flat, -1)
    )
    return (
        compact_payload,
        compact_scale,
        expert_ids,
        output_index_flat.view_as(topk_ids),
        expert_counts,
        permutation,
    )


class PpuGroupedW4A8Strategy(PpuGroupedFP4Strategy):
    """Ordinary DeepGEMM W4A8 routed experts for eager PPU Prefill."""

    name = "ppu_grouped_w4a8"

    def setup_weights(self, layer_weights: Dict) -> None:
        from rtp_llm.utils.model_weight import W

        keys = (
            W.v4_routed_w1_w,
            W.v4_routed_w1_s,
            W.v4_routed_w2_w,
            W.v4_routed_w2_s,
            W.v4_routed_w3_w,
            W.v4_routed_w3_s,
        )
        try:
            w1, s1, w2, s2, w3, s3 = (layer_weights[key] for key in keys)
        except KeyError as error:
            raise KeyError(
                f"ppu_grouped_w4a8 requires all six packed routed tensors; missing {error}"
            ) from error
        inter_local, routed_tp_size = _derive_inter_local_and_tp(
            self.cfg,
            tuple(w1.shape),
            tuple(w2.shape),
            tuple(w3.shape),
            tuple(s1.shape),
            tuple(s2.shape),
            tuple(s3.shape),
        )
        if any(
            weight.dtype not in (torch.int8, torch.uint8) for weight in (w1, w2, w3)
        ):
            raise TypeError("ppu_grouped_w4a8 requires packed int8/uint8 MXFP4 weights")
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if e8m0_dtype is None or any(
            scale.dtype != e8m0_dtype for scale in (s1, s2, s3)
        ):
            raise TypeError("ppu_grouped_w4a8 requires float8_e8m0fnu scales")
        tensors = (w1, s1, w2, s2, w3, s3)
        if any(not tensor.is_cuda for tensor in tensors):
            raise ValueError("ppu_grouped_w4a8 requires CUDA-compatible PPU tensors")
        if any(tensor.device != w1.device for tensor in tensors):
            raise ValueError("ppu_grouped_w4a8 tensors must share one device")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("ppu_grouped_w4a8 tensors must be contiguous")

        import deep_gemm

        grouped_gemm = getattr(
            deep_gemm, "m_grouped_gemm_fp8_nvfp4_bf16_nt_nopad", None
        )
        preprocess = getattr(deep_gemm, "preprocess_weights_and_scales", None)
        mode = getattr(getattr(deep_gemm, "PreprocessMode", None), "W4A8_NVFP4", None)
        if not callable(grouped_gemm) or not callable(preprocess) or mode is None:
            raise RuntimeError("PPU deep_gemm lacks the ordinary W4A8 grouped ABI")

        w13, s13, outer13 = _prepare_mxfp4_as_nvfp4(
            torch.cat((w1, w3), dim=1).contiguous(),
            torch.cat((s1, s3), dim=1).contiguous(),
            deep_gemm,
        )
        w2, s2, outer2 = _prepare_mxfp4_as_nvfp4(w2, s2, deep_gemm)
        self.register_buffer("_ppu_w13", w13, persistent=False)
        self.register_buffer("_ppu_s13", s13, persistent=False)
        self.register_buffer("_ppu_outer13", outer13, persistent=False)
        self.register_buffer("_ppu_w2", w2, persistent=False)
        self.register_buffer("_ppu_s2", s2, persistent=False)
        self.register_buffer("_ppu_outer2", outer2, persistent=False)
        self.inter_local = inter_local
        self.routed_tp_size = routed_tp_size
        self._grouped_gemm = grouped_gemm
        for key in keys:
            layer_weights.pop(key)
        logger.info(
            "DSV4_PPU_GROUPED_OPERATOR_PATH strategy=ppu_grouped_w4a8 "
            "operator=deep_gemm.m_grouped_gemm_fp8_nvfp4_bf16_nt_nopad "
            "operator_module=%s activation_mode=block128_ue8m0_fp32_scale "
            "route_weight=fp32_pre_gemm2",
            _module_path(deep_gemm),
        )

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        *,
        shared: torch.Tensor | None = None,
        route_scale: float | None = None,
        shared_schedule=None,
    ) -> torch.Tensor:
        if self.inter_local <= 0 or self._grouped_gemm is None:
            raise RuntimeError("ppu_grouped_w4a8 weights were not bound")
        if x.ndim != 2 or x.dtype != torch.bfloat16:
            raise TypeError("ppu_grouped_w4a8 input must be BF16 [N,D]")
        if weights.dtype != torch.float32 or weights.shape != indices.shape:
            raise TypeError("route weights must be FP32 and match route ids")
        if indices.dtype != torch.int64 or x.size(0) != indices.size(0):
            raise TypeError("route ids must be int64 [N,K] matching input rows")
        if (shared is None) != (route_scale is None):
            raise ValueError("shared and route_scale must be provided together")

        from rtp_llm.models_py.kernels.ppu.moe.ppu_moe_exact_gather import (
            gather_and_combine_local_loop_compatible,
            gather_local_loop_compatible,
        )

        token_count, dim = x.shape
        if token_count == 0:
            dtype = torch.bfloat16 if shared is not None else torch.float32
            return torch.zeros((0, dim), dtype=dtype, device=x.device)

        payload, scale = _quantize_block128_ue8m0_fp8(x.contiguous())
        (
            compact_payload,
            compact_scale,
            expert_ids,
            output_index,
            expert_counts,
            permutation,
        ) = _compact_fp8_routes_nopad(
            payload, scale, indices.contiguous(), int(self.cfg.n_local_experts)
        )
        total = indices.numel()
        gate_up = torch.empty(
            (total, 2 * self.inter_local), dtype=torch.bfloat16, device=x.device
        )
        if shared_schedule is not None:
            shared_schedule.before_up()
        self._grouped_gemm(
            (compact_payload, compact_scale),
            (self._ppu_w13, self._ppu_s13),
            gate_up,
            expert_ids,
            expert_counts,
            lhs_scale_group_size=128,
            rhs_scales=self._ppu_outer13,
            rhs_scale_block_size=128,
            execution="fused",
        )
        if shared_schedule is not None:
            shared_schedule.after_up()

        gate, up = gate_up.chunk(2, dim=-1)
        gate, up = gate.float(), up.float()
        limit = float(self.cfg.swiglu_limit)
        if limit > 0:
            gate.clamp_(max=limit)
            up.clamp_(min=-limit, max=limit)
        compact_weights = weights.reshape(-1).index_select(0, permutation).float()
        hidden = torch.nn.functional.silu(gate).mul_(up)
        hidden.mul_(compact_weights.unsqueeze(-1))
        hidden = hidden.to(torch.bfloat16).contiguous()
        hidden_payload, hidden_scale = _quantize_block128_ue8m0_fp8(hidden)
        down = torch.empty((total, dim), dtype=torch.bfloat16, device=x.device)
        if shared_schedule is not None:
            shared_schedule.before_down()
        self._grouped_gemm(
            (hidden_payload, hidden_scale),
            (self._ppu_w2, self._ppu_s2),
            down,
            expert_ids,
            expert_counts,
            lhs_scale_group_size=128,
            rhs_scales=self._ppu_outer2,
            rhs_scale_block_size=128,
            execution="fused",
        )

        gather_weights = torch.ones_like(weights, dtype=torch.float32)
        if shared is not None:
            local = torch.empty_like(shared)
            gather_and_combine_local_loop_compatible(
                down,
                indices,
                gather_weights,
                output_index,
                shared,
                local,
                float(route_scale),
            )
            return local
        gathered = torch.empty((token_count, dim), dtype=torch.float32, device=x.device)
        gather_local_loop_compatible(
            down,
            indices,
            gather_weights,
            output_index,
            gathered,
            fused=self._fused_gather,
        )
        return gathered


__all__ = ["PpuGroupedFP4Strategy", "PpuGroupedW4A8Strategy"]
