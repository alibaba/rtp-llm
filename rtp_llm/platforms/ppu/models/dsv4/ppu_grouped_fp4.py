"""M890P grouped-MXFP4 routed experts for the DSV4 TP4/EP1 topology.

This strategy is deliberately separate from the CUDA ``grouped_fp4`` backend.
It consumes the checkpoint's packed MXFP4 tensors, executes two PPU nopad
grouped GEMMs, and returns the normalized routed partial expected by ``MoE``'s
post-W2 route-scale/TP-reduce contract.

There is no fallback in this module.  A wrong topology, storage geometry,
device, or external symbol fails closed.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Dict, Tuple

import torch
from .ppu_moe_config import PpuMoeConfig as MoeCfg

logger = logging.getLogger(__name__)

_ROUTE_WEIGHT_CONTRACT = "post_w2_normalized_then_scale_v1"
_GROUPED_GEMM_SYMBOL = "deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad"
_ACTIVATION_MODE = "eager_swiglu_bf16_then_mxfp4"
_EXACT_GATHER_SYMBOL = (
    "rtp_llm.platforms.ppu.kernels.ppu_moe_exact_gather." "gather_local_loop_compatible"
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
    """The provider seam is intentionally limited to the TP4/EP1 pivot."""

    return int(cfg.tp_size) == 4 and int(cfg.ep_size) == 1


def _runtime_eligible() -> bool:
    """Probe the PPU-only leaf without making generic GPU TP4 select it."""

    if not torch.cuda.is_available():
        return False
    try:
        current_device = torch.cuda.current_device()
        if torch.cuda.get_device_name(current_device) != "ZW-M890P":
            return False
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
        raise RuntimeError("ppu_grouped_fp4 requires exactly tp_size=4 and ep_size=1")

    experts = int(cfg.n_local_experts)
    dim = int(cfg.dim)
    if experts != int(cfg.n_routed_experts):
        raise ValueError(
            "EP1 grouped MXFP4 must bind every routed expert on each TP rank"
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
    """Strict M890P packed-MXFP4 strategy for DSV4 TP4/EP1."""

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
                "ppu_grouped_fp4 setup rejected a non-TP4/EP1 configuration"
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
        if torch.cuda.get_device_name(w1.device) != "ZW-M890P":
            raise RuntimeError("ppu_grouped_fp4 requires ZW-M890P")

        import deep_gemm

        grouped_gemm = getattr(deep_gemm, "m_grouped_gemm_fp4_fp4_bf16_nt_nopad", None)
        if not callable(grouped_gemm):
            raise RuntimeError(
                "PPU deep_gemm lacks m_grouped_gemm_fp4_fp4_bf16_nt_nopad"
            )
        from rtp_llm.platforms.ppu.kernels.ppu_mxfp4 import (
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

        token_count, dim = x.shape
        if token_count == 0:
            return torch.zeros((0, dim), dtype=torch.float32, device=x.device)

        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.expert import (
            require_silu_mul_split,
        )
        from rtp_llm.platforms.ppu.kernels.ppu_moe_exact_gather import (
            gather_local_loop_compatible,
        )
        from rtp_llm.platforms.ppu.kernels.ppu_moe_nopad import (
            compact_mxfp4_routes_nopad,
        )
        from rtp_llm.platforms.ppu.kernels.ppu_mxfp4 import downcast_to_mxfp4

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
        self._grouped_gemm(
            (compact_fp4, compact_scale),
            (self._ppu_w13, self._ppu_s13),
            None,
            gate_up,
            expert_ids,
            expert_counts,
        )

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
        self._grouped_gemm(
            (hidden_fp4, hidden_scale),
            (self._ppu_w2, self._ppu_s2),
            None,
            down,
            expert_ids,
            expert_counts,
        )

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


__all__ = ["PpuGroupedFP4Strategy"]
