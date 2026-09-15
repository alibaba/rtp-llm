"""CUDA-extension MoE front for the DeepSeek-V4 MegaMoE decode path."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from functools import reduce
from operator import mul
from typing import TYPE_CHECKING

import torch

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range

if TYPE_CHECKING:
    from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
        ChunkedFp8Fp4MoeLayer,
    )


MEGA_MOE_FRONT_CAPACITY = 256
_HC_MULT = 4
_HC_WIDTH = 24
_ABI_VERSION = 1
_KERNEL_CONTRACT_VERSION = 3
_TOPK = 6
_TRUE_ENV_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_ENV_VALUES = frozenset(("0", "false", "no", "off", ""))
_AUTO_ENV_VALUES = frozenset(("auto",))


def moe_front_mode() -> str:
    """Parse the MoE-front mode as ``off``, ``auto``, or ``required``."""

    value = os.environ.get("DSV4_MEGA_MOE_FRONT", "0").strip().lower()
    if value in _TRUE_ENV_VALUES:
        return "required"
    if value in _AUTO_ENV_VALUES:
        return "auto"
    if value in _FALSE_ENV_VALUES:
        return "off"
    raise RuntimeError(
        "DSV4_MEGA_MOE_FRONT must be one of "
        f"{sorted(_TRUE_ENV_VALUES | _AUTO_ENV_VALUES | _FALSE_ENV_VALUES)}, "
        f"got {value!r}"
    )


def moe_front_requested() -> bool:
    """Return whether the standalone MoE-front path is explicitly enabled."""

    return moe_front_mode() != "off"


def _parse_arches(value: object) -> set[str]:
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _validate_extension_contract(
    ops, dim: int, experts: int, topk: int, device: torch.device
) -> dict:
    geometry = ops.geometry_moe_front(dim)
    if not isinstance(geometry, Mapping):
        raise RuntimeError(
            "DSV4 MoE-front geometry must be a mapping, "
            f"got {type(geometry).__name__}"
        )
    expected = {
        "abi_version": _ABI_VERSION,
        "kernel_contract_version": _KERNEL_CONTRACT_VERSION,
        "hidden": dim,
        "hc_mult": _HC_MULT,
        "hc_width": _HC_WIDTH,
        "experts": experts,
        "topk": topk,
        "max_m": MEGA_MOE_FRONT_CAPACITY,
        "scale_cols": dim // 128,
        "collapse_ssq_bits": 32,
    }
    mismatches = {
        name: (geometry.get(name), value)
        for name, value in expected.items()
        if geometry.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"DSV4 MoE-front geometry mismatch: {mismatches}")

    capability = tuple(torch.cuda.get_device_capability(device))
    arch = {(10, 0): "sm_100a", (10, 3): "sm_103a"}.get(capability)
    if arch is None:
        raise RuntimeError(
            "DSV4 MoE front requires sm_100a or sm_103a, "
            f"got compute capability {capability}"
        )

    build_info = ops.build_info_moe_front()
    if not isinstance(build_info, Mapping):
        raise RuntimeError("DSV4 MoE-front build info must be a mapping")
    for field in ("target_arches", "production_arch"):
        if arch not in _parse_arches(build_info.get(field, "")):
            raise RuntimeError(
                f"DSV4 MoE-front build does not contain {arch} in {field}: "
                f"{build_info.get(field)!r}"
            )
    if build_info.get("kernel_count") != 4:
        raise RuntimeError(
            "DSV4 MoE-front build must publish four kernels, got "
            f"{build_info.get('kernel_count')!r}"
        )

    source_commit = str(build_info.get("source_commit", ""))
    source_sha256 = str(build_info.get("source_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{8,40}", source_commit):
        raise RuntimeError(
            f"DSV4 MoE-front build has invalid source commit {source_commit!r}"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise RuntimeError(
            f"DSV4 MoE-front build has invalid source SHA256 {source_sha256!r}"
        )
    return dict(geometry)


class MegaMoeFrontAdapter:
    """Stage mHC state into the extension and launch prepacked MegaMoE-SE.

    The extension's DeepGEMM TMA descriptor binds the input address when a plan
    is created. During CUDA Graph capture, a plan is retained for each graph-
    stable input address. Eager plans are temporary so arbitrary allocator
    addresses cannot grow a permanent cache. The extension writes quantized
    activations, routing results, and shared-expert scales directly into
    DeepGEMM's aligned symmetric buffer.
    """

    def __init__(
        self,
        moe: "ChunkedFp8Fp4MoeLayer",
        ffn_hc,
        ffn_norm,
    ) -> None:
        from rtp_kernel import dsv4_moe_front

        if moe.strategy_name != "mega_moe_se":
            raise RuntimeError(
                "DSV4 MoE front requires the MegaMoE-SE strategy; "
                f"selected strategy={moe.strategy_name!r}"
            )

        self.layer_id = int(moe.layer_id)
        self.dim = int(moe.dim)
        self.executor = moe.fused_moe.fused_experts
        self.gate = moe.gate
        self.ffn_hc = ffn_hc
        self.ffn_norm = ffn_norm
        self._ops = dsv4_moe_front

        if int(self.gate.topk) != _TOPK:
            raise RuntimeError(
                "DSV4 MoE front requires TopK-"
                f"{_TOPK}, got {int(self.gate.topk)}"
            )

        device = self.gate.weight.device
        if device.type != "cuda":
            raise RuntimeError(f"DSV4 MoE front requires CUDA weights, got {device}")
        try:
            geometry = _validate_extension_contract(
                dsv4_moe_front,
                self.dim,
                int(self.gate.weight.shape[0]),
                int(self.gate.topk),
                device,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"DSV4 MoE-front validation failed for layer {self.layer_id}: {exc}"
            ) from exc
        if tuple(ffn_hc.fn.shape) != (_HC_WIDTH, _HC_MULT * self.dim):
            raise RuntimeError(
                f"DSV4 MoE-front hc_fn shape mismatch: {tuple(ffn_hc.fn.shape)}"
            )

        self.collapsed = torch.empty(
            (MEGA_MOE_FRONT_CAPACITY, self.dim),
            dtype=torch.bfloat16,
            device=device,
        )
        self.collapse_ssq = torch.empty(
            (MEGA_MOE_FRONT_CAPACITY,), dtype=torch.float32, device=device
        )
        self.normalized_mix = torch.empty(
            (MEGA_MOE_FRONT_CAPACITY, _HC_WIDTH),
            dtype=torch.float32,
            device=device,
        )
        self.normalized = torch.empty_like(self.collapsed)
        self.router_logits = torch.empty(
            (MEGA_MOE_FRONT_CAPACITY, int(self.gate.weight.shape[0])),
            dtype=torch.float32,
            device=device,
        )
        self.post = torch.empty(
            (MEGA_MOE_FRONT_CAPACITY, _HC_MULT),
            dtype=torch.float32,
            device=device,
        )
        self.comb = torch.empty(
            (MEGA_MOE_FRONT_CAPACITY, _HC_MULT, _HC_MULT),
            dtype=torch.float32,
            device=device,
        )

        self.hc_fn = ffn_hc.fn.contiguous()
        self.hc_base = ffn_hc.base.contiguous()
        self.hc_scale = ffn_hc.scale.contiguous()
        self.ffn_norm_weight = ffn_norm.weight.contiguous()
        self.router_weight = self.gate._weight_bf16().contiguous()
        if self.ffn_norm_weight.dtype != torch.bfloat16:
            raise RuntimeError(
                "DSV4 MoE-front learned RMSNorm weight must be BF16, got "
                f"{self.ffn_norm_weight.dtype}"
            )

        self.correction_bias = None
        self.input_ids = None
        self.tid2eid = None
        if self.gate.hash:
            self.input_ids = torch.empty(
                (MEGA_MOE_FRONT_CAPACITY,), dtype=torch.int64, device=device
            )
            self.tid2eid = self.gate.tid2eid.to(torch.int32).contiguous()
        else:
            if self.gate.bias is None:
                raise RuntimeError("learned DSV4 MoE front requires correction bias")
            self.correction_bias = self.gate.bias.to(torch.float32).contiguous()

        self._workspace = dsv4_moe_front.Dsv4MoeFrontWorkspace(device)
        self._graph_plans: dict[tuple[int, int], object] = {}

        if self.layer_id == 0:
            logging.info(
                "[DSV4 MoE front] enabled: geometry=%s strategy=mega_moe_se",
                geometry,
            )

    def _create_plan(self, input_x: torch.Tensor, tokens: int):
        return self._ops.Dsv4MoeFrontPlan(
            input_x,
            self.hc_fn,
            int(tokens),
            workspace=self._workspace,
        )

    def _plan_for(self, input_x: torch.Tensor, tokens: int) -> tuple[object, bool]:
        if not torch.cuda.is_current_stream_capturing():
            return self._create_plan(input_x, tokens), True
        key = (int(tokens), int(input_x.data_ptr()))
        plan = self._graph_plans.get(key)
        if plan is None:
            plan = self._create_plan(input_x, tokens)
            self._graph_plans[key] = plan
        return plan, False

    def supports(self, residual: torch.Tensor) -> bool:
        """Return whether ``M*S`` fits both the front ABI and MoE buffer."""

        if (
            residual.dim() not in (3, 4)
            or tuple(residual.shape[-2:]) != (_HC_MULT, self.dim)
            or not residual.is_cuda
            or residual.dtype != torch.bfloat16
            or not residual.is_contiguous()
        ):
            return False
        tokens = reduce(mul, (int(value) for value in residual.shape[:-2]), 1)
        mega_capacity = int(self.executor._mega_buf.num_max_tokens_per_rank)
        return 0 <= tokens <= min(MEGA_MOE_FRONT_CAPACITY, mega_capacity)

    def forward(
        self, residual: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        leading = tuple(int(value) for value in residual.shape[:-2])
        tokens = reduce(mul, leading, 1)
        buf = self.executor._mega_buf
        capacity = min(MEGA_MOE_FRONT_CAPACITY, int(buf.num_max_tokens_per_rank))
        if tokens < 0 or tokens > capacity:
            raise RuntimeError(
                f"DSV4 MoE front supports 0..{capacity} decode tokens, got {tokens}"
            )
        if tuple(residual.shape[-2:]) != (_HC_MULT, self.dim):
            raise RuntimeError(
                "DSV4 MoE-front residual shape mismatch: "
                f"got {tuple(residual.shape)}, expected [...,{_HC_MULT},{self.dim}]"
            )
        if not residual.is_contiguous() or not input_ids.is_contiguous():
            raise RuntimeError(
                "DSV4 MoE front requires contiguous residual and input_ids for "
                "allocation-free CUDA graph capture"
            )
        input_ids_flat = input_ids.view(-1)
        if int(input_ids_flat.numel()) != tokens:
            raise RuntimeError(
                f"DSV4 MoE-front input id count {input_ids_flat.numel()} != {tokens}"
            )

        if tokens == 0:
            # Empty EP/DP ranks skip the extension but must enter the same
            # DeepGEMM collective as ranks that have local tokens.
            with record_function_range("dsv4.moe.routed_experts"):
                y = self.executor.forward_prepacked(0, residual.device)
            return (
                y.view(*leading, self.dim),
                self.normalized[:0].view(*leading, self.dim),
                self.post[:0].view(*leading, _HC_MULT, 1),
                self.comb[:0].view(*leading, _HC_MULT, _HC_MULT),
            )

        input_x = residual.view(tokens, _HC_MULT, self.dim)
        block_m = int(self.executor._block_m(tokens))

        with record_function_range("dsv4.moe.mega_front"):
            plan, temporary_plan = self._plan_for(input_x, tokens)
            try:
                if self.gate.hash:
                    assert self.input_ids is not None and self.tid2eid is not None
                    hash_input_ids = input_ids_flat
                    if hash_input_ids.dtype != torch.int64:
                        self.input_ids[:tokens].copy_(hash_input_ids)
                        hash_input_ids = self.input_ids
                    plan.run_hash_out(
                        self.hc_base,
                        self.hc_scale,
                        self.ffn_norm_weight,
                        self.router_weight,
                        hash_input_ids,
                        self.tid2eid,
                        self.collapsed,
                        self.collapse_ssq,
                        self.normalized_mix,
                        self.normalized,
                        buf.x[:tokens],
                        buf.x_sf[:tokens],
                        buf.shared_l1_acts_sf,
                        self.router_logits,
                        buf.topk_idx[:tokens],
                        buf.topk_weights[:tokens],
                        self.post,
                        self.comb,
                        block_m,
                        norm_eps=float(self.ffn_norm.variance_epsilon),
                        hc_eps=float(self.ffn_hc.hc_eps),
                        route_scale=float(self.gate.route_scale),
                        use_pdl=True,
                    )
                else:
                    assert self.correction_bias is not None
                    plan.run_learned_out(
                        self.hc_base,
                        self.hc_scale,
                        self.ffn_norm_weight,
                        self.router_weight,
                        self.correction_bias,
                        self.collapsed,
                        self.collapse_ssq,
                        self.normalized_mix,
                        self.normalized,
                        buf.x[:tokens],
                        buf.x_sf[:tokens],
                        buf.shared_l1_acts_sf,
                        buf.topk_idx[:tokens],
                        buf.topk_weights[:tokens],
                        self.post,
                        self.comb,
                        block_m,
                        router_logits=self.router_logits if tokens <= 9 else None,
                        norm_eps=float(self.ffn_norm.variance_epsilon),
                        hc_eps=float(self.ffn_hc.hc_eps),
                        route_scale=float(self.gate.route_scale),
                        use_pdl=True,
                    )
            finally:
                if temporary_plan:
                    plan.close()

        with record_function_range("dsv4.moe.routed_experts"):
            y = self.executor.forward_prepacked(tokens, residual.device)
        return (
            y.view(*leading, self.dim),
            self.normalized[:tokens].view(*leading, self.dim),
            self.post[:tokens].view(*leading, _HC_MULT, 1),
            self.comb[:tokens].view(*leading, _HC_MULT, _HC_MULT),
        )


__all__ = [
    "MEGA_MOE_FRONT_CAPACITY",
    "MegaMoeFrontAdapter",
    "moe_front_mode",
    "moe_front_requested",
]
