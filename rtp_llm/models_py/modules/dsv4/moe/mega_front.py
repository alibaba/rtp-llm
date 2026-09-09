"""CUDA-extension MoE front for the DeepSeek-V4 MegaMoE decode path."""

from __future__ import annotations

import logging
import re
from functools import reduce
from operator import mul
from typing import TYPE_CHECKING, Sequence

import torch

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import (
    HC,
    HC_MIX,
    MAX_BATCH,
)

if TYPE_CHECKING:
    from .moe_layer import MoE


MEGA_MOE_FRONT_CAPACITY = MAX_BATCH
_ABI_VERSION = 1
_KERNEL_CONTRACT_VERSION = 3
_TOPK = 6


def _parse_arches(value: object) -> set[str]:
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _validate_extension_contract(
    dsv4_mega, dim: int, experts: int, topk: int, device: torch.device
) -> dict:
    geometry = dsv4_mega.geometry_moe_front(dim)
    expected = {
        "abi_version": _ABI_VERSION,
        "kernel_contract_version": _KERNEL_CONTRACT_VERSION,
        "hidden": dim,
        "hc_mult": HC,
        "hc_width": HC_MIX,
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

    build_info = dsv4_mega.build_info_moe_front()
    if not isinstance(build_info, dict):
        raise RuntimeError("DSV4 MoE-front build info must be a dictionary")
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
    return geometry


def _capture_tokens_for_batches(
    capture_batches: Sequence[int], gen_num_per_cycle: int
) -> tuple[int, ...]:
    """Expand framework-selected graph batches into MoE-front token widths."""
    gamma = int(gen_num_per_cycle)
    if gamma < 0:
        raise ValueError(f"DSV4 MoE front requires non-negative gamma, got {gamma}")

    # The framework has already parsed and validated capture_batches. This
    # helper only adds the DSpARK/target widths required by the front ABI.
    multipliers = {1}
    if gamma > 0:
        multipliers.update((gamma, gamma + 1))
    values = {
        int(batch) * multiplier
        for batch in capture_batches
        if int(batch) > 0
        for multiplier in multipliers
    }
    # A graph bucket above the front ABI limit remains valid for the ordinary
    # path; Block.supports() selects that path instead of the native front.
    return tuple(sorted(value for value in values if 0 < value <= MAX_BATCH))


class MegaMoeFrontAdapter:
    """Stage mHC state into the extension and launch prepacked MegaMoE-SE.

    The extension's DeepGEMM TMA descriptor binds the input address when a plan
    is created. Each layer therefore owns a stable 256-row staging tensor and a
    plan per configured decode batch. The extension writes quantized activations,
    routing results, and shared-expert scales directly into DeepGEMM's aligned
    symmetric buffer; no RTP gate/quant/pack kernel runs on this path.
    """

    def __init__(
        self,
        moe: "MoE",
        ffn_hc,
        ffn_norm,
        *,
        gen_num_per_cycle: int = 0,
    ) -> None:
        from rtp_kernel import dsv4_mega

        strategy = moe._strategy
        if getattr(strategy, "name", "") != "mega_se":
            raise RuntimeError(
                "DSV4 MoE front requires the MegaMoE-SE strategy; "
                f"selected strategy={getattr(strategy, 'name', '<unknown>')!r}"
            )

        self.layer_id = int(moe.layer_id)
        self.dim = int(moe.dim)
        self.strategy = strategy
        self.gate = moe.gate
        self.ffn_hc = ffn_hc
        self.ffn_norm = ffn_norm
        self._dsv4_mega = dsv4_mega

        if int(moe.n_activated_experts) != _TOPK:
            raise RuntimeError(
                "DSV4 MoE front requires TopK-"
                f"{_TOPK}, got {int(moe.n_activated_experts)}"
            )

        device = self.gate.weight.device
        if device.type != "cuda":
            raise RuntimeError(f"DSV4 MoE front requires CUDA weights, got {device}")
        try:
            geometry = _validate_extension_contract(
                dsv4_mega,
                self.dim,
                int(moe.n_routed_experts),
                int(moe.n_activated_experts),
                device,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"DSV4 MoE-front validation failed for layer {self.layer_id}: {exc}"
            ) from exc
        if tuple(ffn_hc.fn.shape) != (HC_MIX, HC * self.dim):
            raise RuntimeError(
                f"DSV4 MoE-front hc_fn shape mismatch: {tuple(ffn_hc.fn.shape)}"
            )

        self.hidden = torch.empty(
            (MAX_BATCH, HC, self.dim),
            dtype=torch.bfloat16,
            device=device,
        )
        self.collapsed = torch.empty(
            (MAX_BATCH, self.dim), dtype=torch.bfloat16, device=device
        )
        self.collapse_ssq = torch.empty(
            (MAX_BATCH,), dtype=torch.float32, device=device
        )
        self.normalized_mix = torch.empty(
            (MAX_BATCH, HC_MIX), dtype=torch.float32, device=device
        )
        self.normalized = torch.empty_like(self.collapsed)
        self.router_logits = torch.empty(
            (MAX_BATCH, int(moe.n_routed_experts)),
            dtype=torch.float32,
            device=device,
        )
        self.post = torch.empty((MAX_BATCH, HC), dtype=torch.float32, device=device)
        self.comb = torch.empty(
            (MAX_BATCH, HC, HC),
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
            self.input_ids = torch.empty((MAX_BATCH,), dtype=torch.int64, device=device)
            self.tid2eid = self.gate.tid2eid.to(torch.int32).contiguous()
        else:
            if self.gate.bias is None:
                raise RuntimeError("learned DSV4 MoE front requires correction bias")
            self.correction_bias = self.gate.bias.to(torch.float32).contiguous()

        self._plans: dict[int, object] = {}
        self._gen_num_per_cycle = int(gen_num_per_cycle)

        if self.layer_id == 0:
            logging.info(
                "[DSV4 MoE front] enabled: geometry=%s gen_num_per_cycle=%d "
                "strategy=mega_se",
                geometry,
                self._gen_num_per_cycle,
            )

    def _create_plan(self, tokens: int):
        return self._dsv4_mega.Dsv4MoeFrontPlan(self.hidden, self.hc_fn, int(tokens))

    def prepare_capture_plans(self, capture_batches: Sequence[int]) -> tuple[int, ...]:
        """Create plans for the framework's final CUDA Graph capture buckets."""
        capture_tokens = _capture_tokens_for_batches(
            capture_batches, self._gen_num_per_cycle
        )
        for tokens in capture_tokens:
            if tokens not in self._plans:
                self._plans[tokens] = self._create_plan(tokens)
        return capture_tokens

    def _plan_for(self, tokens: int):
        plan = self._plans.get(tokens)
        if plan is not None:
            return plan
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"DSV4 MoE-front plan for {tokens} tokens was not created before "
                "CUDA graph capture; prepare plans from the Graph capture range"
            )
        plan = self._create_plan(tokens)
        self._plans[tokens] = plan
        return plan

    def supports(self, residual: torch.Tensor) -> bool:
        """Return whether ``M*S`` fits both the front ABI and MoE buffer."""

        if residual.dim() < 2:
            return False
        tokens = reduce(mul, (int(value) for value in residual.shape[:-2]), 1)
        mega_capacity = int(self.strategy._mega_buf.num_max_tokens_per_rank)
        return 0 <= tokens <= min(MAX_BATCH, mega_capacity)

    def forward(
        self, residual: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        leading = tuple(int(value) for value in residual.shape[:-2])
        tokens = reduce(mul, leading, 1)
        buf = self.strategy._mega_buf
        capacity = min(MAX_BATCH, int(buf.num_max_tokens_per_rank))
        if tokens < 0 or tokens > capacity:
            raise RuntimeError(
                f"DSV4 MoE front supports 0..{capacity} decode tokens, got {tokens}"
            )
        if tuple(residual.shape[-2:]) != (HC, self.dim):
            raise RuntimeError(
                "DSV4 MoE-front residual shape mismatch: "
                f"got {tuple(residual.shape)}, expected [...,{HC},{self.dim}]"
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
                y = self.strategy.forward_prepacked(0, residual.device)
            return (
                y.view(*leading, self.dim),
                self.normalized[:0].view(*leading, self.dim),
                self.post[:0].view(*leading, HC, 1),
                self.comb[:0].view(*leading, HC, HC),
            )

        # The plan's TMA descriptor is bound to self.hidden. This is the only
        # staging operation; all following front outputs land in final buffers.
        self.hidden[:tokens].copy_(residual.view(tokens, HC, self.dim))
        plan = self._plan_for(tokens)
        block_m = int(self.strategy._block_m(tokens))

        with record_function_range("dsv4.moe.mega_front"):
            if self.gate.hash:
                assert self.input_ids is not None and self.tid2eid is not None
                self.input_ids[:tokens].copy_(input_ids_flat)
                plan.run_hash_out(
                    self.hc_base,
                    self.hc_scale,
                    self.ffn_norm_weight,
                    self.router_weight,
                    self.input_ids,
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

        with record_function_range("dsv4.moe.routed_experts"):
            y = self.strategy.forward_prepacked(tokens, residual.device)
        return (
            y.view(*leading, self.dim),
            self.normalized[:tokens].view(*leading, self.dim),
            self.post[:tokens].view(*leading, HC, 1),
            self.comb[:tokens].view(*leading, HC, HC),
        )


__all__ = ["MEGA_MOE_FRONT_CAPACITY", "MegaMoeFrontAdapter"]
