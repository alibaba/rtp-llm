"""CUDA-extension MoE front for the DeepSeek-V4 MegaMoE decode path."""

from __future__ import annotations

import logging
import os
from functools import reduce
from operator import mul
from typing import TYPE_CHECKING

import torch

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range

if TYPE_CHECKING:
    from .moe_layer import MoE


_CAPACITY_M = 128
_HC_MULT = 4
_HC_WIDTH = 24
_TOPK = 6


def _decode_capture_tokens() -> tuple[int, ...]:
    raw = os.environ.get("DECODE_CAPTURE_CONFIG", "")
    capture_batches: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise RuntimeError(
                f"invalid DECODE_CAPTURE_CONFIG item {item!r} for DSV4 MoE front"
            ) from exc
        if value < 1:
            raise RuntimeError(
                f"DSV4 MoE front requires positive capture batches, got {value}"
            )
        capture_batches.add(value)

    raw_gamma = os.environ.get(
        "GEN_NUM_PER_CIRCLE", os.environ.get("GEN_NUM_PER_CYCLE", "0")
    ).strip()
    try:
        gamma = int(raw_gamma or "0")
    except ValueError as exc:
        raise RuntimeError(
            f"invalid MTP generation width {raw_gamma!r} for DSV4 MoE front"
        ) from exc
    if gamma < 0:
        raise RuntimeError(f"DSV4 MoE front requires non-negative gamma, got {gamma}")

    # One service can capture ordinary decode (B), DSpARK draft (B*gamma), and
    # target verification (B*(gamma+1)). TMA plans bind their input address, so
    # all three token counts must exist before any of those graphs are captured.
    multipliers = {1}
    if gamma > 0:
        multipliers.update((gamma, gamma + 1))
    values = {
        batch * multiplier for batch in capture_batches for multiplier in multipliers
    }
    invalid = sorted(value for value in values if value > _CAPACITY_M)
    if invalid:
        raise RuntimeError(
            "DSV4 MoE front supports capture token counts in "
            f"[1,{_CAPACITY_M}], got {invalid} from batches="
            f"{sorted(capture_batches)} gamma={gamma}"
        )
    return tuple(sorted(values))


class MegaMoeFrontAdapter:
    """Stage mHC state into the extension and launch prepacked MegaMoE-SE.

    The extension's DeepGEMM TMA descriptor binds the input address when a plan
    is created. Each layer therefore owns a stable 128-row staging tensor and a
    plan per configured decode batch. The extension writes quantized activations,
    routing results, and shared-expert scales directly into DeepGEMM's aligned
    symmetric buffer; no RTP gate/quant/pack kernel runs on this path.
    """

    def __init__(self, moe: "MoE", ffn_hc, ffn_norm) -> None:
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

        geometry = dsv4_mega.geometry_moe_front(self.dim)
        expected = {
            "hidden": self.dim,
            "hc_mult": _HC_MULT,
            "hc_width": _HC_WIDTH,
            "experts": int(moe.n_routed_experts),
            "topk": int(moe.n_activated_experts),
            "max_m": _CAPACITY_M,
        }
        mismatches = {
            name: (geometry.get(name), value)
            for name, value in expected.items()
            if geometry.get(name) != value
        }
        if mismatches:
            raise RuntimeError(
                f"DSV4 MoE-front geometry mismatch for layer {self.layer_id}: "
                f"{mismatches}; extension={geometry}"
            )
        if expected["topk"] != _TOPK:
            raise RuntimeError(
                f"DSV4 MoE front requires TopK-{_TOPK}, got {expected['topk']}"
            )

        device = self.gate.weight.device
        if device.type != "cuda":
            raise RuntimeError(f"DSV4 MoE front requires CUDA weights, got {device}")
        if tuple(ffn_hc.fn.shape) != (_HC_WIDTH, _HC_MULT * self.dim):
            raise RuntimeError(
                f"DSV4 MoE-front hc_fn shape mismatch: {tuple(ffn_hc.fn.shape)}"
            )

        self.hidden = torch.empty(
            (_CAPACITY_M, _HC_MULT, self.dim),
            dtype=torch.bfloat16,
            device=device,
        )
        self.collapsed = torch.empty(
            (_CAPACITY_M, self.dim), dtype=torch.bfloat16, device=device
        )
        self.collapse_ssq = torch.empty(
            (_CAPACITY_M,), dtype=torch.float32, device=device
        )
        self.normalized_mix = torch.empty(
            (_CAPACITY_M, _HC_WIDTH), dtype=torch.float32, device=device
        )
        self.normalized = torch.empty_like(self.collapsed)
        self.router_logits = torch.empty(
            (_CAPACITY_M, int(moe.n_routed_experts)),
            dtype=torch.float32,
            device=device,
        )
        self.post = torch.empty(
            (_CAPACITY_M, _HC_MULT), dtype=torch.float32, device=device
        )
        self.comb = torch.empty(
            (_CAPACITY_M, _HC_MULT, _HC_MULT),
            dtype=torch.float32,
            device=device,
        )

        buf = strategy._mega_buf
        if int(buf.num_max_tokens_per_rank) < _CAPACITY_M:
            raise RuntimeError(
                "MegaMoE-SE aligned input capacity is smaller than the MoE-front "
                f"ABI: {buf.num_max_tokens_per_rank} < {_CAPACITY_M}"
            )
        self.x_fp8 = buf.x[:_CAPACITY_M]
        self.x_sf = buf.x_sf[:_CAPACITY_M]
        self.shared_l1_x_sf = buf.shared_l1_acts_sf
        self.topk_ids = buf.topk_idx[:_CAPACITY_M]
        self.topk_weights = buf.topk_weights[:_CAPACITY_M]

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
                (_CAPACITY_M,), dtype=torch.int64, device=device
            )
            self.tid2eid = self.gate.tid2eid.to(torch.int32).contiguous()
        else:
            if self.gate.bias is None:
                raise RuntimeError("learned DSV4 MoE front requires correction bias")
            self.correction_bias = self.gate.bias.to(torch.float32).contiguous()

        self._plans: dict[int, object] = {}
        capture_tokens = _decode_capture_tokens()
        if not capture_tokens:
            logging.warning(
                "[DSV4 MoE front] DECODE_CAPTURE_CONFIG is empty; plans will "
                "be created lazily for eager decode"
            )
        for tokens in capture_tokens:
            self._plans[tokens] = self._create_plan(tokens)

        if self.layer_id == 0:
            logging.info(
                "[DSV4 MoE front] enabled: geometry=%s capture_tokens=%s "
                "strategy=mega_se",
                geometry,
                list(capture_tokens),
            )

    def _create_plan(self, tokens: int):
        return self._dsv4_mega.Dsv4MoeFrontPlan(self.hidden, self.hc_fn, int(tokens))

    def _plan_for(self, tokens: int):
        plan = self._plans.get(tokens)
        if plan is not None:
            return plan
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"DSV4 MoE-front plan for {tokens} tokens was not created before "
                "CUDA graph capture; include this batch in DECODE_CAPTURE_CONFIG"
            )
        plan = self._create_plan(tokens)
        self._plans[tokens] = plan
        return plan

    def supports(self, residual: torch.Tensor) -> bool:
        """Return whether this request fits the fused front's physical buffers."""

        if residual.dim() < 2:
            return False
        tokens = reduce(mul, (int(value) for value in residual.shape[:-2]), 1)
        return 0 <= tokens <= _CAPACITY_M

    def forward(
        self, residual: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        leading = tuple(int(value) for value in residual.shape[:-2])
        tokens = reduce(mul, leading, 1)
        if tokens < 0 or tokens > _CAPACITY_M:
            raise RuntimeError(
                f"DSV4 MoE front supports 0..{_CAPACITY_M} decode tokens, got {tokens}"
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
                y = self.strategy.forward_prepacked(0, residual.device)
            return (
                y.view(*leading, self.dim),
                self.normalized[:0].view(*leading, self.dim),
                self.post[:0].view(*leading, _HC_MULT, 1),
                self.comb[:0].view(*leading, _HC_MULT, _HC_MULT),
            )

        # The plan's TMA descriptor is bound to self.hidden. This is the only
        # staging operation; all following front outputs land in final buffers.
        self.hidden[:tokens].copy_(residual.view(tokens, _HC_MULT, self.dim))
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
                    self.x_fp8,
                    self.x_sf,
                    self.shared_l1_x_sf,
                    self.router_logits,
                    self.topk_ids,
                    self.topk_weights,
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
                    self.x_fp8,
                    self.x_sf,
                    self.shared_l1_x_sf,
                    self.topk_ids,
                    self.topk_weights,
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
            self.post[:tokens].view(*leading, _HC_MULT, 1),
            self.comb[:tokens].view(*leading, _HC_MULT, _HC_MULT),
        )


__all__ = ["MegaMoeFrontAdapter"]
