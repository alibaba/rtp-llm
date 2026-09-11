"""Bind V4.1 checkpoint tensors to the shared32 Mega expert backend.

The runtime supplies its actual EP8/EP16 WORLD group. Topology certification
and CP row assembly remain caller responsibilities; this module never selects
an EP1, dequantized-expert, shared128 or legacy gate fallback.
"""

import os
from collections.abc import Mapping

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.linear import is_supported
from rtp_llm.models_py.modules.dsv41.math import moe_gate


def _require_execution(values):
    if os.environ.get("DSV41_MOE", "0") != "1" or not is_supported(values):
        raise RuntimeError("V4.1 MoE requires DSV41_MOE=1 and CUDA13/Blackwell")
    if torch.is_autocast_enabled():
        raise RuntimeError("V4.1 router and experts require autocast off")


def _geometry(config, draft):
    if type(draft) is not bool:
        raise ValueError("V4.1 draft selection must be explicit boolean metadata")
    text = config.text
    experts = text["dspark_n_routed_experts" if draft else "n_routed_experts"]
    topk = text["dspark_num_experts_per_tok" if draft else "num_experts_per_tok"]
    if (
        text["hidden_size"] != 5120
        or text["moe_intermediate_size"] != 2304
        or (experts, topk) != ((128, 3) if draft else (384, 6))
        or text["n_shared_experts"] != 1
        or text["scoring_func"] != "sqrtsoftplus"
        or text["norm_topk_prob"] is not True
        or text["routed_scaling_factor"] != 1.5
        or text["swiglu_limit"] != 10.0
    ):
        raise ValueError("V4.1 MoE must retain its released expert/router geometry")
    return text["hidden_size"], text["moe_intermediate_size"], experts, topk


def _expert_range(experts, ep_size, ep_rank):
    if type(ep_size) is not int or ep_size not in (8, 16):
        raise ValueError("V4.1 expert ownership requires EP8 or EP16")
    if type(ep_rank) is not int or not 0 <= ep_rank < ep_size:
        raise ValueError("V4.1 expert rank must belong to the actual EP group")
    local = experts // ep_size
    return range(ep_rank * local, (ep_rank + 1) * local)


def _tensor(weights, name, shape, dtype, device):
    value = weights[name]
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != device
        or not value.is_contiguous()
    ):
        raise ValueError(
            f"V4.1 {name} has the wrong checkpoint shape/dtype/device/layout"
        )
    return value


class V41MoERouter(nn.Module):
    def __init__(self, config, weights, *, draft=False):
        super().__init__()
        self.hidden_size, _, self.experts, self.topk = _geometry(config, draft)
        weight = weights["ffn.gate.weight"]
        _require_execution(weight)
        self.register_buffer(
            "weight",
            _tensor(
                weights,
                "ffn.gate.weight",
                (self.experts, self.hidden_size),
                torch.bfloat16,
                weight.device,
            ),
        )
        for name in ("bias", "bias_vl"):
            self.register_buffer(
                name,
                _tensor(
                    weights,
                    "ffn.gate." + name,
                    (self.experts,),
                    torch.float32,
                    weight.device,
                ),
            )

    @torch.inference_mode()
    def forward(self, hidden, image_mask=None):
        _require_execution(hidden)
        if (
            hidden.ndim != 2
            or hidden.shape[1] != self.hidden_size
            or hidden.dtype != torch.bfloat16
            or hidden.device != self.weight.device
            or not hidden.is_contiguous()
        ):
            raise ValueError(
                "V4.1 MoE needs contiguous BF16 [rows,5120] on its weight GPU"
            )
        if image_mask is not None and (
            image_mask.shape != hidden.shape[:1]
            or image_mask.dtype != torch.bool
            or image_mask.device != hidden.device
        ):
            raise ValueError(
                "V4.1 image mask must include every patch and delimiter row"
            )
        torch._assert_async(torch.isfinite(hidden).all(), "nonfinite V4.1 MoE input")
        return moe_gate(
            hidden, self.weight, self.bias, self.bias_vl, image_mask, self.topk
        )


def pack_v41_moe_weights(config, weights: Mapping, *, ep_size, ep_rank, draft=False):
    """Create only the rank-local raw stacks required by Mega's existing loader.

    Expert IDs remain global and ordered. Packing copies raw bytes; the
    strategy owns SF transformation and gate/up interleave. Input tensors and
    the source dictionary remain unchanged and may be released by their owner
    after successful installation.
    """
    from rtp_llm.utils.model_weight import W

    dim, inter, experts, _ = _geometry(config, draft)
    local = _expert_range(experts, ep_size, ep_rank)
    expected = {"ffn.gate." + name for name in ("weight", "bias", "bias_vl")}
    shapes = {"w1": (inter, dim), "w3": (inter, dim), "w2": (dim, inter)}
    for name in shapes:
        expected.update(
            f"ffn.experts.{expert}.{name}.{part}"
            for expert in local
            for part in ("weight", "scale")
        )
        expected.update(
            "ffn.shared_experts." + name + "." + part for part in ("weight", "scale")
        )
    actual = {name for name in weights if name.startswith("ffn.")}
    if actual != expected:
        raise ValueError(
            "V4.1 rank-local MoE inventory mismatch: "
            f"missing={sorted(expected - actual)[:8]}, unexpected={sorted(actual - expected)[:8]}"
        )
    device = weights["ffn.gate.weight"].device
    _require_execution(weights["ffn.gate.weight"])
    raw = {}
    for name, (rows, columns) in shapes.items():
        for expert in local:
            prefix = f"ffn.experts.{expert}.{name}."
            raw[prefix + "weight"] = _tensor(
                weights, prefix + "weight", (rows, columns // 2), torch.int8, device
            )
            raw[prefix + "scale"] = _tensor(
                weights,
                prefix + "scale",
                (rows, columns // 32),
                torch.float8_e8m0fnu,
                device,
            )
        prefix = "ffn.shared_experts." + name + "."
        raw[prefix + "weight"] = _tensor(
            weights, prefix + "weight", (rows, columns), torch.float8_e4m3fn, device
        )
        raw[prefix + "scale"] = _tensor(
            weights,
            prefix + "scale",
            (rows // 32, columns // 32),
            torch.float8_e8m0fnu,
            device,
        )
    for name, value in raw.items():
        if name.endswith(".scale"):
            torch._assert_async(
                (value.view(torch.uint8) != 255).all(), "nonfinite V4.1 expert scale"
            )
    packed = {}
    # float8 concatenation is not implemented by every supported Torch build.
    for name in shapes:
        for part, suffix in (("weight", "w"), ("scale", "s")):
            values = [raw[f"ffn.experts.{expert}.{name}.{part}"] for expert in local]
            packed[getattr(W, "v4_routed_" + name + "_" + suffix)] = torch.stack(
                [value.view(torch.uint8) for value in values]
            ).view(values[0].dtype)
    for part, suffix in (("weight", "w"), ("scale", "s")):
        values = [raw[f"ffn.shared_experts.{name}.{part}"] for name in ("w1", "w3")]
        packed[getattr(W, "v4_shared_w13_" + suffix)] = torch.cat(
            [value.view(torch.uint8) for value in values], dim=0
        ).view(values[0].dtype)
        packed[getattr(W, "v4_shared_w2_" + suffix)] = raw[
            "ffn.shared_experts.w2." + part
        ]
    return packed


class V41MoE(nn.Module):
    @classmethod
    def from_weights(
        cls,
        config,
        layer_id,
        weights,
        *,
        ep_size,
        ep_rank,
        max_tokens_per_rank,
        draft=False,
    ):
        import torch.distributed as dist

        from rtp_llm.models_py.modules.dsv4.moe.strategies import (
            MegaMoEStrategySE,
            MoeCfg,
        )

        dim, inter, experts, topk = _geometry(config, draft)
        local = _expert_range(experts, ep_size, ep_rank)
        if type(layer_id) is not int or not 0 <= layer_id < (3 if draft else 40):
            raise ValueError("V4.1 MoE layer does not belong to its target/draft stack")
        if type(max_tokens_per_rank) is not int or max_tokens_per_rank <= 0:
            raise ValueError("V4.1 MoE requires a positive startup token budget")
        router = V41MoERouter(config, weights, draft=draft)
        if ep_size == 16 and torch.cuda.get_device_capability(router.weight.device) != (
            10,
            0,
        ):
            raise ValueError("V4.1 EP16 is restricted to the GB200 NVLink deployment")
        if (
            not dist.is_initialized()
            or dist.get_world_size() != ep_size
            or dist.get_rank() != ep_rank
        ):
            raise ValueError("V4.1 MoE ownership must match its actual WORLD EP group")
        cfg = MoeCfg(
            layer_id=40 + layer_id if draft else layer_id,
            dim=dim,
            moe_inter_dim=inter,
            n_routed_experts=experts,
            n_activated_experts=topk,
            swiglu_limit=10.0,
            ep_size=ep_size,
            ep_rank=ep_rank,
            n_local_experts=len(local),
            local_expert_start=local.start,
            local_expert_end=local.stop,
            max_tokens_per_rank=max_tokens_per_rank,
            shared_fp8_block_size=32,
        )
        if not MegaMoEStrategySE.can_handle(cfg):
            raise RuntimeError("V4.1 requires the compatible shared32 Mega-SE backend")
        strategy = MegaMoEStrategySE(cfg)
        packed = pack_v41_moe_weights(
            config, weights, ep_size=ep_size, ep_rank=ep_rank, draft=draft
        )
        strategy.setup_weights(packed)
        if packed or not strategy.routed_includes_shared:
            raise RuntimeError(
                "V4.1 Mega-SE did not install the complete expert inventory"
            )
        return cls(router, strategy)

    def __init__(self, router, strategy):
        super().__init__()
        self.router = router
        self.strategy = strategy
        self.max_tokens_per_rank = strategy.cfg.max_tokens_per_rank

    @torch.inference_mode()
    def forward(self, hidden, image_mask=None):
        if hidden.ndim == 2 and hidden.shape[0] > self.max_tokens_per_rank:
            raise ValueError("V4.1 MoE rows exceed the startup token budget")
        weights, indices = self.router(hidden, image_mask)
        # The V4 fused gate path has no VL bias and uses a different FP32 contract.
        output = self.strategy(hidden, weights, indices)
        if (
            output.shape != hidden.shape
            or output.dtype != torch.bfloat16
            or output.device != hidden.device
        ):
            raise RuntimeError(
                "V4.1 Mega-SE changed the expert output geometry or dtype"
            )
        return output
