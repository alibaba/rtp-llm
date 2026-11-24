"""V4 Flash MXFP4 MegaMoE with model-owned symmetric workspace."""

import importlib
import logging
import os

import torch
from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_deepep_fp4 import (
    PpuDeepEPFP4Strategy,
    take_raw_routed_mxfp4_weights,
)
from torch import nn

_W4A8_FP32_CAPABILITY = {
    "output_dtype": "float32",
    "preweighted": True,
    "weight_dtype": "mxfp4_k32_ue8m0",
    "route_weight_stage": "before_l2_quant",
    "g1_rounding": "bf16",
    "activation_quantization": "block128_ue8m0_eps1e-4",
    "input_activation_scale": "block128_ue8m0_fp32_storage",
    "g2_activation_scale": "block128_ue8m0_fp32_storage",
}


class PpuTPMegaMoEW4A8Resources:
    """Two singleton workspaces cover all 256 TP-local experts."""

    def __init__(self):
        self._key = None
        self._buffers = None
        self._group = None

    def bind(self, cfg, module):
        import torch.distributed as dist

        key = (
            cfg.max_tokens_per_rank,
            8,
            cfg.dim,
            cfg.moe_inter_dim // cfg.tp_size,
        )
        if self._buffers is not None:
            if self._key != key:
                raise RuntimeError("TP MegaMoE layers require one workspace layout")
            return self._buffers
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TP MegaMoE requires eager workspace creation")
        world = dist.get_world_size()
        rank = dist.get_rank()
        groups = [dist.new_group([member]) for member in range(world)]
        self._group = groups[rank]
        args = (
            self._group,
            128,
            cfg.max_tokens_per_rank,
            8,
            cfg.dim,
            cfg.moe_inter_dim // cfg.tp_size,
        )
        self._buffers = tuple(
            module.get_symm_buffer_for_mega_moe(
                *args, mma_type="fp8xfp4", quantization="blockwise"
            )
            for _ in range(2)
        )
        self._key = key
        return self._buffers


class PpuTPMegaMoEW4A8Strategy(nn.Module):
    """Exact W4A8 MegaMoE over TP-sharded intermediate weights."""

    name = "ppu_tp_mega_moe_w4a8_fp32"
    preweights_route_scale = True

    def __init__(self, cfg, *, resources, module_name="deep_gemm"):
        super().__init__()
        if (
            cfg.tp_size != 4
            or cfg.ep_size != 1
            or (cfg.dim, cfg.moe_inter_dim, cfg.n_routed_experts) != (4096, 2048, 256)
            or cfg.n_activated_experts not in (6, 8)
        ):
            raise ValueError("TP MegaMoE W4A8 requires the frozen TP4/EP1 topology")
        self.cfg = cfg
        self.routed_tp_size = cfg.tp_size
        self._mega = importlib.import_module(module_name)
        capability = getattr(self._mega, "MEGA_MOE_CAPABILITIES", {}).get(
            "fp8_fp4_mega_moe_fp32"
        )
        required = (
            "get_symm_buffer_for_mega_moe",
            "transform_weights_for_w4a8_mega_moe",
            "mega_moe_pre_dispatch_ue8m0",
            "fp8_fp4_mega_moe_fp32",
        )
        if (
            not isinstance(capability, dict)
            or any(
                capability.get(key) != value
                for key, value in _W4A8_FP32_CAPABILITY.items()
            )
            or any(not callable(getattr(self._mega, name, None)) for name in required)
        ):
            raise RuntimeError("deep_gemm lacks the exact W4A8/FP32 MegaMoE contract")
        self._resources = resources
        self._sym_buffers = None

    def setup_weights(self, layer_weights):
        from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_grouped_fp4 import (
            _derive_inter_local_and_tp,
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
        w1, s1, w2, s2, w3, s3 = (layer_weights[key] for key in keys)
        inter_local, routed_tp_size = _derive_inter_local_and_tp(
            self.cfg,
            tuple(w1.shape),
            tuple(w2.shape),
            tuple(w3.shape),
            tuple(s1.shape),
            tuple(s2.shape),
            tuple(s3.shape),
        )
        if (
            routed_tp_size != self.cfg.tp_size
            or inter_local != self.cfg.moe_inter_dim // self.cfg.tp_size
        ):
            raise ValueError("TP MegaMoE requires TP-sharded routed weights")
        raw13 = torch.cat(
            (w1.view(torch.uint8), w3.view(torch.uint8)), dim=1
        ).contiguous()
        raw_s13 = torch.cat((s1, s3), dim=1).view(torch.uint8).contiguous()
        raw2 = w2.view(torch.uint8).contiguous()
        raw_s2 = s2.view(torch.uint8).contiguous()
        for bank, start in enumerate((0, 128)):
            pair13, pair2 = self._mega.transform_weights_for_w4a8_mega_moe(
                (raw13[start : start + 128], raw_s13[start : start + 128]),
                (raw2[start : start + 128], raw_s2[start : start + 128]),
            )
            for suffix, value in zip(
                ("w13", "s13", "inv13", "w2", "s2", "inv2"), (*pair13, *pair2)
            ):
                if isinstance(value, torch.Tensor):
                    self.register_buffer(f"_{suffix}_{bank}", value, persistent=False)
                else:
                    setattr(self, f"_{suffix}_{bank}", value)
        logging.info(
            "DSV4_PPU_TP_MOE_OPERATOR_PATH strategy=%s "
            "operator=deep_gemm.fp8_fp4_mega_moe_fp32 fast_math=false",
            self.name,
        )

    def _weights(self, bank):
        return (
            (
                getattr(self, f"_w13_{bank}"),
                getattr(self, f"_s13_{bank}"),
                getattr(self, f"_inv13_{bank}"),
            ),
            (
                getattr(self, f"_w2_{bank}"),
                getattr(self, f"_s2_{bank}"),
                getattr(self, f"_inv2_{bank}"),
            ),
        )

    def forward(
        self,
        x,
        weights,
        indices,
        *,
        shared=None,
        route_scale=None,
        shared_schedule=None,
    ):
        from rtp_llm.models_py.kernels.ppu.attention.ppu_topk_padding import (
            pad_inactive_routes,
        )

        if shared_schedule is not None:
            raise ValueError(
                "TP MegaMoE exact path does not support phased shared execution"
            )
        if shared is not None and route_scale is None:
            raise ValueError("shared combine requires route_scale")
        if self._sym_buffers is None:
            self._sym_buffers = self._resources.bind(self.cfg, self._mega)
        if indices.shape[1] != 8:
            indices, weights = pad_inactive_routes(indices, weights, 8)
        routed = []
        scaled = weights * (1.0 if route_scale is None else float(route_scale))
        for bank, start in enumerate((0, 128)):
            valid = (indices >= start) & (indices < start + 128)
            route_ids = torch.where(valid, indices - start, -1).contiguous()
            route_weights = torch.where(valid, scaled, 0.0).contiguous()
            buffer = self._sym_buffers[bank]
            self._mega.mega_moe_pre_dispatch_ue8m0(
                x.contiguous(),
                route_ids,
                route_weights,
                buffer.x,
                buffer.x_sf,
                buffer.topk_idx,
                buffer.topk_weights,
                x.shape[0],
            )
            y = torch.empty_like(x, dtype=torch.float32)
            w13, w2 = self._weights(bank)
            self._mega.fp8_fp4_mega_moe_fp32(
                y,
                w13,
                w2,
                buffer,
                activation_clamp=(
                    self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
                ),
                fast_math=False,
            )
            routed.append(y)
        result = routed[0] + routed[1]
        if shared is None:
            return result
        from rtp_llm.models_py.kernels.ppu.moe.ppu_moe_combine import (
            combine_tp_partials,
        )

        return combine_tp_partials(result.contiguous(), shared, 1.0)


class PpuMegaMoEResources:
    """One serial workspace shared by this model's routed expert layers.

    Shared experts run on their existing stream and never access this storage.
    Warmup creates the allocation before any CUDA Graph capture; every graph
    keeps the model/resources alive and replays on the model's compute stream.
    Concurrent execution of two forwards of the same model is unsupported.
    """

    def __init__(self):
        self._key = None
        self.buffer = None
        self.hierarchy = None

    def bind(self, cfg, wrapper, module, *, mma_type="fp4xfp4"):
        import torch.distributed as dist

        group = dist.group.WORLD
        if (
            group is None
            or dist.get_world_size(group) != cfg.ep_size
            or dist.get_rank(group) != cfg.ep_rank
        ):
            raise RuntimeError("MegaMoE must use the engine's EP WORLD group")
        capacity = wrapper.ll_num_max_token_per_rank
        if cfg.ep_size > 8:
            # The masked-GEMM minimum of 64 does not apply to MegaMoE. Keep a
            # rank-independent bound, including all speculative verification rows.
            logical_capacity = wrapper.config.ll_num_max_token
            if logical_capacity <= 0:
                raise ValueError("MegaMoE requires a positive decode token capacity")
            logical_capacity = (logical_capacity + cfg.tp_size - 1) // cfg.tp_size
            # Native symmetric storage aligns internally; RDMA packets can keep
            # the rank-independent logical bound without transmitting padding.
            exact_ep16_collective = (
                cfg.ep_size == 16
                and cfg.tp_size == 1
                and os.environ.get("DSV4_PPU_MEGAMOE_TRANSPORT", "collective")
                == "collective"
                and os.environ.get("DSV4_PPU_EP16_EXACT_CAPACITY", "0") == "1"
            )
            if (
                os.environ.get("DSV4_PPU_MEGAMOE_TRANSPORT") == "sparse"
                or exact_ep16_collective
            ):
                capacity = min(capacity, logical_capacity)
            else:
                capacity = min(capacity, ((logical_capacity + 15) // 16) * 16)
        # The existing fused V4 gate pads six routes to eight. Keep its exact
        # order, scale bits and inactive -1/0 lanes through the MoE boundary.
        key = (
            id(group),
            cfg.n_routed_experts,
            capacity,
            8,
            cfg.dim,
            cfg.moe_inter_dim,
            mma_type,
        )
        if self.buffer is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MegaMoE requires an eager warmup before graph capture"
                )
            experts = cfg.n_routed_experts
            if cfg.ep_size > 8:
                from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_mega_hierarchy import (
                    PpuMegaMoEHierarchy,
                )

                self.hierarchy = PpuMegaMoEHierarchy(cfg, capacity)
                group = self.hierarchy.local_group
                experts = self.hierarchy.experts
                capacity *= self.hierarchy.nodes
            args = (group, experts, capacity, 8, cfg.dim, cfg.moe_inter_dim)
            self.buffer = (
                module.get_symm_buffer_for_mega_moe(*args)
                if mma_type == "fp4xfp4"
                else module.get_symm_buffer_for_mega_moe(
                    *args, mma_type=mma_type, quantization="blockwise"
                )
            )
            self._key = key
        elif self._key != key:
            raise RuntimeError(
                "MegaMoE layers cannot share different workspace layouts"
            )
        return self.buffer


class PpuMegaMoEFP4Strategy(PpuDeepEPFP4Strategy):
    """EP8 MegaMoE, with hierarchical RDMA transport for EP16/EP32."""

    name = "ppu_mega_moe_fp4"

    @classmethod
    def can_handle(cls, cfg):
        return (
            super().can_handle(cfg)
            and cfg.ep_size in (8, 16, 32)
            and cfg.tp_size == 1
            and (
                cfg.dim,
                cfg.moe_inter_dim,
                cfg.n_routed_experts,
                cfg.n_activated_experts,
            )
            == (4096, 2048, 256, 6)
        )

    def __init__(
        self, cfg, *, resources, module_name="deep_gemm_mega", streamed=False, **kwargs
    ):
        kwargs.setdefault("output_dtype", torch.bfloat16)
        if kwargs["output_dtype"] != torch.bfloat16:
            raise ValueError("PPU MegaMoE currently requires BF16 routed output")
        super().__init__(cfg, **kwargs)
        import importlib

        if module_name != "deep_gemm_mega":
            raise ValueError("MegaMoE package must be deep_gemm_mega")
        deep_gemm = importlib.import_module(module_name)

        required = (
            "get_symm_buffer_for_mega_moe",
            "transform_weights_for_mega_moe",
            "mega_moe_pre_dispatch",
            "fp4_fp4_mega_moe",
        )
        if any(not callable(getattr(deep_gemm, name, None)) for name in required):
            raise RuntimeError(
                "PPU MegaMoE requires a DeepGEMM build with the MegaMoE API"
            )
        self._mega = deep_gemm
        self._streamed = bool(streamed)
        if self._streamed and not callable(
            getattr(deep_gemm, "fp4_fp4_mega_moe_streamed", None)
        ):
            raise RuntimeError("Streamed MegaMoE requires the DeepGEMM streamed API")
        self._resources = resources
        self._sym = None

    def setup_weights(self, layer_weights):
        w13, s13, w2, s2 = take_raw_routed_mxfp4_weights(self.cfg, layer_weights)
        pair13, pair2 = self._mega.transform_weights_for_mega_moe(
            (w13, s13.view(torch.uint8)), (w2, s2.view(torch.uint8))
        )
        for name, value in zip(("_w13", "_s13", "_w2", "_s2"), (*pair13, *pair2)):
            self.register_buffer(name, value, persistent=False)

    def require_staged_shared(self):
        import inspect

        if (
            "after_gemm1"
            not in inspect.signature(self._mega.fp4_fp4_mega_moe).parameters
        ):
            raise RuntimeError(
                "Staged shared execution requires DeepGEMM's after_gemm1 API"
            )

    def forward(self, x, weights, indices, *, active_token_mask=None, after_gemm1=None):
        from rtp_llm.models_py.kernels.ppu.attention.ppu_topk_padding import (
            pad_inactive_routes,
        )

        if self._streamed and after_gemm1 is not None:
            raise ValueError("Streamed MegaMoE does not support after_gemm1")
        if self._sym is None:
            wrapper = self._bind_wrapper()
            self._sym = self._resources.bind(self.cfg, wrapper, self._mega)
        if indices.shape[1] == 8:
            route_ids, route_weights = indices.contiguous(), weights.contiguous()
        else:
            route_ids, route_weights = pad_inactive_routes(indices, weights, 8)
        token_count = x.shape[0]
        hierarchy = self._resources.hierarchy if self.cfg.ep_size > 8 else None
        if hierarchy is not None:
            if self._streamed or after_gemm1 is not None:
                raise ValueError(
                    "Hierarchical MegaMoE supports neither streamed nor staged execution"
                )
            x, route_weights, route_ids = hierarchy.dispatch(
                x, route_weights, route_ids, active_token_mask
            )
            active_token_mask = None
        self._mega.mega_moe_pre_dispatch(
            x.contiguous(),
            route_ids,
            route_weights,
            sym_buffer=self._sym,
            active_token_mask=active_token_mask,
        )
        y = torch.empty_like(x, dtype=torch.bfloat16)
        scheduling = {}
        if hierarchy is not None:
            # Dispatch padding and node-masked routes are not expert work. This
            # hint only selects tiles; actual routed counts still drive execution.
            scheduling["expected_m"] = max(
                1,
                token_count * self.cfg.n_activated_experts // self.cfg.n_local_experts
                + 1,
            )
        self._launch_routed_gemm(y, scheduling, after_gemm1)
        return hierarchy.combine(y, token_count) if hierarchy is not None else y

    # The packaged streamed kernel accepts one decode tile shape only and the
    # binding reports a mismatch through a plain RuntimeError message.
    _STREAMED_TILE_ERROR = "streamed MegaMoE requires"

    def _launch_routed_gemm(self, y, scheduling, after_gemm1):
        weights = ((self._w13, self._s13), (self._w2, self._s2), self._sym)
        extra = dict(
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            preserve_bf16=True,
        )
        if self._streamed:
            try:
                self._mega.fp4_fp4_mega_moe_streamed(y, *weights, **extra, **scheduling)
                return
            except RuntimeError as err:
                if self._STREAMED_TILE_ERROR not in str(err):
                    raise
                # A decode tile the streamed kernel does not cover must degrade
                # to the plain kernel instead of aborting engine startup.
                logging.warning(
                    "[ppu-megamoe] streamed kernel rejected the decode tile shape, "
                    "falling back to fp4_fp4_mega_moe: %s",
                    err,
                )
                self._streamed = False
        fallback = dict(scheduling, fuse_reduce=False)
        if after_gemm1 is not None:
            fallback["after_gemm1"] = after_gemm1
        self._mega.fp4_fp4_mega_moe(y, *weights, **extra, **fallback)


_FP8_FP32_CAPABILITY = {
    "output_dtype": "float32",
    "preweighted": True,
    "route_weight_stage": "before_l2_quant",
    "g1_rounding": "bf16",
    "activation_quantization": "block128_ue8m0_eps1e-4",
    "input_activation_scale": "block128_ue8m0_fp32_storage",
    "g2_activation_scale": "block128_ue8m0_fp32_storage",
}


def promote_mxfp4_to_fp8(raw_weight, raw_scale):
    """Losslessly promote checkpoint MXFP4 into canonical block-128 E4M3."""
    if raw_weight.ndim != 3 or raw_scale.ndim != 3:
        raise ValueError("Routed MXFP4 promotion requires [E,N,K/2] weights and scales")
    if raw_weight.dtype not in (torch.uint8, torch.int8):
        raise ValueError("Routed MXFP4 promotion requires packed byte weights")
    e, n, packed_k = raw_weight.shape
    k = packed_k * 2
    if n % 128 or k % 128 or raw_scale.shape != (e, n, k // 32):
        raise ValueError(
            "Routed MXFP4 promotion requires block-128-compatible geometry"
        )
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=raw_weight.device,
    )
    promoted, scales = [], []
    for expert in range(e):
        packed = raw_weight[expert].view(torch.uint8)
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(n, k).long()
        dense = lut[codes] * raw_scale[expert].float().repeat_interleave(32, dim=-1)
        blocks = dense.reshape(n // 128, 128, k // 128, 128)
        maximum = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-30)
        scale = torch.exp2(torch.ceil(torch.log2(maximum / 448.0)))
        weight = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
        if not torch.equal(weight.float() * scale[:, None, :, None], blocks):
            raise RuntimeError(
                "checkpoint MXFP4 is not losslessly representable by FP8 promotion"
            )
        promoted.append(weight.reshape(n, k).contiguous())
        scales.append(scale.contiguous())
    return torch.stack(promoted), torch.stack(scales)


class PpuMegaMoEFP8Strategy(PpuDeepEPFP4Strategy):
    """Capability-gated Route A: FP8 experts and FP32 route reduction."""

    name = "ppu_mega_moe_fp8_fp32"

    @classmethod
    def can_handle(cls, cfg):
        return (
            super().can_handle(cfg)
            and cfg.ep_size == 8
            and cfg.tp_size == 1
            and (
                cfg.dim,
                cfg.moe_inter_dim,
                cfg.n_routed_experts,
                cfg.n_activated_experts,
            )
            == (4096, 2048, 256, 6)
        )

    @staticmethod
    def package_has_exact_capability(module):
        capability = getattr(module, "MEGA_MOE_CAPABILITIES", {}).get(
            "fp8_fp8_mega_moe_fp32"
        )
        required = (
            "get_symm_buffer_for_mega_moe",
            "transform_weights_for_fp8_mega_moe",
            "mega_moe_pre_dispatch_ue8m0",
            "fp8_fp8_mega_moe_fp32",
        )
        return (
            isinstance(capability, dict)
            and all(
                capability.get(key) == value
                for key, value in _FP8_FP32_CAPABILITY.items()
            )
            and all(callable(getattr(module, name, None)) for name in required)
        )

    @classmethod
    def package_is_available(cls, module_name="deep_gemm_route_a"):
        if module_name != "deep_gemm_route_a":
            return False
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError):
            return False
        return cls.package_has_exact_capability(module)

    def __init__(self, cfg, *, resources, module_name="deep_gemm_route_a", **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected exact MegaMoE options: {sorted(kwargs)}")
        super().__init__(cfg, output_dtype=torch.float32)
        if module_name != "deep_gemm_route_a":
            raise ValueError("Exact MegaMoE package must be isolated Route A")
        self._mega = importlib.import_module(module_name)
        if not self.package_has_exact_capability(self._mega):
            raise RuntimeError("MegaMoE package lacks the exact FP8/FP32 capability")
        self._resources = resources
        self._sym = None

    def setup_weights(self, layer_weights):
        w13, s13, w2, s2 = take_raw_routed_mxfp4_weights(self.cfg, layer_weights)
        pair13 = promote_mxfp4_to_fp8(w13, s13)
        pair2 = promote_mxfp4_to_fp8(w2, s2)
        pair13, pair2 = self._mega.transform_weights_for_fp8_mega_moe(
            pair13, pair2, quantization="blockwise"
        )
        for name, value in zip(("_w13", "_s13", "_w2", "_s2"), (*pair13, *pair2)):
            self.register_buffer(name, value, persistent=False)

    def build_shared_expert(self, expert_weights, swiglu_limit):
        from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_shared_expert import (
            PpuMegaMoEFP8SharedExpert,
        )

        return PpuMegaMoEFP8SharedExpert(
            self.cfg, expert_weights, swiglu_limit=swiglu_limit, module=self._mega
        )

    def forward(self, x, weights, indices, *, active_token_mask=None):
        from rtp_llm.models_py.kernels.ppu.attention.ppu_topk_padding import (
            pad_inactive_routes,
        )

        if self._sym is None:
            wrapper = self._bind_wrapper()
            self._sym = self._resources.bind(
                self.cfg, wrapper, self._mega, mma_type="fp8xfp8"
            )
        if indices.shape[1] != 8 or active_token_mask is not None:
            padded_indices = (
                indices if indices.dtype == torch.int64 else indices.to(torch.int64)
            )
            route_ids, route_weights = pad_inactive_routes(
                padded_indices, weights, 8, active_token_mask
            )
        else:
            route_ids, route_weights = indices, weights
        self._mega.mega_moe_pre_dispatch_ue8m0(
            x.contiguous(),
            route_ids.contiguous(),
            route_weights.contiguous(),
            self._sym.x,
            self._sym.x_sf,
            self._sym.topk_idx,
            self._sym.topk_weights,
            x.shape[0],
        )
        y = torch.empty_like(x, dtype=torch.float32)
        self._mega.fp8_fp8_mega_moe_fp32(
            y,
            (self._w13, self._s13),
            (self._w2, self._s2),
            self._sym,
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            fast_math=True,
        )
        return y


class PpuMegaMoEW4A8Strategy(PpuDeepEPFP4Strategy):
    """EP8 exact W4A8: MXFP4 experts with Route A's UE8M0 activation contract.

    Bit-exactness against Route A comes from keeping the activation side
    identical (1x128 E4M3 + UE8M0 eps 1e-4, per-K128 FP32 rescale) while the
    B side consumes the checkpoint nibbles directly; MXFP4 values are exactly
    representable in E4M3, so every product is unchanged.
    """

    name = "ppu_mega_moe_w4a8_fp32"

    @classmethod
    def can_handle(cls, cfg):
        return (
            super().can_handle(cfg)
            and cfg.ep_size == 8
            and cfg.tp_size == 1
            and (
                cfg.dim,
                cfg.moe_inter_dim,
                cfg.n_routed_experts,
                cfg.n_activated_experts,
            )
            == (4096, 2048, 256, 6)
        )

    @staticmethod
    def package_has_exact_capability(module):
        capability = getattr(module, "MEGA_MOE_CAPABILITIES", {}).get(
            "fp8_fp4_mega_moe_fp32"
        )
        required = (
            "get_symm_buffer_for_mega_moe",
            "transform_weights_for_w4a8_mega_moe",
            "mega_moe_pre_dispatch_ue8m0",
            "fp8_fp4_mega_moe_fp32",
        )
        return (
            isinstance(capability, dict)
            and all(
                capability.get(key) == value
                for key, value in _W4A8_FP32_CAPABILITY.items()
            )
            and all(callable(getattr(module, name, None)) for name in required)
        )

    @classmethod
    def package_is_available(cls, module_name="deep_gemm_route_a"):
        if module_name != "deep_gemm_route_a":
            return False
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError):
            return False
        return cls.package_has_exact_capability(module)

    def __init__(
        self,
        cfg,
        *,
        resources,
        module_name="deep_gemm_route_a",
        max_tokens=96,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unexpected exact W4A8 options: {sorted(kwargs)}")
        super().__init__(cfg, output_dtype=torch.float32)
        if module_name != "deep_gemm_route_a":
            raise ValueError("Exact W4A8 package must be the isolated Route A build")
        self._mega = importlib.import_module(module_name)
        if not self.package_has_exact_capability(self._mega):
            raise RuntimeError("MegaMoE package lacks the exact W4A8/FP32 capability")
        self._max_tokens = int(max_tokens)
        if self._max_tokens <= 0:
            raise ValueError("W4A8 MegaMoE token cap must be positive")
        self._resources = resources
        self._sym = None

    def setup_weights(self, layer_weights):
        w13, s13, w2, s2 = take_raw_routed_mxfp4_weights(self.cfg, layer_weights)
        pair13, pair2 = self._mega.transform_weights_for_w4a8_mega_moe(
            (w13, s13.view(torch.uint8)), (w2, s2.view(torch.uint8))
        )
        for suffix, value in zip(
            ("w13", "s13", "inv13", "w2", "s2", "inv2"), (*pair13, *pair2)
        ):
            if isinstance(value, torch.Tensor):
                self.register_buffer(f"_{suffix}", value, persistent=False)
            else:
                setattr(self, f"_{suffix}", value)

    def build_shared_expert(self, expert_weights, swiglu_limit):
        from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_shared_expert import (
            PpuMegaMoEFP8SharedExpert,
        )

        return PpuMegaMoEFP8SharedExpert(
            self.cfg, expert_weights, swiglu_limit=swiglu_limit, module=self._mega
        )

    def require_staged_shared(self):
        raise RuntimeError("W4A8 MegaMoE does not support staged shared execution")

    def forward(
        self,
        x,
        weights,
        indices,
        *,
        active_token_mask=None,
        after_gemm1=None,
    ):
        from rtp_llm.models_py.kernels.ppu.attention.ppu_topk_padding import (
            pad_inactive_routes,
        )

        if after_gemm1 is not None:
            raise ValueError("W4A8 MegaMoE does not support staged shared execution")
        if x.shape[0] > self._max_tokens:
            raise ValueError(
                "W4A8 MegaMoE input exceeds DSV4_PPU_MEGAMOE_W4A8_MAX_TOKENS="
                f"{self._max_tokens}"
            )
        if self._sym is None:
            wrapper = self._bind_wrapper()
            self._sym = self._resources.bind(
                self.cfg, wrapper, self._mega, mma_type="fp8xfp4"
            )
        # The exact kernel was qualified with int64 global expert ids; keep the
        # inactive -1/0 lanes from pad_inactive_routes untouched.
        route_ids = indices if indices.dtype == torch.int64 else indices.to(torch.int64)
        if route_ids.shape[1] != 8 or active_token_mask is not None:
            route_ids, route_weights = pad_inactive_routes(
                route_ids, weights, 8, active_token_mask
            )
        else:
            route_weights = weights
        self._mega.mega_moe_pre_dispatch_ue8m0(
            x.contiguous(),
            route_ids.contiguous(),
            route_weights.contiguous(),
            self._sym.x,
            self._sym.x_sf,
            self._sym.topk_idx,
            self._sym.topk_weights,
            x.shape[0],
        )
        y = torch.empty_like(x, dtype=torch.float32)
        self._mega.fp8_fp4_mega_moe_fp32(
            y,
            (self._w13, self._s13, self._inv13),
            (self._w2, self._s2, self._inv2),
            self._sym,
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            fast_math=False,
        )
        return y
