"""V4 Flash MXFP4 MegaMoE with model-owned symmetric workspace."""

import os
import torch

from .ppu_deepep_fp4 import PpuDeepEPFP4Strategy, take_raw_routed_mxfp4_weights


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

    def bind(self, cfg, wrapper, module):
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
                and os.environ.get("DSV4_PPU_MEGAMOE_TRANSPORT", "collective") == "collective"
                and os.environ.get("DSV4_PPU_EP16_EXACT_CAPACITY", "0") == "1"
            )
            if os.environ.get("DSV4_PPU_MEGAMOE_TRANSPORT") == "sparse" or exact_ep16_collective:
                capacity = min(capacity, logical_capacity)
            else:
                capacity = min(capacity, ((logical_capacity + 15) // 16) * 16)
        # The existing fused V4 gate pads six routes to eight. Keep its exact
        # order, scale bits and inactive -1/0 lanes through the MoE boundary.
        key = (id(group), cfg.n_routed_experts, capacity, 8, cfg.dim, cfg.moe_inter_dim)
        if self.buffer is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MegaMoE requires an eager warmup before graph capture"
                )
            experts = cfg.n_routed_experts
            if cfg.ep_size > 8:
                from .ppu_mega_hierarchy import PpuMegaMoEHierarchy

                self.hierarchy = PpuMegaMoEHierarchy(cfg, capacity)
                group = self.hierarchy.local_group
                experts = self.hierarchy.experts
                capacity *= self.hierarchy.nodes
            self.buffer = module.get_symm_buffer_for_mega_moe(
                group, experts, capacity, 8, cfg.dim, cfg.moe_inter_dim
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
        return super().can_handle(cfg) and cfg.ep_size in (8, 16, 32) and cfg.tp_size == 1 and (
            cfg.dim,
            cfg.moe_inter_dim,
            cfg.n_routed_experts,
            cfg.n_activated_experts,
        ) == (4096, 2048, 256, 6)

    def __init__(
        self, cfg, *, resources, module_name="deep_gemm_mega", **kwargs
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
        self._resources = resources
        self._sym = None

    def setup_weights(self, layer_weights):
        w13, s13, w2, s2 = take_raw_routed_mxfp4_weights(self.cfg, layer_weights)
        pair13, pair2 = self._mega.transform_weights_for_mega_moe(
            (w13, s13.view(torch.uint8)), (w2, s2.view(torch.uint8))
        )
        for name, value in zip(("_w13", "_s13", "_w2", "_s2"), (*pair13, *pair2)):
            self.register_buffer(name, value, persistent=False)

    def forward(self, x, weights, indices, *, active_token_mask=None):
        from rtp_llm.platforms.ppu.kernels.ppu_topk_padding import pad_inactive_routes

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
                1, token_count * self.cfg.n_activated_experts // self.cfg.n_local_experts + 1
            )
        self._mega.fp4_fp4_mega_moe(
            y,
            (self._w13, self._s13),
            (self._w2, self._s2),
            self._sym,
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            preserve_bf16=True,
            fuse_reduce=False,
            **scheduling,
        )
        return hierarchy.combine(y, token_count) if hierarchy is not None else y
