"""Opt-in Hopper GC MoE strategy for V4.1 Flash batch decode."""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from ..._profiler import record_function_range
from ..mega_sm90_buf import _sm90_shape_supported
from .base import MoeCfg
from .mega_sm90 import SM90MegaMoEStrategy

# All layers with the same EP geometry execute in order on the decode stream.
# Keep their symmetric workspace and output addresses stable across captures.
_BUFFERS: dict[tuple, tuple[object, torch.Tensor]] = {}
_SHARED_RECIPE = (1, 1, 32)


class SM90GCMoESEStrategy(SM90MegaMoEStrategy):
    name = "gc_sm90_se"
    routed_includes_shared = True

    @classmethod
    def can_handle(cls, cfg: MoeCfg, explicit: bool = False) -> bool:
        if not (
            cfg.is_decode_role
            and cfg.shared_fp8_block_size == 32
            and _sm90_shape_supported(
                cfg.dim,
                cfg.moe_inter_dim,
                cfg.n_routed_experts,
                cfg.ep_size,
                cfg.n_activated_experts,
            )
        ):
            return False
        try:
            from gc_moe import (
                get_symm_buffer_for_gc_moe_sm90,
                sm90_fp8_fp4_gc_moe,
                transform_weights_for_gc_moe_sm90,
            )

            from ..gc_sm90_input_packer import triton
        except (ImportError, OSError):
            return False
        return triton is not None and all(
            callable(api)
            for api in (
                get_symm_buffer_for_gc_moe_sm90,
                sm90_fp8_fp4_gc_moe,
                transform_weights_for_gc_moe_sm90,
            )
        )

    def _transform_routed_weights(self, l1, l2):
        from gc_moe import transform_weights_for_gc_moe_sm90

        return transform_weights_for_gc_moe_sm90(l1, l2)

    def setup_weights(self, layer_weights: dict) -> None:
        from gc_moe import transform_weights_for_gc_moe_sm90

        from rtp_llm.utils.model_weight import W

        super().setup_weights(layer_weights)

        def shared_pair(weight_key, scale_key):
            weight = layer_weights.pop(weight_key)
            scale = layer_weights.pop(scale_key)
            n, k = weight.shape
            if tuple(scale.shape) != (n // 32, k // 32):
                raise ValueError("SM90 GC MoE SE requires 32x32 shared weight scales")
            # The GC transform takes a scale for each output row and K32
            # group. Expand the checkpoint scales without changing FP8 bytes.
            return weight, scale.float().repeat_interleave(32, dim=0)

        self._shared_l1, self._shared_l2 = transform_weights_for_gc_moe_sm90(
            shared_pair(W.v4_shared_w13_w, W.v4_shared_w13_s),
            shared_pair(W.v4_shared_w2_w, W.v4_shared_w2_s),
            shared_recipe=_SHARED_RECIPE,
        )

    def setup_runtime(self) -> None:
        import gc_moe

        cfg = self.cfg
        group = dist.group.WORLD
        if group is None or group.size() != cfg.ep_size:
            raise RuntimeError("SM90 GC MoE SE requires the world group to match EP")
        capacity = max(cfg.max_tokens_per_rank, 1)
        device = self._mega_runtime_device
        key = (
            id(group),
            device,
            cfg.n_routed_experts,
            capacity,
            cfg.n_activated_experts,
            cfg.dim,
            cfg.moe_inter_dim,
        )
        if key not in _BUFFERS:
            buf = gc_moe.get_symm_buffer_for_gc_moe_sm90(
                group,
                cfg.n_routed_experts,
                capacity,
                cfg.n_activated_experts,
                cfg.dim,
                cfg.moe_inter_dim,
                num_shared_experts=1,
            )
            y = torch.empty(
                (buf.num_max_tokens_per_rank, cfg.dim),
                dtype=torch.bfloat16,
                device=device,
            )
            _BUFFERS[key] = (buf, y)
            logging.info(
                "SM90 GC MoE SE workspace ready: gc_moe=%s ep=%d capacity=%d",
                gc_moe.__version__,
                cfg.ep_size,
                buf.num_max_tokens_per_rank,
            )
        self._mega_buf, self._mega_y = _BUFFERS[key]
        self._l20z = torch.cuda.get_device_name(device) == "NVIDIA L20Z"

    def can_use_gate_pack_static(self, gate) -> bool:
        # Mega's fused gate writes only K128 inputs. GC also needs K32 inputs.
        return False

    def forward(self, x, weights, indices):
        from gc_moe import sm90_fp8_fp4_gc_moe

        from ..gc_sm90_input_packer import fused_pack_sm90_gc_se_inputs

        buf = self._mega_buf
        tokens = x.size(0)
        with record_function_range("dsv4.moe.gc_sm90_se_pack"):
            fused_pack_sm90_gc_se_inputs(x, weights, indices, buf)
        y = self._mega_y[:tokens]
        # The upstream L20Z schedule pipelines M<=16 and M>=32. The
        # collective bound selects geometry while the tensor shapes still
        # give the kernel the actual row count on each decode rank.
        bound = tokens
        if self._l20z and 16 < tokens < 32 and buf.num_max_tokens_per_rank >= 32:
            bound = 32
        sm90_fp8_fp4_gc_moe(
            y,
            self._mega_l1,
            self._mega_l2,
            buf,
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            # Decode graph pads every EP rank to the same bucket.
            num_tokens_bound=bound,
            shared_l1_weights=self._shared_l1,
            shared_l2_weights=self._shared_l2,
            shared_recipe=_SHARED_RECIPE,
        )
        return y
