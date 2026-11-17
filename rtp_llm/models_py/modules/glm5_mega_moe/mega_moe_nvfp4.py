"""Native NVFP4 routed experts using DeepGEMM 2.8.2 symmetric-memory MoE."""

import logging

import torch
from torch import nn

from .jit_warmup import mega_moe_jit_warmup_enabled
from .mega_buf import get_or_create_mega_output
from .mega_moe import GLM5MegaMoE

_BUFFERS = {}
_WARMED = set()


def _buffer(group, cfg, clone_key=None):
    import deep_gemm

    key = (
        id(group),
        cfg.n_routed_experts,
        cfg.max_tokens_per_rank,
        cfg.n_activated_experts,
        cfg.dim,
        cfg.moe_inter_dim,
        clone_key,
    )
    if key not in _BUFFERS:
        _BUFFERS[key] = deep_gemm.get_symm_buffer_for_mega_moe_nvfp4(
            group,
            cfg.n_routed_experts,
            cfg.max_tokens_per_rank,
            cfg.n_activated_experts,
            cfg.dim,
            cfg.moe_inter_dim,
        )
    return _BUFFERS[key]


def prepare_native_nvfp4_weights(w1, s1, g1, w2, s2, g2):
    """Reorder [up, gate] and pack scale bytes; preserve all checkpoint values."""
    import deep_gemm

    def reorder(t):
        up, gate = t.chunk(2, dim=1)
        return torch.cat((gate, up), dim=1).contiguous()

    e, two_inter, packed_hidden = w1.shape
    hidden, inter = packed_hidden * 2, two_inter // 2
    if (
        w1.dtype != torch.int8
        or w2.dtype != torch.int8
        or s1.dtype != torch.float8_e4m3fn
        or s2.dtype != torch.float8_e4m3fn
        or g1.dtype != torch.float32
        or g2.dtype != torch.float32
        or tuple(g1.shape) != (e, 2)
        or tuple(g2.shape) != (e,)
        or tuple(w2.shape) != (e, hidden, inter // 2)
        or tuple(s1.shape) != (e, two_inter, hidden // 16)
        or tuple(s2.shape) != (e, hidden, inter // 16)
        or hidden % 128
        or inter % 128
    ):
        raise ValueError("Invalid native NVFP4 expert weights/scales or dimensions")
    w1 = reorder(w1)
    s1 = reorder(s1.view(torch.uint8)).view(torch.float8_e4m3fn)
    g1 = reorder(g1)
    # The NVFP4 layout path consumes raw packed E4M3 bytes, never UE8M0.
    sf1 = deep_gemm.transform_sf_into_required_layout(
        s1.contiguous().view(torch.int32), two_inter, hidden, (1, 16), e
    )
    sf2 = deep_gemm.transform_sf_into_required_layout(
        s2.contiguous().view(torch.int32), hidden, inter, (1, 16), e
    )
    return deep_gemm.transform_weights_for_mega_moe_nvfp4(
        (w1, sf1, g1), (w2.contiguous(), sf2, g2.contiguous())
    )


def quantize_native_nvfp4_input(x):
    """DeepGEMM's per-token NVFP4 format without CPU copies during capture."""
    from deep_gemm.utils.math import cast_to_ue4m3_nearest, nvfp4_global_scale

    rows, hidden = x.shape
    values = x.float().reshape(rows, hidden // 16, 16)
    block_scale = values.abs().amax(dim=2) / 6.0
    gsf = nvfp4_global_scale(values.abs().amax(dim=(1, 2))).view(rows, 1)
    sf, sf_codes = cast_to_ue4m3_nearest(block_scale / gsf)
    scaled = (values * ((1.0 / gsf) / sf).unsqueeze(2)).clamp(-6.0, 6.0)
    magnitude = scaled.abs()
    codes = torch.zeros_like(scaled, dtype=torch.uint8)
    # Midpoint ties select the even E2M1 code. Preserve negative zero, too.
    for boundary, round_up in zip(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
        (False, True, False, True, False, True, False),
    ):
        codes += (magnitude >= boundary if round_up else magnitude > boundary).to(
            torch.uint8
        )
    codes |= torch.signbit(scaled).to(torch.uint8) << 3
    pairs = codes.reshape(rows, hidden // 2, 2)
    packed = pairs[:, :, 0] | (pairs[:, :, 1] << 4)
    return (
        packed.view(torch.int8),
        sf_codes.contiguous().view(torch.int32),
        gsf.view(rows),
    )


class GLM5MegaMoENVFP4(GLM5MegaMoE):
    def setup_weights_from_nvfp4(self, w1, s1, g1, w2, s2, g2):
        import deep_gemm
        import torch.distributed as dist

        if not hasattr(deep_gemm, "nvfp4_nvfp4_mega_moe"):
            raise RuntimeError(
                "Native NVFP4 MegaMoE requires DeepGEMM 2.8.2 with mega_nvfp4"
            )
        if not dist.is_initialized() or dist.get_world_size() != self.cfg.ep_size:
            raise ValueError(
                "Native NVFP4 MegaMoE requires WORLD to match the EP group"
            )
        if torch.cuda.get_device_capability(w1.device)[0] != 10:
            raise ValueError("Native NVFP4 MegaMoE requires SM100/SM103")
        self._l1, self._l2 = prepare_native_nvfp4_weights(w1, s1, g1, w2, s2, g2)
        self._mega_l1_w = self._l1[0]  # inherited warmup device selection
        self._mega_group = dist.group.WORLD
        self._mega_buf = _buffer(self._mega_group, self.cfg)
        self._mega_y = get_or_create_mega_output(
            self._mega_buf.num_max_tokens_per_rank,
            self.cfg.dim,
            torch.bfloat16,
            w1.device,
        )
        logging.info(
            "Native NVFP4 MegaMoE: nvfp4_nvfp4_mega_moe, block=16, layer=%d",
            self.cfg.layer_id,
        )
        key = (id(self._mega_buf), self.cfg.swiglu_limit)
        if mega_moe_jit_warmup_enabled() and key not in _WARMED:
            counts = self._resolve_jit_warmup_token_counts(deep_gemm.get_num_sms())
            if counts:
                self._warmup_jit(counts)
            _WARMED.add(key)

    def clone_for_cuda_graph(self, *, share_mega_buf=False):
        clone = object.__new__(type(self))
        nn.Module.__init__(clone)
        clone.cfg = self.cfg
        clone._l1, clone._l2 = self._l1, self._l2
        clone._mega_l1_w = self._mega_l1_w
        clone._mega_group = self._mega_group
        clone._mega_buf = (
            self._mega_buf
            if share_mega_buf
            else _buffer(self._mega_group, self.cfg, id(self._mega_buf))
        )
        clone._mega_y = torch.empty_like(self._mega_y)
        return clone

    def prepacked_input_views(self, tokens):
        raise RuntimeError("NVFP4 cannot consume the FP8 CMP prepack format")

    def forward(self, x, weights, indices, activation_clamp=None):
        import deep_gemm

        buf = self._mega_buf
        tokens = x.size(0)
        if x.dtype != torch.bfloat16 or tokens > buf.num_max_tokens_per_rank:
            raise ValueError("Native NVFP4 input must be BF16 within buffer capacity")
        packed, sf, gsf = quantize_native_nvfp4_input(x)
        buf.x[:tokens].copy_(packed)
        buf.x_sf[:tokens].copy_(sf)
        buf.x_gsf[:tokens].copy_(gsf)
        buf.topk_idx[:tokens].copy_(indices)
        buf.topk_weights[:tokens].copy_(weights)
        self._maybe_pre_kernel_barrier(tokens)
        y = self._mega_y[:tokens]
        deep_gemm.nvfp4_nvfp4_mega_moe(
            y,
            self._l1,
            self._l2,
            buf,
            recipe=(1, 1, 16),
            activation="swiglu",
            activation_clamp=activation_clamp,
            fast_math=False,
        )
        return y
