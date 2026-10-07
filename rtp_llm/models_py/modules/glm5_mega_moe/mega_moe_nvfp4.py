"""MegaMoE NVFP4xNVFP4 strategy backed by DeepGEMM."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from .mega_buf import get_or_create_mega_output
from .mega_moe import (
    GLM5MegaMoE,
    GLM5MegaMoeCfg,
    _activation_clamp_or_none,
    _mega_output_capacity,
    _sync_cuda_graph_warmup_ranks,
)
from .mega_nvfp4_buf import get_or_create_mega_buf_nvfp4
from .mega_nvfp4_input_packer import get_mega_nvfp4_input_packer

logger = logging.getLogger(__name__)

NVFP4_BLOCK = 16


def _pack_raw_ue4m3_scales(
    scale: torch.Tensor,
    *,
    groups: int,
    mn: int,
    k: int,
) -> torch.Tensor:
    """Pack raw positive E4M3 scale bytes and apply DeepGEMM TMA layout."""
    import deep_gemm

    expected = (groups, mn, k // NVFP4_BLOCK)
    if scale.dtype != torch.float8_e4m3fn or tuple(scale.shape) != expected:
        raise TypeError(
            "NVFP4 block scale must be raw E4M3 "
            f"{expected}, got dtype={scale.dtype} shape={tuple(scale.shape)}"
        )
    if scale.shape[-1] % 4 != 0:
        raise ValueError(
            f"NVFP4 scale K groups must be divisible by 4, got {scale.shape[-1]}"
        )
    packed = scale.contiguous().view(torch.int32)
    return deep_gemm.transform_sf_into_required_layout(
        packed, mn, k, (1, NVFP4_BLOCK), groups
    )


class GLM5MegaMoENVFP4(GLM5MegaMoE):
    """Shape-generic routed MoE using ``nvfp4_nvfp4_mega_moe``."""

    def __init__(self, cfg: GLM5MegaMoeCfg):
        super().__init__(cfg)
        self._mega_l1_gsf: Optional[torch.Tensor] = None
        self._mega_l2_gsf: Optional[torch.Tensor] = None

    def clone_for_cuda_graph(self) -> "GLM5MegaMoENVFP4":
        clone = object.__new__(type(self))
        torch.nn.Module.__init__(clone)
        clone.cfg = self.cfg
        clone._mega_l1_w = self._mega_l1_w
        clone._mega_l1_sf = self._mega_l1_sf
        clone._mega_l1_gsf = self._mega_l1_gsf
        clone._mega_l2_w = self._mega_l2_w
        clone._mega_l2_sf = self._mega_l2_sf
        clone._mega_l2_gsf = self._mega_l2_gsf
        clone._mega_buf = self._mega_buf
        clone._mega_y = (
            torch.empty_like(self._mega_y) if self._mega_y is not None else None
        )
        clone._input_packer = get_mega_nvfp4_input_packer()
        clone._mega_group = self._mega_group
        clone._activation_name = self._activation_name
        return clone

    def setup_weights_from_nvfp4(
        self,
        w1_w: torch.Tensor,
        w1_s: torch.Tensor,
        w1_inverse_gsf: torch.Tensor,
        w2_w: torch.Tensor,
        w2_s: torch.Tensor,
        w2_inverse_gsf: torch.Tensor,
    ) -> None:
        """Prepare MiniMax packed NVFP4 routed weights for DeepGEMM.

        RTP's W1 tensors arrive as ``[up | gate]``. Checkpoint global scales
        are multiplicative quantization scales (the inverse of DeepGEMM's
        dequant GSF), so this method both restacks W1 and takes reciprocals.
        """
        import deep_gemm

        cfg = self.cfg
        e = cfg.n_local_experts
        d = cfg.dim
        inter = cfg.moe_inter_dim
        device = (
            w1_w.device
            if w1_w.is_cuda
            else torch.device("cuda", torch.cuda.current_device())
        )
        tensors = (w1_w, w1_s, w1_inverse_gsf, w2_w, w2_s, w2_inverse_gsf)
        with torch.cuda.device(device):
            (
                w1_w,
                w1_s,
                w1_inverse_gsf,
                w2_w,
                w2_s,
                w2_inverse_gsf,
            ) = tuple(t.to(device=device, non_blocking=True) for t in tensors)

        expected_w1 = (e, 2 * inter, d // 2)
        expected_s1 = (e, 2 * inter, d // NVFP4_BLOCK)
        expected_g1 = (e, 2)
        expected_w2 = (e, d, inter // 2)
        expected_s2 = (e, d, inter // NVFP4_BLOCK)
        expected_g2 = (e,)
        for name, tensor, dtype, shape in (
            ("w1", w1_w, torch.int8, expected_w1),
            ("w1_scale", w1_s, torch.float8_e4m3fn, expected_s1),
            ("w1_global_scale", w1_inverse_gsf, torch.float32, expected_g1),
            ("w2", w2_w, torch.int8, expected_w2),
            ("w2_scale", w2_s, torch.float8_e4m3fn, expected_s2),
            ("w2_global_scale", w2_inverse_gsf, torch.float32, expected_g2),
        ):
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise TypeError(
                    f"NVFP4 {name} must be {dtype} {shape}, got "
                    f"{tensor.dtype} {tuple(tensor.shape)}"
                )

        # Standard RTP layout is [up | gate]; DeepGEMM's transform takes
        # [gate | up] and performs its own 8-row interleave afterward.
        w1_up, w1_gate = w1_w.chunk(2, dim=1)
        s1_up, s1_gate = w1_s.chunk(2, dim=1)
        w13 = torch.cat([w1_gate, w1_up], dim=1).contiguous()
        s13_raw = torch.cat([s1_gate, s1_up], dim=1).contiguous()
        inverse_gsf13 = torch.stack(
            [w1_inverse_gsf[:, 1], w1_inverse_gsf[:, 0]], dim=1
        ).contiguous()
        del w1_w, w1_s, w1_up, w1_gate, s1_up, s1_gate, w1_inverse_gsf

        if not bool(torch.isfinite(inverse_gsf13).all()) or bool(
            (inverse_gsf13 <= 0).any()
        ):
            raise ValueError(
                "NVFP4 L1 inverse global scales must be finite and positive"
            )
        if not bool(torch.isfinite(w2_inverse_gsf).all()) or bool(
            (w2_inverse_gsf <= 0).any()
        ):
            raise ValueError(
                "NVFP4 L2 inverse global scales must be finite and positive"
            )
        l1_gsf_raw = inverse_gsf13.reciprocal().contiguous()
        l2_gsf_raw = w2_inverse_gsf.reciprocal().contiguous()
        del inverse_gsf13, w2_inverse_gsf

        logger.info(
            "[MegaMoE NVFP4] preparing weights: layer=%d E=%d D=%d inter=%d",
            cfg.layer_id,
            e,
            d,
            inter,
        )
        with torch.cuda.device(device):
            s13 = _pack_raw_ue4m3_scales(s13_raw, groups=e, mn=2 * inter, k=d)
            s2 = _pack_raw_ue4m3_scales(w2_s, groups=e, mn=d, k=inter)
            (l1_w, l1_sf, l1_gsf), (l2_w, l2_sf, l2_gsf) = (
                deep_gemm.transform_weights_for_mega_moe_nvfp4(
                    (w13, s13, l1_gsf_raw),
                    (w2_w.contiguous(), s2, l2_gsf_raw),
                )
            )
        del w13, s13_raw, s13, l1_gsf_raw, w2_w, w2_s, s2, l2_gsf_raw
        torch.cuda.empty_cache()

        self._mega_l1_w = l1_w
        self._mega_l1_sf = l1_sf
        self._mega_l1_gsf = l1_gsf
        self._mega_l2_w = l2_w
        self._mega_l2_sf = l2_sf
        self._mega_l2_gsf = l2_gsf
        self._setup_buffer_and_warmup()

    def setup_weights_from_fp4(self, *args, **kwargs) -> None:
        raise ValueError("moe_strategy=mega_moe_nvfp4 only accepts NVFP4 weights")

    def setup_weights_from_fp8(self, *args, **kwargs) -> None:
        raise ValueError("moe_strategy=mega_moe_nvfp4 only accepts NVFP4 weights")

    def setup_weights_from_bf16(self, *args, **kwargs) -> None:
        raise ValueError("moe_strategy=mega_moe_nvfp4 only accepts NVFP4 weights")

    def _setup_buffer_and_warmup(self) -> None:
        import torch.distributed as dist

        cfg = self.cfg
        device = self._mega_l1_w.device
        if self._activation_name != "swiglu_oai":
            raise ValueError(
                "moe_strategy=mega_moe_nvfp4 requires MiniMax SwiGLU-OAI "
                "with positive alpha and clamp"
            )
        assert dist.is_initialized(), "MegaMoE NVFP4 requires torch.distributed"
        self._mega_group = dist.group.WORLD
        self._mega_buf = get_or_create_mega_buf_nvfp4(
            group=self._mega_group,
            num_experts=cfg.n_routed_experts,
            num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
            num_topk=cfg.n_activated_experts,
            hidden=cfg.dim,
            intermediate_hidden=cfg.moe_inter_dim,
            activation=self._activation_name,
        )
        self._mega_y = get_or_create_mega_output(
            _mega_output_capacity(self._mega_buf, cfg.max_tokens_per_rank),
            cfg.dim,
            torch.bfloat16,
            device,
        )
        self._input_packer = get_mega_nvfp4_input_packer()
        self._maybe_warmup_jit_once()

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        activation: Optional[str] = None,
        extra_expert_args: Optional[Dict[str, Any]] = None,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        import deep_gemm

        activation_name = (activation or self._activation_name).lower()
        if activation_name != "swiglu_oai":
            raise ValueError(
                "moe_strategy=mega_moe_nvfp4 requires activation='swiglu_oai', "
                f"got {activation!r}"
            )
        alpha = float(
            (extra_expert_args or {}).get("swiglu_alpha", self.cfg.swiglu_alpha)
        )
        limit = float(
            (extra_expert_args or {}).get("swiglu_limit", self.cfg.swiglu_limit)
        )
        if alpha <= 0.0 or limit <= 0.0:
            raise ValueError("swiglu_oai requires positive swiglu_alpha and limit")

        tokens = x.size(0)
        buf = self._mega_buf
        if tokens > buf.num_max_tokens_per_rank:
            raise RuntimeError(
                f"MegaMoE NVFP4 input tokens={tokens} exceeds "
                f"num_max_tokens_per_rank={buf.num_max_tokens_per_rank}"
            )
        if tokens > self._mega_y.size(0):
            raise RuntimeError(
                f"MegaMoE NVFP4 output rows={self._mega_y.size(0)} is smaller "
                f"than input tokens={tokens}"
            )

        if out is not None:
            # DeepGEMM writes only y.size(0) rows, including a partial tail.
            # Its binding does not validate destination stride/device/aliases.
            if (
                out.shape != (tokens, self.cfg.dim)
                or out.dtype != torch.bfloat16
                or not out.is_cuda
                or out.device != x.device
                or not out.is_contiguous()
                or out.data_ptr() % 16
            ):
                raise ValueError("invalid NVFP4 MegaMoE direct-output layout")
            protected = [x, weights, indices, self._mega_y, buf.buffer]
            protected.extend(
                getattr(self, name)
                for name in (
                    "_mega_l1_w",
                    "_mega_l1_sf",
                    "_mega_l1_gsf",
                    "_mega_l2_w",
                    "_mega_l2_sf",
                    "_mega_l2_gsf",
                )
            )
            storage = out.untyped_storage().data_ptr()
            if any(
                tensor.device == out.device
                and tensor.untyped_storage().data_ptr() == storage
                for tensor in protected
            ):
                raise ValueError(
                    "NVFP4 MegaMoE direct output aliases protected storage"
                )

        self._input_packer.pack(x, weights, indices, buf, tokens)
        self._maybe_pre_kernel_barrier(tokens)
        _sync_cuda_graph_warmup_ranks(
            f"mega_moe_nvfp4.layer{self.cfg.layer_id}.before_deepgemm", x.device
        )
        y = self._mega_y[:tokens] if out is None else out
        with torch.cuda.device(x.device):
            deep_gemm.nvfp4_nvfp4_mega_moe(
                y,
                (self._mega_l1_w, self._mega_l1_sf, self._mega_l1_gsf),
                (self._mega_l2_w, self._mega_l2_sf, self._mega_l2_gsf),
                buf,
                recipe=(1, 1, NVFP4_BLOCK),
                activation=activation_name,
                activation_clamp=_activation_clamp_or_none(limit),
                activation_alpha=alpha,
                fast_math=True,
                assume_all_topk_valid=True,
            )
        return y
