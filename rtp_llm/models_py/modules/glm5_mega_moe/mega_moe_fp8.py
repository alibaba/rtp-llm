"""GLM-5 MegaMoE FP8xFP8: DeepGEMM fp8_fp8_mega_moe strategy."""

from __future__ import annotations

import logging

import torch

from .input_packer import get_mega_moe_input_packer
from .mega_buf import get_or_create_mega_output
from .mega_fp8_buf import get_or_create_mega_buf_fp8
from .mega_moe import (
    GLM5MegaMoE,
    GLM5MegaMoeCfg,
    _mega_output_capacity,
    _sync_cuda_graph_warmup_ranks,
)
from .quant_layouts import FP4_BLOCK, prepare_fp8_weight_scale_for_deepgemm

logger = logging.getLogger(__name__)

_CUDA_GRAPH_CLONE_FP8_BUF_CACHE: dict[tuple, object] = {}


def _get_or_create_cuda_graph_clone_buf_fp8(src_buf, group, cfg: GLM5MegaMoeCfg):
    if src_buf is None or group is None:
        return src_buf
    key = (
        id(src_buf),
        id(group),
        cfg.n_routed_experts,
        cfg.max_tokens_per_rank,
        cfg.n_activated_experts,
        cfg.dim,
        cfg.moe_inter_dim,
    )
    cached = _CUDA_GRAPH_CLONE_FP8_BUF_CACHE.get(key)
    if cached is not None:
        return cached

    import deep_gemm

    cached = deep_gemm.get_symm_buffer_for_mega_moe_fp8(
        group=group,
        num_experts=cfg.n_routed_experts,
        num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
        num_topk=cfg.n_activated_experts,
        hidden=cfg.dim,
        intermediate_hidden=cfg.moe_inter_dim,
        use_fp8_dispatch=True,
        activation="swiglu",
    )
    _CUDA_GRAPH_CLONE_FP8_BUF_CACHE[key] = cached
    logging.info(
        "[GLM5 MegaMoE FP8] allocated CUDA graph clone symm buffer: layer=%d "
        "max_tokens_per_rank=%d hidden=%d",
        cfg.layer_id,
        cfg.max_tokens_per_rank,
        cfg.dim,
    )
    return cached


class GLM5MegaMoEFP8(GLM5MegaMoE):
    """GLM-5 MegaMoE wrapper for DeepGEMM ``fp8_fp8_mega_moe``."""

    def clone_for_cuda_graph(self) -> "GLM5MegaMoEFP8":
        clone = object.__new__(type(self))
        torch.nn.Module.__init__(clone)
        clone.cfg = self.cfg
        clone._mega_l1_w = self._mega_l1_w
        clone._mega_l1_sf = self._mega_l1_sf
        clone._mega_l2_w = self._mega_l2_w
        clone._mega_l2_sf = self._mega_l2_sf
        clone._mega_buf = _get_or_create_cuda_graph_clone_buf_fp8(
            self._mega_buf, self._mega_group, self.cfg
        )
        clone._mega_y = (
            torch.empty_like(self._mega_y) if self._mega_y is not None else None
        )
        clone._input_packer = get_mega_moe_input_packer()
        clone._mega_group = self._mega_group
        return clone

    def setup_weights_from_fp8(
        self,
        w1_fp8: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_fp8: torch.Tensor,
        w2_scale: torch.Tensor,
        w3_fp8: torch.Tensor,
        w3_scale: torch.Tensor,
    ) -> None:
        """Setup pre-quantized FP8 per-block expert weights for fp8_fp8_mega_moe.

        Inputs follow DeepGEMM convention: w1 is gate, w3 is up, w2 is down.
        Weight scales are either raw 128x128 per-block scales or DeepGEMM's
        packed int32 layout from the FP8 loader.
        """
        import deep_gemm

        cfg = self.cfg
        E = cfg.n_local_experts
        D = cfg.dim
        inter = cfg.moe_inter_dim

        for name, weight in (
            ("w1_fp8", w1_fp8),
            ("w2_fp8", w2_fp8),
            ("w3_fp8", w3_fp8),
        ):
            if weight.dtype != torch.float8_e4m3fn:
                raise TypeError(
                    f"mega_moe_fp8 requires FP8 e4m3 weights; {name} has {weight.dtype}"
                )

        logger.info(
            "[GLM5 MegaMoE FP8] preparing FP8 weights: layer=%d E=%d D=%d inter=%d",
            cfg.layer_id,
            E,
            D,
            inter,
        )

        w13_fp8 = torch.cat([w1_fp8, w3_fp8], dim=1).contiguous()
        s13 = torch.cat([w1_scale, w3_scale], dim=1).contiguous()
        del w1_fp8, w1_scale, w3_fp8, w3_scale

        if s13.dtype == torch.float32 or w2_scale.dtype == torch.float32:
            if s13.dtype != torch.float32 or w2_scale.dtype != torch.float32:
                raise TypeError(
                    "mega_moe_fp8 requires both FP8 MoE scales to be raw "
                    f"float32 or both prepacked int32/e8m0, got {s13.dtype} "
                    f"and {w2_scale.dtype}"
                )
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import requant_weight_ue8m0

            logger.info(
                "[GLM5 MegaMoE FP8] requantizing FP8 weights to UE8M0 scales: "
                "layer=%d",
                cfg.layer_id,
            )
            w13_fp8, s13_int = requant_weight_ue8m0(w13_fp8, s13)
            del s13
            torch.cuda.empty_cache()
            w2_fp8, s2_int = requant_weight_ue8m0(w2_fp8.contiguous(), w2_scale)
            del w2_scale
            torch.cuda.empty_cache()
        else:
            s13_int = prepare_fp8_weight_scale_for_deepgemm(s13, 2 * inter, D, E)
            s2_int = prepare_fp8_weight_scale_for_deepgemm(w2_scale, D, inter, E)
            del s13, w2_scale
        torch.cuda.empty_cache()

        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe_fp8(
            (w13_fp8, s13_int),
            (w2_fp8.contiguous(), s2_int),
        )
        del w13_fp8, s13_int, w2_fp8, s2_int
        torch.cuda.empty_cache()

        self._mega_l1_w = l1_w
        self._mega_l1_sf = l1_sf
        self._mega_l2_w = l2_w
        self._mega_l2_sf = l2_sf
        self._setup_buffer_and_warmup()

    def setup_weights_from_fp4(self, *args, **kwargs) -> None:
        raise ValueError("moe_strategy=mega_moe_fp8 only accepts FP8 per-block weights")

    def setup_weights_from_bf16(self, *args, **kwargs) -> None:
        raise ValueError("moe_strategy=mega_moe_fp8 only accepts FP8 per-block weights")

    def _setup_buffer_and_warmup(self) -> None:
        import torch.distributed as dist

        cfg = self.cfg
        device = self._mega_l1_w.device

        assert (
            dist.is_initialized()
        ), "GLM5 MegaMoE FP8 requires torch.distributed initialised"
        group = dist.group.WORLD
        self._mega_group = group

        self._mega_buf = get_or_create_mega_buf_fp8(
            group=group,
            num_experts=cfg.n_routed_experts,
            num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
            num_topk=cfg.n_activated_experts,
            hidden=cfg.dim,
            intermediate_hidden=cfg.moe_inter_dim,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        self._mega_y = get_or_create_mega_output(
            _mega_output_capacity(self._mega_buf, cfg.max_tokens_per_rank),
            cfg.dim,
            torch.bfloat16,
            device,
        )
        self._input_packer = get_mega_moe_input_packer()
        self._maybe_warmup_jit_once()

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        import deep_gemm

        T = x.size(0)
        buf = self._mega_buf
        if T > buf.num_max_tokens_per_rank:
            raise RuntimeError(
                f"GLM5 MegaMoE FP8 input tokens={T} exceeds "
                f"num_max_tokens_per_rank={buf.num_max_tokens_per_rank}. "
                "Raise the budget at startup."
            )
        if T > self._mega_y.size(0):
            raise RuntimeError(
                f"GLM5 MegaMoE FP8 output buffer rows={self._mega_y.size(0)} is "
                f"smaller than input tokens={T}."
            )

        self._input_packer.pack(x, weights, indices, buf, T)
        self._maybe_pre_kernel_barrier(T)
        _sync_cuda_graph_warmup_ranks(
            f"glm5.mega_moe_fp8.layer{self.cfg.layer_id}.before_deepgemm",
            x.device,
        )

        y = self._mega_y[:T]
        deep_gemm.fp8_fp8_mega_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=None, #(self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None),
            fast_math=False,
            assume_all_topk_valid=True,
        )
        return y
