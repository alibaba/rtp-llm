"""GLM-5 MegaMoE FP8xFP8: DeepGEMM fp8_fp8_mega_moe strategy."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

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
from .quant_layouts import (
    FP4_BLOCK,
    FP8_BLOCK,
    MXFP8_BLOCK,
    prepare_fp8_weight_scale_for_deepgemm,
)

logger = logging.getLogger(__name__)

_CUDA_GRAPH_CLONE_FP8_BUF_CACHE: dict[tuple, object] = {}


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _infer_fp8_scale_recipe(
    scale: torch.Tensor, mn: int, k: int
) -> Tuple[int, int]:
    """Infer a raw MegaMoE weight-scale recipe from its trailing shape."""
    # Packed scales do not retain enough logical shape information. Existing
    # callers only prepack the legacy 128x128 format; MXFP8 stays raw FP32
    # until this module packs it on the owning CUDA rank.
    if scale.dtype == torch.int32:
        return (FP8_BLOCK, FP8_BLOCK)

    trailing = tuple(scale.shape[-2:])
    recipe_1x32 = (1, MXFP8_BLOCK)
    expected_1x32 = (mn, _ceil_div(k, MXFP8_BLOCK))
    if trailing == expected_1x32 or scale.numel() == mn * expected_1x32[1]:
        return recipe_1x32

    recipe_128x128 = (FP8_BLOCK, FP8_BLOCK)
    expected_128x128 = (
        _ceil_div(mn, FP8_BLOCK),
        _ceil_div(k, FP8_BLOCK),
    )
    if trailing == expected_128x128 or scale.numel() == (
        expected_128x128[0] * expected_128x128[1]
    ):
        return recipe_128x128

    raise ValueError(
        "Cannot infer FP8 MegaMoE weight scale recipe from "
        f"shape={tuple(scale.shape)} for mn={mn}, k={k}. Expected trailing "
        f"dims {expected_1x32} for 1x32 or {expected_128x128} for 128x128."
    )


def _reshape_fp8_scale_for_recipe(
    scale: torch.Tensor,
    num_groups: int,
    mn: int,
    k: int,
    recipe: Tuple[int, int],
) -> torch.Tensor:
    if scale.dtype == torch.int32:
        return scale
    gran_mn, gran_k = recipe
    expected = (
        num_groups,
        _ceil_div(mn, gran_mn),
        _ceil_div(k, gran_k),
    )
    if tuple(scale.shape) == expected:
        return scale
    if scale.numel() == expected[0] * expected[1] * expected[2]:
        return scale.reshape(expected)
    raise ValueError(
        "FP8 MegaMoE scale shape does not match inferred recipe. Got "
        f"shape={tuple(scale.shape)}, expected={expected}, recipe={recipe}."
    )


def _get_or_create_cuda_graph_clone_buf_fp8(
    src_buf,
    group,
    cfg: GLM5MegaMoeCfg,
    num_shared_experts: int = 0,
):
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
        num_shared_experts,
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
        num_shared_experts=num_shared_experts,
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

    def __init__(self, cfg: GLM5MegaMoeCfg):
        super().__init__(cfg)
        self._fp8_weight_recipe: Tuple[int, int] = (FP8_BLOCK, FP8_BLOCK)

    def clone_for_cuda_graph(self) -> "GLM5MegaMoEFP8":
        clone = object.__new__(type(self))
        torch.nn.Module.__init__(clone)
        clone.cfg = self.cfg
        clone._mega_l1_w = self._mega_l1_w
        clone._mega_l1_sf = self._mega_l1_sf
        clone._mega_l2_w = self._mega_l2_w
        clone._mega_l2_sf = self._mega_l2_sf
        clone._num_shared_experts = getattr(self, "_num_shared_experts", 0)
        clone._mega_buf = _get_or_create_cuda_graph_clone_buf_fp8(
            self._mega_buf,
            self._mega_group,
            self.cfg,
            clone._num_shared_experts,
        )
        clone._mega_y = (
            torch.empty_like(self._mega_y) if self._mega_y is not None else None
        )
        clone._input_packer = get_mega_moe_input_packer()
        clone._mega_group = self._mega_group
        clone._fp8_weight_recipe = self._fp8_weight_recipe
        return clone

    def _prepare_fp8_scale(
        self,
        scale: torch.Tensor,
        mn: int,
        k: int,
        num_groups: int,
        recipe: Tuple[int, int],
    ) -> torch.Tensor:
        scale = _reshape_fp8_scale_for_recipe(scale, num_groups, mn, k, recipe)
        return prepare_fp8_weight_scale_for_deepgemm(
            scale,
            mn,
            k,
            num_groups=num_groups,
            recipe=recipe,
        )

    def setup_weights_from_fp8(
        self,
        w1_fp8: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_fp8: torch.Tensor,
        w2_scale: torch.Tensor,
        w3_fp8: torch.Tensor,
        w3_scale: torch.Tensor,
    ) -> None:
        """Setup pre-quantized FP8 expert weights for fp8_fp8_mega_moe.

        Inputs follow DeepGEMM convention: w1 is gate, w3 is up, w2 is down.
        Raw scales may use either 128x128 block FP8 or MXFP8's per-row 1x32
        layout. Packed int32 scales retain the legacy 128x128 interpretation.
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

        l1_recipe = _infer_fp8_scale_recipe(s13, 2 * inter, D)
        l2_recipe = _infer_fp8_scale_recipe(w2_scale, D, inter)
        if l1_recipe != l2_recipe:
            raise ValueError(
                "fp8_fp8_mega_moe requires L1/L2 weights to use the same "
                f"FP8 weight recipe, got L1={l1_recipe}, L2={l2_recipe}"
            )
        self._fp8_weight_recipe = l1_recipe

        if l1_recipe == (FP8_BLOCK, FP8_BLOCK) and (
            s13.dtype == torch.float32 or w2_scale.dtype == torch.float32
        ):
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
            s13_int = self._prepare_fp8_scale(s13, 2 * inter, D, E, l1_recipe)
            s2_int = self._prepare_fp8_scale(w2_scale, D, inter, E, l2_recipe)
            del s13, w2_scale
        torch.cuda.empty_cache()

        logger.info(
            "[GLM5 MegaMoE FP8] transforming weights: layer=%d weight_recipe=%s",
            cfg.layer_id,
            self._fp8_weight_recipe,
        )

        def _transform_weights():
            return deep_gemm.transform_weights_for_mega_moe_fp8(
                (w13_fp8, s13_int),
                (w2_fp8.contiguous(), s2_int),
            )

        if w13_fp8.is_cuda:
            with torch.cuda.device(w13_fp8.device):
                (l1_w, l1_sf), (l2_w, l2_sf) = _transform_weights()
        else:
            (l1_w, l1_sf), (l2_w, l2_sf) = _transform_weights()
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
        activation_clamp: Optional[float] = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            x,
            weights,
            indices,
            inputs_prepacked=False,
            activation_clamp=activation_clamp,
        )

    def forward_prepacked(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(
            x,
            None,
            None,
            inputs_prepacked=True,
            activation_clamp=None,
        )

    def _forward_impl(
        self,
        x: torch.Tensor,
        weights: torch.Tensor | None,
        indices: torch.Tensor | None,
        *,
        inputs_prepacked: bool,
        activation_clamp: Optional[float],
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

        if not inputs_prepacked:
            if weights is None or indices is None:
                raise ValueError("weights and indices are required before packing")
            self._input_packer.pack(x, weights, indices, buf, T)
        self._maybe_pre_kernel_barrier(T)
        _sync_cuda_graph_warmup_ranks(
            f"glm5.mega_moe_fp8.layer{self.cfg.layer_id}.before_deepgemm",
            x.device,
        )

        y = self._mega_y[:T]
        def _run_mega_moe() -> None:
            deep_gemm.fp8_fp8_mega_moe(
                y,
                (self._mega_l1_w, self._mega_l1_sf),
                (self._mega_l2_w, self._mega_l2_sf),
                buf,
                recipe=(1, 1, FP4_BLOCK),
                weight_recipe=self._fp8_weight_recipe,
                activation="swiglu",
                activation_clamp=activation_clamp,
                fast_math=False,
                assume_all_topk_valid=True,
            )

        if x.is_cuda:
            with torch.cuda.device(x.device):
                _run_mega_moe()
        else:
            _run_mega_moe()
        return y
