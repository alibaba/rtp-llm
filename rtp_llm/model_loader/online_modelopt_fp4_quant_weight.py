"""Unified MODELOPT_FP4 + Mega-MoE FP4 + Hybrid FP8 weight loaders.

This module owns ALL load-time weight quantization for FP4-based GLM-5/DeepSeek
deployments:

1. ModelOpt NVFP4 (block_size=16) for MoE w1/w2.
   ``OnlineModelOptFp4MoeWeight`` — BF16 → packed uint8 + per-block fp8 scale +
   per-tensor fp32 scale_2 + input_scale=1.  Triggered by ``MODELOPT_FP4`` quant
   config (no hybrid sub-mode).

2. Mega-MoE FP4 (UE8M0, block_size=32) for MoE w1/w2.
   ``OnlineMegaMoeFp4Weight`` — BF16 → packed int8 + fp32 scale.
   ``OnlineMegaMoeFp4FromFp8Weight`` — FP8 per-block → BF16 → FP4.
   Both run at load time so the BF16/FP8 tensor is released before
   ``MegaMoeFusedWrapper.__init__``. Triggered by ``MOE_STRATEGY=mega_moe``
   env var via ``apply_mega_moe_fp4_wrappers`` in
   ``model_weight_info.py``.

3. Hybrid FP8_PER_BLOCK for non-MoE weights when MODELOPT_FP4 hybrid mode is
   active. ``OnlineModelOptFp4HybridFp8AttnWeight`` — group=128, reuses
   ``LoadQuantPerBlockFp8Weight`` numerics.

The math for #1 follows ModelOptimizer's NVFP4 reference:
    weight_scale_2 = amax(W) / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)
    per_block_scale = per_block_amax / (FLOAT4_E2M1_MAX * weight_scale_2)
    q4 = round(W / (per_block_scale.fp8 * weight_scale_2))   -> packed e2m1 uint8

For #2 the math is ``deep_gemm.utils.per_token_cast_to_fp4(use_ue8m0=True,
gran_k=32)`` to match ``GLM5MegaMoE.setup_weights_from_bf16``.
"""

from typing import Any, Dict, Optional

import torch

from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    ModelOptFp4Config,
    QuantizationConfig,
)
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.utils.model_weight import W, identity

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
NVFP4_BLOCK_SIZE = 16
MEGA_MOE_FP4_BLOCK = 32
FP8_PER_BLOCK = 128

_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_E2M1_ODD_BOUNDS = _E2M1_BOUNDS[[1, 3, 5]]  # banker-rounding ties

_MEGA_MOE_KERNEL_NAMES = (W.moe_w1, W.moe_w2)


# ---------------------------------------------------------------------------
# ModelOpt NVFP4 helpers
# ---------------------------------------------------------------------------


def _cast_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Map float values in [-6, 6] to packed E2M1 ordinals (uint8 in [0, 15])."""
    device = x.device
    sign_bit = (x < 0).to(torch.uint8)
    abs_x = x.abs()
    bounds = _E2M1_BOUNDS.to(device=device, dtype=abs_x.dtype)
    ord_ = torch.searchsorted(bounds, abs_x, out_int32=True).to(torch.uint8)
    odd_bounds = _E2M1_ODD_BOUNDS.to(device=device, dtype=abs_x.dtype)
    equals_odd = torch.any(abs_x.unsqueeze(-1) == odd_bounds, dim=-1).to(torch.uint8)
    return (sign_bit << 3) + ord_ + equals_odd


def quantize_expert_to_nvfp4(
    weight: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
):
    """Quantize a 2D BF16/FP16 weight tensor to NVFP4 (per-block fp8 + per-tensor fp32).

    Args:
        weight: shape ``[N, K]`` with K divisible by ``block_size``.

    Returns:
        Tuple of (packed uint8 ``[N, K // 2]``, per-block fp8 scale ``[N, K // block_size]``,
        per-tensor fp32 scale_2 scalar).
    """
    assert weight.dim() == 2, f"expected 2D, got {tuple(weight.shape)}"
    N, K = weight.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"

    w_fp32 = weight.to(torch.float32)
    amax = w_fp32.abs().amax().clamp_min(1e-8)
    weight_scale_2 = (amax / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)).to(torch.float32)

    blocked = w_fp32.view(N, K // block_size, block_size)
    per_block_amax = blocked.abs().amax(dim=-1)  # [N, K//block_size]
    per_block_scale = per_block_amax / (FLOAT4_E2M1_MAX * weight_scale_2)
    per_block_scale = torch.where(
        per_block_scale == 0,
        torch.ones_like(per_block_scale),
        per_block_scale,
    )
    per_block_scale_fp8 = per_block_scale.to(torch.float8_e4m3fn)

    # Re-dequant fp8 scale to use the same value the kernel will see.
    effective_block_scale = per_block_scale_fp8.to(torch.float32)
    combined = (effective_block_scale * weight_scale_2).unsqueeze(-1)  # [N, K//bs, 1]
    scaled = blocked / combined
    scaled = scaled.clamp(-FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(N, K)

    q = _cast_to_e2m1(scaled)  # uint8 [N, K]
    packed = (q[..., 1::2].to(torch.uint8) << 4) | q[..., 0::2].to(torch.uint8)
    packed = packed.contiguous()  # [N, K // 2]
    return packed, per_block_scale_fp8.contiguous(), weight_scale_2


def _quantize_moe_weight(weight: torch.Tensor, block_size: int = NVFP4_BLOCK_SIZE):
    """Quantize a stacked MoE weight ``[E, N, K]`` to NVFP4."""
    assert weight.dim() == 3, f"expected 3D, got {tuple(weight.shape)}"
    E, N, K = weight.shape
    device = weight.device
    packed = torch.empty([E, N, K // 2], dtype=torch.uint8, device=device)
    block_scale = torch.empty(
        [E, N, K // block_size], dtype=torch.float8_e4m3fn, device=device
    )
    scale_2 = torch.empty([E], dtype=torch.float32, device=device)
    for i in range(E):
        p, s, s2 = quantize_expert_to_nvfp4(weight[i], block_size)
        packed[i].copy_(p)
        block_scale[i].copy_(s)
        scale_2[i] = s2
    return packed, block_scale, scale_2


def _moe_kernel_name(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_w1
    if name == W.moe_w2:
        return W.moe_w2
    raise ValueError(f"unsupported moe weight: {name}")


def _moe_scale_name(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_s1
    if name == W.moe_w2:
        return W.moe_s2
    raise ValueError(f"unsupported moe weight: {name}")


def _moe_scale_2_name(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_w1_s2
    if name == W.moe_w2:
        return W.moe_w2_s2
    raise ValueError(f"unsupported moe weight: {name}")


def _moe_input_scale_name(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_w1_i_s
    if name == W.moe_w2:
        return W.moe_w2_i_s
    raise ValueError(f"unsupported moe weight: {name}")


# ---------------------------------------------------------------------------
# Mega-MoE FP4 (UE8M0) helpers
# ---------------------------------------------------------------------------


def quantize_moe_weight_to_fp4_ue8m0(
    weight: torch.Tensor,
    block_size: int = MEGA_MOE_FP4_BLOCK,
):
    """Quantize a stacked MoE weight ``[E, N, K]`` (BF16/FP16/FP32) to NVFP4
    + UE8M0 scale, matching the per-expert loop in
    ``GLM5MegaMoE.setup_weights_from_bf16``.

    Returns:
        packed: int8 ``[E, N, K // 2]``
        sf    : float32 ``[E, N, K // block_size]``
    """
    from deep_gemm.utils import per_token_cast_to_fp4

    assert weight.dim() == 3, f"expected 3D, got {tuple(weight.shape)}"
    E, N, K = weight.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    device = weight.device

    packed = torch.empty((E, N, K // 2), dtype=torch.int8, device=device)
    sf = torch.empty((E, N, K // block_size), dtype=torch.float32, device=device)
    for i in range(E):
        packed_i, sf_i = per_token_cast_to_fp4(
            weight[i], use_ue8m0=True, gran_k=block_size
        )
        packed[i].copy_(packed_i)
        sf[i].copy_(sf_i)
    return packed, sf


def convert_fp8_moe_to_fp4_ue8m0(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: int = MEGA_MOE_FP4_BLOCK,
):
    """Convert stacked FP8 MoE weights ``[E, N, K]`` to FP4 + UE8M0 scale.

    Dequantizes FP8 per-block → BF16, then requantizes to FP4 (int8 packed)
    with UE8M0 scale factors suitable for DeepGEMM mega_moe.

    Handles two scale layouts:
      - Per-row: [E, N, K//128] — one scale per 128 elements along K
      - 2D-block: [E, N//128, K//128] — one scale per 128×128 block

    Returns:
        packed: int8 [E, N, K // 2]
        sf: float32 [E, N, K // block_size]
    """
    from deep_gemm.utils import per_token_cast_to_fp4

    assert (
        weight_fp8.dim() == 3
    ), f"expected 3D weight_fp8, got {tuple(weight_fp8.shape)}"
    E, N, K = weight_fp8.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    device = weight_fp8.device

    weight_scale = (
        weight_scale.reshape(E, -1) if weight_scale.dim() > 3 else weight_scale
    )
    if weight_scale.dim() == 2:
        weight_scale = weight_scale.unsqueeze(1)

    if weight_scale.shape == (E, N, K // FP8_PER_BLOCK):
        w_float = weight_fp8.float()
        scale_expanded = weight_scale.unsqueeze(-1).expand(
            E, N, K // FP8_PER_BLOCK, FP8_PER_BLOCK
        )
        w_float = (
            w_float.view(E, N, K // FP8_PER_BLOCK, FP8_PER_BLOCK) * scale_expanded
        ).reshape(E, N, K)
    elif weight_scale.shape == (E, N // FP8_PER_BLOCK, K // FP8_PER_BLOCK):
        scale_per_row = weight_scale.repeat_interleave(FP8_PER_BLOCK, dim=1)[:, :N, :]
        w_float = weight_fp8.float()
        scale_expanded = scale_per_row.unsqueeze(-1).expand(
            E, N, K // FP8_PER_BLOCK, FP8_PER_BLOCK
        )
        w_float = (
            w_float.view(E, N, K // FP8_PER_BLOCK, FP8_PER_BLOCK) * scale_expanded
        ).reshape(E, N, K)
        del scale_per_row
    else:
        expected_elements = E * N * (K // FP8_PER_BLOCK)
        if weight_scale.numel() == expected_elements:
            weight_scale = weight_scale.reshape(E, N, K // FP8_PER_BLOCK)
            w_float = weight_fp8.float()
            scale_expanded = weight_scale.unsqueeze(-1).expand(
                E, N, K // FP8_PER_BLOCK, FP8_PER_BLOCK
            )
            w_float = (
                w_float.view(E, N, K // FP8_PER_BLOCK, FP8_PER_BLOCK) * scale_expanded
            ).reshape(E, N, K)
        else:
            raise ValueError(
                f"Cannot interpret scale shape {tuple(weight_scale.shape)} "
                f"for weight shape [E={E}, N={N}, K={K}]"
            )

    w_bf16 = w_float.to(torch.bfloat16)
    del w_float, scale_expanded

    packed = torch.empty((E, N, K // 2), dtype=torch.int8, device=device)
    sf = torch.empty((E, N, K // block_size), dtype=torch.float32, device=device)
    for i in range(E):
        packed[i], sf[i] = per_token_cast_to_fp4(
            w_bf16[i], use_ue8m0=True, gran_k=block_size
        )
    del w_bf16
    return packed, sf


def _mega_moe_scale_name(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_s1
    if name == W.moe_w2:
        return W.moe_s2
    raise ValueError(f"unsupported mega_moe kernel name: {name}")


# ---------------------------------------------------------------------------
# OnlineModelOptFp4MoeWeight (existing) — ModelOpt NVFP4 for MoE
# ---------------------------------------------------------------------------


class OnlineModelOptFp4MoeWeight(CompositeWeight, QuantWeight):
    """Online NVFP4 weight loader for MoE ``moe_w1``/``moe_w2`` only."""

    moe_weight_list = [W.moe_w1, W.moe_w2]

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not isinstance(quant_config, ModelOptFp4Config):
            return False
        if quant_config.is_quanted():
            return False
        if not isinstance(src_weight_info, MoeAtomicWeight):
            return False
        return src_weight_info.name in cls.moe_weight_list

    def __init__(
        self,
        src_weight_info: MoeAtomicWeight,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        kernel = MoeAtomicWeight(
            name=_moe_kernel_name(src_weight_info.name),
            weights=src_weight_info.weights,
            process_fun=src_weight_info.process_fun,
            data_type=None,
            config=src_weight_info.config,
            stacked_ckpt_keys=getattr(src_weight_info, "stacked_ckpt_keys", False),
        )
        scale = MoeAtomicWeight(
            name=_moe_scale_name(src_weight_info.name),
            weights=[],
            process_fun=identity,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale_2 = MoeAtomicWeight(
            name=_moe_scale_2_name(src_weight_info.name),
            weights=[],
            process_fun=identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        input_scale = MoeAtomicWeight(
            name=_moe_input_scale_name(src_weight_info.name),
            weights=[],
            process_fun=identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        sub_weights = {
            kernel.name: kernel,
            scale.name: scale,
            scale_2.name: scale_2,
            input_scale.name: input_scale,
        }
        super().__init__(
            sub_weights,
            quant_config=quant_config,
            name=src_weight_info.name,
            **{k: v for k, v in kwargs.items() if k != "name"},
        )
        self.kernel = kernel
        self.scale = scale
        self.scale_2 = scale_2
        self.input_scale = input_scale
        self._block_size = max(quant_config.group_size(), NVFP4_BLOCK_SIZE)

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        return self.kernel.get_tensor_names(layer_id, load_config)

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        kernel_dict = self.kernel._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        weight = kernel_dict[self.kernel.name]
        packed, block_scale, scale_2 = _quantize_moe_weight(
            weight, block_size=self._block_size
        )
        input_scale = torch.ones([1], dtype=torch.float32, device=weight.device)
        return {
            self.kernel.name: packed,
            self.scale.name: block_scale,
            self.scale_2.name: scale_2,
            self.input_scale.name: input_scale,
        }

    def _postprocess(
        self,
        tensor,
        device: str,
        load_config: LoadConfig,
    ):
        processed = super()._postprocess(tensor, device, load_config)
        kernel_w = processed[self.kernel.name]
        scale_w = processed[self.scale.name]
        kernel_w, scale_w = (
            load_config.exported_device.maybe_prepare_static_weights_for_fp4_moe(
                self.kernel.name,
                self.scale.name,
                kernel_w,
                scale_w,
            )
        )
        processed[self.kernel.name] = kernel_w
        processed[self.scale.name] = scale_w
        return processed


# ---------------------------------------------------------------------------
# OnlineMegaMoeFp4Weight — BF16 → mega-MoE FP4 (UE8M0) at load time
# ---------------------------------------------------------------------------


class OnlineMegaMoeFp4Weight(CompositeWeight, QuantWeight):
    """Load-time FP4+UE8M0 quantizer for mega-MoE stacked MoE weights
    (``moe_w1`` = gate||up, ``moe_w2`` = down) starting from BF16.

    Output layout matches ``GLM5MegaMoE.setup_weights_from_bf16`` bit for bit.
    Doing the work here lets the BF16 tensor be released as soon as it lands
    on the GPU, instead of staying resident until ``MegaMoeFusedWrapper.__init__``.

    Wired in by ``apply_mega_moe_fp4_wrappers`` (in ``model_weight_info.py``)
    when ``MOE_STRATEGY=mega_moe``. Inherits ``QuantWeight`` purely so that
    ``MoeWeight``'s sub-weight assertion passes; the ``support()`` classmethod
    is gated to never fire from the regular ``QuantWeight.create()`` registry
    (the explicit wrap pass is the only entry point).
    """

    moe_weight_list = list(_MEGA_MOE_KERNEL_NAMES)

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        return False  # explicit-wiring only; never auto-selected

    def __init__(
        self,
        src_weight_info: MoeAtomicWeight,
        block_size: int = MEGA_MOE_FP4_BLOCK,
        **kwargs: Any,
    ):
        if src_weight_info.name not in _MEGA_MOE_KERNEL_NAMES:
            raise ValueError(
                f"OnlineMegaMoeFp4Weight only wraps {_MEGA_MOE_KERNEL_NAMES}, "
                f"got {src_weight_info.name}"
            )

        kernel = MoeAtomicWeight(
            name=src_weight_info.name,
            weights=src_weight_info.weights,
            process_fun=src_weight_info.process_fun,
            data_type=None,  # follow load_config.compute_dtype (BF16)
            config=src_weight_info.config,
            stacked_ckpt_keys=getattr(src_weight_info, "stacked_ckpt_keys", False),
        )
        scale = MoeAtomicWeight(
            name=_mega_moe_scale_name(src_weight_info.name),
            weights=[],  # synthesised in _load_raw_tensor
            process_fun=identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )

        sub_weights = {kernel.name: kernel, scale.name: scale}
        super().__init__(
            sub_weights,
            quant_config=None,
            name=src_weight_info.name,
            **{k: v for k, v in kwargs.items() if k != "name"},
        )
        self.kernel = kernel
        self.scale = scale
        self._block_size = block_size

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        return self.kernel.get_tensor_names(layer_id, load_config)

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        kernel_dict = self.kernel._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        weight = kernel_dict[self.kernel.name]
        packed, sf = quantize_moe_weight_to_fp4_ue8m0(weight, self._block_size)
        del weight
        kernel_dict[self.kernel.name] = packed
        kernel_dict[self.scale.name] = sf
        return kernel_dict

    def _split(self, tensor, load_config: LoadConfig):
        split_kernel = self.kernel._split(
            {self.kernel.name: tensor[self.kernel.name]}, load_config
        )
        split_scale = self.scale._split(
            {self.scale.name: tensor[self.scale.name]}, load_config
        )
        out: Dict[str, torch.Tensor] = {}
        out.update(split_kernel)
        out.update(split_scale)
        return out

    def _postprocess(self, tensor, device: str, load_config: LoadConfig):
        # No layout transform here: the deepgemm SF layout + mega-moe transform
        # require the (l1, l2) pair together and run inside MegaMoeFusedWrapper.
        return {
            self.kernel.name: tensor[self.kernel.name],
            self.scale.name: tensor[self.scale.name],
        }


# ---------------------------------------------------------------------------
# OnlineMegaMoeFp4FromFp8Weight — FP8 per-block → mega-MoE FP4 at load time
# ---------------------------------------------------------------------------


class OnlineMegaMoeFp4FromFp8Weight(CompositeWeight, QuantWeight):
    """Load-time FP8 per-block → FP4+UE8M0 quantizer for mega-MoE.

    Takes FP8 per-block quantized weights (``float8_e4m3fn`` kernel +
    ``float32`` scale) and produces FP4 packed int8 + fp32 UE8M0 scale at
    load time, matching ``GLM5MegaMoE.setup_weights_from_fp8`` numerically.

    Wired in by ``apply_mega_moe_fp4_wrappers`` when ``MOE_STRATEGY=mega_moe``
    and the original MoE weight was a ``PerBlockFp8Weight``.
    """

    moe_weight_list = list(_MEGA_MOE_KERNEL_NAMES)

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        return False  # explicit-wiring only

    def __init__(
        self,
        src_kernel: MoeAtomicWeight,
        src_scale: MoeAtomicWeight,
        block_size: int = MEGA_MOE_FP4_BLOCK,
        **kwargs: Any,
    ):
        if src_kernel.name not in _MEGA_MOE_KERNEL_NAMES:
            raise ValueError(
                f"OnlineMegaMoeFp4FromFp8Weight only wraps {_MEGA_MOE_KERNEL_NAMES}, "
                f"got {src_kernel.name}"
            )

        # Sub-weight that loads the FP8 kernel from the checkpoint.
        kernel = MoeAtomicWeight(
            name=src_kernel.name,
            weights=src_kernel.weights,
            process_fun=src_kernel.process_fun,
            data_type=torch.float8_e4m3fn,
            config=src_kernel.config,
            stacked_ckpt_keys=getattr(src_kernel, "stacked_ckpt_keys", False),
        )
        # Sub-weight that loads the FP8 per-block scale; renamed with an
        # ``_fp8_input`` suffix so it does not collide with the FP4 output
        # scale tensor we emit.
        fp8_scale_name = _mega_moe_scale_name(src_kernel.name) + "_fp8_input"
        fp8_scale = MoeAtomicWeight(
            name=fp8_scale_name,
            weights=src_scale.weights,
            process_fun=src_scale.process_fun,
            data_type=torch.float32,
            config=src_scale.config,
            stacked_ckpt_keys=getattr(src_scale, "stacked_ckpt_keys", False),
        )
        # Output FP4 scale (synthesised in _load_raw_tensor).
        out_scale = MoeAtomicWeight(
            name=_mega_moe_scale_name(src_kernel.name),
            weights=[],
            process_fun=identity,
            data_type=torch.float32,
            config=src_kernel.config,
        )

        sub_weights = {
            kernel.name: kernel,
            fp8_scale.name: fp8_scale,
            out_scale.name: out_scale,
        }
        super().__init__(
            sub_weights,
            quant_config=None,
            name=src_kernel.name,
            **{k: v for k, v in kwargs.items() if k != "name"},
        )
        self.kernel = kernel
        self.fp8_scale = fp8_scale
        self.out_scale = out_scale
        self._block_size = block_size

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        names = self.kernel.get_tensor_names(layer_id, load_config)
        names |= self.fp8_scale.get_tensor_names(layer_id, load_config)
        return names

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        kernel_dict = self.kernel._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        scale_dict = self.fp8_scale._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        weight_fp8 = kernel_dict[self.kernel.name]
        weight_scale = scale_dict[self.fp8_scale.name]

        packed, sf = convert_fp8_moe_to_fp4_ue8m0(
            weight_fp8, weight_scale, self._block_size
        )
        del weight_fp8, weight_scale
        return {
            self.kernel.name: packed,
            self.out_scale.name: sf,
        }

    def _split(self, tensor, load_config: LoadConfig):
        split_kernel = self.kernel._split(
            {self.kernel.name: tensor[self.kernel.name]}, load_config
        )
        split_scale = self.out_scale._split(
            {self.out_scale.name: tensor[self.out_scale.name]}, load_config
        )
        out: Dict[str, torch.Tensor] = {}
        out.update(split_kernel)
        out.update(split_scale)
        return out

    def _postprocess(self, tensor, device: str, load_config: LoadConfig):
        return {
            self.kernel.name: tensor[self.kernel.name],
            self.out_scale.name: tensor[self.out_scale.name],
        }


# ---------------------------------------------------------------------------
# OnlineModelOptFp4HybridFp8AttnWeight — FP8_PER_BLOCK for non-MoE in hybrid
# ---------------------------------------------------------------------------


from rtp_llm.model_loader.per_block_fp8_quant_weight import (  # noqa: E402
    LoadQuantPerBlockFp8Weight,
    create_w8a8_fp8_per_block_weight,
)


class OnlineModelOptFp4HybridFp8AttnWeight(LoadQuantPerBlockFp8Weight):
    """FP8_PER_BLOCK loader used by ``MODELOPT_FP4`` hybrid mode for non-MoE weights."""

    _excluded_names = (W.moe_w1, W.moe_w2, W.mla_kc, W.mla_vc)

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not isinstance(quant_config, ModelOptFp4Config):
            return False
        if quant_config.is_quanted():
            return False
        if getattr(quant_config, "hybrid_attn_quant_method", None) != "FP8_PER_BLOCK":
            return False
        if isinstance(src_weight_info, MoeAtomicWeight):
            return False
        name = src_weight_info.name
        if name in cls._excluded_names:
            return False
        return name in cls.w8a8_weight_list

    def __init__(
        self,
        src_weight_info: AtomicWeight,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        # FP4 quant_config has group_size=16; for the FP8 block-wise path used by
        # attention weights we lock to the canonical FP8 per-block size (128).
        self.group_size = Fp8BlockWiseQuantConfig.DEFAULT_FP8_QUANT_BLOCK_SIZE
        params = src_weight_info.extract_params(
            src_weight_info.__class__, src_weight_info, quant_config
        )
        kernel: AtomicWeight = create_w8a8_fp8_per_block_weight(
            src_weight_info, **params
        )
        sub_weights = {kernel.name: kernel}
        scale_name = self.w8a8_weight_list.get(src_weight_info.name)
        scale = None
        if scale_name:
            scale_params = {**params}
            scale_params["name"] = scale_name
            scale = create_w8a8_fp8_per_block_weight(src_weight_info, **scale_params)
            sub_weights.update({scale.name: scale})

        CompositeWeight.__init__(
            self, sub_weights, quant_config=quant_config, *args, **kwargs
        )
        self.kernel = kernel
        self.scale = scale

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        return self.kernel.get_tensor_names(layer_id, load_config)


# ---------------------------------------------------------------------------
# Wiring helper
# ---------------------------------------------------------------------------


def is_mega_moe_strategy() -> bool:
    """Return True when MOE_STRATEGY=mega_moe is set in the env."""
    import os

    return os.environ.get("MOE_STRATEGY") == "mega_moe"


def is_online_fp4gemm_enabled() -> bool:
    """Return True when USE_ONLINE_FP4GEMM=1 is set in the env."""
    import os

    return os.environ.get("USE_ONLINE_FP4GEMM", "0") == "1"


def mxfp4_quantize_linear_weight(weight: torch.Tensor):
    """Quantize a 2D BF16/FP16 weight ``[K, N]`` to MXFP4 (block=32, UE8M0).

    Returns the transposed FP4 packed weight (uint8) and transposed scale
    tensor consumed by ``flashinfer.mm_fp4(backend='cute-dsl', block_size=32,
    use_nvfp4=False)`` from the linear's forward path.

    Numerically and layout-wise identical to the original inline code in
    ``CudaOnlineMxfp4Linear.__init__``. **Do not** call ``.contiguous()`` on
    the returned tensors — ``mm_fp4`` expects the strided
    transposed view (``stride[0]==1``, ``stride[1]==K_packed``) and rejects
    a row-major contiguous version with a "Mismatched strides" error. The
    BF16 source tensor is released internally.
    """
    from flashinfer import mxfp4_quantize

    assert weight.dim() == 2, f"expected 2D, got {tuple(weight.shape)}"
    K, _N = weight.shape
    assert K % 128 == 0, f"K={K} must be divisible by 128 for MXFP4"

    w = weight.T.to(torch.bfloat16).contiguous()  # [N, K]
    w_fp4, w_sf = mxfp4_quantize(w, backend="cute-dsl")
    del w
    return w_fp4.T, w_sf.T


# ---------------------------------------------------------------------------
# Wiring helpers (mega-MoE FP4)
# ---------------------------------------------------------------------------


def wrap_moe_for_mega_moe(weight: WeightModule) -> WeightModule:
    """If ``weight`` is a MoE atomic / per-block FP8 wrapper for moe_w1/moe_w2,
    return a load-time FP4 quantizer that consumes BF16 or FP8 source. Otherwise
    return the weight unchanged.

    This is the single entry point used by ``apply_mega_moe_fp4_wrappers`` in
    ``model_weight_info.py``.
    """
    from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight

    if isinstance(weight, PerBlockFp8Weight) and weight.name in _MEGA_MOE_KERNEL_NAMES:
        kernel = weight.kernel
        scale = weight.scale
        if kernel is None or scale is None:
            return weight
        return OnlineMegaMoeFp4FromFp8Weight(
            src_kernel=kernel,
            src_scale=scale,
        )
    if isinstance(weight, MoeAtomicWeight) and weight.name in _MEGA_MOE_KERNEL_NAMES:
        return OnlineMegaMoeFp4Weight(weight)
    return weight
