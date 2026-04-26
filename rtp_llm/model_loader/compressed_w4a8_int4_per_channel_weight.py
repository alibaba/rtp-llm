"""Loader for pre-quantized compressed-tensors INT4 group weights.

Used by Kimi-K2.5 routed-expert MoE: HF ckpt stores `weight_packed`
(packed int4) + `weight_scale` (bf16). We unpack -> repack into the
cutlass W4A8 layout and convert scales to fp8, matching the output
of `quantize_weight_to_int4b` so the existing
`CutlassExpertsW4a8Int4PerChannel` executor can consume them.
"""

import copy
import re
from typing import Optional

import torch

from rtp_llm.config.quant_config import (
    CompressedW4A8Int4PerChannelQuantConfig,
    QuantizationConfig,
)
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.w8a8_weight import create_w8a8_fp8_weight
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, W, WeightStyle, identity


def _matches_any(name: str, patterns) -> bool:
    for pat in patterns:
        if pat.startswith("re:"):
            if re.search(pat[3:], name):
                return True
        else:
            if pat in name:
                return True
    return False


def repack_compressed_int4_to_cutlass(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
):
    """Convert HF compressed-tensors INT4 layout into the cutlass W4A8 layout.

    Args:
        weight_packed: int32/uint8/int8 tensor of shape [N, K // pack_factor].
            compressed-tensors `pack-quantized` (see `pack_to_int32` in
            `compressed_tensors.compressors.quantized_compressors.pack_quantized`)
            stores each 4-bit weight as **offset-binary** (signed value + 8 →
            unsigned 0..15), packed low-bits-first into the container: for an
            int32 holding w0..w7, bits[0:4]=(w0+8)&0xF, …, bits[28:32]=(w7+8)&0xF.
            Reinterpreted as little-endian int8s, each storage byte holds
            (w0+8) in the low nibble and (w1+8) in the high nibble.
            Online `quantize_weight_to_int4b` instead stores nibbles as 4-bit
            two's complement (signed_value & 0xF). To bring CT bytes into the
            online/cutlass two's-complement convention we XOR every nibble's
            sign bit — i.e. byte ^= 0x88 — which adds 8 (mod 16) to each nibble.
        weight_scale: bf16 tensor of shape [N, K // group_size].
        group_size: per-group size (32 for K2.6, 128 for Qwen3 INT4 ckpt).

    Returns:
        (output_unified_int4, scale_packed) — the same tensors that
        `quantize_weight_to_int4b` produces.
    """
    from rtp_kernel.w4a8_group_gemm import (
        pack_scale_fp8,
        reorder_tensor,
        unified_encode_int4b,
    )

    assert weight_packed.dim() == 2, f"unexpected packed shape: {weight_packed.shape}"

    # Reinterpret int32/int16 storage as int8 so each storage byte holds
    # exactly two int4 nibbles, matching the cutlass kernel's expectation.
    # Little-endian byte order on x86 keeps the nibble order intact.
    if weight_packed.dtype not in (torch.int8, torch.uint8):
        weight_packed = weight_packed.contiguous().view(torch.int8)

    N, packed_k = weight_packed.shape
    K = packed_k * 2
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    assert weight_scale.shape == (
        N,
        K // group_size,
    ), f"scale shape mismatch: {weight_scale.shape} vs ({N}, {K // group_size})"

    # Offset-binary (CT) → two's complement (online/cutlass): flip the sign bit
    # of every nibble. Doing it byte-wise as ^0x88 covers both nibbles at once.
    # int8 ^ 0x88 in PyTorch is well-defined for the bit pattern; do it via
    # uint8 view to avoid any sign-extension surprises.
    packed_uint8 = weight_packed.to(torch.uint8).contiguous()
    packed_uint8 = packed_uint8 ^ 0x88
    packed_int8 = packed_uint8.view(torch.int8).contiguous()

    # Scales: HF stores [N, K/group_size] in bf16. The cutlass kernel wants
    # [K/group_size, N, 8] fp8 (after pack_scale_fp8). Mirror the layout used
    # by `quantize_weight_to_int4b`: transpose to [K/group_size, N], cast to
    # fp8_e4m3fn, then call `pack_scale_fp8`.
    scale_fp8 = weight_scale.to(torch.float8_e4m3fn).t().contiguous()
    scale_packed = pack_scale_fp8(scale_fp8)

    out = unified_encode_int4b(packed_int8)
    out = reorder_tensor(out)
    return out, scale_packed


def _replace_suffix(name: str, suffix: str) -> str:
    """Replace the trailing `.weight` with `suffix` (e.g. `.weight_packed`).

    The standard MoE atomic weight uses ckpt keys like
    `model.layers.{i}.mlp.experts.{expert_id}.up_proj.weight`. For the
    compressed path we need to swap the trailing `.weight` for the
    pre-quantized counterpart.
    """
    if name.endswith(".weight"):
        return name[: -len(".weight")] + suffix
    return name + suffix


def _build_compressed_ckpt_weights(
    src_weights, suffix: str
) -> list:
    new_weights = []
    for cw in src_weights:
        new_weights.append(
            CkptWeightInfo(_replace_suffix(cw.name, suffix), cw.merge_fun)
        )
    return new_weights


class LoadCompressedW4A8Int4PerGroupQuantWeight(CompositeWeight, QuantWeight):
    """Composite weight that materialises (packed_int4, fp8_scale) for MoE.

    For each routed-expert MoE atomic weight (`W.moe_w1` / `W.moe_w2`), build
    a kernel sub-weight that reads `*.weight_packed` (instead of `*.weight`)
    and a scale sub-weight that reads `*.weight_scale`. After loading per
    expert, repack into cutlass layout.
    """

    weight_scale_map = {
        W.moe_w1: (W.moe_s1, None, None),
        W.moe_w2: (W.moe_s2, None, None),
    }

    int4_weight_list = [W.moe_w1, W.moe_w2]

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, CompressedW4A8Int4PerChannelQuantConfig
        ):
            return False
        name = src_weight_info.name
        if name not in cls.int4_weight_list:
            return False
        if src_weight_info.weight_style in [
            WeightStyle.TRT_ENGINE,
            WeightStyle.RTP_SMOOTH_LLM_STYLE,
        ]:
            return False
        # Honour `ignore` patterns from the compressed-tensors config: skip the
        # MoE weight if its ckpt prefix matches an ignore pattern.
        if hasattr(src_weight_info, "weights") and src_weight_info.weights:
            ckpt_name = src_weight_info.weights[0].name
            if _matches_any(ckpt_name, quant_config.ignore_patterns):
                return False
        return True

    def __init__(
        self,
        src_weight_info: AtomicWeight,
        quant_config: QuantizationConfig,
        *args,
        **kwargs,
    ):
        params = src_weight_info.extract_params(
            src_weight_info.__class__, src_weight_info, quant_config
        )

        # Kernel sub-weight: read `.weight_packed` instead of `.weight`.
        # `pack-quantized` INT4 stores 8 nibbles per int32 container; we must
        # preserve the bit pattern through load (no numerical cast to bf16).
        kernel_params = copy.deepcopy(params)
        kernel_params["weights"] = _build_compressed_ckpt_weights(
            src_weight_info.weights, quant_config.weight_pack_suffix
        )
        kernel_params["data_type"] = torch.int32
        kernel: AtomicWeight = create_w8a8_fp8_weight(
            src_weight_info, **kernel_params
        )

        scale_name, _, _ = self.weight_scale_map[src_weight_info.name]
        scale_params = copy.deepcopy(params)
        scale_params["name"] = scale_name
        scale_params["weights"] = _build_compressed_ckpt_weights(
            src_weight_info.weights, quant_config.scale_suffix
        )
        # Scales are bf16 in the ckpt; load_tensor will pass through the dtype.
        scale_params["data_type"] = torch.bfloat16
        scale: AtomicWeight = create_w8a8_fp8_weight(
            src_weight_info, **scale_params
        )

        sub_weights = {kernel.name: kernel, scale.name: scale}
        super().__init__(sub_weights, quant_config=quant_config, *args, **kwargs)
        self.kernel = kernel
        self.scale = scale
        self.group_size = quant_config.group_size()
        self.act_scale = None
        self.act_scale_inv = None

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
        scale_dict = self.scale._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )

        kernel_tensor = kernel_dict[self.kernel.name]
        scale_tensor = scale_dict[self.scale.name]

        assert (
            kernel_tensor.dim() == 3
        ), f"expected stacked-expert kernel [E, N, K_packed], got {kernel_tensor.shape}"
        # `pack-quantized` ckpt stores int4 nibbles in an int32 container;
        # reinterpret the bytes as int8 so each storage byte holds two
        # nibbles, matching the per-expert repack path's expectation.
        if kernel_tensor.dtype not in (torch.int8, torch.uint8):
            kernel_tensor = kernel_tensor.contiguous().view(torch.int8)
        E, N, packed_k = kernel_tensor.shape
        K = packed_k * 2
        assert (
            scale_tensor.shape == (E, N, K // self.group_size)
        ), (
            f"scale shape mismatch: {scale_tensor.shape} vs "
            f"({E}, {N}, {K // self.group_size})"
        )

        quant_kernel = torch.empty(
            (E, N, K // 2), device=kernel_tensor.device, dtype=torch.int8
        )
        out_scale = torch.empty(
            (E, K // self.group_size, N, 8),
            device=kernel_tensor.device,
            dtype=torch.float8_e4m3fn,
        )
        for i in range(E):
            quant_kernel[i], out_scale[i] = repack_compressed_int4_to_cutlass(
                kernel_tensor[i], scale_tensor[i], self.group_size
            )

        return {
            self.kernel.name: quant_kernel.contiguous().to(device),
            self.scale.name: out_scale.contiguous().to(device),
        }
