"""MXFP8 (1x32 microscaling FP8) pre-quantized weight loader for MiniMax-M3.

Reuses ``PerBlockFp8Weight`` for all the ckpt-name derivation, qkv/moe merge,
stacking and TP/EP split logic (identical for the e4m3 kernel and its scale).
Only two things differ from FP8_PER_BLOCK:

* ``support()`` matches :class:`Fp8MxBlockWiseQuantConfig`.
* ``_postprocess()`` keeps the e4m3 kernel as-is and converts the on-disk
  UE8M0 (uint8, loaded as the raw exponent byte values) ``weight_scale_inv``
  into DeepGEMM's int32 ``(1, 32)`` packed layout instead of running the
  128x128 ``requant_weight_ue8m0`` path.
"""

from typing import Any, Dict, Union

import torch

from rtp_llm.config.quant_config import (
    Fp8MxBlockWiseQuantConfig,
    QuantizationConfig,
)
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight
from rtp_llm.model_loader.weight_module import CompositeWeight, WeightModule
from rtp_llm.utils.model_weight import is_v4_weight


class Mxfp8Weight(PerBlockFp8Weight):
    def __init__(
        self,
        src_weight_info: WeightModule,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(src_weight_info, quant_config, *args, **kwargs)
        # TP-split fix for the (1,32) microscale.
        #
        # ``PerBlockFp8Weight`` is built for the 128x128 block-FP8 scale and
        # assigns the qkv scale the ``sp_head_s_gemm_a8_block`` strategy, which
        # divides the head/hidden dims by ``block_size=128`` before splitting.
        # That layout assumption is wrong for MXFP8: the scale is ``[N, K//32]``
        # with one UE8M0 byte per (row, 32-col) block, so its row (N) axis is
        # identical to the kernel's and its col axis is just ``K//32`` (not a
        # 128-block grid). Running the 128-block splitter mangles the scale
        # under TP>1 (rows collapse to ``N/128`` and the tensor is reshaped to
        # a block grid), so ``pack_mxfp8_scale`` then fails
        # ``sf.size(-2) == ceil_div(mn, gran_mn)``.
        #
        # The (1,32) scale shares the kernel's axes exactly (same N rows;
        # K//32 cols that split proportionally to the kernel's K cols), so the
        # kernel's own split function partitions it correctly regardless of
        # whether the split is by-head (dim 0) or even (dim -1). Force the scale
        # to reuse the kernel's split function so it is sharded identically.
        if getattr(self, "scale", None) is not None and getattr(
            self, "kernel", None
        ) is not None:
            kernel_split = self.kernel._get_split_func()
            self.scale._get_split_func = lambda _f=kernel_split: _f

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, Fp8MxBlockWiseQuantConfig
        ):
            return False
        if src_weight_info.name not in cls.w8a8_weight_list:
            return False
        if is_v4_weight(src_weight_info):
            return False
        return True

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ):
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import (
            MX_BLOCK,
            pack_mxfp8_scale,
        )

        # Grab the raw (already merged / TP-EP split) kernel + scale tensors;
        # deliberately skip PerBlockFp8Weight._postprocess so we don't trigger
        # the 128x128 requant_weight_ue8m0 path.
        processed_res = CompositeWeight._postprocess(self, tensor, device, load_config)

        kernel_weight = processed_res[self.kernel.name]
        kernel_weight = load_config.exported_device.maybe_rewrite_weight_by_key(
            "weight", kernel_weight
        )
        processed_res[self.kernel.name] = kernel_weight

        if self.scale is None:
            return processed_res

        scale_weight = processed_res[self.scale.name]
        scale_weight = load_config.exported_device.maybe_rewrite_weight_by_key(
            "scale", scale_weight
        )

        # On-disk UE8M0 is stored as uint8 exponent bytes (bias 127). The
        # generic loader casts it to fp32, preserving the byte *values*, so the
        # real fp32 power-of-two scale is 2^(byte - 127) regardless of dtype.
        # NOTE: store the *fp32* power-of-two scale and DEFER the int32 packing
        # to first forward. DeepGEMM binds its JIT runtime to the first CUDA
        # device it is used on; during multi-rank weight loading that device
        # differs from the rank's device, so calling
        # transform_sf_into_required_layout here fails with
        # CUDA_ERROR_INVALID_HANDLE. Packing lazily at forward (cached on the
        # module) matches how every other DeepGEMM path in rtp-llm is used.
        scale_fp32 = torch.exp2(
            scale_weight.to(device=kernel_weight.device, dtype=torch.float32) - 127.0
        ).contiguous()
        processed_res[self.scale.name] = scale_fp32
        return processed_res
