"""MXFP8 (1x32 microscaling FP8) pre-quantized weight loader.

Reuses ``PerBlockFp8Weight`` for all the ckpt-name derivation, qkv/moe merge,
stacking and TP/EP split logic (identical for the e4m3 kernel and its scale).
Only two things differ from FP8_PER_BLOCK:

* ``support()`` matches :class:`Fp8MxBlockWiseQuantConfig`.
* ``_postprocess()`` keeps the e4m3 kernel as-is and converts the on-disk
  UE8M0 exponent bytes into fp32 powers of two. DeepGEMM layout packing is
  deferred to first forward and cached by the linear/MoE executor.
"""

import functools
from typing import Any, Dict, List, Union

import torch

from rtp_llm.config.quant_config import (
    Fp8MxBlockWiseQuantConfig,
    QuantizationConfig,
)
from rtp_llm.model_loader.attn_weight import MlaAttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    PerBlockFp8Weight,
    create_w8a8_fp8_per_block_weight,
)
from rtp_llm.model_loader.weight_module import CompositeWeight, WeightModule
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    is_v4_weight,
    pad,
    pad_w13,
    stack_,
    stack_moe_w1,
    transpose_slice_k,
    transpose_slice_v,
)


MX_BLOCK = 32


def _dequantize_mxfp8(weight: torch.Tensor, scale_exponents: torch.Tensor) -> torch.Tensor:
    """Materialize a derived BF16 attention-BMM weight from MXFP8 storage.

    This is only used for ``mla_kc``/``mla_vc``. They are consumed by
    ``torch.bmm`` rather than a Linear/MoE GEMM and therefore cannot retain the
    checkpoint's MXFP8 representation.
    """
    if weight.shape[-1] % MX_BLOCK != 0:
        raise ValueError(
            f"MXFP8 K={weight.shape[-1]} must be divisible by {MX_BLOCK}"
        )
    expected = (*weight.shape[:-1], weight.shape[-1] // MX_BLOCK)
    if tuple(scale_exponents.shape) != expected:
        raise ValueError(
            f"MXFP8 scale shape must be {expected}, got {tuple(scale_exponents.shape)}"
        )
    blocked = weight.float().view(*weight.shape[:-1], -1, MX_BLOCK)
    scale = torch.exp2(scale_exponents.float() - 127.0)
    return (blocked * scale.unsqueeze(-1)).reshape(weight.shape).bfloat16()


def _dequantize_mxfp8_split_k(
    ts: List[torch.Tensor],
    head_num: int,
    nope_head_dim: int,
    v_head_dim: int,
    lora_rank: int,
) -> torch.Tensor:
    return transpose_slice_k(
        [_dequantize_mxfp8(ts[0], ts[1])],
        head_num,
        nope_head_dim,
        v_head_dim,
        lora_rank,
    )


def _dequantize_mxfp8_split_v(
    ts: List[torch.Tensor],
    head_num: int,
    nope_head_dim: int,
    v_head_dim: int,
    lora_rank: int,
) -> torch.Tensor:
    return transpose_slice_v(
        [_dequantize_mxfp8(ts[0], ts[1])],
        head_num,
        nope_head_dim,
        v_head_dim,
        lora_rank,
    )


class Mxfp8Weight(PerBlockFp8Weight):
    def __init__(
        self,
        src_weight_info: WeightModule,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        self._checkpoint_scale_suffix = getattr(
            quant_config, "checkpoint_scale_suffix", ".weight_scale_inv"
        )
        self._packed_scale_suffix = getattr(
            quant_config, "packed_scale_suffix", "_scale_inv"
        )
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

    def _get_scale_suffix(self, scale_fmt: object) -> str:
        del scale_fmt
        return self._checkpoint_scale_suffix

    def _scale_name(self, source_name: str) -> str:
        if source_name.endswith(".weight"):
            return source_name[: -len(".weight")] + self._checkpoint_scale_suffix
        return source_name + self._packed_scale_suffix

    @staticmethod
    def _is_excluded(
        quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        excluded = getattr(quant_config, "exclude_modules", set())
        if not excluded:
            return False
        layer_id = getattr(src_weight_info, "layer_id", None)
        for ckpt in getattr(src_weight_info, "weights", ()):
            source_name = ckpt.name
            if layer_id is not None:
                source_name = source_name.format(
                    i=str(layer_id), i_1=str(layer_id + 1), expert_id="{expert_id}"
                )
            module_name = (
                source_name[: -len(".weight")]
                if source_name.endswith(".weight")
                else source_name
            )
            if source_name in excluded or module_name in excluded:
                return True
        return False

    @staticmethod
    def _get_scale_dtype(scale_fmt: object) -> torch.dtype:
        del scale_fmt
        # AtomicWeight converts the raw exponent byte values to fp32 without
        # changing their numeric value. _postprocess performs exp2(value-127).
        return torch.float32

    def _get_qkv_quant_weight(self, src_weight_info, group_size: int):
        del group_size
        return super()._get_qkv_quant_weight(src_weight_info, MX_BLOCK)

    def _get_mla_attn_out_quant_weight(self, src_weight_info, group_size: int):
        del group_size
        return super()._get_mla_attn_out_quant_weight(src_weight_info, MX_BLOCK)

    def _get_ffn_quant_weight(
        self, src_weight_info: FfnAtomicWeight, group_size: int
    ):
        del group_size
        if src_weight_info.name not in (W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13):
            raise ValueError(f"unsupported MXFP8 FFN weight: {src_weight_info.name}")

        weights = src_weight_info.weights
        align_size = src_weight_info.config.align_size
        if src_weight_info.name == W.ffn_w13:
            kernel_process = functools.partial(pad_w13, align_size=align_size, dim=0)
            # MXFP8 scale is [N, K/32], so N uses the same padding as weight.
            scale_process = kernel_process
            return [
                create_w8a8_fp8_per_block_weight(
                    src_weight_info,
                    W.ffn_w13,
                    [CkptWeightInfo(w.name, identity) for w in weights],
                    kernel_process,
                    data_type=torch.float8_e4m3fn,
                    config=src_weight_info.config,
                ),
                create_w8a8_fp8_per_block_weight(
                    src_weight_info,
                    W.ffn_s13,
                    [CkptWeightInfo(self._scale_name(w.name), identity) for w in weights],
                    scale_process,
                    data_type=torch.float32,
                    config=src_weight_info.config,
                ),
            ]

        source_name = weights[0].name
        if src_weight_info.name in (W.ffn_w1, W.ffn_w3):
            kernel_name, scale_name = (
                (W.ffn_w1, W.ffn_s1)
                if src_weight_info.name == W.ffn_w1
                else (W.ffn_w3, W.ffn_s3)
            )
            kernel_process = functools.partial(pad, align_size=align_size, dim=0)
            scale_process = kernel_process
        else:
            kernel_name, scale_name = W.ffn_w2, W.ffn_s2
            kernel_process = functools.partial(pad, align_size=align_size, dim=1)
            # down_proj K is the intermediate dimension represented in blocks.
            scale_process = functools.partial(
                pad, align_size=align_size // MX_BLOCK, dim=1
            )

        return [
            create_w8a8_fp8_per_block_weight(
                src_weight_info,
                kernel_name,
                [CkptWeightInfo(source_name, identity)],
                kernel_process,
                data_type=torch.float8_e4m3fn,
                config=src_weight_info.config,
            ),
            create_w8a8_fp8_per_block_weight(
                src_weight_info,
                scale_name,
                [CkptWeightInfo(self._scale_name(source_name), identity)],
                scale_process,
                data_type=torch.float32,
                config=src_weight_info.config,
            ),
        ]

    def _get_moe_w2_quant_weight(self, src_weight_info: MoeAtomicWeight):
        source_name = src_weight_info.weights[0].name
        scale_name = self._scale_name(source_name)
        stacked = src_weight_info.stacked_ckpt_keys
        kernel = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.moe_w2,
            [CkptWeightInfo(source_name, identity)],
            src_weight_info.process_fun if stacked else stack_,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
            stacked_ckpt_keys=stacked,
        )
        scale = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.moe_s2,
            [CkptWeightInfo(scale_name, identity)],
            src_weight_info.process_fun if stacked else stack_,
            data_type=torch.float32,
            config=src_weight_info.config,
            stacked_ckpt_keys=stacked,
        )
        return [kernel, scale]

    def _get_moe_w1_quant_weight(self, src_weight_info: MoeAtomicWeight):
        kernel_names = [weight.name for weight in src_weight_info.weights]
        scale_names = [self._scale_name(name) for name in kernel_names]
        stacked = src_weight_info.stacked_ckpt_keys
        process_fun = src_weight_info.process_fun if stacked else stack_moe_w1
        kernel = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.moe_w1,
            [CkptWeightInfo(name, identity) for name in kernel_names],
            process_fun,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
            stacked_ckpt_keys=stacked,
        )
        scale = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.moe_s1,
            [CkptWeightInfo(name, identity) for name in scale_names],
            process_fun,
            data_type=torch.float32,
            config=src_weight_info.config,
            stacked_ckpt_keys=stacked,
        )
        return [kernel, scale]

    def _get_mla_kv_c(self, src_weight_info: MlaAttnAtomicWeight):
        is_k = src_weight_info.name == W.mla_kc
        source_name = src_weight_info.weights[0].name
        if not source_name.endswith(".weight"):
            raise ValueError(f"unexpected MXFP8 MLA weight name: {source_name}")
        scale_name = self._scale_name(source_name)
        process_fun = _dequantize_mxfp8_split_k if is_k else _dequantize_mxfp8_split_v
        kernel = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            src_weight_info.name,
            [
                CkptWeightInfo(source_name, identity),
                CkptWeightInfo(scale_name, identity),
            ],
            functools.partial(
                process_fun,
                head_num=src_weight_info.config.head_num,
                nope_head_dim=src_weight_info.nope_head_dim,
                v_head_dim=src_weight_info.v_head_dim,
                lora_rank=src_weight_info.kv_lora_rank,
            ),
            # The process function materializes a BF16 weight for torch.bmm.
            # Keeping this metadata as FP32 makes the loader cast it back to
            # FP32 and forces bmm to produce FP32 into a BF16 output view.
            data_type=torch.bfloat16,
            config=src_weight_info.config,
        )
        return [kernel, None]

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
        if cls._is_excluded(quant_config, src_weight_info):
            return False
        return True

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ):
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
        if kernel_weight.shape[-1] % MX_BLOCK != 0:
            raise ValueError(
                f"MXFP8 K={kernel_weight.shape[-1]} must be divisible by {MX_BLOCK}"
            )
        expected_scale_shape = (
            *kernel_weight.shape[:-1],
            kernel_weight.shape[-1] // MX_BLOCK,
        )
        if tuple(scale_weight.shape) != expected_scale_shape:
            raise ValueError(
                f"MXFP8 scale shape must be {expected_scale_shape}, got "
                f"{tuple(scale_weight.shape)} for {self.kernel.name} "
                f"weight {tuple(kernel_weight.shape)}"
            )

        # On-disk UE8M0 is stored as uint8 exponent bytes (bias 127). The
        # generic loader casts it to fp32, preserving the byte *values*, so the
        # real fp32 power-of-two scale is 2^(byte - 127) regardless of dtype.
        # Store the *fp32* power-of-two scale and defer int32 packing to the
        # owning CUDA module. DeepGEMM binds its JIT runtime to the current
        # CUDA device, which is not guaranteed to match this rank during the
        # generic loader phase. MegaMoE packs during module setup; MXFP8 Linear
        # packs lazily on first forward.
        scale_fp32 = torch.exp2(
            scale_weight.to(device=kernel_weight.device, dtype=torch.float32) - 127.0
        ).contiguous()
        processed_res[self.scale.name] = scale_fp32
        return processed_res
