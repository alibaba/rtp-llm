"""Online FP8_PER_BLOCK quantization for non-MoE weights when MODELOPT_FP4 runs
in hybrid mode (``hybrid_attn_quant_method == "FP8_PER_BLOCK"``).

In ``MODELOPT_FP4`` hybrid mode the model is quantized as:
  - MoE w1/w2  -> NVFP4 via ``OnlineModelOptFp4MoeWeight``
  - everything else (attention, ffn, mla_*, ...) -> FP8 per-block (group=128),
    using the same numerics as ``LoadQuantPerBlockFp8Weight``
"""

from typing import Any, Optional

from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    ModelOptFp4Config,
    QuantizationConfig,
)
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    LoadQuantPerBlockFp8Weight,
    create_w8a8_fp8_per_block_weight,
)
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    WeightModule,
)
from rtp_llm.utils.model_weight import W


class OnlineModelOptFp4HybridFp8AttnWeight(LoadQuantPerBlockFp8Weight):
    """FP8_PER_BLOCK loader used by ``MODELOPT_FP4`` hybrid mode for non-MoE weights."""

    # MoE weights are handled by the FP4 path; mla_kc/mla_vc need pre-quantized scales.
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
