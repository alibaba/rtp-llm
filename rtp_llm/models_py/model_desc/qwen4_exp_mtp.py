"""Python model descriptor for the Qwen4-Exp MTP draft."""

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.mtp_input_projection import (
    Qwen4ExpMTPInputProjection,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.mtp_runtime import (
    Qwen4ExpMTPRuntime,
    normalize_mtp_decode_positions,
    normalize_mtp_prefill_positions,
)
from rtp_llm.models_py.modules.qwen4_exp.weight_views import (
    Qwen4LinearFactory as LinearFactory,
)
from rtp_llm.ops import ParallelismConfig
from rtp_llm.utils.model_weight import W


class Qwen4ExpMTPModel(Qwen4ExpMTPRuntime, Qwen4ExpModel):
    """One-layer draft constructed from legacy weights. Behavior is shared."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
    ) -> None:
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
        )
        if self.layer_num != 1:
            raise RuntimeError("qwen4_exp MTP model descriptor requires one layer")
        if getattr(model_config, "enable_qwen4_ple", False):
            raise RuntimeError("qwen4_exp MTP model descriptor does not support PLE")
        weights = self.weight
        self.mtp_input_projection = Qwen4ExpMTPInputProjection(
            weights.get_global_weight(W.multi_tokens_predict_enorm),
            weights.get_global_weight(W.multi_tokens_predict_hnorm),
            LinearFactory.create_linear_from_weights(
                weights.global_weights,
                W.qwen4_mtp_fc_embedding_w,
                hw_kernel_config=py_hw_kernel_config,
            ),
            LinearFactory.create_linear_from_weights(
                weights.global_weights,
                W.qwen4_mtp_fc_hidden_w,
                hw_kernel_config=py_hw_kernel_config,
            ),
            hidden_size=model_config.hidden_size,
            norm_eps=model_config.layernorm_eps,
        )
