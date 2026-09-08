from typing import Any, Dict, List, Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.hybrid_kv_cache import build_hybrid_kv_cache_spec_descs
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next, Qwen35Dense, Qwen35Moe
from rtp_llm.models.qwen3_next.qwen3_next_weight import (
    Qwen3NextWeight,
    build_qwen35_dense_ffn_weights,
    plus_one,
)
from rtp_llm.ops import HybridAttentionType, KVCacheSpecType, RopeStyle
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose


class Qwen3NextMTPWeight(Qwen3NextWeight):
    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        super().__init__(*args, **kwargs)
        self.prefix = "mtp."
        self.model_prefix = "model."

    def _get_weight_info(self):
        weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.model_prefix + "embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_enorm,
                [
                    CkptWeightInfo(
                        self.prefix + "pre_fc_norm_embedding.weight", plus_one
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_hnorm,
                [CkptWeightInfo(self.prefix + "pre_fc_norm_hidden.weight", plus_one)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_eh_proj,
                [CkptWeightInfo(self.prefix + "fc.weight", identity)],
                transpose,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(self.prefix + "norm.weight", plus_one)],
                identity,
            ),
        ]

        all_layer_weights: List[List[WeightModule]] = []
        for _ in range(self._num_layers):
            layer_weights: List[WeightModule] = []
            layer_weights.extend(self._create_mqa_weight())
            layer_weights.extend(self._create_ffn_weight())
            layer_weights.extend(self._create_layer_norm_weight())
            all_layer_weights.append(layer_weights)

        return ModelWeightInfo(
            layer_weights=all_layer_weights,
            weights=weights,
        )


class Qwen35MoeMTPWeight(Qwen3NextMTPWeight):
    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        super().__init__(*args, **kwargs)
        self.model_prefix = "model.language_model."


class Qwen35DenseMTPWeight(Qwen35MoeMTPWeight):
    def _create_ffn_weight(self) -> List[WeightModule]:
        return build_qwen35_dense_ffn_weights(
            self.prefix, self._align_size, self.ffn_config
        )


class Qwen3NextMTPMixin:
    _mtp_moe_layer_index = (0,)
    _mtp_use_base_rope = False

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        # MTP layers always use full MQA, even when the target layer at the same
        # index uses linear attention.
        config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.NONE
        ]
        config.moe_layer_index = list(cls._mtp_moe_layer_index)
        config.num_layers = 1
        config.is_mtp = True
        if cls._mtp_use_base_rope:
            # Draft MTP consumes text tokens only. Plain RoPE keeps the
            # PyFlashinfer prefill CUDA graph implementation eligible.
            config.attn_config.rope_config.style = RopeStyle.Base
        return config

    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        model_config.kv_cache_spec_descs = build_hybrid_kv_cache_spec_descs(
            [HybridAttentionType.NONE],
            KVCacheSpecType.MHA,
        )

    def _create_python_model(self) -> Optional[Any]:
        from rtp_llm.models_py.model_desc.qwen3_next_mtp import Qwen3NextMTPModel

        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        self.py_model = Qwen3NextMTPModel(
            model_config,
            parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            moe_config=moe_config,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )


class Qwen3NextMTP(Qwen3NextMTPMixin, Qwen3Next):
    @staticmethod
    def get_weight_cls():
        return Qwen3NextMTPWeight


class Qwen35MoeMTP(Qwen3NextMTPMixin, Qwen35Moe):
    _mtp_use_base_rope = True

    @staticmethod
    def get_weight_cls():
        return Qwen35MoeMTPWeight


class Qwen35DenseMTP(Qwen3NextMTPMixin, Qwen35Dense):
    _mtp_moe_layer_index = ()
    _mtp_use_base_rope = True

    @staticmethod
    def get_weight_cls():
        return Qwen35DenseMTPWeight


register_model("qwen3_next_mtp", Qwen3NextMTP, ["Qwen3NextMTPForCausalLM"])
register_model("qwen35_moe_mtp", Qwen35MoeMTP, ["Qwen35MoeMTPForCausalLM"])
register_model("qwen35_dense_mtp", Qwen35DenseMTP)
