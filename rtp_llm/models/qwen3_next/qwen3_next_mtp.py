from typing import Any, Dict, List, Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next
from rtp_llm.models.qwen3_next.qwen3_next_weight import Qwen3NextWeight, plus_one
from rtp_llm.ops import HybridAttentionType
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


class Qwen3NextMTP(Qwen3Next):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        # mtp model attention is mqa, not linear
        config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.NONE
        ]
        config.moe_layer_index = [0]
        config.num_layers = 1
        config.is_mtp = True
        return config

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

    @staticmethod
    def get_weight_cls():
        return Qwen3NextMTPWeight


class Qwen35MoeMTP(Qwen3NextMTP):
    @classmethod
    def get_weight_cls():
        return Qwen35MoeMTPWeight


register_model("qwen3_next_mtp", Qwen3NextMTP, ["Qwen3NextMTPForCausalLM"])
register_model("qwen35_moe_mtp", Qwen35MoeMTP, ["Qwen35MoeMTPForCausalLM"])
