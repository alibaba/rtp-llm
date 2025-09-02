from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.multimodal.multimodal_mixin import (
    BaseMultiModalWeightInfo,
    BaseVitWeights,
    MultiModalMixin,
)
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.models.qwen_v2_audio.processor import Processor
from rtp_llm.utils.util import get_config_from_path


class QWenV2AudioWeightinfo(QWenV2Weight, BaseMultiModalWeightInfo):
    def __init__(self, config: GptInitModelParameters, tp_size: int, tp_rank: int):
        QWenV2Weight.__init__(self, config, tp_size, tp_rank)
        BaseMultiModalWeightInfo.__init__(self, config)

    def _get_weight_info(self):
        qwen_weight = super()._get_weight_info()
        qwen_weight = self._get_vit_info(qwen_weight)
        return qwen_weight


class QWenV2Audio(QWenV2, MultiModalMixin):
    def _init_multimodal(self, config: GptInitModelParameters):
        self.mm_part = Processor(config)
        config.mm_related_params.vit_weights = BaseVitWeights(
            {
                "multi_modal_projector": self.mm_part.multi_modal_projector,
                "audio_tower": self.mm_part.audio_tower,
            },
            with_prefix=True,
        )
        config.mm_related_params.vit_weights._ckpt_prefix = ""

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        return config

    @staticmethod
    def get_weight_cls():
        return QWenV2AudioWeightinfo

    @classmethod
    def _from_hf(cls, config: GptInitModelParameters, ckpt_path: str):
        config_json = get_config_from_path(ckpt_path)
        if not config_json:
            raise Exception(f"failed to get config.json from path: {ckpt_path}")
        sep_token = config_json["audio_token_index"]
        config_json = config_json["text_config"]

        # config.activation_type = config_json["hidden_act"]
        config.inter_size = config_json.get("intermediate_size", 11008)
        config.head_num = config_json.get("num_attention_heads", 32)
        config.head_num_kv = config_json.get("num_key_value_heads", config.head_num)
        config.size_per_head = config_json.get("hidden_size", 4096) // config.head_num
        config.layer_num = config_json.get("num_hidden_layers", 32)
        config.rotary_embedding_base = config_json.get(
            "rope_theta", config.rotary_embedding_base
        )
        config.vocab_size = config_json["vocab_size"]
        config.rotary_embedding_dim = config.size_per_head
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)

        config.mm_sep_tokens = [[sep_token]]  # image_token_index
        config.config_dtype = config_json.get("torch_dtype", None)


register_model("qwen_v2_audio", QWenV2Audio)
