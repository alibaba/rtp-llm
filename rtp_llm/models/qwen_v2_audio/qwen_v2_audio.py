from rtp_llm.config.model_config import ModelConfig, VitParameters
from rtp_llm.config.py_config_modules import VitConfig
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
    def __init__(self, vit_weights, **kwargs):
        QWenV2Weight.__init__(self, **kwargs)
        BaseMultiModalWeightInfo.__init__(self, vit_weights=vit_weights, **kwargs)


class QWenV2Audio(QWenV2, MultiModalMixin):
    def _init_multimodal(
        self,
        mm_model_config,
        vit_config: VitConfig,
    ):
        # mm_related_params is in model_config, not mm_model_config
        self.mm_part = Processor(
            self.model_config.mm_related_params, self.model_config.ckpt_path
        )
        self.model_config.mm_related_params.vit_weights = BaseVitWeights(
            {
                "multi_modal_projector": self.mm_part.multi_modal_projector,
                "audio_tower": self.mm_part.audio_tower,
            },
            with_prefix=True,
        )
        self.model_config.mm_related_params.vit_weights._ckpt_prefix = ""

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.qwen import create_qwen_v2_config

        config = create_qwen_v2_config(ckpt_path)
        return config

    @staticmethod
    def get_weight_cls():
        return QWenV2AudioWeightinfo

    @classmethod
    def _from_hf(cls, config: ModelConfig, ckpt_path: str):
        config_json = get_config_from_path(ckpt_path)
        if not config_json:
            raise Exception(f"failed to get config.json from path: {ckpt_path}")
        sep_token = config_json["audio_token_index"]
        config_json = config_json["text_config"]

        # config.activation_type = config_json["hidden_act"]
        config.inter_size = config_json.get("intermediate_size", 11008)
        config.attn_config.head_num = config_json.get("num_attention_heads", 32)
        config.attn_config.kv_head_num = config_json.get(
            "num_key_value_heads", config.attn_config.head_num
        )
        config.attn_config.size_per_head = (
            config_json.get("hidden_size", 4096) // config.attn_config.head_num
        )
        config.num_layers = config_json.get("num_hidden_layers", 32)
        config.attn_config.rope_config.base = int(
            config_json.get("rope_theta", config.attn_config.rope_config.base)
        )
        config.vocab_size = config_json["vocab_size"]
        config.attn_config.rope_config.dim = config.attn_config.size_per_head
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)

        config.mm_model_config.mm_sep_tokens = [[sep_token]]  # image_token_index
        config.config_dtype = config_json.get("torch_dtype", None)


register_model("qwen_v2_audio", QWenV2Audio)
