from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.utils.util import get_config_from_path


class QWenV2Audio(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config.mm_model_config.is_multimodal = True
        return config

    @staticmethod
    def get_weight_cls():
        return QWenV2Weight

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

        config.mm_relate_params.config["ckpt_path"] = config.ckpt_path


register_model("qwen_v2_audio", QWenV2Audio)
