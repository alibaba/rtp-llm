from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen_v2 import QWenV2


class CosyVoiceQwen(QWenV2):
    @classmethod
    def _create_config_from_json(cls, ckpt_path, config_json):
        # Both service startup and direct HF-config callers use this parser.
        config = super()._create_config_from_json(ckpt_path, config_json)
        config.input_vocab_size = config_json.get(
            "input_vocab_size", config.vocab_size + 151938
        )
        config.mm_model_config.mm_sep_tokens = [[-200]]
        return config


register_model("cosyvoice_qwen", CosyVoiceQwen, ["CosyQwen2ForCausalLM"])
