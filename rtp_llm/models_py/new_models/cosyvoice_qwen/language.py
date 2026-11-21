"""CosyVoice Qwen2 with separate input/output vocabularies and external features."""

from rtp_llm.models_py.modules import MultimodalEmbeddingInjector
from rtp_llm.models_py.new_models.cosyvoice_qwen.inputs import embed_external_features
from rtp_llm.models_py.new_models.qwen2.language import Qwen2ForCausalLM


class CosyVoiceQwenForCausalLM(Qwen2ForCausalLM):
    def __init__(self, model_config, load_config):
        if isinstance(model_config, dict):
            # The runtime requires a native ModelConfig, not an HF mapping.
            from rtp_llm.models.cosyvoice_qwen import CosyVoiceQwen

            raw = model_config
            model_config = CosyVoiceQwen._create_config_from_json(
                raw.get("model_path", ""), raw
            )
            for key in (
                "model_type",
                "enable_fp32_lm_head",
                "normalize_lm_head_weight",
                "logit_scale",
                "enable_output_vocab_pruning",
                "output_vocab_ids",
                "output_vocab_padded_size",
                "hidden_state_capture_layer_ids",
            ):
                if key in raw:
                    setattr(model_config, key, raw[key])
        super().__init__(model_config, load_config)
        self._inject = MultimodalEmbeddingInjector()

    def _input_embeddings(self, inputs):
        return embed_external_features(self.embed_tokens, self._inject, inputs)

    def checkpoint_weight_name_filter(self):
        owns = super().checkpoint_weight_name_filter()
        return lambda name: name.startswith(
            ("model.", "lm_head.", "transformer.")
        ) and (owns is None or owns(name))
