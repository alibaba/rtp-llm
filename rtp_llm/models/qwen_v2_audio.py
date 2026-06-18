from typing import Any

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.util import get_config_from_path


class _MultimodalQwen2MtpModel:
    """Wrapper that adds multimodal embedding injection to Qwen2MtpModel.

    Qwen2MtpModel is text-only; audio features would be silently dropped.
    This wrapper intercepts forward() to inject multimodal features into
    the text embeddings before the MTP decoder layers run.
    """

    def __init__(self, base_model):
        self._base = base_model
        from rtp_llm.models_py.modules import MultimodalEmbeddingInjector
        self._injector = MultimodalEmbeddingInjector()

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        mm_features = inputs.multimodal_inputs.multimodal_features
        mm_feature_locs = inputs.multimodal_inputs.mm_features_locs
        if mm_features is not None and mm_feature_locs is not None and mm_features.numel() > 0:
            # Inject multimodal features into text embeddings by patching
            # the base model's embed_tokens call via a temporary hook.
            # We do this by calling the base forward but intercepting
            # the embedding output through a monkey-patch.
            original_embed = self._base.embed_tokens
            original_forward = self._base.forward

            def patched_embed(input_ids, *args, **kwargs):
                embeds = original_embed(input_ids, *args, **kwargs)
                return self._injector(embeds, mm_features, mm_feature_locs)

            self._base.embed_tokens = patched_embed
            try:
                result = original_forward(inputs, fmha_impl)
            finally:
                self._base.embed_tokens = original_embed
            return result
        return self._base.forward(inputs, fmha_impl)

    def __getattr__(self, name):
        return getattr(self._base, name)


class QWenV2Audio(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        # super()._create_config 内硬编码调 QWenV2._from_hf，会跳过 audio 特定
        # 字段（audio_token_index / text_config 内的 head_num 等）。这里显式
        # 再调一次本类的 _from_hf 让 audio 字段生效。不把 super 那条改成
        # cls._from_hf 是为了不破坏 Qwen2Moe（它的 _from_hf 不调 super 的基础
        # 字段设置，依赖 super 先硬编码调 QWenV2._from_hf）。
        QWenV2Audio._from_hf(config, ckpt_path)
        config.mm_model_config.is_multimodal = True
        return config

    @staticmethod
    def get_weight_cls():
        return QWenV2Weight

    def _create_python_model(self):
        # Override to wrap the Qwen2MtpModel with multimodal injection.
        # Without this, audio features are silently dropped by the text-only model.
        super()._create_python_model()
        self.py_model = _MultimodalQwen2MtpModel(self.py_model)

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

        config.mm_related_params.config["ckpt_path"] = config.ckpt_path


register_model("qwen_v2_audio", QWenV2Audio)
