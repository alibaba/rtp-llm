"""DeepSeek VL V2 language model for the new loader.

语言骨干与 DeepSeek V3.2 完全一致（MLA + MoE），直接继承
``DeepSeekV32ForCausalLM``，权重加载（``load_weights`` / ``WEIGHTS_MAPPER``）
原样继承。

与纯文本 DeepSeek V3.2 的差异：
  1. DeepSeek VL V2 的 ``config.json`` 把语言模型配置嵌套在
     ``language_config`` 子字段下，因此需要覆写 ``_read_config_json``
     把 ``language_config`` 合并到顶层，让 ``_extract_config_values``
     能正确找到 MLA / MoE / RoPE 等字段。
  2. ``forward`` 需要把视觉特征注入到对应占位 token 的 embedding 上
     （``MultimodalEmbeddingInjector``）。DeepSeek VL V2 不使用 deepstack，
     所以只需 embedding 级别的注入。
"""

import logging
from typing import Any, Dict

import torch

from rtp_llm.models_py.modules import MultimodalEmbeddingInjector
from rtp_llm.models_py.new_models.deepseek_v3.language import (
    DeepSeekV32ForCausalLM,
    _read_config_json,
)
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs

logger = logging.getLogger(__name__)


class DeepSeekVLV2ForCausalLM(DeepSeekV32ForCausalLM):
    """DeepSeek VL V2 language model for the new loader.

    Inherits all MLA + MoE weight loading from DeepSeekV32ForCausalLM.
    Overrides ``_read_config_json`` to merge the nested ``language_config``
    section, and ``forward`` to inject multimodal visual features.
    """

    @staticmethod
    def _read_config_json(ckpt_path: str) -> Dict[str, Any]:
        """Read config.json and merge ``language_config`` to top level.

        DeepSeek VL V2's ``config.json`` nests the language model config
        under ``language_config`` (e.g. ``qk_rope_head_dim``,
        ``moe_intermediate_size``, ``rope_scaling``). The parent's
        ``_extract_config_values`` reads these from the top level, so we
        merge ``language_config`` entries into the top-level dict (without
        overwriting any existing top-level keys like ``vision_config``).
        """
        config = _read_config_json(ckpt_path)
        if config and "language_config" in config:
            lang_config = config["language_config"]
            if isinstance(lang_config, dict):
                for key, value in lang_config.items():
                    if key not in config:
                        config[key] = value
        return config

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__(model_config, load_config)
        # Stateless injector: placed here to align with the Qwen3-VL pattern.
        # The injector just copies visual features into the right embedding
        # positions; it has no parameters and does not affect weight loading.
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids

        mm = getattr(inputs, "multimodal_inputs", None)
        mm_features = getattr(mm, "multimodal_features", []) if mm else []
        mm_feature_locs = getattr(mm, "mm_features_locs", None) if mm else None

        inputs_embeds = self.embed_tokens(input_ids)

        if mm_features and mm_feature_locs is not None:
            inputs_embeds = self.multimodal_embedding_injector(
                inputs_embeds, mm_features, mm_feature_locs
            )

        hidden_states = inputs_embeds

        if fmha_impl is None:
            self._ensure_weight_assembled()
            fmha_impl = self.prepare_fmha_impl(inputs)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )

        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = ["DeepSeekVLV2ForCausalLM"]
