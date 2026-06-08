import json
import logging
import os

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity

logger = logging.getLogger(__name__)


class Qwen2_5OmniTalkerWeight(QWenV2Weight):
    def __init__(self, **kwargs):
        super().__init__(prefix="talker.", **kwargs)

    def _get_hf_weight_info(self):
        weights_info = super()._get_hf_weight_info()
        # Replace lm_head → codec_head
        for w in weights_info.weights:
            if w.name == W.lm_head:
                w.weights = [
                    CkptWeightInfo(self.prefix + "codec_head.weight", identity)
                ]
                break
        # Add thinker_to_talker_proj (Linear(3584→896) with bias)
        weights_info.weights.extend([
            AtomicWeight(
                "thinker_to_talker_proj.weight",
                [CkptWeightInfo(self.prefix + "thinker_to_talker_proj.weight", identity)],
                identity,
            ),
            AtomicWeight(
                "thinker_to_talker_proj.bias",
                [CkptWeightInfo(self.prefix + "thinker_to_talker_proj.bias", identity)],
                identity,
            ),
        ])
        return weights_info


class Qwen2_5OmniTalker(QWenV2):
    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.qwen2_5_omni_talker import (
            Qwen2_5OmniTalkerModel,
        )

        self.py_model = Qwen2_5OmniTalkerModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            quant_config=self.model_config.quant_config,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.rope_config.style = 1
        config.has_pre_decoder_layernorm = False
        config.special_tokens.bos_token_id = -1
        config.special_tokens.eos_token_id = 8294

        cls._from_hf(config, ckpt_path)
        return config

    @classmethod
    def _from_hf(cls, config: ModelConfig, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as f:
            root_config = json.load(f)

        talker_config = root_config.get("talker_config", root_config)

        config.inter_size = talker_config["intermediate_size"]
        config.attn_config.head_num = talker_config["num_attention_heads"]
        config.attn_config.kv_head_num = talker_config.get(
            "num_key_value_heads", config.attn_config.head_num
        )
        config.attn_config.size_per_head = (
            int(talker_config.get("head_dim"))
            if "head_dim" in talker_config
            else talker_config["hidden_size"] // config.attn_config.head_num
        )
        config.hidden_size = talker_config["hidden_size"]
        config.num_layers = talker_config["num_hidden_layers"]
        config.vocab_size = talker_config["vocab_size"]
        config.attn_config.rope_config.base = int(
            talker_config.get("rope_theta", 1000000)
        )
        config.attn_config.rope_config.dim = config.attn_config.size_per_head
        config.layernorm_eps = talker_config.get("rms_norm_eps", 1e-06)
        config.tie_word_embeddings = talker_config.get("tie_word_embeddings", False)
        config.config_dtype = talker_config.get("torch_dtype", None)

        embedding_size = talker_config.get("embedding_size")
        if embedding_size is not None:
            config.embedding_size = embedding_size

    @staticmethod
    def get_weight_cls():
        return Qwen2_5OmniTalkerWeight


register_model(
    "qwen2_5_omni_talker",
    Qwen2_5OmniTalker,
    ["Qwen2OmniTalkerForConditionalGeneration"],
)
