from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnWeight
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.base_model import BaseModel
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose
from rtp_llm.utils.util import get_config_from_path


class PhiWeightInfo(ModelDeployWeightInfo):
    def _get_weight_info(self):
        weights = [
            AtomicWeight(
                W.embedding, [CkptWeightInfo("layers.0.wte.weight", identity)], identity
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("layers.25.linear.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head_b,
                [CkptWeightInfo("layers.25.linear.bias", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("layers.25.ln.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta,
                [CkptWeightInfo("layers.25.ln.bias", identity)],
                identity,
            ),
        ]
        attn_config = self.attn_config
        ffn_config = self.ffn_config
        layer_weights = []
        for _ in range(self._num_layers):
            layer_weight = [
                AtomicWeight(
                    W.pre_ln_beta,
                    [CkptWeightInfo("layers.{i_1}.ln.bias", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.pre_ln_gamma,
                    [CkptWeightInfo("layers.{i_1}.ln.weight", identity)],
                    identity,
                ),
                AttnAtomicWeight(
                    W.attn_qkv_w,
                    [CkptWeightInfo("layers.{i_1}.mixer.Wqkv.weight", identity)],
                    transpose,
                    config=attn_config,
                ),
                AttnAtomicWeight(
                    W.attn_qkv_b,
                    [CkptWeightInfo("layers.{i_1}.mixer.Wqkv.bias", identity)],
                    identity,
                    config=attn_config,
                ),
                AttnAtomicWeight(
                    W.attn_o_w,
                    [CkptWeightInfo("layers.{i_1}.mixer.out_proj.weight", identity)],
                    transpose,
                    config=attn_config,
                ),
                AttnAtomicWeight(
                    W.attn_o_b,
                    [CkptWeightInfo("layers.{i_1}.mixer.out_proj.bias", identity)],
                    identity,
                    config=attn_config,
                ),
                FfnWeight(
                    sub_weights=[
                        FfnAtomicWeight(
                            W.ffn_w3,
                            [CkptWeightInfo("layers.{i_1}.mlp.fc1.weight", identity)],
                            transpose,
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_b3,
                            [CkptWeightInfo("layers.{i_1}.mlp.fc1.bias", identity)],
                            identity,
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w2,
                            [CkptWeightInfo("layers.{i_1}.mlp.fc2.weight", identity)],
                            transpose,
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_b2,
                            [CkptWeightInfo("layers.{i_1}.mlp.fc2.bias", identity)],
                            identity,
                            config=ffn_config,
                        ),
                    ],
                    config=ffn_config,
                ),
            ]
            layer_weights.append(layer_weight)
        # close to falcon
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class Phi(BaseModel):
    @staticmethod
    def get_weight_cls():
        return PhiWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_dict = get_config_from_path(ckpt_path)
        if config_dict is None:
            config_dict = {}
        size_per_head = int(
            config_dict.get("n_embd", 2048) / config_dict.get("n_head", 32)
        )
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.head_num = config_dict.get("n_head", 32)
        config.attn_config.size_per_head = size_per_head
        config.inter_size = 4 * config_dict.get("n_embd", 2048)
        config.num_layers = config_dict.get("n_layer", 24)
        config.max_seq_len = config_dict.get("n_positions", 2048)
        config.vocab_size = config_dict.get("vocab_size", 32)
        config.attn_config.rope_config.dim = config_dict.get("rotary_dim", size_per_head)
        config.attn_config.rope_config.style = 1
        config.attn_config.kv_head_num = config_dict.get("n_head", 32)
        config.norm_type = "layernorm"
        config.activation_type = "gelu"
        config.has_positional_encoding = False
        config.has_post_decoder_layernorm = True
        config.has_lm_head_bias = True
        config.tie_word_embeddings = config_dict.get("tie_word_embeddings", False)
        config.config_dtype = config_dict.get("torch_dtype", None)
        return config


register_model("phi", Phi)
