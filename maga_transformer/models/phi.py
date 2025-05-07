from maga_transformer.utils.util import get_config_from_path
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.model_weight import W, CkptWeightInfo, identity, transpose
from maga_transformer.model_loader.model_weight_info import ModelWeightInfo, ModelDeployWeightInfo
from maga_transformer.model_loader.weight_module import AtomicWeight
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, FfnWeight
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight
from maga_transformer.models.base_model import BaseModel
from maga_transformer.model_factory_register import register_model

class PhiWeightInfo(ModelDeployWeightInfo):
    def _get_weight_info(self):
        weights = [
            AtomicWeight(W.embedding, [CkptWeightInfo('layers.0.wte.weight', identity)], identity),
            AtomicWeight(W.lm_head, [CkptWeightInfo('layers.25.linear.weight', identity)], identity),
            AtomicWeight(W.lm_head_b, [CkptWeightInfo('layers.25.linear.bias', identity)], identity),
            AtomicWeight(W.final_ln_gamma, [CkptWeightInfo('layers.25.ln.weight', identity)], identity),
            AtomicWeight(W.final_ln_beta, [CkptWeightInfo('layers.25.ln.bias', identity)], identity),
        ]
        attn_config = self.attn_config
        ffn_config = self.ffn_config
        layer_weights = []
        for _ in range(self._num_layers):
            layer_weight = [
                AtomicWeight(W.pre_ln_beta, [CkptWeightInfo('layers.{i_1}.ln.bias', identity)], identity),
                AtomicWeight(W.pre_ln_gamma, [CkptWeightInfo('layers.{i_1}.ln.weight', identity)], identity),
                AttnAtomicWeight(W.attn_qkv_w, [CkptWeightInfo('layers.{i_1}.mixer.Wqkv.weight', identity)], transpose, config=attn_config),
                AttnAtomicWeight(W.attn_qkv_b, [CkptWeightInfo('layers.{i_1}.mixer.Wqkv.bias', identity)], identity, config=attn_config),
                AttnAtomicWeight(W.attn_o_w, [CkptWeightInfo('layers.{i_1}.mixer.out_proj.weight', identity)], transpose, config=attn_config),
                AttnAtomicWeight(W.attn_o_b, [CkptWeightInfo('layers.{i_1}.mixer.out_proj.bias', identity)], identity, config=attn_config),
                FfnWeight(sub_weights=[
                    FfnAtomicWeight(W.ffn_w3, [CkptWeightInfo('layers.{i_1}.mlp.fc1.weight', identity)], transpose, config=ffn_config),
                    FfnAtomicWeight(W.ffn_b3, [CkptWeightInfo('layers.{i_1}.mlp.fc1.bias', identity)], identity, config=ffn_config),
                    FfnAtomicWeight(W.ffn_w2, [CkptWeightInfo('layers.{i_1}.mlp.fc2.weight', identity)], transpose, config=ffn_config),
                    FfnAtomicWeight(W.ffn_b2, [CkptWeightInfo('layers.{i_1}.mlp.fc2.bias', identity)], identity, config=ffn_config)],
                    config=ffn_config)
            ]
            layer_weights.append(layer_weight)
        # close to falcon
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

class Phi(BaseModel):
    @staticmethod
    def get_weight_cls():
        return PhiWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config_dict = get_config_from_path(ckpt_path)
        size_per_head = int(config_dict.get('n_embd', 2048) / config_dict.get('n_head', 32))
        config = GptInitModelParameters(
            head_num=config_dict.get('n_head', 32),
            size_per_head=size_per_head,
            inter_size=4 * config_dict.get('n_embd', 2048),
            layer_num=config_dict.get('n_layer', 24),
            max_seq_len=config_dict.get('n_positions', 2048),
            vocab_size=config_dict.get('vocab_size', 32),
            rotary_embedding_dim=config_dict.get('rotary_dim', size_per_head),
            rotary_embedding_style=1,
            activation_type='gelu',
            has_positional_encoding=False,
            has_post_decoder_layernorm=True,
            has_lm_head_bias=True,
            tie_word_embeddings = config_dict.get('tie_word_embeddings', False))
        config.head_num_kv = config.head_num
        return config

register_model('phi', Phi)
