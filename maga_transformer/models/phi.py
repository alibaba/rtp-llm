from maga_transformer.utils.util import get_config_from_path
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.model_weight import W, WeightInfo, \
    ModelWeightInfo, ModelDeployWeightInfo, CkptWeightInfo, identity, transpose
from maga_transformer.models.gpt import GPT
from maga_transformer.model_factory_register import register_model

class PhiWeightInfo(ModelDeployWeightInfo):
    def _get_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('layers.0.wte.weight', identity)], identity),
            WeightInfo(W.lm_head, [CkptWeightInfo('layers.25.linear.weight', identity)], identity),
            WeightInfo(W.lm_head_b, [CkptWeightInfo('layers.25.linear.bias', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('layers.25.ln.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [CkptWeightInfo('layers.25.ln.bias', identity)], identity),
        ]
        layer_weights = [
            WeightInfo(W.pre_ln_beta, [CkptWeightInfo('layers.{i_1}.ln.bias', identity)], identity),
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('layers.{i_1}.ln.weight', identity)], identity),
            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('layers.{i_1}.mixer.Wqkv.weight', identity)], transpose),
            WeightInfo(W.attn_qkv_b, [CkptWeightInfo('layers.{i_1}.mixer.Wqkv.bias', identity)], identity),
            WeightInfo(W.attn_o_w, [CkptWeightInfo('layers.{i_1}.mixer.out_proj.weight', identity)], transpose),
            WeightInfo(W.attn_o_b, [CkptWeightInfo('layers.{i_1}.mixer.out_proj.bias', identity)], identity),
            WeightInfo(W.ffn_w1, [CkptWeightInfo('layers.{i_1}.mlp.fc1.weight', identity)], transpose),
            WeightInfo(W.ffn_b1, [CkptWeightInfo('layers.{i_1}.mlp.fc1.bias', identity)], identity),
            WeightInfo(W.ffn_w2, [CkptWeightInfo('layers.{i_1}.mlp.fc2.weight', identity)], transpose),
            WeightInfo(W.ffn_b2, [CkptWeightInfo('layers.{i_1}.mlp.fc2.bias', identity)], identity),
        ]
        # close to falcon
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=W.gpt_style_tp_strategy)

class Phi(GPT):
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
            has_lm_head_bias=True)
        config.head_num_kv = config.head_num
        return config

register_model('phi', Phi)
