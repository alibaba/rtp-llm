import functools
from ..model_loader.weight_module import AtomicWeight
from maga_transformer.utils.model_weight import W, ModelWeightInfo, CkptWeightInfo, \
    identity, transpose, trans_qkv, trans_qkv_b
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, FfnWeight, FfnConfig
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from maga_transformer.model_loader.model_weight_info import ModelWeightInfo, ModelDeployWeightInfo

class GptWeightInfo(ModelDeployWeightInfo):
    def _get_weight_info(self):
        weights = [
            AtomicWeight(W.embedding, [CkptWeightInfo('word_embeddings.weight', identity)], False, identity),
            AtomicWeight(W.pre_decoder_ln_gamma, [CkptWeightInfo('word_embeddings_layernorm.weight', identity)], False, identity),
            AtomicWeight(W.pre_decoder_ln_beta, [CkptWeightInfo('word_embeddings_layernorm.bias', identity)], False, identity),
            AtomicWeight(W.final_ln_gamma, [CkptWeightInfo('ln_f.weight', identity)], False, identity),
            AtomicWeight(W.final_ln_beta, [CkptWeightInfo('ln_f.bias', identity)], False, identity),
        ]
        attn_config: AttnConfig = self.attn_config
        ffn_config: FfnConfig =  self.ffn_config

        layer_weights = []
        for _ in range(self._num_layers):
            layer_weight = [
                AtomicWeight(W.pre_ln_beta, [CkptWeightInfo('model.layers.{i}.input_layernorm.bias', identity)], identity),

                AtomicWeight(W.pre_ln_gamma, [CkptWeightInfo('model.layers.{i}.input_layernorm.weight', identity)], identity),

                AttnAtomicWeight(W.attn_qkv_w, [CkptWeightInfo('model.layers.{i}.attention.query_key_value.weight', identity)],
                                 functools.partial(trans_qkv, hidden_size=self._hidden_size, head_num=self._head_num), config=attn_config),

                AttnAtomicWeight(W.attn_qkv_b, [CkptWeightInfo('model.layers.{i}.attention.query_key_value.bias', identity)],
                                 functools.partial(trans_qkv_b, hidden_size=self._hidden_size, head_num=self._head_num), config=attn_config),

                AttnAtomicWeight(W.attn_o_w, [CkptWeightInfo('model.layers.{i}.self_attention.dense.weight', identity)], transpose, config=attn_config),

                AttnAtomicWeight(W.attn_o_b, [CkptWeightInfo('model.layers.{i}.self_attention.dense.bias', identity)], identity, config=attn_config),

                FfnWeight(sub_weights=[
                    FfnAtomicWeight(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.dense_h_to_4h.weight', identity)], transpose, config=ffn_config),

                    FfnAtomicWeight(W.ffn_b1, [CkptWeightInfo('model.layers.{i}.mlp.dense_h_to_4h.bias', identity)], identity, config=ffn_config),

                    FfnAtomicWeight(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.dense_4h_to_h.weight', identity)], transpose, config=ffn_config),

                    FfnAtomicWeight(W.ffn_b2, [CkptWeightInfo('model.layers.{i}.mlp.dense_4h_to_h.bias', identity)], identity, config=ffn_config)
                    ], config=ffn_config),

                AtomicWeight(W.post_ln_beta, [CkptWeightInfo('model.layers.{i}.post_attention_layernorm.bias', identity)], identity),

                AtomicWeight(W.post_ln_gamma, [CkptWeightInfo('model.layers.{i}.post_attention_layernorm.weight', identity)], identity),
            ]
            layer_weights.append(layer_weight)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)
