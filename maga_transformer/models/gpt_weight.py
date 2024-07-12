import functools
from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo, \
    ModelDeployWeightInfo, CkptWeightInfo, \
    identity, transpose, trans_qkv, trans_qkv_b

class GptWeightInfo(ModelDeployWeightInfo):
    def _get_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('word_embeddings.weight', identity)], False, identity),
            WeightInfo(W.pre_decoder_ln_gamma, [CkptWeightInfo('word_embeddings_layernorm.weight', identity)], False, identity),
            WeightInfo(W.pre_decoder_ln_beta, [CkptWeightInfo('word_embeddings_layernorm.bias', identity)], False, identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('ln_f.weight', identity)], False, identity),
            WeightInfo(W.final_ln_beta, [CkptWeightInfo('ln_f.bias', identity)], False, identity),
        ]

        layer_weights = [
            WeightInfo(W.pre_ln_beta, [CkptWeightInfo('model.layers.{i}.input_layernorm.bias', identity)],
                       False, identity),

            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('model.layers.{i}.input_layernorm.weight', identity)],
                       False, identity),

            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('model.layers.{i}.attention.query_key_value.weight', identity)],
                       True, functools.partial(trans_qkv, hidden_size=self._hidden_size, head_num=self._head_num)),

            WeightInfo(W.attn_qkv_b, [CkptWeightInfo('model.layers.{i}.attention.query_key_value.bias', identity)],
                       False, functools.partial(trans_qkv_b, hidden_size=self._hidden_size, head_num=self._head_num)),

            WeightInfo(W.attn_o_w, [CkptWeightInfo('model.layers.{i}.self_attention.dense.weight', identity)],
                       True, transpose),

            WeightInfo(W.attn_o_b, [CkptWeightInfo('model.layers.{i}.self_attention.dense.bias', identity)],
                       False, identity),

            WeightInfo(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.dense_h_to_4h.weight', identity)],
                       True, transpose),

            WeightInfo(W.ffn_b1, [CkptWeightInfo('model.layers.{i}.mlp.dense_h_to_4h.bias', identity)],
                       False, identity),

            WeightInfo(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.dense_4h_to_h.weight', identity)],
                       True, transpose),

            WeightInfo(W.ffn_b2, [CkptWeightInfo('model.layers.{i}.mlp.dense_4h_to_h.bias', identity)],
                       False, identity),

            WeightInfo(W.post_ln_beta, [CkptWeightInfo('model.layers.{i}.post_attention_layernorm.bias', identity)],
                       False, identity),

            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('model.layers.{i}.post_attention_layernorm.weight', identity)],
                       False, identity),
        ]

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=self._get_gpt_style_tp_strategy())
