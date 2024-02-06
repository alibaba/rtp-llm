import functools

from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo, \
    ModelDeployWeightInfo, CkptWeightInfo,\
    identity, zeros, transpose, trans_qkv, trans_qkv_b

class GPTNeoxWeight(ModelDeployWeightInfo):
    def __init__(self, config, tp_size, tp_rank):
        super().__init__(config, tp_size, tp_rank)
        self.norm = config.norm_type

    def _get_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('gpt_neox.embed_in.weight', identity)], identity),
            WeightInfo(W.lm_head, [CkptWeightInfo('embed_out.weight', identity)], identity)
        ]

        layer_weights = [
            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('gpt_neox.layers.{i}.attention.query_key_value.weight', identity)],
                       functools.partial(trans_qkv, hidden_size=self._hidden_size, head_num=self._head_num)),

            WeightInfo(W.attn_qkv_b, [CkptWeightInfo('gpt_neox.layers.{i}.attention.query_key_value.bias', identity)],
                       functools.partial(trans_qkv_b, hidden_size=self._hidden_size, head_num=self._head_num)),

            WeightInfo(W.attn_o_w, [CkptWeightInfo('gpt_neox.layers.{i}.attention.dense.weight', identity)],
                       transpose),

            WeightInfo(W.attn_o_b, [CkptWeightInfo('gpt_neox.layers.{i}.attention.dense.bias', identity)],
                       identity),

            WeightInfo(W.ffn_w1, [CkptWeightInfo('gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight', identity)],
                       transpose),

            WeightInfo(W.ffn_b1, [CkptWeightInfo('gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias', identity)],
                       identity),

            WeightInfo(W.ffn_w2, [CkptWeightInfo('gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight', identity)],
                       transpose),

            WeightInfo(W.ffn_b2, [CkptWeightInfo('gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias', identity)],
                       identity),
        ]

        # default use parallel residual: x = x + attn(ln1(x)) + mlp(ln2(x))

        if self.norm == 'rmsnorm':
            weights.extend([
                WeightInfo(W.final_ln_gamma, [CkptWeightInfo('gpt_neox.final_layer_norm.scale', identity)], identity),
                WeightInfo(W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size]))
            ])
            layer_weights.extend([
                WeightInfo(W.pre_attn_ln_gamma, [CkptWeightInfo('gpt_neox.layers.{i}.input_layernorm.scale', identity)],
                        identity),
                WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('gpt_neox.layers.{i}.post_attention_layernorm.scale', identity)],
                        identity)
            ])
        elif self.norm == 'layernorm':
            weights.extend([
                WeightInfo(W.final_ln_gamma, [CkptWeightInfo('gpt_neox.final_layer_norm.weight', identity)], identity),
                WeightInfo(W.final_ln_beta, [CkptWeightInfo('gpt_neox.final_layer_norm.bias', identity)], identity)
            ])
            layer_weights.extend([
                WeightInfo(W.pre_attn_ln_gamma, [CkptWeightInfo('gpt_neox.layers.{i}.input_layernorm.weight', identity)],
                        identity),
                WeightInfo(W.pre_attn_ln_beta, [CkptWeightInfo('gpt_neox.layers.{i}.input_layernorm.bias', identity)],
                        identity),
                WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('gpt_neox.layers.{i}.post_attention_layernorm.weight', identity)],
                        identity),
                WeightInfo(W.pre_ln_beta, [CkptWeightInfo('gpt_neox.layers.{i}.post_attention_layernorm.bias', identity)],
                        identity)
            ])
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=W.gpt_style_tp_strategy)

class GPTNeox13BWeight(ModelDeployWeightInfo):
    def _get_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('gpt_neox.embed_in.weight', identity)], identity),
            WeightInfo(W.lm_head, [CkptWeightInfo('embed_out.weight', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('gpt_neox.final_layer_norm.scale', identity)], identity),
            WeightInfo(W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])),
        ]

        layer_weights = [
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('gpt_neox.layers.{i}.input_layernorm.scale', identity)],
                       identity),

            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('gpt_neox.layers.{i}.attention.query_key_value.weight', identity)],
                       functools.partial(trans_qkv, hidden_size=self._hidden_size, head_num=self._head_num)),

            WeightInfo(W.attn_qkv_b, [CkptWeightInfo('gpt_neox.layers.{i}.attention.query_key_value.bias', identity)],
                       functools.partial(trans_qkv_b, hidden_size=self._hidden_size, head_num=self._head_num)),

            WeightInfo(W.attn_o_w, [CkptWeightInfo('gpt_neox.layers.{i}.attention.dense.weight', identity)],
                       transpose),

            WeightInfo(W.attn_o_b, [CkptWeightInfo('gpt_neox.layers.{i}.attention.dense.bias', identity)],
                       identity),

            WeightInfo(W.ffn_w1, [CkptWeightInfo('gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight', identity)],
                       transpose),

            WeightInfo(W.ffn_b1, [CkptWeightInfo('gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias', identity)],
                       identity),

            WeightInfo(W.ffn_w2, [CkptWeightInfo('gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight', identity)],
                       transpose),

            WeightInfo(W.ffn_b2, [CkptWeightInfo('gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias', identity)],
                       identity),

            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('gpt_neox.layers.{i}.post_attention_layernorm.scale', identity)],
                       identity),
        ]

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=W.gpt_style_tp_strategy)