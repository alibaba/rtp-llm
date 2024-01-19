
from typing import List
import functools
import torch

from maga_transformer.utils.chatglm2_quantization import extract_weight_to_half

from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo, ModelDeployWeightInfo, \
    CkptWeightInfo, concat_1, identity, transpose, trans_qkv, trans_qkv_b, trans_lora_qkv



class GlmWeightInfo(ModelDeployWeightInfo):
    def _process_meta(self, meta_dicts, weight_keys):
        if 'transformer.prefix_encoder.embedding.weight' in weight_keys:
            self._has_prefix_encoder = True

    def _get_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('transformer.word_embeddings.weight', concat_1)], identity),
            WeightInfo(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('transformer.final_layernorm.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [CkptWeightInfo('transformer.final_layernorm.bias', identity)], identity),
        ]

        if self._has_prefix_encoder:
            weights.append(WeightInfo(W.prefix_w, [CkptWeightInfo('transformer.prefix_encoder.embedding.weight', identity)], identity))

        layer_weights = [
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('transformer.layers.{i}.input_layernorm.weight', identity)],
                       identity),

            WeightInfo(W.pre_ln_beta, [CkptWeightInfo('transformer.layers.{i}.input_layernorm.bias', identity)],
                       identity),

            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('transformer.layers.{i}.attention.query_key_value.weight', identity)],
                       functools.partial(trans_qkv, hidden_size=self._hidden_size, head_num=self._head_num)),

            WeightInfo(W.attn_qkv_b, [CkptWeightInfo('transformer.layers.{i}.attention.query_key_value.bias', identity)],
                       functools.partial(trans_qkv_b, hidden_size=self._hidden_size, head_num=self._head_num)),

            WeightInfo(W.attn_o_w, [CkptWeightInfo('transformer.layers.{i}.attention.dense.weight', identity)],
                       transpose),

            WeightInfo(W.attn_o_b, [CkptWeightInfo('transformer.layers.{i}.attention.dense.bias', identity)],
                       identity),

            WeightInfo(W.ffn_w1, [CkptWeightInfo('transformer.layers.{i}.mlp.dense_h_to_4h.weight', identity)],
                       transpose),

            WeightInfo(W.ffn_b1, [CkptWeightInfo('transformer.layers.{i}.mlp.dense_h_to_4h.bias', identity)],
                       identity),

            WeightInfo(W.ffn_w2, [CkptWeightInfo('transformer.layers.{i}.mlp.dense_4h_to_h.weight', identity)],
                       transpose),

            WeightInfo(W.ffn_b2, [CkptWeightInfo('transformer.layers.{i}.mlp.dense_4h_to_h.bias', identity)],
                       identity),

            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('transformer.layers.{i}.post_attention_layernorm.weight', identity)],
                       identity),

            WeightInfo(W.post_ln_beta, [CkptWeightInfo('transformer.layers.{i}.post_attention_layernorm.bias', identity)],
                       identity),
        ]

        assert self._src_quantization_bit == 0 or self._src_quantization_bit in [4, 8]
        if self._src_quantization_bit in [4, 8]:
            for idx, layer_weight in enumerate(layer_weights):
                new_weight = layer_weight.weights + [CkptWeightInfo(layer_weight.weights[0].name + '_scale', functools.partial(identity, allow_empty = True))]
                layer_weights[idx] = WeightInfo(layer_weight.name, new_weight,
                                                functools.partial(extract_weight_to_half, source_bit_width = self._src_quantization_bit, sufix_func = layer_weight.process_fun))


        model_weight_info = ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=W.gpt_style_tp_strategy)
        model_weight_info.set_lora(qkv_fun=functools.partial(trans_lora_qkv, head_num=self._head_num, head_size=self._size_per_head))
        return model_weight_info
