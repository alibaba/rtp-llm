import functools
import logging

from maga_transformer.utils.model_weight import (W, CkptWeightInfo, concat_1,
                                                 concat_0, identity, zeros, transpose, merge_qkv_b)
from maga_transformer.models.multimodal.multimodal_mixin import BaseVitWeights, BaseMultiModalWeightInfo
from maga_transformer.models.llama_weight import qkv_rerange, merge_qkv_hf
from maga_transformer.model_loader.model_weight_info import ModelDeployWeightInfo, AtomicWeight, ModelWeightInfo
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, FfnWeight

class LlamaWeightNames:
    WQ = "language_model.model.layers.{i}.self_attn.q_proj.weight"
    WK = "language_model.model.layers.{i}.self_attn.k_proj.weight"
    WV = "language_model.model.layers.{i}.self_attn.v_proj.weight"
    WO = "language_model.model.layers.{i}.self_attn.o_proj.weight"
    FFW1 = "language_model.model.layers.{i}.mlp.gate_proj.weight"
    FFW2 = "language_model.model.layers.{i}.mlp.down_proj.weight"
    FFW3 = "language_model.model.layers.{i}.mlp.up_proj.weight"
    ATTEN_NORM = "language_model.model.layers.{i}.input_layernorm.weight"
    FFN_NORM = "language_model.model.layers.{i}.post_attention_layernorm.weight"
    TOKEN_EMBEDDING = "language_model.model.embed_tokens.weight"
    NORM = "language_model.model.norm.weight"
    OUTPUT = "language_model.lm_head.weight"

class Internlm2WeightNames:
    W_QKV = "language_model.model.layers.{i}.attention.wqkv.weight"
    WO = "language_model.model.layers.{i}.attention.wo.weight"
    FFW1 = "language_model.model.layers.{i}.feed_forward.w1.weight"
    FFW2 = "language_model.model.layers.{i}.feed_forward.w2.weight"
    FFW3 = "language_model.model.layers.{i}.feed_forward.w3.weight"
    ATTEN_NORM = "language_model.model.layers.{i}.attention_norm.weight"
    FFN_NORM = "language_model.model.layers.{i}.ffn_norm.weight"
    TOKEN_EMBEDDING = "language_model.model.tok_embeddings.weight"
    NORM = "language_model.model.norm.weight"
    OUTPUT = "language_model.output.weight"

class QwenWeightName(LlamaWeightNames):
    BQ = "language_model.model.layers.{i}.self_attn.q_proj.bias"
    BK = "language_model.model.layers.{i}.self_attn.k_proj.bias"
    BV = "language_model.model.layers.{i}.self_attn.v_proj.bias"

class InternVLVitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = ""
        self._ft_prefix = "self.mm_part."

class InternVLWeightInfo(ModelDeployWeightInfo, BaseMultiModalWeightInfo):
    def __init__(self, config, tp_size, tp_rank):
        ModelDeployWeightInfo.__init__(self, config, tp_size, tp_rank)
        BaseMultiModalWeightInfo.__init__(self, config)
        self._names = None
        self._merge_qkv = None
        self._merge_qkv_b = None

    def _process_meta(self, meta_dicts, weight_keys):
        if Internlm2WeightNames.W_QKV.format(i="0") in weight_keys:
            logging.info("load internlm2 style weight for llm part")
            self._names = Internlm2WeightNames
            self._merge_qkv = qkv_rerange
        elif QwenWeightName.BQ.format(i="0") in weight_keys:
            logging.info("load qwen style weight for llm part")
            self._names = QwenWeightName
            self._merge_qkv = merge_qkv_hf
            self._merge_qkv_b = merge_qkv_b
        elif LlamaWeightNames.WQ.format(i="0") in weight_keys:
            logging.info("load llama style weight for llm part")
            self._names = LlamaWeightNames
            self._merge_qkv = merge_qkv_hf
        else:
            raise Exception("unknown weights format")

    def _get_weight_info(self):
        weights = [
            AtomicWeight(W.embedding, [CkptWeightInfo(self._names.TOKEN_EMBEDDING, concat_1)], identity),
            AtomicWeight(W.final_ln_gamma, [CkptWeightInfo(self._names.NORM, identity)], identity),
            AtomicWeight(W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])),
            AtomicWeight(W.lm_head, [CkptWeightInfo(self._names.OUTPUT, concat_0)], identity)
        ]
        attn_config = self.attn_config
        ffn_config = self.ffn_config
        layer_weights = []
        for _ in range(self._num_layers):
                    
            layer_weight = [
                AtomicWeight(W.pre_ln_gamma, [CkptWeightInfo(self._names.ATTEN_NORM, identity)], identity),
                AtomicWeight(W.post_ln_gamma, [CkptWeightInfo(self._names.FFN_NORM, identity)], identity),
                AttnAtomicWeight(W.attn_o_w, [CkptWeightInfo(self._names.WO, concat_1)], transpose),
                FfnWeight(sub_weights=[
                FfnAtomicWeight(W.ffn_w1, [CkptWeightInfo(self._names.FFW1, concat_0)], transpose, config=ffn_config),
                FfnAtomicWeight(W.ffn_w3, [CkptWeightInfo(self._names.FFW3, concat_0)], transpose, config=ffn_config),
                FfnAtomicWeight(W.ffn_w2, [CkptWeightInfo(self._names.FFW2, concat_1)], transpose, config=ffn_config)
            ], config=ffn_config)
            ]

            func = functools.partial(self._merge_qkv,
                                        hidden_size=self._hidden_size,
                                        head_num_kv=self._head_num_kv,
                                        head_num=self._head_num)

            if self._names == Internlm2WeightNames:
                layer_weight.append(
                    AttnAtomicWeight(
                        W.attn_qkv_w,
                        [CkptWeightInfo(self._names.W_QKV, identity)],
                        func, config=attn_config))
            else:
                layer_weight.append(
                    AttnAtomicWeight(
                        W.attn_qkv_w,
                        [
                            CkptWeightInfo(self._names.WQ, concat_0),
                            CkptWeightInfo(self._names.WK, concat_0),
                            CkptWeightInfo(self._names.WV, concat_0)
                        ],
                        func, config=attn_config))
                if self._names == QwenWeightName:
                    layer_weight.append(
                        AttnAtomicWeight(W.attn_qkv_b,
                                [CkptWeightInfo(self._names.BQ, identity),
                                CkptWeightInfo(self._names.BK, identity),
                                CkptWeightInfo(self._names.BV, identity)],
                                functools.partial(self._merge_qkv_b), config=attn_config)
                    )
            layer_weights.append(layer_weight)


        model_weights = ModelWeightInfo(layer_weights=layer_weights, weights=weights)
        model_weights = self._get_vit_info(model_weights)
        return model_weights