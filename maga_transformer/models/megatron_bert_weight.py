import torch
from typing import List, NamedTuple, Any
from pydantic import BaseModel

from maga_transformer.utils.model_weight import W, CkptWeightInfo, concat_0, transpose, identity
from maga_transformer.model_loader.weight_module import WeightModule, AtomicWeight
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, FfnWeight, FfnConfig
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from maga_transformer.model_loader.model_weight_info import ModelWeightInfo, ModelDeployWeightInfo

def merge_qkv_hf(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_weight = torch.concat([q.T, k.T, v.T], dim=1).contiguous()
    return qkv_weight

class HfWeightNames(BaseModel):
    Q_W: str = 'encoder.layer.{i}.attention.self.query.weight'
    Q_B: str = 'encoder.layer.{i}.attention.self.query.bias'
    K_W: str = 'encoder.layer.{i}.attention.self.key.weight'
    K_B: str = 'encoder.layer.{i}.attention.self.key.bias'
    V_W: str = 'encoder.layer.{i}.attention.self.value.weight'
    V_B: str = 'encoder.layer.{i}.attention.self.value.bias'
    O_W: str = 'encoder.layer.{i}.attention.output.dense.weight'
    O_B: str = 'encoder.layer.{i}.attention.output.dense.bias'

    POST_LN_W: str = 'encoder.layer.{i}.ln.weight'
    POST_LN_B: str = 'encoder.layer.{i}.ln.bias'

    FFN_PRE_LAYERNORM_W: str = 'encoder.layer.{i}.ln.weight'
    FFN_PRE_LAYERNORM_B: str = 'encoder.layer.{i}.ln.bias'
    FFN_INTER_DENSE_W: str = 'encoder.layer.{i}.intermediate.dense.weight'
    FFN_INTER_DENSE_B: str = 'encoder.layer.{i}.intermediate.dense.bias'
    FFN_OUTPUT_DENSE_W: str = 'encoder.layer.{i}.output.dense.weight'
    FFN_OUTPUT_DENSE_B: str = 'encoder.layer.{i}.output.dense.bias'

    FFN_NORM: str = 'model.layers.{i}.post_attention_layernorm.weight'

    TOKEN_EMBEDDING: str = 'embeddings.word_embeddings.weight'
    POSITION_EMBEDDING: str = 'embeddings.position_embeddings.weight'
    TOKEN_TYPE_EMBEDDING: str = 'embeddings.token_type_embeddings.weight'

class MegatronBertWeightInfo(ModelDeployWeightInfo):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.prefix_name = 'bert'
        self.model_name = 'megatron_bert'
        self._names = HfWeightNames()

    @staticmethod
    def _contains(keys: List[str], val: str):
        for key in keys:
            if val in key:
                return True
        return False

    def _append_name_prefix(self, names: HfWeightNames, weight_keys: List[str]):
        prefix = self.prefix_name + '.'
        if self._contains(weight_keys, prefix):
            for key, value in names.model_dump().items():
                setattr(names, key, prefix + value)

    def _process_meta(self, meta_dicts: Any, weight_keys: List[str]):
        self._append_name_prefix(self._names, weight_keys)
        if self._contains(weight_keys, 'beta') and self._contains(weight_keys, 'gamma'):
            self._names.POST_LN_W = self._names.POST_LN_W.replace('weight', 'gamma')
            self._names.POST_LN_B = self._names.POST_LN_B.replace('bias', 'beta')

    def _get_weight_info(self):
        weights: List[WeightModule] = [
            AtomicWeight(W.embedding, [CkptWeightInfo(self._names.TOKEN_EMBEDDING)]),
            AtomicWeight(W.positional_embedding, [CkptWeightInfo(self._names.POSITION_EMBEDDING)]),
            AtomicWeight(W.token_type_embedding, [CkptWeightInfo(self._names.TOKEN_TYPE_EMBEDDING)]),
        ]
        layer_weights = [
            AtomicWeight(W.pre_ln_gamma, [CkptWeightInfo('bert.encoder.layer.{i}.attention.ln.weight', identity)],
                       identity),
            AtomicWeight(W.pre_ln_beta, [CkptWeightInfo('bert.encoder.layer.{i}.attention.ln.bias', identity)],
                       identity),

            AttnAtomicWeight(W.attn_qkv_w, [
                CkptWeightInfo(self._names.Q_W),
                CkptWeightInfo(self._names.K_W),
                CkptWeightInfo(self._names.V_W)], merge_qkv_hf),

            AttnAtomicWeight(W.attn_qkv_b, [
                CkptWeightInfo(self._names.Q_B),
                CkptWeightInfo(self._names.K_B),
                CkptWeightInfo(self._names.V_B)], concat_0),

            AttnAtomicWeight(W.attn_o_w, [CkptWeightInfo(self._names.O_W)], transpose),
            AttnAtomicWeight(W.attn_o_b, [CkptWeightInfo(self._names.O_B)]),

            AtomicWeight(W.post_ln_beta, [CkptWeightInfo(self._names.POST_LN_B)]),
            AtomicWeight(W.post_ln_gamma, [CkptWeightInfo(self._names.POST_LN_W)]),

            FfnAtomicWeight(W.ffn_w3, [CkptWeightInfo(self._names.FFN_INTER_DENSE_W)], transpose),
            FfnAtomicWeight(W.ffn_b3, [CkptWeightInfo(self._names.FFN_INTER_DENSE_B)]),

            FfnAtomicWeight(W.ffn_w2, [CkptWeightInfo(self._names.FFN_OUTPUT_DENSE_W)], transpose),
            FfnAtomicWeight(W.ffn_b2, [CkptWeightInfo(self._names.FFN_OUTPUT_DENSE_B)]),
        ]
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)
