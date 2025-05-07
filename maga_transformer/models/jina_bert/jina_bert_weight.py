import torch
import functools
from typing import List, Any
from pydantic import BaseModel

from maga_transformer.utils.model_weight import W, CkptWeightInfo, concat_0, transpose
from maga_transformer.model_loader.weight_module import WeightModule, AtomicWeight
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, FfnWeight, FfnConfig
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from maga_transformer.model_loader.model_weight_info import ModelWeightInfo, ModelDeployWeightInfo

def merge_qkv_hf(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_weight = torch.concat([q.T, k.T, v.T], dim=1).contiguous()
    return qkv_weight


def merge_qkv_transpose_concat0(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_weight = torch.concat([q.T, k.T, v.T], dim=0).contiguous()
    return qkv_weight

def slice_index_transepose(ts: List[torch.Tensor], index: int, inter_size: int):
    t = ts[0]
    return t[index * inter_size: (index + 1) * inter_size, :].transpose(0, 1).contiguous()

# from [torch.Size(1), torch.Size(1), torch.Size(1)] to torch.Size(3 * hidden_size)
def expand_scale(ts: List[torch.Tensor], hidden_size:int):
    new_ts: List[torch.Tensor] = []
    for t in ts:
        assert t.shape == torch.Size([1]), "tensor shape should be [1], actual: " + str(t.shape)
        new_ts.append(t.expand(hidden_size))
    return torch.concat(new_ts, dim=-1)

class BaseWeightNames(BaseModel):
    Q_W: str = 'encoder.layer.{i}.attention.self.query.weight'
    Q_B: str = 'encoder.layer.{i}.attention.self.query.bias'
    K_W: str = 'encoder.layer.{i}.attention.self.key.weight'
    K_B: str = 'encoder.layer.{i}.attention.self.key.bias'
    V_W: str = 'encoder.layer.{i}.attention.self.value.weight'
    V_B: str = 'encoder.layer.{i}.attention.self.value.bias'
    O_W: str = 'encoder.layer.{i}.attention.output.dense.weight'
    O_B: str = 'encoder.layer.{i}.attention.output.dense.bias'

    POST_LN_W: str = 'encoder.layer.{i}.attention.output.LayerNorm.weight'
    POST_LN_B: str = 'encoder.layer.{i}.attention.output.LayerNorm.bias'

    TOKEN_EMBEDDING: str = 'embeddings.word_embeddings.weight'
    POSITION_EMBEDDING: str = 'embeddings.position_embeddings.weight'
    TOKEN_TYPE_EMBEDDING: str = 'embeddings.token_type_embeddings.weight'
    EMB_NORM_W: str = 'embeddings.LayerNorm.weight'
    EMB_NORM_B: str = 'embeddings.LayerNorm.bias'

class QKNormHfWeightNames(BaseWeightNames):
    Q_LN_W: str = 'encoder.layer.{i}.attention.self.layer_norm_q.weight'
    Q_LN_B: str = 'encoder.layer.{i}.attention.self.layer_norm_q.bias'
    K_LN_W: str = 'encoder.layer.{i}.attention.self.layer_norm_k.weight'
    K_LN_B: str = 'encoder.layer.{i}.attention.self.layer_norm_k.bias'

    POST_LN_2_W: str = "encoder.layer.{i}.layer_norm_1.weight"
    POST_LN_2_B: str = "encoder.layer.{i}.layer_norm_1.bias"

    FFN_GATE_W: str = 'encoder.layer.{i}.mlp.up_gated_layer.weight'
    FFN_DOWN_W: str = 'encoder.layer.{i}.mlp.down_layer.weight'
    FFN_DOWN_B: str = 'encoder.layer.{i}.mlp.down_layer.bias'
    FFN_OUTPUT_LAYERNORM_W: str = "encoder.layer.{i}.layer_norm_2.weight"
    FFN_OUTPUT_LAYERNORM_B: str = "encoder.layer.{i}.layer_norm_2.bias"

class JinaBertWeightInfo(ModelDeployWeightInfo):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model_name = 'bert'
        self._names = QKNormHfWeightNames()

    @staticmethod
    def _contains(keys: List[str], val: str):
        for key in keys:
            if val in key:
                return True
        return False

    def _append_name_prefix(self, names: QKNormHfWeightNames, weight_keys: List[str]):
        prefix = self.model_name + '.'
        if self._contains(weight_keys, prefix):
            for key, value in names.model_dump().items():
                setattr(names, key, prefix + value)
        if self._contains(weight_keys, 'beta') and self._contains(weight_keys, 'gamma'):
            names.POST_LN_W = names.POST_LN_W.replace('weight', 'gamma')
            names.POST_LN_B = names.POST_LN_B.replace('bias', 'beta')
            names.FFN_OUTPUT_LAYERNORM_W = names.FFN_OUTPUT_LAYERNORM_W.replace('weight', 'gamma')
            names.FFN_OUTPUT_LAYERNORM_B = names.FFN_OUTPUT_LAYERNORM_B.replace('bias', 'beta')
            names.EMB_NORM_W = names.EMB_NORM_W.replace('weight', 'gamma')
            names.EMB_NORM_B = names.EMB_NORM_B.replace('bias', 'beta')

    def _process_meta(self, meta_dicts: Any, weight_keys: List[str]):
        self._append_name_prefix(self._names, weight_keys)


    def _get_base_weight_info(self) -> List[WeightModule]:
        return [
            AtomicWeight(W.embedding, [CkptWeightInfo(self._names.TOKEN_EMBEDDING)]),
            AtomicWeight(W.token_type_embedding, [CkptWeightInfo(self._names.TOKEN_TYPE_EMBEDDING)]),
            AtomicWeight(W.pre_decoder_ln_beta, [CkptWeightInfo(self._names.EMB_NORM_B)]),
            AtomicWeight(W.pre_decoder_ln_gamma, [CkptWeightInfo(self._names.EMB_NORM_W)]),
        ]

    def _get_weight_info(self):
        weights = self._get_base_weight_info()
        layer_weights = [
            AttnAtomicWeight(W.attn_qkv_w, [
                CkptWeightInfo(self._names.Q_W),
                CkptWeightInfo(self._names.K_W),
                CkptWeightInfo(self._names.V_W)], merge_qkv_hf),

            AttnAtomicWeight(W.attn_qkv_b, [
                CkptWeightInfo(self._names.Q_B),
                CkptWeightInfo(self._names.K_B),
                CkptWeightInfo(self._names.V_B)], concat_0),

            AtomicWeight(W.q_ln_gamma, [CkptWeightInfo(self._names.Q_LN_W)]),
            AtomicWeight(W.q_ln_beta, [CkptWeightInfo(self._names.Q_LN_B)]),
            AtomicWeight(W.k_ln_gamma, [CkptWeightInfo(self._names.K_LN_W)]),
            AtomicWeight(W.k_ln_beta, [CkptWeightInfo(self._names.K_LN_B)]),

            AttnAtomicWeight(W.attn_o_w, [CkptWeightInfo(self._names.O_W)], transpose),
            AttnAtomicWeight(W.attn_o_b, [CkptWeightInfo(self._names.O_B)]),

            AtomicWeight(W.post_ln_beta, [CkptWeightInfo(self._names.POST_LN_B)]),
            AtomicWeight(W.post_ln_gamma, [CkptWeightInfo(self._names.POST_LN_W)]),

            AtomicWeight(W.post_ln_2_gamma, [CkptWeightInfo(self._names.POST_LN_2_W)]),
            AtomicWeight(W.post_ln_2_beta, [CkptWeightInfo(self._names.POST_LN_2_B)]),

            # gate
            FfnAtomicWeight(W.ffn_w1, [CkptWeightInfo(self._names.FFN_GATE_W)], functools.partial(slice_index_transepose, index=1, inter_size=self._inter_size)),

            # up
            FfnAtomicWeight(W.ffn_w3, [CkptWeightInfo(self._names.FFN_GATE_W)], functools.partial(slice_index_transepose, index=0, inter_size=self._inter_size)),

            # down
            FfnAtomicWeight(W.ffn_w2, [CkptWeightInfo(self._names.FFN_DOWN_W)], transpose),
            FfnAtomicWeight(W.ffn_b2, [CkptWeightInfo(self._names.FFN_DOWN_B)]),

            AtomicWeight(W.post_ffn_ln_beta, [CkptWeightInfo(self._names.FFN_OUTPUT_LAYERNORM_B)]),
            AtomicWeight(W.post_ffn_ln_gamma, [CkptWeightInfo(self._names.FFN_OUTPUT_LAYERNORM_W)]),
        ]
        return ModelWeightInfo(layer_weights=layer_weights,  weights=weights)