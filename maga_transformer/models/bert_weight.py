import torch
import functools
from typing import List, Any, Union
from pydantic import BaseModel

from maga_transformer.utils.model_weight import W, CkptWeightInfo, concat_0, transpose
from maga_transformer.model_loader.model_weight_info import ModelWeightInfo, ModelDeployWeightInfo
from maga_transformer.model_loader.weight_module import WeightModule, AtomicWeight
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, FfnWeight, FfnConfig
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight, AttnConfig


def get_tensor_reciprocal(ts: List[torch.Tensor]) -> torch.Tensor:
    return 1.0 / ts[0].reshape(-1)

def get_tensor_from_scalar(ts: List[torch.Tensor]) -> torch.Tensor:
    return ts[0].reshape(-1)

def merge_qkv_hf(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_weight = torch.concat([q.T, k.T, v.T], dim=1).contiguous()
    return qkv_weight


def merge_qkv_transpose_concat0(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_weight = torch.concat([q.T, k.T, v.T], dim=0).contiguous()
    return qkv_weight


# from [torch.Size(1), torch.Size(1), torch.Size(1)] to torch.Size(3 * hidden_size)
def expand_scale(ts: List[torch.Tensor], hidden_size:int):
    new_ts: List[torch.Tensor] = []
    for t in ts:
        tmp_t = t.reshape(-1)
        if tmp_t.shape == torch.Size([1]):
            new_ts.append(tmp_t.expand(hidden_size))
        elif tmp_t.shape == torch.Size([hidden_size]):
            new_ts.append(tmp_t)
        else:
            raise Exception(f"unknown scale shape: {t.shape}")
    return torch.concat(new_ts, dim=-1)


class HfWeightNames(BaseModel):
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

    FFN_INTER_DENSE_W: str = 'encoder.layer.{i}.intermediate.dense.weight'
    FFN_INTER_DENSE_B: str = 'encoder.layer.{i}.intermediate.dense.bias'
    FFN_OUTPUT_DENSE_W: str = 'encoder.layer.{i}.output.dense.weight'
    FFN_OUTPUT_DENSE_B: str = 'encoder.layer.{i}.output.dense.bias'
    FFN_OUTPUT_LAYERNORM_W: str = 'encoder.layer.{i}.output.LayerNorm.weight'
    FFN_OUTPUT_LAYERNORM_B: str = 'encoder.layer.{i}.output.LayerNorm.bias'

    TOKEN_EMBEDDING: str = 'embeddings.word_embeddings.weight'
    POSITION_EMBEDDING: str = 'embeddings.position_embeddings.weight'
    TOKEN_TYPE_EMBEDDING: str = 'embeddings.token_type_embeddings.weight'
    EMB_NORM_W: str = 'embeddings.LayerNorm.weight'
    EMB_NORM_B: str = 'embeddings.LayerNorm.bias'



class BertWeightInfo(ModelDeployWeightInfo):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model_name = 'bert'
        self._names = HfWeightNames()


    def _append_name_prefix(self, names: HfWeightNames, weight_keys: List[str]):
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
            AtomicWeight(W.positional_embedding, [CkptWeightInfo(self._names.POSITION_EMBEDDING)]),
            AtomicWeight(W.token_type_embedding, [CkptWeightInfo(self._names.TOKEN_TYPE_EMBEDDING)]),
            AtomicWeight(W.pre_decoder_ln_beta, [CkptWeightInfo(self._names.EMB_NORM_B)]),
            AtomicWeight(W.pre_decoder_ln_gamma, [CkptWeightInfo(self._names.EMB_NORM_W)]),
        ]

    def _get_weight_info(self):
        return self._get_hf_weight_info()


    def _get_hf_weight_info(self):
        weights = self._get_base_weight_info()

        layer_weights = []
        for id in range(self._num_layers):
            attn_config = AttnConfig(
                hidden_size=self._hidden_size,
                size_per_head=self._size_per_head,
                head_num=self._head_num,
                head_num_kv=self._head_num_kv,
                need_post_ln=id != self._num_layers - 1,
            )
            ffn_config = FfnConfig(
                is_gated_activation=self._is_gated_activation,
                inter_padding_size=self._inter_padding_size,
                is_moe=False
            )

            layer_weight = [
                AttnAtomicWeight(W.attn_qkv_w, [
                    CkptWeightInfo(self._names.Q_W),
                    CkptWeightInfo(self._names.K_W),
                    CkptWeightInfo(self._names.V_W)], merge_qkv_hf, config=attn_config),

                AttnAtomicWeight(W.attn_qkv_b, [
                    CkptWeightInfo(self._names.Q_B),
                    CkptWeightInfo(self._names.K_B),
                    CkptWeightInfo(self._names.V_B)], concat_0, config=attn_config),

                AttnAtomicWeight(W.attn_o_w, [CkptWeightInfo(self._names.O_W)], transpose, config=attn_config),
                AttnAtomicWeight(W.attn_o_b, [CkptWeightInfo(self._names.O_B)], config=attn_config),

                AtomicWeight(W.post_ln_beta, [CkptWeightInfo(self._names.POST_LN_B)]),
                AtomicWeight(W.post_ln_gamma, [CkptWeightInfo(self._names.POST_LN_W)]),
                FfnWeight(sub_weights=[
                    FfnAtomicWeight(W.ffn_w3, [CkptWeightInfo(self._names.FFN_INTER_DENSE_W)], transpose, config=ffn_config),
                    FfnAtomicWeight(W.ffn_b3, [CkptWeightInfo(self._names.FFN_INTER_DENSE_B)], config=ffn_config),

                    FfnAtomicWeight(W.ffn_w2, [CkptWeightInfo(self._names.FFN_OUTPUT_DENSE_W)], transpose, config=ffn_config),
                    FfnAtomicWeight(W.ffn_b2, [CkptWeightInfo(self._names.FFN_OUTPUT_DENSE_B)], config=ffn_config)
                    ], config=ffn_config),
                AtomicWeight(W.post_ffn_ln_beta, [CkptWeightInfo(self._names.FFN_OUTPUT_LAYERNORM_B)]),
                AtomicWeight(W.post_ffn_ln_gamma, [CkptWeightInfo(self._names.FFN_OUTPUT_LAYERNORM_W)]),
            ]
            layer_weights.append(layer_weight)
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

class RobertaWeightInfo(BertWeightInfo):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model_name = 'roberta'
