import torch
import functools
from typing import List, Any, Union
from pydantic import BaseModel

from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo, ModelDeployWeightInfo, CkptWeightInfo, concat_0, transpose

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

class SmoothQuantWeightNames(BaseModel):
    Q_W: str = 'encoder.layer.{i}.attention.self.query.qweight'
    Q_B: str = 'encoder.layer.{i}.attention.self.query.bias'
    Q_S: str = 'encoder.layer.{i}.attention.self.query.scales'
    K_W: str = 'encoder.layer.{i}.attention.self.key.qweight'
    K_B: str = 'encoder.layer.{i}.attention.self.key.bias'
    K_S: str = 'encoder.layer.{i}.attention.self.key.scales'
    V_W: str = 'encoder.layer.{i}.attention.self.value.qweight'
    V_B: str = 'encoder.layer.{i}.attention.self.value.bias'
    V_S: str = 'encoder.layer.{i}.attention.self.value.scales'
    O_W: str = 'encoder.layer.{i}.attention.output.dense.qweight'
    O_B: str = 'encoder.layer.{i}.attention.output.dense.bias'
    O_S: str = 'encoder.layer.{i}.attention.output.dense.scales'

    POST_LN_W: str = 'encoder.layer.{i}.attention.output.LayerNorm.weight'
    POST_LN_B: str = 'encoder.layer.{i}.attention.output.LayerNorm.bias'

    FFN_INTER_DENSE_W: str = 'encoder.layer.{i}.intermediate.dense.qweight'
    FFN_INTER_DENSE_B: str = 'encoder.layer.{i}.intermediate.dense.bias'
    FFN_INTER_DENSE_S: str = 'encoder.layer.{i}.intermediate.dense.scales'
    FFN_OUTPUT_DENSE_W: str = 'encoder.layer.{i}.output.dense.qweight'
    FFN_OUTPUT_DENSE_B: str = 'encoder.layer.{i}.output.dense.bias'
    FFN_OUTPUT_DENSE_S: str = 'encoder.layer.{i}.output.dense.scales'
    FFN_OUTPUT_LAYERNORM_W: str = 'encoder.layer.{i}.output.LayerNorm.weight'
    FFN_OUTPUT_LAYERNORM_B: str = 'encoder.layer.{i}.output.LayerNorm.bias'

    TOKEN_EMBEDDING: str = 'embeddings.word_embeddings.weight'
    POSITION_EMBEDDING: str = 'embeddings.position_embeddings.weight'
    TOKEN_TYPE_EMBEDDING: str = 'embeddings.token_type_embeddings.weight'
    EMB_NORM_W: str = 'embeddings.LayerNorm.weight'
    EMB_NORM_B: str = 'embeddings.LayerNorm.bias'

    ATTN_INPUT_SMOOTHER: str = "encoder.layer.{i}.input.dense.smoother"
    ATTN_OUTPUT_SMOOTHER: str = "encoder.layer.{i}.output.dense.smoother"
    FFN_SMOOTHER: str = "encoder.layer.{i}.intermediate.dense.smoother"

class PerTensorQuantWeightNames(SmoothQuantWeightNames):
    EMB_NORM_S: str = "encoder.layer.0.attention.self.qkv_input_scale"
    ATTEN_OUTPUT_S: str = "encoder.layer.{i}.attention.output.dense_input_scale"
    POST_LN_S: str = 'encoder.layer.{i}.intermediate.dense_input_scale'
    FFN_INTER_INPUT_S: str = "encoder.layer.{i}.output.dense_input_scale"
    FFN_OUTPUT_LAYERNORM_S: str = "encoder.layer.{i_1}.attention.self.qkv_input_scale"

class BertWeightInfo(ModelDeployWeightInfo):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model_name = 'bert'
        self._names = HfWeightNames()
        self._smooth_quant_names = SmoothQuantWeightNames()
        self._pertensor_quant_names = PerTensorQuantWeightNames()

    def _append_name_prefix(self, names: Union[HfWeightNames, SmoothQuantWeightNames], weight_keys: List[str]):
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
        self._append_name_prefix(self._smooth_quant_names, weight_keys)
        self._append_name_prefix(self._pertensor_quant_names, weight_keys)

    def _get_base_weight_info(self) -> List[WeightInfo]:
        return [
            WeightInfo(W.embedding, [CkptWeightInfo(self._names.TOKEN_EMBEDDING)]),
            WeightInfo(W.positional_embedding, [CkptWeightInfo(self._names.POSITION_EMBEDDING)]),
            WeightInfo(W.token_type_embedding, [CkptWeightInfo(self._names.TOKEN_TYPE_EMBEDDING)]),
            WeightInfo(W.pre_decoder_ln_beta, [CkptWeightInfo(self._names.EMB_NORM_B)]),
            WeightInfo(W.pre_decoder_ln_gamma, [CkptWeightInfo(self._names.EMB_NORM_W)]),
        ]

    def _get_weight_info(self):
        if self._quant_algo.isSmoothQuant() or self._quant_algo.isPerTensorQuant():
            return self._get_quant_weight_info()
        else:
            return self._get_hf_weight_info()

    def _get_quant_layer_weight_info(self) -> List[List[WeightInfo]]:
        layer_weights: List[List[WeightInfo]] = []
        for layer_id in range(self._num_layers):
            layer = [
                WeightInfo(W.attn_qkv_w, [
                    CkptWeightInfo(self._smooth_quant_names.Q_W),
                    CkptWeightInfo(self._smooth_quant_names.K_W),
                    CkptWeightInfo(self._smooth_quant_names.V_W)], merge_qkv_transpose_concat0),

                WeightInfo(W.attn_qkv_b, [
                    CkptWeightInfo(self._smooth_quant_names.Q_B),
                    CkptWeightInfo(self._smooth_quant_names.K_B),
                    CkptWeightInfo(self._smooth_quant_names.V_B)], concat_0),

                WeightInfo(W.attn_qkv_s, [
                    CkptWeightInfo(self._smooth_quant_names.Q_S),
                    CkptWeightInfo(self._smooth_quant_names.K_S),
                    CkptWeightInfo(self._smooth_quant_names.V_S)], functools.partial(expand_scale, hidden_size=self._hidden_size)),

                WeightInfo(W.attn_o_w, [CkptWeightInfo(self._smooth_quant_names.O_W)], transpose),
                WeightInfo(W.attn_o_b, [CkptWeightInfo(self._smooth_quant_names.O_B)]),
                WeightInfo(W.attn_o_s,
                        [CkptWeightInfo(self._smooth_quant_names.O_S)],
                        functools.partial(expand_scale, hidden_size=self._hidden_size)),

                WeightInfo(W.post_ln_beta, [CkptWeightInfo(self._smooth_quant_names.POST_LN_B)]),
                WeightInfo(W.post_ln_gamma, [CkptWeightInfo(self._smooth_quant_names.POST_LN_W)]),

                WeightInfo(W.ffn_w3, [CkptWeightInfo(self._smooth_quant_names.FFN_INTER_DENSE_W)], transpose),
                WeightInfo(W.ffn_b3, [CkptWeightInfo(self._smooth_quant_names.FFN_INTER_DENSE_B)]),
                WeightInfo(W.ffn_s3, [CkptWeightInfo(self._smooth_quant_names.FFN_INTER_DENSE_S)], functools.partial(expand_scale, hidden_size=self._inter_size)),

                WeightInfo(W.ffn_w2, [CkptWeightInfo(self._smooth_quant_names.FFN_OUTPUT_DENSE_W)], transpose),
                WeightInfo(W.ffn_b2, [CkptWeightInfo(self._smooth_quant_names.FFN_OUTPUT_DENSE_B)]),
                WeightInfo(W.ffn_s2, [CkptWeightInfo(self._smooth_quant_names.FFN_OUTPUT_DENSE_S)], functools.partial(expand_scale, hidden_size=self._hidden_size)),

                WeightInfo(W.post_ffn_ln_beta, [CkptWeightInfo(self._smooth_quant_names.FFN_OUTPUT_LAYERNORM_B)]),
                WeightInfo(W.post_ffn_ln_gamma, [CkptWeightInfo(self._smooth_quant_names.FFN_OUTPUT_LAYERNORM_W)]),
            ]
            if self._quant_algo.isSmoothQuant():
                layer.extend([
                    WeightInfo(W.attn_i_smoother, [CkptWeightInfo(self._smooth_quant_names.ATTN_INPUT_SMOOTHER)]),
                    WeightInfo(W.attn_o_smoother, [CkptWeightInfo(self._smooth_quant_names.ATTN_OUTPUT_SMOOTHER)]),
                    WeightInfo(W.ffn_smoother, [CkptWeightInfo(self._smooth_quant_names.FFN_SMOOTHER)]),
                ])
            if self._quant_algo.isPerTensorQuant():
                layer.extend([
                    WeightInfo(W.attention_output_static_quant, [CkptWeightInfo(self._pertensor_quant_names.ATTEN_OUTPUT_S)], get_tensor_reciprocal, torch.float32),
                    WeightInfo(W.attention_output_static_quant_reciprocal, [CkptWeightInfo(self._pertensor_quant_names.ATTEN_OUTPUT_S)], get_tensor_from_scalar, torch.float32),
                    WeightInfo(W.post_ln_static_quant, [CkptWeightInfo(self._pertensor_quant_names.POST_LN_S)], get_tensor_reciprocal, torch.float32),
                    WeightInfo(W.post_ln_static_quant_reciprocal, [CkptWeightInfo(self._pertensor_quant_names.POST_LN_S)], get_tensor_from_scalar, torch.float32),
                    WeightInfo(W.ffn_intermediate_weight2_static_quant, [CkptWeightInfo(self._pertensor_quant_names.FFN_INTER_INPUT_S)], get_tensor_reciprocal, torch.float32),
                    WeightInfo(W.ffn_intermediate_weight2_static_quant_reciprocal, [CkptWeightInfo(self._pertensor_quant_names.FFN_INTER_INPUT_S)], get_tensor_from_scalar, torch.float32),
                ])
                if layer_id != self._num_layers - 1:
                    layer.extend([
                        WeightInfo(W.post_ffn_ln_static_quant, [CkptWeightInfo(self._pertensor_quant_names.FFN_OUTPUT_LAYERNORM_S)], get_tensor_reciprocal, torch.float32),
                        WeightInfo(W.post_ffn_ln_static_quant_reciprocal, [CkptWeightInfo(self._pertensor_quant_names.FFN_OUTPUT_LAYERNORM_S)], get_tensor_from_scalar, torch.float32),
                    ])
            layer_weights.append(layer)
        return layer_weights

    def _get_quant_weight_info(self):
        weights = self._get_base_weight_info()
        if self._quant_algo.isPerTensorQuant():
            weights.extend([
                WeightInfo(W.pre_decoder_ln_static_quant, [CkptWeightInfo(self._pertensor_quant_names.EMB_NORM_S)], get_tensor_reciprocal, torch.float32),
                WeightInfo(W.pre_decoder_ln_static_quant_reciprocal, [CkptWeightInfo(self._pertensor_quant_names.EMB_NORM_S)], get_tensor_from_scalar, data_type=torch.float32),
            ])
        layer_weights = self._get_quant_layer_weight_info()        
        return ModelWeightInfo(layer_weights=layer_weights,
                               weights=weights,
                               tp_strategy=self._get_gpt_style_tp_strategy())

    def _get_hf_weight_info(self):
        weights = self._get_base_weight_info()
        layer_weights = [
            WeightInfo(W.attn_qkv_w, [
                CkptWeightInfo(self._names.Q_W),
                CkptWeightInfo(self._names.K_W),
                CkptWeightInfo(self._names.V_W)], merge_qkv_hf),

            WeightInfo(W.attn_qkv_b, [
                CkptWeightInfo(self._names.Q_B),
                CkptWeightInfo(self._names.K_B),
                CkptWeightInfo(self._names.V_B)], concat_0),

            WeightInfo(W.attn_o_w, [CkptWeightInfo(self._names.O_W)], transpose),
            WeightInfo(W.attn_o_b, [CkptWeightInfo(self._names.O_B)]),

            WeightInfo(W.post_ln_beta, [CkptWeightInfo(self._names.POST_LN_B)]),
            WeightInfo(W.post_ln_gamma, [CkptWeightInfo(self._names.POST_LN_W)]),

            WeightInfo(W.ffn_w3, [CkptWeightInfo(self._names.FFN_INTER_DENSE_W)], transpose),
            WeightInfo(W.ffn_b3, [CkptWeightInfo(self._names.FFN_INTER_DENSE_B)]),

            WeightInfo(W.ffn_w2, [CkptWeightInfo(self._names.FFN_OUTPUT_DENSE_W)], transpose),
            WeightInfo(W.ffn_b2, [CkptWeightInfo(self._names.FFN_OUTPUT_DENSE_B)]),

            WeightInfo(W.post_ffn_ln_beta, [CkptWeightInfo(self._names.FFN_OUTPUT_LAYERNORM_B)]),
            WeightInfo(W.post_ffn_ln_gamma, [CkptWeightInfo(self._names.FFN_OUTPUT_LAYERNORM_W)]),
        ]
        return ModelWeightInfo(layer_weights=layer_weights,
                               weights=weights,
                               tp_strategy=self._get_gpt_style_tp_strategy())

class RobertaWeightInfo(BertWeightInfo):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model_name = 'roberta'
