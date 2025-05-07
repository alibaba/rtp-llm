import functools
import torch
from typing import Any, List, Union, Dict
from maga_transformer.model_loader.load_config import LoadConfig
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from maga_transformer.model_loader.weight_module import AtomicWeight, CompositeWeight, QuantWeight, WeightModule
from maga_transformer.utils.model_weight import W, CkptWeightInfo, identity, kv_split, mla_pad, \
    mla_pad_scale, stack_, stack_moe_w1, concat_0,\
    multipy_identity, pad, transpose_slice_k, transpose_slice_v,\
        pad_w13
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight, MlaAttnAtomicWeight


W_SUFFIX = '.weight'
B_SUFFIX = ".bias"
QW_SUFFIX = '.weight'
QS_SUFFIX = '.weight_scale_inv'


def dequant_weight_split_k(ts: List[torch.Tensor], block_size: int, head_num: int, nope_head_dim: int, v_head_dim: int, lora_rank: int) -> torch.Tensor:
    from maga_transformer.models.deepseek_dequant import weight_dequant
    return transpose_slice_k([weight_dequant(ts[0], ts[1], block_size)],
                             head_num, nope_head_dim, v_head_dim, lora_rank)

def dequant_weight_split_v(ts: List[torch.Tensor], block_size: int, head_num: int, nope_head_dim: int, v_head_dim: int, lora_rank: int) -> torch.Tensor:
    from maga_transformer.models.deepseek_dequant import weight_dequant
    return transpose_slice_v([weight_dequant(ts[0], ts[1], block_size)],
                             head_num, nope_head_dim, v_head_dim, lora_rank)


class W8A8Fp8PerBlockAtomicWeight(AtomicWeight):
    gpt_style_tp_strategy = W.gemm_block_fp8_gpt_style_tp_strategy()
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _get_split_func(self):
        return self.gpt_style_tp_strategy[self.name]

class W8A8Fp8PerBlockAttnAtomicWeight(AttnAtomicWeight, W8A8Fp8PerBlockAtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

class W8A8Fp8PerBlockMlaAttnAtomicWeight(MlaAttnAtomicWeight, W8A8Fp8PerBlockAtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

class W8A8Fp8PerBlockFfnAtomicWeight(FfnAtomicWeight, W8A8Fp8PerBlockAtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

class W8A8Fp8PerBlockMoeAtomicWeight(MoeAtomicWeight, W8A8Fp8PerBlockAtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

def create_w8a8_fp8_per_block_weight(src_weight_info: WeightModule, *args: Any, **kwargs: Any) -> W8A8Fp8PerBlockAtomicWeight :
    if isinstance(src_weight_info, MlaAttnAtomicWeight):
        return W8A8Fp8PerBlockMlaAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AttnAtomicWeight):
        return W8A8Fp8PerBlockAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, MoeAtomicWeight):
        return W8A8Fp8PerBlockMoeAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, FfnAtomicWeight):
        return W8A8Fp8PerBlockFfnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AtomicWeight):
        return W8A8Fp8PerBlockAtomicWeight(*args, **kwargs)
    raise NotImplementedError(f"Unsupported weight type: {src_weight_info}")

class PerBlockFp8Weight(CompositeWeight, QuantWeight):
    w8a8_weight_list = [
        W.attn_qkv_w,
        W.attn_o_w,
        W.mla_k_nope_w,
        W.mla_v_w,
        W.mla_kc,
        W.mla_vc,
        W.mla_q_b_w,
        W.mla_fusedqkrope_w,
        W.ffn_w1,
        W.ffn_w2,
        W.ffn_w3,
        W.ffn_w13,
        W.ffn_b13,
        W.moe_w1,
        W.moe_w2
    ]

    @classmethod
    def support(cls, quant_algo: Any, src_weight_info: WeightModule) -> bool:
        name = src_weight_info.name
        return quant_algo.isGroupwise() and quant_algo.isFp8() and name in cls.w8a8_weight_list


    def __init__(self, src_weight_info: WeightModule, quant_algo: Any,  *args: Any, **kwargs: Any):
        kernel: WeightModule = None
        scale: WeightModule = None
        self.group_size = quant_algo.getGroupSize()

        if src_weight_info.name == W.attn_qkv_w:
            kernel, scale = self._get_qkv_quant_weight(src_weight_info, self.group_size)
        elif src_weight_info.name == W.attn_o_w:
            kernel, scale = self._get_attn_out_quant_weight(src_weight_info, self.group_size)
        elif src_weight_info.name in [W.mla_k_nope_w, W.mla_v_w]:
            kernel, scale = self._get_mla_kv_nope_quant_weight(src_weight_info, self.group_size)
        elif src_weight_info.name in [W.mla_kc, W.mla_vc]:
            kernel, scale = self._get_mla_kv_c(src_weight_info)
        elif src_weight_info.name == W.mla_q_b_w:
            kernel, scale = self._get_mla_q_quant_weight(src_weight_info)
        elif src_weight_info.name == W.mla_fusedqkrope_w:
            kernel, scale = self._get_fusedqkrope_quant_weight(src_weight_info)
        elif src_weight_info.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13]:
            kernel, scale = self._get_ffn_quant_weight(src_weight_info, self.group_size)
        elif src_weight_info.name == W.moe_w1:
            kernel, scale = self._get_moe_w1_quant_weight(src_weight_info)
        elif src_weight_info.name == W.moe_w2:
            kernel, scale = self._get_moe_w2_quant_weight(src_weight_info)

        sub_weights = {kernel.name : kernel}
        if scale is not None:
            sub_weights.update({scale.name : scale})
        super().__init__(sub_weights, quant_algo=quant_algo,*args, **kwargs)
        self.kernel = sub_weights.get(kernel.name)
        self.scale = sub_weights.get(scale.name) if scale is not None else None

    def _get_qkv_quant_weight(self, src_weight_info: AttnAtomicWeight, group_size: int):
        assert src_weight_info.name == W.attn_qkv_w
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1 or len(weights) == 3
        qkv_w_list = [CkptWeightInfo(sub_w.name[:-len(W_SUFFIX)] + QW_SUFFIX, sub_w.merge_fun) for sub_w in weights]
        qkv_s_list = [CkptWeightInfo(sub_w.name[:-len(W_SUFFIX)] + QS_SUFFIX, sub_w.merge_fun) for sub_w in weights]
        kernel = create_w8a8_fp8_per_block_weight(src_weight_info, W.attn_qkv_w,
                                                  [qkv_w_list], concat_0, data_type=torch.float8_e4m3fn, config=src_weight_info.config)

        scale = create_w8a8_fp8_per_block_weight(W.attn_qkv_s,
                                                 qkv_s_list, concat_0, data_type=torch.float32, config=src_weight_info.config)
        return [kernel, scale]


    def _get_attn_out_quant_weight(self, src_weight_info: MlaAttnAtomicWeight, group_size: int):
        assert src_weight_info.name == W.attn_o_w
        w_name = src_weight_info.weights[0].name[:-len(W_SUFFIX)]
        kernel = MlaAttnAtomicWeight(W.attn_o_w, [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                    functools.partial(mla_pad, head_num=src_weight_info.config.head_num, nope_head_dim=src_weight_info.nope_head_dim, rope_head_dim=0),
                    data_type=torch.float8_e4m3fn, config=src_weight_info.config)
        scale = MlaAttnAtomicWeight(W.attn_o_s, [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                    functools.partial(mla_pad_scale, head_num=src_weight_info.config.head_num, nope_head_dim=src_weight_info.nope_head_dim, rope_head_dim=0, group_size=group_size),
                    data_type=torch.float32, config=src_weight_info.config)

        return [kernel, scale]


    def _get_mla_kv_nope_quant_weight(self, src_weight_info: MlaAttnAtomicWeight, group_size: int):
        is_k = src_weight_info.name == W.mla_k_nope_w
        w_name = src_weight_info.weights[0].name[:-len(W_SUFFIX)]
        if is_k:
            w, s = [W.mla_k_nope_w,W.mla_k_nope_s]
        else:
            w, s = [W.mla_v_w, W.mla_v_s]

        kernel = MlaAttnAtomicWeight(w, [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                    functools.partial(kv_split, kv_lora_rank=src_weight_info.kv_lora_rank, nope_head_dim=src_weight_info.nope_head_dim, v_head_dim=src_weight_info.v_head_dim, idx=0 if is_k else 1),
                    data_type=torch.float8_e4m3fn, config=src_weight_info.config)
        scale = MlaAttnAtomicWeight(s, [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                    functools.partial(kv_split, kv_lora_rank=src_weight_info.kv_lora_rank // group_size, nope_head_dim=src_weight_info.nope_head_dim // group_size, v_head_dim=src_weight_info.v_head_dim // group_size, idx= 0 if is_k else 1),
                    data_type=torch.float32, config=src_weight_info.config)

        return [kernel, scale]

    def _get_mla_kv_c(self, src_weight_info: MlaAttnAtomicWeight):
        is_k = src_weight_info.name == W.mla_kc
        process_func = dequant_weight_split_k if is_k else dequant_weight_split_v
        w_name = src_weight_info.weights[0].name[:-len(W_SUFFIX)]
        w = W.mla_kc if is_k else W.mla_vc

        kernel = MlaAttnAtomicWeight(w, [CkptWeightInfo(w_name + QW_SUFFIX, identity), CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                    functools.partial(process_func, block_size=128, head_num=src_weight_info.config.head_num, nope_head_dim=src_weight_info.nope_head_dim, v_head_dim=src_weight_info.v_head_dim, lora_rank=src_weight_info.kv_lora_rank),
                    config=src_weight_info.config)
        return [kernel, None]

    def _get_mla_q_quant_weight(self, src_weight_info: MlaAttnAtomicWeight):
        assert src_weight_info.name == W.mla_q_b_w
        assert src_weight_info.config.q_use_lora
        w_name = src_weight_info.weights[0].name[:-len(W_SUFFIX)]

        kernel = MlaAttnAtomicWeight(W.mla_q_b_w, [CkptWeightInfo(w_name + QW_SUFFIX, src_weight_info.weights[0].merge_fun)],
                identity, data_type=torch.float8_e4m3fn, config=src_weight_info.config)
        scale = MlaAttnAtomicWeight(W.mla_q_b_s, [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                identity, data_type=torch.float32, config=src_weight_info.config)
        return [kernel, scale]

    def _get_fusedqkrope_quant_weight(self, src_weight_info: MlaAttnAtomicWeight):
        assert src_weight_info.name == W.mla_fusedqkrope_w
        assert src_weight_info.config.q_use_lora
        q_w_name = src_weight_info.weights[0].name[:-len(W_SUFFIX)]
        k_w_name = src_weight_info.weights[1].name[:-len(W_SUFFIX)]
        kernel = MlaAttnAtomicWeight(W.mla_fusedqkrope_w, [
            CkptWeightInfo(q_w_name + QW_SUFFIX, identity),
            CkptWeightInfo(k_w_name + QW_SUFFIX, src_weight_info.weights[1].merge_fun)], concat_0,
            data_type=torch.float8_e4m3fn, config=src_weight_info.config)
        scale = MlaAttnAtomicWeight(W.mla_fusedqkrope_s, [
            CkptWeightInfo(q_w_name + QS_SUFFIX, identity),
            CkptWeightInfo(k_w_name + QS_SUFFIX, identity)], concat_0, data_type=torch.float32, config=src_weight_info.config)

        return [kernel, scale]

    def _get_ffn_quant_weight(self, src_weight_info: FfnAtomicWeight, group_size: int):
        assert src_weight_info.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13]
        weights = src_weight_info.weights
        w_name = src_weight_info.weights[0].name[:-len(W_SUFFIX)]
        if src_weight_info.name == W.ffn_w13:
            w, s = (W.ffn_w13, W.ffn_s13)
            w1_name = weights[0].name[:-len(W_SUFFIX)]
            w3_name = weights[1].name[:-len(W_SUFFIX)]

            return [
                FfnAtomicWeight(
                    w, [CkptWeightInfo(w1_name + QW_SUFFIX, identity), CkptWeightInfo(w3_name + QW_SUFFIX, identity)],
                        functools.partial(pad_w13, inter_padding_size=src_weight_info.config.inter_padding_size, dim=0),
                        data_type=torch.float8_e4m3fn,
                        config=src_weight_info.config),
                FfnAtomicWeight(
                    s, [CkptWeightInfo(w1_name + QS_SUFFIX, identity), CkptWeightInfo(w3_name + QS_SUFFIX, identity)],
                    functools.partial(pad_w13, inter_padding_size=src_weight_info.config.inter_padding_size // group_size, dim=0), data_type=torch.float32,
                    config=src_weight_info.config)
            ]
        elif src_weight_info.name in [W.ffn_w1, W.ffn_w3]:
            if src_weight_info.name == W.ffn_w1:
                w, s = [W.ffn_w1, W.ffn_s1]
            else:
                w, s = [W.ffn_w3, W.ffn_s3]

            kernel = FfnAtomicWeight(w, [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                                     functools.partial(pad, inter_padding_size=src_weight_info.config.inter_padding_size, dim=0),
                                     data_type=torch.float8_e4m3fn, config=src_weight_info.config)
            scale = FfnAtomicWeight(s, [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                                    functools.partial(pad, inter_padding_size=src_weight_info.config.inter_padding_size // group_size, dim=0),
                                    data_type=torch.float32, config=src_weight_info.config)
            return [kernel, scale]
        else:
            kernel = FfnAtomicWeight(W.ffn_w2, [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                                     functools.partial(pad, inter_padding_size=src_weight_info.config.inter_padding_size, dim=1),
                                     data_type=torch.float8_e4m3fn, config=src_weight_info.config)
            scale = FfnAtomicWeight(W.ffn_s2, [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                                    functools.partial(pad, inter_padding_size=src_weight_info.config.inter_padding_size // group_size, dim=1),
                                    data_type=torch.float32, config=src_weight_info.config)
            return [kernel, scale]

    def _get_moe_w2_quant_weight(self, src_weight_info: MoeAtomicWeight):
        assert src_weight_info.name in [W.moe_w2]
        w_name = src_weight_info.weights[0].name[:-len(W_SUFFIX)]
        kernel = MoeAtomicWeight(W.moe_w2, [CkptWeightInfo(w_name + QW_SUFFIX, identity)], stack_,
                                 data_type=torch.float8_e4m3fn, config=src_weight_info.config)
        scale = MoeAtomicWeight(W.moe_s2, [CkptWeightInfo(w_name + QS_SUFFIX,
                                functools.partial(multipy_identity, scale=src_weight_info.config.routed_scaling_factor))], stack_,
                                data_type=torch.float32, config=src_weight_info.config)
        return [kernel, scale]

    def _get_moe_w1_quant_weight(self, src_weight_info: MoeAtomicWeight):
        assert src_weight_info.name in [W.moe_w1]
        kernel = MoeAtomicWeight(W.moe_w1, [CkptWeightInfo(w.name[:-len(W_SUFFIX)] + QW_SUFFIX, identity) for w in src_weight_info.weights],
                                stack_moe_w1, data_type=torch.float8_e4m3fn, config=src_weight_info.config)
        scale = MoeAtomicWeight(W.moe_s1, [CkptWeightInfo(w.name[:-len(W_SUFFIX)] + QS_SUFFIX, identity) for w in src_weight_info.weights],
                                stack_moe_w1, data_type=torch.float32, config=src_weight_info.config)
        return [kernel, scale]

    def _postprocess(self, tensor: Union[torch.Tensor, Dict[str, torch.Tensor]], device: str, load_config: LoadConfig):
        # need reshape for kernel weight
        processed_res = super()._postprocess(tensor, device, load_config)
        kernel_weight = processed_res[self.kernel.name]
        kernel_weight = kernel_weight.reshape(kernel_weight.shape[-1], -1) if kernel_weight.dim() == 2 else kernel_weight
        processed_res[self.kernel.name] = kernel_weight
        if self.scale is not None:
            scale_weight = processed_res[self.scale.name]
            scale_weight = scale_weight.reshape(scale_weight.shape[-1], -1) if  scale_weight.dim() == 2 else scale_weight
            processed_res[self.scale.name] = scale_weight

        return processed_res
