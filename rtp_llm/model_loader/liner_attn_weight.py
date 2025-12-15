from typing import Any, Callable, Dict, List, Optional, Union

import torch

from rtp_llm.config.gpt_init_model_parameters import LinearAttentionConfig
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


class LinearAttnConfig(object):
    def __init__(self, linear_attention_config: LinearAttentionConfig):
        self.linear_num_key_heads = linear_attention_config.linear_num_key_heads
        self.linear_num_value_heads = linear_attention_config.linear_num_value_heads
        self.linear_key_head_dim = linear_attention_config.linear_key_head_dim
        self.linear_value_head_dim = linear_attention_config.linear_value_head_dim


# qkvz layout: [hidden_size, head_k, xx] -> [hidden, local_head_k, xx]
def split_qkvz(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    local_head_num_k = linear_config.linear_num_key_heads // load_config.tp_size
    start_head_num_k = local_head_num_k * load_config.tp_rank
    end_head_num_k = start_head_num_k + local_head_num_k
    qkvz = (
        t.view(t.size(0), linear_config.linear_num_key_heads, -1)[
            :, start_head_num_k:end_head_num_k, :
        ]
        .reshape(t.size(0), -1)
        .contiguous()
    )
    return qkvz


def split_qkvz_t(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    t = split_qkvz(t.transpose(0, 1), load_config, linear_config)
    return t.transpose(0, 1).contiguous()


# ba layout: [hidden_size, head_k, 2 + 2]
def split_ba(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    local_head_num_k = linear_config.linear_num_key_heads // load_config.tp_size
    start_head_num_k = local_head_num_k * load_config.tp_rank
    end_head_num_k = start_head_num_k + local_head_num_k
    ba = (
        t.view(t.size(0), linear_config.linear_num_key_heads, -1)[
            :, start_head_num_k:end_head_num_k, :
        ]
        .reshape(t.size(0), -1)
        .contiguous()
    )
    return ba


# layout [head_num_v]
def split_head_linear(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    local_head_num_v = linear_config.linear_num_value_heads // load_config.tp_size
    start_head_num_v = local_head_num_v * load_config.tp_rank
    end_head_num_v = start_head_num_v + local_head_num_v
    return t[start_head_num_v:end_head_num_v]


# layout [head_num_k * head_dim(Q), head_num_k * head_dim(K), head_num_v * head_dim(V), 1, kernel_size]
def split_conv1d(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    q, k, v = torch.split(
        t,
        [
            linear_config.linear_num_key_heads * linear_config.linear_key_head_dim,
            linear_config.linear_num_key_heads * linear_config.linear_key_head_dim,
            linear_config.linear_num_value_heads * linear_config.linear_value_head_dim,
        ],
        dim=0,
    )
    local_head_num_k = linear_config.linear_num_key_heads // load_config.tp_size
    start_k = local_head_num_k * load_config.tp_rank * linear_config.linear_key_head_dim
    end_k = start_k + local_head_num_k * linear_config.linear_key_head_dim
    local_head_num_v = linear_config.linear_num_value_heads // load_config.tp_size
    start_v = (
        local_head_num_v * load_config.tp_rank * linear_config.linear_value_head_dim
    )
    end_v = start_v + local_head_num_v * linear_config.linear_value_head_dim
    q = q[start_k:end_k].contiguous()
    k = k[start_k:end_k].contiguous()
    v = v[start_v:end_v].contiguous()
    return torch.cat([q, k, v], dim=0)


# weight: [head_num_v * head_size_v, hidden_size] -> [local_head_v * head_size_v, hidden_size]
def split_out_linear(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    _, n = t.shape
    t = t.view(linear_config.linear_num_value_heads, -1, n)
    local_head_num_v = linear_config.linear_num_value_heads // load_config.tp_size
    start_head_num_v = local_head_num_v * load_config.tp_rank
    end_head_num_v = start_head_num_v + local_head_num_v
    return t[start_head_num_v:end_head_num_v, :, :].reshape(-1, n)


def split_out_linear_t(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    t = split_out_linear(t.transpose(0, 1), load_config, linear_config)
    return t.transpose(0, 1).contiguous()


def sp_id(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    return t


_linear_attn_split_stratey = {
    W.linear_attn_qkvz_w: split_qkvz,
    W.linear_attn_ba_w: split_ba,
    W.linear_attn_alog: split_head_linear,
    W.linear_attn_dt_b: split_head_linear,
    W.linear_attn_conv1d_w: split_conv1d,
    W.linear_attn_out_w: split_out_linear,
    W.linear_attn_norm_w: sp_id,
}


_linear_attn_w8a8_per_block_split_strategy = {
    W.linear_attn_qkvz_w: split_qkvz_t,
    W.linear_attn_qkvz_s: split_qkvz_t,
    W.linear_attn_out_w: split_out_linear_t,
    W.linear_attn_out_s: split_out_linear_t,
}


class LinearAttnAtomicWeight(AtomicWeight):
    def __init__(
        self,
        name: str,
        weights: List[CkptWeightInfo],
        process_fun: Callable[[List[torch.Tensor]], torch.Tensor],
        config: LinearAttnConfig,
        data_type: Optional[torch.dtype] = None,
    ):
        super().__init__(name, weights, process_fun, data_type)
        self.config = config
        self.split_func_factory = _linear_attn_split_stratey

    def _split(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        load_config: LoadConfig,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(tensor, dict):
            tensor = tensor[self.name]
        if load_config.tp_size <= 1:
            return {self.name: tensor}
        else:
            return {
                self.name: self.split_func_factory[self.name](
                    tensor, load_config, self.config
                )
            }


class W8A8Fp8PerBlockLinearAttnAtomicWeight(LinearAttnAtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.split_func_factory = _linear_attn_w8a8_per_block_split_strategy
