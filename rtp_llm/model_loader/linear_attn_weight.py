from typing import Any, Callable, Dict, List, Optional, Union

import torch

from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.ops import LinearAttentionConfig
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
    BLOCK_SIZE = 128
    origin_qkvz_size = (
        linear_config.linear_key_head_dim * linear_config.linear_num_key_heads
        + linear_config.linear_value_head_dim * linear_config.linear_num_value_heads
    ) * 2
    if t.shape[1] == origin_qkvz_size:
        q, k, v, z = torch.split(
            t,
            [
                linear_config.linear_key_head_dim * linear_config.linear_num_key_heads,
                linear_config.linear_key_head_dim * linear_config.linear_num_key_heads,
                linear_config.linear_value_head_dim
                * linear_config.linear_num_value_heads,
                linear_config.linear_value_head_dim
                * linear_config.linear_num_value_heads,
            ],
            dim=1,
        )
    elif t.shape[1] == origin_qkvz_size // BLOCK_SIZE:
        q, k, v, z = torch.split(
            t,
            [
                linear_config.linear_key_head_dim
                // BLOCK_SIZE
                * linear_config.linear_num_key_heads,
                linear_config.linear_key_head_dim
                // BLOCK_SIZE
                * linear_config.linear_num_key_heads,
                linear_config.linear_value_head_dim
                // BLOCK_SIZE
                * linear_config.linear_num_value_heads,
                linear_config.linear_value_head_dim
                // BLOCK_SIZE
                * linear_config.linear_num_value_heads,
            ],
            dim=1,
        )
    else:
        raise ValueError(
            f"Invalid input shape 0 for scale / weight: {t.shape}, expected: {origin_qkvz_size} or {origin_qkvz_size // BLOCK_SIZE}"
        )
    q = torch.split(q, q.shape[1] // load_config.tp_size, dim=1)[
        load_config.tp_rank
    ].contiguous()
    k = torch.split(k, k.shape[1] // load_config.tp_size, dim=1)[
        load_config.tp_rank
    ].contiguous()
    v = torch.split(v, v.shape[1] // load_config.tp_size, dim=1)[
        load_config.tp_rank
    ].contiguous()
    z = torch.split(z, z.shape[1] // load_config.tp_size, dim=1)[
        load_config.tp_rank
    ].contiguous()
    return torch.cat([q, k, v, z], dim=1)


def split_qkvz_t(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    t = split_qkvz(t.transpose(0, 1), load_config, linear_config)
    return t.transpose(0, 1).contiguous()


# ba layout: [hidden_size, head_v + head_v]
def split_ba(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    pack_head_num = t.shape[1]
    assert pack_head_num % 2 == 0, "pack_head_num must be even"
    b, a = torch.split(t, [pack_head_num // 2, pack_head_num // 2], dim=1)
    b = b.split(b.shape[1] // load_config.tp_size, dim=1)[
        load_config.tp_rank
    ].contiguous()
    a = a.split(a.shape[1] // load_config.tp_size, dim=1)[
        load_config.tp_rank
    ].contiguous()
    return torch.cat([b, a], dim=1)


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
    # The returned tensor is retained for the lifetime of the model.  A view
    # would keep the complete, transposed output projection alive on every TP
    # rank even though each rank only consumes its local heads.  Materialize
    # the TP-local payload so the loader source storage can be released.
    return (
        t[start_head_num_v:end_head_num_v, :, :]
        .reshape(-1, n)
        .clone(memory_format=torch.contiguous_format)
    )


def split_out_linear_t(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    t = split_out_linear(t.transpose(0, 1), load_config, linear_config)
    return t.transpose(0, 1).contiguous()


def sp_id(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    return t


# KDA fused qkv: [hidden, q+k+v] -> per-section TP split on dim=1.
def split_kda_qkv(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    q_size = linear_config.linear_num_key_heads * linear_config.linear_key_head_dim
    k_size = linear_config.linear_num_key_heads * linear_config.linear_key_head_dim
    v_size = linear_config.linear_num_value_heads * linear_config.linear_value_head_dim
    q, k, v = torch.split(t, [q_size, k_size, v_size], dim=1)
    q = q.split(q.shape[1] // load_config.tp_size, dim=1)[
        load_config.tp_rank
    ].contiguous()
    k = k.split(k.shape[1] // load_config.tp_size, dim=1)[
        load_config.tp_rank
    ].contiguous()
    v = v.split(v.shape[1] // load_config.tp_size, dim=1)[
        load_config.tp_rank
    ].contiguous()
    return torch.cat([q, k, v], dim=1)


def split_kda_qkvg_fa_beta_sections(
    tensor: torch.Tensor,
    q_size: int,
    k_size: int,
    v_size: int,
    g_size: int,
    f_a_size: int,
    beta_size: int,
    *,
    dim: int = -1,
) -> tuple[torch.Tensor, ...]:
    """Split the shared K3 fused-projection layout into its six sections."""

    section_sizes = (q_size, k_size, v_size, g_size, f_a_size, beta_size)
    if any(size <= 0 for size in section_sizes):
        raise ValueError(
            "KDA fused projection section widths must be positive, got "
            f"{section_sizes}"
        )
    actual_width = tensor.shape[dim]
    expected_width = sum(section_sizes)
    if actual_width != expected_width:
        raise ValueError(
            "KDA fused projection width does not match its layout: "
            f"shape={tuple(tensor.shape)}, dim={dim}, "
            f"sections={section_sizes}, expected={expected_width}, "
            f"actual={actual_width}"
        )
    return torch.split(tensor, section_sizes, dim=dim)


# K3 fused projection: shard Q/K/V/G by head while replicating F_A/beta.
def split_kda_qkvg_fa_beta(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    q_size = linear_config.linear_num_key_heads * linear_config.linear_key_head_dim
    k_size = linear_config.linear_num_key_heads * linear_config.linear_key_head_dim
    v_size = linear_config.linear_num_value_heads * linear_config.linear_value_head_dim
    g_size = linear_config.linear_num_value_heads * linear_config.linear_value_head_dim
    beta_size = linear_config.linear_num_value_heads
    f_a_size = t.shape[1] - q_size - k_size - v_size - g_size - beta_size
    if f_a_size <= 0:
        raise ValueError(
            "KDA fused projection must contain a non-empty F_A section: "
            f"shape={tuple(t.shape)}, qkvg={[q_size, k_size, v_size, g_size]}, "
            f"beta={beta_size}"
        )
    if any(size % load_config.tp_size for size in (q_size, k_size, v_size, g_size)):
        raise ValueError(
            "KDA Q/K/V/G widths must be divisible by TP: "
            f"qkvg={[q_size, k_size, v_size, g_size]}, tp={load_config.tp_size}"
        )

    q, k, v, g, f_a, beta = split_kda_qkvg_fa_beta_sections(
        t,
        q_size,
        k_size,
        v_size,
        g_size,
        f_a_size,
        beta_size,
        dim=1,
    )

    def _local(section: torch.Tensor) -> torch.Tensor:
        width = section.shape[1] // load_config.tp_size
        begin = load_config.tp_rank * width
        return section.narrow(1, begin, width)

    return torch.cat(
        [_local(q), _local(k), _local(v), _local(g), f_a, beta],
        dim=1,
    ).contiguous()


# KDA split: TP split on dim=1 (b_proj, LoRA up projections, full-rank gate).
def split_kda_tp_dim1(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    return t.split(t.shape[1] // load_config.tp_size, dim=1)[
        load_config.tp_rank
    ].contiguous()


# KDA dt_bias layout [num_heads * head_dim] -> [local_heads * head_dim].
def split_kda_dt_bias(
    t: torch.Tensor, load_config: LoadConfig, linear_config: LinearAttnConfig
) -> torch.Tensor:
    num_heads = linear_config.linear_num_value_heads
    head_dim = linear_config.linear_key_head_dim
    local_heads = num_heads // load_config.tp_size
    t = t.reshape(num_heads, head_dim)
    start = local_heads * load_config.tp_rank
    return t[start : start + local_heads].reshape(-1)


_linear_attn_split_stratey = {
    W.linear_attn_qkvz_w: split_qkvz,
    W.linear_attn_ba_w: split_ba,
    W.linear_attn_alog: split_head_linear,  # GDN/KDA shared: [num_heads]
    W.linear_attn_dt_b: split_head_linear,  # GDN-only: [num_heads]
    W.linear_attn_conv1d_w: split_conv1d,
    W.linear_attn_out_w: split_out_linear,
    W.linear_attn_norm_w: sp_id,
    # KDA (Kimi Delta Attention) fused weights.
    W.linear_attn_qkv_w: split_kda_qkv,
    W.linear_attn_qkvg_fa_beta_w: split_kda_qkvg_fa_beta,
    W.linear_attn_b_w: split_kda_tp_dim1,
    W.linear_attn_f_a_w: sp_id,  # forget-gate LoRA down: rank not sharded
    W.linear_attn_f_b_w: split_kda_tp_dim1,
    # Reserved for the kimi_linear low-rank output gate; K3's manifest loads the
    # full-rank g_w below instead, so these two have no K3 load path yet.
    W.linear_attn_g_a_w: sp_id,  # output-gate LoRA down: rank not sharded
    W.linear_attn_g_b_w: split_kda_tp_dim1,
    W.linear_attn_g_w: split_kda_tp_dim1,  # K3 full-rank output gate
    W.linear_attn_dt_b_kda: split_kda_dt_bias,  # KDA-only: [num_heads * head_dim]
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
