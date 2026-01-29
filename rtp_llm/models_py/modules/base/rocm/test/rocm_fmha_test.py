import itertools
import math
from typing import List, Optional, Tuple
from unittest import SkipTest, TestCase, main

import aiter
import torch
import torch.nn.functional as F
from aiter import dtypes
from einops import rearrange, repeat
from rtp_llm.ops.compute_ops import paged_attention_atrex


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim_q)
        k: (batch_size, seqlen_k, nheads_k, head_dim_q)
        v: (batch_size, seqlen_k, nheads_k, head_dim_v)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim_v)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf")
        )
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def run_torch(
    q,
    k,
    v,
    bias=None,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window,
    upcast=True,
    reorder_ops=False,
):
    (_, seqlen_q, _, _) = q.shape
    (_, seqlen_k, _, _) = k.shape

    if bias is not None:
        attn_bias = bias
    else:
        attn_bias = None

    out, _ = attention_ref(
        q,
        k,
        v,
        None,
        None,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        window_size=window_size,
        upcast=upcast,
        reorder_ops=reorder_ops,
    )

    if dout == None:
        return out
    elif bias is not None:
        dq, dk, dv, dbias = torch.autograd.grad(out, (q, k, v, bias), dout)
        # If seqlen_q > seqlen_k with mask, pytorch will output NaN.
        # Align with ck behavior here
        dbias = torch.nan_to_num(dbias, nan=0.0)
        return out, dq, dk, dv, dbias
    else:
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        return out, dq, dk, dv, None


def run_ck(
    q,
    k,
    v,
    bias=None,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    deterministic=False,
    return_lse=True,
    return_attn_probs=False,
):
    out, _, S_dmask = aiter.flash_attn_func(
        q,
        k,
        v,
        dropout_p,
        causal=causal,
        window_size=window_size,
        bias=bias,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_lse=return_lse,
        return_attn_probs=return_attn_probs,
    )

    if dout == None:
        return out, None
    elif bias is not None:
        dq, dk, dv, dbias = torch.autograd.grad(out, (q, k, v, bias), dout)
        return out, dropout_mask, dq, dk, dv, dbias
    else:
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        return out, None, dq, dk, dv, None


_PARTITION_SIZE = 512
_PARTITION_SIZE_ROCM = 256
_DEVICE_PROPERTIES = torch.cuda.get_device_properties("cuda")
_ON_NAVI = (
    hasattr(_DEVICE_PROPERTIES, "gcnArchName")
    and "gfx1" in torch.cuda.get_device_properties("cuda").gcnArchName
)


def _use_rocm_custom_paged_attention(
    qtype: torch.dtype,
    head_size: int,
    block_size: int,
    gqa_ratio: int,
    max_seq_len: int,
) -> bool:
    # rocm custom page attention not support on navi (gfx1*)
    return (
        not _ON_NAVI
        and (qtype == torch.half or qtype == dtypes.bf16)
        and (head_size == 64 or head_size == 128)
        and (block_size == 16 or block_size == 32)
        and (gqa_ratio >= 1 and gqa_ratio <= 16)
        and max_seq_len <= 65536
    )


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    nkvhead,
    k_scale=torch.Tensor,  # [1] or [nkvhead, 1, seq_lenth]
    v_scale=torch.Tensor,  # [1] or [nkvhead, 1, seq_lenth]
    attn_mask: Optional[torch.Tensor] = None,
    dtype=None,
) -> torch.Tensor:
    p_scale = 1.0
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query.float(), key.float())

    # [nqhead, q_len, ctx_len]
    nqhead, q_len, ctx_len = attn_weights.shape
    attn_weights = attn_weights.view(nqhead // nkvhead, nkvhead, q_len, ctx_len)
    attn_weights *= k_scale
    attn_weights = attn_weights.view(nqhead, q_len, ctx_len)

    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()

    attn_weights = torch.softmax(attn_weights, dim=-1)

    attn_weights = attn_weights.view(nqhead // nkvhead, nkvhead, q_len, ctx_len)
    attn_weights *= v_scale
    attn_weights = attn_weights.view(nqhead, q_len, ctx_len)
    # if v_scale != 1.0:
    #     attn_weights, p_scale = aiter.per_tensor_quant(
    #         attn_weights,  quant_dtype=dtypes.i8)
    #     # attn_weights,  quant_dtype=key.dtype)
    #     # attn_weights = attn_weights.float()*p_scale

    out = torch.einsum("hqk,khd->qhd", attn_weights.float(), value.float())
    out *= p_scale
    return out.to(dtype)


def run_native(
    query,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    max_seq_len,
    kv_cache_dtype,
    num_kv_heads,
    scale,
    alibi_slopes,
    k_scale_cache,
    v_scale_cache,
    num_queries_per_kv,
    dtype,
):
    output = torch.zeros_like(query).to(dtype)
    num_query_heads = query.shape[1]
    num_kv_heads = v_cache.shape[1]
    head_size = v_cache.shape[2]
    block_size = v_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()

    # (num_blocks, num_heads, head_size // x, block_size, x)
    k_cache = (
        k_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_kv_heads, head_size)
    )
    # (num_blocks, num_heads, head_size, block_size)
    v_cache = v_cache.permute(0, 3, 1, 2).contiguous().view(-1, num_kv_heads, head_size)
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]
        ctx_len = int(seq_lens_lst[i])

        idx = [
            int(block_table[j // block_size]) * block_size + (j % block_size)
            for j in range(ctx_len)
        ]
        if k_cache.dtype == dtypes.fp8:
            keys = k_cache.view(dtypes.i8)[idx].view(dtypes.fp8)
            values = v_cache.view(dtypes.i8)[idx].view(dtypes.fp8)
        else:
            keys = k_cache[idx]
            values = v_cache[idx]
        if k_scale_cache.numel() > 1:
            k_scale = k_scale_cache[:, idx].contiguous().view(num_kv_heads, 1, ctx_len)
            v_scale = v_scale_cache[:, idx].contiguous().view(num_kv_heads, 1, ctx_len)
        else:
            k_scale = k_scale_cache  # [1]
            v_scale = v_scale_cache  # [1]

        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(ctx_len).int()
            alibi_bias = (position_ids - ctx_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)

        out = ref_masked_attention(
            q, keys, values, scale, num_kv_heads, k_scale, v_scale, alibi_bias, dtype
        )
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)
    return output  # , 1


def run_aiter(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    kv_cache_dtype: str,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
    fp8_out_scale=None,
) -> torch.Tensor:
    # Whether to use rocm custom paged attention or not
    num_seqs, num_heads, head_size = query.shape
    block_size = value_cache.shape[3]
    gqa_ratio = num_heads // num_kv_heads
    use_custom = _use_rocm_custom_paged_attention(
        query.dtype, head_size, block_size, gqa_ratio, max_seq_len
    )
    output = torch.empty_like(query)
    if use_custom:
        max_num_partitions = (
            max_seq_len + _PARTITION_SIZE_ROCM - 1
        ) // _PARTITION_SIZE_ROCM
        assert _PARTITION_SIZE_ROCM % block_size == 0
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=dtypes.fp32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)
        cpa_fp8_out = False
        if fp8_out_scale is not None:
            output = torch.empty_like(output, dtype=dtypes.fp8)
            cpa_fp8_out = True
        aiter.paged_attention_rocm(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale,
            fp8_out_scale if cpa_fp8_out else None,
            _PARTITION_SIZE_ROCM,
        )
        if cpa_fp8_out:
            return output.view(num_seqs, num_heads * head_size)
    else:
        assert use_custom == True, "rocm custom paged attention should be used"

    return output


def run_atrex(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    kv_cache_dtype: str,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
    fp8_out_scale=None,
) -> torch.Tensor:
    # Whether to use rocm custom paged attention or not
    num_seqs, num_heads, head_size = query.shape
    block_size = value_cache.shape[3]    
    max_num_partitions = (
        max_seq_len + _PARTITION_SIZE - 1
    ) // _PARTITION_SIZE
    assert _PARTITION_SIZE % block_size == 0
    x = 16 // key_cache.element_size()
    grp_size = num_heads // num_kv_heads
    # init output
    output = torch.empty_like(query).view((num_seqs, num_heads, head_size))
    exp_sums = torch.empty(
        size=(num_seqs, num_kv_heads, max_num_partitions, grp_size),
        dtype=torch.float32,
        device=output.device,
    )
    max_logits = torch.empty_like(exp_sums)
    # init tmp_output
    tmp_output = torch.empty(
        size=(num_seqs, num_kv_heads, max_num_partitions, grp_size, head_size),
        dtype=output.dtype,
        device=output.device,
    )
    query = query.view((num_seqs, num_heads, head_size))
    paged_attention_atrex(
        output,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        key_cache,
        value_cache,
        seq_lens,
        block_tables,
        scale,
        max_seq_len,
        alibi_slopes,
    )

    return output


def test_flash_attn_output(
    batch_size,
    nheads,
    seqlen_q,
    seqlen_k,
    d,
    d_v,
    dropout_p,
    causal,
    local,
    bias_type,
    deterministic,
    mha_type,
    dtype,
):
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))

    return_lse = True
    return_attn_probs = True

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d_v,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    attn_bias = None
    alibi_slopes = None
    if bias_type == "bias":
        attn_bias = torch.randn(
            seqlen_q, seqlen_k, device="cuda", dtype=dtype, requires_grad=True
        )
    elif bias_type == "alibi":
        alibi_slopes = torch.rand(batch_size, nheads, device="cuda", dtype=dtypes.fp32)

    dout = torch.randn(
        batch_size,
        seqlen_q,
        nheads,
        d_v,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    out, dropout_mask, dq, dk, dv, dbias = run_ck(
        q,
        k,
        v,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        causal,
        window_size,
        deterministic,
        return_lse,
        return_attn_probs,
    )

    out_ref, dq_ref, dk_ref, dv_ref, dbias_ref = run_torch(
        q,
        k,
        v,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        None,
        causal,
        window_size,
    )

    out_pt, dq_pt, dk_pt, dv_pt, dbias_pt = run_torch(
        q,
        k,
        v,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        None,
        causal,
        window_size,
        upcast=False,
        reorder_ops=True,
    )

    return out, out_ref, out_pt


# 以下是从test_pa.py移植过来的test_paged_attention函数及其依赖代码

uniform_range = (-1, 1)
STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": dtypes.bf16,
    "float": dtypes.fp32,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}


def get_kv_cache_torch_dtype(
    cache_dtype,
    model_dtype=None,
):
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype


def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype,
    model_dtype=None,
    seed: int = 0,
    device="cuda",
):

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    x = 16 // torch_dtype.itemsize
    k_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    k_caches = []
    for _ in range(num_layers):
        k_cache = torch.empty(size=k_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            k_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        k_caches.append(k_cache)

    v_cache_shape = (num_blocks, num_heads, head_size, block_size)
    v_caches = []
    for _ in range(num_layers):
        v_cache = torch.empty(size=v_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            v_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        v_caches.append(v_cache)
    return k_caches, v_caches


def asm_V_shuffle(VC):
    # [num_blocks, num_kv_heads, head_size, block_size]
    x = 16 // VC.element_size()
    num_blocks, num_kv_heads, head_size, block_size = VC.shape
    VC = VC.view(num_blocks, num_kv_heads, head_size, block_size // x, x)
    # [num_blocks, num_kv_heads, block_size/X, head_size, X]
    VC = VC.permute(0, 1, 3, 2, 4).contiguous()
    return VC


def test_paged_attention(
    ctx_lens: int,
    num_seqs: int,
    num_heads: int,
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    device: str,
):
    torch.set_default_device(device)
    # Using default kv_scale
    k_scale = v_scale = torch.tensor(1.0, device=device, dtype=dtypes.fp32)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=dtypes.fp32)
    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    max_seq_len = ctx_lens
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = max_num_blocks_per_seq * num_seqs

    query = torch.empty_strided(
        (num_seqs, num_query_heads, head_size),
        ((num_query_heads + 2 * num_kv_heads) * head_size, head_size, 1),
        dtype=dtype,
    )
    query.uniform_(*uniform_range)

    seq_lens = [ctx_lens for _ in range(num_seqs)]
    seq_lens = torch.tensor(seq_lens, dtype=torch.int)

    # Create the block tables.
    import random

    block_tables_lst = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)

    block_tables = torch.tensor(block_tables_lst, dtype=torch.int)

    # Create the KV caches.
    k_caches, v_caches = kv_cache_factory(
        num_blocks,
        block_size,
        1,
        num_kv_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        seed,
        device,
    )
    k_cache, v_cache = k_caches[0], v_caches[0]

    # Test run_aiter
    out_aiter = run_aiter(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        kv_cache_dtype,
        num_kv_heads,
        scale,
        alibi_slopes,
        k_scale,
        v_scale,
    )

    out_atrex = run_atrex(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        kv_cache_dtype,
        num_kv_heads,
        scale,
        alibi_slopes,
        k_scale,
        v_scale,
    )

    out_native = run_native(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        kv_cache_dtype,
        num_kv_heads,
        scale,
        alibi_slopes,
        k_scale,
        v_scale,
        num_queries_per_kv,
        dtype,
    )
    if torch.allclose(out_native, out_aiter, atol=1e-2, rtol=1e-2):
        print("run_aiter test passed")
    else:
        print(
            f"Output difference: max diff = {(out_native - out_aiter).abs().max().item()}"
        )

    if torch.allclose(out_native, out_atrex, atol=1e-2, rtol=1e-2):
        print("run_atrex test passed")
    else:
        print(
            f"Output difference: max diff = {(out_native - out_atrex).abs().max().item()}"
        )
        return {"test_status": "failed"}

    # Test run_aiter_asm (only for bf16)
    # if dtype == dtypes.bf16:
    #     out_aiter_asm = aiter.pa_fwd_asm(
    #         query.contiguous(),  # this kernel need contiguous buffer
    #         k_cache,
    #         asm_V_shuffle(v_cache),
    #         block_tables,
    #         seq_lens,
    #         max_num_blocks_per_seq,
    #     )
    #     print("run_aiter_asm test passed")

    #     # Check if outputs are close
    #     if torch.allclose(out_aiter, out_aiter_asm, atol=1e-2, rtol=1e-2):
    #         print("run_aiter and run_aiter_asm outputs are close")
    #     else:
    #         print(f"Output difference: max diff = {(out_aiter - out_aiter_asm).abs().max().item()}")

    print(
        f"Test completed: {ctx_lens=}, {num_seqs=}, {num_heads=}, {head_size=}, {use_alibi=}, {block_size=}, {dtype=}, {kv_cache_dtype=}"
    )
    return {"test_status": "passed"}


class FmhaTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    MHATYPES = ["mha"]
    DETERMINISTICS = [False]
    BIAS_TYPES = ["no"]
    CAUSAL_OPTIONS = [True]
    LOCAL_OPTIONS = [False]
    DROPOUT_P = [0.0]
    BATCH_SIZES = [5]
    NHEADS = [6]
    SEQLENS = [
        (113, 203),
    ]
    HEAD_DIMS = [
        (256, 256),
    ]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def test_fmha(self):
        for params in itertools.product(
            self.BATCH_SIZES,
            self.NHEADS,
            self.SEQLENS,
            self.HEAD_DIMS,
            self.DROPOUT_P,
            self.CAUSAL_OPTIONS,
            self.LOCAL_OPTIONS,
            self.BIAS_TYPES,
            self.DETERMINISTICS,
            self.MHATYPES,
            self.DTYPES,
        ):
            (
                batch_size,
                nheads,
                (seqlen_q, seqlen_k),
                (d, d_v),
                dropout_p,
                causal,
                local,
                bias_type,
                deterministic,
                mha_type,
                dtype,
            ) = params

            with self.subTest(
                batch_size=batch_size,
                nheads=nheads,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                d=d,
                d_v=d_v,
                dropout_p=dropout_p,
                causal=causal,
                local=local,
                bias_type=bias_type,
                deterministic=deterministic,
                mha_type=mha_type,
                dtype=dtype,
            ):
                out, out_ref, out_pt = test_flash_attn_output(
                    batch_size,
                    nheads,
                    seqlen_q,
                    seqlen_k,
                    d,
                    d_v,
                    dropout_p,
                    causal,
                    local,
                    bias_type,
                    deterministic,
                    mha_type,
                    dtype,
                )
                print(f"Output max diff: {(out - out_ref).abs().max().item()}")
                print(
                    f"Output Pytorch max diff: {(out_pt - out_ref).abs().max().item()}"
                )
                self.assertTrue(torch.allclose(out, out_ref, atol=1e-2, rtol=1e-2))

    def test_paged_attention_benchmark(self):
        # 简单的测试配置
        num_heads = (32, 8)
        ctx_len = 128
        torch_dtype = dtypes.bf16
        result = test_paged_attention(
            ctx_len, 128, num_heads, 128, False, 16, torch_dtype, "auto", 0, "cuda:0"
        )
        self.assertIsInstance(result, dict)
        self.assertIn("test_status", result)
        self.assertEqual(result["test_status"], "passed")
        print("Paged attention test completed successfully")


if __name__ == "__main__":
    main()
