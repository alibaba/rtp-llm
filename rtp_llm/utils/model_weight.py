import logging
import math
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import torch

from rtp_llm.utils.util import check_with_info


def get_pad_size(size: int, align_size: int) -> int:
    """Calculate padding size to align to align_size."""
    return (align_size - (size % align_size)) % align_size


def w_half1_t(ts: List[torch.Tensor], inter_size: int):
    return ts[0][:inter_size, ...].T.contiguous()


def w_half2_t(ts: List[torch.Tensor], inter_size: int):
    return ts[0][inter_size:, ...].T.contiguous()


def w_half1(ts: List[torch.Tensor], inter_size: int):
    return ts[0][:inter_size, ...].contiguous()


def w_half2(ts: List[torch.Tensor], inter_size: int):
    return ts[0][inter_size:, ...].contiguous()


def concat_0(ts: List[torch.Tensor]) -> torch.Tensor:
    if len(ts) == 1:
        return ts[0]
    # torch.concat() dose not support fp8 in current rocm torch version
    if ts[0].dtype in [
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    ]:
        dtype = ts[0].dtype
        out_u8 = torch.concat([x.view(torch.uint8) for x in ts], dim=0).contiguous()
        return out_u8.view(dtype)
    else:
        return torch.concat(ts, dim=0).contiguous()


def concat_1(ts: List[torch.Tensor]) -> torch.Tensor:
    if len(ts) == 1:
        return ts[0]
    return torch.concat(ts, dim=1).contiguous()


def pad(ts: List[torch.Tensor], align_size: int, dim: int):
    """Pad tensor to align_size along the specified dimension.

    Args:
        ts: List containing the tensor to pad
        align_size: Alignment size for padding (0 means no padding needed)
        dim: Dimension to pad (0 or 1)
    """
    if align_size == 0:
        return ts[0].contiguous()

    # Calculate padding size based on tensor shape and align_size
    size_to_align = ts[0].shape[dim]
    pad_size = get_pad_size(size_to_align, align_size)

    logging.debug(
        "align_size: %s, size_to_align: %s, pad_size: %s, dim: %s",
        align_size,
        size_to_align,
        pad_size,
        dim,
    )

    if pad_size == 0:
        return ts[0].contiguous()

    if dim == 0:
        pad_shape = [pad_size, ts[0].shape[1]]
    elif dim == 1:
        pad_shape = [ts[0].shape[0], pad_size]
    else:
        raise Exception("unknown padding dim: " + str(dim))

    z = torch.zeros(pad_shape, device=ts[0].device).to(ts[0].dtype)
    return torch.cat((ts[0], z), dim).to(ts[0].device).contiguous()


def transpose_pad(ts: List[torch.Tensor], align_size: int, dim: int):
    """Pad tensor to align_size along the specified dimension, then transpose.

    Args:
        ts: List containing the tensor to pad
        align_size: Alignment size for padding (0 means no padding needed)
        dim: Dimension to pad (0 or 1)
    """
    if align_size == 0:
        return ts[0].T.contiguous()

    # Calculate padding size based on tensor shape and align_size
    size_to_align = ts[0].shape[dim]
    pad_size = get_pad_size(size_to_align, align_size)

    if pad_size == 0:
        return ts[0].T.contiguous()

    if dim == 0:
        pad_shape = [pad_size, ts[0].shape[1]]
    elif dim == 1:
        pad_shape = [ts[0].shape[0], pad_size]
    else:
        raise Exception("unknown padding dim: " + str(dim))

    z = torch.zeros(pad_shape, device=ts[0].device).to(ts[0].dtype)
    return torch.cat((ts[0], z), dim).T.to(ts[0].device).contiguous()


def b_half_merge(ts: List[torch.Tensor]):
    n_ts_1 = []
    n_ts_2 = []
    for t in ts:
        t_a = t.chunk(2, dim=-1)
        n_ts_1.append(t_a[0].cuda())
        n_ts_2.append(t_a[1].cuda())
    return concat_0([concat_0(n_ts_1), concat_0(n_ts_2)])


def zeros(ts: List[torch.Tensor], shape: List[int]) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.half).contiguous()


def ones(ts: List[torch.Tensor], shape: List[int] = [1]) -> torch.Tensor:
    return torch.ones(shape, dtype=torch.half).contiguous()


def transpose(ts: List[torch.Tensor]) -> torch.Tensor:
    return ts[0].t().contiguous()


def identity(ts: List[torch.Tensor], allow_empty: bool = False) -> torch.Tensor:
    if len(ts) == 0:
        if allow_empty:
            return None
        else:
            raise Exception("ts is empty")
    return ts[0].contiguous()


def multipy_identity(ts: List[torch.Tensor], scale: float) -> torch.Tensor:
    t = identity(ts)
    return t * scale


def div(ts: List[torch.Tensor], allow_empty: bool = False) -> torch.Tensor:
    if len(ts) == 0:
        if allow_empty:
            return None
        else:
            raise Exception("ts is empty")
    return (1.0 / ts[0]).to(torch.float32).contiguous()


def get_tensor_reciprocal(ts: List[torch.Tensor]) -> torch.Tensor:
    return 1.0 / ts[0].reshape(-1)


def get_tensor_from_scalar(ts: List[torch.Tensor]) -> torch.Tensor:
    return ts[0].reshape(-1)


def get_list_tensor_reciprocal(ts: List[torch.Tensor]) -> torch.Tensor:
    for i in range(len(ts)):
        if (ts[i] == 0).any():
            raise ValueError(
                f"Tensor at index {i} contains zero elements, causing division by zero."
            )
        ts[i] = 1.0 / ts[i].reshape(-1)
    return concat_0(ts)


def get_list_tensor_from_scalar(ts: List[torch.Tensor]) -> torch.Tensor:
    for i in range(len(ts)):
        ts[i] = ts[i].reshape(-1)
    return concat_0(ts)


def tolerate_failed(
    ts: List[torch.Tensor], origin_func: Callable[[List[torch.Tensor]], torch.Tensor]
) -> torch.Tensor:
    try:
        return origin_func(ts)
    except Exception as _:
        return None


def choose_available(
    ts: List[Optional[torch.Tensor]],
    origin_func_list: List[Callable[[List[torch.Tensor]], torch.Tensor]],
) -> torch.Tensor:
    for t, func in zip(ts, origin_func_list):
        if t is not None and len(ts) > 0:
            return func([t])
    raise ValueError(f"all tensor is empty, but not allow empty")


def shift_one(ts: List[torch.Tensor], allow_empty: bool = False) -> torch.Tensor:
    if len(ts) == 0 and allow_empty:
        return None
    return (ts[0] + 1.0).contiguous()


def sp_0(t: torch.Tensor, tp: int, tp_rank: int, **kwargs: Any) -> torch.Tensor:
    return torch.split(t, t.shape[0] // tp, dim=0)[tp_rank]


def ffn_sp_0(
    t: torch.Tensor,
    tp: int,
    tp_rank: int,
    ep: int,
    ep_rank: int,
    dp: int,
    dp_rank: int,
    ffn_tp_rank: int,
    ffn_tp_size: int,
    **kwargs: Any,
) -> torch.Tensor:
    return torch.split(t, t.shape[0] // ffn_tp_size, dim=0)[ffn_tp_rank]


def sp_1(t: torch.Tensor, tp: int, tp_rank: int, **kwargs: Any) -> torch.Tensor:
    return torch.split(t, t.shape[1] // tp, dim=1)[tp_rank]


def sp_neg1(t: torch.Tensor, tp: int, tp_rank: int, **kwargs: Any) -> torch.Tensor:
    return torch.split(t, t.shape[-1] // tp, dim=-1)[tp_rank]


def ffn_sp_neg1(
    t: torch.Tensor,
    tp: int,
    tp_rank: int,
    ep: int,
    ep_rank: int,
    dp: int,
    dp_rank: int,
    ffn_tp_rank: int,
    ffn_tp_size: int,
    **kwargs: Any,
) -> torch.Tensor:
    return torch.split(t, t.shape[-1] // ffn_tp_size, dim=-1)[ffn_tp_rank]


def sp_neg1_part_by_head(
    t: torch.Tensor,
    tp: int,
    tp_rank: int,
    head_num: int,
    size_per_head: int,
    **kwargs: Any,
) -> torch.Tensor:
    t_0 = torch.split(
        t[:, : head_num * size_per_head], head_num * size_per_head // tp, dim=-1
    )[tp_rank]
    t_1 = t[:, head_num * size_per_head :]
    return torch.concat([t_0, t_1], dim=-1)


def sp_id(t: torch.Tensor, tp: int, tp_rank: int, **kwargs: Any) -> torch.Tensor:
    return t


def sp_moe_neg1(
    t: torch.Tensor,
    tp: int,
    tp_rank: int,
    ep: int,
    ep_rank: int,
    dp: int,
    dp_rank: int,
    use_stack_weight: bool,
    **kwargs: Any,
) -> torch.Tensor:
    if use_stack_weight:
        assert len(t.shape) == 3, "t.shape: " + str(t.shape)
        return t.split(t.shape[0] // ep, dim=0)[ep_rank]
    else:
        raise ValueError("use_stack_weight is False")
        return t


def sp_moe_w1(
    t: torch.Tensor,
    tp: int,
    tp_rank: int,
    ep: int,
    ep_rank: int,
    dp: int,
    dp_rank: int,
    use_stack_weight: bool,
    **kwargs: Any,
) -> torch.Tensor:
    # [expert_num, 2*n, k]
    if use_stack_weight:
        assert len(t.shape) == 3, "t.shape: " + str(t.shape)
        return t.split(t.shape[0] // ep, dim=0)[ep_rank]
    else:
        raise ValueError("use_stack_weight is False")
        return t


def stack_(ts: List[torch.Tensor]):
    return stack_0(ts)


def stack_pad(ts: List[torch.Tensor], moe_align_size: int, dim: int):
    """Stack tensors and pad to moe_align_size along the specified dimension.

    Args:
        ts: List of tensors to stack
        moe_align_size: Alignment size for MoE padding
        dim: Dimension to pad (1 or 2 after stacking)
    """
    t = torch.stack(ts, dim=0)

    # Calculate padding size based on stacked tensor shape and moe_align_size
    if dim == 1:
        size_to_align = t.shape[1]
    elif dim == 2:
        size_to_align = t.shape[2]
    else:
        raise Exception("moe unknown padding dim: " + str(dim))

    pad_size = get_pad_size(size_to_align, moe_align_size)

    if pad_size == 0:
        return t

    if dim == 1:
        pad_shape = [t.shape[0], pad_size, t.shape[2]]
    elif dim == 2:
        pad_shape = [t.shape[0], t.shape[1], pad_size]

    z = torch.zeros(pad_shape, device=t.device).half()
    return torch.concat([t, z], dim)


def stack_moe_w1_pad(ts: List[torch.Tensor], moe_align_size: int, dim: int):
    """Stack MoE w1/w3 (gate/up) tensors and pad to moe_align_size.

    Args:
        ts: List of tensors (first half are gate weights, second half are up weights)
        moe_align_size: Alignment size for MoE padding
        dim: Dimension to pad (1 after stacking)
    """
    gate_ = ts[: len(ts) // 2]
    up_ = ts[len(ts) // 2 :]
    w1 = torch.stack(gate_, dim=0)
    w3 = torch.stack(up_, dim=0)

    if dim != 1:
        raise Exception("moe unknown padding dim: " + str(dim))

    # Calculate padding size based on stacked tensor shape and moe_align_size
    size_to_align = w1.shape[1]
    pad_size = get_pad_size(size_to_align, moe_align_size)

    if pad_size > 0:
        pad_shape = [w1.shape[0], pad_size, w1.shape[2]]
        z = torch.zeros(pad_shape, device=w1.device).half()
        w1 = torch.cat((w1, z), dim=1)
        w3 = torch.cat((w3, z), dim=1)

    x = torch.concat([w1, w3], dim=1)
    return x


def stack_0(ts: List[torch.Tensor]) -> torch.Tensor:
    if len(ts) == 1:
        return ts[0].unsqueeze(0)
    # torch.stack() does not support fp8 in current rocm torch version
    if ts[0].dtype in [
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    ]:
        dtype = ts[0].dtype
        out_u8 = torch.concat(
            [x.view(torch.uint8).unsqueeze(0) for x in ts], dim=0
        ).contiguous()
        return out_u8.view(dtype)
    else:
        return torch.stack(ts, dim=0).contiguous()


def stack_moe_w1(ts: List[torch.Tensor]):
    gate = ts[: len(ts) // 2]
    up = ts[len(ts) // 2 :]
    ws = []
    for w1, w3 in zip(gate, up):
        ws.append(concat_0([w1, w3]))
    x = stack_0(ws)
    return x


def get_sp_tensor(
    t: torch.Tensor,
    head_num: int,
    head_num_kv: int,
    size_per_head: int,
    tp: int,
    tp_rank: int,
    **kwargs,
):
    t = t.reshape([-1, (head_num + head_num_kv * 2) * size_per_head])
    q_hidden = head_num * size_per_head
    kv_hidden = head_num_kv * size_per_head
    if len(t.shape) == 1:
        t = t.unsqueeze(0)
    qs = sp_neg1(t[:, :q_hidden], tp, tp_rank)
    if head_num_kv == 1:
        ks = t[:, q_hidden : q_hidden + kv_hidden]
        vs = t[:, q_hidden + kv_hidden :]
    else:
        ks = sp_neg1(t[:, q_hidden : q_hidden + kv_hidden], tp, tp_rank)
        vs = sp_neg1(t[:, q_hidden + kv_hidden :], tp, tp_rank)
    return torch.concat([qs, ks, vs], dim=1).contiguous()


# MHA layout: [D, head*size_per_head, head*size_per_head, head*size_per_head] == [D, 3, D] (sp_neg)
# MQA layout: [D, head*size_per_head, kv_head*size_per_head, kv_head*size_per_head] (sp_head)
def sp_head(
    t: torch.Tensor,
    hidden_size: int,
    head_num: int,
    head_num_kv: int,
    size_per_head: int,
    bits: int,
    **kwargs: Any,
) -> torch.Tensor:
    # quant
    if len(t.shape) == 2 and t.dtype == torch.int32:
        nums = 32 // bits
        # awq
        if (
            t.shape[0] == hidden_size
            and t.shape[1] == ((head_num + head_num_kv * 2) * size_per_head) // nums
        ):
            size_per_head = size_per_head // nums
    return get_sp_tensor(
        t,
        head_num=head_num,
        head_num_kv=head_num_kv,
        size_per_head=size_per_head,
        **kwargs,
    )


def sp_head_s(t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return get_sp_tensor(t, **kwargs)


def sp_head_z(
    t: torch.Tensor, size_per_head: int, bits: int, **kwargs: Any
) -> torch.Tensor:
    size_per_head = size_per_head // (32 // bits)
    z = get_sp_tensor(t, size_per_head=size_per_head, **kwargs)
    return z


def sp_head_b(t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return get_sp_tensor(t, **kwargs)


def sp_head_qk_norm(
    t: torch.Tensor, tp, tp_rank, head_num, head_num_kv, size_per_head, **kwargs: Any
) -> torch.Tensor:
    q_hidden = head_num * size_per_head
    t = t.reshape(1, -1)
    qs = sp_neg1(t[:, :q_hidden], tp, tp_rank)
    if head_num_kv == 1:
        ks = t[:, q_hidden:]
    else:
        ks = sp_neg1(t[:, q_hidden:], tp, tp_rank)
    return torch.concat([qs, ks], dim=1).contiguous()


def sp_head_lora(t: torch.Tensor, hidden_size, **kwargs: Any) -> torch.Tensor:
    hidden_size = t.shape[0]
    return get_sp_tensor(t, hidden_size=hidden_size, **kwargs)


def sp_head_gemm_a8(t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return get_sp_tensor(t.reshape([t.shape[0], -1]).T, **kwargs).T


def get_sp_tensor_blocked(
    t: torch.Tensor,
    head_num: int,
    head_num_kv: int,
    size_per_head: int,
    tp: int,
    tp_rank: int,
    **kwargs,
):
    block_size = 128
    check_with_info(
        (head_num + head_num_kv * 2) * size_per_head % block_size == 0,
        "illegal head_num or size_per_head",
    )
    t = t.reshape([-1, (head_num + head_num_kv * 2) * size_per_head // block_size])
    q_hidden = head_num * size_per_head // block_size
    kv_hidden = head_num_kv * size_per_head // block_size
    if len(t.shape) == 1:
        t = t.unsqueeze(0)
    qs = sp_neg1(t[:, :q_hidden], tp, tp_rank)
    if head_num_kv == 1:
        ks = t[:, q_hidden : q_hidden + kv_hidden]
        vs = t[:, q_hidden + kv_hidden :]
    else:
        ks = sp_neg1(t[:, q_hidden : q_hidden + kv_hidden], tp, tp_rank)
        vs = sp_neg1(t[:, q_hidden + kv_hidden :], tp, tp_rank)
    return torch.concat([qs, ks, vs], dim=1).contiguous()


def sp_head_s_gemm_a8_block(t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return get_sp_tensor_blocked(t.T, **kwargs).T


def sp_head_s_gemm_a8_channel(t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return sp_head_s(t.T, **kwargs).T


def sp_head_s_gemm_a8(t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return sp_head_s(t, **kwargs)


def sp_attn_gate(
    t: torch.Tensor,
    tp: int,
    tp_rank: int,
    head_num: int,
    hidden_size: int,
    size_per_head: int,
    **kwargs: Any,
):
    local_head_num = head_num // tp
    start_idx = local_head_num * tp_rank
    end_idx = local_head_num * (tp_rank + 1)
    t = t[:, start_idx * size_per_head : end_idx * size_per_head]
    return t


def trans_qkv(
    ts: List[torch.Tensor], hidden_size: int, head_num: int, size_per_head: int = -1
) -> torch.Tensor:
    if size_per_head == -1:
        size_per_head = hidden_size // head_num
    return (
        ts[0]
        .T.reshape(hidden_size, head_num, 3, size_per_head)
        .permute(0, 2, 1, 3)
        .reshape(hidden_size, 3 * head_num * size_per_head)
        .contiguous()
    )


def qkv_transpose(ts, hidden_size):
    return ts[0].reshape(hidden_size, -1)


def trans_qkv_b(
    ts: List[torch.Tensor], hidden_size: int, head_num: int
) -> torch.Tensor:
    return (
        ts[0]
        .reshape(head_num, 3, hidden_size // head_num)
        .permute(1, 0, 2)
        .reshape(3 * hidden_size)
        .contiguous()
    )


def qkv_transpose(ts, hidden_size):
    return ts[0].reshape(hidden_size, -1)


def qkv_gather(
    ts: List[torch.Tensor],
    dim0: int,
    head_num: int,
    head_num_kv: int,
    size_per_head: int = -1,
) -> torch.Tensor:
    t = ts[0].t().contiguous().reshape(dim0, -1)
    if size_per_head == -1:
        size_per_head = t.shape[1] // (head_num + head_num_kv * 2)
    new_idxs: List[int] = []
    q2kv_ratio = head_num // head_num_kv
    for q2kv_idx in range(head_num_kv):
        base_idx = (q2kv_ratio + 2) * q2kv_idx
        new_idxs.extend(list(range(base_idx, base_idx + q2kv_ratio)))
    for q2kv_idx in range(head_num_kv):
        new_idxs.append((q2kv_ratio + 2) * q2kv_idx + q2kv_ratio)
    for q2kv_idx in range(head_num_kv):
        new_idxs.append((q2kv_ratio + 2) * q2kv_idx + q2kv_ratio + 1)
    return t.reshape(dim0, head_num + head_num_kv * 2, size_per_head)[
        :, new_idxs, :
    ].reshape(dim0, -1)


def sp_0_pad8(t: torch.Tensor, tp: int, tp_rank: int, **kwargs: Any) -> torch.Tensor:
    align_size = tp * 8
    paded_size = int(math.ceil(t.shape[0] * 1.0 / align_size) * align_size)
    pad_size = int(paded_size - t.shape[0])
    per_slice_size = int(paded_size / tp)
    if pad_size != 0 and tp_rank == tp - 1:
        if len(t.shape) == 2:
            return torch.concat(
                [
                    t[tp_rank * per_slice_size :, :],
                    torch.zeros([pad_size, t.shape[1]], device=t.device).to(t.dtype),
                ],
                dim=0,
            )
        else:
            return torch.concat(
                [
                    t[tp_rank * per_slice_size :, :],
                    torch.zeros([pad_size], device=t.device).to(t.dtype),
                ],
                dim=0,
            )
    else:
        if len(t.shape) == 2:
            return t[tp_rank * per_slice_size : (tp_rank + 1) * per_slice_size, :]
        else:
            return t[tp_rank * per_slice_size : (tp_rank + 1) * per_slice_size]


def merge_qkv_hf(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_weight = torch.concat([q.T, k.T, v.T], dim=1).contiguous()
    return qkv_weight


def merge_qkv_transpose_concat0(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_weight = torch.concat([q.T, k.T, v.T], dim=0).contiguous()
    return qkv_weight


def merge_qkv_b(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_b = torch.concat([q, k, v], dim=0).contiguous()
    return qkv_b


def trans_lora_qkv(ts: List[torch.Tensor], head_num: int, head_size: int):
    split = 3
    r = ts[0].shape[1]
    return (
        ts[0]
        .T.reshape(r, head_num, split, head_size)
        .permute(0, 2, 1, 3)
        .reshape(r, split, head_num * head_size)
        .contiguous()
    )


def merge_qkv_lora_A(
    ts: List[torch.Tensor],
    allow_empty=False,
    hidden_size: int = None,
    head_num: int = None,
    head_num_kv: int = None,
    size_per_head: int = -1,
):
    q, k, v = ts
    rank = -1
    if q is not None:
        rank = int(q.numel() // hidden_size)
    elif k is not None:
        rank = int(k.numel() // hidden_size)
    else:
        rank = int(v.numel() // hidden_size)
    logging.debug("merge_qkv_lora_A rank %d", rank)
    if allow_empty:
        if q is None:
            q = torch.zeros(rank, hidden_size)
            logging.info("lora_B  is empty, use zeros instead")
        if k is None:
            k = torch.zeros(rank, hidden_size)
            logging.info("lora_B  is empty, use zeros instead")
        if v is None:
            v = torch.zeros(rank, hidden_size)
            logging.info("lora_B  is empty, use zeros instead")

    try:
        qkv_weight = torch.concat([q.T, k.T, v.T], dim=1).contiguous()
        return qkv_weight
    except:
        raise Exception(
            f"merge_qkv_lora_A failed: q shape {q.shape}, k shape {k.shape}, v shape {v.shape}"
        )


def merge_qkv_lora_B(
    ts: List[torch.Tensor],
    allow_empty=False,
    hidden_size: int = None,
    head_num: int = None,
    head_num_kv: int = None,
    size_per_head: int = -1,
):
    q, k, v = ts
    if q is not None:
        rank = int(q.numel() // (head_num * size_per_head))
    elif k is not None:
        rank = int(k.numel() // (head_num_kv * size_per_head))
    else:
        rank = int(v.numel() // (head_num_kv * size_per_head))
    logging.debug("merge_qkv_lora_B rank %d", rank)

    if allow_empty:
        if q is None:
            q = torch.zeros(head_num * size_per_head, rank)
        if k is None:
            k = torch.zeros(head_num_kv * size_per_head, rank)
        if v is None:
            v = torch.zeros(head_num_kv * size_per_head, rank)
    t_q = torch.zeros_like(q)
    t_k = torch.zeros_like(k)
    t_v = torch.zeros_like(v)

    return torch.cat(
        (
            torch.cat((q, t_q, t_q), dim=1),
            torch.cat((t_k, k, t_k), dim=1),
            torch.cat((t_v, t_v, v), dim=1),
        )
    ).T.contiguous()


def merge_te_qkv(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_weight = torch.concat([q, k, v], dim=0).contiguous()
    return qkv_weight


def merge_block_scale(ts: List[torch.Tensor]):
    check_with_info(len(ts) == 3, "qkv scale should have 3 tensors")
    out_scale = torch.concat(ts, dim=0).contiguous()
    return out_scale


# from [torch.Size(1), torch.Size(1), torch.Size(1)] to torch.Size(3 * hidden_size)
def expand_scale(ts: List[torch.Tensor], hidden_size: int):
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


# mla function


def yarn_get_mscale(scale: float = 1, mscale: float = 1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def kv_split(
    ts: List[torch.Tensor],
    kv_lora_rank: int,
    nope_head_dim: int,
    v_head_dim: int,
    idx: int,
):
    res_list = (
        ts[0]
        .reshape(-1, nope_head_dim + v_head_dim, kv_lora_rank)
        .split([nope_head_dim, v_head_dim], dim=1)
    )
    res = res_list[idx]
    res = res.reshape(-1, kv_lora_rank)
    # [head_num*head_dim, lora_rank]
    return res.contiguous()


def kv_split1(
    ts: List[torch.Tensor], kv_lora_rank: int, nope_head_dim: int, v_head_dim: int
) -> torch.Tensor:
    k, _ = (
        ts[0]
        .transpose(0, 1)
        .reshape(kv_lora_rank, -1, nope_head_dim + v_head_dim)
        .split([nope_head_dim, v_head_dim], dim=-1)
    )
    k = k.reshape(kv_lora_rank, -1)
    # [lora_rank, head_num * head_dim]
    return k.contiguous()


def kv_split2(
    ts: List[torch.Tensor], kv_lora_rank: int, nope_head_dim: int, v_head_dim: int
) -> torch.Tensor:
    _, v = (
        ts[0]
        .transpose(0, 1)
        .reshape(kv_lora_rank, -1, nope_head_dim + v_head_dim)
        .split([nope_head_dim, v_head_dim], dim=-1)
    )
    v = v.reshape(kv_lora_rank, -1)
    return v.contiguous()


def mla_pad(
    ts: List[torch.Tensor], head_num: int, nope_head_dim: int, rope_head_dim: int
) -> torch.Tensor:
    t = ts[0]
    t = t.reshape(-1, head_num, nope_head_dim)
    z = torch.zeros((t.shape[0], head_num, rope_head_dim), device=t.device).to(t.dtype)
    t = torch.cat([t, z], dim=-1)
    t = t.reshape(-1, head_num * (nope_head_dim + rope_head_dim))
    return t.contiguous()


def mla_pad_t(
    ts: List[torch.Tensor], head_num: int, nope_head_dim: int, rope_head_dim: int
) -> torch.Tensor:
    t = ts[0]
    t = t.reshape(-1, head_num, nope_head_dim)
    z = torch.zeros(t.shape[0], head_num, rope_head_dim, device=t.device).to(t.dtype)
    t = torch.cat([t, z], dim=-1)
    t = t.reshape(-1, head_num * (nope_head_dim + rope_head_dim))
    return t.T.contiguous()


def transpose_slice_k(
    ts: List[torch.Tensor],
    head_num: int,
    nope_head_dim: int,
    v_head_dim: int,
    lora_rank: int,
) -> torch.Tensor:
    t = ts[0]
    t = t.transpose(0, 1).view(lora_rank, head_num, nope_head_dim + v_head_dim)
    return t[:, :, :nope_head_dim].permute(1, 2, 0).contiguous()


def transpose_slice_v(
    ts: List[torch.Tensor],
    head_num: int,
    nope_head_dim: int,
    v_head_dim: int,
    lora_rank: int,
) -> torch.Tensor:
    t = ts[0]
    t = t.transpose(0, 1).view(lora_rank, head_num, nope_head_dim + v_head_dim)
    return t[:, :, nope_head_dim:].transpose(0, 1).contiguous()


def mla_pad_scale(
    ts: List[torch.Tensor],
    head_num: int,
    nope_head_dim: int,
    rope_head_dim: int,
    group_size: int,
) -> torch.Tensor:
    t = ts[0]
    t = t.reshape(-1, head_num * nope_head_dim // group_size)
    z = torch.zeros(
        t.shape[0], head_num * rope_head_dim // group_size, device=t.device
    ).to(t.dtype)
    t = torch.cat([t, z], dim=-1)
    t = t.reshape(-1, head_num * (nope_head_dim + rope_head_dim) // group_size)
    return t.contiguous()


def concat_0_tranpose(ts: List[torch.Tensor]):
    return torch.concat(ts, dim=0).transpose(0, 1).contiguous()


def transpose_kv_rope(ts: List[torch.Tensor], kv_lora_rank: int, rope_size: int):
    rope_size_half = rope_size // 2
    kva = ts[0]
    kva[kv_lora_rank:, :] = (
        kva[kv_lora_rank:, :]
        .reshape([rope_size_half, 2, -1])
        .transpose(0, 1)
        .reshape([rope_size, -1])
    )
    return kva.reshape(ts[0].shape).contiguous()


def transpose_q_rope(
    ts: List[torch.Tensor], head_num: int, nope_head_dim: int, rope_size: int
):
    rope_size_half = rope_size // 2
    q = ts[0]
    q = q.reshape([head_num, nope_head_dim + rope_size, -1])
    q[:, nope_head_dim:, :] = (
        q[:, nope_head_dim:, :]
        .reshape([head_num, rope_size_half, 2, -1])
        .transpose(1, 2)
        .reshape([head_num, rope_size, -1])
    )
    return q.reshape(ts[0].shape).contiguous()


# for w1 w3
def pad_w13(ts: List[torch.Tensor], align_size: int, dim: int):
    """Pad w1 and w3 tensors to align_size and concatenate them.

    Args:
        ts: List containing w1 and w3 tensors
        align_size: Alignment size for padding
        dim: Dimension to pad and concatenate
    """
    w1 = pad([ts[0]], align_size, dim)
    w3 = pad([ts[1]], align_size, dim)
    if w1.dtype in [
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    ]:
        dtype = w1.dtype
        out_u8 = torch.concat(
            [w1.view(torch.uint8), w3.view(torch.uint8)], dim=dim
        ).contiguous()
        return out_u8.view(dtype)
    else:
        return torch.concat([w1, w3], dim=dim).contiguous()


def transpose_w13(ts: List[torch.Tensor]):
    w1 = transpose([ts[0]])
    w3 = transpose([ts[1]])
    return torch.concat([w1, w3], dim=-1).contiguous()


def transpose_w13_2(ts: List[torch.Tensor]):
    w1 = transpose([ts[0]])
    w3 = transpose([ts[1]])
    return torch.concat([w1, w3], dim=0).contiguous()


def concat_w13(ts: List[torch.Tensor]):
    return torch.concat(ts, dim=-1).contiguous()


def concat_w13_2(ts: List[torch.Tensor]):
    # torch.concat() dose not support fp8 in current rocm torch version
    if ts[0].dtype in [
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    ]:
        dtype = ts[0].dtype
        out_u8 = torch.concat([x.view(torch.uint8) for x in ts], dim=0).contiguous()
        return out_u8.view(dtype)
    else:
        return torch.concat(ts, dim=0).contiguous()


def ffn_sp_neg1_w13(
    t: torch.Tensor,
    tp: int,
    tp_rank: int,
    ep: int,
    ep_rank: int,
    dp: int,
    dp_rank: int,
    ffn_tp_rank: int,
    ffn_tp_size: int,
    **kwargs: Any,
) -> torch.Tensor:
    w1, w3 = torch.chunk(t, 2, dim=-1)
    w1 = ffn_sp_neg1(
        w1, tp, tp_rank, ep, ep_rank, dp, dp_rank, ffn_tp_rank, ffn_tp_size, **kwargs
    )
    w3 = ffn_sp_neg1(
        w3, tp, tp_rank, ep, ep_rank, dp, dp_rank, ffn_tp_rank, ffn_tp_size, **kwargs
    )
    return concat_w13([w1, w3])


def ffn_sp_0_w13(
    t: torch.Tensor,
    tp: int,
    tp_rank: int,
    ep: int,
    ep_rank: int,
    dp: int,
    dp_rank: int,
    ffn_tp_rank: int,
    ffn_tp_size: int,
    **kwargs: Any,
) -> torch.Tensor:
    w1, w3 = torch.chunk(t, 2, dim=-1)
    w1 = ffn_sp_0(
        w1, tp, tp_rank, ep, ep_rank, dp, dp_rank, ffn_tp_rank, ffn_tp_size, **kwargs
    )
    w3 = ffn_sp_0(
        w3, tp, tp_rank, ep, ep_rank, dp, dp_rank, ffn_tp_rank, ffn_tp_size, **kwargs
    )
    return concat_w13([w1, w3])


def sp_0_w13(
    t: torch.Tensor,
    tp: int,
    tp_rank: int,
    ep: int,
    ep_rank: int,
    dp: int,
    dp_rank: int,
    ffn_tp_rank: int,
    ffn_tp_size: int,
    **kwargs: Any,
) -> torch.Tensor:
    w1, w3 = torch.chunk(t, 2, dim=0)
    w1 = sp_0(w1, ffn_tp_size, ffn_tp_rank, **kwargs)
    w3 = sp_0(w3, ffn_tp_size, ffn_tp_rank, **kwargs)
    return torch.concat([w1, w3], dim=0)


def split_slopes_tp(slopes: torch.Tensor, head_num: int, tp: int, tp_rank: int):
    local_head_num = 1 if head_num == 1 else head_num // tp
    start_pos = local_head_num * tp_rank
    return slopes[start_pos : start_pos + local_head_num]


def get_slopes(n: int) -> List[float]:
    def get_slopes_power_of_2(n: int) -> List[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))

        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def slopes(ts: List[torch.Tensor], n: int):
    slopes = torch.Tensor(get_slopes(n))
    return slopes


class W:
    # global
    embedding = "embedding"
    lm_head = "lm_head"
    lm_head_b = "lm_head_b"
    prefix_w = "transformer.prefix_encoder.embedding.weight"
    pre_decoder_ln_gamma = "pre_decoder_layernorm.gamma"
    pre_decoder_ln_beta = "pre_decoder_layernorm.bias"
    positional_embedding = "position_encoding.weight"
    token_type_embedding = "token_type_embedding.weight"
    final_ln_gamma = "final_layernorm.gamma"
    final_ln_beta = "final_layernorm.beta"

    # mtp
    multi_tokens_predict_enorm = "multi_tokens_predict_enorm.weight"
    multi_tokens_predict_hnorm = "multi_tokens_predict_hnorm.weight"
    multi_tokens_predict_eh_proj = "multi_tokens_predict_eh_proj.weight"
    multi_tokens_predict_final_ln_gamma = "multi_tokens_predict_final_layernorm.gamma"
    multi_tokens_predict_final_ln_beta = "multi_tokens_predict_final_layernorm.beta"

    # eagle3
    eagle3_fc_proj = "eagle3_fc.weight"
    eagle3_fc_norm_gamma = "eagle3_fc.gamma"
    eagle3_input_norm_gamma = "eagle3_input.gamma"

    # attn
    pre_ln_gamma = "pre_layernorm_weights.gamma"
    pre_ln_beta = "pre_layernorm_weights.beta"
    pre_attn_ln_gamma = "pre_attn_layernorm_weights.gamma"
    pre_attn_ln_beta = "pre_attn_layernorm_weights.beta"
    attn_qkv_w = "self_attention_weights.query_weight.kernel"
    attn_qkv_b = "self_attention_weights.query_weight.bias"
    attn_ln_gamma = "self_attention_weights.attention_layernorm.gamma"
    attn_ln_beta = "self_attention_weights.attention_layernorm.beta"
    qk_ln_gamma = "self_attention_weights.qk_layernorm.gamma"
    attn_o_w = "self_attention_weights.attention_output_weight.kernel"
    attn_o_b = "self_attention_weights.attention_output_weight.bias"
    post_ln_gamma = "post_layernorm_weights.gamma"
    post_ln_beta = "post_layernorm_weights.beta"
    linear_bias_slopes = "linear_bias_slopes"
    attn_gate_w = "self_attention_weights.gate.weight"
    attn_gate_s = "self_attention_weights.gate.scale"

    # linear_attn_weights
    linear_attn_qkvz_w = "linear_attn.in_proj_qkvz.weight"
    linear_attn_qkvz_s = "linear_attn.in_proj_qkvz.scale"
    linear_attn_ba_w = "linear_attn.in_proj_ba.weight"
    linear_attn_norm_w = "linear_attn.norm.weight"
    linear_attn_dt_b = "linear_attn.dt_bias"
    linear_attn_conv1d_w = "linear_attn.conv1d.weight"
    linear_attn_alog = "linear_attn.A_log"
    linear_attn_out_w = "linear_attn.out_proj.weight"
    linear_attn_out_s = "linear_attn.out_proj.scale"

    # jina_bert
    q_ln_gamma = "self_attention_weights.q_layernorm.gamma"
    q_ln_beta = "self_attention_weights.q_layernorm.beta"
    k_ln_gamma = "self_attention_weights.k_layernorm.gamma"
    k_ln_beta = "self_attention_weights.k_layernorm.beta"

    post_ln_2_gamma = "post_layernorm_weights_2.gamma"
    post_ln_2_beta = "post_layernorm_weights_2.beta"

    # mla
    mla_fusedqkrope_w = "self_attention_weights.mla.fusedqkrope.kernel"
    mla_fusedqkrope_no_lora_w = "self_attention_weights.mla.fusedqkrope_no_lora.kernel"
    mla_q_b_w = "self_attention_weights.mla.query_b_weight.kernel"
    mla_k_nope_w = "self_attention_weights.mla.key_nope_weight.kernel"
    mla_v_w = "self_attention_weights.mla.value_weight.kernel"
    mla_q_a_ln_gamma = "self_attention_weights.mla.query_a_layernorm_weight.gamma"
    mla_q_a_ln_beta = "self_attention_weights.mla.query_a_layernorm_weight.beta"
    mla_kv_a_ln_gamma = "self_attention_weights.mla.key_value_a_layernorm_weight.gamma"
    mla_kv_a_ln_beta = "self_attention_weights.mla.key_value_a_layernorm_weight.beta"

    mla_fusedqkrope_s = "self_attention_weights.mla.fusedqkrope.weight_only_quant_scale"
    mla_fusedqkrope_no_lora_s = (
        "self_attention_weights.mla.fusedqkrope_no_lora.weight_only_quant_scale"
    )
    mla_q_b_s = "self_attention_weights.mla.query_b_weight.weight_only_quant_scale"
    mla_k_nope_s = "self_attention_weights.mla.key_nope_weight.weight_only_quant_scale"

    mla_v_s = "self_attention_weights.mla.value_weight.weight_only_quant_scale"

    # mla + absorb
    mla_kc = "self_attention_weights.mla.kc.kernel"
    mla_vc = "self_attention_weights.mla.vc.kernel"

    mla_kc_s = "self_attention_weights.mla.kc.weight_only_quant_scale"
    mla_vc_s = "self_attention_weights.mla.vc.weight_only_quant_scale"

    # ffn
    ffn = "__ffn_weights__"
    ffn_w1 = "ffn_weights.intermediate_weight.kernel"
    ffn_b1 = "ffn_weights.intermediate_weight.bias"
    ffn_w3 = "ffn_weights.intermediate_weight3.kernel"
    ffn_b3 = "ffn_weights.intermediate_weight3.bias"
    ffn_w13 = "ffn_weights.intermediate_weight13.kernel"
    ffn_b13 = "ffn_weights.intermediate_weight13.bias"
    ffn_ln_gamma = "ffn_weights.dense_layernorm.gamma"
    ffn_ln_beta = "ffn_weights.dense_layernorm.beta"
    ffn_w2 = "ffn_weights.intermediate_weight2.kernel"
    ffn_b2 = "ffn_weights.intermediate_weight2.bias"
    post_ffn_ln_gamma = "post_ffn_layernorm_weights.gamma"
    post_ffn_ln_beta = "post_ffn_layernorm_weights.beta"

    # partial moe
    moe = "__moe_weights__"
    shared_expert_gate = "ffn_weights.shared_expert_gate.kernel"
    moe_w1 = "partial_moe_weights.intermediate_weight.kernel"
    moe_b1 = "partial_moe_weights.intermediate_weight.bias"
    moe_w2 = "partial_moe_weights.intermediate_weight2.kernel"
    moe_b2 = "partial_moe_weights.intermediate_weight2.bias"
    moe_gate = "partial_moe_weights.gate.kernel"

    # eplb
    log2phy = "moe_eplb.log2phy"
    logic_expert_cnt = "moe_eplb.logic_expert_cnt"

    # deepseek3 noaux_tc
    e_score_correction_b = "partial_moe_weights.e_score_correction_bias"

    # cross attn
    cross_attn_pre_ln_gamma = "cross_attention_weights_pre_layernorm.gamma"
    cross_attn_pre_ln_beta = "cross_attention_weights_pre_layernorm.beta"
    cross_attn_qkv_w = "cross_attention_weights.query_weight.weight"
    cross_attn_qkv_b = "cross_attention_weights.query_weight.bias"
    cross_attn_o_w = "cross_attention_weights.output_weight.weight"
    cross_attn_o_b = "cross_attention_weights.output_weight.bias"

    # lora
    attn_qkv_w_lora_a = "self_attention_weights.query_weight.kernel.lora_A"
    attn_qkv_w_lora_b = "self_attention_weights.query_weight.kernel.lora_B"
    attn_o_w_lora_a = "self_attention_weights.attention_output_weight.kernel.lora_A"
    attn_o_w_lora_b = "self_attention_weights.attention_output_weight.kernel.lora_B"
    ffn_w1_lora_a = "ffn_weights.intermediate_weight.kernel.lora_A"
    ffn_w1_lora_b = "ffn_weights.intermediate_weight.kernel.lora_B"
    ffn_w3_lora_a = "ffn_weights.intermediate_weight3.kernel.lora_A"
    ffn_w3_lora_b = "ffn_weights.intermediate_weight3.kernel.lora_B"
    ffn_w2_lora_a = "ffn_weights.intermediate_weight2.kernel.lora_A"
    ffn_w2_lora_b = "ffn_weights.intermediate_weight2.kernel.lora_B"

    # gptq
    attn_qkv_z = "self_attention_weights.query_weight.zero"
    attn_qkv_s = "self_attention_weights.query_weight.weight_only_quant_scale"
    attn_o_z = "self_attention_weights.attention_output_weight.zero"
    attn_o_s = "self_attention_weights.attention_output_weight.weight_only_quant_scale"
    ffn_z1 = "ffn_weights.intermediate_weight.zero"
    ffn_s1 = "ffn_weights.intermediate_weight.weight_only_quant_scale"
    ffn_z3 = "ffn_weights.intermediate_weight3.zero"
    ffn_s3 = "ffn_weights.intermediate_weight3.weight_only_quant_scale"
    ffn_z13 = "ffn_weights.intermediate_weight13.zero"
    ffn_s13 = "ffn_weights.intermediate_weight13.weight_only_quant_scale"
    ffn_act_s = "ffn_weights.intermediate_weight2.act_quant_scale"  # gpt_xx model awq quant act need div scales
    ffn_z2 = "ffn_weights.intermediate_weight2.zero"
    ffn_s2 = "ffn_weights.intermediate_weight2.weight_only_quant_scale"
    moe_z1 = "partial_moe_weights.intermediate_weight.zero"
    moe_s1 = "partial_moe_weights.intermediate_weight.weight_only_quant_scale"
    moe_z2 = "partial_moe_weights.intermediate_weight2.zero"
    moe_s2 = "partial_moe_weights.intermediate_weight2.weight_only_quant_scale"

    # sq
    attn_i_smoother = "self_attention_weights.query_weight.smoother"
    attn_o_smoother = "self_attention_weights.attention_output_weight.smoother"
    attn_o_shift = "self_attention_weights.attention_output_weight.shift"
    ffn_smoother = "ffn_weights.intermediate_weight2.smoother"

    # per tensor quant
    pre_decoder_ln_static_quant = "pre_decoder_layernorm.static_quant"
    pre_decoder_ln_static_quant_reciprocal = (
        "pre_decoder_layernorm.static_quant_reciprocal"
    )
    pre_ln_static_quant = "pre_layernorm_weights.static_quant"
    pre_ln_static_quant_reciprocal = "pre_layernorm_weights.static_quant_reciprocal"
    attention_output_static_quant = (
        "self_attention_weights.attention_output_weight.static_quant"
    )
    attention_output_static_quant_reciprocal = (
        "self_attention_weights.attention_output_weight.static_quant_reciprocal"
    )
    post_ln_static_quant = "post_layernorm_weights.static_quant"
    post_ln_static_quant_reciprocal = "post_layernorm_weights.static_quant_reciprocal"
    ffn_intermediate_weight2_static_quant = (
        "ffn_weights.intermediate_weight2.static_quant"
    )
    ffn_intermediate_weight2_static_quant_reciprocal = (
        "ffn_weights.intermediate_weight2.static_quant_reciprocal"
    )
    ffn_intermediate_weight3_static_quant = (
        "ffn_weights.intermediate_weight3.static_quant"
    )
    ffn_intermediate_weight3_static_quant_reciprocal = (
        "ffn_weights.intermediate_weight3.static_quant_reciprocal"
    )

    post_ffn_ln_static_quant = "post_ffn_layernorm_weights.static_quant"
    post_ffn_ln_static_quant_reciprocal = (
        "post_ffn_layernorm_weights.static_quant_reciprocal"
    )

    # moe per static tensor quant
    moe_w1_input_s = "moe_w1_activation.static_quant"
    moe_w1_input_sr = "moe_w1_activation.static_quant_reciprocal"
    moe_w2_input_s = "moe_w2_activation.static_quant"
    moe_w2_input_sr = "moe_w2_activation.static_quant_reciprocal"

    # rotary embedding cos sin cache
    rope_cos_sin_cache = "rotary_embedding.cos_sin_cache"

    gpt_style_tp_strategy: Dict[str, Any] = {
        embedding: sp_neg1,
        lm_head: sp_0_pad8,
        lm_head_b: sp_0_pad8,
        pre_decoder_ln_gamma: sp_id,
        pre_decoder_ln_beta: sp_id,
        final_ln_gamma: sp_id,
        final_ln_beta: sp_id,
        pre_ln_gamma: sp_id,
        pre_ln_beta: sp_id,
        linear_bias_slopes: split_slopes_tp,
        rope_cos_sin_cache: sp_id,
        # deepseekv3-mtp
        multi_tokens_predict_enorm: sp_id,
        multi_tokens_predict_hnorm: sp_id,
        multi_tokens_predict_eh_proj: sp_id,
        multi_tokens_predict_final_ln_gamma: sp_id,
        multi_tokens_predict_final_ln_beta: sp_id,
        eagle3_fc_proj: sp_id,
        eagle3_fc_norm_gamma: sp_id,
        eagle3_input_norm_gamma: sp_id,
        pre_attn_ln_gamma: sp_id,
        pre_attn_ln_beta: sp_id,
        qk_ln_gamma: sp_head_qk_norm,
        q_ln_gamma: sp_id,
        k_ln_gamma: sp_id,
        attn_qkv_w: sp_head,
        attn_qkv_z: sp_head_z,
        attn_qkv_s: sp_head_s,
        attn_qkv_b: sp_head_b,
        attn_o_w: sp_0,
        attn_o_z: sp_0,
        attn_o_s: sp_0,
        attn_o_b: sp_id,
        attn_i_smoother: sp_0,
        attn_o_smoother: sp_0,
        attn_o_shift: sp_0,
        attn_gate_w: sp_attn_gate,
        # mla
        mla_q_b_w: sp_neg1,
        mla_fusedqkrope_w: sp_id,
        mla_fusedqkrope_s: sp_id,
        mla_fusedqkrope_no_lora_w: sp_neg1_part_by_head,
        mla_fusedqkrope_no_lora_s: sp_neg1_part_by_head,
        mla_k_nope_w: sp_neg1,
        mla_v_w: sp_neg1,
        mla_v_s: sp_neg1,
        mla_q_a_ln_gamma: sp_id,
        mla_q_a_ln_beta: sp_id,
        mla_kv_a_ln_gamma: sp_id,
        mla_kv_a_ln_beta: sp_id,
        mla_q_b_s: sp_neg1,
        mla_fusedqkrope_s: sp_id,
        mla_k_nope_s: sp_neg1,
        mla_kc: sp_0,
        mla_vc: sp_0,
        mla_kc_s: sp_0,
        mla_vc_s: sp_0,
        cross_attn_pre_ln_gamma: sp_id,
        cross_attn_pre_ln_beta: sp_id,
        cross_attn_qkv_w: sp_head,
        cross_attn_qkv_b: sp_head_b,
        cross_attn_o_w: sp_0,
        cross_attn_o_b: sp_id,
        ffn_w1: ffn_sp_neg1,
        ffn_z1: ffn_sp_neg1,
        ffn_s1: ffn_sp_neg1,
        ffn_b1: ffn_sp_neg1,
        ffn_w3: ffn_sp_neg1,
        ffn_z3: ffn_sp_neg1,
        ffn_s3: ffn_sp_neg1,
        ffn_b3: ffn_sp_neg1,
        ffn_w13: ffn_sp_neg1_w13,
        ffn_z13: ffn_sp_neg1_w13,
        ffn_s13: ffn_sp_neg1_w13,
        ffn_b13: ffn_sp_neg1_w13,
        ffn_w2: ffn_sp_0,
        ffn_z2: ffn_sp_0,
        ffn_s2: ffn_sp_0,
        ffn_b2: sp_id,
        ffn_act_s: ffn_sp_0,
        ffn_smoother: ffn_sp_0,
        moe_w1: sp_moe_w1,
        moe_z1: sp_moe_w1,
        moe_s1: sp_moe_w1,
        moe_b1: sp_moe_neg1,
        moe_w2: sp_moe_neg1,
        moe_z2: sp_moe_neg1,
        moe_s2: sp_moe_neg1,
        moe_b2: sp_moe_neg1,
        e_score_correction_b: sp_id,
        post_ln_beta: sp_id,
        post_ln_gamma: sp_id,
        positional_embedding: sp_neg1,
        attn_qkv_w_lora_a: sp_id,
        attn_qkv_w_lora_b: sp_head_lora,
        attn_o_w_lora_a: sp_0,
        attn_o_w_lora_b: sp_id,
        ffn_w1_lora_a: sp_id,
        ffn_w1_lora_b: sp_neg1,
        ffn_w3_lora_a: sp_id,
        ffn_w3_lora_b: sp_neg1,
        ffn_w2_lora_a: sp_0,
        ffn_w2_lora_b: sp_id,
        moe_gate: sp_id,
        shared_expert_gate: sp_id,
        post_ffn_ln_beta: sp_id,
        post_ffn_ln_gamma: sp_id,
        token_type_embedding: sp_neg1,
        attention_output_static_quant_reciprocal: sp_id,
    }

    weights_list = [
        embedding,
        lm_head,
        lm_head_b,
        pre_decoder_ln_gamma,
        pre_decoder_ln_beta,
        positional_embedding,
        final_ln_gamma,
        final_ln_beta,
        prefix_w,
    ]

    fp32_weights_list = [
        e_score_correction_b,
    ]

    skip_weights_list = [
        attn_qkv_w,
        attn_qkv_b,
        attn_ln_gamma,
        attn_ln_beta,
        qk_ln_gamma,
        attn_o_w,
    ]


class CkptWeightInfo:
    name: str
    merge_fun: Callable[[List[torch.Tensor]], torch.Tensor]

    # hf checkpoint没有tensor做拆分的ckpt，所以默认函数可以是identity
    def __init__(
        self,
        name: str,
        merge_fun: Callable[[List[torch.Tensor]], torch.Tensor] = identity,
    ) -> None:
        self.name = name
        self.merge_fun = merge_fun

    def tensor_name(self, layer_id: Optional[int]):
        if layer_id is not None:
            return self.name.format(i=str(layer_id), i_1=str(layer_id + 1))
        return self.name

    def __str__(self) -> str:
        return f"CkptWeightInfo[{self.name}]"

    def __repr__(self) -> str:
        return self.__str__()


class WeightStyle(Enum):
    NONE = 0
    TRT_ENGINE = 1
    TRANSFORMER_ENGINE = 2
    RTP_LLM_STYLE = 3
    RTP_SMOOTH_LLM_STYLE = (
        4  # for ooold weight converted by rtp_llm.utils.smooth_quant_convert
    )


FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN = -352.0
