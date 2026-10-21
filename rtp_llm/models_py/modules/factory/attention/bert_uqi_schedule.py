from array import array
from dataclasses import dataclass
from typing import Optional

import torch

@dataclass
class BertUqiTwoPassSchedule:
    has_b: bool
    perm: Optional[torch.Tensor]  # [total] int64 device; None => identity
    inv_perm: Optional[torch.Tensor]  # [total] int64 device
    b_rows: Optional[torch.Tensor]  # [total_b] int64 device, rows in PERMUTED layout
    qo_indptr_p1: torch.Tensor  # int32 CPU: [2N+1] if has_b else [N+1] (=cu_seqlens_host)
    qo_indptr_p2: Optional[torch.Tensor]  # [N+1] int32 CPU
    kv_indptr_p2: Optional[torch.Tensor]  # [N+1] int32 CPU (= cu_seqlens_host)


def derive_bert_uqi_segment_ids_hostlen(
    input_ids: torch.Tensor,
    cu_seqlens_dev: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    b_start_token_id: int = 2,
    sep_token_id: int = 102,
) -> torch.Tensor:
    """与 `derive_bert_uqi_segment_ids` 逐位等价, 但所有数据相关形状取自
    cu_seqlens_host —— 无隐藏 device sync (repeat_interleave 的 lengths 在 host)。
    """
    device = input_ids.device
    total = int(input_ids.shape[0])
    out = torch.zeros(total, dtype=torch.int32, device=device)
    num_seq = int(cu_seqlens_host.numel()) - 1
    if total == 0 or num_seq <= 0:
        return out
    cu_host = cu_seqlens_host.to(dtype=torch.long)
    lengths_host = cu_host[1:] - cu_host[:-1]
    seq_of_tok = torch.repeat_interleave(
        torch.arange(num_seq), lengths_host
    ).to(device, non_blocking=True)
    cu = cu_seqlens_dev.to(device=device, dtype=torch.long)
    lengths = cu[1:] - cu[:-1]
    local_pos = torch.arange(total, device=device) - cu[seq_of_tok]
    BIG = total + 1
    big = torch.full((total,), BIG, dtype=torch.long, device=device)
    cls_src = torch.where(input_ids == b_start_token_id, local_pos, big)
    first_cls = torch.full(
        (num_seq,), BIG, dtype=torch.long, device=device
    ).scatter_reduce(0, seq_of_tok, cls_src, reduce="amin", include_self=True)
    first_cls_tok = first_cls[seq_of_tok]
    sep_after = (input_ids == sep_token_id) & (local_pos >= first_cls_tok)
    sep_src = torch.where(sep_after, local_pos, big)
    first_sep = torch.full(
        (num_seq,), BIG, dtype=torch.long, device=device
    ).scatter_reduce(0, seq_of_tok, sep_src, reduce="amin", include_self=True)
    first_sep_tok = first_sep[seq_of_tok]
    has_sep = first_sep_tok < BIG
    b_end_tok = torch.where(has_sep, first_sep_tok + 1, lengths[seq_of_tok])
    in_b = (local_pos >= first_cls_tok) & (local_pos < b_end_tok)
    out[in_b] = 1
    return out


def build_bert_uqi_two_pass_schedule_from_bounds(
    b_starts_host: torch.Tensor,
    b_lens_host: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    device: torch.device,
) -> BertUqiTwoPassSchedule:
    """C++ 元数据路径: 段边界已由引擎在 host token 上扫出(PyWrappedModel),
    schedule 全部 host 构建 —— 零 GPU derive kernel、零 D2H 同步、零排序
    (permutation 由边界闭式公式直接算出)。语义与 derive+build 等价。

    Args:
        b_starts_host: [N] int32 CPU, 画像段起点(序列内), -1=无画像段。
        b_lens_host:   [N] int32 CPU, 画像段长度, 0=无。
        cu_seqlens_host: [N+1] int32 CPU (pinned 镜像)。
        device: perm/b_rows 的目标设备。
    """
    num_seq = int(cu_seqlens_host.numel()) - 1
    cu_i32_host = cu_seqlens_host.to(dtype=torch.int32)
    cu_host = cu_seqlens_host.to(dtype=torch.long)
    total = int(cu_host[-1]) if num_seq > 0 else 0
    b_lens = b_lens_host.to(torch.long)
    b_starts = b_starts_host.to(torch.long)
    total_b = int(b_lens.sum()) if num_seq > 0 else 0
    if num_seq <= 0 or total == 0 or total_b == 0:
        return BertUqiTwoPassSchedule(
            has_b=False, perm=None, inv_perm=None, b_rows=None,
            qo_indptr_p1=cu_i32_host, qo_indptr_p2=None, kv_indptr_p2=None,
        )
    lengths = cu_host[1:] - cu_host[:-1]
    a_lens = lengths - b_lens
    seg_lens = torch.stack([a_lens, b_lens], dim=1).reshape(-1)  # [2N]
    qo_p1 = torch.zeros(2 * num_seq + 1, dtype=torch.int32)
    qo_p1[1:] = seg_lens.cumsum(0).to(torch.int32)
    qo_p2 = torch.zeros(num_seq + 1, dtype=torch.int32)
    qo_p2[1:] = b_lens.cumsum(0).to(torch.int32)
    # permutation 闭式构造(免排序): token t(序列内位置 local)的新位置 =
    #   B 段内:  a_len + (local - b_start)
    #   B 段前:  local
    #   B 段后:  local - b_len
    pos = torch.arange(total)
    seq_of_tok = torch.repeat_interleave(torch.arange(num_seq), lengths)
    local = pos - cu_host[:-1][seq_of_tok]
    bs_t = b_starts[seq_of_tok]
    bl_t = b_lens[seq_of_tok]
    has_b_t = bs_t >= 0
    in_b = has_b_t & (local >= bs_t) & (local < bs_t + bl_t)
    after_b = has_b_t & (local >= bs_t + bl_t)
    new_local = torch.where(
        in_b, a_lens[seq_of_tok] + (local - bs_t), local - after_b * bl_t
    )
    new_pos = cu_host[:-1][seq_of_tok] + new_local  # = inv_perm (token t 的新位置)
    perm_host = torch.empty(total, dtype=torch.long)
    perm_host[new_pos] = pos  # perm[新位置] = 旧下标
    # B rows in permuted layout: 序列 i 尾部 [cu[i]+a_i, cu[i+1])
    seq_of_b = torch.repeat_interleave(torch.arange(num_seq), b_lens)
    within = torch.arange(total_b) - qo_p2[:-1].to(torch.long)[seq_of_b]
    b_rows_host = cu_host[:-1][seq_of_b] + a_lens[seq_of_b] + within
    return BertUqiTwoPassSchedule(
        has_b=True,
        perm=perm_host.to(device, non_blocking=True),
        inv_perm=new_pos.to(device, non_blocking=True),
        b_rows=b_rows_host.to(device, non_blocking=True),
        qo_indptr_p1=qo_p1, qo_indptr_p2=qo_p2, kv_indptr_p2=cu_i32_host,
    )


def build_bert_uqi_two_pass_schedule(
    seg_ids: torch.Tensor,
    cu_seqlens_dev: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
) -> BertUqiTwoPassSchedule:
    """从段标记构建两趟 attention 的执行计划。

    perm 把每条序列重排为 [A...|B...] (A/B 内部保持原相对顺序, stable);
    qo_indptr_p1 为 2N 段 [A_0][B_0]...[A_{N-1}][B_{N-1}] 的前缀和 (CPU int32,
    直接喂 FlashInfer plan, 其内部 `.to("cpu")` 变 no-op); pass 2 的 kv indptr
    就是 cu_seqlens_host。唯一 device->host 同步 = b_lens 的 [N] int 拷贝。
    """
    device = seg_ids.device
    total = int(seg_ids.shape[0])
    num_seq = int(cu_seqlens_host.numel()) - 1
    cu_i32_host = cu_seqlens_host.to(dtype=torch.int32)
    if total == 0 or num_seq <= 0:
        return BertUqiTwoPassSchedule(
            has_b=False, perm=None, inv_perm=None, b_rows=None,
            qo_indptr_p1=cu_i32_host, qo_indptr_p2=None, kv_indptr_p2=None,
        )
    cu_host = cu_seqlens_host.to(dtype=torch.long)
    lengths_host = cu_host[1:] - cu_host[:-1]
    seq_of_tok_host = torch.repeat_interleave(torch.arange(num_seq), lengths_host)
    seq_of_tok = seq_of_tok_host.to(device, non_blocking=True)
    b_lens_dev = torch.zeros(num_seq, dtype=torch.long, device=device).scatter_add_(
        0, seq_of_tok, seg_ids.to(torch.long)
    )
    b_lens_host = b_lens_dev.cpu()  # 全路径唯一的 device->host 同步 ([N] ints)
    total_b = int(b_lens_host.sum())
    if total_b == 0:
        return BertUqiTwoPassSchedule(
            has_b=False, perm=None, inv_perm=None, b_rows=None,
            qo_indptr_p1=cu_i32_host, qo_indptr_p2=None, kv_indptr_p2=None,
        )
    a_lens_host = lengths_host - b_lens_host
    seg_lens = torch.stack([a_lens_host, b_lens_host], dim=1).reshape(-1)  # [2N]
    qo_p1 = torch.zeros(2 * num_seq + 1, dtype=torch.int32)
    qo_p1[1:] = seg_lens.cumsum(0).to(torch.int32)
    qo_p2 = torch.zeros(num_seq + 1, dtype=torch.int32)
    qo_p2[1:] = b_lens_host.cumsum(0).to(torch.int32)
    # stable sort by (seq, seg): A 保持相对顺序在前, B 在后
    key = seq_of_tok * 2 + seg_ids.to(torch.long)
    perm = torch.argsort(key, stable=True)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(total, device=device)
    # B rows in permuted layout: 序列 i 的尾部 [cu[i]+a_i, cu[i+1]); 全在 host 构建
    seq_of_b = torch.repeat_interleave(torch.arange(num_seq), b_lens_host)
    within = torch.arange(total_b) - qo_p2[:-1].to(torch.long)[seq_of_b]
    b_rows_host = cu_host[:-1][seq_of_b] + a_lens_host[seq_of_b] + within
    b_rows = b_rows_host.to(device, non_blocking=True)
    return BertUqiTwoPassSchedule(
        has_b=True, perm=perm, inv_perm=inv_perm, b_rows=b_rows,
        qo_indptr_p1=qo_p1, qo_indptr_p2=qo_p2, kv_indptr_p2=cu_i32_host,
    )


def build_uqi_graph_layout(lengths, starts, b_lengths, token_capacity):
    """Return (length, B start, B length) for fixed slots plus padding sequence."""
    if not (len(lengths) == len(starts) == len(b_lengths)):
        raise ValueError("UQI graph metadata sizes differ")
    total_tokens = sum(lengths)
    if any(n < 0 for n in lengths) or total_tokens > token_capacity:
        raise ValueError("UQI graph token token_capacity exceeded")
    result = []
    for n, start, count in zip(lengths, starts, b_lengths):
        if count < 0 or (count and (start < 0 or start + count > n)):
            raise ValueError("invalid UQI segment bounds")
        result.append((n, start if count else n, count))
    padding = token_capacity - total_tokens
    result.append((padding, padding, 0))
    return result


def build_uqi_aiter_layout(lengths, starts, b_lengths, token_capacity, max_seq_len):
    """Fixed-slot AITER indptrs, stable permutations and output masks."""
    layout = build_uqi_graph_layout(lengths, starts, b_lengths, token_capacity)
    if max_seq_len <= 0 or any(n > max_seq_len for n in lengths):
        raise ValueError("UQI sequence exceeds AITER profile")
    perm, split_lengths, full_lengths, b_mask = [], [], [], []
    offset = 0
    for n, start, count in layout[:-1]:
        perm.extend(range(offset, offset + start))
        perm.extend(range(offset + start + count, offset + n))
        perm.extend(range(offset + start, offset + start + count))
        split_lengths.extend((n - count, count))
        full_lengths.append(n)
        b_mask.extend([False] * (n - count) + [True] * count)
        offset += n
    perm.extend(range(offset, token_capacity))
    b_mask.extend([False] * (token_capacity - offset))
    # Bound synthetic padding sequences individually: no bucket-sized quadratic
    # attention. Slot count stays constant when real token b_lengths change.
    padding_slots = (token_capacity + max_seq_len - 1) // max_seq_len
    padding = [
        min(max_seq_len, max(0, token_capacity - offset - i * max_seq_len))
        for i in range(padding_slots)
    ]

    def indptr(lens):
        ptr = [0]
        for n in lens:
            ptr.append(ptr[-1] + n)
        return ptr

    inv_perm = [0] * token_capacity
    for new, old in enumerate(perm):
        inv_perm[old] = new
    return (
        perm,
        inv_perm,
        indptr(split_lengths + padding),
        indptr(full_lengths + padding),
        b_mask,
        [True] * offset + [False] * (token_capacity - offset),
    )


class BertUqiAiterGraphLayout:
    """Reusable packed metadata for one graph bucket, without per-token loops.

    All six arrays share one byte buffer, allowing a single H2D copy. Native
    int64/int32 views match the PyTorch views of the device buffer. Only the
    request slots are visited in Python; token ranges are copied in bulk.
    """

    def __init__(self, token_capacity: int, request_capacity: int, max_seq_len: int):
        if (
            token_capacity <= 0
            or token_capacity > 2**31 - 1
            or request_capacity <= 0
            or max_seq_len <= 0
        ):
            raise ValueError("invalid UQI graph token_capacity")
        self.token_capacity = token_capacity
        self.request_capacity = request_capacity
        self.max_seq_len = max_seq_len
        self.padding_slots = (token_capacity + max_seq_len - 1) // max_seq_len
        self.split_indptr_size = 2 * request_capacity + self.padding_slots + 1
        self.full_indptr_size = request_capacity + self.padding_slots + 1
        self.inv_perm_offset = token_capacity * 8
        self.split_indptr_offset = self.inv_perm_offset + token_capacity * 8
        self.full_indptr_offset = self.split_indptr_offset + self.split_indptr_size * 4
        self.b_mask_offset = self.full_indptr_offset + self.full_indptr_size * 4
        self.valid_mask_offset = self.b_mask_offset + token_capacity
        self.buffer = bytearray(self.valid_mask_offset + token_capacity)
        view = memoryview(self.buffer)
        self.perm = view[: self.inv_perm_offset].cast("q")
        self.inv_perm = view[self.inv_perm_offset : self.split_indptr_offset].cast("q")
        self.split_indptr = view[
            self.split_indptr_offset : self.full_indptr_offset
        ].cast("i")
        self.full_indptr = view[self.full_indptr_offset : self.b_mask_offset].cast("i")
        self.b_mask = view[self.b_mask_offset : self.valid_mask_offset]
        self.valid_mask = view[self.valid_mask_offset :]
        self._identity = memoryview(array("q", range(token_capacity)))
        self._zeros = bytes(token_capacity)
        self._ones = b"\x01" * token_capacity

    def update(self, lengths, starts, b_lengths) -> None:
        if len(lengths) != self.request_capacity:
            raise ValueError("UQI graph request token_capacity changed")
        layout = build_uqi_graph_layout(lengths, starts, b_lengths, self.token_capacity)
        if any(n > self.max_seq_len for n in lengths):
            raise ValueError("UQI sequence exceeds AITER profile")

        self.perm[:] = self._identity
        self.inv_perm[:] = self._identity
        self.b_mask[:] = self._zeros
        self.split_indptr[0] = self.full_indptr[0] = 0
        offset = 0
        for i, (n, start, count) in enumerate(layout[:-1]):
            a_end = offset + n - count
            end = offset + n
            if count:
                b_start, b_end = offset + start, offset + start + count
                self.perm[b_start:a_end] = self._identity[b_end:end]
                self.perm[a_end:end] = self._identity[b_start:b_end]
                self.inv_perm[b_start:b_end] = self._identity[a_end:end]
                self.inv_perm[b_end:end] = self._identity[b_start:a_end]
                self.b_mask[a_end:end] = self._ones[:count]
            self.split_indptr[2 * i + 1] = a_end
            self.split_indptr[2 * i + 2] = end
            self.full_indptr[i + 1] = end
            offset = end
        self.valid_mask[:offset] = self._ones[:offset]
        self.valid_mask[offset:] = self._zeros[: self.token_capacity - offset]
        for i in range(self.padding_slots):
            end = min(self.token_capacity, offset + (i + 1) * self.max_seq_len)
            self.split_indptr[2 * self.request_capacity + i + 1] = end
            self.full_indptr[self.request_capacity + i + 1] = end


def derive_bert_uqi_bounds(
    tokens, lengths, text_mask=None, cls_token_id=2, sep_token_id=102
):
    """Find the first text CLS..SEP segment per request using host metadata."""
    if tokens is None or tokens.is_cuda or lengths.is_cuda:
        raise ValueError("BERT UQI metadata requires host token IDs and lengths")
    if (
        lengths.ndim != 1
        or bool((lengths < 0).any())
        or int(lengths.sum()) != tokens.numel()
    ):
        raise ValueError("BERT UQI token/length metadata mismatch")
    if text_mask is not None and text_mask.numel():
        if text_mask.is_cuda or text_mask.numel() != tokens.numel():
            raise ValueError("BERT UQI text mask must match host token IDs")
        tokens = tokens.masked_fill(~text_mask.bool(), -1)
    starts = torch.full_like(lengths, -1, dtype=torch.int32)
    b_lengths = torch.zeros_like(lengths, dtype=torch.int32)
    offset = 0
    for index, length in enumerate(lengths.tolist()):
        sequence = tokens[offset : offset + length]
        markers = (sequence == cls_token_id).nonzero().flatten()
        if markers.numel():
            start = int(markers[0])
            endings = (sequence[start:] == sep_token_id).nonzero().flatten()
            end = start + int(endings[0]) + 1 if endings.numel() else length
            starts[index], b_lengths[index] = start, end - start
        offset += length
    return starts, b_lengths


def valid_bert_uqi_bounds(attn_inputs):
    """Graph eligibility for BERT user segments, without planning or mutation."""
    lengths = attn_inputs.input_lengths
    starts, b_lengths = attn_inputs.uqi_b_starts, attn_inputs.uqi_b_lens
    for bounds in (starts, b_lengths):
        if (
            bounds is None
            or bounds.is_cuda
            or bounds.dtype != torch.int32
            or bounds.ndim != 1
            or not bounds.is_contiguous()
            or bounds.numel() != lengths.numel()
        ):
            return False
    if bool((b_lengths < 0).any()):
        return False
    active = b_lengths > 0
    return not bool(
        (
            active
            & ((starts < 0) | (starts > lengths) | (b_lengths > lengths - starts))
        ).any()
    )
