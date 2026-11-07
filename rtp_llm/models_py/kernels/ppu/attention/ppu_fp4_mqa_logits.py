"""PPU Prefill FP4 scores with native BF16 product and head reduction.

The 64 MiB bound applies to dot/product scratch; caller chunking bounds logits.
"""
import torch
from rtp_llm.models_py.kernels.ppu.moe.ppu_mxfp4 import gemm_fp4_fp4_bf16_nt, prepare_fp4_weight_scale_mxfp4
DOT_BYTE_LIMIT = 64 * 1024 ** 2

def _rows(rows, heads, keys, scratch_bytes):
    if not 0 < scratch_bytes <= DOT_BYTE_LIMIT:
        raise ValueError('scratch_bytes must be positive and at most64MiB')
    cap = scratch_bytes // (2 * heads * keys)
    if cap < 1:
        raise ValueError('One full-head/full-K query exceeds bounded FP4 score scratch')
    return min(rows, cap)

def _prepared(packed, scale):
    if packed.dtype != torch.int8 or scale.dtype != torch.int32 or packed.shape[-1] != 64:
        raise ValueError('FP4 score requires original int8 packed K128 and int32 UE8M0 scales')
    if not packed.is_contiguous() or not scale.is_contiguous() or tuple(scale.shape) != tuple(packed.shape[:-1]) + (1,):
        raise ValueError('FP4 score payload/scale layout mismatch')
    rows = packed.numel() // 64
    data = packed.view(torch.uint8).reshape(rows, 64)
    raw = scale.view(torch.uint8).reshape(rows, 4)
    return (data, prepare_fp4_weight_scale_mxfp4(raw))

def fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, clean_logits=False, max_seqlen_k=0, logits_dtype=torch.float32, *, scratch_bytes=DOT_BYTE_LIMIT):
    """Compute nonpaged BF16 MQA scores from the original FP4 payload.

    Invalid columns are initialized -inf, a stronger defined contract than old
    clean_logits=False. Callers compare only valid columns against old DG output.
    """
    if logits_dtype != torch.bfloat16 or max_seqlen_k not in (None, 0):
        raise ValueError('Candidate qualified only for BF16 uncompressed/full-width score output')
    (packed, scale) = q
    (kpacked, kscale) = kv
    if packed.ndim != 3 or kpacked.ndim != 2 or weights.shape != packed.shape[:2]:
        raise ValueError('FP4 score Q/K/weight shape mismatch')
    (rows, heads, _) = packed.shape
    keys = kpacked.shape[0]
    if heads not in (32, 64) or keys <= 0 or weights.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('Unsupported FP4 score shape/dtype')
    if any((v.device != packed.device for v in [scale, kpacked, kscale, weights, cu_seq_len_k_start, cu_seq_len_k_end])):
        raise ValueError('FP4 score device mismatch')
    if any((v.dtype != torch.int32 or tuple(v.shape) != (rows,) for v in [cu_seq_len_k_start, cu_seq_len_k_end])):
        raise ValueError('FP4 score per-query bounds must be int32[rows]')
    result = torch.empty((rows, keys), dtype=torch.bfloat16, device=packed.device)
    if rows == 0:
        return result
    # The installed no-bias FP4 epilogue requires even physical N.
    physical_keys = keys + keys % 2
    if physical_keys != keys:
        kpacked = torch.cat((kpacked, torch.zeros_like(kpacked[:1])), dim=0)
        kscale = torch.cat((kscale, kscale[:1]), dim=0)
    step = _rows(rows, heads, physical_keys, scratch_bytes)
    prepared_k = _prepared(kpacked, kscale)
    columns = torch.arange(keys, device=packed.device)
    for start in range(0, rows, step):
        end = min(rows, start + step)
        prepared_q = _prepared(packed[start:end], scale[start:end])
        dot = gemm_fp4_fp4_bf16_nt(prepared_q, prepared_k).view(end - start, heads, physical_keys)
        # Keep the BF16 product before Torch accumulates the head sum.
        dot.relu_()
        dot.mul_(weights[start:end].to(torch.bfloat16).unsqueeze(-1))
        result[start:end].copy_(dot.sum(dim=1)[:, :keys])
        del dot, prepared_q
        valid = (columns[None, :] >= cu_seq_len_k_start[start:end, None]) & (columns[None, :] < cu_seq_len_k_end[start:end, None])
        result[start:end].masked_fill_(~valid, float('-inf'))
    return result

def native_topk(scores, starts, ends, out=None, *, topk=512):
    """Native Torch topk order, then bounds masking and caller-width -1 padding."""
    if scores.ndim != 2 or scores.dtype != torch.bfloat16 or topk <= 0:
        raise ValueError('Native Prefill topk requires BF16[M,T] and positive K')
    (rows, keys) = scores.shape
    if out is None:
        out = torch.empty((rows, topk), device=scores.device, dtype=torch.int32)
    if out.dtype != torch.int32 or tuple(out.shape) != (rows, topk) or out.device != scores.device:
        raise ValueError('Native topk output must be int32[M,K] on score device')
    if any((v.dtype != torch.int32 or tuple(v.shape) != (rows,) or v.device != scores.device for v in [starts, ends])):
        raise ValueError('Native topk bounds must be int32[M] on score device')
    count = min(topk, keys)
    out.fill_(-1)
    if not count or not rows:
        return out
    # Preserve native TopK tie order and its [1, rows, keys] call shape.
    selected = scores.unsqueeze(0).topk(count, dim=-1).indices[0]
    valid = (selected >= starts[:, None]) & (selected < ends[:, None])
    out[:, :count].copy_(selected.masked_fill(~valid, -1))
    return out
