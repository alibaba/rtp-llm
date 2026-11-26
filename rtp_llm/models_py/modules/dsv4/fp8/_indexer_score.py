"""DSv4 Indexer FP8-paged score path via DeepGEMM.

Wraps ``deep_gemm.fp8_paged_mqa_logits`` so it can drop into the indexer
decode loop in place of the bf16 ``v4_indexer_score`` Triton kernel
when the cache is FP8 packed (132B/slot).

End-to-end shape contract:

  q_fp8       [B, next_n, H, D]            float8_e4m3fn  (per-(t,h) quant)
  w_fold      [B*next_n, H]                fp32           (per-token Q
                                                            scale folded in)
  kv_cache    [num_blocks, block_size, 1, D+4]  uint8     (132B per slot:
                                                            128 FP8 K + 4B fp32 scale)
  context_lens[B, next_n]                  int32          (live K length per row)
  block_table [B, max_blocks]              int32          (logical→physical block id)

Returns ``[B*next_n, max_ctx_len] fp32`` logits — same semantics as
``v4_indexer_score``: each row is the per-K-token score after fused
einsum + ReLU + per-head weighted sum.

Caller is responsible for FP8 quantizing Q via
:func:`indexer_q_fp8_quant_fold`, building the block_table, and the
2D context_lens shape DeepGEMM requires.
"""

from __future__ import annotations

import re as _re
from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
)


def get_fp8_paged_mqa_logits_metadata(backend, *args, **kwargs):
    """Pair legacy FP8 logits with its 256-token schedule when available."""
    metadata = getattr(backend, "get_legacy_paged_mqa_logits_metadata", None)
    if not callable(metadata):
        metadata = backend.get_paged_mqa_logits_metadata
    return metadata(*args, **kwargs)


def fp8_paged_mqa_uses_2d_context_lens(backend=None) -> bool:
    """Whether ``backend.fp8_paged_mqa_logits`` expects 2D ``[B, next_n]`` lengths.

    DeepGEMM >= 2.8 (such as CUDA 13's 2.8.2) requires contiguous 2D
    ``[B, next_n]`` context lengths. Older wheels such as CUDA 12.9's
    ``2.1.1+ee8d9c7`` index ``context_lens`` as 1D ``[B]`` per request;
    passing ``[B, next_n]`` when ``next_n > 1`` silently reads wrong
    request lengths when request lengths differ.
    """
    dg = _deep_gemm if backend is None else backend
    if dg is None:
        return True
    if callable(getattr(dg, "get_legacy_paged_mqa_logits_metadata", None)):
        return True
    if callable(getattr(dg, "fp8_fp4_paged_mqa_logits", None)):
        return True
    ver = getattr(dg, "__version__", "")
    match = _re.match(r"(\d+)\.(\d+)\.(\d+)", ver) if isinstance(ver, str) else None
    if match:
        return tuple(int(x) for x in match.groups()) >= (2, 8, 0)
    return False


def prepare_fp8_paged_mqa_inputs(
    q_fp8: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    device: Optional[torch.device | str | int] = None,
    backend=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapt paged MQA inputs for DeepGEMM's length rank and next_n limits.

    DeepGEMM's SM90 legacy paged MQA kernel only instantiates ``next_n``
    in ``{1, 2}``; pre-2.8 wheels (such as CUDA 12.9's ``2.1.1+ee8d9c7``)
    likewise only instantiate ``next_n`` in ``{1, 2}`` on SM90 and SM100.
    For wider verification batches (or unequal per-token lengths on 1D
    wheels), flatten queries to ``next_n=1`` with matching row lengths and
    expanded block-table rows.
    """
    q_contig = q_fp8.contiguous()
    batch, next_n, heads, head_dim = q_contig.shape
    rows = batch * next_n
    target_device = torch.device(q_contig.device if device is None else device)
    uses_2d = fp8_paged_mqa_uses_2d_context_lens(backend)
    major: Optional[int] = None
    if target_device.type == "cuda" and torch.cuda.is_available():
        with torch.cuda.device(target_device):
            major = torch.cuda.get_device_capability(target_device)[0]
    bt_expanded = (
        block_table.contiguous()
        if block_table.shape[0] == rows
        else block_table.repeat_interleave(next_n, dim=0).contiguous()
    )

    if uses_2d:
        if context_lens.dim() == 1:
            if context_lens.numel() == batch:
                ctx_2d = context_lens.view(batch, 1).expand(batch, next_n)
            elif context_lens.numel() == rows:
                ctx_2d = context_lens.view(batch, next_n)
            else:
                raise ValueError(
                    f"context_lens numel={context_lens.numel()} incompatible with "
                    f"batch={batch}, next_n={next_n}"
                )
        else:
            ctx_2d = context_lens
        sm90_gt2 = next_n > 2 and major == 9
        if sm90_gt2:
            return (
                q_contig.reshape(rows, 1, heads, head_dim),
                ctx_2d.reshape(rows, 1).contiguous(),
                bt_expanded,
            )
        return q_contig, ctx_2d.contiguous(), block_table.contiguous()

    # 1D [B] ABI (e.g. CUDA 12.9 DeepGEMM 2.1.1+ee8d9c7): SM90 and SM100
    # both instantiate only next_n in {1, 2}.
    pre28_gt2 = next_n > 2 and (major is None or major in (9, 10))
    if context_lens.dim() == 1:
        if next_n > 1 and context_lens.numel() == rows:
            return (
                q_contig.reshape(rows, 1, heads, head_dim),
                context_lens.contiguous(),
                bt_expanded,
            )
        if context_lens.numel() != batch:
            raise ValueError(
                f"context_lens numel={context_lens.numel()} incompatible with "
                f"batch={batch}, next_n={next_n}"
            )
        if pre28_gt2:
            return (
                q_contig.reshape(rows, 1, heads, head_dim),
                context_lens.repeat_interleave(next_n).contiguous(),
                bt_expanded,
            )
        return q_contig, context_lens.contiguous(), block_table.contiguous()

    if next_n == 1:
        return (
            q_contig,
            context_lens.reshape(batch).contiguous(),
            block_table.contiguous(),
        )
    return (
        q_contig.reshape(rows, 1, heads, head_dim),
        context_lens.reshape(rows).contiguous(),
        bt_expanded,
    )


try:
    if not torch.cuda.is_available():
        raise ImportError("no CUDA device available for deep_gemm")
    import deep_gemm as _deep_gemm

    # DeepGEMM 2.8 renamed the group-128 MQA logits entry points to the
    # fp8_fp4_* family and switched q to a (tensor, sf|None) tuple.
    _HAS_DEEP_GEMM = hasattr(_deep_gemm, "fp8_paged_mqa_logits") and hasattr(
        _deep_gemm, "get_paged_mqa_logits_metadata"
    )
    _HAS_DEEP_GEMM_MQA = hasattr(_deep_gemm, "fp8_mqa_logits")
    # Newer wheels restore the group-128 FP32-weight/output APIs alongside
    # the MX APIs, with an explicit legacy metadata entry point. Older
    # 2.8.0 wheels use the shared metadata name; 2.8.1 rejects these calls
    # on SM100 and keeps the existing Triton fallback.
    _match = _re.match(r"(\d+)\.(\d+)\.(\d+)", getattr(_deep_gemm, "__version__", ""))
    _DEEP_GEMM_SM100_LEGACY_OK = callable(
        getattr(_deep_gemm, "get_legacy_paged_mqa_logits_metadata", None)
    ) or (bool(_match) and tuple(int(x) for x in _match.groups()) < (2, 8, 1))
except ImportError:
    _deep_gemm = None
    _HAS_DEEP_GEMM = False
    _HAS_DEEP_GEMM_MQA = False
    _DEEP_GEMM_SM100_LEGACY_OK = False


def _is_cuda_runtime_device(device) -> bool:
    if device is None:
        return torch.cuda.is_available()
    target = torch.device(device)
    return target.type == "cuda" and torch.cuda.is_available()


def _legacy_mqa_abi_ok(backend=None) -> bool:
    dg = _deep_gemm if backend is None else backend
    if dg is None:
        return False
    if callable(getattr(dg, "get_legacy_paged_mqa_logits_metadata", None)):
        return True
    match = _re.match(r"(\d+)\.(\d+)\.(\d+)", getattr(dg, "__version__", ""))
    return bool(match) and tuple(int(x) for x in match.groups()) < (2, 8, 1)


def _sm100_needs_triton_fallback(device, backend=None) -> bool:
    """Whether this wheel rejects the group-128 legacy logits calls on SM100."""
    return torch.cuda.get_device_capability(device)[0] == 10 and not (
        _DEEP_GEMM_SM100_LEGACY_OK
        if backend is None
        else _legacy_mqa_abi_ok(backend)
    )


def deepgemm_supports_block_kv(device, block_kv: int, backend=None) -> bool:
    """DeepGEMM's legacy SM90/SM100 paged-MQA metadata contract."""
    if not _is_cuda_runtime_device(device):
        return False
    dg = _deep_gemm if backend is None else backend
    major = torch.cuda.get_device_capability(device)[0]
    if major not in (9, 10):
        return False
    if not fp8_paged_mqa_uses_2d_context_lens(dg):
        return block_kv == 64
    if block_kv == 64:
        return True
    if block_kv == 32:
        return major == 10
    if block_kv == 128:
        return major == 10 and callable(
            getattr(dg, "get_legacy_paged_mqa_logits_metadata", None)
        )
    return False


def supports_paged_mqa_logits(
    device, block_kv: Optional[int] = None, *, backend=None
) -> bool:
    """Whether legacy DeepGEMM SM90/SM100 paged MQA logits accepts ``device``."""
    if not _is_cuda_runtime_device(device):
        return False
    if backend is None:
        has_paged = _HAS_DEEP_GEMM
    else:
        has_paged = hasattr(backend, "fp8_paged_mqa_logits") and (
            hasattr(backend, "get_legacy_paged_mqa_logits_metadata")
            or hasattr(backend, "get_paged_mqa_logits_metadata")
        )
    if not has_paged or _sm100_needs_triton_fallback(device, backend=backend):
        return False
    if torch.cuda.get_device_capability(device)[0] not in (9, 10):
        return False
    return block_kv is None or deepgemm_supports_block_kv(
        device, block_kv, backend=backend
    )


def supports_mqa_logits(device, *, backend=None) -> bool:
    """Whether legacy DeepGEMM SM90/SM100 MQA logits accepts ``device``."""
    if not _is_cuda_runtime_device(device):
        return False
    if torch.cuda.get_device_capability(device)[0] not in (9, 10):
        return False
    has_mqa = (
        _HAS_DEEP_GEMM_MQA
        if backend is None
        else hasattr(backend, "fp8_mqa_logits")
    )
    return has_mqa and not _sm100_needs_triton_fallback(device, backend=backend)


def has_fp8_paged_mqa_logits() -> bool:
    """Kernel availability — not a device/path decision (always usable).

    The group-128 paged kernel is always servable: DeepGEMM when the wheel
    accepts the call, the Triton fallback otherwise. Ask
    :func:`supports_paged_mqa_logits` which path actually runs.
    """
    return True


def has_fp8_mqa_logits() -> bool:
    """Kernel availability — not a device/path decision; see above."""
    return True


_num_sms_cache: dict[int, int] = {}


def _get_num_sms(device: torch.device) -> int:
    # The schedule must cover the SM count DeepGEMM launches with, which
    # deep_gemm.set_num_sms() can lower below the physical count.
    getter = getattr(_deep_gemm, "get_num_sms", None)
    if callable(getter):
        with torch.cuda.device(device):
            return int(getter())
    dev_idx = (
        torch.cuda.current_device()
        if torch.device(device).index is None
        else int(torch.device(device).index)
    )
    cached = _num_sms_cache.get(dev_idx)
    if cached is None:
        cached = torch.cuda.get_device_properties(dev_idx).multi_processor_count
        _num_sms_cache[dev_idx] = cached
    return cached


def fp8_paged_indexer_score(
    q_fp8: torch.Tensor,  # [B, next_n, H, D] float8_e4m3fn
    w_fold: torch.Tensor,  # [B*next_n, H]    fp32
    kv_pool_uint8: torch.Tensor,  # [total_slots, 132] uint8 — flat pool view
    block_table: torch.Tensor,  # [B, max_blocks] int32 — logical→physical
    context_lens: torch.Tensor,  # [B, next_n] int32 — live K length per row
    block_size: int,  # tokens per cache block
    max_ctx_len: int,  # output T dim
) -> torch.Tensor:
    """One-shot FP8 paged indexer logits via DeepGEMM.

    Returns ``[B*next_n, max_ctx_len] fp32`` — feed straight to topk.
    Padded columns past per-row ``context_lens[b, n]`` are left as
    whatever DeepGEMM writes (use ``clean_logits=True`` if the
    downstream topk needs ``-inf`` there; default False to save the
    extra mask).
    """
    assert q_fp8.dtype == torch.float8_e4m3fn, f"q_fp8 dtype={q_fp8.dtype}"
    assert q_fp8.dim() == 4 and q_fp8.shape[-1] == INDEXER_HEAD_DIM
    assert w_fold.dtype == torch.float32 and w_fold.dim() == 2
    assert kv_pool_uint8.dtype == torch.uint8
    assert kv_pool_uint8.shape[-1] == INDEXER_ENTRY_BYTES
    assert block_table.dtype == torch.int32 and block_table.dim() == 2
    assert context_lens.dtype == torch.int32 and context_lens.dim() == 2
    # DeepGEMM kv_cache shape: [num_blocks, block_size, 1, D+4] uint8.
    # Our pool is a flat [total_slots, 132] view; reshape into the 4D
    # layout (no copy — just a metadata change).
    total_slots = kv_pool_uint8.shape[0]
    assert (
        total_slots % block_size == 0
    ), f"total_slots={total_slots} not divisible by block_size={block_size}"
    num_blocks = total_slots // block_size
    kv_4d = kv_pool_uint8.view(num_blocks, block_size, 1, INDEXER_ENTRY_BYTES)

    if not supports_paged_mqa_logits(q_fp8.device, block_size):
        from rtp_llm.models_py.triton_kernels.legacy_fp8_indexer_score import (
            legacy_fp8_paged_mqa_logits,
        )

        return legacy_fp8_paged_mqa_logits(
            q_fp8, kv_4d, w_fold, context_lens, block_table, max_ctx_len
        )
    with torch.cuda.device(q_fp8.device):
        num_sms = _get_num_sms(q_fp8.device)
        q_in, ctx_in, bt_in = prepare_fp8_paged_mqa_inputs(
            q_fp8, context_lens, block_table, device=q_fp8.device, backend=_deep_gemm
        )
        # Legacy SM90/SM100 schedulers size work from the row's final length;
        # monotonic cummax avoids under-scheduling decreasing MTP rows while
        # passing the original ctx_in to logits preserves masking.
        ctx_sched = (
            ctx_in.cummax(dim=1).values.contiguous()
            if (ctx_in.dim() == 2 and ctx_in.shape[1] > 1)
            else ctx_in
        )
        schedule = get_fp8_paged_mqa_logits_metadata(
            _deep_gemm, ctx_sched, block_size, num_sms
        )
        return _deep_gemm.fp8_paged_mqa_logits(
            q_in,
            kv_4d,
            w_fold.contiguous(),
            ctx_in,
            bt_in,
            schedule,
            max_ctx_len,
        )


# ---------------------------------------------------------------------------
# Prefill (non-paged) wrapper around ``deep_gemm.fp8_mqa_logits``.
#
# Shape contract:
#   q_fp8        [M, H, D]    float8_e4m3fn  (M = total query tokens)
#   w_fold       [M, H]       fp32           (per-(token, head) Q scale folded in)
#   k_quant      [N, D]       float8_e4m3fn  (N = total key tokens — gathered
#                                              contiguous from the FP8 cache)
#   k_scale      [N]          float32
#   cu_seqlen_ks [M]          int32          (K start, inclusive)
#   cu_seqlen_ke [M]          int32          (K end,   exclusive)
#
# Returns ``[M, N] fp32`` logits — same semantics as ``v4_indexer_score``
# but laid out flat over total query tokens (the indexer prefill caller
# reshapes back to ``[B, S, T]``).
# ---------------------------------------------------------------------------


def _mask_relative_mqa_logits(
    out: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    num_keys: int,
) -> torch.Tensor:
    """Mask out-of-interval and out-of-bounds columns for relative MQA logits."""
    max_seqlen_k = out.shape[1]
    if max_seqlen_k == 0:
        return out
    starts = cu_seqlen_ks.to(device=out.device, dtype=torch.int64).unsqueeze(1)
    ends = cu_seqlen_ke.to(device=out.device, dtype=torch.int64).unsqueeze(1)
    rel = torch.arange(max_seqlen_k, device=out.device, dtype=torch.int64).unsqueeze(0)
    abs_pos = starts + rel
    valid = (
        (abs_pos >= starts)
        & (abs_pos < ends)
        & (abs_pos >= 0)
        & (abs_pos < num_keys)
    )
    out.masked_fill_(~valid, float("-inf"))
    return out


def _slice_relative_mqa_logits(
    out: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    *,
    clean_logits: bool,
    max_seqlen_k: int,
) -> torch.Tensor:
    if max_seqlen_k < 0:
        raise ValueError(f"max_seqlen_k must be non-negative, got {max_seqlen_k}")
    rows, cols = out.shape
    starts = cu_seqlen_ks.to(device=out.device, dtype=torch.long).unsqueeze(1)
    ends = cu_seqlen_ke.to(device=out.device, dtype=torch.long).unsqueeze(1)
    if max_seqlen_k == 0:
        if clean_logits:
            positions = torch.arange(cols, device=out.device).unsqueeze(0)
            valid = (positions >= starts) & (positions < ends)
            out.masked_fill_(~valid, float("-inf"))
        return out
    if cols == 0:
        return torch.full(
            (rows, max_seqlen_k),
            float("-inf"),
            dtype=out.dtype,
            device=out.device,
        )
    rel = torch.arange(max_seqlen_k, device=out.device).unsqueeze(0)
    abs_pos = starts + rel
    valid = (abs_pos >= starts) & (abs_pos < ends) & (abs_pos >= 0) & (abs_pos < cols)
    gathered = out.gather(1, abs_pos.clamp(0, cols - 1))
    gathered.masked_fill_(~valid, float("-inf"))
    return gathered


def fp8_mqa_indexer_score(
    q_fp8: torch.Tensor,  # [M, H, D] float8_e4m3fn
    w_fold: torch.Tensor,  # [M, H]    fp32
    k_quant: torch.Tensor,  # [N, D]    float8_e4m3fn
    k_scale: torch.Tensor,  # [N]       float32
    cu_seqlen_ks: torch.Tensor,  # [M]       int32
    cu_seqlen_ke: torch.Tensor,  # [M]       int32
    *,
    clean_logits: bool = False,
    max_seqlen_k: int = 0,
) -> torch.Tensor:
    """One-shot non-paged FP8 indexer logits via DeepGEMM.

    Returns ``[M, N] fp32`` (M = total Q tokens this chunk; N = total K
    tokens in the gathered workspace). Caller reshapes back to ``[B, S, T]``.

    ``clean_logits=False`` matches what we want — entries past
    ``cu_seqlen_ke[m]`` are left untouched; the topk-with-causal-mask path
    in :class:`Indexer.forward` re-applies its own ``q_pos`` causal cap.
    """
    assert q_fp8.dtype == torch.float8_e4m3fn and q_fp8.dim() == 3
    assert q_fp8.shape[-1] == INDEXER_HEAD_DIM
    assert w_fold.dtype == torch.float32 and w_fold.dim() == 2
    assert w_fold.shape[0] == q_fp8.shape[0]
    assert k_quant.dtype == torch.float8_e4m3fn and k_quant.dim() == 2
    assert k_quant.shape[-1] == INDEXER_HEAD_DIM
    assert k_scale.dtype == torch.float32 and k_scale.dim() == 1
    assert k_scale.shape[0] == k_quant.shape[0]
    assert cu_seqlen_ks.dtype == torch.int32 and cu_seqlen_ks.dim() == 1
    assert cu_seqlen_ke.dtype == torch.int32 and cu_seqlen_ke.dim() == 1
    assert cu_seqlen_ks.shape[0] == q_fp8.shape[0]
    assert cu_seqlen_ke.shape[0] == q_fp8.shape[0]

    if not supports_mqa_logits(q_fp8.device):
        from rtp_llm.models_py.triton_kernels.legacy_fp8_indexer_score import (
            legacy_fp8_mqa_logits,
        )

        return legacy_fp8_mqa_logits(
            q_fp8,
            (k_quant, k_scale),
            w_fold,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=clean_logits,
            max_seqlen_k=max_seqlen_k,
        )

    with torch.cuda.device(q_fp8.device):
        if fp8_paged_mqa_uses_2d_context_lens(_deep_gemm):
            return _deep_gemm.fp8_mqa_logits(
                q_fp8.contiguous(),
                (k_quant.contiguous(), k_scale.contiguous()),
                w_fold.contiguous(),
                cu_seqlen_ks.contiguous(),
                cu_seqlen_ke.contiguous(),
                clean_logits,
                max_seqlen_k,
            )
        raw = _deep_gemm.fp8_mqa_logits(
            q_fp8.contiguous(),
            (k_quant.contiguous(), k_scale.contiguous()),
            w_fold.contiguous(),
            cu_seqlen_ks.contiguous(),
            cu_seqlen_ke.contiguous(),
            bool(clean_logits) if max_seqlen_k == 0 else False,
        )
        return _slice_relative_mqa_logits(
            raw,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=clean_logits,
            max_seqlen_k=max_seqlen_k,
        )
