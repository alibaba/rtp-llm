"""Fused kernels for the whisper decoder (Phase C lane contract).

Two device-side replacements for the eager whisper decoder path; both keep
the reference math (see ``model_desc/whisper.py`` for the Phase A contract):

1. ``fused_residual_clamp`` -- one Triton kernel replacing the ATen
   ``residual + x`` add plus the ``torch.nan_to_num`` fp16 guard pair. The
   add stays an fp16 add (same rounding as the eager op: the Triton sum is
   cast back to the pointer dtype before the guard) and the guard
   reproduces ``nan_to_num`` semantics exactly: nan -> +clamp, +inf ->
   +clamp, -inf -> -clamp, finite values untouched. The replacement value
   is applied in fp32 and rounded back on store, which lands on the same
   fp16 value the reference produces when it casts ``finfo.max - 1000``
   (64504 -> 64512) into the fp16 output.

2. ``WhisperCrossAttnRaggedRunner`` -- FlashInfer ragged-prefill attention
   for ``WhisperCrossAttention`` (non-causal, per-row K/V). The pinned
   flashinfer 0.6.9+8c4f4dcf ``BatchPrefillWithRaggedKVCacheWrapper`` is
   itself the direct (non-paged) k/v-tensor path, so no paged plumbing is
   needed. One wrapper is shared by all decoder layers: the model plans
   once per forward (qo/kv indptr from the row structure, which is
   layer-invariant) and each layer runs (q, k, v) against that plan --
   the same plan/run split ``PyFlashinferPrefillAttnOp`` uses. The softmax
   scale reaches the kernel through ``plan(sm_scale=...)`` instead of the
   eager ``q * scaling`` elementwise mul: the scale is applied in fp32 on
   the qk accumulator, which is strictly more accurate than pre-rounding
   the scaled q in fp16 and stays well inside the fp16 parity band the
   greedy decoder test asserts.

Both paths fall back to the original eager math whenever CUDA, Triton or
FlashInfer are unavailable, so CPU runs and non-fp16 dtypes keep the exact
reference behavior.

Ragged K/V layout: rows whose K/V share one contiguous storage and tile it
in order from offset 0 (the prefill ``fuse_row_kv`` views) are fed to
FlashInfer zero-copy; any other row mix is concatenated once per row
composition and cached per layer, so steady decode batches do not re-copy
their 1500 x 1280 cross K/V every step.
"""

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# Kill switches for remote triage; both default to enabled.
_FUSED_RESIDUAL_ENABLED = os.environ.get("WHISPER_FUSED_RESIDUAL_CLAMP", "1") != "0"
_CROSS_ATTN_FLASHINFER_ENABLED = os.environ.get("WHISPER_CROSS_ATTN_FLASHINFER", "1") != "0"

try:  # triton is a GPU-only dependency; keep module importable on CPU hosts
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# 1. fused residual add + fp16 clamp (replaces add + nan_to_num)
# ---------------------------------------------------------------------------
_RESIDUAL_BLOCK = 1024


if _HAS_TRITON:

    @triton.jit
    def _fused_residual_clamp_kernel(
        residual_ptr,
        x_ptr,
        out_ptr,
        n_rows,
        hidden,
        clamp,
        BLOCK: tl.constexpr,
    ):
        # Grid: (n_rows, cdiv(hidden, BLOCK)). Row index rides grid axis 0
        # (2^31 limit) and is widened to int64 before the row-stride
        # multiply, mirroring the other models_py Triton kernels.
        row = tl.program_id(0).to(tl.int64)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        col_mask = col < hidden
        offsets = row * hidden + col
        residual = tl.load(residual_ptr + offsets, mask=col_mask, other=0)
        x = tl.load(x_ptr + offsets, mask=col_mask, other=0)
        # fp16 add, rounded exactly like the eager ATen add: force the sum
        # back to the pointer dtype so the kernel is bit-exact whether
        # Triton emits an fp16 add or a promoted fp32 add.
        acc = (residual + x).to(residual_ptr.dtype.element_ty)
        acc_f32 = acc.to(tl.float32)
        # torch.nan_to_num(out, nan=clamp, posinf=clamp, neginf=-clamp):
        # finite values pass through untouched (the reference clamps only
        # inf/nan; whisper activations never approach finfo.max, so the
        # exact mapping is used rather than an unconditional clamp).
        is_nan = acc_f32 != acc_f32
        is_pos_inf = acc_f32 == float("inf")
        is_neg_inf = acc_f32 == float("-inf")
        out_f32 = acc_f32
        out_f32 = tl.where(is_pos_inf, clamp, out_f32)
        out_f32 = tl.where(is_neg_inf, -clamp, out_f32)
        out_f32 = tl.where(is_nan, clamp, out_f32)
        tl.store(out_ptr + offsets, out_f32.to(out_ptr.dtype.element_ty), mask=col_mask)


def _fused_residual_clamp_supported(residual: torch.Tensor, x: torch.Tensor) -> bool:
    if not _FUSED_RESIDUAL_ENABLED or not _HAS_TRITON:
        return False
    if residual.dtype != torch.float16 or x.dtype != torch.float16:
        return False  # the reference clamps fp16 only
    if not residual.is_cuda or not x.is_cuda:
        return False
    return residual.shape == x.shape and residual.is_contiguous() and x.is_contiguous()


def fused_residual_clamp(residual: torch.Tensor, x: torch.Tensor, clamp: Optional[float] = None) -> torch.Tensor:
    """``nan_to_num``-guarded fp16 ``residual + x`` in one Triton kernel.

    Bit-exact replacement for the eager pair ``residual + x`` followed by
    ``torch.nan_to_num`` with ``nan/posinf = finfo.max - 1000`` (fp16 only,
    like the transformers modeling_whisper reference). Non-fp16, non-CUDA or
    non-contiguous inputs take the original eager path.
    """
    if not _fused_residual_clamp_supported(residual, x):
        out = residual + x
        if out.dtype == torch.float16:
            clamp_value = torch.finfo(out.dtype).max - 1000
            out = torch.nan_to_num(out, nan=clamp_value, posinf=clamp_value, neginf=-clamp_value)
        return out
    if clamp is None:
        clamp = float(torch.finfo(residual.dtype).max) - 1000.0
    out = torch.empty_like(residual)
    hidden = residual.shape[-1]
    n_rows = residual.numel() // hidden if hidden else 0
    if n_rows == 0 or hidden == 0:
        return out
    grid = (n_rows, triton.cdiv(hidden, _RESIDUAL_BLOCK))
    _fused_residual_clamp_kernel[grid](
        residual,
        x,
        out,
        n_rows,
        hidden,
        clamp,
        BLOCK=_RESIDUAL_BLOCK,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# 2. ragged cross-attention (FlashInfer, non-causal, per-row K/V)
# ---------------------------------------------------------------------------
class _RaggedKVEntry:
    """Single-entry per-layer cache of the concatenated ragged K/V."""

    __slots__ = ("key", "k", "v", "refs")

    def __init__(self, key, k: torch.Tensor, v: torch.Tensor, refs: List[torch.Tensor]):
        self.key = key
        self.k = k
        self.v = v
        # Strong references to the source K/V keep the cache key's data
        # pointers from being recycled by a new allocation (the pool may
        # evict the slot that otherwise owns them).
        self.refs = refs


def _ragged_from_shared_views(views: List[torch.Tensor]) -> Optional[torch.Tensor]:
    """Zero-copy ragged K/V when the views tile one storage in row order.

    Returns ``None`` when the views do not share a storage / do not start
    at storage offset 0 / are not contiguous -- the caller then falls back
    to ``torch.cat``.
    """
    first = views[0]
    if not first.is_contiguous() or first.dim() != 2:
        return None
    width = first.shape[1]
    base = first.untyped_storage().data_ptr()
    elem = first.element_size()
    offset_rows = 0
    for view in views:
        if (
            view.dim() != 2
            or view.dtype != first.dtype
            or view.shape[1] != width
            or not view.is_contiguous()
            or view.untyped_storage().data_ptr() != base
            or view.data_ptr() != base + offset_rows * width * elem
        ):
            return None
        offset_rows += view.shape[0]
    return torch.as_strided(first, (offset_rows, width), (width, 1), 0)


def fuse_row_kv(
    pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Concatenate per-row (k, v) into ragged tensors and return row views.

    A single row is passed through unchanged (no copy). Multi-row inputs
    are concatenated once; the returned per-row views alias the ragged
    tensors, so the cross-KV pool stores views into one contiguous buffer
    per layer and later same-batch decode steps read it zero-copy.
    """
    if not pairs:
        return [], None, None
    if len(pairs) == 1:
        k, v = pairs[0]
        return [(k, v)], k, v
    k_ragged = torch.cat([pair[0] for pair in pairs], dim=0)
    v_ragged = torch.cat([pair[1] for pair in pairs], dim=0)
    views: List[Tuple[torch.Tensor, torch.Tensor]] = []
    offset = 0
    for k, v in pairs:
        length = k.shape[0]
        views.append((k_ragged[offset : offset + length], v_ragged[offset : offset + length]))
        offset += length
    return views, k_ragged, v_ragged


class WhisperCrossAttnRaggedRunner:
    """Shared FlashInfer ragged-prefill runner for whisper cross attention.

    The model calls :meth:`begin_forward` once per forward with the
    layer-invariant row structure (qo/kv indptr), then each layer's
    ``WhisperCrossAttention`` calls :meth:`attention` with its own rows and
    the freshly projected q. ``sm_scale`` carries the ``head_dim ** -0.5``
    softmax scale, so callers must NOT pre-scale q.
    """

    def __init__(self, head_num: int, head_dim: int, backend: str = "auto") -> None:
        self.head_num = int(head_num)
        self.head_dim = int(head_dim)
        self.hidden = self.head_num * self.head_dim
        self.sm_scale = float(self.head_dim) ** -0.5
        self._backend = backend
        # Set by begin_forward(); False falls the layer back to eager math.
        self.active = False
        self._wrapper = None
        self._workspace_owned = False
        self._device: Optional[torch.device] = None
        self._disabled = False
        self._warned = False
        # Row structure captured by the last begin_forward().
        self._all_slices: List[slice] = []
        self._kv_mask: List[bool] = []
        self._kv_row_slices: List[slice] = []
        self._kv_lens: List[int] = []
        self._qo_indptr: Optional[torch.Tensor] = None
        self._kv_indptr: Optional[torch.Tensor] = None
        self._all_kv = False
        self._q_dtype: Optional[torch.dtype] = None
        # Per-layer single-entry ragged K/V cache.
        self._layer_kv: Dict[int, _RaggedKVEntry] = {}

    # -- lifecycle ---------------------------------------------------------
    def _release_workspace(self) -> None:
        if not self._workspace_owned or self._wrapper is None:
            return
        try:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
                release_py_flashinfer_workspace_buffer,
            )

            release_py_flashinfer_workspace_buffer(self._wrapper._float_workspace_buffer)
        except Exception:
            pass
        self._workspace_owned = False

    def __del__(self):
        try:
            self._release_workspace()
        except Exception:
            pass

    def _get_wrapper(self, device: torch.device):
        if self._wrapper is not None and self._device == device:
            return self._wrapper
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            get_py_flashinfer_workspace_buffer,
        )

        if self._device != device:
            # Cached ragged tensors live on the old device; drop them.
            self._layer_kv.clear()
        self._release_workspace()
        workspace = get_py_flashinfer_workspace_buffer(device)
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD", backend=self._backend)
        self._workspace_owned = True
        self._wrapper = wrapper
        self._device = device
        return wrapper

    # -- per-forward plan ----------------------------------------------------
    def begin_forward(self, rows: Sequence[Tuple[slice, Optional[Tuple[torch.Tensor, torch.Tensor]]]],
                      hidden_states: torch.Tensor) -> None:
        """Plan the ragged wrapper from the row structure shared by all layers.

        Cheap no-op when the ragged path does not apply (env kill switch,
        non-CUDA device, no K/V rows, unsupported shapes/dtypes); the
        decoder layers then keep the eager reference math.
        """
        self.active = False
        if self._disabled or not _CROSS_ATTN_FLASHINFER_ENABLED:
            return
        device = hidden_states.device
        if device.type != "cuda" or hidden_states.dim() != 2:
            return
        if hidden_states.shape[-1] != self.hidden:
            return
        kv_rows = [(row_slice, kv) for row_slice, kv in rows if kv is not None]
        if not kv_rows:
            return
        kv_dtype = kv_rows[0][1][0].dtype
        for _, (k, v) in kv_rows:
            if (
                k.dim() != 2
                or v.dim() != 2
                or k.dtype != kv_dtype
                or v.dtype != kv_dtype
                or k.dtype != hidden_states.dtype
                or k.shape[1] != self.hidden
                or v.shape[1] != self.hidden
                or k.device != device
                or v.device != device
            ):
                return
        # Host int32 indptrs: plan() syncs on device-resident indptrs
        # (``indptr.to("cpu")``), host tensors avoid that D2H copy.
        qo = [0]
        kv_indptr = [0]
        for row_slice, _ in kv_rows:
            qo.append(qo[-1] + (row_slice.stop - row_slice.start))
        for _, (k, _) in kv_rows:
            kv_indptr.append(kv_indptr[-1] + k.shape[0])
        qo_indptr = torch.tensor(qo, dtype=torch.int32)
        kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
        try:
            wrapper = self._get_wrapper(device)
            wrapper.plan(
                qo_indptr,
                kv_indptr,
                self.head_num,
                self.head_num,
                self.head_dim,
                head_dim_vo=self.head_dim,
                causal=False,
                pos_encoding_mode="NONE",
                use_fp16_qk_reduction=False,
                sm_scale=self.sm_scale,
                q_data_type=hidden_states.dtype,
                kv_data_type=kv_dtype,
                o_data_type=hidden_states.dtype,
            )
        except Exception as exc:
            self._disabled = True
            if not self._warned:
                logger.warning(
                    "whisper cross-attention FlashInfer ragged plan failed (%s); "
                    "falling back to eager cross attention", exc,
                )
                self._warned = True
            return
        self._all_slices = [row_slice for row_slice, _ in rows]
        self._kv_mask = [kv is not None for _, kv in rows]
        self._kv_row_slices = [row_slice for row_slice, _ in kv_rows]
        self._kv_lens = [kv[0].shape[0] for _, kv in kv_rows]
        self._qo_indptr = qo_indptr
        self._kv_indptr = kv_indptr
        self._all_kv = len(kv_rows) == len(rows)
        self._q_dtype = hidden_states.dtype
        self.active = True

    # -- per-layer attention ---------------------------------------------------
    def attention(self, layer_idx: int, rows: Sequence[Tuple[slice, Optional[Tuple[torch.Tensor, torch.Tensor]]]],
                  q: torch.Tensor) -> Optional[torch.Tensor]:
        """Run this layer's cross attention against the shared plan.

        ``q`` is the unscaled ``q_proj(hidden_states)`` output ``[T, D]``
        (the softmax scale lives in the kernel's ``sm_scale``). Returns the
        ragged attention output ``[sum_kv_row_tokens, D]`` for the K/V rows
        in batch order, or ``None`` when this layer's rows no longer match
        the planned structure (caller falls back to eager math).
        """
        if not self.active:
            return None
        kv_rows = [(row_slice, kv) for row_slice, kv in rows if kv is not None]
        if len(kv_rows) != len(self._kv_row_slices) or q.dtype != self._q_dtype:
            return None
        for (row_slice, kv), planned_slice, planned_len in zip(kv_rows, self._kv_row_slices, self._kv_lens):
            if row_slice != planned_slice or kv[0].shape[0] != planned_len:
                return None
        k_ragged, v_ragged = self._layer_ragged(layer_idx, kv_rows)
        if self._all_kv:
            q_ragged = q.reshape(-1, self.head_num, self.head_dim)
        else:
            q_ragged = torch.cat([q[row_slice] for row_slice, _ in kv_rows], dim=0)
            q_ragged = q_ragged.reshape(-1, self.head_num, self.head_dim)
        # FlashInfer NHD contract: [nnz, num_heads, head_dim]; the row K/V are
        # [L, num_heads * head_dim] contiguous, so these are views.
        out = self._wrapper.run(
            q_ragged,
            k_ragged.reshape(-1, self.head_num, self.head_dim),
            v_ragged.reshape(-1, self.head_num, self.head_dim),
        )
        return out.reshape(out.shape[0], -1)

    def _layer_ragged(self, layer_idx: int, kv_rows) -> Tuple[torch.Tensor, torch.Tensor]:
        ks = [kv[1][0] for kv in kv_rows]
        vs = [kv[1][1] for kv in kv_rows]
        key = tuple((k.data_ptr(), v.data_ptr(), k.shape[0]) for k, v in zip(ks, vs))
        entry = self._layer_kv.get(layer_idx)
        if entry is not None and entry.key == key:
            return entry.k, entry.v
        if len(ks) == 1:
            k_ragged = ks[0] if ks[0].is_contiguous() else ks[0].contiguous()
            v_ragged = vs[0] if vs[0].is_contiguous() else vs[0].contiguous()
            entry = _RaggedKVEntry(key, k_ragged, v_ragged, list(ks) + list(vs))
        else:
            k_ragged = _ragged_from_shared_views(ks)
            v_ragged = _ragged_from_shared_views(vs)
            if k_ragged is None or v_ragged is None:
                k_ragged = torch.cat(ks, dim=0)
                v_ragged = torch.cat(vs, dim=0)
                entry = _RaggedKVEntry(key, k_ragged, v_ragged, list(ks) + list(vs))
            else:
                # Zero-copy: the as_strided results already keep the shared
                # storage alive; no extra refs are needed.
                entry = _RaggedKVEntry(key, k_ragged, v_ragged, [])
        self._layer_kv[layer_idx] = entry
        return entry.k, entry.v

    def assemble_output(self, o_ragged: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Spread the ragged o_proj output back to the full batch shape.

        Rows without cross K/V (text-only requests) come out as zeros,
        exactly like the eager path. When every row has K/V the ragged
        output already is the batch-ordered result and is returned as-is
        (no copy, no ``torch.cat``).
        """
        if self._all_kv:
            return o_ragged
        width = like.shape[-1]
        pieces: List[torch.Tensor] = []
        position = 0
        for row_slice, has_kv in zip(self._all_slices, self._kv_mask):
            length = row_slice.stop - row_slice.start
            if has_kv:
                pieces.append(o_ragged[position : position + length])
                position += length
            else:
                pieces.append(o_ragged.new_zeros((length, width)))
        if len(pieces) == 1:
            return pieces[0]
        return torch.cat(pieces, dim=0)


__all__ = [
    "WhisperCrossAttnRaggedRunner",
    "fuse_row_kv",
    "fused_residual_clamp",
]
