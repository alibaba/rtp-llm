"""Context-Parallel helpers for DeepSeek-V4.

RTP-LLM's CP repurposes the TP process group as the CP group (see
``ParallelismConfig::get_attn_tp_size`` — returns 1 when CP enabled).
The C++ ``ZigZagProcessor`` splits the padded prefill tokens across the
CP group with a zigzag layout and attaches
``attention_inputs.context_parallel_info`` carrying:

- ``prefill_qkv_padding_mask``  : ``[padded_seq_len] int32`` — 1 for
  real tokens, 0 for padding. Padding sits at global positions
  ``[input_len, padded_seq_len)``.
- ``prefill_qkv_restore_indice``: ``[padded_seq_len] int32`` —
  ``restore[global_pos] = gathered_flat_idx`` (where gathered is the
  concat of per-rank chunks: ``cp_rank * chunk_len + local_idx``).

V4's model-side CP work:
- Attention: rank-local Q (chunk_len tokens) × FULL KV (stripped to
  ``input_len``).  RoPE uses **global** positions.
- Compressor / Indexer: all-gather kv/score before S-pool, so every
  rank's ``kv_cache`` buffer holds the full compressed KV for decode.
- MoE: stays rank-local — DeepEP dispatches the ``chunk_len`` tokens
  naturally; no gather needed.

All derived per-forward quantities are bundled into ``CPContext`` and
stashed on each module via ``_cp_ctx`` before ``forward`` runs.  A
``None`` value means "no CP" and every module falls through to the
single-rank path unchanged.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather


@dataclass
class CPContext:
    """Derived CP metadata for one prefill forward."""

    cp_size: int
    cp_rank: int
    # Rank-local chunk length (== input_ids.size(1) on the adapter side;
    # framework-padded multiple of ``cp_size*2/cp_size = 2``).
    chunk_length: int
    # Total padded global seqlen = cp_size * chunk_length.
    padded_seq_len: int
    # Real un-padded global seqlen (= user's prefill input length).
    seq_len_full: int
    # [chunk_length] int64 — local idx i -> position in the current CP
    # prefill shard, before adding any reused prefix offset.
    relative_positions: torch.Tensor
    # Prefix length already resident in KV cache for continuation prefill.
    prefix_length: int
    # [chunk_length] int64 — local idx i -> absolute sequence position
    # (prefix_length + relative_positions).  For padding local idxs, the
    # position is ≥ seq_len_total; their attention output is discarded by
    # the framework's strip-pad gather.
    global_positions: torch.Tensor
    # [chunk_length] bool — True if local idx maps to a real token
    # (padding_mask[relative_pos] == 1), False for padding slots.
    local_is_real: torch.Tensor
    # [seq_len_full] int64 — gathered-flat index for each real token in
    # GLOBAL order. ``gathered.index_select(0, unpad_restore)`` yields the
    # full un-padded sequence.
    unpad_restore: torch.Tensor
    # Total sequence length after this prefill step (prefix + current input).
    seq_len_total: int
    # Raw cp_info, kept for any caller needing extra fields.
    cp_info: object


@dataclass
class CPSyncGatherHandle:
    """Completed synchronous CP gather result."""

    full_2d: torch.Tensor


@dataclass
class CPCudaAsyncGatherHandle:
    """In-flight CUDA/NCCL CP gather result."""

    cp_ctx: CPContext
    gathered: torch.Tensor
    work: Any
    stream: Any
    local_2d: torch.Tensor


class CPGatherImplBase:
    """Base interface for explicit CP gather implementations."""

    def start(
        self,
        local_2d: torch.Tensor,
        cp_ctx: CPContext,
        stream: Optional[Any] = None,
    ) -> Any:
        raise NotImplementedError

    def wait(self, handle: Any) -> torch.Tensor:
        raise NotImplementedError


class SyncCPGatherImpl(CPGatherImplBase):
    """Reference implementation using the synchronous collective wrapper."""

    def start(
        self,
        local_2d: torch.Tensor,
        cp_ctx: CPContext,
        stream: Optional[Any] = None,
    ) -> CPSyncGatherHandle:
        del stream
        return CPSyncGatherHandle(full_2d=cp_all_gather_full(local_2d, cp_ctx))

    def wait(self, handle: Any) -> torch.Tensor:
        if not isinstance(handle, CPSyncGatherHandle):
            raise TypeError(f"SyncCPGatherImpl.wait expected CPSyncGatherHandle, got {type(handle)!r}")
        return handle.full_2d


class CudaAsyncCPGatherImpl(CPGatherImplBase):
    """Production CUDA implementation.

    This path intentionally fails fast when CUDA distributed all-gather cannot
    be launched. CPU/reference execution should select ``SyncCPGatherImpl``
    explicitly instead of relying on a silent fallback.
    """

    def start(
        self,
        local_2d: torch.Tensor,
        cp_ctx: CPContext,
        stream: Optional[Any] = None,
    ) -> CPCudaAsyncGatherHandle:
        local_2d = _cp_gather_2d(local_2d, cp_ctx)
        if not local_2d.is_cuda:
            raise RuntimeError("CudaAsyncCPGatherImpl requires CUDA tensor input")
        if not torch.distributed.is_initialized():
            raise RuntimeError("CudaAsyncCPGatherImpl requires initialized torch.distributed")

        process_group = collective_torch._get_group(Group.TP)
        world_size = torch.distributed.get_world_size(process_group)
        if world_size != cp_ctx.cp_size:
            raise RuntimeError(
                f"CP gather world_size({world_size}) != cp_ctx.cp_size({cp_ctx.cp_size})"
            )

        gathered = torch.empty(
            (world_size * local_2d.size(0), local_2d.size(1)),
            device=local_2d.device,
            dtype=local_2d.dtype,
        )

        current_stream = torch.cuda.current_stream(local_2d.device)
        gather_stream = stream or torch.cuda.Stream(device=local_2d.device)
        gather_stream.wait_stream(current_stream)
        try:
            with torch.cuda.stream(gather_stream):
                work = torch.distributed.all_gather_into_tensor(
                    gathered,
                    local_2d,
                    group=process_group,
                    async_op=True,
                )
        except Exception as exc:
            raise RuntimeError("failed to launch CUDA CP all_gather_into_tensor") from exc

        return CPCudaAsyncGatherHandle(
            cp_ctx=cp_ctx,
            gathered=gathered,
            work=work,
            stream=gather_stream,
            local_2d=local_2d,
        )

    def wait(self, handle: Any) -> torch.Tensor:
        if not isinstance(handle, CPCudaAsyncGatherHandle):
            raise TypeError(
                f"CudaAsyncCPGatherImpl.wait expected CPCudaAsyncGatherHandle, got {type(handle)!r}"
            )
        torch.cuda.current_stream(handle.gathered.device).wait_stream(handle.stream)
        handle.work.wait()
        return _cp_restore_gathered_full_2d(handle.gathered, handle.cp_ctx)


def build_cp_context(
    cp_info,
    cp_size: int,
    cp_rank: int,
    chunk_length: int,
    device: torch.device,
    position_offset: int = 0,
) -> CPContext:
    """Compute the per-forward derived CPContext from framework metadata."""
    padding_mask = cp_info.prefill_qkv_padding_mask
    restore_indices = cp_info.prefill_qkv_restore_indice
    if padding_mask.device != device:
        padding_mask = padding_mask.to(device)
        restore_indices = restore_indices.to(device)
    padded_seq_len = int(padding_mask.shape[0])

    # For B=1 single prefill stream (V4 is B=1 only), cp_size * chunk_length
    # must equal padded_seq_len.  Assert so future multi-stream support
    # triggers a clean failure instead of silent index corruption.
    assert cp_size * chunk_length == padded_seq_len, (
        f"cp_size({cp_size}) * chunk_length({chunk_length}) != "
        f"padded_seq_len({padded_seq_len}) — multi-stream CP not yet supported"
    )
    pair_size = chunk_length // 2
    assert (
        pair_size * 2 == chunk_length
    ), f"chunk_length({chunk_length}) must be even for zigzag CP"

    # Formula-based current-input positions (matches ZigZagProcessor::plan).
    arange_pair = torch.arange(pair_size, dtype=torch.long, device=device)
    even_positions = cp_rank * pair_size + arange_pair
    odd_positions = padded_seq_len - (cp_rank + 1) * pair_size + arange_pair
    relative_positions = torch.cat([even_positions, odd_positions])  # [chunk_length]

    local_is_real = padding_mask[relative_positions] == 1  # [chunk_length] bool
    unpad_restore = restore_indices[padding_mask == 1].to(torch.long)  # [seq_len_full]
    seq_len_full = int(unpad_restore.shape[0])
    prefix_length = int(position_offset)
    global_positions = relative_positions + prefix_length
    seq_len_total = prefix_length + seq_len_full

    return CPContext(
        cp_size=int(cp_size),
        cp_rank=int(cp_rank),
        chunk_length=int(chunk_length),
        padded_seq_len=padded_seq_len,
        seq_len_full=seq_len_full,
        relative_positions=relative_positions,
        prefix_length=prefix_length,
        global_positions=global_positions,
        local_is_real=local_is_real,
        unpad_restore=unpad_restore,
        seq_len_total=seq_len_total,
        cp_info=cp_info,
    )


def _cp_gather_2d(
    local_2d: torch.Tensor,
    cp_ctx: CPContext,
) -> torch.Tensor:
    """Validate CP gather input and return contiguous token-major 2D data.

    Contract:
    - input shape is flattened ``[T_local, H]``;
    - ``T_local`` must equal ``cp_ctx.chunk_length``;
    - no batch dimension is accepted here, so callers own any
      ``[1, T, H]`` squeeze/unsqueeze at module boundaries.
    """
    if local_2d.dim() != 2:
        raise ValueError(f"CP gather expects 2D [T_local, H], got shape {tuple(local_2d.shape)}")
    if local_2d.size(0) != cp_ctx.chunk_length:
        raise ValueError(
            f"CP gather T_local({local_2d.size(0)}) != cp_ctx.chunk_length({cp_ctx.chunk_length})"
        )
    if local_2d.size(1) <= 0:
        raise ValueError(f"CP gather hidden dimension must be positive, got shape {tuple(local_2d.shape)}")
    return local_2d.contiguous()


def _cp_restore_gathered_full_2d(
    gathered: torch.Tensor,
    cp_ctx: CPContext,
) -> torch.Tensor:
    if gathered.dim() != 2:
        raise ValueError(f"CP gathered tensor must be 2D [T_padded, H], got shape {tuple(gathered.shape)}")
    expected_rows = cp_ctx.cp_size * cp_ctx.chunk_length
    if gathered.size(0) != expected_rows:
        raise ValueError(f"CP gathered rows({gathered.size(0)}) != expected rows({expected_rows})")
    full = gathered.index_select(0, cp_ctx.unpad_restore)  # [seq_len_full, H]
    if full.size(0) != cp_ctx.seq_len_full:
        raise ValueError(f"CP restored rows({full.size(0)}) != cp_ctx.seq_len_full({cp_ctx.seq_len_full})")
    return full


def cp_all_gather_full(
    local_2d: torch.Tensor,
    cp_ctx: CPContext,
) -> torch.Tensor:
    """Synchronously gather CP-local ``[T_local, H]`` to full ``[T_full, H]``.

    The API is intentionally 2D and token-major. Sequence/batch layout belongs
    to the caller; passing ``[B, S, H]`` or arbitrary trailing dimensions is a
    contract violation.
    """
    local_2d = _cp_gather_2d(local_2d, cp_ctx)
    gathered = all_gather(local_2d, group=Group.TP)
    # gathered: [cp_size * chunk_length, H]
    return _cp_restore_gathered_full_2d(gathered, cp_ctx)


def cp_all_gather_full_async(
    local_2d: torch.Tensor,
    cp_ctx: CPContext,
    stream: Optional[Any] = None,
) -> Any:
    """Start CP gather for flattened ``[T_local, H]`` input.

    Default implementation is selected by ``DSV4_CP_GATHER_IMPL``:
    ``async`` (default) for CUDA/NCCL production, or ``sync`` for explicit
    reference execution/tests. The async path has no implicit fallback.
    """
    return get_cp_gather_impl().start(local_2d, cp_ctx, stream=stream)


def cp_wait_gather_full(handle: Any) -> torch.Tensor:
    """Wait for a deferred CP gather and return flattened ``[T_full, H]``."""
    if isinstance(handle, CPSyncGatherHandle):
        return SyncCPGatherImpl().wait(handle)
    if isinstance(handle, CPCudaAsyncGatherHandle):
        return CudaAsyncCPGatherImpl().wait(handle)
    raise TypeError(f"unsupported CP gather handle type: {type(handle)!r}")


def build_cp_gather_impl(mode: Optional[str] = None) -> CPGatherImplBase:
    mode = (mode or os.environ.get("DSV4_CP_GATHER_IMPL", "async")).strip().lower()
    if mode == "async":
        return CudaAsyncCPGatherImpl()
    if mode == "sync":
        return SyncCPGatherImpl()
    raise ValueError(f"unsupported DSV4_CP_GATHER_IMPL={mode!r}; expected 'async' or 'sync'")


_CP_GATHER_IMPL: Optional[CPGatherImplBase] = None


def get_cp_gather_impl() -> CPGatherImplBase:
    global _CP_GATHER_IMPL
    if _CP_GATHER_IMPL is None:
        _CP_GATHER_IMPL = build_cp_gather_impl()
    return _CP_GATHER_IMPL


def cp_freqs_cis_local(
    freqs_cis: torch.Tensor,
    cp_ctx: CPContext,
) -> torch.Tensor:
    """Select ``freqs_cis`` rows at the GLOBAL positions of this rank's
    local tokens → ``[chunk_length, rope_dim/2]`` complex tensor suitable
    for ``apply_rotary_emb`` against a rank-local Q/K of length
    ``chunk_length``.

    Padding slots pick a valid (in-range) row from ``freqs_cis``; their
    attention output is discarded by the framework after the exit
    all-gather, so the specific rotation angle doesn't matter as long as
    it doesn't NaN or OOB."""
    pos = cp_ctx.global_positions
    if pos.device != freqs_cis.device:
        pos = pos.to(freqs_cis.device)
    # freqs_cis is complex [max_seq_len, rope_dim//2].  Clamp positions
    # to a valid range — padding slots compute a RoPE that gets thrown
    # away, but must not index past the end.
    max_s = freqs_cis.size(0)
    pos = pos.clamp_max(max_s - 1)
    return freqs_cis.index_select(0, pos)


# Legacy shim kept so existing callers (old scaffold) still link while
# we move them over.  New code should use CPContext + cp_all_gather_full.
def cp_all_gather_to_full(
    local: torch.Tensor,
    cp_info,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """Deprecated: use ``build_cp_context`` + ``cp_all_gather_full``.

    ``local`` follows the same strict flattened contract:
    ``[chunk_length, H] -> [seq_len_full, H]``.
    """
    device = local.device
    if local.dim() != 2:
        raise ValueError(f"cp_all_gather_to_full expects 2D [T_local, H], got shape {tuple(local.shape)}")
    chunk_length = local.size(0)
    ctx = build_cp_context(cp_info, cp_size, cp_rank, chunk_length, device)
    return cp_all_gather_full(local, ctx)


def cp_should_gather(cp_ctx: Optional[CPContext], start_pos: int) -> bool:
    """Prefill-only gate: gather runs iff a CPContext is bound.

    ``start_pos`` is intentionally ignored: CP continuation prefill also
    receives only the rank-local suffix, so compressor/indexer state must still
    all-gather the current input before pooling.
    """
    return cp_ctx is not None and cp_ctx.cp_size > 1
