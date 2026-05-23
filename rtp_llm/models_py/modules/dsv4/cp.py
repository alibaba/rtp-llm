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
from typing import Any, Optional, Tuple, Union

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
    # [chunk_length] int64 — local idx i -> position in the padded
    # request-concat stream consumed by ``prefill_qkv_padding_mask`` /
    # ``prefill_qkv_restore_indice``.
    relative_positions: torch.Tensor
    # Prefix length already resident in KV cache for continuation prefill.
    prefix_length: int
    # [chunk_length] int64 — local idx i -> absolute sequence position
    # (prefix_length + relative_positions).  For padding local idxs, the
    # position is clamped to a valid token in the same request; their
    # attention output is discarded by the framework's strip-pad gather.
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
    # [chunk_length] int32 — request id for each rank-local token.  Under
    # B==1 this is all zeros.  Under B>1 it mirrors the framework's flat
    # rank-local order: all local tokens for request 0, then request 1, ...
    req_id_per_token: Optional[torch.Tensor] = None
    # [B] int64 — per-request prefix/start position.  Kept because CP+B>1
    # cannot be represented by the legacy scalar ``prefix_length``.
    prefix_lengths: Optional[torch.Tensor] = None
    # ----- Pool-write side (CP "global" view) ------------------------------
    # The framework's ``attention_inputs.{cu_seqlens, input_lengths}`` are
    # rank-local (already split by ZigZagProcessor). After gather the KV tensor
    # has length ``seq_len_full`` in GLOBAL request order, so SWA paged-tail
    # slot mappings need the global per-request view.
    input_lengths_global: Optional[torch.Tensor] = None
    cu_seqlens_global: Optional[torch.Tensor] = None
    # True when unpadding is just ``gathered[:seq_len_full]``.  This is
    # computed while building CPContext from CPU restore metadata to avoid a
    # per-forward CUDA comparison on the hot path.
    unpad_restore_is_prefix: bool = False
    # Rank-local per-request chunk lengths after ZigZagProcessor splitting.
    # Each request chunk is laid out as [front half, back half] locally.
    chunk_lengths_per_req: Optional[Tuple[int, ...]] = None
    # Stage 5b: True when the prefill node's CSA / HCA / INDEXER paged pools
    # are RR-sharded across CP ranks (logical block ``b`` owned by rank
    # ``b % cp_size``). When False, attention's compressed-K read goes
    # straight to the local pool. Plumbed from
    # ``parallelism_config.prefill_cp_config.kv_cache_sharded``.
    kv_cache_sharded: bool = False


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
            raise TypeError(
                f"SyncCPGatherImpl.wait expected CPSyncGatherHandle, got {type(handle)!r}"
            )
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
            raise RuntimeError(
                "CudaAsyncCPGatherImpl requires initialized torch.distributed"
            )

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
            raise RuntimeError(
                "failed to launch CUDA CP all_gather_into_tensor"
            ) from exc

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
    position_offset: Union[int, torch.Tensor] = 0,
    kv_cache_sharded: bool = False,
) -> CPContext:
    """Compute the per-forward derived CPContext from framework metadata."""
    padding_mask = cp_info.prefill_qkv_padding_mask
    restore_indices = cp_info.prefill_qkv_restore_indice
    if padding_mask.device != device:
        padding_mask = padding_mask.to(device)
    if restore_indices.device != device:
        restore_indices = restore_indices.to(device)
    padded_seq_len = int(padding_mask.shape[0])

    if cp_size * chunk_length != padded_seq_len:
        raise ValueError(
            f"cp_size({cp_size}) * chunk_length({chunk_length}) != "
            f"padded_seq_len({padded_seq_len})"
        )

    actual_input_lengths_cpu = getattr(
        cp_info, "prefill_actual_input_lengths_cpu", None
    )
    input_lengths_global: Optional[torch.Tensor] = None
    cu_seqlens_global: Optional[torch.Tensor] = None
    if actual_input_lengths_cpu is not None and actual_input_lengths_cpu.numel() > 0:
        input_lengths_global = actual_input_lengths_cpu.to(
            device=device, dtype=torch.int32
        ).contiguous()
        zero = torch.zeros(1, dtype=torch.int32, device=device)
        cu_seqlens_global = torch.cat(
            [zero, torch.cumsum(input_lengths_global, dim=0).to(torch.int32)]
        ).contiguous()

    chunk_lengths_obj = getattr(cp_info, "prefill_cp_chunk_lengths", None)
    if chunk_lengths_obj is not None and chunk_lengths_obj.numel() > 0:
        chunk_lengths = [int(v) for v in chunk_lengths_obj.detach().cpu().tolist()]
    elif input_lengths_global is not None and input_lengths_global.numel() == 1:
        chunk_lengths = [chunk_length]
    else:
        # Test/legacy fallback: a single stream with the caller-provided
        # aggregate rank-local chunk length.
        chunk_lengths = [chunk_length]

    assert sum(chunk_lengths) == chunk_length, (
        f"sum(prefill_cp_chunk_lengths)={sum(chunk_lengths)} != "
        f"chunk_length={chunk_length}"
    )
    for i, per_req_chunk in enumerate(chunk_lengths):
        assert per_req_chunk % 2 == 0, (
            f"prefill_cp_chunk_lengths[{i}]={per_req_chunk} must be even "
            "for zigzag CP"
        )

    if input_lengths_global is not None:
        B = int(input_lengths_global.numel())
    else:
        B = len(chunk_lengths)
    assert B == len(
        chunk_lengths
    ), f"num global lengths ({B}) != num CP chunks ({len(chunk_lengths)})"

    if isinstance(position_offset, torch.Tensor):
        prefix_lengths = position_offset.to(
            device=device, dtype=torch.long
        ).contiguous()
        if prefix_lengths.numel() == 1 and B > 1:
            prefix_lengths = prefix_lengths.expand(B).contiguous()
    else:
        prefix_lengths = torch.full(
            (B,), int(position_offset), dtype=torch.long, device=device
        )
    assert (
        prefix_lengths.numel() >= B
    ), f"prefix_lengths has {prefix_lengths.numel()} entries, expected at least {B}"
    prefix_lengths = prefix_lengths[:B].contiguous()

    if input_lengths_global is not None:
        real_lengths = input_lengths_global.to(device=device, dtype=torch.long)
    else:
        # No actual lengths means no padding information beyond the mask.
        # For the single-stream fallback this collapses to seq_len_full below.
        real_lengths = torch.tensor(
            [int((padding_mask == 1).sum().item())],
            dtype=torch.long,
            device=device,
        )

    # C++ ZigZagProcessor applies the zigzag plan independently per prefill
    # stream/request, then concatenates the rank-local chunks.  Generate the
    # same padded-concat coordinates here; the previous single-stream formula
    # is only valid for B==1.
    padded_positions = []
    per_req_positions = []
    req_ids = []
    padded_seq_offset = 0
    for req_id, per_req_chunk in enumerate(chunk_lengths):
        pair_size = per_req_chunk // 2
        padded_len = per_req_chunk * cp_size
        arange_pair = torch.arange(pair_size, dtype=torch.long, device=device)
        even_padded = padded_seq_offset + cp_rank * pair_size + arange_pair
        odd_padded = (
            padded_seq_offset + padded_len - (cp_rank + 1) * pair_size + arange_pair
        )
        req_relative = torch.cat(
            [even_padded - padded_seq_offset, odd_padded - padded_seq_offset]
        )
        if req_id < int(real_lengths.numel()):
            max_real_pos = max(int(real_lengths[req_id].item()) - 1, 0)
        else:
            max_real_pos = max(padded_len - 1, 0)
        per_req_positions.append(req_relative.clamp_max(max_real_pos))
        padded_positions.append(torch.cat([even_padded, odd_padded]))
        req_ids.append(
            torch.full((per_req_chunk,), req_id, dtype=torch.int32, device=device)
        )
        padded_seq_offset += padded_len

    relative_positions = torch.cat(padded_positions).contiguous()
    local_positions = torch.cat(per_req_positions).contiguous()
    req_id_per_token = torch.cat(req_ids).contiguous()

    local_is_real = padding_mask[relative_positions] == 1  # [chunk_length] bool
    unpad_restore_is_prefix = False
    if input_lengths_global is not None:
        seq_len_full = int(input_lengths_global.to(torch.long).sum().item())
    else:
        seq_len_full = int((padding_mask == 1).sum().item())

    if input_lengths_global is not None and int(input_lengths_global.numel()) == 1:
        # Single stream metadata stores real tokens as a prefix, so avoid
        # boolean indexing on the hot path and keep the current async gather
        # prefix fast path.
        restore_prefix = restore_indices[:seq_len_full]
        if restore_prefix.device.type == "cpu":
            expected = torch.arange(seq_len_full, dtype=restore_prefix.dtype)
            unpad_restore_is_prefix = bool(torch.equal(restore_prefix.cpu(), expected))
        unpad_restore = restore_prefix.to(torch.long)
    else:
        # Multi-request CP has padding after each request's real-token prefix,
        # so restore rows must be selected with the full concat padding mask.
        unpad_restore = restore_indices[padding_mask == 1].to(torch.long)
        seq_len_full = int(unpad_restore.shape[0])
    prefix_per_token = prefix_lengths.gather(0, req_id_per_token.to(torch.long))
    global_positions = (prefix_per_token + local_positions).contiguous()
    prefix_length = int(prefix_lengths[0].item()) if prefix_lengths.numel() > 0 else 0
    if input_lengths_global is not None:
        seq_len_total = int((prefix_lengths + real_lengths[:B]).max().item())
    else:
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
        req_id_per_token=req_id_per_token,
        prefix_lengths=prefix_lengths,
        local_is_real=local_is_real,
        unpad_restore=unpad_restore,
        seq_len_total=seq_len_total,
        cp_info=cp_info,
        input_lengths_global=input_lengths_global,
        cu_seqlens_global=cu_seqlens_global,
        unpad_restore_is_prefix=unpad_restore_is_prefix,
        chunk_lengths_per_req=tuple(chunk_lengths),
        kv_cache_sharded=bool(kv_cache_sharded),
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
        raise ValueError(
            f"CP gather expects 2D [T_local, H], got shape {tuple(local_2d.shape)}"
        )
    if local_2d.size(0) != cp_ctx.chunk_length:
        raise ValueError(
            f"CP gather T_local({local_2d.size(0)}) != cp_ctx.chunk_length({cp_ctx.chunk_length})"
        )
    if local_2d.size(1) <= 0:
        raise ValueError(
            f"CP gather hidden dimension must be positive, got shape {tuple(local_2d.shape)}"
        )
    if local_2d.is_contiguous():
        return local_2d
    return local_2d.contiguous()


def _cp_restore_gathered_full_2d(
    gathered: torch.Tensor,
    cp_ctx: CPContext,
) -> torch.Tensor:
    if gathered.dim() != 2:
        raise ValueError(
            f"CP gathered tensor must be 2D [T_padded, H], got shape {tuple(gathered.shape)}"
        )
    expected_rows = cp_ctx.cp_size * cp_ctx.chunk_length
    if gathered.size(0) != expected_rows:
        raise ValueError(
            f"CP gathered rows({gathered.size(0)}) != expected rows({expected_rows})"
        )
    if cp_ctx.unpad_restore_is_prefix:
        full = gathered[: cp_ctx.seq_len_full]  # [seq_len_full, H], view
    else:
        full = gathered.index_select(0, cp_ctx.unpad_restore)  # [seq_len_full, H]
    if full.size(0) != cp_ctx.seq_len_full:
        raise ValueError(
            f"CP restored rows({full.size(0)}) != cp_ctx.seq_len_full({cp_ctx.seq_len_full})"
        )
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
    raise ValueError(
        f"unsupported DSV4_CP_GATHER_IMPL={mode!r}; expected 'async' or 'sync'"
    )


_CP_GATHER_IMPL: Optional[CPGatherImplBase] = None


def get_cp_gather_impl() -> CPGatherImplBase:
    global _CP_GATHER_IMPL
    if _CP_GATHER_IMPL is None:
        _CP_GATHER_IMPL = build_cp_gather_impl()
    return _CP_GATHER_IMPL


def cp_all_gather_full_varlen(
    local_flat: torch.Tensor,
    cp_ctx: CPContext,
) -> torch.Tensor:
    """**Varlen B>=1 path** (default ``DSV4_VARLEN_PREFILL=1``):
    all-gather a flat ``[chunk_length, *F]`` rank-local tensor across the
    CP (== TP) group and strip padding → ``[seq_len_full, *F]`` in
    GLOBAL per-request-concat order. No leading B dim.

    For B==1 this is mathematically identical to
    :func:`cp_all_gather_full` (just without the extra unsqueeze). For
    B>1 ``seq_len_full`` is the sum of per-request real lengths and the
    output is the request-concatenated KV stream — every consumer in
    the FP8 prefill stack (compressor/indexer/SWA write/attention)
    treats this as a single virtual sequence (matching the existing
    non-CP B>1 behaviour documented in
    ``prefill/forward.py::forward_layers``).
    """
    assert local_flat.dim() >= 1
    assert (
        local_flat.size(0) == cp_ctx.chunk_length
    ), f"local_flat.size(0)={local_flat.size(0)} != chunk_length={cp_ctx.chunk_length}"
    trailing = local_flat.shape[1:]
    local_2d = local_flat.reshape(cp_ctx.chunk_length, -1).contiguous()
    gathered = all_gather(local_2d, group=Group.TP)
    full = _cp_restore_gathered_full_2d(gathered, cp_ctx)
    return full.view((cp_ctx.seq_len_full,) + trailing)


def cp_gather_last_by_request(
    local_2d: torch.Tensor,
    cp_ctx: CPContext,
) -> torch.Tensor:
    """Gather only each request's final prefill row across CP ranks.

    ``local_2d`` is rank-local ``[chunk_length, H]`` in the same layout as the
    CP-split input.  The result is ``[B, H]`` in request order.  This avoids the
    old full-sequence all-gather used only to slice one row per request for MTP.
    """
    assert local_2d.dim() == 2
    assert (
        local_2d.size(0) == cp_ctx.chunk_length
    ), f"local_2d.size(0)={local_2d.size(0)} != chunk_length={cp_ctx.chunk_length}"
    assert (
        cp_ctx.input_lengths_global is not None
    ), "CP last-hidden gather requires input_lengths_global"
    assert cp_ctx.prefix_lengths is not None, "CP last-hidden gather requires prefixes"
    assert (
        cp_ctx.req_id_per_token is not None
    ), "CP last-hidden gather requires req_id_per_token"

    device = local_2d.device
    B = int(cp_ctx.input_lengths_global.numel())
    H = int(local_2d.size(1))

    req_ids = cp_ctx.req_id_per_token.to(device=device, dtype=torch.long).reshape(-1)
    prefixes = cp_ctx.prefix_lengths.to(device=device, dtype=torch.long).reshape(-1)[:B]
    input_lengths = cp_ctx.input_lengths_global.to(
        device=device, dtype=torch.long
    ).reshape(-1)[:B]
    target_positions = prefixes + input_lengths - 1
    target_per_token = target_positions.gather(0, req_ids)

    global_positions = cp_ctx.global_positions.to(device=device, dtype=torch.long)
    local_is_real = cp_ctx.local_is_real.to(device=device)
    owner_mask = local_is_real & (global_positions == target_per_token)

    local_last = local_2d.new_zeros((B, H))
    owner_req_ids = req_ids[owner_mask]
    local_last.index_copy_(0, owner_req_ids, local_2d[owner_mask])

    # Zigzag CP assigns each real global token to exactly one rank, so each
    # request's final token contributes from one rank only.  Summing the gathered
    # [B, H] rows is therefore equivalent to selecting the unique owner row.
    # RTP-LLM CP reuses the TP process group, matching the full hidden gather
    # path above.
    gathered = all_gather(local_last.contiguous(), group=Group.TP)
    return gathered.view(cp_ctx.cp_size, B, H).sum(dim=0).contiguous()


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
        raise ValueError(
            f"cp_all_gather_to_full expects 2D [T_local, H], got shape {tuple(local.shape)}"
        )
    chunk_length = local.size(0)
    ctx = build_cp_context(cp_info, cp_size, cp_rank, chunk_length, device)
    return cp_all_gather_full(local, ctx)


def combine_topk_swa_indices_cp_varlen(
    topk_indices: torch.Tensor,
    global_positions: torch.Tensor,
    sp_int: int,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
    req_id_per_token: Optional[torch.Tensor] = None,
    prefix_lengths: Optional[torch.Tensor] = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """**Varlen B>=1 path** (default ``DSV4_VARLEN_PREFILL=1``):
    multi-request-aware vectorized replacement for
    ``_swa_ops_triton.combine_topk_swa_indices`` under CP.

    For B==1 single-request prefill this is bit-equal to
    :func:`combine_topk_swa_indices_cp_b1`. For B>1 it emits indices in
    the flattened workspace coordinate system:

      * compressed slots: ``req_id * M + topk_indices[t, k]``
      * SWA slots: ``req_id * M + N + local_slot``

    ``global_positions`` are absolute per-request positions
    (``prefix_lengths[req] + local_pos``), not padded-concat positions.
    """
    if req_id_per_token is None or prefix_lengths is None:
        return combine_topk_swa_indices_cp_b1(
            topk_indices=topk_indices,
            global_positions=global_positions,
            sp_int=sp_int,
            window_size=window_size,
            compress_ratio=compress_ratio,
            topk=topk,
            M=M,
            N=N,
        )

    device = topk_indices.device
    T = int(global_positions.shape[0])
    combined_topk_padded = ((topk + window_size + 127) // 128) * 128
    combined_indices = torch.full(
        (T, combined_topk_padded), -1, dtype=torch.int32, device=device
    )
    combined_lens = torch.zeros((T,), dtype=torch.int32, device=device)
    if T == 0:
        return combined_indices, combined_lens

    req = req_id_per_token.to(device=device, dtype=torch.int64)
    prefix = prefix_lengths.to(device=device, dtype=torch.int64)
    gp = global_positions.to(device=device, dtype=torch.int64)
    sp_b = prefix.gather(0, req)
    P_b = torch.clamp_max(sp_b, window_size - 1)
    gather_start = sp_b - P_b

    topk_len = torch.minimum((gp + 1) // compress_ratio, gp.new_full((), topk)).to(
        torch.int64
    )
    swa_len = torch.minimum(gp + 1, gp.new_full((), window_size)).to(torch.int64)
    combined_lens = (topk_len + swa_len).to(torch.int32).contiguous()
    req_base = req * int(M)

    if topk > 0:
        off_k = torch.arange(topk, device=device, dtype=torch.int64).unsqueeze(0)
        mask_k = off_k < topk_len.unsqueeze(1)
        topk_vals = req_base.unsqueeze(1) + topk_indices.to(torch.int64)
        combined_indices[:, :topk] = torch.where(
            mask_k,
            topk_vals.to(torch.int32),
            torch.full((T, topk), -1, dtype=torch.int32, device=device),
        )

    off_w = torch.arange(window_size, device=device, dtype=torch.int64).unsqueeze(0)
    mask_w = off_w < swa_len.unsqueeze(1)
    swa_val = (
        req_base.unsqueeze(1)
        + int(N)
        + (gp - gather_start).unsqueeze(1)
        - swa_len.unsqueeze(1)
        + 1
        + off_w
    ).to(torch.int32)
    target_col = topk_len.unsqueeze(1) + off_w
    flat_target = (
        torch.arange(T, device=device, dtype=torch.int64).unsqueeze(1)
        * combined_topk_padded
        + target_col
    )
    combined_indices.view(-1).scatter_(0, flat_target[mask_w], swa_val[mask_w])
    return combined_indices, combined_lens


def build_cp_full_prefill_positions(
    cp_ctx: CPContext,
    device: torch.device,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """Return full gathered per-token compressor metadata for CP prefill.

    Output order matches ``cp_all_gather_full[_varlen]`` after
    ``unpad_restore``: request 0 real tokens, request 1 real tokens, ...

    Returns ``(positions, b_idx, seq_start_per_req, cu_seq_per_req)``.
    """
    assert (
        cp_ctx.input_lengths_global is not None
    ), "CP full prefill positions require input_lengths_global"
    lengths = cp_ctx.input_lengths_global.to(device=device, dtype=torch.long)
    if cp_ctx.prefix_lengths is not None:
        prefixes = cp_ctx.prefix_lengths.to(device=device, dtype=torch.long)
    else:
        prefixes = torch.full(
            (int(lengths.numel()),),
            int(cp_ctx.prefix_length),
            dtype=torch.long,
            device=device,
        )

    positions = []
    b_idx = []
    for req_id in range(int(lengths.numel())):
        length = int(lengths[req_id].item())
        start = int(prefixes[req_id].item())
        if length <= 0:
            continue
        positions.append(
            torch.arange(start, start + length, dtype=torch.long, device=device)
        )
        b_idx.append(torch.full((length,), req_id, dtype=torch.long, device=device))
    if positions:
        pos = torch.cat(positions).contiguous()
        req = torch.cat(b_idx).contiguous()
    else:
        pos = torch.empty((0,), dtype=torch.long, device=device)
        req = torch.empty((0,), dtype=torch.long, device=device)

    zero = torch.zeros(1, dtype=torch.int32, device=device)
    cu_seq = torch.cat(
        [zero, torch.cumsum(lengths.to(torch.int32), dim=0)]
    ).contiguous()
    return (
        pos,
        req,
        prefixes.to(device=device, dtype=torch.int32).contiguous(),
        cu_seq,
    )


def combine_topk_swa_indices_cp_b1(
    topk_indices: torch.Tensor,
    global_positions: torch.Tensor,
    sp_int: int,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """**Legacy B==1 path** (used under ``DSV4_VARLEN_PREFILL=0``):
    vectorized replacement for ``_swa_ops_triton.combine_topk_swa_indices``.

    The Triton kernel computes ``pos = start_pos + token_idx_in_query``
    assuming Q is a contiguous sequence slice; under zigzag CP the
    rank-local Q rows correspond to non-contiguous global positions, so
    that formula breaks. This helper consumes ``cp_ctx.global_positions``
    directly to compute per-token ``pos`` and emits the same ``[T,
    align(topk+win, 128)]`` int32 layout — sentinel ``-1`` in the unused
    tail — plus ``[T]`` int32 ``combined_lens = topk_len + swa_len``.

    For each Q-row ``t`` with global position ``gp[t]``:
      * ``topk_len = min((gp+1) // ratio, topk)``
      * ``swa_len  = min(gp+1, win)``
      * row[0:topk_len]                 = topk_indices[t, 0:topk_len]
      * row[topk_len:topk_len+swa_len]  = N + (gp - gather_start) - swa_len + 1 + off
        where ``gather_start = sp_int - P``, ``P = min(sp_int, win-1)``.

    Workspace stride ``M`` is currently unused on this legacy B==1 helper
    (``M*batch_idx == 0``), but kept in the signature so callers can share the
    same dispatch shape as the B>=1 varlen helper.
    """
    device = topk_indices.device
    T = int(global_positions.shape[0])
    combined_topk_padded = ((topk + window_size + 127) // 128) * 128
    combined_indices = torch.full(
        (T, combined_topk_padded), -1, dtype=torch.int32, device=device
    )
    combined_lens = torch.zeros((T,), dtype=torch.int32, device=device)
    if T == 0:
        return combined_indices, combined_lens

    gp = global_positions.to(device=device, dtype=torch.int64)
    P = min(sp_int, window_size - 1)
    gather_start = sp_int - P

    topk_len = torch.minimum((gp + 1) // compress_ratio, gp.new_full((), topk)).to(
        torch.int64
    )
    swa_len = torch.minimum(gp + 1, gp.new_full((), window_size)).to(torch.int64)
    combined_lens = (topk_len + swa_len).to(torch.int32).contiguous()

    if topk > 0:
        off_k = torch.arange(topk, device=device, dtype=torch.int64).unsqueeze(0)
        mask_k = off_k < topk_len.unsqueeze(1)
        combined_indices[:, :topk] = torch.where(
            mask_k,
            topk_indices.to(torch.int32),
            torch.full_like(topk_indices, -1, dtype=torch.int32),
        )

    off_w = torch.arange(window_size, device=device, dtype=torch.int64).unsqueeze(0)
    mask_w = off_w < swa_len.unsqueeze(1)
    swa_val = (
        N + (gp - gather_start).unsqueeze(1) - swa_len.unsqueeze(1) + 1 + off_w
    ).to(torch.int32)
    target_col = topk_len.unsqueeze(1) + off_w
    flat_target = (
        torch.arange(T, device=device, dtype=torch.int64).unsqueeze(1)
        * combined_topk_padded
        + target_col
    )
    flat_target = flat_target[mask_w]
    flat_val = swa_val[mask_w]
    combined_indices.view(-1).scatter_(0, flat_target, flat_val)
    return combined_indices, combined_lens


# ---------------------------------------------------------------------------
# CP-sharded paged-pool block gather (Stage 5b).
#
# Under ``prefill_cp_config.kv_cache_sharded=true`` the CSA / HCA / INDEXER
# paged pools are RR-sharded across ``cp_size`` ranks: logical block ``b`` is
# physically owned by rank ``b % cp_size``. Prefill attention's
# ``_attn_via_workspace`` path needs the full request prefix to compute
# correctly, so we gather the missing (cp_size-1)/cp_size fraction from peer
# ranks via NCCL and interleave into logical order.
#
# This module ships two pieces so the interleave (pure tensor reshape) is
# unit-testable on CPU without a distributed init:
#
#   * ``cp_interleave_gathered_pool_blocks`` — pure tensor reorder
#   * ``cp_gather_request_pool_blocks``      — NCCL wrapper + interleave
# ---------------------------------------------------------------------------
def cp_interleave_gathered_pool_blocks(
    gathered: torch.Tensor,
    cp_size: int,
    total_logical_blocks: int,
) -> torch.Tensor:
    """Reorder rank-major NCCL all_gather output to logical block order.

    Args:
      gathered: ``[cp_size * L, *block_shape]`` — rank-major concatenation
                produced by ``all_gather`` of each rank's ``[L, *block_shape]``
                contribution. Rank ``r``'s contribution lives at rows
                ``[r*L, (r+1)*L)``.
      cp_size:  number of CP ranks.
      total_logical_blocks: actual logical block count for the request
                (``L * cp_size`` may overshoot when total is not divisible
                by ``cp_size``; trailing rows are dropped).

    Returns:
      ``[total_logical_blocks, *block_shape]`` in logical order.
      Logical block ``b`` is at row ``b``, sourced from rank ``b % cp_size``'s
      local block ``b // cp_size``.
    """
    if gathered.dim() < 1:
        raise ValueError(
            f"gathered must have rank >= 1, got shape {tuple(gathered.shape)}"
        )
    if cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    n = gathered.size(0)
    if n % cp_size != 0:
        raise ValueError(f"gathered.size(0)={n} not divisible by cp_size={cp_size}")
    L = n // cp_size
    if total_logical_blocks > n:
        raise ValueError(
            f"total_logical_blocks({total_logical_blocks}) > gathered rows({n})"
        )
    block_shape = gathered.shape[1:]
    # Rank-major view: [cp_size, L, *block_shape] -> permute to
    # [L, cp_size, *block_shape] so flatten yields logical order
    # row[i*cp_size + r] == rank r local block i, equivalent to logical
    # block (r + i*cp_size).
    viewed = gathered.view(cp_size, L, *block_shape)
    perm_dims = (1, 0) + tuple(range(2, viewed.dim()))
    out = viewed.permute(*perm_dims).contiguous().view(cp_size * L, *block_shape)
    return out[:total_logical_blocks]


def cp_gather_request_pool_blocks(
    local_pool: torch.Tensor,
    local_block_table_for_req: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    total_logical_blocks: int,
) -> torch.Tensor:
    """Gather one request's owned-block slice from each CP rank into logical order.

    Each rank packs its locally-owned blocks into a contiguous tensor, calls
    NCCL ``all_gather``, then reorders the result so logical block ``b`` is at
    row ``b``. Logical block ``b`` is owned by rank ``b % cp_size``.

    Args:
      local_pool: ``[num_pool_blocks, *block_shape]`` — this rank's full pool
                  storage (e.g. ``[N, block_size, slot_bytes]`` for FP8 paged
                  CSA / HCA / INDEXER pools).
      local_block_table_for_req: ``[L]`` int64 / int32 — physical pool block
                  ids, one per locally-owned logical position. Must be the
                  rank's view of the request's block table after Stage 5a's
                  ``HybridKVCacheAllocator`` allocation
                  (``L = ceil(total_logical_blocks / cp_size)``).
      cp_size, cp_rank: CP geometry.
      total_logical_blocks: logical block count for the request, used to
                  trim trailing padded rows when total is not a multiple of
                  ``cp_size``.

    Returns:
      ``[total_logical_blocks, *block_shape]`` logical-order assembly of the
      request's full prefix data.
    """
    if local_pool.dim() < 2:
        raise ValueError(
            f"local_pool must have rank >= 2, got shape {tuple(local_pool.shape)}"
        )
    if local_block_table_for_req.dim() != 1:
        raise ValueError(
            f"local_block_table_for_req must be 1D, got shape "
            f"{tuple(local_block_table_for_req.shape)}"
        )
    L = int(local_block_table_for_req.numel())
    if cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if not (0 <= cp_rank < cp_size):
        raise ValueError(f"cp_rank({cp_rank}) out of range [0, {cp_size})")

    expected_L = (total_logical_blocks + cp_size - 1) // cp_size
    if L != expected_L:
        raise ValueError(
            f"local_block_table size {L} != ceil(total_logical_blocks="
            f"{total_logical_blocks} / cp_size={cp_size}) = {expected_L}"
        )

    # Pack owned blocks contiguously.
    local_owned = local_pool.index_select(
        0, local_block_table_for_req.to(device=local_pool.device, dtype=torch.long)
    ).contiguous()

    # NCCL all_gather. Each rank contributes [L, *block_shape] -> output is
    # [cp_size * L, *block_shape] rank-major.
    gathered = all_gather(local_owned, group=Group.TP)

    return cp_interleave_gathered_pool_blocks(gathered, cp_size, total_logical_blocks)


# ---------------------------------------------------------------------------
# Stage 5b — per-iteration restore-indices builder (branch
# `_get_topk_ragged_cp_roundrobin` pattern).
#
# Each CP rank holds the OWNED K of every request (1/cp_size of the global
# token stream, RR by ``logical_block_idx % cp_size``). For prefill
# attention / indexer to see the full K, each layer:
#
#   1. gathers its LOCAL owned K from the local pool (one cp_gather op,
#      per-request ``cu_local_kv_seqlens`` is the count of owned tokens),
#   2. ``all_gather`` across cp_size ranks → ``[cp_size * local_kv_len, D]``
#      rank-major,
#   3. reorders into logical token order via the precomputed
#      ``restore_indices`` ``[total_kv_len]`` index tensor.
#
# This builder produces step (3)'s ``restore_indices`` once at prefill
# planning time. It is request-batched (concatenates per-request restore
# slices) so a single tensor covers the whole iteration.
# ---------------------------------------------------------------------------
def build_kv_allgather_restore_indices(
    per_req_total_kv_lens: torch.Tensor,
    cp_size: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute the per-iteration restore index tensor.

    The all_gather output is laid out as ``[cp_size, sum_b L_b, D]`` (rank
    major; rank ``r``'s contribution lives at row range
    ``[r * sum_L, (r+1) * sum_L)``). We want to reorder this into logical
    token order (per request, then concatenated across requests).

    Owner rule: logical block ``b`` owned by rank ``b % cp_size``. Within a
    block, all ``block_size`` tokens have the same owner. So for request r
    with ``T_r`` total kv tokens:

      logical token ``t`` (in [0, T_r)) is owned by rank
      ``(t // block_size) % cp_size``, and lives at the rank's local
      position ``(t // block_size) // cp_size * block_size + t %
      block_size``.

    Args:
      per_req_total_kv_lens: ``[B]`` int64 — total KV tokens per request
            (== prefix_len + new_len for prefill iterations after the
            writer has written this iteration's K).
      cp_size: number of CP ranks.
      block_size: tokens per pool block (==
            ``cache_config.seq_size_per_block`` for the relevant pool;
            CSA/HCA/INDEXER pool block_size is uniform per pool).
      device: target device.

    Returns:
      ``restore_indices`` ``[total_kv_len]`` int64 — applied as
      ``flat_global = gathered_flat[restore_indices]``. Each row of
      ``flat_global`` is the K for that logical token, in
      request-concatenated logical order.
    """
    if cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if per_req_total_kv_lens.dim() != 1:
        raise ValueError(
            f"per_req_total_kv_lens must be 1D, got shape "
            f"{tuple(per_req_total_kv_lens.shape)}"
        )

    # Vectorized formulation (mirrors branch
    # ``flashmla_sparse_cp_impl.py:565-572`` plan()).
    total_kv_lens = per_req_total_kv_lens.to(dtype=torch.int64, device=device)
    if int(total_kv_lens.numel()) == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    local_per_req = cp_padded_local_kv_lens(total_kv_lens, cp_size, block_size)
    cu_local_per_req = torch.zeros(
        int(local_per_req.numel()) + 1, dtype=torch.int64, device=device
    )
    cu_local_per_req[1:] = torch.cumsum(local_per_req, dim=0)
    total_local_kv = int(cu_local_per_req[-1].item())  # per-rank padded local

    total_real = int(total_kv_lens.sum().item())
    if total_real == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    # Flat ``[total_real]`` arange of "logical token position within request".
    # Fully vectorized: positions_flat[i] = i - cu_starts[req_ids[i]] where
    # cu_starts is the exclusive prefix-sum of per-request lengths. Replaces
    # a per-request Python loop + B .item() syncs with two GPU ops.
    req_ids = torch.repeat_interleave(
        torch.arange(int(total_kv_lens.numel()), dtype=torch.int64, device=device),
        total_kv_lens,
    )
    cu_starts = torch.zeros(
        int(total_kv_lens.numel()), dtype=torch.int64, device=device
    )
    cu_starts[1:] = torch.cumsum(total_kv_lens[:-1], dim=0)
    positions_flat = (
        torch.arange(total_real, dtype=torch.int64, device=device) - cu_starts[req_ids]
    )
    cu_offsets = cu_local_per_req[req_ids]  # per-token rank-local request base
    global_block_idx = positions_flat // block_size
    token_in_block = positions_flat % block_size
    owner = global_block_idx % cp_size
    local_block_idx = global_block_idx // cp_size
    local_pos_in_req = local_block_idx * block_size + token_in_block
    # rank-major flat index:
    #   owner * total_local_kv + (cu_offset_of_request_for_this_rank + local_pos)
    # The cu_offset is the same value for every rank because all ranks pad
    # to identical L_r per request.
    restore = (owner * total_local_kv + cu_offsets + local_pos_in_req).contiguous()
    assert int(restore.numel()) == total_real, (
        f"restore size {int(restore.numel())} != sum(per_req_total_kv_lens) "
        f"{total_real}"
    )
    return restore


def cp_padded_local_kv_lens(
    per_req_total_kv_lens: torch.Tensor, cp_size: int, block_size: int
) -> torch.Tensor:
    """Uniform per-rank local KV lengths for CP all_gather.

    For request length ``T_r``, each rank contributes
    ``ceil(T_r / (block_size * cp_size)) * block_size`` rows so NCCL
    ``all_gather`` sees identical shapes. Rows past the actual owned count are
    padding and must be initialized by the caller before gather.
    """
    if cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    virtual_block_size = block_size * cp_size
    lens = per_req_total_kv_lens.to(torch.int64)
    n_virtual_blocks = (lens + virtual_block_size - 1) // virtual_block_size
    return n_virtual_blocks * block_size


def cp_actual_owned_kv_lens(
    per_req_total_kv_lens: torch.Tensor,
    cp_size: int,
    block_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """Actual KV-token count written by ``cp_rank`` for each request.

    Ownership is block round-robin: logical block ``b`` belongs to
    ``b % cp_size``. The last owned block may be partial, so this can be
    smaller than :func:`cp_padded_local_kv_lens`.
    """
    if cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank({cp_rank}) out of range [0, {cp_size})")

    T = per_req_total_kv_lens.to(torch.int64)
    total_blocks = (T + block_size - 1) // block_size
    raw = total_blocks - cp_rank
    n_owned = torch.where(
        raw > 0,
        (raw + cp_size - 1) // cp_size,
        torch.zeros_like(raw),
    )
    last_blk_idx = (n_owned - 1).clamp_min(0) * cp_size + cp_rank
    last_blk_size = torch.minimum(
        torch.full_like(T, block_size),
        (T - last_blk_idx * block_size).clamp_min(0),
    )
    return torch.where(
        n_owned > 0,
        (n_owned - 1) * block_size + last_blk_size,
        torch.zeros_like(T),
    )


def cp_should_gather(cp_ctx: Optional[CPContext], start_pos: int) -> bool:
    """Prefill-only gate: gather runs iff a CPContext is bound.

    ``start_pos`` is intentionally ignored: CP continuation prefill also
    receives only the rank-local suffix, so compressor/indexer state must still
    all-gather the current input before pooling.
    """
    return cp_ctx is not None and cp_ctx.cp_size > 1
