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

from dataclasses import dataclass
from typing import Optional

import torch

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
    # [chunk_length] int64 — local idx i -> global position in
    # [0, padded_seq_len).  For padding local idxs, the global position
    # is ≥ seq_len_full; their attention output is discarded by the
    # framework's strip-pad gather.
    global_positions: torch.Tensor
    # [chunk_length] bool — True if local idx maps to a real token
    # (padding_mask[global_pos] == 1), False for padding slots.
    local_is_real: torch.Tensor
    # [seq_len_full] int64 — gathered-flat index for each real token in
    # GLOBAL order. ``gathered.index_select(0, unpad_restore)`` yields the
    # full un-padded sequence.
    unpad_restore: torch.Tensor
    # Raw cp_info, kept for any caller needing extra fields.
    cp_info: object


def build_cp_context(
    cp_info, cp_size: int, cp_rank: int, chunk_length: int, device: torch.device,
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
    assert pair_size * 2 == chunk_length, (
        f"chunk_length({chunk_length}) must be even for zigzag CP"
    )

    # Formula-based global positions (matches ZigZagProcessor::plan).
    arange_pair = torch.arange(pair_size, dtype=torch.long, device=device)
    even_positions = cp_rank * pair_size + arange_pair
    odd_positions = padded_seq_len - (cp_rank + 1) * pair_size + arange_pair
    global_positions = torch.cat([even_positions, odd_positions])  # [chunk_length]

    local_is_real = padding_mask[global_positions] == 1            # [chunk_length] bool
    unpad_restore = restore_indices[padding_mask == 1].to(torch.long)  # [seq_len_full]
    seq_len_full = int(unpad_restore.shape[0])

    return CPContext(
        cp_size=int(cp_size),
        cp_rank=int(cp_rank),
        chunk_length=int(chunk_length),
        padded_seq_len=padded_seq_len,
        seq_len_full=seq_len_full,
        global_positions=global_positions,
        local_is_real=local_is_real,
        unpad_restore=unpad_restore,
        cp_info=cp_info,
    )


def cp_all_gather_full(
    local: torch.Tensor, cp_ctx: CPContext,
) -> torch.Tensor:
    """All-gather a rank-local ``[B, chunk_length, *F]`` tensor across the
    CP (== TP) group and strip padding → ``[B, seq_len_full, *F]`` in
    GLOBAL logical order.  ``B`` must be 1 (V4 invariant)."""
    assert local.dim() >= 2
    B = local.size(0)
    assert B == 1
    assert local.size(1) == cp_ctx.chunk_length, (
        f"local.size(1)={local.size(1)} != chunk_length={cp_ctx.chunk_length}"
    )
    trailing = local.shape[2:]

    # collective_torch.all_gather concatenates along dim 0 for 1-D/2-D
    # tensors.  Flatten trailing dims so the helper sees a simple 2-D.
    local_flat = local.reshape(cp_ctx.chunk_length, -1).contiguous()
    gathered = all_gather(local_flat, group=Group.TP)
    # gathered: [cp_size * chunk_length, prod(F)]
    full = gathered.index_select(0, cp_ctx.unpad_restore)   # [seq_len_full, prod(F)]
    return full.view((1, cp_ctx.seq_len_full) + trailing)


def cp_freqs_cis_local(
    freqs_cis: torch.Tensor, cp_ctx: CPContext,
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
    """Deprecated: use ``build_cp_context`` + ``cp_all_gather_full``."""
    device = local.device
    chunk_length = local.size(1)
    ctx = build_cp_context(cp_info, cp_size, cp_rank, chunk_length, device)
    return cp_all_gather_full(local, ctx)


def cp_should_gather(cp_ctx: Optional[CPContext], start_pos: int) -> bool:
    """Prefill-only gate: gather runs iff a CPContext is bound, cp_size > 1,
    and we're at start_pos == 0 (prefill)."""
    return cp_ctx is not None and cp_ctx.cp_size > 1 and start_pos == 0
