"""DSV4 Indexer TopK backend selection.

The DSV4 indexer score kernel already fuses score generation; the remaining
hot path is the row-wise ``torch.topk`` plus mask/copy post-processing.  vLLM
and SGLang avoid this for DeepSeek sparse attention with custom row-wise TopK
kernels. Selection uses the automatic exact backend; model-specific bounded
selection is exposed through separate candidate APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

_FAST_TOPK_VALUES = (2048,)
_PERSISTENT_TOPK_VALUES = (512, 1024, 2048)


def native_exact_topk_enabled():
    return True


def native_candidate_selection_supported(columns, topk, *, prefix="block"):
    """Shared selector/prepare policy; a running merge selects 2K down to K."""
    return topk in (512, 1024, 2048) and columns >= 0


def native_page_rr_selection_supported(owner_width, topk, *, shards, page, start):
    """Use one static policy for Decode preparation and CUDA Graph replay."""
    columns = owner_width * shards
    # At most K visible columns use the same ordered copy in both selectors.
    # For longer spans, radix selection preserves exact members but its emission
    # order can change floating-point accumulation in FP8 sparse attention.
    return (
        0 < shards <= 256
        and (shards & (shards - 1)) == 0
        and page >= 4
        and (page & (page - 1)) == 0
        and owner_width > 0
        and start >= 0
        and owner_width % page == start % page == 0
        and (start + owner_width) * shards < 2**31
        and native_candidate_selection_supported(columns, topk)
        and columns <= topk
    )


def _flatten_score(score: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = tuple(score.shape)
    if score.dim() == 2:
        return score.contiguous(), shape
    if score.dim() == 3:
        rows = score.shape[0] * score.shape[1]
        return score.reshape(rows, score.shape[-1]).contiguous(), shape
    raise ValueError(f"score must be [rows,T] or [B,S,T], got {shape}")


def _reshape_indices(
    indices: torch.Tensor, score_shape: tuple[int, ...]
) -> torch.Tensor:
    if len(score_shape) == 2:
        return indices
    return indices.view(score_shape[0], score_shape[1], indices.shape[-1])


def _normalize_lengths(
    score: torch.Tensor,
    topk: int,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    rows = 1
    for dim in score.shape[:-1]:
        rows *= dim
    if lengths is None:
        lengths = torch.full(
            (rows,), score.shape[-1], device=score.device, dtype=torch.int32
        )
    else:
        lengths = lengths.to(device=score.device, dtype=torch.int32).reshape(-1)
    if lengths.numel() != rows:
        raise ValueError(
            f"lengths rows mismatch: expected {rows}, got {lengths.numel()}"
        )
    return lengths.clamp_(min=0, max=score.shape[-1])


def _apply_offset(indices: torch.Tensor, offset: int | torch.Tensor) -> torch.Tensor:
    if isinstance(offset, int) and offset == 0:
        return indices
    if isinstance(offset, torch.Tensor):
        off = offset.to(device=indices.device, dtype=torch.int32).reshape(-1, 1)
        flat = indices.reshape(-1, indices.shape[-1])
        return torch.where(flat >= 0, flat + off, flat).view_as(indices)
    return torch.where(indices >= 0, indices + int(offset), indices)


class IndexerTopKBackend(ABC):
    """Abstract row-wise TopK backend for DSV4 indexer scores.

    Implementations must return int32 indices, use ``-1`` for rows with fewer
    than ``topk`` valid entries, and preserve the current DSV4 semantics where
    offset is only added to non-negative indices.  Index order is not part of
    the contract; sparse attention consumes the selected set.
    """

    name: str

    @abstractmethod
    def select(
        self,
        score: torch.Tensor,
        topk: int,
        *,
        lengths: Optional[torch.Tensor] = None,
        offset: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        raise NotImplementedError


class TorchIndexerTopKBackend(IndexerTopKBackend):
    """Exact PyTorch backend matching the pre-optimization implementation."""

    name = "torch"

    def __init__(self, *, stable_ties: bool = False):
        self.stable_ties = stable_ties

    def select(
        self,
        score: torch.Tensor,
        topk: int,
        *,
        lengths: Optional[torch.Tensor] = None,
        offset: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        flat, shape = _flatten_score(score)
        lengths_i32 = _normalize_lengths(score, topk, lengths)
        k_eff = min(int(topk), flat.shape[-1])
        out = torch.full(
            (flat.shape[0], int(topk)),
            -1,
            dtype=torch.int32,
            device=flat.device,
        )
        if k_eff > 0:
            if lengths is not None:
                col = torch.arange(
                    flat.shape[-1], device=flat.device, dtype=torch.int32
                ).view(1, -1)
                flat_for_topk = torch.where(
                    col < lengths_i32.view(-1, 1),
                    flat,
                    torch.full_like(flat, float("-inf")),
                )
            else:
                flat_for_topk = flat
            if self.stable_ties:
                idx = flat_for_topk.argsort(dim=-1, descending=True, stable=True)[
                    :, :k_eff
                ].to(torch.int32)
            else:
                idx = flat_for_topk.topk(k_eff, dim=-1).indices.to(torch.int32)
            idx = torch.where(
                idx < lengths_i32.view(-1, 1), idx, torch.full_like(idx, -1)
            )
            out[:, :k_eff].copy_(idx)
        out = _apply_offset(out, offset)
        return _reshape_indices(out, shape)

    def select_candidates(
        self,
        score,
        positions,
        topk,
        *,
        workspace=None,
        prefix="block",
        out=None,
        scores_are_sanitized=False,
        starts=None,
        ends=None,
    ):
        """Direct selection of bounded score/global-position candidates.

        The unique integer key includes the tie rule before selection. This
        avoids a full stable argsort and remains exact across KV blocks and
        PageRR owners. ``workspace`` supplies reusable, phase-owned buffers.
        """
        if (
            score.ndim != 2
            or score.dtype != torch.float32
            or positions.dtype != torch.int32
            or positions.shape not in (score.shape, (score.shape[1],))
            or positions.device != score.device
            or topk <= 0
        ):
            raise ValueError(
                "TopK candidates require FP32 scores/INT32 global positions"
            )
        rows, columns = score.shape
        if starts is not None or ends is not None:
            col = torch.arange(columns, device=score.device).view(1, -1)
            valid = positions >= 0
            if starts is not None:
                valid = valid & (col >= starts.view(-1, 1))
            if ends is not None:
                valid = valid & (col < ends.view(-1, 1))
            score = torch.where(valid, score, -float("inf"))

        def buffer(name, shape, dtype):
            if workspace is not None:
                return workspace(prefix + "." + name, shape, dtype)
            return torch.empty(shape, dtype=dtype, device=score.device)

        output = (
            out
            if out is not None
            else (
                buffer("scores", (rows, topk), torch.float32),
                buffer("positions", (rows, topk), torch.int32),
            )
        )
        if (
            output[0].shape != (rows, topk)
            or output[1].shape != (rows, topk)
            or output[0].dtype != torch.float32
            or output[1].dtype != torch.int32
        ):
            raise ValueError("incorrect bounded TopK output buffers")
        selected = min(topk, columns)
        if selected == 0:
            output[0].fill_(-float("inf"))
            output[1].fill_(-1)
            return output
        keys = buffer("keys", (rows, columns), torch.int64)
        values = buffer("selected_keys", (rows, selected), torch.int64)
        slots = buffer("selected_slots", (rows, selected), torch.int64)
        if score.is_cuda:
            from rtp_llm.models_py.kernels.cuda.indexer_topk_keys import (
                pack_indexer_topk_keys,
                unpack_indexer_topk_keys,
            )

            pack_indexer_topk_keys(score, positions, keys)
            # Members use the complete score/global-position key. Consumers
            # need that set, not a sorted visit order for sparse attention.
            torch.topk(keys, selected, dim=-1, sorted=False, out=(values, slots))
            unpack_indexer_topk_keys(values, *output)
        else:
            # Small constructed references use the same key contract on CPU.
            bits = score.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            bits = torch.where(score == 0, 0, bits)
            ordered = torch.where(
                (bits & 0x80000000) != 0, bits ^ 0xFFFFFFFF, bits ^ 0x80000000
            )
            upper = ordered ^ 0x80000000
            upper = torch.where(upper >= 0x80000000, upper - 0x100000000, upper)
            packed = (upper << 32) | (0xFFFFFFFF - positions.to(torch.int64))
            valid = torch.isfinite(score) & (positions >= 0)
            keys.copy_(torch.where(valid, packed, torch.iinfo(torch.int64).min))
            torch.topk(keys, selected, dim=-1, sorted=True, out=(values, slots))
            ordered = ((values >> 32) & 0xFFFFFFFF) ^ 0x80000000
            bits = torch.where(
                (ordered & 0x80000000) != 0, ordered ^ 0x80000000, ordered ^ 0xFFFFFFFF
            )
            valid = values != torch.iinfo(torch.int64).min
            output[0].fill_(-float("inf"))
            output[1].fill_(-1)
            output[0][:, :selected].copy_(
                torch.where(
                    valid,
                    bits.to(torch.int32).contiguous().view(torch.float32),
                    -float("inf"),
                )
            )
            output[1][:, :selected].copy_(
                torch.where(
                    valid,
                    (0xFFFFFFFF - (values & 0xFFFFFFFF)).to(torch.int32),
                    -1,
                )
            )
        return output

    def merge_candidates(self, running, current, *, workspace=None):
        """Merge at most 2K candidates into the same FP32/INT32 running K."""
        rows, topk = running[0].shape
        if current[0].shape != (rows, topk) or current[1].shape != (rows, topk):
            raise ValueError("running/current TopK shapes differ")

        def buffer(name, dtype):
            if workspace is not None:
                return workspace("merge." + name, (rows, 2 * topk), dtype)
            return torch.empty((rows, 2 * topk), dtype=dtype, device=running[0].device)

        scores = buffer("input_scores", torch.float32)
        positions = buffer("input_positions", torch.int32)
        scores[:, :topk].copy_(running[0])
        scores[:, topk:].copy_(current[0])
        positions[:, :topk].copy_(running[1])
        positions[:, topk:].copy_(current[1])
        return self.select_candidates(
            scores, positions, topk, workspace=workspace, prefix="merge", out=running
        )


class NativeExactIndexerTopKBackend(TorchIndexerTopKBackend):
    """Bounded GLM radix selection with the complete global-position tie key.

    The input is read-only; bounds exclude unwritten MQA columns, while holes
    and non-finite values are handled during selection. Unordered pairs feed
    SFA directly or a real 2K-to-K running merge. No separate repair or sort.
    """

    name = "native_exact"

    def select_candidates(
        self,
        score,
        positions,
        topk,
        *,
        workspace=None,
        prefix="block",
        out=None,
        scores_are_sanitized=False,
        starts=None,
        ends=None,
    ):
        if not score.is_cuda or not native_candidate_selection_supported(
            score.shape[-1], topk, prefix=prefix
        ):
            return super().select_candidates(
                score,
                positions,
                topk,
                workspace=workspace,
                prefix=prefix,
                out=out,
                starts=starts,
                ends=ends,
            )
        if (
            score.ndim != 2
            or score.dtype != torch.float32
            or score.stride(1) != 1
            or positions.dtype != torch.int32
            or positions.stride(-1) != 1
            or positions.shape not in (score.shape, (score.shape[1],))
            or positions.device != score.device
        ):
            raise ValueError(
                "Native exact TopK requires FP32 scores/INT32 global positions"
            )
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        rows, columns = score.shape

        def buffer(name, shape, dtype):
            return (
                workspace(prefix + ".native." + name, shape, dtype)
                if workspace is not None
                else torch.empty(shape, device=score.device, dtype=dtype)
            )

        selected_scores, selected_positions = (
            out
            if out is not None
            else (
                buffer("scores", (rows, topk), torch.float32),
                buffer("positions", (rows, topk), torch.int32),
            )
        )
        if (
            selected_scores.shape != (rows, topk)
            or selected_positions.shape != (rows, topk)
            or selected_scores.dtype != torch.float32
            or selected_positions.dtype != torch.int32
        ):
            raise ValueError("incorrect bounded TopK output buffers")
        rtp_llm_ops.dsa_select_topk_candidates(
            score, positions, starts, ends, selected_scores, selected_positions, topk
        )
        return selected_scores, selected_positions

    def select_page_rr_candidates(
        self,
        received,
        query_positions,
        topk,
        *,
        workspace,
        out,
        shard_rank,
        shards,
        page,
        start,
    ):
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        shape = (received.size(0) // shards, topk)
        scores, positions = (
            out
            if out is not None
            else (
                workspace("block.native.scores", shape, torch.float32),
                workspace("block.native.positions", shape, torch.int32),
            )
        )
        rtp_llm_ops.dsa_select_page_rr_candidates(
            received,
            query_positions,
            scores,
            positions,
            topk,
            shards,
            shard_rank,
            page,
            start,
        )
        return scores, positions


class FastIndexerTopKBackend(IndexerTopKBackend):
    """SGLang/TileLang-style row-wise TopK backend.

    RTP already carries ``fast_topk_v2`` adapted from SGLang's
    ``sgl-kernel/csrc/elementwise/topk.cu`` and TileLang's DSv3.2 selector.
    That kernel remains a DeepSeek V3.2/TileLang-specialized topk=2048 path.
    """

    name = "fast"

    def select(
        self,
        score: torch.Tensor,
        topk: int,
        *,
        lengths: Optional[torch.Tensor] = None,
        offset: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        if not (
            score.is_cuda
            and score.dtype == torch.float32
            and int(topk) in _FAST_TOPK_VALUES
        ):
            raise RuntimeError(
                "DSV4 fast indexer TopK requires CUDA float32 scores and "
                "topk=2048; "
                f"got device={score.device}, dtype={score.dtype}, topk={int(topk)}"
            )
        flat, shape = _flatten_score(score)
        lengths_i32 = _normalize_lengths(score, topk, lengths)

        from rtp_llm.models_py.kernels.cuda.fast_topk import fast_topk_v2

        out = fast_topk_v2(flat, lengths_i32, int(topk))
        out = _apply_offset(out, offset)
        return _reshape_indices(out, shape)


class PersistentIndexerTopKBackend(IndexerTopKBackend):
    """Persistent TopK branch for long decode rows.

    vLLM uses a persistent TopK kernel for DeepSeek sparse indexer rows when
    ``topk`` is 512/1024/2048. This backend calls RTP's port of that kernel.
    """

    name = "persistent"

    def select(
        self,
        score: torch.Tensor,
        topk: int,
        *,
        lengths: Optional[torch.Tensor] = None,
        offset: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        if not (
            score.is_cuda
            and score.dtype == torch.float32
            and int(topk) in _PERSISTENT_TOPK_VALUES
        ):
            raise RuntimeError(
                "DSV4 persistent indexer TopK requires CUDA float32 scores and "
                "topk in {512, 1024, 2048}; "
                f"got device={score.device}, dtype={score.dtype}, topk={int(topk)}"
            )
        flat, shape = _flatten_score(score)
        lengths_i32 = _normalize_lengths(score, topk, lengths)

        from rtp_llm.ops.compute_ops import rtp_llm_ops

        persistent_topk = getattr(rtp_llm_ops, "persistent_topk")
        out = torch.empty(
            (flat.shape[0], int(topk)), dtype=torch.int32, device=flat.device
        )
        workspace = torch.empty((1 << 20,), dtype=torch.uint8, device=flat.device)
        persistent_topk(
            flat, lengths_i32, out, workspace, int(topk), int(flat.shape[1])
        )
        out = _apply_offset(out, offset)
        return _reshape_indices(out, shape)


class AutoIndexerTopKBackend(IndexerTopKBackend):
    """Use the fastest exact backend available for the current shape."""

    name = "auto"

    def __init__(self) -> None:
        self._fast = FastIndexerTopKBackend()
        self._persistent = PersistentIndexerTopKBackend()
        self._torch = TorchIndexerTopKBackend()

    def select(
        self,
        score: torch.Tensor,
        topk: int,
        *,
        lengths: Optional[torch.Tensor] = None,
        offset: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        if int(topk) <= 0 or score.shape[-1] == 0:
            return self._torch.select(score, topk, lengths=lengths, offset=offset)
        # The fused TopK kernels are only used for full-row selection. Decode
        # can request topk=512 while only a short compressed prefix is valid;
        # keep that masked case on the exact torch path.
        if lengths is not None:
            return self._torch.select(score, topk, lengths=lengths, offset=offset)
        if (
            score.is_cuda
            and score.dtype == torch.float32
            and int(topk) in _FAST_TOPK_VALUES
        ):
            try:
                return self._fast.select(score, topk, lengths=lengths, offset=offset)
            except (ImportError, AttributeError):
                pass
        if (
            score.is_cuda
            and score.dtype == torch.float32
            and int(topk) in _PERSISTENT_TOPK_VALUES
        ):
            try:
                return self._persistent.select(
                    score, topk, lengths=lengths, offset=offset
                )
            except (ImportError, AttributeError):
                pass
        return self._torch.select(score, topk, lengths=lengths, offset=offset)


def get_indexer_topk_backend() -> IndexerTopKBackend:
    return AutoIndexerTopKBackend()


def select_indexer_topk(
    score: torch.Tensor,
    topk: int,
    *,
    lengths: Optional[torch.Tensor] = None,
    offset: int | torch.Tensor = 0,
) -> torch.Tensor:
    return get_indexer_topk_backend().select(
        score, topk, lengths=lengths, offset=offset
    )
