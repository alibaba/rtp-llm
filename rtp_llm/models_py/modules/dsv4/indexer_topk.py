"""DSV4 Indexer TopK backend selection.

The DSV4 indexer score kernel already fuses score generation; the remaining
hot path is the row-wise ``torch.topk`` plus mask/copy post-processing.  vLLM
and SGLang avoid this for DeepSeek sparse attention with custom row-wise TopK
kernels.  TileLang HISA is different: it is an algorithmic replacement for the
indexer, not an exact TopK kernel, so it stays behind an explicit experimental
backend.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import torch

_FAST_TOPK_VALUES = (2048,)
_PERSISTENT_TOPK_VALUES = (512, 1024, 2048)


def _backend_name() -> str:
    return os.environ.get("DSV4_INDEXER_TOPK_BACKEND", "auto").strip().lower()


def _flatten_score(score: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = tuple(score.shape)
    if score.dim() == 2:
        return score.contiguous(), shape
    if score.dim() == 3:
        return score.reshape(-1, score.shape[-1]).contiguous(), shape
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
    rows = score.numel() // score.shape[-1]
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
            idx = flat_for_topk.topk(k_eff, dim=-1).indices.to(torch.int32)
            idx = torch.where(
                idx < lengths_i32.view(-1, 1), idx, torch.full_like(idx, -1)
            )
            out[:, :k_eff].copy_(idx)
        out = _apply_offset(out, offset)
        return _reshape_indices(out, shape)


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
        persistent_topk(flat, lengths_i32, out, workspace, int(topk), int(flat.shape[1]))
        out = _apply_offset(out, offset)
        return _reshape_indices(out, shape)


class HisaIndexerTopKBackend(IndexerTopKBackend):
    """Experimental TileLang HISA branch.

    HISA changes the indexer algorithm by hierarchical sparse selection; it is
    not an exact replacement for DSV4 ``score.topk``.  Keep it explicit so no
    production path silently changes sparse attention recall.
    """

    name = "hisa"

    def select(
        self,
        score: torch.Tensor,
        topk: int,
        *,
        lengths: Optional[torch.Tensor] = None,
        offset: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "DSV4_INDEXER_TOPK_BACKEND=hisa is experimental and not implemented"
        )


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
                return self._persistent.select(score, topk, lengths=lengths, offset=offset)
            except (ImportError, AttributeError):
                pass
        return self._torch.select(score, topk, lengths=lengths, offset=offset)


def get_indexer_topk_backend() -> IndexerTopKBackend:
    name = _backend_name()
    if name == "torch":
        return TorchIndexerTopKBackend()
    if name == "fast":
        return FastIndexerTopKBackend()
    if name == "persistent":
        return PersistentIndexerTopKBackend()
    if name == "hisa":
        return HisaIndexerTopKBackend()
    if name != "auto":
        raise ValueError(
            "invalid DSV4_INDEXER_TOPK_BACKEND="
            f"{name!r}; expected auto|torch|fast|persistent|hisa"
        )
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
