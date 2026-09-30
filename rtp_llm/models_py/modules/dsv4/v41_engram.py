"""Bounded Engram lookup, group-32 projection and per-lane residual update."""

import math

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv4.v41_engram_host import HostEngramEmbedding
from rtp_llm.models_py.modules.dsv4.v41_engram_inject import gated_engram_residual
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear


class V41Engram(nn.Module):
    """Consume one layer's explicit hashes without owning request history.

    The worker owns table registration/teardown; this module keeps a reference.
    Hashing runs once per model forward, and the caller selects this layer's IDs.
    Weights are immutable and on their final device, like V41MXFP8Linear weights.
    Lookup and WKV temporaries contain at most chunk_tokens rows. Graph capture
    is rejected until the model/worker integrates table and graph lifetimes.
    """

    def __init__(
        self,
        embedding: HostEngramEmbedding,
        projection: V41MXFP8Linear,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        *,
        eps: float,
        chunk_tokens: int = 128,
    ):
        super().__init__()
        if not isinstance(embedding, HostEngramEmbedding) or not isinstance(
            projection, V41MXFP8Linear
        ):
            raise ValueError(
                "V4.1 Engram requires a host embedding and group-32 projection"
            )
        if type(chunk_tokens) is not int or chunk_tokens <= 0:
            raise ValueError("Engram chunk_tokens must be a positive integer")
        if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("Engram norm epsilon must be finite and positive")
        if (
            q_weight.ndim != 2
            or min(q_weight.shape) <= 0
            or k_weight.shape != q_weight.shape
            or any(
                t.dtype != torch.bfloat16
                or t.device != projection.weight.device
                or not t.is_contiguous()
                for t in (q_weight, k_weight)
            )
        ):
            raise ValueError(
                "Engram gate weights must be contiguous BF16 [lanes,dim] on the projection device"
            )
        self.lanes, self.dim = q_weight.shape
        table = embedding._table
        table_dim = table.source.weight.shape[1]
        if projection.K % table_dim or projection.N != (self.lanes + 1) * self.dim:
            raise ValueError("Engram lookup, WKV and residual geometry disagree")
        self.heads = projection.K // table_dim
        self.backend = "eager" if projection.backend == "eager" else "triton"
        if self.backend == "triton" and (
            self.heads,
            table_dim,
            self.lanes,
            self.dim,
        ) != (24, 256, 4, 5120):
            raise ValueError(
                "V4.1 Engram requires 24x256 lookup and 4x5120 residual geometry"
            )
        self.layer_id = table.source.layer_id
        self.embedding, self.projection = embedding, projection
        self.q_weight, self.k_weight = q_weight, k_weight
        self.eps, self.chunk_tokens = eps, chunk_tokens

    def _validate(self, hidden, hash_ids, valid_rows, out):
        if (
            hidden.ndim != 3
            or hidden.shape[1:] != (self.lanes, self.dim)
            or hidden.dtype != torch.bfloat16
            or not hidden.is_contiguous()
            or hidden.device != self.projection.weight.device
        ):
            raise ValueError(
                "Engram residual must be contiguous BF16 [tokens,lanes,dim] on the weight device"
            )
        tokens = hidden.shape[0]
        if (
            hash_ids.shape != (tokens, self.heads)
            or hash_ids.dtype not in (torch.int32, torch.int64)
            or hash_ids.device != hidden.device
            or valid_rows.shape != (tokens,)
            or valid_rows.dtype != torch.bool
            or not valid_rows.is_contiguous()
            or valid_rows.device != hidden.device
        ):
            raise ValueError(
                "Engram requires layer-local integer hashes and a contiguous device bool token mask"
            )
        if torch.is_grad_enabled() and any(
            t.requires_grad
            for t in (hidden, self.q_weight, self.k_weight, self.projection.weight)
        ):
            raise ValueError("V4.1 Engram is inference-only")
        if out is None:
            return
        if (
            out.shape != hidden.shape
            or out.dtype != hidden.dtype
            or out.device != hidden.device
            or not out.is_contiguous()
            or out.requires_grad
        ):
            raise ValueError(
                "Engram output must match the residual and be contiguous without gradients"
            )
        if not tokens:
            return
        storage = out.untyped_storage().data_ptr()
        if (
            storage == hidden.untyped_storage().data_ptr()
            and out.data_ptr() != hidden.data_ptr()
        ):
            raise ValueError("Engram output alias with hidden must be exact")
        protected = (
            hash_ids,
            valid_rows,
            self.q_weight,
            self.k_weight,
            self.projection.weight,
            self.projection.scales,
            self.projection._packed_scales,
            self.embedding._table._storage,
        )
        if any(
            t is not None and storage == t.untyped_storage().data_ptr()
            for t in protected
        ):
            raise ValueError(
                "Engram output must not share storage with hashes, masks, weights or host tables"
            )

    def forward(self, hidden, hash_ids, valid_rows, *, out=None):
        """Apply a layer to [T,lanes,dim]; exact hidden/output alias is allowed.

        Metadata errors fail before any writes. A provider error propagates and
        may leave completed chunks in caller output; callers must abort the round.
        """
        table = self.embedding._table
        # Exclude teardown between chunks; close() drains queued CUDA work later.
        with table._mutex:
            table._check_open()
            self._validate(hidden, hash_ids, valid_rows, out)
            if self.backend == "triton":
                if table._runtime is None or table._device != hidden.device:
                    raise ValueError(
                        "Engram table must be registered on the residual device"
                    )
                with torch.cuda.device(hidden.device):
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "Engram composition graph capture requires integrated worker lifecycle"
                        )
            output = torch.empty_like(hidden) if out is None else out
            for start in range(0, hidden.shape[0], self.chunk_tokens):
                end = min(start + self.chunk_tokens, hidden.shape[0])
                rows = slice(start, end)
                values = self.embedding(
                    hash_ids[rows], valid_rows[rows], backend=self.backend
                )
                projected = self.projection(values.flatten(1))
                gated_engram_residual(
                    hidden[rows],
                    projected,
                    self.q_weight,
                    self.k_weight,
                    valid_rows[rows],
                    eps=self.eps,
                    backend=self.backend,
                    out=output[rows],
                )
                del values, projected
            return output
