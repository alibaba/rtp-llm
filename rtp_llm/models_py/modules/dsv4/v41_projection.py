"""V4.1 projection, normalization and cache publication in reference order.

Inputs are flattened, post-attention-norm BF16 token rows. Positions, physical
slots and pair predecessors belong to the scheduler, including CP exchange.
Weights/frequencies remain immutable on their final device. The eager backend
is a CPU oracle; SM120 execution uses the group-32 FlashInfer projection.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rtp_llm.config.dsv41_config import V41LayerSources
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb_batched
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache, _shares_storage
from rtp_llm.models_py.modules.dsv4.v41_compressor import (
    V41CompressionFrame,
    V41PairMetadata,
    V41PairState,
    prepare_ratio2,
)
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear


def _rms_norm(x, weight):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-20)
    return (weight * value).to(x.dtype)


def _rotate(x, frequencies, *, inverse=False):
    # The existing helper takes a request axis and a singleton sequence axis.
    result = x.clone()
    apply_rotary_emb_batched(
        result[..., -64:].unsqueeze(1), frequencies, inverse=inverse
    )
    return result


def _tensor(value, shape, dtype, device, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or value.dtype != dtype
        or value.device != device
        or not value.is_contiguous()
        or value.requires_grad
    ):
        raise ValueError(f"V4.1 {name} shape/dtype/device/stride mismatch")
    if device.type == "cpu" and not bool(torch.isfinite(value).all()):
        raise ValueError(f"V4.1 {name} must be finite")


def _inputs(x, positions, valid, projection, frequencies, max_rows):
    rows = x.shape[0] if x.ndim == 2 else -1
    if rows < 0 or rows > max_rows:
        raise ValueError("V4.1 projection exceeds its token row bound")
    device = projection.weight.device
    # Dead rows can contain NaN; sanitize them before any GEMM or normalization.
    for value, shape, dtypes in (
        (x, (rows, projection.K), (torch.bfloat16,)),
        (positions, (rows,), (torch.int32, torch.int64)),
        (valid, (rows,), (torch.bool,)),
    ):
        if (
            value.shape != shape
            or value.dtype not in dtypes
            or value.device != device
            or not value.is_contiguous()
            or value.requires_grad
        ):
            raise ValueError("V4.1 projection input shape/dtype/device/stride mismatch")
    if device.type == "cpu":
        if not bool(torch.isfinite(x[valid]).all()) or bool(
            ((positions[valid] < 0) | (positions[valid] >= len(frequencies))).any()
        ):
            raise ValueError("V4.1 live rows must be finite with valid RoPE positions")
    return torch.where(valid[:, None], x, 0), torch.where(valid, positions, 0).long()


@dataclass(frozen=True)
class V41KVInputs:
    """Projected token rows before cross-rank pairing and cache publication."""

    owner: object
    swa: torch.Tensor
    values: torch.Tensor | None
    scores: torch.Tensor | None
    positions: torch.Tensor
    valid_rows: torch.Tensor


@dataclass(frozen=True)
class V41KVFrame:
    """Owned projected rows; preparing a frame does not modify any cache.

    Publish once on the preparation stream. Scheduler readiness must advance
    only after publish succeeds. A device/store failure can leave partial writes
    and is terminal for the forward; this is not a rollback transaction.
    """

    swa: torch.Tensor
    global_kv: torch.Tensor | None
    index_k: torch.Tensor | None
    valid_rows: torch.Tensor
    complete: torch.Tensor
    global_tag: str | None
    pair: V41CompressionFrame | None
    backend: str
    stream_id: int | None

    def publish(
        self,
        swa_cache: V41PagedCache,
        swa_slots: torch.Tensor,
        *,
        global_cache: V41PagedCache | None = None,
        global_slots: torch.Tensor | None = None,
        index_cache: V41PagedCache | None = None,
        index_slots: torch.Tensor | None = None,
    ):
        if (
            self.stream_id is not None
            and torch.cuda.current_stream(self.swa.device).cuda_stream != self.stream_id
        ):
            raise RuntimeError("V4.1 publication must use its preparation stream")
        destinations = [(swa_cache, swa_slots, self.swa, "swa_kv", self.valid_rows)]
        if self.global_tag is None:
            if any(
                item is not None
                for item in (global_cache, global_slots, index_cache, index_slots)
            ):
                raise ValueError("V4.1 non-owner cannot publish GLOBAL/index caches")
        else:
            destinations.extend(
                (
                    (
                        global_cache,
                        global_slots,
                        self.global_kv,
                        self.global_tag,
                        self.complete,
                    ),
                    (
                        index_cache,
                        index_slots,
                        self.index_k,
                        "indexer_kv",
                        self.complete,
                    ),
                )
            )
        writes = []
        for cache, slots, values, tag, active in destinations:
            if (
                not isinstance(cache, V41PagedCache)
                or cache.tag != tag
                or slots is None
            ):
                raise ValueError("V4.1 cache publication destination/tag mismatch")
            if slots.shape != active.shape:
                raise ValueError("V4.1 publication slots must match token rows")
            if (
                slots.dtype not in (torch.int32, torch.int64)
                or slots.device != cache.raw.device
                or not slots.is_contiguous()
            ):
                raise ValueError(
                    "V4.1 publication slots must be contiguous device integers"
                )
            selected = torch.where(active, slots, -1)
            cache._check_slots(selected, self.backend, writing=True)
            cache._check_rows(values, selected, values.shape[-1], floating=True)
            if self.backend == "eager":
                if not bool(torch.isfinite(values[active]).all()):
                    raise ValueError("V4.1 published live values must be finite")
                if tag != "swa_kv" and bool((selected[active] < 0).any()):
                    raise ValueError(
                        "V4.1 every completed latent needs GLOBAL/index slots"
                    )
            writes.append((cache, selected, values))
        # Validate all destinations and aliases before the first cache mutation.
        protected = [
            self.swa,
            self.global_kv,
            self.index_k,
            self.valid_rows,
            self.complete,
        ]
        protected.extend(slots for _, slots, _, _, _ in destinations)
        if self.pair is not None:
            protected.extend(
                (self.pair.state.raw, self.pair.pending_rows, self.pair.write_slots)
            )
        for i, (cache, _, _) in enumerate(writes):
            if any(
                _shares_storage(cache.raw, other.raw) for other, _, _ in writes[:i]
            ) or any(
                value is not None and _shares_storage(cache.raw, value)
                for value in protected
            ):
                raise ValueError(
                    "V4.1 publication caches must not alias inputs or each other"
                )
        for cache, slots, values in writes:
            cache.write(values, slots, backend=self.backend)
        if self.pair is not None:
            self.pair.commit()


class V41KVProjection:
    """SWA for every layer; compressed/index keys only for their source layer.

    Ratio-2 BF16 checkpoint weights are promoted to FP32 once at construction,
    matching the reference compressor. TF32 must be disabled for those GEMMs.
    GLOBAL/index physical slots are separate, especially at ratio 2 where index
    pages have twice as many physical entries. SWA slots may skip earlier rows
    when the caller publishes a final ring after attending to a chunk scratch.
    """

    def __init__(
        self,
        sources: V41LayerSources,
        wkv: V41MXFP8Linear,
        kv_norm: torch.Tensor,
        frequencies: torch.Tensor,
        *,
        compressor_weight=None,
        compressor_norm=None,
        gate_weight=None,
        index_weight=None,
        index_norm=None,
        max_rows=128,
    ):
        if (
            not isinstance(wkv, V41MXFP8Linear)
            or wkv.N != 512
            or sources.compress_ratio not in (0, 1, 2)
            or (sources.owns_kv and not sources.compress_ratio)
            or type(max_rows) is not int
            or not 1 <= max_rows <= 128
        ):
            raise ValueError(
                "V4.1 KV projection requires admitted geometry and row bound"
            )
        self.sources, self.wkv, self.max_rows = sources, wkv, max_rows
        device = wkv.weight.device
        _tensor(kv_norm, (512,), torch.bfloat16, device, "KV norm")
        if frequencies.ndim != 2 or not 1 <= frequencies.shape[0] <= 32768:
            raise ValueError("V4.1 RoPE table exceeds the bounded 32K profile")
        _tensor(frequencies, (len(frequencies), 32), torch.complex64, device, "RoPE")
        self.kv_norm, self.frequencies = kv_norm, frequencies
        optional = (compressor_weight, compressor_norm, index_weight, index_norm)
        if sources.owns_kv:
            for value, shape, name in (
                (compressor_weight, (512, wkv.K), "compressor weight"),
                (compressor_norm, (512,), "compressor norm"),
                (index_weight, (128, 512), "index key weight"),
                (index_norm, (128,), "index key norm"),
            ):
                _tensor(value, shape, torch.bfloat16, device, name)
            if sources.compress_ratio == 2:
                _tensor(
                    gate_weight, (512, wkv.K), torch.bfloat16, device, "compressor gate"
                )
                compressor_weight, gate_weight = (
                    compressor_weight.float(),
                    gate_weight.float(),
                )
            elif gate_weight is not None:
                raise ValueError("V4.1 ratio-1 projection has no compression gate")
        elif any(value is not None for value in (*optional, gate_weight)):
            raise ValueError(
                "V4.1 non-owner must not load compressed/index key weights"
            )
        self.compressor_weight, self.compressor_norm = (
            compressor_weight,
            compressor_norm,
        )
        self.gate_weight, self.index_weight, self.index_norm = (
            gate_weight,
            index_weight,
            index_norm,
        )
        self.backend = "eager" if wkv.backend == "eager" else "triton"

    def project(self, hidden, positions, valid_rows) -> V41KVInputs:
        """Project local rows; an explicit CP exchange may gather these outputs."""
        x, safe_positions = _inputs(
            hidden, positions, valid_rows, self.wkv, self.frequencies, self.max_rows
        )
        values, scores = None, None
        if self.sources.owns_kv:
            if self.sources.compress_ratio == 2:
                if hidden.is_cuda and torch.backends.cuda.matmul.allow_tf32:
                    raise RuntimeError(
                        "V4.1 FP32 compressor projections require TF32 disabled"
                    )
                values = F.linear(x.float(), self.compressor_weight)
                scores = F.linear(x.float(), self.gate_weight)
            else:
                values = F.linear(x, self.compressor_weight)
        swa = _rotate(
            _rms_norm(self.wkv(x), self.kv_norm), self.frequencies[safe_positions]
        )
        return V41KVInputs(self, swa, values, scores, positions, valid_rows)

    def finish(
        self,
        projected: V41KVInputs,
        *,
        pair_state: V41PairState | None = None,
        pair_metadata: V41PairMetadata | None = None,
    ) -> V41KVFrame:
        """Pool complete visible rows and prepare writes, without mutating caches."""
        if not isinstance(projected, V41KVInputs) or projected.owner is not self:
            raise ValueError("V4.1 KV inputs belong to a different projection")
        positions, valid_rows = projected.positions, projected.valid_rows
        device = self.wkv.weight.device
        rows = valid_rows.numel()
        if rows > self.max_rows or positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("V4.1 projected KV rows exceed their admitted geometry")
        _tensor(positions, (rows,), positions.dtype, device, "projected positions")
        _tensor(valid_rows, (rows,), torch.bool, device, "projected row mask")
        _tensor(projected.swa, (rows, 512), torch.bfloat16, device, "projected SWA")
        if self.sources.owns_kv:
            dtype = (
                torch.float32 if self.sources.compress_ratio == 2 else torch.bfloat16
            )
            _tensor(
                projected.values, (rows, 512), dtype, device, "projected compressor"
            )
        elif projected.values is not None:
            raise ValueError("V4.1 non-owner cannot carry compressor projections")
        if self.sources.owns_kv and self.sources.compress_ratio == 2:
            _tensor(
                projected.scores, (rows, 512), torch.float32, device, "projected gates"
            )
        elif projected.scores is not None:
            raise ValueError("V4.1 compressor gates belong only to ratio-2 sources")
        safe_positions = torch.where(valid_rows, positions, 0).long()
        pair = None
        latent = None
        needs_pair = self.sources.owns_kv and self.sources.compress_ratio == 2
        if needs_pair:
            if pair_state is None or pair_metadata is None:
                raise ValueError(
                    "V4.1 ratio-2 projection requires explicit pair state/metadata"
                )
            # One metadata object must describe both projection and predecessor rows.
            if (
                pair_metadata.positions is not positions
                or pair_metadata.valid_rows is not valid_rows
            ):
                raise ValueError(
                    "V4.1 pair positions/mask must be the projection metadata"
                )
            pair = prepare_ratio2(
                projected.values,
                projected.scores,
                self.compressor_norm,
                pair_state,
                pair_metadata,
                backend=self.backend,
            )
            latent, complete = pair.latent, pair.complete
        else:
            if pair_state is not None or pair_metadata is not None:
                raise ValueError("V4.1 pair state belongs only to ratio-2 KV owners")
            complete = (
                valid_rows.clone()
                if self.sources.owns_kv
                else torch.zeros_like(valid_rows)
            )
            if self.sources.owns_kv:
                latent = _rms_norm(projected.values, self.compressor_norm)
        global_kv, index_k, tag = None, None, None
        if latent is not None:
            group_positions = torch.where(
                complete, safe_positions + 1 - self.sources.compress_ratio, 0
            )
            group_freqs = self.frequencies[group_positions]
            # Index keys consume the pre-RoPE latent. GLOBAL gets its own rotation.
            index_k = _rotate(
                _rms_norm(F.linear(latent, self.index_weight), self.index_norm),
                group_freqs,
            )
            global_kv = _rotate(latent, group_freqs)
            index_k = torch.where(complete[:, None], index_k, 0)
            global_kv = torch.where(complete[:, None], global_kv, 0)
            tag = "csa_kv" if self.sources.compress_ratio == 2 else "hca_kv"
        return V41KVFrame(
            projected.swa,
            global_kv,
            index_k,
            valid_rows.clone(),
            complete,
            tag,
            pair,
            self.backend,
            (
                torch.cuda.current_stream(device).cuda_stream
                if device.type == "cuda"
                else None
            ),
        )

    def prepare(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        valid_rows: torch.Tensor,
        *,
        pair_state: V41PairState | None = None,
        pair_metadata: V41PairMetadata | None = None,
        exchange=None,
    ) -> V41KVFrame:
        projected = self.project(hidden, positions, valid_rows)
        if exchange is not None:
            projected = exchange(projected)
        return self.finish(
            projected, pair_state=pair_state, pair_metadata=pair_metadata
        )
