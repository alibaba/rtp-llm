"""Fixed-width owner compression and ratio2 rollback for decode CUDA graphs.

The caller owns request slots, canonical page mapping and graph lifetime. Call
``prepare`` before warmup/capture/replay, ``forward`` and ``store`` inside the
target graph, then ``select_pair`` with the actual materialized row counts.
Ordinary gamma5 verification uses six rows and seven complete pair snapshots.
This component does not enable model Graph execution or publish cache state.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from rtp_llm.models_py.modules.dsv41.cache_layout import CacheRegion
from rtp_llm.models_py.modules.dsv41.compact_writer import write_compact
from rtp_llm.models_py.modules.dsv41.compressor import (
    CompressorRoPE,
    OwnerCompressor,
    _rotate,
    is_supported,
)
from rtp_llm.models_py.modules.dsv41.math import rms_norm
from torch import nn

PAIR_SNAPSHOT_BYTES = 4112


def _tensor(value, shape, dtype, device, name):
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != tuple(shape)
        or value.dtype != dtype
        or value.device != device
        or not value.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous {dtype} {shape} on {device}")


@dataclass(frozen=True)
class V41DecodePairState:
    """Engine byte ABI, with an optional snapshot axis before the last axis."""

    storage: torch.Tensor

    @property
    def partial_kv(self):
        return self.storage[..., :2048].view(torch.float32)

    @property
    def partial_score(self):
        return self.storage[..., 2048:4096].view(torch.float32)

    @property
    def positions(self):
        return self.storage[..., 4096:4104].view(torch.int64).squeeze(-1)

    @property
    def valid(self):
        return self.storage[..., 4104:4108].view(torch.int32).squeeze(-1)


def normalize_empty_pair_checkpoint(
    region, snapshot, positions, state_ready, *, reuse_unit
):
    """Materialize the position omitted by a canonical empty memory checkpoint.

    The caller validates execution metadata and supplies the entire pair region,
    including speculative snapshots and padding. Only its private snapshot is
    updated; malformed/nonempty input retains the normal reader's strict checks.
    """
    if (
        region.ndim != 2
        or region.dtype != torch.uint8
        or region.shape[1] < PAIR_SNAPSHOT_BYTES
        or region.stride(1) != 1
        or type(reuse_unit) is not int
        or reuse_unit <= 0
        or reuse_unit % 2
    ):
        raise ValueError("empty pair normalization requires a complete pair region")
    rows, device = region.shape[0], region.device
    _tensor(snapshot, (rows, PAIR_SNAPSHOT_BYTES), torch.uint8, device, "pair snapshot")
    _tensor(positions, (rows,), torch.int64, device, "restored pair positions")
    _tensor(state_ready, (rows,), torch.bool, device, "restored pair readiness")
    empty = state_ready & (positions > 0) & (positions <= 1048576)
    empty &= positions % reuse_unit == 0
    empty &= (region == 0).all(-1)
    state = V41DecodePairState(snapshot)
    state.positions.copy_(torch.where(empty, positions, state.positions))


def _inverse_frequencies(device):
    rope = CompressorRoPE()
    dimensions = torch.arange(0, rope.dimension, 2, dtype=torch.float32, device=device)
    frequencies = 1.0 / (rope.theta ** (dimensions / rope.dimension))

    def corrected_dimension(rotations):
        return (
            rope.dimension
            * math.log(rope.original_context / (rotations * 2 * math.pi))
            / (2 * math.log(rope.theta))
        )

    low = max(math.floor(corrected_dimension(rope.beta_fast)), 0)
    high = min(math.ceil(corrected_dimension(rope.beta_slow)), rope.dimension - 1)
    ramp = (
        (torch.arange(rope.dimension // 2, dtype=torch.float32, device=device) - low)
        / max(high - low, 1e-3)
    ).clamp(0, 1)
    smooth = 1 - ramp
    return frequencies / rope.factor * (1 - smooth) + frequencies * smooth


class V41DecodeOwnerCompressor(nn.Module):
    """Alias loaded owner weights and retain fixed buffers for one graph bucket.

    Result rows stay in request-major query order. ``emitted`` identifies real
    compressed rows; incomplete pairs and padding never write a compact page.
    Inputs and snapshots have fixed addresses, including the packed pair ABI.
    Bind one instance per owner and graph bucket, and retain its resources for
    the whole graph lifetime. ``prepare`` does not change any pool backing.
    """

    def __init__(
        self,
        owner,
        index_weight,
        index_norm,
        *,
        batch_size,
        query_width=6,
    ):
        super().__init__()
        if not isinstance(owner, OwnerCompressor):
            raise TypeError("decode compression requires a loaded V4.1 owner")
        if (
            type(batch_size) is not int
            or batch_size <= 0
            or type(query_width) is not int
            or query_width not in (1, 6)
        ):
            raise ValueError(
                "decode compression requires a fixed B x 1 or B x 6 bucket"
            )
        device = owner.wkv.device
        _tensor(index_weight, (128, 512), torch.bfloat16, device, "index weight")
        if index_norm.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("index norm must retain its BF16 or FP32 weight")
        _tensor(index_norm, (128,), index_norm.dtype, device, "index norm")
        if not is_supported(torch.empty(0, device=device, dtype=torch.bfloat16)):
            raise RuntimeError("decode owner compression requires Blackwell CUDA")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("allocate the decode compressor before graph capture")
        self.owner = owner
        self.batch_size, self.query_width = batch_size, query_width
        self.register_buffer("index_weight", index_weight)
        self.register_buffer("index_norm", index_norm)
        self.register_buffer("inverse_frequencies", _inverse_frequencies(device))
        self.register_buffer(
            "start_positions", torch.zeros(batch_size, dtype=torch.int64, device=device)
        )
        self.register_buffer(
            "valid_rows", torch.zeros(batch_size, dtype=torch.int32, device=device)
        )
        self.register_buffer(
            "row_offsets", torch.arange(query_width, dtype=torch.int64, device=device)
        )
        self.register_buffer(
            "snapshot_offsets",
            torch.arange(query_width + 1, dtype=torch.int64, device=device),
        )
        for name, width, dtype in (
            ("unrotated", 512, torch.bfloat16),
            ("global_values", 512, torch.bfloat16),
            ("index_values", 128, torch.bfloat16),
        ):
            self.register_buffer(
                name,
                torch.zeros(
                    (batch_size, query_width, width), dtype=dtype, device=device
                ),
            )
        for name, dtype in (
            ("group_positions", torch.int64),
            ("visible_lengths", torch.int32),
            ("emitted", torch.bool),
            ("global_slots", torch.int64),
            ("index_slots", torch.int64),
            ("global_status", torch.int32),
            ("index_status", torch.int32),
        ):
            self.register_buffer(
                name, torch.zeros((batch_size, query_width), dtype=dtype, device=device)
            )
        if owner.ratio == 2:
            for name, shape in (
                ("pair_input", (batch_size, PAIR_SNAPSHOT_BYTES)),
                ("pair_snapshots", (batch_size, query_width + 1, PAIR_SNAPSHOT_BYTES)),
                ("selected_pair", (batch_size, PAIR_SNAPSHOT_BYTES)),
            ):
                self.register_buffer(
                    name, torch.zeros(shape, dtype=torch.uint8, device=device)
                )

    @classmethod
    def from_owner(cls, owner, index_weight, index_norm, *, batch_size, query_width=6):
        return cls(
            owner,
            index_weight,
            index_norm,
            batch_size=batch_size,
            query_width=query_width,
        )

    @torch.inference_mode()
    def prepare(self, start_positions, valid_rows, *, pair_state=None):
        """Refresh fixed inputs; a missing pair is an explicit even-boundary reset.

        ``pair_state`` contains the accepted snapshot restored by the engine,
        including an odd PD/chunk boundary's real FP32 projection and score.
        Zero valid rows denote padding and preserve an existing pair unchanged.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "prepare decode owner inputs before graph capture/replay"
            )
        device = self.start_positions.device
        _tensor(
            start_positions, (self.batch_size,), torch.int64, device, "start positions"
        )
        _tensor(valid_rows, (self.batch_size,), torch.int32, device, "valid row counts")
        torch._assert_async(
            ((valid_rows >= 0) & (valid_rows <= self.query_width)).all(),
            "invalid decode query count",
        )
        live = valid_rows > 0
        torch._assert_async(
            (
                ~live
                | ((start_positions >= 0) & (start_positions + valid_rows <= 1048576))
            ).all(),
            "decode owner exceeds model context",
        )
        self.start_positions.copy_(start_positions)
        self.valid_rows.copy_(valid_rows)
        if self.owner.ratio == 1:
            if pair_state is not None:
                raise ValueError("ratio1 owner cannot consume ratio2 snapshots")
            return
        if pair_state is None:
            torch._assert_async(
                (~live | (start_positions % 2 == 0)).all(),
                "odd decode start requires its real pair state",
            )
            self.pair_input.zero_()
            V41DecodePairState(self.pair_input).positions.copy_(start_positions)
        else:
            raw = pair_state.storage
            _tensor(
                raw,
                (self.batch_size, PAIR_SNAPSHOT_BYTES),
                torch.uint8,
                device,
                "packed pair input",
            )
            self.pair_input.copy_(raw)
        pair = V41DecodePairState(self.pair_input)
        torch._assert_async(
            (
                ~live
                | (
                    (pair.positions == start_positions)
                    & (pair.valid == start_positions % 2)
                )
            ).all(),
            "pair snapshot differs from the materialized start",
        )

    def _save_snapshots(self, projected, score):
        pair = V41DecodePairState(self.pair_input)
        snapshots = V41DecodePairState(self.pair_snapshots)
        counts = torch.minimum(self.valid_rows[:, None], self.snapshot_offsets[None, :])
        positions = self.start_positions[:, None] + counts
        index = (counts - 1).clamp_min(0).unsqueeze(-1).expand(-1, -1, 512)
        partial_kv = projected.gather(1, index)
        partial_score = score.gather(1, index)
        keep_new = ((counts > 0) & (positions % 2 != 0)).unsqueeze(-1)
        keep_old = (counts == 0).unsqueeze(-1)
        self.pair_snapshots.copy_(
            self.pair_input[:, None, :].expand_as(self.pair_snapshots)
        )
        snapshots.partial_kv.copy_(
            torch.where(
                keep_old,
                pair.partial_kv[:, None, :],
                torch.where(keep_new, partial_kv, 0),
            )
        )
        snapshots.partial_score.copy_(
            torch.where(
                keep_old,
                pair.partial_score[:, None, :],
                torch.where(keep_new, partial_score, 0),
            )
        )
        snapshots.positions.copy_(
            torch.where(counts > 0, positions, pair.positions[:, None])
        )
        snapshots.valid.copy_(
            torch.where(counts > 0, positions % 2, pair.valid[:, None])
        )

    @torch.inference_mode()
    def forward(self, normalized_hidden):
        _tensor(
            normalized_hidden,
            (self.batch_size, self.query_width, 5120),
            torch.bfloat16,
            self.start_positions.device,
            "normalized owner input",
        )
        active = self.row_offsets[None, :] < self.valid_rows[:, None]
        positions = self.start_positions[:, None] + self.row_offsets[None, :]
        hidden = normalized_hidden.masked_fill(~active[:, :, None], 0)
        if self.owner.ratio == 2:
            projected = F.linear(hidden.float(), self.owner.wkv)
            score = F.linear(hidden.float(), self.owner.wgate)
            pair = V41DecodePairState(self.pair_input)
            torch._assert_async(
                (
                    ~active[:, :, None]
                    | (torch.isfinite(projected) & torch.isfinite(score))
                ).all(),
                "nonfinite ratio2 projection cannot become committed pair state",
            )
            previous_kv = torch.cat(
                (pair.partial_kv[:, None, :], projected[:, :-1, :]), dim=1
            )
            previous_score = torch.cat(
                (pair.partial_score[:, None, :], score[:, :-1, :]), dim=1
            )
            grouped_kv = torch.stack((previous_kv, projected), dim=2)
            grouped_score = torch.stack((previous_score, score), dim=2)
            pooled = (grouped_kv * grouped_score.softmax(dim=2)).sum(dim=2)
            latent = rms_norm(pooled.to(hidden.dtype), self.owner.norm_weight)
            emitted = active & (positions % 2 != 0)
            self._save_snapshots(projected, score)
        else:
            latent = rms_norm(F.linear(hidden, self.owner.wkv), self.owner.norm_weight)
            emitted = active
        self.emitted.copy_(emitted)
        self.unrotated.copy_(latent.masked_fill(~emitted[:, :, None], 0))
        self.visible_lengths.copy_(
            torch.where(active, (positions + 1) // self.owner.ratio, 0)
        )
        self.group_positions.copy_(
            torch.where(emitted, positions - (self.owner.ratio - 1), 0)
        )
        phases = (
            self.group_positions.flatten().float()[:, None]
            * self.inverse_frequencies[None, :]
        )
        frequencies = torch.polar(torch.ones_like(phases), phases)
        flat = self.unrotated.flatten(0, 1)
        index = rms_norm(F.linear(flat, self.index_weight), self.index_norm)
        self.global_values.copy_(_rotate(flat, frequencies).view_as(self.global_values))
        self.index_values.copy_(_rotate(index, frequencies).view_as(self.index_values))
        return self

    @torch.inference_mode()
    def store(self, global_pages, index_pages, global_slots, index_slots):
        """Store emitted rows at the caller's current physical mappings.

        Slot tensors cover all query rows, even a ratio2 row that emits nothing.
        The non-padding destinations must be unique across the live batch.
        Check both returned writer statuses after replay and before publication.
        """
        for pages, region, slots, destination in (
            (global_pages, CacheRegion.GLOBAL, global_slots, self.global_slots),
            (index_pages, CacheRegion.INDEX_K, index_slots, self.index_slots),
        ):
            if (
                pages.region != region
                or pages.entries_per_page
                != self.owner.layout.token_block_size // self.owner.ratio
            ):
                raise ValueError(
                    "decode compact pool differs from the owner's region/ratio"
                )
            _tensor(
                slots,
                (self.batch_size, self.query_width),
                torch.int64,
                self.start_positions.device,
                "owner physical slots",
            )
            destination.copy_(torch.where(self.emitted, slots, -1))
        return (
            write_compact(
                self.global_values.flatten(0, 1),
                global_pages,
                self.global_slots.flatten(),
                status=self.global_status.flatten(),
            ),
            write_compact(
                self.index_values.flatten(0, 1),
                index_pages,
                self.index_slots.flatten(),
                status=self.index_status.flatten(),
            ),
        )

    @torch.inference_mode()
    def select_pair(self, kept_rows):
        """Select retained materialized rows, including accept/EOS/stop truncation.

        Normal verification retains ``1 + accepted_proposals`` rows. Sampled
        correction/bonus tokens are not materialized and are not counted here.
        Copy the result to the request's state page before preparing another step.
        """
        if self.owner.ratio != 2:
            raise ValueError("ratio1 owner has no speculative pair snapshots")
        _tensor(
            kept_rows,
            (self.batch_size,),
            torch.int32,
            self.start_positions.device,
            "retained materialized row counts",
        )
        torch._assert_async(
            ((kept_rows >= 0) & (kept_rows <= self.valid_rows)).all(),
            "retained rows exceed the completed verify rows",
        )
        indices = (
            kept_rows.to(torch.int64)
            .clamp(0, self.query_width)[:, None, None]
            .expand(-1, 1, PAIR_SNAPSHOT_BYTES)
        )
        self.selected_pair.copy_(self.pair_snapshots.gather(1, indices).squeeze(1))
        return V41DecodePairState(self.selected_pair)
