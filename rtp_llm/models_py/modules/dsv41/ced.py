"""V4.1 prefill phase planning and bounded-tail state contracts.

Runtime code must apply these plans and report completion after the corresponding
GPU work. Planning alone does not materialize KV, aux rows, or checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

if TYPE_CHECKING:
    import torch

from rtp_llm.models_py.modules.dsv41.cache_layout import (
    SWA_WINDOW,
    TARGET_LAYERS,
    CacheIdentity,
    CacheLayout,
    MemoryCheckpoint,
    _identity,
    layer_sources,
)


class ReplayMode(str, Enum):
    FULL = "full"
    BOUNDED = "bounded_checkpoint_v1"


@dataclass(frozen=True)
class ReplayConfig:
    mode: ReplayMode = ReplayMode.FULL
    window: int = SWA_WINDOW
    tail_policy_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ReplayMode(self.mode))
        if self.window != SWA_WINDOW or self.tail_policy_version != 1:
            raise ValueError("this release defines only window128 tail policy v1")

    @property
    def fingerprint(self) -> str:
        return _identity(asdict(self))

    def cache_identity(self, model_revision: str, layout: CacheLayout) -> CacheIdentity:
        return CacheIdentity(model_revision, layout.fingerprint, self.fingerprint)


@dataclass(frozen=True)
class RowRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid half-open token range")

    def __len__(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PrefillExtend:
    encoder_rows: RowRange
    decoder_rows: Optional[RowRange]
    replay_floor: Optional[int]
    uses_checkpoint_history: bool
    checkpoint_end: Optional[int]
    final_handoff: bool
    retain_l20_rows: RowRange
    required_protected_end: int

    def query_rows(self, layer: int) -> Optional[RowRange]:
        if not 0 <= layer < TARGET_LAYERS:
            raise ValueError("target query planning requires a target layer")
        return self.encoder_rows if layer <= 20 else self.decoder_rows

    def global_projection_rows(self, layer: int) -> Optional[RowRange]:
        return self.encoder_rows if layer_sources(layer).writes_global else None

    def index_query_rows(self, layer: int) -> Optional[RowRange]:
        return self.query_rows(layer) if layer_sources(layer).scores_queries else None

    @property
    def needs_decoder_ep(self) -> bool:
        # All EP ranks participate even when their local selected row set is empty.
        return self.decoder_rows is not None


@dataclass(frozen=True)
class PrefillPlan:
    config: ReplayConfig
    identity: CacheIdentity
    extends: Tuple[PrefillExtend, ...]
    protected_checkpoint_end: int
    fallback_reason: Optional[str] = None


@dataclass(frozen=True)
class LateCompletion:
    valid_aux_rows: RowRange
    target_swa_complete: bool = False
    draft_swa_complete: bool = False


def _validate_images(images: Sequence[RowRange], total_tokens: int) -> None:
    previous_end = 0
    for image in images:
        if not len(image) or image.start < previous_end or image.end > total_tokens:
            raise ValueError("image spans must be nonempty, ordered and disjoint")
        previous_end = image.end


def checkpoint_boundary(
    total_tokens: int, reuse_unit: int, images: Sequence[RowRange] = ()
) -> int:
    if total_tokens < 0 or reuse_unit <= 0:
        raise ValueError("invalid context length or reuse unit")
    _validate_images(images, total_tokens)
    boundary = total_tokens // reuse_unit * reuse_unit
    for image in reversed(images):
        if image.start < boundary < image.end:
            boundary = image.start // reuse_unit * reuse_unit
    return boundary


def build_prefill_plan(
    *,
    total_tokens: int,
    chunk_tokens: int,
    layout: CacheLayout,
    model_revision: str,
    config: ReplayConfig,
    images: Sequence[RowRange] = (),
    restored_checkpoint: Optional[MemoryCheckpoint] = None,
    all_prompt_logprobs: bool = False,
    all_hidden_states: bool = False,
) -> PrefillPlan:
    if total_tokens <= 0 or total_tokens > 1048576:
        raise ValueError("prefill context must fit the model's 1048576 token limit")
    if chunk_tokens <= 0:
        raise ValueError("chunk token budget must be positive")
    _validate_images(images, total_tokens)
    if any(len(image) > chunk_tokens for image in images):
        raise ValueError("chunk budget cannot fit a complete image span")
    fallback = None
    if config.mode == ReplayMode.BOUNDED and (all_prompt_logprobs or all_hidden_states):
        fallback = "all_prompt_logprobs" if all_prompt_logprobs else "all_hidden_states"
        config = replace(config, mode=ReplayMode.FULL)
    identity = config.cache_identity(model_revision, layout)
    checkpoint_end = checkpoint_boundary(total_tokens, layout.reuse_unit, images)
    start = 0
    complete_end = 0
    if restored_checkpoint is not None:
        if not restored_checkpoint.is_complete(layout, identity):
            raise ValueError(
                "restored checkpoint is incomplete or has another identity"
            )
        start = restored_checkpoint.materialized_end
        if start >= total_tokens:
            raise ValueError(
                "restored checkpoint must precede the next token to compute"
            )
        if any(image.start < start < image.end for image in images):
            raise ValueError("restored checkpoint splits an image")
        complete_end = start
    extends = []
    while start < total_tokens:
        end = min(start + chunk_tokens, total_tokens)
        if start < checkpoint_end < end:
            end = checkpoint_end
        for image in images:
            if image.start < end < image.end:
                end = image.start
                break
        if end <= start:
            raise ValueError("chunk selection made no progress")
        publish = end == checkpoint_end and checkpoint_end > 0
        handoff = end == total_tokens
        required = publish or handoff
        decoder_rows = None
        floor = None
        use_history = False
        if config.mode == ReplayMode.FULL:
            decoder_rows = RowRange(start, end)
            floor = 0
            use_history = start > 0
        elif required:
            # Only a complete same-mode boundary permits short-suffix continuation.
            use_history = complete_end > 0 and end - complete_end <= config.window
            decoder_start = complete_end if use_history else max(0, end - config.window)
            decoder_rows = RowRange(decoder_start, end)
            floor = (
                max(0, complete_end - config.window) if use_history else decoder_start
            )
        extends.append(
            PrefillExtend(
                RowRange(start, end),
                decoder_rows,
                floor,
                use_history,
                end if publish else None,
                handoff,
                RowRange(max(0, end - config.window), end),
                checkpoint_end if start >= checkpoint_end and checkpoint_end else 0,
            )
        )
        if required:
            complete_end = end
        start = end
    return PrefillPlan(config, identity, tuple(extends), checkpoint_end, fallback)


@dataclass(frozen=True)
class PrefillProgress:
    encoder_materialized_end: int = 0
    decoder_checkpoint_end: int = 0
    protected_checkpoint_end: int = 0

    def __post_init__(self) -> None:
        if not (
            0
            <= self.protected_checkpoint_end
            <= self.decoder_checkpoint_end
            <= self.encoder_materialized_end
        ):
            raise ValueError("CED progress boundaries are inconsistent")

    def encoder_completed(self, extend: PrefillExtend) -> PrefillProgress:
        rows = extend.encoder_rows
        if extend.required_protected_end > self.protected_checkpoint_end:
            raise ValueError("protect checkpoint N before materializing its suffix")
        if rows.start != self.encoder_materialized_end or not len(rows):
            raise ValueError(
                "encoder completion must extend the current contiguous prefix"
            )
        return replace(self, encoder_materialized_end=rows.end)

    def decoder_completed(
        self, extend: PrefillExtend, completion: LateCompletion
    ) -> PrefillProgress:
        rows = extend.decoder_rows
        if rows is None or rows.end > self.encoder_materialized_end:
            raise ValueError("decoder completion requires materialized encoder rows")
        if (
            completion.valid_aux_rows != rows
            or not completion.target_swa_complete
            or not completion.draft_swa_complete
        ):
            raise ValueError(
                "decoder checkpoint requires valid aux and complete target/draft SWA"
            )
        if rows.end < self.decoder_checkpoint_end:
            raise ValueError("decoder completion cannot move backwards")
        if extend.uses_checkpoint_history and rows.start != self.decoder_checkpoint_end:
            raise ValueError(
                "decoder continuation requires its planned checkpoint history"
            )
        return replace(self, decoder_checkpoint_end=rows.end)

    def checkpoint_protected(
        self, checkpoint: MemoryCheckpoint, layout: CacheLayout, identity: CacheIdentity
    ) -> PrefillProgress:
        if not checkpoint.is_complete(layout, identity):
            raise ValueError("checkpoint backing/copy/history is not complete")
        if not checkpoint.backing_protected:
            raise ValueError(
                "checkpoint backing needs a held reference or private copy"
            )
        if checkpoint.materialized_end != self.decoder_checkpoint_end:
            raise ValueError(
                "checkpoint must be protected at its materialized boundary"
            )
        return replace(self, protected_checkpoint_end=checkpoint.materialized_end)

    def require_handoff(self, total_tokens: int, protected_end: int) -> None:
        if self.encoder_materialized_end != total_tokens:
            raise ValueError("encoder has not materialized the handoff boundary")
        if self.decoder_checkpoint_end != total_tokens:
            raise ValueError("decoder/aux state has not reached the handoff boundary")
        if protected_end and self.protected_checkpoint_end != protected_end:
            raise ValueError("required prefix checkpoint was not protected")


@dataclass(frozen=True)
class L20TailRow:
    position: int
    logical_row: int
    image_mask: bool
    topk: Tuple[int, ...]
    candidate_blocks: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.position < 0 or self.logical_row < 0:
            raise ValueError("retained tail rows require absolute logical positions")
        visible = self.position + 1
        for ids, capacity, upper in (
            (self.topk, 512, visible),
            (self.candidate_blocks, 2048, (visible + 7) // 8),
        ):
            if ids != tuple(sorted(set(ids))) or any(i < 0 or i >= upper for i in ids):
                raise ValueError("tail index IDs must be unique, sorted and causal")
            if len(ids) != min(capacity, upper):
                raise ValueError(
                    "tail index valid count does not cover the selected set"
                )
        if visible % 8 and self.position // 8 not in self.candidate_blocks:
            raise ValueError("L20 candidates must retain the current partial block")


@dataclass(frozen=True)
class L20TailMetadata:
    """Metadata paired with owned HC/pre_mix storage, never physical page IDs."""

    request_id: str
    forward_epoch: int
    replay_fingerprint: str
    rows: Tuple[L20TailRow, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id or self.forward_epoch < 0 or not self.replay_fingerprint:
            raise ValueError(
                "tail metadata requires request, epoch and replay identity"
            )
        positions = tuple(row.position for row in self.rows)
        if len(positions) > SWA_WINDOW or (
            positions and positions != tuple(range(positions[0], positions[-1] + 1))
        ):
            raise ValueError(
                "retained L20 tail must be consecutive and at most 128 rows"
            )
        if len({row.logical_row for row in self.rows}) != len(self.rows):
            raise ValueError("retained logical row IDs must be unique")

    def append(
        self, rows: Sequence[L20TailRow], *, forward_epoch: int
    ) -> L20TailMetadata:
        if forward_epoch < self.forward_epoch:
            raise ValueError("retained tail cannot accept an older forward epoch")
        if not rows:
            return replace(self, forward_epoch=forward_epoch)
        if self.rows and rows[0].position != self.rows[-1].position + 1:
            raise ValueError("retained tail updates must follow global token order")
        positions = tuple(row.position for row in rows)
        if positions != tuple(range(positions[0], positions[-1] + 1)):
            raise ValueError("new L20 tail rows are not consecutive")
        return replace(
            self,
            rows=(self.rows + tuple(rows))[-SWA_WINDOW:],
            forward_epoch=forward_epoch,
        )

    def selection(
        self,
        rows: RowRange,
        *,
        request_id: str,
        forward_epoch: int,
        replay_fingerprint: str,
    ) -> Tuple[int, ...]:
        if (request_id, forward_epoch, replay_fingerprint) != (
            self.request_id,
            self.forward_epoch,
            self.replay_fingerprint,
        ):
            raise ValueError("retained tail request/epoch/replay identity is stale")
        positions = {row.position: index for index, row in enumerate(self.rows)}
        if any(position not in positions for position in range(rows.start, rows.end)):
            raise ValueError(
                "required L20 HC/pre_mix/top-k/candidates are no longer retained"
            )
        return tuple(positions[position] for position in range(rows.start, rows.end))


def local_tail_indices(
    global_positions: Sequence[int], rows: RowRange
) -> Tuple[int, ...]:
    """Select CP-local rows while retaining their absolute position interpretation."""
    if len(set(global_positions)) != len(global_positions):
        raise ValueError("CP row map contains duplicate global positions")
    return tuple(
        index
        for index, position in enumerate(global_positions)
        if rows.start <= position < rows.end
    )


@dataclass(frozen=True)
class AuxRowMap:
    request_id: str
    forward_epoch: int
    replay_fingerprint: str
    positions: Tuple[int, ...]
    logical_rows: Tuple[int, ...]
    image_mask: Tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.request_id or self.forward_epoch < 0 or not self.replay_fingerprint:
            raise ValueError("aux rows require request, epoch and replay identity")
        if not (len(self.positions) == len(self.logical_rows) == len(self.image_mask)):
            raise ValueError("aux row metadata lengths disagree")
        if self.positions != tuple(sorted(set(self.positions))):
            raise ValueError("valid aux positions must be unique and ordered")
        if any(position < 0 for position in self.positions):
            raise ValueError("valid aux positions must be non-negative")
        if len(set(self.logical_rows)) != len(self.logical_rows) or any(
            row < 0 for row in self.logical_rows
        ):
            raise ValueError("logical aux row IDs must be unique and non-negative")

    def selection(
        self,
        required_positions: Sequence[int],
        *,
        request_id: str,
        forward_epoch: int,
        replay_fingerprint: str,
    ) -> Tuple[int, ...]:
        if (request_id, forward_epoch, replay_fingerprint) != (
            self.request_id,
            self.forward_epoch,
            self.replay_fingerprint,
        ):
            raise ValueError("aux request/epoch/replay identity is stale")
        required = tuple(required_positions)
        if required != tuple(sorted(set(required))):
            raise ValueError("requested aux positions must be unique and ordered")
        indices = {position: index for index, position in enumerate(self.positions)}
        if any(position not in indices for position in required):
            raise ValueError("requested aux row has not been computed")
        return tuple(indices[position] for position in required)


def select_aux_hidden(
    hidden: torch.Tensor,
    row_map: AuxRowMap,
    required_positions: Sequence[int],
    *,
    request_id: str,
    forward_epoch: int,
    replay_fingerprint: str,
) -> torch.Tensor:
    """Gather valid aux before main_proj/main_norm and each draft wkv projection."""
    import torch

    if hidden.ndim != 2 or tuple(hidden.shape) != (len(row_map.positions), 15360):
        raise ValueError("V4.1 aux must contain only valid rows with width 15360")
    indices = row_map.selection(
        required_positions,
        request_id=request_id,
        forward_epoch=forward_epoch,
        replay_fingerprint=replay_fingerprint,
    )
    selection = torch.tensor(indices, dtype=torch.long, device=hidden.device)
    return hidden.index_select(0, selection)
