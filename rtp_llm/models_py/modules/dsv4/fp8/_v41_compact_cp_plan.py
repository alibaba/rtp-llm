"""Host-only geometry and wire layout for admitted V4.1 CP4 chunks.

Base contract: single-request (B1) 4096-token chunks. Under
``DSV41_D2_COMPACT_GENERAL`` (Batch 5 #9/D2, default 0) the packed
geometry generalizes to 8192-token chunks (local 2048 = 2x1024 zigzag
halves) and multi-request packed chunks (uniform per-request counts);
the wire ABI per request packet is unchanged (header field 10 carries
the chunk token count). Flag off keeps the exact B1/4096 admission.
"""

import os
import sys
from dataclasses import dataclass

MODES = ("off", "projected", "packed")
PRODUCERS = ((2, 2), (8, 2), (14, 2), (20, 1))
HEADER_BYTES = 128
RECORD_BYTES = 364
TAIL_RECORD_BYTES = 4112
WIRE_VERSION = 1
CHUNK_WINDOW = 32768
SUPPORTED_CHUNK_TOKENS = (4096, 8192)

_D2_FLAG = "DSV41_D2_COMPACT_GENERAL"
_D2_ANNOUNCED = False


def d2_general_enabled() -> bool:
    """D2: generalized compact-CP geometry (default 0)."""
    return os.environ.get(_D2_FLAG, "0") == "1"


def _announce_d2(detail: str) -> None:
    global _D2_ANNOUNCED
    if not _D2_ANNOUNCED:
        _D2_ANNOUNCED = True
        print(
            f"[D2GEN] generalized compact CP engaged ({detail})",
            file=sys.stderr,
            flush=True,
        )


def parse_mode(value):
    if value not in MODES:
        raise ValueError("DSV41_CP_COMPACT_MODE must be off, projected or packed")
    return value


@dataclass(frozen=True)
class CompactCPPlan:
    mode: str
    request_id: int
    generation: int
    cache_incarnation: int
    execution_round: int
    chunk_id: int
    start: int
    rank: int
    layer: int
    ratio: int
    chunk_tokens: int = 4096

    def __post_init__(self):
        parse_mode(self.mode)
        if self.mode == "off":
            raise ValueError("An off request has no compact payload plan")
        for name in (
            "request_id",
            "generation",
            "cache_incarnation",
            "execution_round",
            "chunk_id",
            "start",
            "rank",
            "layer",
            "ratio",
            "chunk_tokens",
        ):
            value = getattr(self, name)
            minimum = -(1 << 63) if name == "request_id" else 0
            if type(value) is not int or not minimum <= value < (1 << 63):
                raise ValueError(f"Invalid compact CP {name}")
        if self.chunk_tokens not in SUPPORTED_CHUNK_TOKENS:
            raise ValueError("Compact CP chunk tokens must be 4096 or 8192")
        if (
            self.rank >= 4
            or self.start % self.chunk_tokens
            or self.start + self.chunk_tokens > CHUNK_WINDOW
        ):
            raise ValueError("Compact CP requires an aligned 32K-window CP4 chunk")
        if (self.layer, self.ratio) not in PRODUCERS:
            raise ValueError("Unknown compact CP producer")

    @property
    def half(self):
        """Zigzag half width in global tokens (512 at 4096, 1024 at 8192)."""
        return self.chunk_tokens // 8

    @property
    def local_rows(self):
        return self.chunk_tokens // 4

    @property
    def intervals(self):
        half = self.half
        return (
            (self.start + self.rank * half, self.start + (self.rank + 1) * half),
            (self.start + (7 - self.rank) * half, self.start + (8 - self.rank) * half),
        )

    @property
    def boundary_positions(self):
        return tuple(
            position
            for begin, end in self.intervals
            for position in range(begin + self.ratio - 1, end, self.ratio)
        )

    @property
    def rows(self):
        return self.local_rows // self.ratio

    @property
    def tail_count(self):
        return 2 if self.ratio == 2 and self.rank == 0 else 0

    @property
    def send_bytes(self):
        return (
            HEADER_BYTES
            + self.rows * RECORD_BYTES
            + (2 * TAIL_RECORD_BYTES if self.ratio == 2 else 0)
        )

    @property
    def header(self):
        return (
            WIRE_VERSION,
            MODES.index(self.mode),
            0,
            self.layer,
            self.request_id,
            self.generation,
            self.cache_incarnation,
            self.execution_round,
            self.chunk_id,
            self.start,
            self.chunk_tokens,
            self.ratio,
            self.rank,
            self.rows,
            self.tail_count,
            0,
        )


def producer_specs(stage, pp_cut=21):
    """Stage-local owners; source IDs and per-producer wire arithmetic stay global."""
    if pp_cut not in (20, 21):
        raise ValueError("V4.1 compact CP requires cut 20 or 21")
    if stage == 0:
        return PRODUCERS if pp_cut == 21 else PRODUCERS[:-1]
    return PRODUCERS[-1:] if stage == 1 and pp_cut == 20 else ()


def resolve_geometry(
    mode, cp, spans, *, stage, generation_count, device_type, capability, pp_cut=21
):
    """Return the agreed geometry decision before any payload is allocated.

    The native preflight validates CP tensors against these host mirrors. This
    function only narrows that existing admission; it does not validate tensors.
    """
    parse_mode(mode)
    if mode == "off":
        return "off", "disabled"
    if not producer_specs(stage, pp_cut):
        return "off", "no-producers"
    if device_type != "cuda" or capability != (12, 0):
        return "off", "architecture"
    if generation_count != 0 or not spans:
        return "off", "request-geometry"
    if not d2_general_enabled():
        if len(spans) != 1:
            return "off", "request-geometry"
        span = spans[0]
        if span.count != 4096 or span.start % 4096 or not 0 <= span.start <= 28672:
            return "off", "chunk-geometry"
        if cp is None or (
            cp.cp_size != 4
            or not 0 <= cp.cp_rank < 4
            or cp.kv_cache_sharded
            or cp.chunk_length != 1024
            or cp.padded_seq_len != 4096
            or cp.seq_len_full != 4096
            or tuple(cp.chunk_lengths_per_req or ()) != (1024,)
            or tuple(cp.input_lengths_global_host or ()) != (4096,)
            or tuple(cp.prefix_lengths_host or ()) != (span.start,)
            or getattr(cp, "swa_replay_start", None) is not None
            or getattr(cp, "swa_replay_starts_host", None) is not None
            or getattr(cp, "gather_restore_positions", None) is not None
        ):
            return "off", "cp-geometry"
        return mode, "eligible"
    # D2 generalized admission: uniform per-request chunk counts of 4096
    # or 8192, window-aligned starts, and a cp context that exactly packs
    # the spans (local rows = count/4 per request, zigzag CP4).
    counts = {span.count for span in spans}
    if len(counts) != 1:
        return "off", "mixed-request-geometry"
    count = counts.pop()
    if count not in SUPPORTED_CHUNK_TOKENS:
        return "off", "chunk-geometry"
    if any(
        span.start % count or not 0 <= span.start <= CHUNK_WINDOW - count
        for span in spans
    ):
        return "off", "chunk-geometry"
    n = len(spans)
    local = count // 4
    if cp is None or (
        cp.cp_size != 4
        or not 0 <= cp.cp_rank < 4
        or cp.kv_cache_sharded
        or cp.chunk_length != n * local
        or cp.padded_seq_len != 4 * n * local
        or cp.seq_len_full != n * count
        or tuple(cp.chunk_lengths_per_req or ()) != (local,) * n
        or tuple(cp.input_lengths_global_host or ()) != (count,) * n
        or tuple(cp.prefix_lengths_host or ()) != tuple(span.start for span in spans)
        or getattr(cp, "swa_replay_start", None) is not None
        or getattr(cp, "swa_replay_starts_host", None) is not None
        or getattr(cp, "gather_restore_positions", None) is not None
    ):
        return "off", "cp-geometry"
    _announce_d2(f"spans={n} chunk_tokens={count}")
    return mode, "eligible"
