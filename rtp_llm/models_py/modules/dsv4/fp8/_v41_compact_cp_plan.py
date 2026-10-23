"""Host-only geometry and wire layout for admitted V4.1 B1 CP4 chunks."""

from dataclasses import dataclass

MODES = ("off", "projected", "packed")
PRODUCERS = ((2, 2), (8, 2), (14, 2), (20, 1))
HEADER_BYTES = 128
RECORD_BYTES = 364
TAIL_RECORD_BYTES = 4112
WIRE_VERSION = 1


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
        ):
            value = getattr(self, name)
            minimum = -(1 << 63) if name == "request_id" else 0
            if type(value) is not int or not minimum <= value < (1 << 63):
                raise ValueError(f"Invalid compact CP {name}")
        if self.rank >= 4 or self.start % 4096 or self.start > 28672:
            raise ValueError("Compact CP requires an aligned 32K B1 CP4 chunk")
        if (self.layer, self.ratio) not in PRODUCERS:
            raise ValueError("Unknown compact CP producer")

    @property
    def intervals(self):
        return (
            (self.start + self.rank * 512, self.start + (self.rank + 1) * 512),
            (self.start + (7 - self.rank) * 512, self.start + (8 - self.rank) * 512),
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
        return 1024 // self.ratio

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
            4096,
            self.ratio,
            self.rank,
            self.rows,
            self.tail_count,
            0,
        )


def resolve_geometry(
    mode, cp, spans, *, stage, generation_count, device_type, capability
):
    """Return the agreed geometry decision before any payload is allocated.

    The native preflight validates CP tensors against these host mirrors. This
    function only narrows that existing admission; it does not validate tensors.
    """
    parse_mode(mode)
    if mode == "off":
        return "off", "disabled"
    if stage != 0:
        return "off", "no-producers"
    if device_type != "cuda" or capability != (12, 0):
        return "off", "architecture"
    if generation_count != 0 or len(spans) != 1:
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
