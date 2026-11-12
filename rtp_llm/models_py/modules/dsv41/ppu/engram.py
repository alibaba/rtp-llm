# SPDX-License-Identifier: Apache-2.0
# Kernels and protocol pieces adapted from sglang-for-sail (commit 512b03d,
# Apache-2.0):
#   python/sglang/srt/layers/engram.py            (zero-fill + all-reduce lookup)
#   python/sglang/kernels/ops/embeddings/engram_hash.py    (Triton n-gram hash)
#   python/sglang/kernels/ops/embeddings/engram_gather.py  (software E4M3 gather)
#   python/sglang/kernels/ops/embeddings/engram_gate.py    (fused gate)
# Torch reference math adapted from rtp-llm
# rtp_llm/models_py/modules/dsv41/engram.py, which is the numerical authority
# for the hash, the FP8/E8M0 dequantization and the gated inject.
"""ENP8 host-sharded Engram memory-table lookup for DeepSeek V4.1 Flash.

Each ENP rank owns one contiguous slice of the per-layer Engram table and
keeps it in host RAM as FP8-E4M3 payload bytes plus E8M0 block scales (32
value groups along the embedding dim). A lookup is a collective over the ENP
group (plan section 4.3):

1. Query row IDs (global table row ids, int64, ``-1`` = invalid) are split
   into rounds of at most ``max_rows_per_round`` rows, sized by the
   GROUP-WIDE maximum so every rank executes the same rounds; a rank with
   fewer rows pads its tail with ``-1`` and empty ranks join every round.
2. Per round the row IDs are all-gathered in FIXED rank order.
3. Each rank reads ONLY the rows it owns from its host shard, dequantizes
   them to bf16 and writes them into a zero-initialized contribution tensor
   at the requesting position; non-owned positions stay zero.
4. The contributions are all-reduced (sum). Exactly one rank owns each row,
   so the sum never accumulates two live values.
5. Each rank slices out its own request rows, preserving input order.

triton and torch.distributed are imported lazily; this module loads on a
CPU-only host. cudaHostRegister-style pinning from sglang is deliberately
NOT copied: PPU uses ``torch.Tensor.pin_memory()`` (lazily, only when a PPU
device is present) and CPU hosts keep plain tensors.
"""

from __future__ import annotations

import json
import math
import os
import types
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F

__all__ = [
    "EngramShardLayout",
    "HostShard",
    "EngramHashParams",
    "LocalEnpmGroup",
    "DistributedEnpmGroup",
    "DEAD_ID",
    "FP8_BLOCK_SIZE",
    "PAD_SCALE_BYTE",
    "COPY_BATCH_ROWS",
    "compute_hash_multipliers",
    "compressed_vocab_map",
    "build_token_window",
    "build_token_windows",
    "build_engram_primes",
    "compute_engram_hash_ids",
    "dequant_rows",
    "engram_lookup",
    "wkv_project",
    "engram_gate_inject",
    "is_ppu",
    "make_padding_rows",
    "shard_bytes_estimate",
]

DEAD_ID = -1
FP8_BLOCK_SIZE = 32
# Padding / alignment rows load as zero payload with a unit (2**0) scale.
PAD_SCALE_BYTE = 0x7F
# Upper bound on rows moved host->device in one copy batch.
COPY_BATCH_ROWS = 4096
LOCAL_PREFILL_ENGRAM_ROUNDS_ENV = "RTP_V41_PPU_LOCAL_PREFILL_ENGRAM_ROUNDS"
LOCAL_PREFILL_ENGRAM_ROUND_CAP = 98304
_ROW_ALIGNMENT = 256


def is_ppu() -> bool:
    """True when this process sees a PPU accelerator device."""
    try:
        from rtp_llm.device.device_type import DeviceType, get_device_type

        return get_device_type() == DeviceType.Ppu
    except Exception:
        return False


def _maybe_pin(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Pin host tensors for PPU device gathers; CPU hosts keep plain tensors."""
    if tensor.device.type != "cpu":
        return tensor, tensor.is_pinned()
    if tensor.is_pinned():
        return tensor, True
    if not is_ppu():
        return tensor, False
    return tensor.pin_memory(), True


# ---------------------------------------------------------------------------
# ENP8 sharding math (from DeepInfer EngramLayout, authoritative)
# ---------------------------------------------------------------------------


def shard_bytes_estimate(rows: int, dim: int, block_size: int = FP8_BLOCK_SIZE) -> int:
    """Host bytes of ``rows`` table rows (payload + E8M0 scales)."""
    return rows * dim + rows * (dim // block_size)


@dataclass(frozen=True)
class EngramShardLayout:
    """Contiguous ENP row split with 256-byte-aligned shard boundaries.

    alignment = lcm(256, enp); padded_rows = ceil(num_embeddings / alignment)
    * alignment; num_per_part = padded_rows // enp; row_start(rank) =
    rank * num_per_part; owner(row) = row // num_per_part; valid_rows(rank) =
    min(num_per_part, num_embeddings - row_start(rank)). Padding rows
    (>= num_embeddings) are invalid: they load as zero payload with unit
    scale and never read a shard.
    """

    num_embeddings: int
    enp_size: int
    alignment: int
    num_per_part: int
    padded_rows: int

    @classmethod
    def from_table(cls, num_embeddings: int, enp_size: int) -> "EngramShardLayout":
        num_embeddings = int(num_embeddings)
        enp_size = int(enp_size)
        if num_embeddings < 0:
            raise ValueError("num_embeddings must be non-negative")
        if enp_size < 1:
            raise ValueError("enp_size must be positive")
        alignment = math.lcm(_ROW_ALIGNMENT, enp_size)
        padded_rows = -(-num_embeddings // alignment) * alignment
        num_per_part = padded_rows // enp_size
        return cls(
            num_embeddings=num_embeddings,
            enp_size=enp_size,
            alignment=alignment,
            num_per_part=num_per_part,
            padded_rows=padded_rows,
        )

    def row_start(self, rank: int) -> int:
        return int(rank) * self.num_per_part

    def valid_rows(self, rank: int) -> int:
        start = self.row_start(rank)
        if start >= self.num_embeddings:
            return 0
        return min(self.num_per_part, self.num_embeddings - start)

    def owner(self, row: int) -> int:
        """Owning rank of a global row id (only meaningful for valid rows)."""
        return int(row) // self.num_per_part

    def row_is_global_valid(self, row: int) -> bool:
        return 0 <= int(row) < self.num_embeddings

    def owns(self, rank: int, row: int) -> bool:
        start = self.row_start(rank)
        return start <= int(row) < start + self.valid_rows(rank)


# ---------------------------------------------------------------------------
# Host shard
# ---------------------------------------------------------------------------


def make_padding_rows(count: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Padding rows: zero FP8 payload + unit (0x7F) E8M0 scale."""
    payload = torch.zeros((count, dim), dtype=torch.float8_e4m3fn)
    scale = torch.full(
        (count, dim // FP8_BLOCK_SIZE), PAD_SCALE_BYTE, dtype=torch.uint8
    )
    return payload, scale


@dataclass
class HostShard:
    """One rank's owned Engram rows, resident in host RAM.

    ``payload`` is FP8-E4M3 ``[valid_rows, dim]`` and ``scale`` is uint8
    E8M0 ``[valid_rows, dim // 32]`` (32-value groups along the embedding
    dim). Only the owned slice is ever materialized on this rank; the full
    table never lives here. ``reads`` counts table rows read (test oracle
    for owner-only access).
    """

    payload: torch.Tensor
    scale: torch.Tensor
    layout: EngramShardLayout
    rank: int
    dim: int
    pinned: bool = False
    reads: int = field(default=0)

    def __post_init__(self) -> None:
        if self.payload.ndim != 2 or self.payload.shape[1] != self.dim:
            raise ValueError("HostShard payload must be [rows, dim]")
        expected = (self.payload.shape[0], self.dim // FP8_BLOCK_SIZE)
        if tuple(self.scale.shape) != expected:
            raise ValueError(f"HostShard scale must be {expected}")
        if self.payload.device.type != "cpu" or self.scale.device.type != "cpu":
            raise ValueError("Engram host shards must stay in host RAM")

    @classmethod
    def from_tensors(
        cls,
        full_payload: torch.Tensor,
        full_scale: torch.Tensor,
        layout: EngramShardLayout,
        rank: int,
    ) -> "HostShard":
        """Slice ONLY this rank's owned rows out of a full table.

        The full tensors are supplied by the caller (synthetic tables in
        tests, or the checkpoint loader); this rank copies out and retains
        just ``[row_start, row_start + valid_rows)``.
        """
        if full_payload.ndim != 2:
            raise ValueError("full_payload must be [num_embeddings, dim]")
        if full_payload.shape[0] != layout.num_embeddings:
            raise ValueError(
                f"payload rows {full_payload.shape[0]} != num_embeddings "
                f"{layout.num_embeddings}"
            )
        if full_payload.dtype != torch.float8_e4m3fn:
            raise ValueError("Engram payload must be float8_e4m3fn")
        dim = int(full_payload.shape[1])
        if dim % FP8_BLOCK_SIZE:
            raise ValueError("embedding dim must be a multiple of 32")
        expected = (layout.num_embeddings, dim // FP8_BLOCK_SIZE)
        if tuple(full_scale.shape) != expected:
            raise ValueError(f"full_scale must be {expected}")
        if full_scale.dtype != torch.uint8:
            if full_scale.element_size() != 1:
                raise ValueError("Engram scales must be byte-sized")
            full_scale = full_scale.view(torch.uint8)
        start = layout.row_start(rank)
        count = layout.valid_rows(rank)
        # Select unchanged bytes: CPU index ops do not implement Float8.
        payload = (
            full_payload.view(torch.uint8)[start : start + count]
            .clone()
            .view(torch.float8_e4m3fn)
        )
        scale = full_scale[start : start + count].clone()
        payload, pinned = _maybe_pin(payload)
        scale, _ = _maybe_pin(scale)
        return cls(
            payload=payload,
            scale=scale,
            layout=layout,
            rank=int(rank),
            dim=dim,
            pinned=pinned,
        )

    @classmethod
    def load_owned_rows(
        cls,
        payload: Union[torch.Tensor, Callable[[], torch.Tensor]],
        scale: Union[torch.Tensor, Callable[[], torch.Tensor]],
        layout: EngramShardLayout,
        rank: int,
    ) -> "HostShard":
        """Materialize this rank's OWNED rows from lazily-opened checkpoint data.

        Clean seam for weight integration: the caller may pass zero-argument
        callables that open a safetensors mapping and slice the owned row
        range (real checkpoint loading is wired later). Only the owned slice
        is materialized; padding rows are never stored (they resolve to zero
        payload + 0x7F scale on lookup). Extra trailing rows (an allocated
        padding tail) are dropped.
        """
        if callable(payload):
            payload = payload()
        if callable(scale):
            scale = scale()
        if payload.ndim != 2:
            raise ValueError("owned payload must be [rows, dim]")
        dim = int(payload.shape[1])
        if dim % FP8_BLOCK_SIZE:
            raise ValueError("embedding dim must be a multiple of 32")
        count = layout.valid_rows(rank)
        if payload.shape[0] < count:
            raise ValueError(
                f"owned payload has {payload.shape[0]} rows, rank {rank} needs {count}"
            )
        if payload.dtype != torch.float8_e4m3fn:
            raise ValueError("Engram payload must be float8_e4m3fn")
        expected = (count, dim // FP8_BLOCK_SIZE)
        if scale.dtype != torch.uint8:
            if scale.element_size() != 1:
                raise ValueError("Engram scales must be byte-sized")
            scale = scale.view(torch.uint8)
        if tuple(scale.shape) != expected:
            if (
                scale.ndim == 2
                and scale.shape[0] >= count
                and scale.shape[1] == expected[1]
            ):
                scale = scale[:count]
            else:
                raise ValueError(f"owned scale must be {expected}")
        payload = payload[:count].view(torch.uint8).clone().view(torch.float8_e4m3fn)
        scale = scale.clone()
        payload, pinned = _maybe_pin(payload)
        scale, _ = _maybe_pin(scale)
        return cls(
            payload=payload,
            scale=scale,
            layout=layout,
            rank=int(rank),
            dim=dim,
            pinned=pinned,
        )

    @property
    def bytes_estimate(self) -> int:
        """Host bytes held by this shard (payload + scales)."""
        return int(
            self.payload.numel() * self.payload.element_size()
            + self.scale.numel() * self.scale.element_size()
        )

    def read_payload(self, rows: torch.Tensor) -> torch.Tensor:
        """Owned-row payload gather (increments the read counter)."""
        rows = rows.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        self.reads += int(rows.numel())
        # CPU index_select does not implement Float8; select unchanged bytes.
        return (
            self.payload.view(torch.uint8)
            .index_select(0, rows)
            .view(torch.float8_e4m3fn)
        )

    def read_scale(self, rows: torch.Tensor) -> torch.Tensor:
        rows = rows.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        return self.scale.index_select(0, rows)


# ---------------------------------------------------------------------------
# Hash math (rtp-llm engram.py is the numerical authority)
# ---------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    # Adapted from rtp-llm models_py/modules/dsv41/engram.py (Miller-Rabin).
    if n < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % prime == 0:
            return n == prime
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        value = pow(a, d, n)
        if value in (1, n - 1):
            continue
        for _ in range(r - 1):
            value = value * value % n
            if value == n - 1:
                break
        else:
            return False
    return True


def _find_next_prime(start: int, seen_primes: set) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_engram_primes(
    layer_ids: Sequence[int],
    max_ngram_size: int,
    n_heads: int,
    vocab_size: int,
    num_embeddings: Union[int, Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per (layer, n-gram size, head) bucket primes plus row offsets.

    Primes are unique and ascending, drawn from one shared sequence starting
    above ``vocab_size - 1``; offsets are the cumsum of the bucket sizes per
    layer. Adapts rtp-llm EngramLayout prime generation.
    """
    layer_ids = tuple(int(x) for x in layer_ids)
    if max_ngram_size < 2 or n_heads < 1:
        raise ValueError("Engram requires at least one head and a 2-gram")
    if isinstance(num_embeddings, int):
        counts = [num_embeddings] * len(layer_ids)
    else:
        counts = [int(x) for x in num_embeddings]
    if len(counts) != len(layer_ids):
        raise ValueError("Engram layer IDs and embedding counts must match")
    seen: set = set()
    primes = []
    for _ in layer_ids:
        per_layer = []
        for _ in range(max_ngram_size - 1):
            current = int(vocab_size) - 1
            sizes = []
            for _ in range(n_heads):
                current = _find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_layer.append(tuple(sizes))
        primes.append(tuple(per_layer))
    flat = [[p for group in layer for p in group] for layer in primes]
    offsets = []
    for sizes in flat:
        running, row = 0, []
        for size in sizes:
            row.append(running)
            running += size
        offsets.append(row)
    for count, sizes in zip(counts, flat):
        if sum(sizes) > count:
            raise ValueError("Engram prime buckets exceed checkpoint table size")
    return (
        torch.tensor(primes, dtype=torch.int64),
        torch.tensor(offsets, dtype=torch.int64),
    )


def compute_hash_multipliers(
    layer_ids: Sequence[int], max_ngram_size: int, compressed_vocab_size: int
) -> torch.Tensor:
    """One odd int64 multiplier per (layer, lookback), bit-identical to rtp-llm.

    multipliers = rng(10007 * layer).integers(0, bound) * 2 + 1 with
    bound = max(1, (int64max // compressed_vocab_size) // 2).
    """
    import numpy as np

    bound = max(1, (np.iinfo(np.int64).max // int(compressed_vocab_size)) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * int(layer_id))
        values = generator.integers(0, bound, size=int(max_ngram_size), dtype=np.int64)
        rows.append(torch.from_numpy(values * 2 + 1))
    return (
        torch.stack(rows)
        if rows
        else torch.empty((0, max_ngram_size), dtype=torch.int64)
    )


@dataclass(frozen=True)
class EngramHashParams:
    """Hash constants for one Engram configuration (the op's input producer)."""

    layer_ids: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    compressed_vocab_size: int
    pad_id: int
    token_map: torch.Tensor  # int64 [vocab]
    multipliers: torch.Tensor  # int64 [L, max_ngram_size]
    primes: torch.Tensor  # int64 [L, max_ngram_size - 1, n_heads]
    offsets: torch.Tensor  # int64 [L, (max_ngram_size - 1) * n_heads]

    @property
    def num_layers(self) -> int:
        return len(self.layer_ids)

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @classmethod
    def build(
        cls,
        *,
        layer_ids: Sequence[int],
        max_ngram_size: int,
        n_heads: int,
        vocab_size: int,
        compressed_vocab_size: int,
        num_embeddings: Union[int, Sequence[int]],
        token_map: Union[Sequence[int], torch.Tensor],
        pad_token_id: int,
    ) -> "EngramHashParams":
        layer_ids = tuple(int(x) for x in layer_ids)
        token_map_t = torch.as_tensor(token_map, dtype=torch.int64)
        if token_map_t.ndim != 1:
            raise ValueError("token_map must be 1-D")
        if int(token_map_t.numel()) and int(token_map_t.max()) + 1 != int(
            compressed_vocab_size
        ):
            raise ValueError(
                f"compressed vocab mismatch: tokenizer={int(token_map_t.max()) + 1}, "
                f"config={compressed_vocab_size}"
            )
        primes, offsets = build_engram_primes(
            layer_ids, max_ngram_size, n_heads, vocab_size, num_embeddings
        )
        multipliers = compute_hash_multipliers(
            layer_ids, max_ngram_size, int(compressed_vocab_size)
        )
        return cls(
            layer_ids=layer_ids,
            max_ngram_size=int(max_ngram_size),
            n_heads=int(n_heads),
            compressed_vocab_size=int(compressed_vocab_size),
            pad_id=int(token_map_t[int(pad_token_id)]),
            token_map=token_map_t,
            multipliers=multipliers,
            primes=primes,
            offsets=offsets,
        )


def compressed_vocab_map(tokenizer) -> tuple[list[int], int]:
    """Token id -> compressed id plus the compressed vocab size.

    Uses the checkpoint tokenizer's exact training-time normalization
    (adapted from rtp-llm engram.py; ``tokenizers`` is imported lazily).
    """
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    keys: dict[str, int] = {}
    lookup = []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            key = normalizer.normalize_str(text) or text
        if key not in keys:
            keys[key] = len(keys)
        lookup.append(keys[key])
    return lookup, len(keys)


def load_token_map(tokenizer_path: str, cache_path=None) -> list[int]:
    """Load the tokenizer's Engram compressed-vocab map with optional JSON caching."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as handle:
            return json.load(handle)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            os.path.dirname(os.path.abspath(tokenizer_path)), trust_remote_code=False
        )
    except Exception:
        from tokenizers import Tokenizer

        class FastTokenizer:
            backend_tokenizer = Tokenizer.from_file(tokenizer_path)

            def __len__(self):
                return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)

        tokenizer = FastTokenizer()

    lookup, _ = compressed_vocab_map(tokenizer)
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as handle:
            json.dump(lookup, handle)
    return lookup


def build_token_windows(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lookback_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Windows [T, depth + 1] = [current, prev, prev-2, ...] without crossing
    requests; ``lookback_token_ids[b]`` is most-recent-first, -1 before the
    sequence starts. Adapted from rtp-llm engram.py ``make_token_windows``.
    """
    ids = input_ids.detach().to(device="cpu", dtype=torch.int64).flatten()
    boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    lookback = lookback_token_ids.detach().to(device="cpu", dtype=torch.int64)
    depth = lookback.shape[1]
    windows = torch.empty((ids.numel(), depth + 1), dtype=torch.int64, device="cpu")
    for request, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start == end:
            continue
        history = torch.cat((lookback[request].flip(0), ids[start:end]))
        # unfold presents oldest first; hashing uses current first.
        windows[start:end] = history.unfold(0, depth + 1, 1).flip(1)
    return windows


build_token_window = build_token_windows


def _torch_window_hash(
    windows: torch.Tensor,
    dead_mask: Optional[torch.Tensor],
    params: EngramHashParams,
) -> torch.Tensor:
    """Pure-torch reference: rtp-llm engram.py hash math, int64 ops."""
    blocked = windows < 0
    if dead_mask is not None:
        blocked = blocked | dead_mask.to(dtype=torch.bool)
    # An image or sequence boundary terminates every longer n-gram.
    blocked = blocked.to(torch.int32).cumsum(dim=1) > 0
    mapped = params.token_map[windows.clamp_min(0)]
    mapped = torch.where(blocked, params.pad_id, mapped)
    layers = params.num_layers
    heads = params.n_heads
    hashes = torch.empty(
        (windows.shape[0], layers, params.n_hash_cols),
        dtype=torch.int64,
        device=windows.device,
    )
    rolling = torch.zeros((windows.shape[0], layers), dtype=torch.int64)
    for shift in range(params.max_ngram_size):
        rolling ^= mapped[:, shift, None] * params.multipliers[None, :, shift]
        if shift:
            start = (shift - 1) * heads
            end = start + heads
            hashes[:, :, start:end] = (
                rolling[:, :, None] % params.primes[None, :, shift - 1]
                + params.offsets[None, :, start:end]
            )
    return hashes


def compute_engram_hash_ids(
    window: torch.Tensor,
    params: EngramHashParams,
    dead_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Hash token windows [T, n] into row ids [T, L, (n - 1) * heads] int64.

    Column 0 is the current token, column s its s-th predecessor; a negative
    id (or ``dead_mask``) terminates every longer n-gram. The torch path is
    the rtp-llm reference; on a PPU device the sglang-for-sail Triton hash
    (adapted) runs instead with bit-identical int64 arithmetic.
    """
    windows = window.detach()
    if windows.dtype != torch.int64:
        windows = windows.to(torch.int64)
    if windows.dim() != 2:
        raise ValueError("token windows must be [T, max_ngram_size]")
    if windows.shape[1] != params.max_ngram_size:
        raise ValueError(
            f"window width {windows.shape[1]} != max_ngram_size {params.max_ngram_size}"
        )
    origin = windows.device
    if origin.type != "cpu":
        kernels = _triton_kernels()
        if (
            kernels is not None
            and params.max_ngram_size & (params.max_ngram_size - 1) == 0
            and params.n_heads & (params.n_heads - 1) == 0
        ):
            dead = (
                None
                if dead_mask is None
                else dead_mask.detach().to(device=origin, dtype=torch.int8).contiguous()
            )
            return kernels.engram_hash_windows(
                windows.contiguous(),
                dead,
                params.token_map.to(origin),
                params.multipliers.to(origin),
                params.primes.to(origin),
                params.offsets.to(origin),
                params.pad_id,
            )
    windows = windows.to(device="cpu")
    dead = None if dead_mask is None else dead_mask.detach().to(device="cpu")
    return _torch_window_hash(windows, dead, params).to(device=origin)


# ---------------------------------------------------------------------------
# Dequantization
# ---------------------------------------------------------------------------


def _scale_to_float(scales: torch.Tensor) -> torch.Tensor:
    """E8M0 exponent bytes to float32: (byte << 23) viewed as f32 (rtp-llm)."""
    if scales.dtype == torch.uint8:
        return (scales.to(torch.int32) << 23).view(torch.float32)
    return scales.to(torch.float32)


def dequant_rows(payload: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """bf16 = float8(row) * 2**(scale - 127), torch reference (rtp-llm).

    ``payload`` [K, dim] fp8-e4m3, ``scale`` [K, dim // 32] uint8 E8M0 with
    32-value groups along the embedding dim.
    """
    if payload.ndim != 2:
        raise ValueError("payload must be [rows, dim]")
    dim = int(payload.shape[1])
    if dim % FP8_BLOCK_SIZE:
        raise ValueError("embedding dim must be a multiple of 32")
    expected = (payload.shape[0], dim // FP8_BLOCK_SIZE)
    if tuple(scale.shape) != expected:
        raise ValueError(f"scale must be {expected}")
    values = payload.to(torch.float32)
    scale_f = _scale_to_float(scale).repeat_interleave(FP8_BLOCK_SIZE, dim=1)
    return (values * scale_f).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Lookup protocol
# ---------------------------------------------------------------------------


def _round_lengths(group_max: int, max_rows: int) -> list[int]:
    if group_max <= 0:
        return []
    lengths = []
    remaining = group_max
    while remaining > 0:
        take = min(max_rows, remaining)
        lengths.append(take)
        remaining -= take
    return lengths


def _round_chunk(flat: torch.Tensor, offset: int, length: int) -> torch.Tensor:
    """This rank's round chunk, tail-padded with the invalid row id -1."""
    chunk = torch.full((length,), DEAD_ID, dtype=torch.int64, device=flat.device)
    count = min(length, flat.numel() - offset)
    if count > 0:
        chunk[:count] = flat[offset : offset + count]
    return chunk


def _lookup_round(
    gathered_ids: torch.Tensor,
    rank: int,
    shard: HostShard,
    layout: EngramShardLayout,
    device: torch.device,
) -> torch.Tensor:
    """Owner contribution for one round: bf16 [numel, dim], zeros elsewhere.

    Only rows this rank owns are read from the host shard; invalid rows
    (negative, past the table, or padding) contribute zeros. Host->device
    copies are chunked at COPY_BATCH_ROWS rows.
    """
    ids = gathered_ids.detach().reshape(-1)
    n = int(ids.numel())
    dim = shard.dim
    if n == 0:
        return torch.zeros((0, dim), dtype=torch.bfloat16, device=device)
    row_start = layout.row_start(rank)
    row_hi = row_start + layout.valid_rows(rank)
    if shard.pinned and ids.device.type != "cpu":
        kernels = _triton_kernels()
        if kernels is not None:
            # PPU fast path (sglang-for-sail engram_gather): the kernel
            # reads only [row_lo, row_hi) rows and zero-fills the rest.
            out = torch.empty((n, dim), dtype=torch.bfloat16, device=ids.device)
            kernels.engram_gather(
                shard.payload.data_ptr(),
                shard.scale.data_ptr(),
                ids.contiguous(),
                out,
                dim,
                FP8_BLOCK_SIZE,
                row_lo=row_start,
                row_hi=row_hi,
            )
            return out
    host_ids = ids.to(device="cpu", dtype=torch.int64)
    owned = (host_ids >= row_start) & (host_ids < row_hi)
    out = torch.zeros((n, dim), dtype=torch.bfloat16, device=device)
    positions = owned.nonzero(as_tuple=False).reshape(-1)
    total = int(positions.numel())
    if total == 0:
        return out
    local = host_ids[positions] - row_start
    for begin in range(0, total, COPY_BATCH_ROWS):
        end = min(begin + COPY_BATCH_ROWS, total)
        payload = shard.read_payload(local[begin:end])
        scale = shard.read_scale(local[begin:end])
        values = dequant_rows(payload, scale)
        out[positions[begin:end]] = values.to(device=device, dtype=torch.bfloat16)
    return out


class DistributedEnpmGroup:
    """torch.distributed adapter exposing the ENP collective surface.

    rank/size/all_gather_ids/all_reduce_sum/all_max_count is exactly the
    protocol LocalEnpmGroup simulates, so the two are interchangeable.
    """

    def __init__(self, process_group=None):
        self.pg = process_group

    @property
    def rank(self) -> int:
        import torch.distributed as dist

        return dist.get_rank(self.pg)

    @property
    def size(self) -> int:
        import torch.distributed as dist

        return dist.get_world_size(self.pg)

    def all_gather_ids(self, local: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        local = local.contiguous()
        out = torch.empty(
            self.size * local.numel(), dtype=local.dtype, device=local.device
        )
        dist.all_gather_into_tensor(out, local, group=self.pg)
        return out

    def all_reduce_sum(self, values: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        dist.all_reduce(values, op=dist.ReduceOp.SUM, group=self.pg)
        return values

    def all_max_count(self, count: int, device) -> int:
        import torch.distributed as dist

        scalar = torch.tensor([int(count)], dtype=torch.int64, device=device)
        dist.all_reduce(scalar, op=dist.ReduceOp.MAX, group=self.pg)
        return int(scalar.item())

    def reduce_scatter_sum(self, values, local_rows):
        import torch.distributed as dist

        local = values.new_empty((local_rows, values.shape[1]))
        dist.reduce_scatter_tensor(local, values, group=self.pg)
        return local


@dataclass
class _LocalEngramLease:
    active: bool = True
    records: list = field(default_factory=list)
    schedule: object = field(default=None, repr=False)


def _engram_host_tensor_key(tensor):
    version = None if tensor.is_inference() else tensor._version
    return (tensor._cdata, tensor.data_ptr(), tensor.shape, tensor.stride(),
            tensor.storage_offset(), tensor.dtype, tensor.device, version)


def _capture_local_engram_bindings(bindings, rank):
    """Keep the exact native host tensor identities alive with their keys."""
    import torch.distributed as dist

    captured, geometry = [], []
    try:
        for layer in (1, 14):
            shard, layout, adapter = bindings[layer]
            tensors = shard.payload, shard.scale
            if not (
                isinstance(shard, HostShard) and isinstance(layout, EngramShardLayout)
                and type(adapter) is DistributedEnpmGroup and adapter.pg is dist.group.WORLD
                and shard.layout is layout and shard.rank == rank and layout.enp_size == 8
                and shard.dim == 256
                and tensors[0].dtype == torch.float8_e4m3fn and tensors[1].dtype == torch.uint8
                and tensors[0].shape == (layout.valid_rows(rank), 256)
                and tensors[1].shape == (layout.valid_rows(rank), 8)
                and all(t.device.type == "cpu" and t.layout == torch.strided and t.is_contiguous()
                        and not t.is_neg() and not t.is_conj() for t in tensors)
            ):
                raise ValueError("invalid native Engram binding")
            fields = (layout.num_embeddings, layout.padded_rows, layout.num_per_part, layout.enp_size, shard.dim)
            if any(type(value) is not int or not 0 <= value < 2**62 for value in fields):
                raise ValueError("invalid native Engram geometry")
            geometry.extend(fields)
            captured.append((layer, shard, layout, adapter, fields,
                             tuple(_engram_host_tensor_key(t) for t in tensors), tensors))
    except (AttributeError, KeyError, TypeError, ValueError):
        return (), (0,) * 10, True
    return tuple(captured), tuple(geometry), False


@dataclass(frozen=True)
class LocalPrefillEngramSchedule:
    """One jointly admitted eager WORLD8 forward, including idle owners."""

    group: object
    rank: int
    epoch: int
    device: torch.device
    round_cap: int
    layers: tuple
    _bindings: tuple = field(repr=False, compare=False)
    _lease: _LocalEngramLease = field(repr=False, compare=False)

    def claim(self, adapter, shard, layout, rows, layer_id):
        import torch.distributed as dist

        if not (
            self._lease.active and self._lease.schedule is self
            and type(adapter) is DistributedEnpmGroup
            and adapter.pg is self.group is dist.group.WORLD
            and dist.is_initialized() and adapter.size == 8
            and adapter.rank == self.rank == shard.rank
            and layout.enp_size == 8 and shard.dim == 256 and shard.pinned
            and rows.device == self.device and rows.is_cuda
            and rows.layout == torch.strided and not rows.is_neg() and not rows.is_conj()
            and rows.ndim == 2 and rows.shape[1] == 24
            and len(self._lease.records) < len(self.layers)
            and layer_id == self.layers[len(self._lease.records)]
        ):
            raise RuntimeError("local Prefill Engram schedule binding/lifecycle changed after admission")
        bound = self._bindings[len(self._lease.records)]
        fields = (layout.num_embeddings, layout.padded_rows, layout.num_per_part, layout.enp_size, shard.dim)
        if (bound[0] != layer_id or bound[1] is not shard or bound[2] is not layout
                or bound[3] is not adapter or bound[4] != fields
                or bound[5] != tuple(_engram_host_tensor_key(t) for t in (shard.payload, shard.scale))):
            raise RuntimeError("local Prefill Engram host binding changed after admission")
        record = dict(layer=int(layer_id), local_rows=rows.numel(), group_max_rows=None,
                      rounds=[], reduce_scatter=True, complete=False, contribution_bytes=0)
        self._lease.records.append(record)
        return record

    def close(self):
        if self._lease.active and self._lease.schedule is not self:
            raise RuntimeError("local Prefill Engram lease belongs to another schedule")
        complete = (self._lease.active and len(self._lease.records) == len(self.layers)
                    and all(record["complete"] for record in self._lease.records))
        self._lease.active = False
        self._lease.schedule = None
        return complete


def prepare_local_prefill_engram_schedule(
    device, *, requested, supported, epoch, process_group, fatal=False,
    round_cap=LOCAL_PREFILL_ENGRAM_ROUND_CAP, layers=(1, 14), evidence=None, bindings=None,
):
    """All P-role WORLD8 owners enter, even with the rollout flag disabled.

    Unsupported owners agree on the old schedule. Flag/cap/epoch, group or
    lifecycle disagreement is a coherent error before any layer collective.
    No caller may turn a rejected, expired or partially used lease into a
    local fallback after this agreement.
    """
    import torch.distributed as dist

    if (not dist.is_initialized() or dist.get_world_size(dist.group.WORLD) != 8
            or device.type != "cuda" or torch.cuda.is_current_stream_capturing()):
        raise RuntimeError("local Prefill Engram agreement requires eager CUDA/PPU WORLD8")
    invalid = (type(requested) is not bool or type(supported) is not bool
               or type(epoch) is not int or not 1 <= epoch < 2**62
               or type(round_cap) is not int or round_cap != LOCAL_PREFILL_ENGRAM_ROUND_CAP
               or not isinstance(layers, (tuple, list)) or tuple(layers) != (1, 14)
               or process_group is not dist.group.WORLD)
    cap = round_cap if type(round_cap) is int and 0 <= round_cap < 2**62 else 0
    generation = epoch if type(epoch) is int and 0 <= epoch < 2**62 else 0
    enabled = int(requested) if type(requested) is bool else 0
    captured, geometry, binding_bad = _capture_local_engram_bindings(bindings, dist.get_rank(dist.group.WORLD))
    dimensions = [item for value in geometry for item in (value, -value)]
    packet = torch.tensor([enabled, -enabled, cap, -cap, generation, -generation,
                           int(supported is not True), int(bool(fatal or invalid or (enabled and binding_bad))),
                           *dimensions],
                          dtype=torch.int64, device=device)
    # The control group is WORLD even if a local binding is invalid. Peers
    # must reach the same reject decision, not enter different groups.
    dist.all_reduce(packet, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
    values = packet.tolist()
    on, neg_on, maximum, neg_minimum, current, neg_previous, unsupported, rejected = values[:8]
    result = evidence if evidence is not None else {}
    result.clear()
    result.update(decision="rejected", epoch=generation, round_cap=cap, packet=values,
                  rank=dist.get_rank(dist.group.WORLD), consumed_layers=[])
    mismatch = any(values[i] != -values[i + 1] for i in range(8, len(values), 2))
    if rejected or on != -neg_on or maximum != -neg_minimum or current != -neg_previous or mismatch:
        raise RuntimeError(f"local Prefill Engram configuration/lifecycle disagreement: {values!r}")
    if not on or unsupported:
        result["decision"] = "disabled" if not on else "fallback"
        return None
    lease = _LocalEngramLease()
    result.update(decision="adopted", round_cap=maximum, consumed_layers=lease.records)
    schedule = LocalPrefillEngramSchedule(dist.group.WORLD, dist.get_rank(dist.group.WORLD),
                                         current, device, maximum, tuple(layers), captured, lease)
    lease.schedule = schedule
    return schedule


class LocalEnpmGroup:
    """Single-process simulation of the ENP group for CPU tests.

    Holds every rank's HostShard and drives the exact engram_lookup round
    protocol: fixed-rank-order id gather, owner-only host reads into zero
    contributions, sum combine, per-rank slicing in original query order.
    Swapping this for a DistributedEnpmGroup (torch.distributed) changes no
    protocol logic: both run the shared _round_lengths / _round_chunk /
    _lookup_round sequence.
    """

    def __init__(
        self, shards: Sequence[HostShard], layout: Optional[EngramShardLayout] = None
    ):
        self.shards = list(shards)
        if not self.shards:
            raise ValueError("LocalEnpmGroup needs at least one shard")
        self.layout = layout if layout is not None else self.shards[0].layout
        self.size = len(self.shards)
        self.rank = 0
        self.rounds_executed = 0
        self.round_lengths: list[int] = []
        self.gathered_rounds: list[torch.Tensor] = []
        self.collectives: list[tuple[str, int]] = []

    def lookup(
        self,
        per_rank_rows: Sequence[Optional[torch.Tensor]],
        *,
        layer_id: int = 0,
        max_rows_per_round: int = COPY_BATCH_ROWS,
    ) -> list[torch.Tensor]:
        if len(per_rank_rows) != self.size:
            raise ValueError("one row request per rank required")
        flats: list[torch.Tensor] = []
        shapes: list[tuple[int, ...]] = []
        for rows in per_rank_rows:
            if rows is None:
                rows = torch.empty(0, dtype=torch.int64)
            flats.append(rows.detach().reshape(-1).to(torch.int64))
            shapes.append(tuple(rows.shape))
        device = flats[0].device
        group_max = max(f.numel() for f in flats)
        lengths = _round_lengths(group_max, int(max_rows_per_round))
        self.round_lengths = lengths
        self.rounds_executed = len(lengths)
        self.gathered_rounds = []
        self.collectives = []
        pieces: list[list[torch.Tensor]] = [[] for _ in range(self.size)]
        offset = 0
        for length in lengths:
            chunks = [_round_chunk(f, offset, length) for f in flats]
            gathered = torch.cat(chunks, dim=0)  # fixed rank order
            self.gathered_rounds.append(gathered.clone())
            self.collectives.append(("all_gather_ids", length))
            contributions = [
                _lookup_round(
                    gathered, rank, self.shards[rank], self.layout, gathered.device
                )
                for rank in range(self.size)
            ]
            reduced = contributions[0].clone()
            for extra in contributions[1:]:
                reduced += extra
            self.collectives.append(("all_reduce_sum", length))
            for rank in range(self.size):
                pieces[rank].append(reduced[rank * length : (rank + 1) * length])
            offset += length
        dim = self.shards[0].dim
        results = []
        for rank, (flat, shape) in enumerate(zip(flats, shapes)):
            if pieces[rank]:
                out = torch.cat(pieces[rank], dim=0)[: flat.numel()]
            else:
                out = torch.zeros((0, dim), dtype=torch.bfloat16, device=device)
            results.append(out.reshape(*shape, dim) if shape else out.reshape(0, dim))
        return results


def _resolve_group(group, layout: EngramShardLayout) -> Optional[object]:
    if group is None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return DistributedEnpmGroup()
        if layout.enp_size == 1:
            return None
        # Offline single-process over an ENP>1 shard (wave2-fix): the lone
        # rank can only serve its OWN rows — other ENP ranks' rows
        # zero-fill, exactly the owner semantics of ``_lookup_round``. The
        # offline real-weight drivers load one ENP shard and run no process
        # group; raising here made every device smoke impossible.
        warnings.warn(
            "engram_lookup runs offline with ENP>1: only this rank's shard "
            "rows are visible; rows owned by other ENP ranks contribute zeros",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if isinstance(group, DistributedEnpmGroup):
        return group
    if hasattr(group, "all_gather_ids"):
        return group
    return DistributedEnpmGroup(group)


def engram_lookup(
    rows: torch.Tensor,
    *,
    layer_id: int,
    shard: HostShard,
    layout: EngramShardLayout,
    group=None,
    max_rows_per_round: int = COPY_BATCH_ROWS,
    uniform_rank_rows: bool = False,
    round_schedule: Optional[LocalPrefillEngramSchedule] = None,
) -> torch.Tensor:
    """Collective host-sharded Engram lookup -> bf16 ``[*rows.shape, dim]``.

    ``rows`` holds int64 GLOBAL table row ids; invalid ids (-1 or >=
    num_embeddings) produce zero rows. Input query order is preserved and the
    result is returned on the requesting rank. When ``group`` is None and
    torch.distributed is not initialized, ENP size 1 runs a single-process
    direct lookup.
    """
    if not torch.is_tensor(rows) or rows.dim() == 0:
        raise ValueError("rows must be a non-scalar integer tensor")
    if rows.dtype.is_floating_point:
        raise TypeError("engram row ids must be integral")
    if max_rows_per_round < 1:
        raise ValueError("max_rows_per_round must be positive")
    flat = rows.detach().reshape(-1).to(torch.int64)
    device = rows.device
    adapter = _resolve_group(group, layout)
    record = None
    if round_schedule is not None:
        if not isinstance(round_schedule, LocalPrefillEngramSchedule) or uniform_rank_rows:
            raise RuntimeError("local Prefill Engram requires its independent padded schedule")
        record = round_schedule.claim(adapter, shard, layout, rows, layer_id)
        max_rows_per_round = round_schedule.round_cap
    if adapter is None:
        # Single-process direct lookup: the lone shard owns every valid row.
        out = _lookup_round(flat, shard.rank, shard, layout, device)
        return out.reshape(*rows.shape, shard.dim)
    fixed_rows = False
    if uniform_rank_rows and isinstance(adapter, DistributedEnpmGroup) and rows.is_cuda:
        import torch.distributed as dist

        # Full CP8 has the same padded rows per rank. Only use this proof
        # when the eight participating ENP ranks are the whole CP8 world.
        fixed_rows = adapter.size == dist.get_world_size() == 8
    group_max = flat.numel() if fixed_rows else adapter.all_max_count(flat.numel(), device)
    lengths = _round_lengths(group_max, int(max_rows_per_round))
    if record is not None:
        record.update(group_max_rows=group_max, rounds=list(lengths))
    rank = adapter.rank
    pieces: list[torch.Tensor] = []
    offset = 0
    for length in lengths:
        chunk = flat.narrow(0, offset, length) if fixed_rows else _round_chunk(flat, offset, length)
        gathered = adapter.all_gather_ids(chunk)
        contribution = _lookup_round(gathered, rank, shard, layout, gathered.device)
        if fixed_rows or round_schedule is not None:
            pieces.append(adapter.reduce_scatter_sum(contribution, length))
        else:
            reduced = adapter.all_reduce_sum(contribution)
            pieces.append(reduced[rank * length : (rank + 1) * length])
        offset += length
        if record is not None:
            record["contribution_bytes"] = max(record["contribution_bytes"], contribution.numel() * contribution.element_size())
    if fixed_rows and len(pieces) == 1:
        out = pieces[0]
    elif round_schedule is not None and len(pieces) == 1:
        out = pieces[0][: flat.numel()]
    elif pieces:
        out = torch.cat(pieces, dim=0)[: flat.numel()]
    else:
        out = torch.zeros((0, shard.dim), dtype=torch.bfloat16, device=device)
    if record is not None:
        record["complete"] = True
    return out.reshape(*rows.shape, shard.dim)


# ---------------------------------------------------------------------------
# Model-side WKV projection and gated inject
# ---------------------------------------------------------------------------


def wkv_project(flat_rows: torch.Tensor, wkv_bf16: torch.Tensor) -> torch.Tensor:
    """Plain bf16 linear over the flattened lookup rows.

    ``flat_rows`` [T, n_hash_cols * head_dim] -> [T, (hc_mult + 1) * hidden];
    the real V4.1 projection is wkv_bf16 [25600, 6144].
    """
    if wkv_bf16.ndim != 2:
        raise ValueError("wkv must be [out, in]")
    if flat_rows.shape[-1] != wkv_bf16.shape[1]:
        raise ValueError(
            f"flat_rows last dim {flat_rows.shape[-1]} != wkv input {wkv_bf16.shape[1]}"
        )
    x = flat_rows if flat_rows.dtype == wkv_bf16.dtype else flat_rows.to(wkv_bf16.dtype)
    return F.linear(x, wkv_bf16)


def engram_gate_inject(
    h: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    value: torch.Tensor,
    *,
    eps: float = 1e-20,
    clamp: float = 1e-6,
    token_mask: Optional[torch.Tensor] = None,
    fused_mask: bool = False,
) -> torch.Tensor:
    """Gated Engram inject, matching rtp-llm engram.py:626-639 exactly.

    ``h`` [T, hc, dim]; ``value`` is the WKV output [T, (hc + 1) * dim] with
    one key per hc copy followed by the shared value. The gate is
    sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)) with dot normalized per
    (token, copy); at dot == 0 the reference keeps the POSITIVE root.
    On a PPU device the sglang-for-sail fused Triton gate (adapted) runs.
    """
    if h.ndim != 3:
        raise ValueError("h must be [T, hc, dim]")
    tokens, copies, dim = h.shape
    if (
        (token_mask is None or (
            fused_mask and token_mask.shape == (tokens,)
            and token_mask.dtype == torch.bool and token_mask.device == h.device
            and value.shape == (tokens, (copies + 1) * dim)
            and q_weight.shape == k_weight.shape == (copies, dim)
            and all(
                tensor.device == h.device and tensor.is_contiguous()
                for tensor in (h, value, q_weight, k_weight)
            )
        ))
        and h.device.type != "cpu"
        and h.dtype in (torch.bfloat16, torch.float32)
        and value.dtype == h.dtype
        and q_weight.dtype in (torch.bfloat16, torch.float32)
        and k_weight.dtype in (torch.bfloat16, torch.float32)
    ):
        kernels = _triton_kernels()
        if kernels is not None:
            return kernels.fused_engram_gate(
                h, value, q_weight, k_weight, eps, clamp, token_mask=token_mask,
            )
    key = value[:, : copies * dim].reshape(tokens, copies, dim).float()
    shared = value[:, copies * dim :].reshape(tokens, 1, dim).float()
    hidden = h.float()
    dot = (hidden * q_weight.float() * k_weight.float() * key).sum(-1)
    dot *= torch.rsqrt(hidden.square().mean(-1) + eps)
    dot *= torch.rsqrt(key.square().mean(-1) + eps) / math.sqrt(dim)
    gate_input = dot.abs().clamp_min(clamp).sqrt()
    # where(dot < 0, -sqrt, +sqrt): the positive root survives dot == 0
    # (and -0.0), which torch.copysign would not preserve.
    gate_input = torch.where(dot < 0, -gate_input, gate_input)
    gate = torch.sigmoid(gate_input)
    if token_mask is not None:
        gate = torch.where(token_mask.to(device=h.device)[:, None], gate, 0.0)
    result = (hidden + gate[..., None] * shared).to(h.dtype)
    return result


# ---------------------------------------------------------------------------
# PPU Triton kernels (lazily compiled; sglang-for-sail copies)
# ---------------------------------------------------------------------------

_KERNELS: Optional[types.SimpleNamespace] = None
_KERNELS_FAILED = False


def _software_decode_e4m3() -> bool:
    """Decode raw E4M3 bytes in the gather kernel.

    SM80-class PPU lowering rejects float8e4nv in Triton IR; the software
    decode covers it. Off by default; override with
    RTP_V41_ENGRAM_SW_E4M3=1.
    """
    return bool(int(os.environ.get("RTP_V41_ENGRAM_SW_E4M3", "0")))


def _triton_kernels() -> Optional[types.SimpleNamespace]:
    """Compile the sglang-for-sail Triton kernels on first use.

    triton is imported only here so this module loads on a CPU-only host.
    Returns None when triton is unavailable.
    """
    global _KERNELS, _KERNELS_FAILED
    if _KERNELS is not None:
        return _KERNELS
    if _KERNELS_FAILED:
        return None
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice
    except Exception:
        _KERNELS_FAILED = True
        return None

    # ------------------------------------------------------------------
    # Adapted from sglang-for-sail
    # python/sglang/kernels/ops/embeddings/engram_gather.py (512b03d):
    # software-E4M3 decode gather for device/pinned-host tables. The
    # CUDA-VMM peer path (_PeerShards, engram_gather_peer) is NOT copied.
    # ------------------------------------------------------------------
    _E8M0_ZERO = 2.0**-127

    @triton.jit
    def _e4m3fn_to_float(payload):
        # Decode raw E4M3FN bytes without introducing an FP8 type into Triton IR.
        # This also covers SM80 PPU, where float8e4nv is rejected during lowering.
        bits = payload.to(tl.uint32)
        exponent = (bits >> 3) & 15
        mantissa = bits & 7
        normal = ((exponent + 120) << 23) | (mantissa << 20)
        subnormal = (mantissa.to(tl.float32) * (1.0 / 512.0)).to(
            tl.uint32, bitcast=True
        )
        magnitude = tl.where(exponent == 0, subnormal, normal)
        # E4M3FN has finite exponent-15 values through 448, and two signed NaNs.
        magnitude = tl.where((bits & 127) == 127, 0x7FC00000, magnitude)
        return (magnitude | ((bits & 128) << 24)).to(tl.float32, bitcast=True)

    @triton.jit
    def _engram_gather_kernel(
        w_ptr,
        s_ptr,
        ids_ptr,
        dst_ptr,
        out_ptr,
        row_lo,
        row_hi,
        DIM: tl.constexpr,
        BLK: tl.constexpr,
        E8M0_ZERO: tl.constexpr,
        HAS_DST: tl.constexpr,
        DECODE_E4M3_BYTES: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        idx = tl.load(ids_ptr + row).to(tl.int64)
        dst = tl.load(dst_ptr + row).to(tl.int64) if HAS_DST else row
        # The table holds rows [row_lo, row_hi); an id outside it is not read and
        # comes out as zeros, which is what the sharded all-reduce sums.
        owned = (idx >= row_lo) & (idx < row_hi)
        local = tl.where(owned, idx - row_lo, 0)
        s = s_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
        offs = tl.arange(0, DIM)
        if DECODE_E4M3_BYTES:
            w = w_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
            payload = tl.load(w + local * DIM + offs, mask=owned, other=0)
            vals = _e4m3fn_to_float(payload)
        else:
            w = w_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
            vals = tl.load(w + local * DIM + offs, mask=owned, other=0.0).to(tl.float32)
        exps = tl.load(s + local * (DIM // BLK) + offs // BLK, mask=owned, other=0).to(
            tl.int32
        )
        # 2**(e - 127) from the exponent bits: exact, no exp2 rounding or denormal flush.
        scale = (exps << 23).to(tl.float32, bitcast=True)
        scale = tl.where(exps == 0, E8M0_ZERO, scale)
        scale = tl.where(exps == 255, float("nan"), scale)
        out = tl.where(owned, vals * scale, 0.0)
        tl.store(out_ptr + dst * DIM + offs, out.to(tl.bfloat16))

    def engram_gather(
        weight_ptr: int,
        scale_ptr: int,
        ids: torch.Tensor,
        out: torch.Tensor,
        dim: int,
        block_size: int,
        row_lo: int = 0,
        row_hi: int = 2**62,
        sort_ids: bool = False,
        decode_e4m3_bytes: Optional[bool] = None,
    ) -> torch.Tensor:
        """Gather rows ``ids`` ([N] int) into ``out`` ([N, dim] bf16, contiguous).

        ``weight_ptr`` addresses [rows, dim] fp8 e4m3 bytes and ``scale_ptr``
        [rows, dim // block_size] e8m0 bytes for global rows [row_lo, row_hi);
        both may live in device or pinned host memory. Ids outside the range
        produce zero rows.
        """
        assert dim > 0 and dim & (dim - 1) == 0 and block_size > 0, (dim, block_size)
        assert dim % block_size == 0, (dim, block_size)
        assert ids.device.type != "cpu" and ids.is_contiguous()
        assert ids.dtype in (torch.int32, torch.int64)
        assert out.is_contiguous() and out.dtype == torch.bfloat16
        assert out.device == ids.device and out.shape == (ids.numel(), dim), out.shape
        if decode_e4m3_bytes is None:
            decode_e4m3_bytes = _software_decode_e4m3()
        n = ids.numel()
        if n:
            dst = None
            if sort_ids:
                ids, dst = torch.sort(ids)
            _engram_gather_kernel[(n,)](
                weight_ptr,
                scale_ptr,
                ids,
                dst if dst is not None else ids,
                out,
                row_lo,
                row_hi,
                DIM=dim,
                BLK=block_size,
                E8M0_ZERO=_E8M0_ZERO,
                HAS_DST=dst is not None,
                DECODE_E4M3_BYTES=decode_e4m3_bytes,
            )
        return out

    # ------------------------------------------------------------------
    # Adapted from sglang-for-sail
    # python/sglang/kernels/ops/embeddings/engram_hash.py (512b03d):
    # per-request history rows, MODE_DECODE/VERIFY/EXTEND, commit.
    # ------------------------------------------------------------------
    MODE_DECODE = 0
    MODE_VERIFY = 1
    MODE_EXTEND = 2

    @triton.jit
    def _engram_commit_history_kernel(
        history_ptr,
        tokens_ptr,
        slots_ptr,
        commit_ptr,
        live_ptr,
        gate_ptr,
        HISTORY_STRIDE: tl.constexpr,
        HISTORY_WIDTH: tl.constexpr,
        TOKEN_STRIDE: tl.constexpr,
        BLOCK: tl.constexpr,
        GATED: tl.constexpr,
    ):
        row = tl.program_id(0)
        if GATED:
            # Captured rows past the live batch, or a step the graph does not commit.
            if (tl.load(live_ptr + row) == 0) | (tl.load(gate_ptr) == 0):
                return
        slot = tl.load(slots_ptr + row).to(tl.int64)
        commit = tl.load(commit_ptr + row)
        col = tl.arange(0, BLOCK)
        source_col = commit + col
        valid = col < HISTORY_WIDTH
        old = tl.load(
            history_ptr + slot * HISTORY_STRIDE + source_col,
            mask=valid & (source_col < HISTORY_WIDTH),
            other=0,
        )
        new = tl.load(
            tokens_ptr + row * TOKEN_STRIDE + source_col - HISTORY_WIDTH,
            mask=valid & (source_col >= HISTORY_WIDTH),
            other=0,
        )
        values = tl.where(source_col < HISTORY_WIDTH, old, new)
        # Every lane must finish reading the old row before any lane overwrites it.
        tl.debug_barrier()
        tl.store(history_ptr + slot * HISTORY_STRIDE + col, values, mask=valid)

    def engram_commit_history(
        history: torch.Tensor,
        verify_ids: torch.Tensor,
        req_slots: torch.Tensor,
        commit_lens: torch.Tensor,
        live_rows: Optional[torch.Tensor] = None,
        gate: Optional[torch.Tensor] = None,
    ) -> None:
        """Update distinct live request slots with anchor + correct drafts, not bonus.
        Skips rows whose ``live_rows`` entry is 0, and all rows when ``gate[0]`` is 0.
        """
        bs = req_slots.numel()
        width = history.shape[1]
        if bs == 0 or width == 0:
            return
        assert history.stride(1) == 1 and verify_ids.stride(1) == 1
        assert req_slots.stride(0) == 1 and commit_lens.stride(0) == 1
        _engram_commit_history_kernel[(bs,)](
            history,
            verify_ids,
            req_slots,
            commit_lens,
            live_rows if live_rows is not None else commit_lens,
            gate if gate is not None else commit_lens,
            HISTORY_STRIDE=history.stride(0),
            HISTORY_WIDTH=width,
            TOKEN_STRIDE=verify_ids.stride(0),
            BLOCK=triton.next_power_of_2(width),
            GATED=live_rows is not None,
            num_warps=4,
        )

    @triton.jit
    def _engram_hash_kernel(
        ids_ptr,
        pos_ptr,
        row_ptr,
        starts_ptr,
        slots_ptr,
        hist_ptr,
        token_map_ptr,
        mult_ptr,
        primes_ptr,
        offsets_ptr,
        out_loc_ptr,
        tokens_out_ptr,
        out_ptr,
        num_tokens,
        num_real,
        extend_bs,
        search_steps,
        pad_id,
        image_token_id,
        mm_pad_shift,
        MODE: tl.constexpr,
        BLOCK: tl.constexpr,
        HIST_VIA_SLOTS: tl.constexpr,
        HAS_IMAGE: tl.constexpr,
        COMMIT: tl.constexpr,
        WRITE_TOKENS: tl.constexpr,
        N: tl.constexpr,
        L: tl.constexpr,
        H: tl.constexpr,
        BLOCK_T: tl.constexpr,
        ROWS_FROM_STARTS: tl.constexpr = False,
    ):
        COLS: tl.constexpr = (N - 1) * H
        t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        tmask = t < num_tokens
        real = t < num_real
        # Request row r and the token's offset inside its run for this forward.
        if MODE == 0:
            r = t.to(tl.int64)
            off = t * 0
        elif MODE == 1:
            r = (t // BLOCK).to(tl.int64)
            off = t - (t // BLOCK) * BLOCK
        else:
            if ROWS_FROM_STARTS:
                # Upper bound handles adjacent starts from zero-length requests.
                # Resolve rows in registers instead of materializing repeat_interleave.
                # The batch size and step count are runtime values: as constexprs every
                # new batch size would compile and device-load a fresh specialization
                # mid-serving.
                lo = tl.zeros((BLOCK_T,), tl.int32)
                hi = tl.zeros((BLOCK_T,), tl.int32) + extend_bs
                for _ in range(search_steps):
                    mid = (lo + hi) // 2
                    start = tl.load(
                        starts_ptr + mid, mask=real & (mid < extend_bs), other=0
                    )
                    right = (mid < extend_bs) & (start <= t)
                    lo = tl.where(right, mid + 1, lo)
                    hi = tl.where(right, hi, mid)
                r = tl.maximum(lo - 1, 0).to(tl.int64)
            else:
                r = tl.load(row_ptr + t, mask=real, other=0).to(tl.int64)
            off = t - tl.load(starts_ptr + r, mask=real, other=0).to(tl.int32)
        if HIST_VIA_SLOTS:
            hrow = tl.load(slots_ptr + r, mask=real, other=0).to(tl.int64)
        else:
            hrow = r
        pos = tl.load(pos_ptr + t, mask=tmask, other=0).to(tl.int32)

        t2 = t[:, None]
        s = tl.arange(0, N)[None, :]
        tmask2 = tmask[:, None] & (s >= 0)
        real2 = real[:, None] & (s >= 0)
        off2 = off[:, None]
        # Predecessor at shift s: an earlier token of the same run when s <= off, else
        # history[row, n - 2 - (s - off - 1)]; the history is oldest first.
        from_batch = tl.load(ids_ptr + tl.maximum(t2 - s, 0), mask=tmask2, other=0).to(
            tl.int64
        )
        hcol = tl.minimum(tl.maximum(N - 2 - (s - off2 - 1), 0), N - 2)
        from_hist = tl.load(
            hist_ptr + hrow[:, None] * (N - 1) + hcol, mask=tmask2, other=0
        ).to(tl.int64)
        tok = tl.where(s <= off2, from_batch, from_hist)
        tok = tl.where(real2, tok, 0)
        blk = (pos[:, None] < s) | (real2 == 0)
        if HAS_IMAGE:
            # Scheduler-provided history still carries the multimodal pad ids.
            tok = tl.where(tok >= mm_pad_shift, image_token_id, tok)
            blk = blk | (tok == image_token_id)
        # Once a shift is blocked every older shift is too (cummax along s).
        blocked = tl.cumsum(blk.to(tl.int32), axis=1) > 0
        if WRITE_TOKENS:
            tl.store(tokens_out_ptr + t2 * N + s, tok.to(tl.int32), mask=tmask2)
        if COMMIT:
            # Decode: the token and its n - 2 newest predecessors become the request's
            # history, oldest first. Graph-padded rows (out_cache_loc 0) write nothing.
            live = tl.load(out_loc_ptr + t, mask=real, other=0) != 0
            cmask = real2 & live[:, None] & (s <= N - 2)
            tl.store(
                hist_ptr + hrow[:, None] * (N - 1) + (N - 2 - s),
                tok.to(tl.int32),
                mask=cmask,
            )
        mapped = tl.load(token_map_ptr + tok, mask=tmask2, other=0).to(tl.int64)
        comp = tl.where(blocked, pad_id, mapped)

        h = tl.arange(0, H)[None, :]
        omask = tmask[:, None] & (h >= 0)
        for l in tl.static_range(L):
            mult = tl.load(mult_ptr + l * N + s)
            prod = comp * mult
            for i in tl.static_range(1, N):
                # (i + 1)-gram hash: XOR of the first i + 1 shifts' products.
                rolling = tl.xor_sum(tl.where(s <= i, prod, 0), axis=1)
                primes = tl.load(primes_ptr + l * COLS + (i - 1) * H + h)
                offsets = tl.load(offsets_ptr + l * COLS + (i - 1) * H + h)
                val = rolling[:, None] % primes + offsets
                tl.store(
                    out_ptr + t2 * (L * COLS) + l * COLS + (i - 1) * H + h,
                    val,
                    mask=omask,
                )

    def _launch_hash_kernel(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        mode: int,
        history: torch.Tensor,
        token_map: torch.Tensor,
        multipliers: torch.Tensor,
        primes: torch.Tensor,
        offsets: torch.Tensor,
        pad_id: int,
        num_real: Optional[int],
        req_slots: Optional[torch.Tensor],
        block: int,
        row: Optional[torch.Tensor],
        starts: Optional[torch.Tensor],
        image_token_id: Optional[int],
        mm_pad_shift: int,
        out_cache_loc: Optional[torch.Tensor],
        write_tokens: bool,
        block_t: int,
        extend_bs: int = 0,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = input_ids.shape[0]
        L, N = multipliers.shape
        H = primes.shape[-1]
        assert N & (N - 1) == 0 and H & (H - 1) == 0, (N, H)
        assert primes.shape == (L, N - 1, H) and offsets.shape == (L, (N - 1) * H)
        assert history.dim() == 2 and history.shape[1] == N - 1, history.shape
        if mode == MODE_EXTEND:
            assert (row is not None or extend_bs > 0) and starts is not None
        if out_cache_loc is not None:
            assert (
                mode == MODE_DECODE and req_slots is not None
            ), "commit is decode-only"
            assert out_cache_loc.shape[0] == num_tokens, out_cache_loc.shape
        if num_real is None:
            num_real = num_tokens
        device = input_ids.device
        out = torch.empty(num_tokens, L, (N - 1) * H, dtype=torch.int64, device=device)
        tokens = (
            torch.empty(num_tokens, N, dtype=torch.int32, device=device)
            if write_tokens
            else None
        )
        if num_tokens == 0:
            return out, tokens
        dummy = out  # unused pointer slots; never dereferenced under their constexprs
        _engram_hash_kernel[(triton.cdiv(num_tokens, block_t),)](
            input_ids,
            positions,
            row if row is not None else dummy,
            starts if starts is not None else dummy,
            req_slots if req_slots is not None else dummy,
            history,
            token_map,
            multipliers,
            primes,
            offsets,
            out_cache_loc if out_cache_loc is not None else dummy,
            tokens if tokens is not None else dummy,
            out,
            num_tokens,
            num_real,
            extend_bs,
            extend_bs.bit_length(),
            pad_id,
            image_token_id if image_token_id is not None else -1,
            mm_pad_shift,
            MODE=mode,
            BLOCK=block,
            HIST_VIA_SLOTS=req_slots is not None,
            HAS_IMAGE=image_token_id is not None,
            COMMIT=out_cache_loc is not None,
            WRITE_TOKENS=write_tokens,
            N=N,
            L=L,
            H=H,
            BLOCK_T=block_t,
            ROWS_FROM_STARTS=extend_bs > 0,
            num_warps=4,
        )
        return out, tokens

    @triton.jit
    def _engram_commit_extend_history_kernel(
        history_ptr,
        tokens_ptr,
        slots_ptr,
        starts_ptr,
        lens_ptr,
        out_loc_ptr,
        num_tokens,
        N: tl.constexpr,
        HAS_OUT_LOC: tl.constexpr,
    ):
        r = tl.program_id(0)
        length = tl.load(lens_ptr + r)
        start = tl.load(starts_ptr + r)
        last = tl.minimum(tl.maximum(start + length - 1, 0), num_tokens - 1)
        live = length > 0
        if HAS_OUT_LOC:
            live = live & (tl.load(out_loc_ptr + last) != 0)
        slot = tl.load(slots_ptr + r).to(tl.int64)
        col = tl.arange(0, N)
        value = tl.load(
            tokens_ptr + last * N + (N - 2 - col),
            mask=live & (col < N - 1),
            other=0,
        )
        tl.store(history_ptr + slot * (N - 1) + col, value, mask=live & (col < N - 1))

    def engram_hash_extend_and_commit(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        history: torch.Tensor,
        commit_history: torch.Tensor,
        req_slots: torch.Tensor,
        starts: torch.Tensor,
        lengths: torch.Tensor,
        num_real: int,
        history_via_slots: bool,
        out_cache_loc: Optional[torch.Tensor],
        token_map: torch.Tensor,
        multipliers: torch.Tensor,
        primes: torch.Tensor,
        offsets: torch.Tensor,
        pad_id: int,
        image_token_id: Optional[int] = None,
        mm_pad_shift: int = 0,
    ) -> torch.Tensor:
        """Eager extend: hash without a row-map tensor, then commit in one launch.

        Hashing must finish before history is updated: different token programs may
        still read a request's old history. Keep these as two ordered kernels.
        Empty/padded requests leave live history untouched; the spare padding row is
        intentionally not updated. Request slots must be distinct for live requests.
        """
        bs = req_slots.numel()
        assert 0 <= num_real <= input_ids.numel()
        assert starts.numel() == lengths.numel() == bs
        assert history.is_contiguous() and commit_history.is_contiguous()
        assert all(
            t.ndim == 1 and t.stride(0) == 1 for t in (req_slots, starts, lengths)
        )
        out, tokens = _launch_hash_kernel(
            input_ids,
            positions,
            mode=MODE_EXTEND,
            history=history,
            token_map=token_map,
            multipliers=multipliers,
            primes=primes,
            offsets=offsets,
            pad_id=pad_id,
            num_real=num_real,
            req_slots=req_slots if history_via_slots else None,
            block=1,
            row=None,
            starts=starts,
            image_token_id=image_token_id,
            mm_pad_shift=mm_pad_shift,
            out_cache_loc=None,
            write_tokens=True,
            block_t=32,
            extend_bs=bs,
        )
        if input_ids.numel() and bs:
            _engram_commit_extend_history_kernel[(bs,)](
                commit_history,
                tokens,
                req_slots,
                starts,
                lengths,
                out_cache_loc if out_cache_loc is not None else req_slots,
                input_ids.numel(),
                N=multipliers.shape[1],
                HAS_OUT_LOC=out_cache_loc is not None,
                num_warps=4,
            )
        return out

    def engram_hash_ids(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        mode: int,
        history: torch.Tensor,
        token_map: torch.Tensor,
        multipliers: torch.Tensor,
        primes: torch.Tensor,
        offsets: torch.Tensor,
        pad_id: int,
        num_real: Optional[int] = None,
        req_slots: Optional[torch.Tensor] = None,
        block: int = 1,
        row: Optional[torch.Tensor] = None,
        starts: Optional[torch.Tensor] = None,
        image_token_id: Optional[int] = None,
        mm_pad_shift: int = 0,
        block_t: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hash ids [T, L, (n - 1) * heads] int64 and the predecessor table [T, n] int32.
        Reads ``history`` and never writes it.

        ``mode``: MODE_DECODE (row = t, offset 0), MODE_VERIFY (row = t // block),
        MODE_EXTEND (``row`` [num_real] and ``starts`` [bs] give each token's request and
        the run's first token). ``history`` is [rows, n - 1] oldest first; with
        ``req_slots`` given it is indexed by ``req_slots[row]``, else by ``row``.
        Tokens at or past ``num_real`` are padding: PAD ids, zero predecessors.
        """
        out, tokens = _launch_hash_kernel(
            input_ids,
            positions,
            mode=mode,
            history=history,
            token_map=token_map,
            multipliers=multipliers,
            primes=primes,
            offsets=offsets,
            pad_id=pad_id,
            num_real=num_real,
            req_slots=req_slots,
            block=block,
            row=row,
            starts=starts,
            image_token_id=image_token_id,
            mm_pad_shift=mm_pad_shift,
            out_cache_loc=None,
            write_tokens=True,
            block_t=block_t,
        )
        return out, tokens

    def engram_hash_ids_and_commit(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        history: torch.Tensor,
        req_slots: torch.Tensor,
        out_cache_loc: torch.Tensor,
        token_map: torch.Tensor,
        multipliers: torch.Tensor,
        primes: torch.Tensor,
        offsets: torch.Tensor,
        pad_id: int,
        image_token_id: Optional[int] = None,
        mm_pad_shift: int = 0,
        block_t: int = 32,
    ) -> torch.Tensor:
        """One decode step: hash ids [T, L, (n - 1) * heads] for the T = bs tokens, and
        ``history[req_slots[t]]`` advanced in place to the token and its n - 2 newest
        predecessors (oldest first). Rows whose ``out_cache_loc`` is 0 are CUDA-graph
        padding and leave the table untouched.
        """
        out, _ = _launch_hash_kernel(
            input_ids,
            positions,
            mode=MODE_DECODE,
            history=history,
            token_map=token_map,
            multipliers=multipliers,
            primes=primes,
            offsets=offsets,
            pad_id=pad_id,
            num_real=None,
            req_slots=req_slots,
            block=1,
            row=None,
            starts=None,
            image_token_id=image_token_id,
            mm_pad_shift=mm_pad_shift,
            out_cache_loc=out_cache_loc,
            write_tokens=False,
            block_t=block_t,
        )
        return out

    # Window-hash kernel: the hashing stage of _engram_hash_kernel above,
    # taking an explicit [T, n] window (the rtp-llm window contract) instead
    # of ids + positions + history. int64 arithmetic, bit-identical to the
    # torch reference.
    @triton.jit
    def _engram_window_hash_kernel(
        win_ptr,
        dead_ptr,
        token_map_ptr,
        mult_ptr,
        primes_ptr,
        offsets_ptr,
        out_ptr,
        num_tokens,
        pad_id,
        HAS_DEAD: tl.constexpr,
        CHECK_WINDOW: tl.constexpr,
        VOCAB_SIZE: tl.constexpr,
        IMAGE_ID: tl.constexpr,
        N: tl.constexpr,
        L: tl.constexpr,
        H: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        COLS: tl.constexpr = (N - 1) * H
        t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        tmask = t < num_tokens
        t2 = t[:, None]
        s = tl.arange(0, N)[None, :]
        tmask2 = tmask[:, None] & (s >= 0)
        win = tl.load(win_ptr + t2 * N + s, mask=tmask2, other=-1).to(tl.int64)
        blk = win < 0
        if CHECK_WINDOW:
            in_range = (win >= 0) & (win < VOCAB_SIZE)
            blk = ~in_range | (win == IMAGE_ID)
            current_dead = tl.sum((blk & (s == 0)).to(tl.int32), axis=1) > 0
            map_mask = tmask2 & in_range
        else:
            map_mask = tmask2
        if HAS_DEAD:
            d = tl.load(dead_ptr + t2 * N + s, mask=tmask2, other=0)
            blk = blk | (d != 0)
        # Once a shift is blocked every older shift is too.
        blocked = tl.cumsum(blk.to(tl.int32), axis=1) > 0
        mapped = tl.load(token_map_ptr + tl.maximum(win, 0), mask=map_mask, other=0).to(
            tl.int64
        )
        comp = tl.where(blocked, pad_id, mapped)
        h = tl.arange(0, H)[None, :]
        omask = tmask[:, None] & (h >= 0)
        for l in tl.static_range(L):
            mult = tl.load(mult_ptr + l * N + s)
            prod = comp * mult
            for i in tl.static_range(1, N):
                # (i + 1)-gram hash: XOR of the first i + 1 shifts' products.
                rolling = tl.xor_sum(tl.where(s <= i, prod, 0), axis=1)
                primes = tl.load(primes_ptr + l * COLS + (i - 1) * H + h)
                offsets = tl.load(offsets_ptr + l * COLS + (i - 1) * H + h)
                val = rolling[:, None] % primes + offsets
                if CHECK_WINDOW:
                    val = tl.where(current_dead[:, None], -1, val)
                tl.store(
                    out_ptr + t2 * (L * COLS) + l * COLS + (i - 1) * H + h,
                    val,
                    mask=omask,
                )

    def engram_hash_windows(
        window: torch.Tensor,
        dead_mask: Optional[torch.Tensor],
        token_map: torch.Tensor,
        multipliers: torch.Tensor,
        primes: torch.Tensor,
        offsets: torch.Tensor,
        pad_id: int,
        block_t: int = 64,
        *,
        image_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Hash [T, n] windows (column 0 = current token) to [T, L, (n-1)*H] int64."""
        num_tokens = window.shape[0]
        L, N = multipliers.shape
        H = primes.shape[-1]
        assert N & (N - 1) == 0 and H & (H - 1) == 0, (N, H)
        assert primes.shape == (L, N - 1, H) and offsets.shape == (L, (N - 1) * H)
        window = window.contiguous()
        out = torch.empty(
            num_tokens, L, (N - 1) * H, dtype=torch.int64, device=window.device
        )
        if num_tokens == 0:
            return out
        dead = out if dead_mask is None else dead_mask.contiguous()
        _engram_window_hash_kernel[(triton.cdiv(num_tokens, block_t),)](
            window,
            dead,
            token_map,
            multipliers,
            primes,
            offsets,
            out,
            num_tokens,
            pad_id,
            HAS_DEAD=dead_mask is not None,
            CHECK_WINDOW=image_token_id is not None,
            VOCAB_SIZE=token_map.numel(),
            IMAGE_ID=image_token_id if image_token_id is not None else -1,
            N=N,
            L=L,
            H=H,
            BLOCK_T=block_t,
            num_warps=4,
        )
        return out

    @triton.jit
    def _engram_graph_ids_kernel(
        rows, valid, output,
        BATCH: tl.constexpr, COLS: tl.constexpr, CAPACITY: tl.constexpr,
        ROW_STRIDE: tl.constexpr, COL_STRIDE: tl.constexpr, MASK_STRIDE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row = index // COLS
        live = tl.load(valid + row * MASK_STRIDE, row < BATCH, other=False)
        ids = tl.load(
            rows + row * ROW_STRIDE + index % COLS * COL_STRIDE,
            mask=(row < BATCH) & live, other=-1,
        )
        tl.store(output + index, ids, index < CAPACITY)

    def engram_graph_ids(rows, valid, output):
        block = 128
        _engram_graph_ids_kernel[(triton.cdiv(output.numel(), block),)](
            rows, valid, output, rows.shape[0], rows.shape[1], output.numel(),
            rows.stride(0), rows.stride(1), valid.stride(0), block, num_warps=4,
        )

    # ------------------------------------------------------------------
    # Adapted from sglang-for-sail
    # python/sglang/kernels/ops/embeddings/engram_gate.py (512b03d):
    # fused FP32 gate with a single final cast to the activation dtype.
    # ------------------------------------------------------------------
    @triton.jit
    def _engram_gate_kernel(
        X,
        KV,
        QW,
        KW,
        IDS,
        TOKEN_MASK,
        O,
        IMAGE_ID,
        D: tl.constexpr,
        HC: tl.constexpr,
        EPS: tl.constexpr,
        CLAMP: tl.constexpr,
        B: tl.constexpr,
        HAS_IDS: tl.constexpr,
        HAS_MASK: tl.constexpr,
        MASK_STRIDE: tl.constexpr,
    ):
        row = tl.program_id(0)
        token = row // HC
        hc = row % HC
        col = tl.arange(0, B)
        mask = col < D
        x = tl.load(X + row * D + col, mask, 0).to(tl.float32)
        key = tl.load(KV + token * (HC + 1) * D + hc * D + col, mask, 0).to(tl.float32)
        q_weight = tl.load(QW + hc * D + col, mask, 0).to(tl.float32)
        k_weight = tl.load(KW + hc * D + col, mask, 0).to(tl.float32)
        rstd_x = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        rstd_key = tl.rsqrt(tl.sum(key * key, 0) / D + EPS)
        if HAS_MASK:
            dot = tl.sum(((x * q_weight) * k_weight) * key, 0) * rstd_x
            dot = dot * (rstd_key * (D**-0.5))
            root = tl.sqrt(tl.maximum(tl.abs(dot), CLAMP))
            gate = tl.sigmoid(tl.where(dot < 0, -root, root))
            gate = tl.where(dot != dot, dot, gate)
            gate = tl.where(tl.load(TOKEN_MASK + token * MASK_STRIDE), gate, 0.0)
        else:
            dot = tl.sum((x * (q_weight * k_weight)) * key, 0) * (rstd_x * rstd_key) * (D**-0.5)
            gate = tl.sigmoid(libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), CLAMP)), dot))
        value = tl.load(KV + token * (HC + 1) * D + HC * D + col, mask, 0).to(
            tl.float32
        )
        out = x + gate * value
        if HAS_IDS:
            out = tl.where(tl.load(IDS + token) == IMAGE_ID, x, out)
        tl.store(O + row * D + col, out, mask)

    def fused_engram_gate(
        x: torch.Tensor,
        kv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        clamp_value: float,
        input_ids: Optional[torch.Tensor] = None,
        image_token_id: Optional[int] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rows whose ``input_ids`` entry is ``image_token_id`` keep ``x`` unchanged."""
        assert x.ndim == 3 and kv.ndim == 2
        t, hc, d = x.shape
        assert kv.shape == (t, (hc + 1) * d)
        assert q_weight.shape == k_weight.shape == (hc, d)
        assert input_ids is None or (
            input_ids.shape == (t,)
            and input_ids.device.type != "cpu"
            and image_token_id is not None
        )
        assert all(
            a.device.type != "cpu" and a.is_contiguous() and a.device == x.device
            for a in (x, kv, q_weight, k_weight)
        )
        assert all(
            a.dtype in (torch.bfloat16, torch.float32)
            for a in (x, kv, q_weight, k_weight)
        )
        out = torch.empty_like(x)
        if t:
            _engram_gate_kernel[(t * hc,)](
                x,
                kv,
                q_weight,
                k_weight,
                input_ids,
                token_mask,
                out,
                image_token_id,
                d,
                hc,
                eps,
                clamp_value,
                triton.next_power_of_2(d),
                input_ids is not None,
                token_mask is not None,
                token_mask.stride(0) if token_mask is not None else 0,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return out

    _KERNELS = types.SimpleNamespace(
        triton=triton,
        MODE_DECODE=MODE_DECODE,
        MODE_VERIFY=MODE_VERIFY,
        MODE_EXTEND=MODE_EXTEND,
        engram_gather=engram_gather,
        engram_commit_history=engram_commit_history,
        engram_hash_ids=engram_hash_ids,
        engram_hash_ids_and_commit=engram_hash_ids_and_commit,
        engram_hash_extend_and_commit=engram_hash_extend_and_commit,
        engram_hash_windows=engram_hash_windows,
        engram_graph_ids=engram_graph_ids,
        fused_engram_gate=fused_engram_gate,
    )
    return _KERNELS
