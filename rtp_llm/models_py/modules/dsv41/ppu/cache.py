"""Per-rank V4.1 KV/state pools for the PPU backend (L2 prefill producer).

One :class:`PPUKVCache` instance owns every pool this rank needs for the
requests it serves (plan section 5, the L2/L3 seam contract):

=================  ===========================  ============================  ========================
pool tag           entry size                  owner layer                   notes
=================  ===========================  ============================  ========================
``global_kv_2``    288 B (FP4 E2M1 + E4M3)     kv-source layers, ratio 2     one pool PER producer
``global_kv_1``    288 B (FP4 E2M1 + E4M3)     kv-source layers, ratio 1     one pool PER producer
``indexer_kv``     68 B (MX FP4 E2M1 + UE8M0)  index-source layers           consumers share by source
``swa_kv``         528 B (FP8 E4M3 + UE8M0)    EVERY layer                   per-layer pool, window ring
``csa_state``      1024 x fp32                 ratio-2 kv-source layers      per-RAW-token compressor state
=================  ===========================  ============================  ========================

Pools are keyed ``(owner_layer_id, tag)`` because several layers produce the
same tag (2/8/14 all produce ``global_kv_2``); a consumer layer never copies —
it reads its kv/index source layer's pool through
:meth:`PPUKVCache.read_source` / :meth:`PPUKVCache.cache_view`.

Physical layout: each pool is a 2D ``uint8`` tensor ``[blocks, block_entries *
entry_bytes]`` (``csa_state`` is fp32 ``[blocks, block_entries * 1024]``);
absolute slot = ``block * block_entries + offset``, exactly the convention of
``rtp_llm.models_py.modules.dsv41.ppu.kv_codec`` (``insert_rows`` /
``gather_rows``). Block tables map request positions to slots; v1 allocates
SWA/CSA capacity for the full request length (no ring compaction yet — the
position -> slot mapping stays the identity the L1 reader uses).

The fixture format (``export_fixture`` / ``load_fixture``) is the P -> D state
hand-off contract: a directory with ``manifest.json`` (format version,
producer snapshot, source CP, per-request lengths, valid regions, config
signature, payload checksum) and ``pools.pt`` (the raw pool tensors, block
tables and per-request bookkeeping). ``load_fixture`` refuses incompatible
fixtures loudly (version, checksum or config signature mismatch).

All top-level imports are stdlib + torch only.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from rtp_llm.models_py.modules.dsv41.ppu import kv_codec
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import PoolDescriptor

FORMAT_VERSION = 1
PAYLOAD_FILE = "pools.pt"
MANIFEST_FILE = "manifest.json"

GLOBAL_KV_TAGS = ("global_kv_2", "global_kv_1")
POOL_TAGS = ("global_kv_2", "global_kv_1", "indexer_kv", "swa_kv", "csa_state")

# Entry geometry fixed by the V4.1 cache format (plan section 5).
GLOBAL_KV_ENTRY_BYTES = 288
INDEXER_KV_ENTRY_BYTES = 68
SWA_KV_ENTRY_BYTES = 528
CSA_STATE_FLOATS = 1024  # [values(512) | scores(512)] per raw token

DEFAULT_BLOCK_ENTRIES = 128
DEFAULT_SWA_BLOCK_ENTRIES = 512
DEFAULT_CSA_BLOCK_ENTRIES = 128

PoolKey = Tuple[int, str]


def _config_int(config, *names, default=None) -> int:
    for name in names:
        if name in config:
            return int(config[name])
    if default is None:
        raise KeyError(f"V4.1 config is missing any of {list(names)}")
    return int(default)


@dataclass(frozen=True)
class CacheGeometry:
    """Layer-role summary a cache is built from (subset of the model config).

    ``kv_source_layer_ids`` / ``index_source_layer_ids`` are the producer sets
    from the checkpoint config; ``compress_ratios`` is sized like the model's
    (at least ``num_layers``). Engram/router fields are irrelevant here.
    """

    num_layers: int
    compress_ratios: Tuple[int, ...]
    kv_source_layer_ids: Tuple[int, ...]
    index_source_layer_ids: Tuple[int, ...]
    window: int = 128
    block_entries: int = DEFAULT_BLOCK_ENTRIES
    swa_block_entries: int = DEFAULT_SWA_BLOCK_ENTRIES

    def ratio(self, layer_id: int) -> int:
        if 0 <= layer_id < len(self.compress_ratios):
            return int(self.compress_ratios[layer_id])
        return 0

    def kv_source_for(self, layer_id: int) -> Optional[int]:
        """Newest kv-source layer at or below ``layer_id`` (None for SWA-only)."""
        sources = [s for s in self.kv_source_layer_ids if s <= layer_id]
        return max(sources) if sources else None

    def index_source_for(self, layer_id: int) -> Optional[int]:
        sources = [s for s in self.index_source_layer_ids if s <= layer_id]
        return max(sources) if sources else None

    @classmethod
    def from_config(cls, config: dict) -> "CacheGeometry":
        num_layers = _config_int(config, "num_hidden_layers", "num_layers")
        ratios = tuple(int(r) for r in config["compress_ratios"])[:num_layers]
        if len(ratios) < num_layers:
            raise ValueError(
                f"compress_ratios has {len(ratios)} entries, need {num_layers}"
            )
        kv_sources = tuple(int(x) for x in config.get("kv_source_layer_ids", []))
        index_sources = tuple(int(x) for x in config.get("index_source_layer_ids", []))
        for source in kv_sources:
            if not 0 <= source < num_layers:
                raise ValueError(f"kv source layer {source} out of range")
            if ratios[source] not in (1, 2):
                raise ValueError(
                    f"kv source layer {source} must have ratio 1 or 2, "
                    f"got {ratios[source]}"
                )
        for source in index_sources:
            if not 0 <= source < num_layers:
                raise ValueError(f"index source layer {source} out of range")
        return cls(
            num_layers=num_layers,
            compress_ratios=ratios,
            kv_source_layer_ids=kv_sources,
            index_source_layer_ids=index_sources,
            window=int(
                _config_int(config, "sliding_window", "window_size", default=128)
            ),
            block_entries=int(config.get("kv_block_entries", DEFAULT_BLOCK_ENTRIES)),
            swa_block_entries=int(
                config.get("swa_block_entries", DEFAULT_SWA_BLOCK_ENTRIES)
            ),
        )

    def signature(self) -> str:
        """Stable identity of the pool topology (fixture compatibility)."""
        payload = dict(
            num_layers=self.num_layers,
            compress_ratios=list(self.compress_ratios),
            kv_source_layer_ids=list(self.kv_source_layer_ids),
            index_source_layer_ids=list(self.index_source_layer_ids),
            window=self.window,
            block_entries=self.block_entries,
            swa_block_entries=self.swa_block_entries,
            format_version=FORMAT_VERSION,
        )
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def pool_keys(self) -> List[PoolKey]:
        keys: List[PoolKey] = []
        for producer in self.kv_source_layer_ids:
            ratio = self.ratio(producer)
            tag = "global_kv_2" if ratio == 2 else "global_kv_1"
            keys.append((producer, tag))
            keys.append((producer, "indexer_kv"))
            if ratio == 2:
                keys.append((producer, "csa_state"))
        for layer in range(self.num_layers):
            keys.append((layer, "swa_kv"))
        return keys


@dataclass
class PPURequestSlots:
    """Slot handles returned by :meth:`PPUKVCache.alloc`.

    Every tensor holds ABSOLUTE pool slots (``block * block_entries +
    offset``) for the request's CURRENT total length; ``alloc`` returns the
    full slot vectors so prefill can write every entry, while decode can pass
    sliced tails. ``global_slots`` / ``indexer_slots`` / ``csa_slots`` are
    keyed by the PRODUCING layer id, ``swa_slots`` by the (owning) layer id.
    """

    request_id: int
    n_tokens: int
    total_tokens: int
    global_slots: Dict[int, torch.Tensor] = field(default_factory=dict)
    indexer_slots: Dict[int, torch.Tensor] = field(default_factory=dict)
    swa_slots: Dict[int, torch.Tensor] = field(default_factory=dict)
    csa_slots: Dict[int, torch.Tensor] = field(default_factory=dict)


class _BlockAllocator:
    """Free-list over one pool's blocks; the pool grows on demand."""

    def __init__(self, blocks: int = 0):
        self.allocated = int(blocks)
        self.free: List[int] = list(range(blocks))

    def take(self, count: int) -> List[int]:
        while len(self.free) < count:
            self.free.append(self.allocated)
            self.allocated += 1
        return [self.free.pop(0) for _ in range(count)]


class PPUKVCache:
    """The rank-local V4.1 KV/state pools.

    Construction: ``PPUKVCache(config_dict_or_geometry, device=...)``. Pools
    start empty and grow as requests allocate blocks.

    Life cycle per request:
      1. ``slots = cache.alloc(request_id, n_tokens)`` — reserves pool space
         for ``n_tokens`` NEW tokens (repeat calls extend the request; total
         length is tracked internally);
      2. ``cache.write(layer_id, slots, global_kv=..., indexer_k=...,
         swa_kv=..., csa_state=...)`` — scatters packed rows produced by
         ``ops.encode_kv`` into the pools this LAYER owns;
      3. consumers read through ``cache.read_source(...)`` /
         ``cache.cache_view(layer_id)`` — never a copy;
      4. ``cache.export_fixture(path, meta)`` freezes the state for the D side
         (or diagnostics); ``PPUKVCache.load_fixture(path)`` restores it.
    """

    def __init__(
        self,
        config=None,
        geometry: Optional[CacheGeometry] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        if geometry is None:
            if config is None:
                raise ValueError("provide a config dict or a CacheGeometry")
            geometry = CacheGeometry.from_config(config)
        self.geometry = geometry
        self.device = torch.device(device) if device is not None else None
        self._pools: Dict[PoolKey, torch.Tensor] = {}
        self._allocators: Dict[PoolKey, _BlockAllocator] = {}
        # pool key -> request id -> list of block ids (position ordered)
        self._block_tables: Dict[PoolKey, Dict[int, List[int]]] = {}
        self._request_tokens: Dict[int, int] = {}
        # pool key -> request id -> valid entry count
        self._valid_entries: Dict[PoolKey, Dict[int, int]] = {}
        for key in geometry.pool_keys():
            self._pools[key] = self._empty_pool(key)
            self._allocators[key] = _BlockAllocator()
            self._block_tables[key] = {}
            self._valid_entries[key] = {}

    # ------------------------------------------------------------------ geometry

    def _entries_per_block(self, tag: str) -> int:
        geo = self.geometry
        if tag == "swa_kv":
            return geo.swa_block_entries
        if tag == "csa_state":
            return geo.block_entries
        return geo.block_entries

    def _entry_shape(self, key: PoolKey) -> Tuple[int, int]:
        """(entries per block, row width in elements) for one pool."""
        tag = key[1]
        entries = self._entries_per_block(tag)
        if tag == "global_kv_2" or tag == "global_kv_1":
            return entries, GLOBAL_KV_ENTRY_BYTES
        if tag == "indexer_kv":
            return entries, INDEXER_KV_ENTRY_BYTES
        if tag == "swa_kv":
            return entries, SWA_KV_ENTRY_BYTES
        if tag == "csa_state":
            return entries, CSA_STATE_FLOATS
        raise ValueError(f"unknown pool tag {tag!r}")

    def _empty_pool(self, key: PoolKey) -> torch.Tensor:
        entries, width = self._entry_shape(key)
        dtype = torch.float32 if key[1] == "csa_state" else torch.uint8
        shape = (0, entries * width)
        if self.device is not None:
            return torch.zeros(shape, dtype=dtype, device=self.device)
        return torch.zeros(shape, dtype=dtype)

    def _grow_pool(self, key: PoolKey, blocks: List[int]) -> None:
        wanted = max(blocks) + 1 if blocks else 0
        pool = self._pools[key]
        if pool.shape[0] >= wanted:
            return
        entries, width = self._entry_shape(key)
        dtype = pool.dtype
        device = pool.device
        extra = torch.zeros(
            (wanted - pool.shape[0], entries * width), dtype=dtype, device=device
        )
        self._pools[key] = torch.cat((pool, extra), dim=0)

    def _row_width(self, key: PoolKey) -> int:
        return self._entry_shape(key)[1]

    def pool(self, owner_layer: int, tag: str) -> torch.Tensor:
        """The raw 2D pool tensor ``(owner_layer, tag)`` (read-only by convention)."""
        return self._pools[(int(owner_layer), tag)]

    def pool_keys(self) -> List[PoolKey]:
        return list(self._pools.keys())

    def descriptors(self, owner_layer: int) -> Dict[str, PoolDescriptor]:
        """L1 ``kv_codec`` descriptors for one owner layer's pools."""
        geo = self.geometry
        out: Dict[str, PoolDescriptor] = {}
        for owner, tag in self._pools:
            if owner != owner_layer:
                continue
            if tag in GLOBAL_KV_TAGS:
                out[tag] = PoolDescriptor(
                    entry_bytes=GLOBAL_KV_ENTRY_BYTES,
                    ratio=geo.ratio(owner),
                    planar=False,
                    block_entries=self._entries_per_block(tag),
                    kind="fp4_e2m1_e4m3",
                )
            elif tag == "indexer_kv":
                out[tag] = PoolDescriptor(
                    entry_bytes=INDEXER_KV_ENTRY_BYTES,
                    ratio=geo.ratio(owner),
                    planar=True,
                    block_entries=self._entries_per_block(tag),
                    kind="fp4_e2m1_ue8m0",
                )
            elif tag == "swa_kv":
                out[tag] = PoolDescriptor(
                    entry_bytes=SWA_KV_ENTRY_BYTES,
                    ratio=1,
                    planar=False,
                    block_entries=self._entries_per_block(tag),
                    window=geo.window,
                    kind="fp8_e4m3_ue8m0",
                )
        return out

    # ------------------------------------------------------------------ alloc

    def request_length(self, request_id: int) -> int:
        return int(self._request_tokens.get(int(request_id), 0))

    def _slots_for_blocks(
        self, key: PoolKey, request_id: int, count: int
    ) -> torch.Tensor:
        table = self._block_tables[key][int(request_id)]
        entries = self._entries_per_block(key[1])
        slots = torch.zeros(count, dtype=torch.int64)
        for entry in range(count):
            block = entry // entries
            offset = entry % entries
            slots[entry] = table[block] * entries + offset
        return slots

    def _extend(self, key: PoolKey, request_id: int, entries_needed: int) -> None:
        entries = self._entries_per_block(key[1])
        table = self._block_tables[key].setdefault(int(request_id), [])
        have = len(table) * entries
        if entries_needed > have:
            blocks = self._allocators[key].take(
                (entries_needed - have + entries - 1) // entries
            )
            self._grow_pool(key, blocks)
            table.extend(blocks)

    def alloc(self, request_id: int, n_tokens: int) -> PPURequestSlots:
        """Reserve pool space for ``n_tokens`` new tokens of one request.

        Returns the full slot vectors for the request's new TOTAL length
        (existing tokens included): ``global_slots``/``indexer_slots`` hold
        ``total // ratio`` entries per producer, ``swa_slots`` one entry per
        layer per token, ``csa_slots`` one fp32 state per raw token for
        ratio-2 producers. Repeated calls extend the same request.
        """
        request_id = int(request_id)
        n_tokens = int(n_tokens)
        if n_tokens < 0:
            raise ValueError("n_tokens must be non-negative")
        geo = self.geometry
        previous = self._request_tokens.get(request_id, 0)
        total = previous + n_tokens
        self._request_tokens[request_id] = total

        slots = PPURequestSlots(
            request_id=request_id, n_tokens=n_tokens, total_tokens=total
        )
        for producer in geo.kv_source_layer_ids:
            ratio = geo.ratio(producer)
            entries = total // ratio
            key = (producer, "global_kv_2" if ratio == 2 else "global_kv_1")
            self._extend(key, request_id, entries)
            slots.global_slots[producer] = self._slots_for_blocks(
                key, request_id, entries
            )
            self._valid_entries[key][request_id] = entries
            index_key = (producer, "indexer_kv")
            self._extend(index_key, request_id, entries)
            slots.indexer_slots[producer] = self._slots_for_blocks(
                index_key, request_id, entries
            )
            self._valid_entries[index_key][request_id] = entries
            if ratio == 2:
                state_key = (producer, "csa_state")
                self._extend(state_key, request_id, total)
                slots.csa_slots[producer] = self._slots_for_blocks(
                    state_key, request_id, total
                )
                self._valid_entries[state_key][request_id] = total
        for layer in range(geo.num_layers):
            key = (layer, "swa_kv")
            self._extend(key, request_id, total)
            slots.swa_slots[layer] = self._slots_for_blocks(key, request_id, total)
            self._valid_entries[key][request_id] = total
        return slots

    # ------------------------------------------------------------------ write / read

    def _check_rows(self, packed: torch.Tensor, key: PoolKey, what: str) -> None:
        width = self._row_width(key)
        if packed.dtype == torch.uint8 and key[1] != "csa_state":
            if packed.dim() != 2 or packed.shape[1] != width:
                raise ValueError(
                    f"{what} rows must be [N, {width}] uint8, got {tuple(packed.shape)}"
                )
            return
        if key[1] == "csa_state":
            if packed.dtype != torch.float32 or packed.shape[-1] != width:
                raise ValueError(
                    f"csa_state rows must be [N, {width}] fp32, got "
                    f"{tuple(packed.shape)} {packed.dtype}"
                )
            return
        raise ValueError(f"{what} rows must be uint8, got {packed.dtype}")

    def _insert(self, key: PoolKey, slots: torch.Tensor, rows: torch.Tensor) -> None:
        pool = self._pools[key]
        entries = self._entries_per_block(key[1])
        width = self._row_width(key)
        # Pools are always 2D [blocks, entries * width]; a literal reshape
        # with -1 is ambiguous (RuntimeError) for empty 0-block pools.
        flat = pool
        slots = slots.to(torch.int64).reshape(-1)
        if slots.numel() != rows.shape[0]:
            raise ValueError(
                f"slot count {slots.numel()} != row count {rows.shape[0]} for {key}"
            )
        for index in range(slots.numel()):
            slot = int(slots[index])
            if slot < 0:
                continue
            block, offset = divmod(slot, entries)
            flat[block, offset * width : (offset + 1) * width] = rows[index]

    def write(
        self,
        layer_id: int,
        slots: PPURequestSlots,
        *,
        global_kv: Optional[torch.Tensor] = None,
        indexer_k: Optional[torch.Tensor] = None,
        swa_kv: Optional[torch.Tensor] = None,
        csa_state: Optional[torch.Tensor] = None,
    ) -> None:
        """Scatter packed rows into the pools OWNED by ``layer_id``.

        ``global_kv`` / ``indexer_k`` are packed uint8 rows from
        ``ops.encode_kv(..., pool_tag="global_kv_2"/"global_kv_1"/"indexer_kv")``
        sized to the producer's entry count; ``swa_kv`` are 528B rows sized to
        the token count; ``csa_state`` is fp32 ``[T, 1024]`` (values|scores per
        raw token). Supplying a pool this layer does not own raises.
        """
        layer_id = int(layer_id)
        geo = self.geometry
        if global_kv is not None or indexer_k is not None:
            if layer_id not in geo.kv_source_layer_ids:
                raise ValueError(
                    f"layer {layer_id} is not a kv-source layer; it owns no "
                    "global/indexer pools"
                )
            ratio = geo.ratio(layer_id)
            tag = "global_kv_2" if ratio == 2 else "global_kv_1"
            if global_kv is not None:
                self._check_rows(global_kv, (layer_id, tag), "global_kv")
                self._insert((layer_id, tag), slots.global_slots[layer_id], global_kv)
            if indexer_k is not None:
                self._check_rows(indexer_k, (layer_id, "indexer_kv"), "indexer_k")
                self._insert(
                    (layer_id, "indexer_kv"), slots.indexer_slots[layer_id], indexer_k
                )
        if csa_state is not None:
            if layer_id not in geo.kv_source_layer_ids or geo.ratio(layer_id) != 2:
                raise ValueError(
                    f"layer {layer_id} is not a ratio-2 kv-source layer; it owns "
                    "no csa_state pool"
                )
            self._check_rows(csa_state, (layer_id, "csa_state"), "csa_state")
            self._insert((layer_id, "csa_state"), slots.csa_slots[layer_id], csa_state)
        if swa_kv is not None:
            self._check_rows(swa_kv, (layer_id, "swa_kv"), "swa_kv")
            self._insert((layer_id, "swa_kv"), slots.swa_slots[layer_id], swa_kv)

    def _gather(self, key: PoolKey, slots: torch.Tensor) -> torch.Tensor:
        pool = self._pools[key]
        entries = self._entries_per_block(key[1])
        width = self._row_width(key)
        # Pools are always 2D [blocks, entries * width]; a literal reshape
        # with -1 is ambiguous (RuntimeError) for empty 0-block pools.
        flat = pool
        slots = slots.to(torch.int64).reshape(-1)
        rows = slots.shape[0]
        if key[1] == "csa_state":
            out = torch.zeros((rows, width), dtype=torch.float32, device=pool.device)
        else:
            out = torch.zeros((rows, width), dtype=torch.uint8, device=pool.device)
        for index in range(rows):
            slot = int(slots[index])
            if slot < 0:
                continue
            block, offset = divmod(slot, entries)
            out[index] = flat[block, offset * width : (offset + 1) * width]
        return out

    def insert_rows(
        self, owner_layer: int, tag: str, slots: torch.Tensor, rows: torch.Tensor
    ) -> None:
        """Partial-write escape hatch: scatter ``rows`` at explicit slots.

        Same pool convention as :meth:`write` but with caller-chosen slot
        tensors (e.g. a chunk's entry slice ``slots[b:e]`` or RAW-position
        slots from :meth:`slots_at_positions`). Rows must be packed uint8
        (or fp32 for ``csa_state``).
        """
        key = (int(owner_layer), tag)
        if key not in self._pools:
            raise ValueError(f"pool ({owner_layer}, {tag}) does not exist")
        self._check_rows(rows, key, tag)
        self._insert(key, slots, rows)

    def slots_at_positions(
        self, owner_layer: int, tag: str, request_id: int, positions: torch.Tensor
    ) -> torch.Tensor:
        """Absolute slots of the swa/csa pool at RAW positions (any order)."""
        key = (int(owner_layer), tag)
        if key not in self._pools:
            raise ValueError(f"pool ({owner_layer}, {tag}) does not exist")
        entries = self._entries_per_block(tag)
        table = self._block_tables[key].get(int(request_id))
        if table is None:
            raise ValueError(f"request {request_id} has no blocks in {key}")
        pos = positions.to(torch.int64).reshape(-1)
        blocks = pos // entries
        if pos.numel() and int(blocks.max()) >= len(table):
            raise ValueError("position outside the request's block table")
        table_tensor = torch.tensor(table, dtype=torch.int64, device=pos.device)
        return table_tensor[blocks] * entries + pos % entries

    def read(
        self,
        layer_id: int,
        slots: Optional[PPURequestSlots] = None,
        request_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Packed rows of every pool ``layer_id`` OWNS.

        With ``slots`` given (or ``request_id`` resolved to this cache's
        handles) the full valid region of that request is gathered; without
        either, every valid entry of the pool is returned. Consumer layers
        must use :meth:`read_source` instead — they own only ``swa_kv``.
        """
        layer_id = int(layer_id)
        out: Dict[str, torch.Tensor] = {}
        geo = self.geometry
        for key in self._pools:
            owner, tag = key
            if owner != layer_id:
                continue
            slot_tensor = None
            if slots is not None:
                if tag in GLOBAL_KV_TAGS:
                    slot_tensor = slots.global_slots.get(layer_id)
                elif tag == "indexer_kv":
                    slot_tensor = slots.indexer_slots.get(layer_id)
                elif tag == "csa_state":
                    slot_tensor = slots.csa_slots.get(layer_id)
                else:
                    slot_tensor = slots.swa_slots.get(layer_id)
            elif request_id is not None:
                count = self._valid_entries[key].get(int(request_id), 0)
                slot_tensor = self._slots_for_blocks(key, int(request_id), count)
            else:
                per_request = self._valid_entries[key]
                if not per_request:
                    slot_tensor = torch.zeros(0, dtype=torch.int64)
                else:
                    request = max(per_request, key=lambda r: per_request[r])
                    slot_tensor = self._slots_for_blocks(
                        key, request, per_request[request]
                    )
            if slot_tensor is None:
                raise ValueError(f"no slots for pool {key} in the given handles")
            out[tag] = self._gather(key, slot_tensor)
        return out

    def read_source(
        self,
        source_layer_id: int,
        request_id: int,
        *,
        global_entries: Optional[int] = None,
        indexer_entries: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Packed (global_kv rows, indexer_kv rows) of one SOURCE layer.

        The reference a consumer layer needs: the source's pools, gathered for
        the request's valid entries (or the counts the caller asks for, e.g.
        a decode tail). No copy of the pool itself — only the packed rows.
        """
        source = int(source_layer_id)
        geo = self.geometry
        if source not in geo.kv_source_layer_ids:
            raise ValueError(f"layer {source} is not a kv-source layer")
        ratio = geo.ratio(source)
        tag = "global_kv_2" if ratio == 2 else "global_kv_1"
        total = self._valid_entries[(source, tag)].get(int(request_id), 0)
        gcount = total if global_entries is None else int(global_entries)
        icount = total if indexer_entries is None else int(indexer_entries)
        gslots = self._slots_for_blocks((source, tag), int(request_id), gcount)
        islots = self._slots_for_blocks((source, "indexer_kv"), int(request_id), icount)
        return self._gather((source, tag), gslots), self._gather(
            (source, "indexer_kv"), islots
        )

    def read_swa(self, layer_id: int, request_id: int, positions: torch.Tensor):
        """Packed 528B SWA rows of ``layer_id`` at RAW positions (any order)."""
        key = (int(layer_id), "swa_kv")
        entries = self._entries_per_block("swa_kv")
        table = self._block_tables[key][int(request_id)]
        pos = positions.to(torch.int64).reshape(-1)
        blocks = pos // entries
        if blocks.numel() and int(blocks.max()) >= len(table):
            raise ValueError("position outside the request's swa block table")
        slots = (
            torch.tensor(table, dtype=torch.int64, device=pos.device)[blocks] * entries
            + pos % entries
        )
        return self._gather(key, slots)

    def read_csa(self, source_layer_id: int, request_id: int, positions: torch.Tensor):
        """fp32 ``[len(positions), 1024]`` CSA states at RAW positions."""
        source = int(source_layer_id)
        key = (source, "csa_state")
        if key not in self._pools:
            raise ValueError(f"layer {source} owns no csa_state pool")
        entries = self._entries_per_block("csa_state")
        table = self._block_tables[key][int(request_id)]
        pos = positions.to(torch.int64).reshape(-1)
        blocks = pos // entries
        if blocks.numel() and int(blocks.max()) >= len(table):
            raise ValueError("position outside the request's csa block table")
        slots = (
            torch.tensor(table, dtype=torch.int64, device=pos.device)[blocks] * entries
            + pos % entries
        )
        return self._gather(key, slots)

    # ------------------------------------------------------------------ L1 cache view

    def cache_view(self, layer_id: int):
        """A ``V41CacheView``-shaped view of THIS layer's readable pools.

        Fields (attributes, matching the frozen contract dataclass):
          * ``pools``: tag -> raw 2D pool tensor;
          * ``block_tables``: tag -> int64 ``[max_request_id + 1, max_blocks]``
            (row = request id VALUE, exactly what L1's attention reader
            indexes);
          * ``descriptors``: tag -> :class:`PoolDescriptor` with
            ``producer_layer_id`` / ``compression_ratio`` set, so the L1
            reader resolves the global pool unambiguously.
        Consumers see their kv/index SOURCE layer's pools plus their own
        ``swa_kv``; producers see their own; SWA-only layers see only
        ``swa_kv`` (they must not call the sparse-attention global path).
        """
        layer_id = int(layer_id)
        geo = self.geometry
        pools = {}
        tables = {}
        descriptors = {}
        source = geo.kv_source_for(layer_id)
        if source is not None:
            ratio = geo.ratio(source)
            tag = "global_kv_2" if ratio == 2 else "global_kv_1"
            pools[tag] = self._pools[(source, tag)]
            tables[tag] = self._table_tensor((source, tag))
            desc = PoolDescriptor(
                entry_bytes=GLOBAL_KV_ENTRY_BYTES,
                ratio=ratio,
                planar=False,
                block_entries=self._entries_per_block(tag),
                kind="fp4_e2m1_e4m3",
            )
            descriptors[tag] = _ViewDescriptor(desc, producer_layer_id=source)
        tag = "swa_kv"
        pools[tag] = self._pools[(layer_id, tag)]
        tables[tag] = self._table_tensor((layer_id, tag))
        descriptors[tag] = _ViewDescriptor(
            PoolDescriptor(
                entry_bytes=SWA_KV_ENTRY_BYTES,
                ratio=1,
                planar=False,
                block_entries=self._entries_per_block(tag),
                window=geo.window,
                kind="fp8_e4m3_ue8m0",
            ),
            producer_layer_id=layer_id,
        )
        return _CacheView(pools=pools, block_tables=tables, descriptors=descriptors)

    def _table_tensor(self, key: PoolKey) -> torch.Tensor:
        table = self._block_tables[key]
        requests = sorted(table.keys())
        width = max((len(blocks) for blocks in table.values()), default=0)
        # The table lives WITH its pool (the cache device when set): L1's
        # reader indexes it with device-side positions — a CPU table under
        # cuda pools is a device-mismatch crash (wave2-fix).
        device = self.device if self.device is not None else torch.device("cpu")
        out = torch.zeros(
            (max(requests, default=0) + 1, max(width, 1)),
            dtype=torch.int64,
            device=device,
        )
        for request, blocks in table.items():
            out[request, : len(blocks)] = torch.tensor(
                blocks, dtype=torch.int64, device=device
            )
        return out

    # ------------------------------------------------------------------ fixtures

    def valid_regions(self) -> Dict[str, Dict[str, int]]:
        """``{"<owner>:<tag>": {"<request_id>": valid_entries}}`` snapshot."""
        regions = {}
        for key, per_request in self._valid_entries.items():
            regions[f"{key[0]}:{key[1]}"] = {
                str(request): int(count) for request, count in per_request.items()
            }
        return regions

    def select_requests(self, request_ids: Sequence[int]) -> "PPUKVCache":
        """Copy only these requests, compacting whole blocks in logical order.

        Indexer blocks are planar: copy complete blocks rather than treating
        their bytes as interleaved entries. CSA phase and SWA positions stay
        in their original logical slots. No other DP owner's blocks survive.
        """
        selected = sorted(set(int(rid) for rid in request_ids))
        missing = set(selected) - self._request_tokens.keys()
        if missing:
            raise ValueError(f"unknown cache requests: {sorted(missing)}")
        result = PPUKVCache(geometry=self.geometry, device=self.device)
        result._request_tokens = {rid: self._request_tokens[rid] for rid in selected}
        for key, pool in self._pools.items():
            blocks = []
            for rid in selected:
                old = self._block_tables[key].get(rid, [])
                start = len(blocks)
                blocks.extend(old)
                result._block_tables[key][rid] = list(range(start, len(blocks)))
                if rid in self._valid_entries[key]:
                    result._valid_entries[key][rid] = self._valid_entries[key][rid]
            indices = torch.tensor(blocks, dtype=torch.int64, device=pool.device)
            result._pools[key] = pool.index_select(0, indices)
            allocator = _BlockAllocator(len(blocks))
            allocator.free.clear()
            result._allocators[key] = allocator
        return result

    def state_checksum(self) -> str:
        """Stable state identity, independent of torch.save's archive metadata.

        Pools are zero-initialized, including padding. This diagnostic hash
        covers the complete allocated blocks and their logical mapping.
        """
        digest = hashlib.sha256()
        digest.update(self.geometry.signature().encode())
        digest.update(json.dumps(self._request_tokens, sort_keys=True).encode())
        for key in sorted(self._pools):
            tensor = self._pools[key].detach().cpu().contiguous()
            metadata = (
                key,
                str(tensor.dtype),
                list(tensor.shape),
                self._block_tables[key],
                self._valid_entries[key],
            )
            digest.update(json.dumps(metadata, sort_keys=True).encode())
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def export_fixture(self, path, meta: Optional[dict] = None) -> dict:
        """Freeze this cache into a fixture directory (the P -> D contract).

        Writes ``manifest.json`` + ``pools.pt`` under ``path``. The manifest
        records the format version, the producer snapshot, the source CP,
        per-request lengths, the valid regions, the pool-geometry signature
        and the SHA256 of the payload file. Returns the manifest.
        """
        meta = dict(meta or {})
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "pools": {f"{k[0]}:{k[1]}": v.clone() for k, v in self._pools.items()},
            "block_tables": {
                f"{k[0]}:{k[1]}": {str(r): list(b) for r, b in v.items()}
                for k, v in self._block_tables.items()
            },
            "request_tokens": dict(self._request_tokens),
            "valid_entries": {
                f"{k[0]}:{k[1]}": {str(r): int(c) for r, c in v.items()}
                for k, v in self._valid_entries.items()
            },
        }
        payload_path = root / PAYLOAD_FILE
        torch.save(payload, payload_path)
        checksum = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        manifest = {
            "format_version": FORMAT_VERSION,
            "producer_snapshot": {
                "module": __name__,
                "snapshot": meta.get(
                    "snapshot", os.environ.get("RTP_V41_SNAPSHOT_ID", "unrecorded")
                ),
                "exported_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "source_cp": int(meta.get("source_cp", 1)),
            "geometry_signature": self.geometry.signature(),
            "geometry": {
                "num_layers": self.geometry.num_layers,
                "compress_ratios": list(self.geometry.compress_ratios),
                "kv_source_layer_ids": list(self.geometry.kv_source_layer_ids),
                "index_source_layer_ids": list(self.geometry.index_source_layer_ids),
                "window": self.geometry.window,
                "block_entries": self.geometry.block_entries,
                "swa_block_entries": self.geometry.swa_block_entries,
            },
            "lengths": {str(k): int(v) for k, v in self._request_tokens.items()},
            "valid_regions": self.valid_regions(),
            "pool_layouts": {
                f"{key[0]}:{key[1]}": {
                    "dtype": str(tensor.dtype),
                    "stride": list(tensor.stride()),
                    "block_elements": int(tensor.shape[1]),
                }
                for key, tensor in self._pools.items()
            },
            "checksum": checksum,
            "meta": {k: v for k, v in meta.items() if k != "snapshot"},
        }
        (root / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )
        return manifest

    @classmethod
    def load_fixture(cls, path) -> Tuple["PPUKVCache", dict]:
        """Restore a cache from a fixture directory; refuse mismatches loudly.

        Verifies the format version, the payload checksum and the pool
        geometry signature before rebuilding. Returns ``(cache, manifest)``.
        """
        root = Path(path)
        manifest_path = root / MANIFEST_FILE
        payload_path = root / PAYLOAD_FILE
        if not manifest_path.is_file() or not payload_path.is_file():
            raise ValueError(f"fixture {root} is missing its manifest or payload")
        manifest = json.loads(manifest_path.read_text())
        if int(manifest["format_version"]) != FORMAT_VERSION:
            raise ValueError(
                f"fixture format {manifest['format_version']} != {FORMAT_VERSION}; "
                "refusing to load an incompatible V4.1 cache fixture"
            )
        checksum = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        if checksum != manifest["checksum"]:
            raise ValueError("fixture payload checksum mismatch; refusing to load")
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        geometry = CacheGeometry(
            num_layers=int(manifest["geometry"]["num_layers"]),
            compress_ratios=tuple(manifest["geometry"]["compress_ratios"]),
            kv_source_layer_ids=tuple(manifest["geometry"]["kv_source_layer_ids"]),
            index_source_layer_ids=tuple(
                manifest["geometry"]["index_source_layer_ids"]
            ),
            window=int(manifest["geometry"]["window"]),
            block_entries=int(manifest["geometry"]["block_entries"]),
            swa_block_entries=int(manifest["geometry"]["swa_block_entries"]),
        )
        if geometry.signature() != manifest["geometry_signature"]:
            raise ValueError(
                "fixture pool geometry does not match this code's topology; "
                "refusing to load"
            )
        cache = cls(geometry=geometry)
        expected_pools = {f"{key[0]}:{key[1]}" for key in cache._pools}
        if set(payload["pools"]) != expected_pools:
            raise ValueError("fixture pool tags differ from the cache topology")
        for name, tensor in payload["pools"].items():
            owner, tag = name.split(":", 1)
            key = (int(owner), tag)
            if key not in cache._pools:
                raise ValueError(f"fixture carries unknown pool {name}")
            expected_pool = cache._pools[key]
            if (
                tensor.ndim != 2
                or tensor.shape[1] != expected_pool.shape[1]
                or tensor.dtype != expected_pool.dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"fixture pool {name} has incompatible dtype/stride/block width"
                )
            layout = {
                "dtype": str(tensor.dtype),
                "stride": list(tensor.stride()),
                "block_elements": int(tensor.shape[1]),
            }
            if (
                "pool_layouts" in manifest
                and manifest["pool_layouts"].get(name) != layout
            ):
                raise ValueError(
                    f"fixture pool {name} differs from its layout contract"
                )
            if tuple(tensor.shape) != tuple(cache._pools[key].shape):
                cache._grow_pool(key, list(range(tensor.shape[0])))
                cache._pools[key] = tensor.clone().to(cache._pools[key].dtype)
            else:
                cache._pools[key] = tensor.clone().to(cache._pools[key].dtype)
            # The restored rows are all IN USE (the block tables reference
            # them): the allocator's free list must stay EMPTY, or the next
            # cross-block alloc hands out an occupied block and the new
            # token's entry overwrites live data (a post-prefill fixture
            # continued past its block boundary corrupts entry 0).
            allocator = _BlockAllocator(tensor.shape[0])
            allocator.free.clear()
            cache._allocators[key] = allocator
        for name, per_request in payload["block_tables"].items():
            owner, tag = name.split(":", 1)
            cache._block_tables[(int(owner), tag)] = {
                int(r): [int(b) for b in blocks] for r, blocks in per_request.items()
            }
        for name, per_request in payload["valid_entries"].items():
            owner, tag = name.split(":", 1)
            cache._valid_entries[(int(owner), tag)] = {
                int(r): int(c) for r, c in per_request.items()
            }
        cache._request_tokens = {
            int(r): int(v) for r, v in payload["request_tokens"].items()
        }
        return cache, manifest

    def bytes_summary(self) -> Dict[str, int]:
        """Host/device bytes held per pool tag (diagnostics)."""
        summary: Dict[str, int] = {}
        for key, pool in self._pools.items():
            tag = key[1]
            summary[tag] = summary.get(tag, 0) + pool.numel() * pool.element_size()
        return summary


class _ViewDescriptor:
    """A ``PoolDescriptor`` plus the producer identity L1's reader looks for."""

    def __init__(self, descriptor: PoolDescriptor, producer_layer_id: int):
        self.__dict__.update(
            entry_bytes=descriptor.entry_bytes,
            ratio=descriptor.ratio,
            planar=descriptor.planar,
            block_entries=descriptor.block_entries,
            window=descriptor.window,
            kind=descriptor.kind,
        )
        self.producer_layer_id = int(producer_layer_id)
        self.source_layer_id = int(producer_layer_id)
        self.layer_id = int(producer_layer_id)

    @property
    def compression_ratio(self) -> int:
        return self.ratio


class _CacheView:
    """Duck-typed ``V41CacheView`` (pools / block_tables / descriptors)."""

    def __init__(self, pools, block_tables, descriptors):
        self.pools = pools
        self.block_tables = block_tables
        self.descriptors = descriptors


__all__ = [
    "FORMAT_VERSION",
    "CacheGeometry",
    "PPUKVCache",
    "PPURequestSlots",
    "GLOBAL_KV_ENTRY_BYTES",
    "INDEXER_KV_ENTRY_BYTES",
    "SWA_KV_ENTRY_BYTES",
    "CSA_STATE_FLOATS",
]
