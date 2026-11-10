"""PPU views of public RR pages and native fixed-pool byte slices.

No storage is allocated here for a resident cache. Read assemblies are owned
by one read call (or one Attention layer's source scope), never by the model.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.cp import cp_gather_request_pool_blocks
from rtp_llm.models_py.modules.dsv41.cp_cache import (
    local_slice_offset,
    ring_coordinates,
    round_robin_owner,
    round_robin_page,
)
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import (
    NativeCacheBridge, _FULL, _Pool, _WIDTH,
)


def bind_cache_placement(config, parallelism, is_decode):
    """Freeze rank-local allocation facts; protocol identity omits local rank."""
    cp = parallelism.prefill_cp_config
    enabled = bool(cp.kv_cache_sharded)
    size = int(cp.prefill_cp_size if is_decode else parallelism.tp_size)
    if enabled and size <= 1:
        raise ValueError("sharded PPU cache requires an explicit source CP size")
    return dict(config, _ppu_cp_cache={
        "enabled": enabled, "size": size if enabled else 1,
        "rank": int(parallelism.tp_rank) if enabled and not is_decode else 0,
        "sliced_prefill": enabled and not is_decode,
    })


class _LogicalRead(NativeCacheBridge):
    """Reuse the native coordinate/planar copier on a temporary logical view."""

    def __init__(self, geometry, key, pool):
        self.logical = SimpleNamespace(geometry=geometry)
        self.pools = {key: pool}


@dataclass(frozen=True)
class PoolPlacement:
    size: int
    rank: int
    physical_span: int
    pages: int
    ratio: int
    full_entries: int
    slice_elements: int
    table: object

    @classmethod
    def create(cls, cp, key, pool, native, meta, geometry):
        size, rank = int(cp["size"]), int(cp["rank"])
        if size <= 1 or not 0 <= rank < size:
            raise ValueError("invalid native CP placement")
        physical_span = int(native.get_seq_size_per_block(pool.framework_tag))
        if physical_span <= 0 or physical_span % pool.span:
            raise ValueError("physical span must contain complete kernel pages")
        table = getattr(meta, "kv_cache_block_id_device", None)
        if table is None or not table.numel():
            table = getattr(meta, "kv_cache_block_id", None)
        if table is None or table.ndim != 2:
            raise ValueError("native CP placement requires the physical tag table")
        pages = physical_span // pool.span if key[1] in _FULL else 1
        if pool.tensor.shape[0] % pages:
            raise ValueError("native physical block has incomplete kernel pages")
        width = _WIDTH[key[1]]
        if key[1] in _FULL:
            entries, sliced = pool.entries, pool.tensor.shape[1]
            if sliced < entries * width:
                raise ValueError("native CP page is smaller than its payload")
        elif key[1] == "csa_state":
            # PAYLOAD layout: the API reports this rank's entry count.
            entries, sliced = pool.entries * size, pool.entries * width
            if pool.tensor.shape[1] != sliced:
                raise ValueError("CSA payload slice must contain complete rows")
        else:
            # BLOCK_STRIDE layout: entries describes the full ring, while
            # allocation contains one contiguous (possibly padded) slice.
            entries, sliced = pool.entries, pool.tensor.shape[1]
            if entries * width > sliced * size:
                raise ValueError("SWA stride slices cannot cover the complete ring")
        ratio = geometry.ratio(key[0]) if key[1] in _FULL else 1
        return cls(size, rank, physical_span, pages, ratio, entries, sliced, table.to(pool.tensor.device))

    def owner(self, key, positions):
        if key[1] in _FULL:
            return round_robin_owner(positions, self.physical_span, self.size)
        return ((positions % self.full_entries) * _WIDTH[key[1]]) // self.slice_elements

    def owned(self, key, positions):
        if key[1] in _FULL:
            return self.owner(key, positions) == self.rank
        start = (positions % self.full_entries) * _WIDTH[key[1]]
        return (start < (self.rank + 1) * self.slice_elements) & (start + _WIDTH[key[1]] > self.rank * self.slice_elements)

    def write_ring(self, key, pool, request, positions, rows):
        """Copy only the rank's byte/element intersection, even across a row."""
        columns, offsets = ring_coordinates(positions, pool.span, self.full_entries)
        if positions.numel() and (int(positions.min()) < 0 or int(columns.max()) >= pool.table.shape[1]):
            raise ValueError("native CP ring position outside its physical table")
        blocks = pool.table[int(request), columns].long()
        if blocks.numel() and (int(blocks.min()) < 0 or int(blocks.max()) >= pool.tensor.shape[0]):
            raise ValueError("native CP ring refers to an unallocated physical block")
        width = _WIDTH[key[1]]
        for start in range(0, positions.numel(), 4096):
            row_offsets = offsets[start:start + 4096, None]
            logical = row_offsets * width + torch.arange(width, device=positions.device)
            local = local_slice_offset(logical, self.rank, self.slice_elements)
            live = (local >= 0) & (local < self.slice_elements)
            b = blocks[start:start + 4096, None].expand_as(local)
            pool.tensor[b[live], local[live]] = rows[start:start + 4096].to(pool.tensor.device)[live]

    def coordinates(self, key, pool, request, positions):
        if key[1] in _FULL:
            columns, page_offsets = round_robin_page(
                positions, self.physical_span, pool.span, self.size
            )
            offsets = page_offsets // self.ratio
        else:
            columns, ring_offsets = ring_coordinates(positions, pool.span, self.full_entries)
            offsets = local_slice_offset(
                ring_offsets, self.rank, self.slice_elements // _WIDTH[key[1]]
            )
        if positions.numel() and (int(positions.min()) < 0 or int(columns.max()) >= pool.table.shape[1]):
            raise ValueError("native CP position outside local tag table")
        blocks = pool.table[int(request), columns].long()
        if blocks.numel() and (int(blocks.min()) < 0 or int(blocks.max()) >= pool.tensor.shape[0]):
            raise ValueError("native CP position refers to an unallocated page")
        if offsets.numel() and (int(offsets.min()) < 0 or int(offsets.max()) * _WIDTH[key[1]] >= self.slice_elements):
            raise ValueError("native CP row is outside its owned slice")
        return blocks, offsets

    def gather(self, key, pool, request, positions, geometry):
        """All ranks participate, including ranks with no logical owned rows."""
        if key[1] in _FULL:
            total = int(positions.max()) // self.physical_span + 1
            local_count = (total + self.size - 1) // self.size
            local = pool.tensor.view(-1, self.pages, pool.tensor.shape[1])
            table = self.table[int(request), :local_count]
            if table.numel() != local_count:
                raise ValueError("native CP physical table is too short")
            assembled = cp_gather_request_pool_blocks(local, table, self.size, self.rank, total)
            tensor = assembled.reshape(-1, pool.tensor.shape[1])
            entries = pool.entries
            logical_table = torch.arange(tensor.shape[0], device=tensor.device)[None]
        else:
            columns = torch.unique(positions // pool.span, sorted=True)
            if int(columns[0]) < 0 or int(columns[-1]) >= self.table.shape[1]:
                raise ValueError("native CP ring position outside physical table")
            table = self.table[int(request), columns]
            if bool(((table < 0) | (table >= pool.tensor.shape[0])).any()):
                raise ValueError("native CP ring read refers to an unallocated physical block")
            # Public interleave treats CP slices as sub-blocks, then joins
            # just the requested physical columns back into full blocks.
            count = columns.numel()
            assembled = cp_gather_request_pool_blocks(pool.tensor, table, self.size, self.rank, count * self.size)
            tensor = assembled.reshape(count, -1)
            entries = self.full_entries
            # Keep absolute positions so ring offsets survive sparse columns.
            logical_table = torch.full((1, self.table.shape[1]), -1, device=tensor.device, dtype=torch.long)
            logical_table[0, columns] = torch.arange(count, device=tensor.device)
        view = _Pool(pool.framework_tag, pool.handle, tensor, logical_table, entries, pool.span)
        return _LogicalRead(geometry, key, view)._copy_rows(key, 0, positions)
