"""Fixed-capacity Engram hashing and ENP lookup for PPU decode Graphs.

The scheduler owns token history and supplies a most-recent-first window.
Every Graph bucket uses the same model-wide ENP capacity, including idle DP
ranks. Only query IDs and table contributions change between replays; neither
the communication shape nor the host table backing storage changes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv41.ppu.engram import (
    DEAD_ID,
    FP8_BLOCK_SIZE,
    EngramHashParams,
    _software_decode_e4m3,
    _torch_window_hash,
    _triton_kernels,
    dequant_rows,
)


@dataclass
class _LookupBuffers:
    # Strong references keep pinned table memory alive for captured pointers.
    shard: Any
    payload: torch.Tensor
    scale: torch.Tensor
    layout: Any
    process_group: Any
    rank: int
    size: int
    ids: torch.Tensor
    gathered: torch.Tensor
    contribution: torch.Tensor
    payload_pointer: int
    scale_pointer: int
    row_start: int
    row_stop: int


class GraphEngram:
    """Model-owned, non-reentrant ENP workspace shared by decode buckets.

    ``lookup`` returns a view into persistent storage. Its caller must consume
    the rows before another lookup of that layer. Graph replay preserves this
    ordering on the model stream, just as the eager device path does.
    """

    def __init__(self, ops, geometry, *, device, max_batch_size, fuse_metadata=False):
        self.device = torch.device(device)
        self.max_batch_size = int(max_batch_size)
        if self.max_batch_size < 1:
            raise ValueError("Graph Engram requires a positive maximum batch size")
        self.layer_ids = tuple(geometry.engram_layer_ids)
        self.window_size = int(geometry.engram_max_ngram_size)
        self.hash_columns = (self.window_size - 1) * int(geometry.engram_n_heads)
        self.image_token_id = int(geometry.image_token_id)
        self._bindings = {}
        self._kernels = None
        self.params = None
        if not self.layer_ids:
            return
        params = EngramHashParams.build(
            layer_ids=self.layer_ids,
            max_ngram_size=self.window_size,
            n_heads=geometry.engram_n_heads,
            vocab_size=geometry.engram_vocab_size,
            compressed_vocab_size=geometry.engram_compressed_vocab_size,
            num_embeddings=geometry.engram_num_embeddings,
            token_map=list(geometry.engram_token_map),
            pad_token_id=geometry.engram_pad_token_id,
        )
        # Constants are materialized before warmup/capture, never per step.
        self.params = replace(
            params,
            token_map=params.token_map.to(self.device),
            multipliers=params.multipliers.to(self.device),
            primes=params.primes.to(self.device),
            offsets=params.offsets.to(self.device),
        )
        self._device_path = self.device.type != "cpu"
        self._fuse_metadata = fuse_metadata and self._device_path
        self._decode_e4m3_bytes = _software_decode_e4m3()
        if self._device_path:
            self._kernels = _triton_kernels()
            if self._kernels is None:
                raise RuntimeError(
                    "Graph Engram requires the device hash/gather kernels"
                )
            if self.window_size & (self.window_size - 1) or (
                self.params.n_heads & (self.params.n_heads - 1)
            ):
                raise ValueError(
                    "Graph Engram hash window and head counts must be powers of two"
                )
        capacity = self.max_batch_size * self.hash_columns
        for index, layer_id in enumerate(self.layer_ids):
            try:
                shard, layout, group = ops._engram[layer_id]
            except KeyError as error:
                raise RuntimeError(
                    f"Graph Engram layer {layer_id} has no bound shard"
                ) from error
            if shard.layout != layout or shard.dim != geometry.engram_head_dim:
                raise ValueError(
                    f"Graph Engram layer {layer_id} shard geometry differs"
                )
            if layout.num_embeddings != geometry.engram_num_embeddings[index]:
                raise ValueError("Graph Engram hash and table row counts differ")
            if shard.payload.shape[0] != layout.valid_rows(shard.rank):
                raise ValueError(
                    "Graph Engram shard must contain exactly its owned rows"
                )
            if (
                shard.payload.dtype != torch.float8_e4m3fn
                or shard.scale.dtype != torch.uint8
            ):
                raise TypeError(
                    "Graph Engram requires FP8 payload and E8M0 scale bytes"
                )
            if not shard.payload.is_contiguous() or not shard.scale.is_contiguous():
                raise ValueError("Graph Engram host tables must be contiguous")
            if self._device_path and not (
                shard.pinned and shard.payload.is_pinned() and shard.scale.is_pinned()
            ):
                raise RuntimeError(
                    "Graph Engram requires pinned, device-readable host shards"
                )
            if self._device_path and shard.dim & (shard.dim - 1):
                raise ValueError("Graph Engram embedding width must be a power of two")
            process_group = getattr(group, "pg", None)
            if layout.enp_size > 1:
                if group is not None and not hasattr(group, "pg"):
                    raise ValueError(
                        "Graph Engram requires a distributed ENP process group"
                    )
                if not dist.is_available() or not dist.is_initialized():
                    raise RuntimeError(
                        "Graph Engram ENP requires initialized collectives"
                    )
                size = dist.get_world_size(process_group)
                rank = dist.get_rank(process_group)
                if (size, rank) != (layout.enp_size, shard.rank):
                    raise ValueError(
                        "Graph Engram process group differs from shard ownership"
                    )
            else:
                size, rank = 1, 0
                if shard.rank != 0:
                    raise ValueError(
                        "Single-rank Graph Engram requires owner rank zero"
                    )
            self._bindings[layer_id] = _LookupBuffers(
                shard=shard,
                payload=shard.payload,
                scale=shard.scale,
                layout=layout,
                process_group=process_group,
                rank=rank,
                size=size,
                ids=torch.full(
                    (capacity,), DEAD_ID, dtype=torch.int64, device=self.device
                ),
                gathered=torch.full(
                    (size * capacity,), DEAD_ID, dtype=torch.int64, device=self.device
                ),
                contribution=torch.empty(
                    (size * capacity, shard.dim),
                    dtype=torch.bfloat16,
                    device=self.device,
                ),
                payload_pointer=shard.payload.data_ptr(),
                scale_pointer=shard.scale.data_ptr(),
                row_start=layout.row_start(rank),
                row_stop=layout.row_start(rank) + layout.valid_rows(rank),
            )

    def validate(self):
        """Validate table lifetimes from the runner's graph-external prepare."""
        for binding in self._bindings.values():
            if (
                binding.shard.payload is not binding.payload
                or binding.shard.scale is not binding.scale
                or binding.payload.data_ptr() != binding.payload_pointer
                or binding.scale.data_ptr() != binding.scale_pointer
            ):
                raise RuntimeError(
                    "Graph Engram table backing storage changed; recapture is required"
                )

    def hash(self, windows):
        """Hash scheduler windows on device without reconstructing history."""
        if windows.ndim != 2 or windows.shape[1] != self.window_size:
            raise ValueError(
                "Graph Engram windows must have the configured n-gram width"
            )
        if windows.shape[0] > self.max_batch_size or windows.device != self.device:
            raise ValueError(
                "Graph Engram windows exceed capacity or use another device"
            )
        if windows.dtype not in (torch.int32, torch.int64):
            raise TypeError("Graph Engram windows must contain integer token IDs")
        if self.params is None:
            return torch.empty(
                (windows.shape[0], 0, self.hash_columns),
                dtype=torch.int64,
                device=self.device,
            )
        if self._fuse_metadata:
            return self._kernels.engram_hash_windows(
                windows, None, self.params.token_map, self.params.multipliers,
                self.params.primes, self.params.offsets, self.params.pad_id,
                image_token_id=self.image_token_id,
            )
        # Invalid IDs must be made safe even for a masked table load. An image
        # token terminates the n-gram history, matching the eager hash contract.
        invalid = (windows < 0) | (windows >= self.params.token_map.numel())
        safe_windows = windows.masked_fill(invalid, -1).to(torch.int64).contiguous()
        dead = (windows == self.image_token_id) | invalid
        if self._device_path:
            hashes = self._kernels.engram_hash_windows(
                safe_windows,
                dead.to(torch.int8).contiguous(),
                self.params.token_map,
                self.params.multipliers,
                self.params.primes,
                self.params.offsets,
                self.params.pad_id,
            )
        else:
            hashes = _torch_window_hash(safe_windows, dead, self.params)
        live = ~dead[:, 0]
        return hashes.masked_fill(~live[:, None, None], DEAD_ID)

    def lookup(self, rows, *, layer_id, valid_mask):
        """Return owned-table contributions in this rank's original row order."""
        if layer_id not in self._bindings:
            raise ValueError(f"Graph Engram layer {layer_id} has no bound lookup")
        binding = self._bindings[layer_id]
        if rows.ndim != 2 or rows.shape[1] != self.hash_columns:
            raise ValueError("Graph Engram row IDs must be [batch, hash_columns]")
        batch = rows.shape[0]
        if batch > self.max_batch_size or rows.device != self.device:
            raise ValueError(
                "Graph Engram lookup exceeds capacity or uses another device"
            )
        if rows.dtype not in (torch.int32, torch.int64):
            raise TypeError("Graph Engram row IDs must be integer tensors")
        if valid_mask.dtype != torch.bool or valid_mask.shape != (batch,):
            raise ValueError("Graph Engram validity must be one bool per captured row")
        if valid_mask.device != self.device:
            raise ValueError("Graph Engram validity must remain on the model device")
        if self._fuse_metadata:
            self._kernels.engram_graph_ids(rows, valid_mask, binding.ids)
        else:
            binding.ids.fill_(DEAD_ID)
            binding.ids[: rows.numel()].copy_(
                rows.masked_fill(~valid_mask[:, None], DEAD_ID).reshape(-1)
            )
        if binding.size > 1:
            dist.all_gather_into_tensor(
                binding.gathered, binding.ids, group=binding.process_group
            )
        else:
            binding.gathered.copy_(binding.ids)
        if self._device_path:
            self._kernels.engram_gather(
                binding.payload_pointer,
                binding.scale_pointer,
                binding.gathered,
                binding.contribution,
                binding.shard.dim,
                FP8_BLOCK_SIZE,
                row_lo=binding.row_start,
                row_hi=binding.row_stop,
                decode_e4m3_bytes=self._decode_e4m3_bytes,
            )
        else:
            self._lookup_cpu(binding)
        if binding.size > 1:
            dist.all_reduce(
                binding.contribution, op=dist.ReduceOp.SUM, group=binding.process_group
            )
        start = binding.rank * binding.ids.numel()
        return binding.contribution[start : start + rows.numel()].view(
            batch, self.hash_columns, binding.shard.dim
        )

    @staticmethod
    def _lookup_cpu(binding):
        """CPU test oracle; accelerator execution never enters this branch."""
        if binding.row_start == binding.row_stop:
            binding.contribution.zero_()
            return
        ids = binding.gathered
        owned = (ids >= binding.row_start) & (ids < binding.row_stop)
        local = (ids - binding.row_start).clamp(
            0, binding.row_stop - binding.row_start - 1
        )
        payload = binding.payload.view(torch.uint8).index_select(0, local)
        values = dequant_rows(
            payload.view(torch.float8_e4m3fn), binding.scale.index_select(0, local)
        )
        binding.contribution.copy_(values.masked_fill(~owned[:, None], 0))
