"""Native cache binding and fixed-address metadata for V4.1 PPU decode Graph."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from rtp_llm.models_py.modules.dsv41.kv_cache_utils import (
    as_attention_inputs_by_tag,
    primary_attention_inputs,
)
from rtp_llm.models_py.modules.dsv41.ppu.cache import CacheGeometry
from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import (
    position_slots,
    read_packed_rows,
    write_packed_rows,
)
from rtp_llm.models_py.modules.dsv41.ppu.native_protocol import framework_pool_tag
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy

_FULL = frozenset(("global_kv_1", "global_kv_2", "indexer_kv"))
_WIDTH = {
    "global_kv_1": 288,
    "global_kv_2": 288,
    "indexer_kv": 68,
    "swa_kv": 528,
    "csa_state": 1024,
}


def _identity(tensor):
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


class NativeGraphCache:
    """Bind framework pools directly; only their block-table contents change.

    Canonical PPU tags select framework groups independently. Full pools use
    kernel-granularity tables, while SWA/CSA use physical-granularity tables.
    The indexer scale plane starts after its RAW-token payload capacity, even
    for a ratio-2 producer with only half that many compressed keys.
    """

    def __init__(self, config, kv_cache, attention_inputs, *, device):
        if kv_cache is None:
            raise ValueError("PPU decode Graph requires framework-owned cache pools")
        self.config = dict(config)
        self.geometry = CacheGeometry.from_config(config)
        self.policy = PrefillPolicy.from_config(config)
        self.max_seq_len = int(config.get("max_position_embeddings", 32768))
        self.device = torch.device(device)
        self.native = kv_cache
        self.entries, self.spans, self.tables = {}, {}, {}
        self._pools, self._handles, self._pool_identity = {}, {}, {}
        self._table_fields, self._table_identity = {}, {}
        self._framework_tags, self._table_framework_tags = {}, {}
        by_tag = as_attention_inputs_by_tag(attention_inputs, kv_cache)
        for owner, tag in self.geometry.pool_keys():
            framework_tag = framework_pool_tag(
                owner, tag, geometry=self.geometry, policy=self.policy
            )
            self._framework_tags[(owner, tag)] = framework_tag
            table_tag = self._table_tag(owner, tag)
            if framework_tag not in by_tag:
                raise ValueError(
                    f"missing Graph attention inputs for pool {framework_tag}"
                )
            handle = kv_cache.get_layer_cache(owner, framework_tag)
            tensor = handle.kv_cache_base
            if (
                tensor.ndim < 2
                or not tensor.is_contiguous()
                or tensor.device != self.device
            ):
                raise ValueError(
                    f"Graph pool {(owner, tag)} must be contiguous on {self.device}"
                )
            tensor = tensor.view(tensor.shape[0], math.prod(tensor.shape[1:]))
            entries = int(kv_cache.get_entries_per_block(framework_tag))
            span = int(
                kv_cache.get_kernel_seq_size_per_block(framework_tag)
                if tag in _FULL
                else kv_cache.get_seq_size_per_block(framework_tag)
            )
            dtype = torch.float32 if tag == "csa_state" else torch.uint8
            if entries <= 0 or span <= 0 or tensor.dtype != dtype:
                raise ValueError(
                    f"invalid Graph pool dtype/geometry for {(owner, tag)}"
                )
            if tensor.shape[1] < entries * _WIDTH[tag] or tensor.shape[0] < 1:
                raise ValueError(
                    f"Graph pool {(owner, tag)} has insufficient physical capacity"
                )
            if tag == "swa_kv" and entries < self.geometry.window:
                raise ValueError(
                    "SWA native ring cannot hold the complete attention window"
                )
            if tag == "csa_state" and entries < 2:
                raise ValueError("CSA native ring cannot hold a ratio-2 boundary")
            self._pools[(owner, tag)] = tensor
            self._handles[(owner, tag)] = handle
            self._pool_identity[(owner, tag)] = _identity(tensor)
            self.entries[table_tag], self.spans[table_tag] = entries, span
            if table_tag in self.tables:
                continue
            field = (
                "kv_cache_kernel_block_id_device"
                if tag in _FULL
                else "kv_cache_block_id_device"
            )
            table = getattr(by_tag[framework_tag], field, None)
            if table is None or not table.numel():
                table = getattr(
                    by_tag[framework_tag], field.removesuffix("_device"), None
                )
            if table is None or table.ndim != 2 or table.shape[1] == 0:
                raise ValueError(
                    f"Graph pool {framework_tag} requires its own nonempty block table"
                )
            self.tables[table_tag] = table.to(device=self.device)
            self._table_fields[table_tag] = field
            self._table_identity[table_tag] = _identity(self.tables[table_tag])
            self._table_framework_tags[table_tag] = framework_tag

    def _table_tag(self, owner, tag):
        framework_tag = self._framework_tags[(int(owner), tag)]
        return "decoder_swa_kv" if framework_tag == "decoder_swa_kv" else tag

    def pool(self, owner, tag):
        return self._pools[(int(owner), tag)]

    def slots(self, owner, tag, raw_positions):
        """Translate raw positions [B, ...], returning -1 for padding."""
        if raw_positions.device != self.device:
            raise ValueError("Graph positions must already reside on the cache device")
        full = tag in _FULL
        table_tag = self._table_tag(owner, tag)
        ratio = self.geometry.ratio(int(owner)) if full else 1
        if ratio <= 0:
            raise ValueError("Global/indexer Graph cache owner must have ratio 1 or 2")
        return position_slots(
            raw_positions,
            self.tables[table_tag],
            span=self.spans[table_tag],
            entries=self.entries[table_tag],
            ratio=ratio,
            full=full,
            pool_blocks=self.pool(owner, tag).shape[0],
        )

    def read_rows(self, owner, tag, slots):
        return read_packed_rows(
            self.pool(owner, tag),
            slots,
            width=_WIDTH[tag],
            entries=self.entries[self._table_tag(owner, tag)],
            planar=tag == "indexer_kv",
        )

    def write_rows(self, owner, tag, slots, rows):
        write_packed_rows(
            self.pool(owner, tag),
            slots,
            rows,
            width=_WIDTH[tag],
            entries=self.entries[self._table_tag(owner, tag)],
            planar=tag == "indexer_kv",
        )

    def prepare(self, attention_inputs):
        """Refresh owned host-fallback tables outside capture, retaining storage."""
        by_tag = as_attention_inputs_by_tag(attention_inputs, self.native)
        for tag, destination in self.tables.items():
            framework_tag = self._table_framework_tags[tag]
            if framework_tag not in by_tag:
                raise ValueError(
                    f"missing Graph attention inputs for pool {framework_tag}"
                )
            field = self._table_fields[tag]
            source = getattr(by_tag[framework_tag], field, None)
            if source is None or not source.numel():
                source = getattr(
                    by_tag[framework_tag], field.removesuffix("_device"), None
                )
            if source is None or tuple(source.shape) != tuple(destination.shape):
                raise ValueError(
                    f"Graph block-table geometry changed for {framework_tag}"
                )
            if _identity(destination) != self._table_identity[tag]:
                raise RuntimeError("Graph cache table allocation changed after capture")
            if (
                source.device != destination.device
                or source.data_ptr() != destination.data_ptr()
            ):
                destination.copy_(source, non_blocking=True)
        for key, pool in self._pools.items():
            current = self.native.get_layer_cache(
                key[0], self._framework_tags[key]
            ).kv_cache_base
            current = current.view(current.shape[0], math.prod(current.shape[1:]))
            if (
                _identity(current) != self._pool_identity[key]
                or _identity(pool) != self._pool_identity[key]
            ):
                raise RuntimeError(
                    "Framework cache allocation changed after Graph binding"
                )


@dataclass
class DecodeGraphMetadata:
    positions: torch.Tensor
    valid_mask: torch.Tensor
    request_ids: torch.Tensor
    batch_size: int
    q_len_per_req: int = 1


class PPUDecodeGraphImpl:
    """C++ Graph preparation protocol for ordinary one-token PPU decode."""

    cuda_graph_uses_device_metadata = True
    cuda_graph_uses_valid_request_count = True
    cuda_graph_uses_physical_block_tables = True

    def __init__(self, config, kv_cache, attention_inputs, *, device, batch_size=None):
        self.cache = NativeGraphCache(config, kv_cache, attention_inputs, device=device)
        self.device = self.cache.device
        primary = primary_attention_inputs(attention_inputs, kv_cache)
        lengths = getattr(primary, "input_lengths", None)
        if batch_size is None:
            batch_size = (
                int(lengths.shape[0])
                if lengths is not None and lengths.numel()
                else next(iter(self.cache.tables.values())).shape[0]
            )
        self.batch_size = int(batch_size)
        if self.batch_size <= 0 or any(
            t.shape[0] < self.batch_size for t in self.cache.tables.values()
        ):
            raise ValueError("PPU decode Graph batch must fit all tagged tables")
        self.cuda_graph_valid_request_count_device = torch.full(
            (), self.batch_size, dtype=torch.int32, device=self.device
        )
        self.metadata = DecodeGraphMetadata(
            positions=torch.zeros(
                self.batch_size, dtype=torch.int64, device=self.device
            ),
            valid_mask=torch.ones(
                self.batch_size, dtype=torch.bool, device=self.device
            ),
            request_ids=torch.arange(
                self.batch_size, dtype=torch.int64, device=self.device
            ),
            batch_size=self.batch_size,
        )
        self._metadata_in_graph = False
        self._position_source = None
        self._position_is_plus_one = False
        self._bind_positions(attention_inputs, allow_host=True)
        self._update_metadata()

    @property
    def cuda_graph_preparation_in_graph(self):
        return self._metadata_in_graph

    def support_cuda_graph(self):
        return True

    def _bind_positions(self, attention_inputs, *, allow_host):
        primary = primary_attention_inputs(attention_inputs, self.cache.native)
        source = getattr(primary, "sequence_lengths_plus_1_device", None)
        plus_one = source is not None and source.numel() > 0
        if not plus_one:
            source = getattr(primary, "sequence_lengths", None)
        if source is None or source.numel() < self.batch_size:
            raise ValueError(
                "PPU Graph decode requires current per-request sequence lengths"
            )
        if source.device != self.device:
            if not allow_host:
                raise ValueError("PPU Graph metadata requires device sequence lengths")
            source = source.to(device=self.device)
        self._position_source = source.reshape(-1)[: self.batch_size]
        self._position_is_plus_one = plus_one

    def _update_metadata(self):
        positions = self._position_source.to(dtype=torch.int64)
        if self._position_is_plus_one:
            positions = positions - 1
        positions = positions.clamp(min=0, max=self.cache.max_seq_len - 1)
        self.metadata.positions.copy_(positions)
        self.metadata.valid_mask.copy_(
            self.metadata.request_ids < self.cuda_graph_valid_request_count_device
        )

    def prepare_cuda_graph(self, attention_inputs, *, valid_request_count=None):
        if valid_request_count is None:
            raise ValueError("PPU Graph replay requires an explicit live request count")
        self.cache.prepare(attention_inputs)
        self._bind_positions(attention_inputs, allow_host=not self._metadata_in_graph)
        destination = self.cuda_graph_valid_request_count_device
        if isinstance(valid_request_count, torch.Tensor):
            if (
                valid_request_count.numel() != 1
                or valid_request_count.device != self.device
                or valid_request_count.dtype != torch.int32
            ):
                raise ValueError(
                    "PPU Graph live request count must be one device int32"
                )
            if valid_request_count.data_ptr() != destination.data_ptr():
                destination.copy_(valid_request_count.reshape(()))
        else:
            if not 0 <= valid_request_count <= self.batch_size:
                raise ValueError("PPU Graph live request count exceeds captured batch")
            destination.fill_(valid_request_count)
        if not self._metadata_in_graph:
            self._update_metadata()

    def prepare_in_graph(self, attention_inputs):
        # The C++ runner updates the same device length allocation each replay.
        # Rebind only at eager/capture entry; Python is not needed during replay.
        self._bind_positions(attention_inputs, allow_host=False)
        self._update_metadata()
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            self._metadata_in_graph = True
