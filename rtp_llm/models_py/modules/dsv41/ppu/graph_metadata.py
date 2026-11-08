"""Device metadata shared by PPU Decode and DSpark Graph adapters."""

import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv41.kv_cache_utils import (
    as_attention_inputs_by_tag,
    primary_attention_inputs,
)


def device_rows(attention_inputs, name, device, rows):
    value = getattr(attention_inputs, name, None)
    if (value is None or value.device != device or value.ndim != 1 or value.numel() < rows
        or value.dtype not in (torch.int32, torch.int64)):
        raise ValueError(f"PPU Graph requires device {name} for every capture row")
    return value[:rows]


def require_device_tables(attention_inputs, native, device):
    for tag, entry in as_attention_inputs_by_tag(attention_inputs, native).items():
        name = (
            "kv_cache_kernel_block_id_device"
            if tag in ("hca_kv", "csa_kv", "indexer_kv")
            else "kv_cache_block_id_device"
        )
        table = getattr(entry, name, None)
        if table is None or table.device != device or table.ndim != 2 or not table.numel():
            raise ValueError(f"PPU Graph requires a device page table for {tag}")


def refresh_device_metadata(impl, attention_inputs):
    """Keep captured pointers fixed even when a direct caller has new inputs.

    The runner normally passes slices of its own fixed buffers. Do not rebind
    the position source to a new allocation after capture.
    """
    native = impl.cache.native
    require_device_tables(attention_inputs, native, impl.device)
    primary = primary_attention_inputs(attention_inputs, native)
    source = device_rows(primary, "sequence_lengths_plus_1_device", impl.device, impl.batch_size)
    if not impl._position_is_plus_one:
        raise ValueError("grouped Graph requires the public device length-plus-one field")
    impl.cache.prepare(attention_inputs)
    if source.data_ptr() != impl._position_source.data_ptr():
        impl._position_source.copy_(source, non_blocking=True)


class DecodeGraphReceipt:
    """Optional bounded smoke telemetry; observe completed copies by event."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.rank = dist.get_rank()
        self.impl = None
        self.armed = False
        self.pending = []
        self.submitted = 0
        self.restore_checked = False

    def bind(self, impl, ids, windows):
        self.impl, self.ids, self.windows = impl, ids, windows
        self.reserved = {
            key: impl.cache.pool(*key)[0].view(torch.uint8).clone()
            for key in impl.cache.geometry.pool_keys()
        }

    def observe_restored_prefix(self, model, inputs):
        """N6 fingerprint, sampled before the first synchronous smoke replay."""
        from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache

        if self.restore_checked or inputs.is_fake_stream or inputs.warmup:
            return
        attn = primary_attention_inputs(inputs.attention_inputs, model.kv_cache)
        if attn.is_prefill:
            return
        prefix = tuple(map(int, attn.sequence_lengths.tolist()))
        if prefix != (257,):
            raise ValueError("restoration smoke expects the first recovered prefix at 257")
        runtime = model.shared_runtime
        cache = SharedNativeCache(
            model.shared_weight_views.config, runtime.geometry, model.kv_cache,
            inputs.attention_inputs, prefix, device=runtime.device,
        )
        with (self.directory / f"rank{self.rank}-restored-prefix.json").open("x") as output:
            json.dump({
                "schema": "n9-native-restored-prefix-v1", "rank": self.rank, "prefix_lengths": prefix,
                "native_cache": cache.fingerprint(prefix),
            }, output, indent=2)
            output.write("\n")
        self.restore_checked = True

    def observe(self):
        if not self.armed or self.submitted >= 2:
            return
        impl = self.impl
        metadata, runtime = impl.metadata, impl.runtime
        cache = impl.cache
        pos = torch.where(metadata.valid_mask, metadata.positions, -1)
        positions = pos[:, None]
        arrays = {
            "positions": pos,
            "valid": metadata.valid_mask,
            "token_ids": self.ids[:impl.batch_size],
            "engram_windows": self.windows[:impl.batch_size],
            "layer_calls": runtime.layer_calls,
            "received": runtime.grouped.received,
            "swa_slots": cache.slots(0, "swa_kv", positions),
            "reserved_unchanged": torch.stack([
                (cache.pool(*key)[0].view(torch.uint8) == before).all()
                for key, before in self.reserved.items()
            ]),
        }
        # Read the native pools bound at capture, downstream of replay.
        # These observations are scoped to the current live slot per domain.
        payloads = {}
        for layer, tag in cache.geometry.pool_keys():
            raw = positions
            if tag in ("global_kv_2", "indexer_kv") and cache.geometry.ratio(layer) == 2:
                raw = torch.where(positions >= 0, positions - (positions & 1), -1)
                raw = torch.where((positions & 1) != 0, raw, -1)
            slots = cache.slots(layer, tag, raw)
            payloads[f"{layer}:{tag}"] = cache.read_rows(layer, tag, slots).view(torch.uint8)
        tensors = {**arrays, **payloads}
        mirrors = {
            name: torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)
            for name, value in tensors.items()
        }
        for name, value in tensors.items():
            mirrors[name].copy_(value, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.submitted += 1
        self.pending.append((
            event, mirrors, tuple(payloads), self.submitted, {
                "ids": self.ids.data_ptr(),
                "windows": self.windows.data_ptr(),
                "positions": metadata.positions.data_ptr(),
                "count": impl.cuda_graph_valid_request_count_device.data_ptr(),
                **{f"table:{tag}": table.data_ptr() for tag, table in cache.tables.items()},
            },
        ))

    def flush(self):
        while self.pending and self.pending[0][0].query():
            _, mirrors, payloads, step, addresses = self.pending.pop(0)
            domains = {}
            for key in payloads:
                value = mirrors.pop(key)
                domains[key] = {
                    "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                    "nonzero_bytes": int(torch.count_nonzero(value)),
                }
            record = {
                "schema": "n9-native-grouped-decode-replay-v1",
                "rank": self.rank,
                "replay": step,
                "addresses": addresses,
                "domains": domains,
                **{name: tensor.tolist() for name, tensor in mirrors.items()},
            }
            path = self.directory / f"rank{self.rank}-replay{step}.json"
            temporary = path.with_suffix(".json.tmp")
            with temporary.open("x") as output:
                json.dump(record, output, indent=2)
                output.write("\n")
            temporary.replace(path)
