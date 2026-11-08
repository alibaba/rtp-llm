"""Fixed device metadata and native-pool adapters for DSpark model capture."""

import hashlib
import json
import os
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv41.kv_cache_utils import primary_attention_inputs
from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import NativeGraphCache, _FULL, _WIDTH
from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import position_slots
from rtp_llm.models_py.modules.dsv41.ppu.shared_graph_metadata import (
    device_rows,
    require_device_tables,
)
from rtp_llm.models_py.speculative.dspark_proposer_mixin import map_context_rows


class SharedDSparkGraphImpl:
    """One graph's fixed row/batch geometry, over the current tagged tables.

    Packed rows stay request-major and contiguous. The captured rectangle in
    the public target loop is just a view of the token slab; device lengths
    determine each logical row's request and position.
    """

    cuda_graph_uses_device_metadata = True
    cuda_graph_uses_valid_request_count = True
    cuda_graph_uses_physical_block_tables = True

    def __init__(self, model, inputs, mode, *, runtime=None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare DSpark Graph resources before capture")
        if not model._is_decode_role or model.kv_cache is None:
            raise ValueError("DSpark Graph requires D's native cache")
        self.mode, self.runtime = mode, runtime
        self._model = weakref.ref(model)
        self.device = model.shared_runtime.device
        attn = primary_attention_inputs(inputs.attention_inputs, model.kv_cache)
        if not attn.is_target_verify or getattr(attn, "context_parallel_info", None) is not None:
            raise ValueError("DSpark Graph is for decode-side verify/propose/commit only")
        self.batch_size = batch = attn.input_lengths.numel()
        self.rows = rows = inputs.input_ids.numel()
        if not 0 < batch <= model._max_generate_batch_size or rows <= 0 or rows % batch:
            raise ValueError("DSpark Graph needs a fixed public batch/token bucket")
        self.width = rows // batch
        if mode == "proposal":
            if self.width != model.shared_mode.query_width:
                raise ValueError("proposal width differs from the public query block")
        elif not 0 < self.width <= model.shared_runtime.policy.max_rollback_tokens:
            raise ValueError("DSpark Graph bucket exceeds rollback capacity")
        if mode == "dense" and (
            attn.is_packed_target_verify or self.width != model.shared_runtime.policy.max_rollback_tokens
        ):
            raise ValueError("dense verify requires the full public query width")
        if mode == "packed" and (
            not attn.is_packed_target_verify or self.width >= model.shared_runtime.policy.max_rollback_tokens
        ):
            raise ValueError("packed verify requires a smaller public packed bucket")
        self.requests = torch.arange(batch, device=self.device)
        self.cuda_graph_valid_request_count_device = torch.full(
            (), batch, dtype=torch.int32, device=self.device
        )
        self._in_graph = False
        self._replays = 0
        self._reported_idle = False
        require_device_tables(inputs.attention_inputs, model.kv_cache, self.device)
        self.native_cache = self.cache = NativeGraphCache(
            dict(model.shared_weight_views.config, max_position_embeddings=int(model._v4_args.max_seq_len)),
            model.kv_cache, inputs.attention_inputs, device=self.device,
        )
        self.prefixes, self.lengths = self._sources(attn)
        if mode in ("dense", "packed"):
            self.cache = VerifyGraphJournal(self.native_cache, rows)
            self.consumed_prefixes = torch.empty_like(self.prefixes)
            self.consumed_lengths = torch.empty_like(self.lengths)
            self.consumed_ids = torch.empty_like(inputs.input_ids)
            self.consumed_windows = torch.empty_like(inputs.engram_token_windows)
            self.armed_verify = None
            self.staged_completion = 0

    def _sources(self, attn):
        return tuple(
            device_rows(attn, name, self.device, self.batch_size)
            for name in ("prefix_lengths_device", "input_lengths_device")
        )

    @property
    def cuda_graph_preparation_in_graph(self):
        return self._in_graph

    def support_cuda_graph(self):
        return True

    def prepare_cuda_graph(self, attention_inputs, *, valid_request_count):
        native = self.native_cache.native
        require_device_tables(attention_inputs, native, self.device)
        self.native_cache.prepare(attention_inputs)
        attn = primary_attention_inputs(attention_inputs, native)
        for destination, source in zip((self.prefixes, self.lengths), self._sources(attn)):
            if source.data_ptr() != destination.data_ptr():
                # The draft's public proposer also reads these stable mirrors.
                destination.copy_(source, non_blocking=True)
        count = self.cuda_graph_valid_request_count_device
        if isinstance(valid_request_count, torch.Tensor):
            if (count.shape, count.dtype, count.device) != (
                valid_request_count.shape, valid_request_count.dtype, valid_request_count.device
            ):
                raise ValueError("DSpark Graph requires an int32 device count scalar")
            if count.data_ptr() != valid_request_count.data_ptr():
                count.copy_(valid_request_count)
        elif type(valid_request_count) is int and 0 <= valid_request_count <= self.batch_size:
            count.fill_(valid_request_count)
        else:
            raise ValueError("invalid DSpark Graph live request count")

    def live_requests(self):
        if torch.cuda.is_current_stream_capturing():
            self._in_graph = True
        return self.requests < self.cuda_graph_valid_request_count_device

    def prepare_in_graph(self, attention_inputs):
        lengths = torch.where(self.live_requests(), self.lengths.long(), 0)
        prefixes = self.prefixes.long()
        self.consumed_prefixes.copy_(prefixes)
        self.consumed_lengths.copy_(lengths)
        starts = lengths.cumsum(0) - lengths
        req, raw = map_context_rows(starts, lengths, prefixes + lengths, self.rows)
        valid = (req >= 0) & (raw >= 0) & (raw < self.native_cache.max_seq_len)
        self.metadata = SimpleNamespace(
            batch_size=1 if self.mode == "packed" else self.batch_size,
            q_len_per_req=self.rows if self.mode == "packed" else self.width,
            positions=raw.long().clamp(0, self.native_cache.max_seq_len - 1),
            valid_mask=valid,
        )
        self.cache.bind(req.long(), starts, prefixes, valid)

    def assert_inputs_bound(self, inputs):
        attn = primary_attention_inputs(inputs.attention_inputs, self.native_cache.native)
        if inputs.input_ids.numel() != self.rows or any(
            source.data_ptr() != destination.data_ptr()
            for source, destination in zip(self._sources(attn), (self.prefixes, self.lengths))
        ):
            raise ValueError("DSpark Graph forward requires its captured input buffers")

    def arm_verify(self, request_ids):
        """Bind the direct driver's request identities to exactly one forward."""
        if self.mode not in ("dense", "packed") or torch.cuda.is_current_stream_capturing():
            raise ValueError("bind verify request identities before replay")
        self.armed_verify = (int(self.cache.completions.item()), request_ids.clone())

    def finish_cuda_graph_replay(self, inputs, outputs):
        model = self._model()
        verify = self.mode in ("dense", "packed")
        if verify and (not inputs.warmup or inputs.request_id is not None):
            model.shared_runtime.stage_graph_verify(
                inputs, self, outputs.mtp_target_hidden_states, native_replay=True
            )
        directory = os.environ.get("RTP_V41_SHARED_EVIDENCE_DIR")
        if not directory or inputs.warmup or (inputs.is_fake_stream and self._reported_idle):
            return
        self._reported_idle |= inputs.is_fake_stream
        self._replays += 1
        attn = primary_attention_inputs(inputs.attention_inputs, model.kv_cache)
        rank = dist.get_rank() if dist.is_initialized() else 0
        record = {
            "rank": rank,
            "mode": self.mode,
            "bucket": [self.batch_size, self.width],
            "replay": self._replays,
            "idle": inputs.is_fake_stream,
            "prefixes": attn.prefix_lengths.tolist(),
            "lengths": attn.input_lengths.tolist(),
        }
        if os.environ.get("RTP_V41_TEST_DSPARK_GRAPH_BATCH") == "1" and not inputs.is_fake_stream:
            record["observed_ns"] = time.monotonic_ns()
            if verify:
                record["target_call"] = model.shared_runtime.calls
                record["request_ids"] = inputs.request_id.reshape(-1).tolist()
                rows = inputs.input_ids.numel()
                record["tokens"] = self.consumed_ids[:rows].reshape(-1).tolist()
                record["positions"] = self.metadata.positions[:rows].reshape(-1).tolist()
        if self.mode == "commit":
            record["native_writes_per_layer"] = self.writes.tolist()
        if verify or self.mode == "commit":
            aux = outputs.mtp_target_hidden_states if verify else inputs.input_hiddens
            record["aux_sha256"] = hashlib.sha256(aux.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        path = Path(directory) / f"graph-rank{rank}-{self.mode}-{self.batch_size}-{self.width}-{self._replays:05d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as output:
            json.dump(record, output)


class VerifyGraphJournal:
    """Fixed raw-position payloads; only the existing accept hook publishes."""

    def __init__(self, native, rows):
        self.native_cache = native
        self.native, self.max_seq_len = native.native, native.max_seq_len
        self.generations = {
            key: torch.empty(rows, dtype=torch.long, device=native.device)
            for key in native.geometry.pool_keys()
        }
        self.payloads = {
            key: native.pool(*key).new_empty((rows, _WIDTH[key[1]]))
            for key in self.generations
        }
        self.completions = torch.zeros((), dtype=torch.int64, device=native.device)
        self.consumed_tables = {tag: torch.empty_like(table) for tag, table in native.tables.items()}

    def bind(self, requests, starts, prefixes, valid):
        self.requests, self.valid = requests.clamp_min(0), valid
        self.starts, self.prefixes = starts, prefixes
        for tag, table in self.native_cache.tables.items():
            self.consumed_tables[tag].copy_(table)

    def _native_slots(self, owner, tag, raw):
        cache = self.native_cache
        table_tag = cache._table_tag(owner, tag)
        full = tag in _FULL
        return position_slots(
            raw, cache.tables[table_tag][self.requests],
            span=cache.spans[table_tag], entries=cache.entries[table_tag],
            ratio=cache.geometry.ratio(owner) if full else 1, full=full,
            pool_blocks=cache.pool(owner, tag).shape[0],
        )

    def slots(self, owner, tag, raw):
        # A journal slot is a raw position in this query row's request. Check
        # actual allocation first, without conflating aliased ring generations.
        return torch.where(self._native_slots(owner, tag, raw) >= 0, raw, -1)

    def write_rows(self, owner, tag, slots, rows):
        self.generations[owner, tag].copy_(slots)
        self.payloads[owner, tag].copy_(rows)

    def read_rows(self, owner, tag, slots):
        key = owner, tag
        native = self.native_cache.read_rows(owner, tag, self._native_slots(owner, tag, slots))
        shape = (self.requests.numel(),) + (1,) * (slots.ndim - 1)
        starts = self.starts[self.requests].reshape(shape)
        prefixes = self.prefixes[self.requests].reshape(shape)
        # A ratio-2 payload is written at the odd generation and is indexed by
        # its preceding even absolute position.
        lag = int(tag in _FULL and self.native_cache.geometry.ratio(owner) == 2)
        offset = slots + lag - prefixes
        index = (starts + offset).clamp(0, self.requests.numel() - 1)
        live = (
            (slots >= 0) & (offset >= 0) & self.valid[index]
            & (self.requests[index] == self.requests.reshape(shape))
            & (self.generations[key][index] == slots)
        )
        return torch.where(live[..., None], self.payloads[key][index], native)

    def stage(self, cache, layout):
        """Materialize this replay's live journal for the N7 transaction."""
        from rtp_llm.models_py.modules.dsv41.ppu.shared_verify import SharedCacheVersion

        for key, positions in self.generations.items():
            table = self.consumed_tables[self.native_cache._table_tag(*key)]
            pool = cache.pools[key]
            batch, columns = len(layout.lengths), pool.table.shape[1]
            if (pool.tensor.data_ptr() != self.native_cache.pool(*key).data_ptr()
                or not torch.equal(table[:batch, :columns], pool.table[:batch])):
                raise RuntimeError(f"verify journal native binding changed for {key}")
            for request, (start, length) in enumerate(zip(layout.starts, layout.lengths)):
                section = slice(start, start + length)
                raw = positions[section]
                live = raw >= 0
                if bool(live.any()):
                    # Singleton CSA versions preserve every raw generation
                    # through the existing exact accepted-row publication.
                    pos, payload = raw[live], self.payloads[key][section][live]
                    if key[1] == "csa_state":
                        cache.journal.extend(
                            SharedCacheVersion(key, request, p[None], row[None])
                            for p, row in zip(pos, payload)
                        )
                    else:
                        cache.journal.append(SharedCacheVersion(key, request, pos, payload))
