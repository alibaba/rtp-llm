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
from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import (
    BatchedVerifyPublication,
    position_slots,
    try_enqueue_verify_table_check,
    try_plan_verify_publication,
)
from rtp_llm.models_py.modules.dsv41.ppu.graph_metadata import (
    device_rows,
    require_device_tables,
)
from rtp_llm.models_py.modules.dsv41.ppu.graph_packed_attention import (
    GRAPH_VISIBLE_INDEXER_ENV,
    VERIFY_PACKED_ATTENTION_ENV,
    GraphPackedAttention,
)
from rtp_llm.models_py.modules.dsv41.ppu.packed_attention import DEPENDENCY_ENV, enabled
from rtp_llm.models_py.speculative.dspark_proposer_mixin import map_context_rows


_REFERENCE_REPLAYS = []
_VERIFY_FUSED_PUBLICATION_ENV = "RTP_V41_PPU_VERIFY_FUSED_PUBLICATION"
_VERIFY_BATCHED_PUBLICATION_ENV = "RTP_V41_PPU_VERIFY_BATCHED_PUBLICATION"
_PRIVATE_DRAFT_KV_ENV = "RTP_V41_PPU_DSPARK_PRIVATE_DRAFT_KV"
_JOINT_KV_PUBLISH_ENV = "RTP_V41_PPU_DSPARK_JOINT_KV_PUBLISH"


def flush_reference_replays(directory, rank):
    if not _REFERENCE_REPLAYS:
        return
    path = Path(directory) / f"replay-rank{rank}.jsonl"
    with path.open("x") as output:
        for replay in _REFERENCE_REPLAYS:
            output.write(json.dumps(replay) + "\n")
    _REFERENCE_REPLAYS.clear()


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
        self.visible_indexer = mode in ("dense", "packed") and enabled(GRAPH_VISIBLE_INDEXER_ENV)
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
        self.private_draft_journal = None
        self._private_draft_replay = 0
        if mode in ("dense", "packed"):
            self.cache = VerifyGraphJournal(self.native_cache, rows)
            if (
                runtime is not None and runtime.grouped is None
                and enabled(VERIFY_PACKED_ATTENTION_ENV, default=True)
                and os.environ.get(DEPENDENCY_ENV)
                and GraphPackedAttention.supports_verify(self.native_cache, rows)
            ):
                self.packed_attention = GraphPackedAttention(
                    self.native_cache, model.shared_weight_views.geometry, journal=self.cache,
                )
            self.consumed_prefixes = torch.empty_like(self.prefixes)
            self.consumed_lengths = torch.empty_like(self.lengths)
            self.consumed_ids = torch.empty_like(inputs.input_ids)
            self.consumed_windows = torch.empty_like(inputs.engram_token_windows)
            self.armed_verify = None
            self.staged_completion = 0
        elif (
            mode == "commit" and (batch, self.width) == (8, 4)
            and not attn.is_packed_target_verify and enabled(_PRIVATE_DRAFT_KV_ENV)
            and self.native_cache.geometry.window >= self.width
            and tuple(self.native_cache.geometry.pool_keys()) == tuple((layer, "swa_kv") for layer in range(3))
        ):
            self.private_draft_journal = _PrivateDraftCommitJournal(
                self.native_cache, rows, self.prefixes, self.lengths,
            )
            self.cache = self.private_draft_journal

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

    def stage_private_draft_commit(self, inputs):
        """Queue one receipt on the Draft replay stream."""
        from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache

        journal = self.private_draft_journal
        native = self.native_cache
        attn = primary_attention_inputs(inputs.attention_inputs, native.native)
        if (
            journal is None or inputs.is_fake_stream or inputs.warmup
            or attn.is_packed_target_verify or attn.input_lengths.numel() != self.batch_size
            or inputs.input_ids.numel() != self.rows
        ):
            raise ValueError("private Draft KV requires a live dense B8/Q4 replay")
        require_device_tables(inputs.attention_inputs, native.native, self.device)
        cache = SharedNativeCache(
            native.config, native.geometry, native.native, inputs.attention_inputs, (),
            device=self.device, defer_producer_slots=True,
        )
        return _PrivateDraftCommitReceipt(self, journal, cache)

    def publish_dense_private_draft_commit(self, inputs):
        """Preserve dense writes for a Commit replay without an armed lease."""
        if not inputs.is_fake_stream:
            self.private_draft_journal.publish_dense()

    def finish_cuda_graph_replay(self, inputs, outputs):
        model = self._model()
        verify = self.mode in ("dense", "packed")
        if verify and (not inputs.warmup or inputs.request_id is not None):
            runtime = model.shared_runtime
            aux = outputs.mtp_target_hidden_states
            if not runtime.try_stage_deferred_graph_verify(inputs, self, aux):
                runtime.stage_graph_verify(inputs, self, aux, native_replay=True)
        if self.private_draft_journal is not None:
            self._private_draft_replay += 1
            if inputs.warmup:
                self.publish_dense_private_draft_commit(inputs)
            model.note_private_draft_replay(self, inputs)
        reference_dir = os.environ.get("RTP_V41_REFERENCE_PERF_DIR")
        if reference_dir and not inputs.warmup and not inputs.is_fake_stream:
            if len(_REFERENCE_REPLAYS) < 16384:
                _REFERENCE_REPLAYS.append((time.monotonic_ns(), self.mode, self.batch_size, self.visible_indexer))
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
            counter = (
                "staged_writes_per_layer" if self.private_draft_journal is not None
                else "native_writes_per_layer"
            )
            record[counter] = self.writes.tolist()
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
        self.rows = rows
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
        self._batched_publication = None
        if rows == 32 and enabled(_VERIFY_BATCHED_PUBLICATION_ENV, default=True):
            self._batched_publication = self._build_publication_batch()

    def _build_publication_batch(self):
        cache = self.native_cache
        pools = []
        for (owner, tag), raw in self.generations.items():
            table_tag = cache._table_tag(owner, tag)
            full = tag in _FULL
            pools.append((
                raw, self.payloads[owner, tag], cache.pool(owner, tag),
                self.consumed_tables[table_tag], cache.spans[table_tag],
                cache.entries[table_tag], cache.geometry.ratio(owner) if full else 1,
                full, tag == "csa_state", tag == "indexer_kv",
            ))
        return BatchedVerifyPublication.try_create(pools, self.rows, cache.device)

    def bind(self, requests, starts, prefixes, valid):
        self.requests, self.valid = requests.clamp_min(0), valid
        self.starts, self.prefixes = starts, prefixes
        self.row_tables = {}
        for tag, table in self.native_cache.tables.items():
            self.consumed_tables[tag].copy_(table)
            self.row_tables[tag] = table[self.requests]

    def _native_slots(self, owner, tag, raw):
        cache = self.native_cache
        table_tag = cache._table_tag(owner, tag)
        full = tag in _FULL
        return position_slots(
            raw, self.row_tables[table_tag],
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

    def binding_checks(self, cache):
        checked = set()
        for key in self.generations:
            table_tag = self.native_cache._table_tag(*key)
            consumed = self.consumed_tables[table_tag]
            pool = cache.pools[key]
            if (pool.tensor.data_ptr() != self.native_cache.pool(*key).data_ptr()
                or pool.span != self.native_cache.spans[table_tag]
                or pool.entries != self.native_cache.entries[table_tag]):
                raise RuntimeError(f"verify journal native binding changed for {key}")
            table = pool.table
            identity = (table_tag, table.data_ptr(), table.shape, table.stride())
            if identity not in checked:
                checked.add(identity)
                yield key, consumed, table

    def validate_bindings(self, cache, layout):
        """Check the per-replay table snapshot once per shared table."""
        batch = len(layout.lengths)
        for key, consumed, table in self.binding_checks(cache):
            if not torch.equal(consumed[:batch, :table.shape[1]], table[:batch]):
                raise RuntimeError(f"verify journal native block table changed for {key}")

    def try_enqueue_binding_check(self, cache, batch, *, checks=None):
        """Check native pool bindings now and defer the exact table receipt."""
        if checks is None:
            checks = tuple(self.binding_checks(cache))
        return try_enqueue_verify_table_check(checks, batch)

    def joint_publication_inputs(self, cache, last_accepted, *, checks=None):
        if self._batched_publication is None or not enabled(_VERIFY_FUSED_PUBLICATION_ENV, default=True):
            return None
        if checks is None:
            checks = tuple(self.binding_checks(cache))
        # Captured tables can be wider than the scheduler's live tables.
        # Bound every possible published raw position to the checked columns.
        if any(last_accepted >= live.shape[1] * cache.pools[key].span for key, _, live in checks):
            return None
        return (
            self._batched_publication, self.requests, self.prefixes, self.valid,
            checks,
        )

    def _prepare_publication(self, ends):
        cache = self.native_cache
        if (
            ends.is_cuda and 0 < self.rows <= 64 and self.generations
            and enabled(_VERIFY_FUSED_PUBLICATION_ENV, default=True)
        ):
            plans = torch.empty(
                (len(self.generations), self.rows + 2), dtype=torch.int64, device=ends.device,
            )
            prepared = {}
            for index, ((owner, tag), raw) in enumerate(self.generations.items()):
                table_tag = cache._table_tag(owner, tag)
                full = tag in _FULL
                plan = plans[index]
                if not try_plan_verify_publication(
                    raw, self.requests, self.prefixes, ends, self.valid,
                    self.row_tables[table_tag], plan,
                    span=cache.spans[table_tag], entries=cache.entries[table_tag],
                    ratio=cache.geometry.ratio(owner) if full else 1,
                    full=full, trim_csa=tag == "csa_state",
                    pool_blocks=cache.pool(owner, tag).shape[0],
                ):
                    break
                prepared[owner, tag] = plan[:self.rows]
            else:
                # One host transfer validates the entire transaction before
                # backing up or writing any destination.
                count, invalid = plans[:, self.rows:].sum(dim=0).tolist()
                if invalid:
                    raise RuntimeError("accepted verify journal row has no native slot")
                return prepared, count
        row_end = ends[self.requests]
        row_base = self.prefixes[self.requests]
        prepared, accepted, invalid = {}, [], []
        for (owner, tag), raw in self.generations.items():
            if tag in _FULL:
                ratio = cache.geometry.ratio(owner)
                compressed = raw // ratio
                keep = (compressed >= row_base // ratio) & (compressed < row_end // ratio)
            else:
                keep = (raw >= row_base) & (raw < row_end)
                if tag == "csa_state":
                    entries = cache.entries[cache._table_tag(owner, tag)]
                    keep &= raw >= row_end - entries
            keep &= self.valid & (raw >= 0)
            slots = self._native_slots(owner, tag, torch.where(keep, raw, -1))
            accepted.append(keep.sum())
            invalid.append((keep & (slots < 0)).any())
            # A ring or shared page can alias. Match the legacy request-major
            # publication order and scatter each physical destination once.
            ordered, order = slots.sort(stable=True)
            ordered[:-1].masked_fill_(ordered[:-1] == ordered[1:], -1)
            prepared[owner, tag] = torch.empty_like(slots).scatter_(0, order, ordered)
        if invalid and bool(torch.stack(invalid).any()):
            raise RuntimeError("accepted verify journal row has no native slot")

        count = int(torch.stack(accepted).sum()) if accepted else 0
        return prepared, count

    def publish(self, layout, counts):
        """Publish the accepted prefix from the fixed device rows, per native pool."""
        ends = [base + count for base, count in zip(layout.prefixes, counts)]
        ends.extend([0] * (self.prefixes.numel() - len(ends)))
        ends = torch.tensor(ends, dtype=self.prefixes.dtype, device=self.prefixes.device)
        if self._batched_publication is not None and enabled(_VERIFY_FUSED_PUBLICATION_ENV, default=True):
            count = self._batched_publication.publish(self.requests, self.prefixes, ends, self.valid)
            if count is not None:
                return count
        prepared, count = self._prepare_publication(ends)
        cache = self.native_cache
        backups = {key: cache.read_rows(*key, slots) for key, slots in prepared.items()}
        written = []
        try:
            for key, slots in prepared.items():
                written.append(key)
                cache.write_rows(*key, slots, self.payloads[key])
        except Exception:
            for key in reversed(written):
                cache.write_rows(*key, prepared[key], backups[key])
            raise
        return count

    def stage(self, cache, layout):
        """Materialize this replay's live journal for the N7 transaction."""
        from rtp_llm.models_py.modules.dsv41.ppu.verify_transaction import SharedCacheVersion

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


class _PrivateDraftCommitJournal(VerifyGraphJournal):
    """Three SWA scratch pools with no native destination in the capture."""

    is_private_draft_commit = True

    def __init__(self, native, rows, prefixes, lengths):
        super().__init__(native, rows)
        self.source_prefixes, self.source_lengths = prefixes, lengths
        self.layout = torch.empty((2, prefixes.numel()), dtype=torch.int64, device=prefixes.device)

    def bind_commit(self, requests, valid, live):
        self.layout[0].copy_(self.source_prefixes)
        self.layout[1].copy_(torch.where(live, self.source_lengths, 0))
        self.bind(requests, None, self.layout[0], valid)

    def publish_dense(self):
        from rtp_llm.models_py.modules.dsv41.ppu.graph_swa_write import unique_commit_swa_slots

        # Legacy dense Commit skips absent slots and keeps the last feature
        # row for each physical address, including rejected tail rows.
        for key, raw in self.generations.items():
            slots = unique_commit_swa_slots(self._native_slots(*key, raw))
            self.native_cache.write_rows(*key, slots, self.payloads[key])


class _PrivateDraftCommitReceipt:
    def __init__(self, impl, journal, cache):
        self.impl, self.journal, self.cache = impl, journal, cache
        self.replay = impl._private_draft_replay
        self.closed = False
        self.layout = self.ready = None
        if not enabled(_JOINT_KV_PUBLISH_ENV):
            self.layout = torch.empty(
                journal.layout.shape, dtype=journal.layout.dtype, device="cpu", pin_memory=True,
            )
            self.layout.copy_(journal.layout, non_blocking=True)
            self.ready = torch.cuda.Event()
            self.ready.record(torch.cuda.current_stream(impl.device))

    def _check_replay(self):
        if self.closed or self.replay != self.impl._private_draft_replay:
            raise RuntimeError("private Draft KV receipt is stale or already published")

    def _check_native_bindings(self):
        native = self.cache.native
        for (owner, _), pool in self.cache.pools.items():
            current = native.get_layer_cache(owner, pool.framework_tag).kv_cache_base
            if (
                current.ndim < 2 or not current.is_contiguous()
                or current.device != pool.tensor.device or current.dtype != pool.tensor.dtype
                or current.data_ptr() != pool.tensor.data_ptr()
                or current.shape[0] != pool.tensor.shape[0]
                or current.numel() != pool.tensor.numel()
                or native.get_entries_per_block(pool.framework_tag) != pool.entries
                or native.get_seq_size_per_block(pool.framework_tag) != pool.span
            ):
                raise RuntimeError("private Draft KV native allocation changed after replay")
            table = self.cache.inputs[pool.framework_tag].kv_cache_block_id_device
            if (
                table is None or table.device != pool.table.device or table.dtype != pool.table.dtype
                or table.data_ptr() != pool.table.data_ptr() or table.shape != pool.table.shape
                or table.stride() != pool.table.stride()
            ):
                raise RuntimeError("private Draft KV live page table rebound after replay")

    def _check_capacity(self, bases, last_accepted):
        if max(bases) + self.impl.width > self.journal.max_seq_len:
            raise ValueError("private Draft KV replay exceeds native cache capacity")
        if any(last_accepted // pool.span >= pool.table.shape[1] for pool in self.cache.pools.values()):
            raise RuntimeError("accepted private Draft KV row has no live page")

    def check_joint_publication(self, bases, last_accepted):
        self._check_replay()
        self._check_native_bindings()
        self._check_capacity(bases, last_accepted)

    def joint_publication_inputs(self, bases, last_accepted):
        self.check_joint_publication(bases, last_accepted)
        return self.joint_inputs_if_covered(last_accepted)

    def joint_inputs_if_covered(self, last_accepted):
        inputs = self.journal.joint_publication_inputs(self.cache, last_accepted)
        if inputs is None:
            return None
        return (*inputs, self.journal.layout[1])

    def publish(self, prefixes, accepted_rows):
        from rtp_llm.models_py.modules.dsv41.ppu.verify import VerifyLayout

        self._check_replay()
        self.closed = True
        batch, width = self.impl.batch_size, self.impl.width
        if any(
            not isinstance(value, torch.Tensor) or value.numel() != batch
            or value.dtype not in (torch.int32, torch.int64)
            or value.device not in (torch.device("cpu"), self.impl.device)
            for value in (prefixes, accepted_rows)
        ):
            raise ValueError("private Draft KV acceptance must contain one prefix and count per request")
        counts = tuple(accepted_rows.reshape(-1).tolist())
        if any(not 0 <= count <= width for count in counts):
            raise ValueError("private Draft KV accepted rows exceed the dense replay")
        self._check_native_bindings()
        # The executor has already joined the Draft replay stream.
        bindings = self.journal.try_enqueue_binding_check(self.cache, batch)
        bases = tuple(prefixes.reshape(-1).tolist())
        if self.ready is not None:
            self.ready.synchronize()
        snapshot = self.journal.layout if self.layout is None else self.layout
        consumed_prefixes, consumed_lengths = map(tuple, snapshot.tolist())
        if bases != consumed_prefixes or consumed_lengths != (width,) * batch:
            raise ValueError("private Draft KV acceptance does not match the replay layout")
        layout = VerifyLayout.from_lengths_and_prefixes(consumed_lengths, consumed_prefixes, batch * width)
        last_accepted = max(
            (base + count - 1 for base, count in zip(bases, counts) if count), default=-1,
        )
        self._check_capacity(bases, last_accepted)
        if bindings is None:
            self.journal.validate_bindings(self.cache, layout)
        else:
            bindings.validate()
        if not any(counts):
            return 0
        return self.journal.publish(layout, counts)
