"""Eager generation-aware cache leases and complete stage attention plans.

The native allocator must retain the supplied pages until release. This object
does not allocate native blocks, restore prefixes or complete external PP/PD
tickets. One serialized forward is allowed; failures after execution poison the
session. Unsafe prefill ring overwrites use bounded invocation-owned scratch.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from types import MappingProxyType

import torch

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.models.dsv41_kv_cache import dsv41_cache_bindings
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache, _overlaps
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Span,
)
from rtp_llm.models_py.modules.dsv4.v41_compressor import V41PairMetadata, V41PairState
from rtp_llm.models_py.modules.dsv4.v41_stage import V41Stage
from rtp_llm.models_py.modules.dsv4.v41_stage_attention import (
    V41AttentionPlan,
    V41ForwardRows,
)


@dataclass
class _Request:
    identity: V41CacheIdentity
    tables: dict
    visible: int = 0
    chunk: int = 0
    cancelled: bool = False


def _cp_signature(forward, phase, spans):
    return hashlib.sha256(
        json.dumps(
            [forward.signature.hex(), phase, [s.wire_values() for s in spans]],
            separators=(",", ":"),
        ).encode("ascii")
    ).digest()


class V41StageCache:
    """Own logical page reservations, not the native allocator's physical leases.

    Generations increase across all admissions to this worker session. A request
    starts at token zero; successful forwards advance every local layer together.
    Tables are tag-local native block IDs shared by all layers using that tag.
    """

    def __init__(
        self,
        config,
        caches,
        *,
        role,
        pp_rank,
        max_positions=32768,
        max_scratch_bytes,
        backend="triton",
    ):
        if not isinstance(config, V41Config):
            raise ValueError("V4.1 cache planning requires checkpoint metadata")
        config.validate()
        if (
            role not in ("prefill", "decode")
            or type(pp_rank) is not int
            or pp_rank not in ((0, 1) if role == "prefill" else (0,))
        ):
            raise ValueError("V4.1 cache planning requires CEP4PP2 or DP8EP8 ownership")
        if type(max_positions) is not int or not 1 <= max_positions <= min(
            32768, config.text["max_position_embeddings"]
        ):
            raise ValueError("V4.1 cache planning requires a bounded 32K context")
        if type(max_scratch_bytes) is not int or max_scratch_bytes < 0:
            raise ValueError("V4.1 scratch budget must be nonnegative")
        bindings = dsv41_cache_bindings(
            config.text, 2 if role == "prefill" else 1, pp_rank
        )
        expected = {(b.source_layer_id, b.tag) for b in bindings}
        if set(caches) != expected:
            raise ValueError(
                "V4.1 native caches do not cover exact stage/source bindings"
            )
        self.caches = MappingProxyType(dict(caches))
        self.device = next(iter(caches.values())).raw.device
        if backend not in ("triton", "eager") or self.device.type != (
            "cpu" if backend == "eager" else "cuda"
        ):
            raise ValueError(
                "V4.1 cache planner requires explicit CPU eager or CUDA Triton"
            )
        if backend == "triton" and torch.cuda.get_device_capability(self.device) != (
            12,
            0,
        ):
            raise ValueError("V4.1 cache planning requires SM120")
        raw = []
        self._capacities = {}
        for (_, tag), cache in self.caches.items():
            cls = V41PairState if tag == "csa_state" else V41PagedCache
            if (
                not isinstance(cache, cls)
                or (tag != "csa_state" and cache.tag != tag)
                or cache.gen_num_per_cycle != 1
                or cache.raw.device != self.device
                or cache.raw.requires_grad
            ):
                raise ValueError(
                    "V4.1 cache planning requires matching inference pools and gen=1"
                )
            if any(_overlaps(cache.raw, prior) for prior in raw):
                raise ValueError("V4.1 native cache regions overlap")
            raw.append(cache.raw)
            self._capacities[tag] = min(
                self._capacities.get(tag, cache.raw.shape[0]), cache.raw.shape[0]
            )
        self.role, self.pp_rank, self.backend = role, pp_rank, backend
        self.begin, self.end = (
            ((0, 21) if pp_rank == 0 else (21, 40)) if role == "prefill" else (0, 40)
        )
        self.sources = config.layer_sources()[self.begin : self.end]
        self.max_positions, self.max_scratch_bytes = max_positions, max_scratch_bytes
        self._requests, self._owners = {}, {}
        self._last_generation, self._active = -1, None
        self._failed, self._drained, self._pid = False, True, os.getpid()
        self._undrained = None

    @classmethod
    def from_native(cls, config, kv_cache, **kwargs):
        role, rank = kwargs["role"], kwargs["pp_rank"]
        bindings = dsv41_cache_bindings(
            config.text, 2 if role == "prefill" else 1, rank
        )
        expected = {(b.storage_layer_id, b.tag): b for b in bindings}
        count = 40 if role == "decode" else 21 if rank == 0 else 19
        if kv_cache.layer_count != count:
            raise ValueError("V4.1 native cache has the wrong stage-local layer count")
        views = {}
        for layer in range(count):
            groups = kv_cache.get_layer_cache_groups(layer)
            if {g.tag for g in groups} != {
                tag for slot, tag in expected if slot == layer
            } or len({g.tag for g in groups}) != len(groups):
                raise ValueError("V4.1 native cache has missing or extra layer tags")
            for group in groups:
                binding = expected[layer, group.tag]
                if group.tag == "csa_state":
                    if (
                        group.layer_id != layer
                        or group.seq_size_per_block != 256
                        or group.read_replica_source_layer_id is not None
                        or (
                            group.kv_scale_base is not None
                            and group.kv_scale_base.numel()
                        )
                    ):
                        raise ValueError("V4.1 native pair-state metadata mismatch")
                    view = V41PairState(group.kv_cache_base)
                else:
                    view = V41PagedCache.from_layer_cache(group, binding)
                views[binding.source_layer_id, binding.tag] = view
        result = cls(config, views, **kwargs)
        result._native_owner = kv_cache
        return result

    def _check(self, *, idle=False, healthy=True):
        if os.getpid() != self._pid:
            raise RuntimeError("V4.1 cache owner cannot cross a fork")
        if healthy and self._failed:
            raise RuntimeError("V4.1 cache session failed; recreate after draining")
        if idle and self._active is not None:
            raise RuntimeError("V4.1 cache session has an in-flight forward")
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("V4.1 cache admission cannot run during capture")

    def _state(self, identity):
        if not isinstance(identity, V41CacheIdentity):
            raise ValueError("V4.1 cache lease requires an explicit identity")
        state = self._requests.get(identity.request_id)
        if state is None or state.identity != identity:
            raise ValueError("V4.1 unknown or stale cache identity")
        return state

    def _tables(self, blocks):
        if set(blocks) != set(self._capacities):
            raise ValueError("V4.1 lease must cover every stage cache tag")
        tables = {tag: tuple(pages) for tag, pages in blocks.items()}
        full_lengths = set()
        for tag, pages in tables.items():
            if any(
                type(p) is not int or not 1 <= p < self._capacities[tag] for p in pages
            ) or len(set(pages)) != len(pages):
                raise ValueError("V4.1 lease has duplicate or invalid physical pages")
            if tag in ("swa_kv", "csa_state"):
                if len(pages) != 1:
                    raise ValueError(
                        "V4.1 initial exact path requires one state page per request/tag"
                    )
            else:
                if len(pages) > (self.max_positions + 255) // 256:
                    raise ValueError("V4.1 full allocation exceeds context bound")
                full_lengths.add(len(pages))
        if len(full_lengths) != 1:
            raise ValueError("V4.1 FULL tags must cover identical logical page counts")
        return tables

    def _reserve(self, identity, tables):
        if any(
            self._owners.get((tag, page), identity) != identity
            for tag, pages in tables.items()
            for page in pages
        ):
            raise ValueError("V4.1 allocation overlaps another active request")
        self._owners.update(
            {(tag, page): identity for tag, pages in tables.items() for page in pages}
        )

    def begin_request(self, identity, blocks):
        self._check(idle=True)
        if not isinstance(identity, V41CacheIdentity) or identity.request_id < 0:
            raise ValueError("V4.1 request rows require nonnegative cache identities")
        if (
            identity.request_id in self._requests
            or identity.generation <= self._last_generation
        ):
            raise ValueError(
                "V4.1 admission requires a new request and increasing generation"
            )
        tables = self._tables(blocks)
        self._reserve(identity, tables)
        self._requests[identity.request_id] = _Request(identity, tables)
        self._last_generation = identity.generation

    def extend_request(self, identity, blocks):
        self._check(idle=True)
        state, tables = self._state(identity), self._tables(blocks)
        if state.cancelled or any(
            tables[tag][: len(pages)] != pages for tag, pages in state.tables.items()
        ):
            raise ValueError("V4.1 active allocations only allow append-only growth")
        self._reserve(identity, tables)
        state.tables = tables

    def block_tables(self, identity):
        self._check()
        state = self._state(identity)
        if state.cancelled:
            raise ValueError("V4.1 request is cancelled")
        return dict(state.tables)

    def cancel(self, identity):
        self._check(idle=True, healthy=False)
        self._state(identity).cancelled = True

    def drain(self):
        self._check(idle=True, healthy=False)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._drained = True
        self._undrained = None

    def release(self, identity):
        """Caller must also drain external transport/read tickets before release."""
        self._check(idle=True, healthy=False)
        state = self._state(identity)
        if not self._drained:
            raise RuntimeError("V4.1 cache writes have not drained")
        for tag, pages in state.tables.items():
            for page in pages:
                del self._owners[tag, page]
        del self._requests[identity.request_id]

    def prepare(self, rows, spans):
        """Plan full visible CP rows; peers must agree planning success before run."""
        self._check(idle=True)
        if not isinstance(rows, V41ForwardRows):
            raise ValueError("V4.1 cache planning requires explicit forward rows")
        rows, spans = rows.snapshot(), tuple(spans)
        if (
            rows.positions.device != self.device
            or any(not isinstance(s, V41L20Span) for s in spans)
            or len({s.identity.request_id for s in spans}) != len(spans)
        ):
            raise ValueError(
                "V4.1 cache planning requires distinct trusted request spans"
            )
        expected = set()
        for span in spans:
            state = self._state(span.identity)
            end = span.start + span.count
            if (
                state.cancelled
                or span.start != state.visible
                or span.chunk_id != state.chunk
            ):
                raise ValueError(
                    "V4.1 chunk must continue its admitted generation without gaps/replay"
                )
            if end > self.max_positions or (self.role == "decode" and span.count != 1):
                raise ValueError(
                    "V4.1 chunk exceeds context or single-token decode bounds"
                )
            if any(
                end > len(pages) * 256
                for tag, pages in state.tables.items()
                if tag not in ("swa_kv", "csa_state")
            ):
                raise ValueError("V4.1 chunk exceeds its FULL allocation")
            expected.update(
                (span.identity.request_id, pos) for pos in range(span.start, end)
            )
        records = list(
            zip(
                rows.request_ids.cpu().tolist(),
                rows.positions.cpu().tolist(),
                rows.valid_rows.cpu().tolist(),
            )
        )
        if {(req, pos) for req, pos, live in records if live} != expected:
            raise ValueError(
                "V4.1 forward rows do not cover exactly the admitted chunks"
            )
        batch = V41CacheBatch(self, rows, spans, records)
        self._active = batch
        self._drained = False
        try:
            batch._build()
        except Exception:
            # Only invocation-owned scratch may have been written during prepare.
            self._active = None
            try:
                self.drain()
            except Exception:
                self._failed, self._undrained = True, batch
            raise
        return batch

    def prepare_cp(self, forward, spans):
        """Collectively admit generations and local plans before any native write."""
        forward._check()
        batch, error, signature = None, None, bytes(32)
        try:
            if (self.role, self.pp_rank, self.max_positions, self.device) != (
                "prefill",
                forward.session.plan.pp_rank,
                forward.session.plan.max_positions,
                forward.session.device,
            ):
                raise ValueError("V4.1 cache owner and CP plan disagree")
            batch = self.prepare(forward.rows, spans)
            signature = _cp_signature(forward, "cache-prepare", batch.spans)
        except Exception as exc:
            error = exc
        try:
            forward.session._consensus(signature, error)
        except Exception:
            if batch is not None:
                batch.discard()
            raise
        batch._cp_forward = forward
        return batch


class V41CacheBatch:
    """One-use plans, scratch and continuation commit for a serialized stage call."""

    def __init__(self, owner, rows, spans, records):
        self.owner, self.rows, self.spans, self._records = owner, rows, spans, records
        self._stream = (
            torch.cuda.current_stream(owner.device).cuda_stream
            if owner.device.type == "cuda"
            else None
        )
        self._plans, self._copies = {}, []
        self._started = False
        self._cp_forward = None
        self.scratch_bytes = 0

    @property
    def plans(self):
        return MappingProxyType(self._plans)

    def _check(self):
        self.owner._check()
        if self.owner._active is not self or self._started:
            raise RuntimeError("V4.1 cache batch is stale, completed or executing")
        if (
            self._stream is not None
            and torch.cuda.current_stream(self.owner.device).cuda_stream != self._stream
        ):
            raise RuntimeError("V4.1 cache batch must stay on its preparation stream")

    def discard(self):
        """Discard a prepared batch before any stage execution."""
        self._check()
        self.owner._active = None
        try:
            self.owner.drain()
        except Exception:
            self.owner._failed, self.owner._undrained = True, self
            raise

    def _build(self):
        owner, n = self.owner, len(self._records)
        device = owner.device
        by_id = {s.identity.request_id: s for s in self.spans}
        tables = {
            req: owner._state(span.identity).tables for req, span in by_id.items()
        }
        entries = owner.caches[owner.begin, "swa_kv"].layout.entries
        scratch = any(min(s.start, 127) + s.count > entries for s in self.spans)
        offsets, pages = {}, 0
        for req, span in by_id.items():
            first = max(0, span.start - 127)
            offsets[req] = (pages * entries, first)
            pages += (span.start + span.count - first + entries - 1) // entries
        if scratch:
            self.scratch_bytes = (
                pages
                * owner.caches[owner.begin, "swa_kv"].layout.stride_bytes
                * len(owner.sources)
            )
            if self.scratch_bytes > owner.max_scratch_bytes:
                raise ValueError(
                    "V4.1 prefill scratch exceeds its admitted byte budget"
                )

        def integer(values, shape=None):
            value = torch.tensor(values, dtype=torch.int64, device=device)
            return value if shape is None else value.reshape(shape)

        def swa_slot(req, pos, *, native=False):
            if scratch and not native:
                base, first = offsets[req]
                return base + pos - first
            return tables[req]["swa_kv"][0] * entries + pos % entries

        writes = integer(
            [swa_slot(req, pos) if live else -1 for req, pos, live in self._records]
        )
        reads = integer(
            [
                (
                    [swa_slot(req, p) for p in range(max(0, pos - 127), pos + 1)]
                    + [-1] * (128 - min(pos + 1, 128))
                    if live
                    else [-1] * 128
                )
                for req, pos, live in self._records
            ],
            (n, 128),
        )
        prefix_native, prefix_scratch, retain_native, retain_scratch = [], [], [], []
        if scratch:
            for req, span in by_id.items():
                for pos in range(max(0, span.start - 127), span.start):
                    prefix_native.append(swa_slot(req, pos, native=True))
                    prefix_scratch.append(swa_slot(req, pos))
                end = span.start + span.count
                for pos in range(max(0, end - 128), end):
                    retain_native.append(swa_slot(req, pos, native=True))
                    retain_scratch.append(swa_slot(req, pos))
        prefix_native, prefix_scratch, retain_native, retain_scratch = map(
            integer, (prefix_native, prefix_scratch, retain_native, retain_scratch)
        )
        width = (owner.max_positions + 255) // 256
        page_maps = {
            tag: integer(
                [
                    (
                        list(tables[req][tag]) + [-1] * (width - len(tables[req][tag]))
                        if live
                        else [-1] * width
                    )
                    for req, _, live in self._records
                ],
                (n, width),
            ).int()
            for tag in owner._capacities
            if tag not in ("swa_kv", "csa_state")
        }
        lookup = {
            (req, pos): row
            for row, (req, pos, live) in enumerate(self._records)
            if live
        }
        for sources in owner.sources:
            layer, ratio = sources.layer_id, sources.compress_ratio
            native = owner.caches[layer, "swa_kv"]
            swa = native
            if scratch:
                swa = V41PagedCache(
                    torch.empty(
                        (pages, native.layout.stride_bytes),
                        dtype=torch.uint8,
                        device=device,
                    ),
                    "swa_kv",
                )
                swa.scatter(
                    native.gather(prefix_native, backend=owner.backend),
                    prefix_scratch,
                    backend=owner.backend,
                )
                self._copies.append((native, swa, retain_native, retain_scratch))
            args = dict(swa=swa, swa_writes=writes, swa_reads=reads)
            if ratio:
                source = sources.kv_source_layer_id
                tag = "csa_kv" if ratio == 2 else "hca_kv"
                global_cache, index_cache = (
                    owner.caches[source, tag],
                    owner.caches[source, "indexer_kv"],
                )
                args.update(
                    kv_source_layer_id=source,
                    global_cache=global_cache,
                    index_cache=index_cache,
                    global_pages=page_maps[tag],
                    index_pages=page_maps["indexer_kv"],
                )
                if sources.owns_kv:
                    for name, cache in (
                        ("global_writes", global_cache),
                        ("index_writes", index_cache),
                    ):
                        args[name] = integer(
                            [
                                (
                                    tables[req][cache.tag][pos // 256]
                                    * cache.layout.entries
                                    + (pos % 256) // ratio
                                    if live and (pos + 1) % ratio == 0
                                    else -1
                                )
                                for req, pos, live in self._records
                            ]
                        )
                    if ratio == 2:
                        pair = owner.caches[layer, "csa_state"]
                        previous_rows, previous_slots, state_writes = [], [], []
                        for req, pos, live in self._records:
                            previous = (
                                lookup.get((req, pos - 1), -1)
                                if live and pos % 2
                                else -1
                            )
                            old = live and pos % 2 and previous < 0
                            previous_rows.append(previous)
                            previous_slots.append(
                                tables[req]["csa_state"][0] * pair.entries
                                + (pos - 1) % pair.entries
                                if old
                                else -1
                            )
                            pending = (
                                live
                                and pos % 2 == 0
                                and pos == by_id[req].start + by_id[req].count - 1
                            )
                            state_writes.append(
                                tables[req]["csa_state"][0] * pair.entries
                                + pos % pair.entries
                                if pending
                                else -1
                            )
                        args.update(
                            pair_state=pair,
                            pair_metadata=V41PairMetadata(
                                self.rows.request_ids,
                                self.rows.positions,
                                self.rows.valid_rows,
                                integer(previous_rows),
                                integer(previous_slots),
                                integer(state_writes),
                            ),
                        )
            self._plans[layer] = V41AttentionPlan(**args)

    def _execute(self, run):
        self._check()
        owner = self.owner
        self._started, owner._drained = True, False
        try:
            result = run(self.plans)
            error = None
            try:
                for native, scratch, destination, source in self._copies:
                    native.scatter(
                        scratch.gather(source, backend=owner.backend),
                        destination,
                        backend=owner.backend,
                    )
                if owner.device.type == "cuda":
                    torch.cuda.synchronize(owner.device)
            except Exception as exc:
                error = exc
            if self._cp_forward is not None:
                self._cp_forward.session._consensus(
                    _cp_signature(self._cp_forward, "cache-commit", self.spans), error
                )
            elif error is not None:
                raise error
            for span in self.spans:
                state = owner._state(span.identity)
                state.visible, state.chunk = span.start + span.count, span.chunk_id + 1
            owner._drained = True
            return result
        except Exception:
            owner._failed = True
            try:
                if owner.device.type == "cuda":
                    torch.cuda.synchronize(owner.device)
                owner._drained = True
            except Exception:
                pass
            raise
        finally:
            owner._active = None
            if not owner._drained:
                owner._undrained = self

    def _stage(self, stage):
        if not isinstance(stage, V41Stage) or (
            stage.role,
            stage.pp_rank,
            stage.device,
            stage.max_positions,
        ) != (
            self.owner.role,
            self.owner.pp_rank,
            self.owner.device,
            self.owner.max_positions,
        ):
            raise ValueError(
                "V4.1 cache batch belongs to a different stage/device/context"
            )

    def _preflight(self, stage, cp_forward, validate):
        error = None
        try:
            self._check()
            self._stage(stage)
            if cp_forward is not self._cp_forward:
                raise ValueError(
                    "V4.1 CP cache execution requires its prepare_cp admission"
                )
            if cp_forward is not None:
                self.rows.require_same(cp_forward.rows)
            validate()
        except Exception as exc:
            error = exc
        if self._cp_forward is not None:
            self._cp_forward.session._consensus(
                _cp_signature(self._cp_forward, "cache-execute", self.spans), error
            )
        elif error is not None:
            raise error

    def run(self, stage, state, *, windows=None, cp_forward=None):
        def validate():
            if stage.begin == 21:
                raise ValueError(
                    "V4.1 stage-one cache execution requires the PP adapter"
                )

        self._preflight(stage, cp_forward, validate)
        if cp_forward is None:
            return self._execute(
                lambda plans: stage.run(state, self.rows, plans, windows=windows)
            )
        return self._execute(
            lambda plans: cp_forward.run(
                stage, state, cp_forward.local_rows, plans, windows=windows
            )
        )

    def run_pp(self, stage, tensors, receiver, dispatch, *, cp_forward=None):
        from rtp_llm.models_py.modules.dsv4.v41_stage_pp import run_v41_pp_stage

        def validate():
            if tuple(dispatch.spans) != self.spans:
                raise ValueError(
                    "V4.1 PP dispatch differs from the admitted cache chunks"
                )
            self.rows.require_same(
                dispatch.forward_rows(self.owner.device)
                if cp_forward is None
                else cp_forward.rows
            )

        self._preflight(stage, cp_forward, validate)
        return self._execute(
            lambda plans: run_v41_pp_stage(
                stage, tensors, receiver, dispatch, plans, cp_forward=cp_forward
            )
        )
