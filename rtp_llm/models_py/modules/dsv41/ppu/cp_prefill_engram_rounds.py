"""Independent full-CP8 Engram round certificate, using the native ENP ABI.

The entry vote proves a uniform flattened count and protocol, not equal
request boundaries across ranks. Hash owners are published by their actual
producer; the native CPContext birth is authenticated only after public
Prefill binds it and before either layer starts its lookup collectives.
"""

from dataclasses import dataclass, field
import weakref

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv41.ppu import engram
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import env_value


CP8_ENGRAM_ROUNDS_ENV = "RTP_V41_PPU_CP8_YOCO_ENGRAM_RS"
CP8_MODE_TAG = 1 << 60
CP8_PEER_SHARDS_TAG = 1 << 59
CP8_MAX_FLAT_ROWS = 1 << 40


def cp8_engram_topology(model, rank):
    args, parallel = model._v4_args, model.parallelism_config
    return (getattr(parallel, "tp_size", None) == 8 and getattr(parallel, "tp_rank", None) == rank
            and tuple(getattr(args, name, None) for name in ("tp_size", "dp_size", "ep_size")) == (1, 1, 8)
            and getattr(args, "dp_rank", None) == 0 and getattr(args, "ep_rank", None) == rank
            and getattr(model, "_prefill_cp_size", None) == 8)


def cp8_engram_prefill_profile(policy):
    return ((policy.mode, policy.keep_tokens, policy.window, policy.max_rollback_tokens,
             policy.encoder_replay) == ("compact", 140, 128, 0, False)
            and env_value("RTP_V41_PPU_CP8_YOCO_FAST_PREFILL", "0") == "1")


def _plain(value):
    return (type(value) is torch.Tensor and value.layout == torch.strided
            and not value.is_neg() and not value.is_conj() and value.is_contiguous())


def _device_supported(device):
    return (device.type == "cuda" and engram.is_ppu()
            and torch.cuda.get_device_capability(device) == (8, 9)
            and not torch.cuda.is_current_stream_capturing())


def _device_rows_supported(rows, device):
    return rows.is_cuda and rows.device == device


def _entry_snapshot(runtime, state):
    """Read existing CPU metadata only; never read accelerator tensor values."""
    try:
        inputs, attn, cache, model = state.inputs, state.attn, state.cache, runtime.model
        info = attn.context_parallel_info
        lengths, prefixes = state.lengths, state.query_prefixes
        host = info.prefill_actual_input_lengths_cpu, info.prefill_prefix_lengths_cpu
        if (not all(_plain(value) and value.device.type == "cpu" and value.ndim == 1
                    and value.dtype in (torch.int32, torch.int64) for value in host)
                or tuple(host[0].tolist()) != lengths or tuple(host[1].tolist()) != prefixes
                or not lengths or len(lengths) != len(prefixes)
                or any(type(n) is not int or n <= 0 for n in lengths)
                or any(type(p) is not int or p < 0 for p in prefixes)):
            return None
        chunks = tuple((n + 15) // 16 * 2 for n in lengths)
        source = getattr(info, "prefill_cp_chunk_lengths", None)
        if source is not None and (not _plain(source) or source.ndim != 1
                                   or source.dtype not in (torch.int32, torch.int64)):
            return None
        if source is not None and source.device.type == "cpu" and tuple(source.tolist()) != chunks:
            return None
        ids, windows = inputs.input_ids, inputs.engram_token_windows
        if (not _plain(ids) or ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64)
                or ids.device != runtime.device or ids.numel() != sum(chunks)
                or not _plain(windows) or windows.dtype not in (torch.int32, torch.int64)
                or windows.shape != (ids.numel(), runtime.geometry.engram_max_ngram_size)
                or windows.device not in (torch.device("cpu"), runtime.device)
                or cache.lengths != tuple(p + n for p, n in zip(prefixes, lengths))):
            return None
        tensors = (*host, source, ids, windows)
        owners = (inputs, attn, cache, model, model.parallelism_config, model._v4_args,
                  runtime.geometry, runtime.hash_params, info, *tensors)
        key = (tuple(id(owner) for owner in owners[:9]), lengths, prefixes, chunks,
               tuple(engram._engram_host_tensor_key(value) if value is not None else None for value in tensors))
        return key, owners, chunks
    except (AttributeError, TypeError, ValueError):
        return None


@dataclass(frozen=True)
class _CPEntry:
    runtime: object
    state_ref: object
    epoch: int
    snapshot: tuple
    owners: tuple
    chunks: tuple
    tokens: int

    def current(self, *, bound):
        state = self.state_ref()
        if (state is None or self.runtime.current is not state or state.runtime is not self.runtime
                or self.runtime._engram_epoch != self.epoch or state.compacted
                or not state.prefill or state.idle
                or any(getattr(state, name, False) for name in ("is_graph", "is_graph_commit", "verify", "proposal"))
                or getattr(state.attn, "is_cuda_graph", False)
                or (self.runtime.device.type == "cuda" and torch.cuda.is_current_stream_capturing())
                or torch.is_grad_enabled()):
            return False
        current = _entry_snapshot(self.runtime, state)
        if current is None or current[0] != self.snapshot:
            return False
        if not bound:
            return state.original_context is None and state.context is None
        cp = state.original_context
        return (cp is state.context and cp is not None and cp.cp_size == 8 and cp.kv_cache_sharded
                and cp.gather_restore_positions is None and cp.chunk_length == self.tokens
                and cp.chunk_lengths_per_req == self.chunks
                and cp.input_lengths_global_host == state.lengths
                and cp.prefix_lengths_host == state.query_prefixes
                and cp.cp_rank == dist.get_rank(dist.group.WORLD)
                and state._cp_rows_phase() and state._cp_full_rows_valid())


@dataclass
class _CPLease:
    active: bool = True
    schedule: object = None
    publications: dict = field(default_factory=dict)
    records: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CP8PrefillEngramSchedule:
    group: object
    rank: int
    epoch: int
    device: torch.device
    round_cap: int
    local_flat_rows: int
    group_max_rows: int
    _entry: _CPEntry = field(repr=False, compare=False)
    _bindings: tuple = field(repr=False, compare=False)
    _lease: _CPLease = field(repr=False, compare=False)
    layers: tuple = (1, 14)
    peer_shards: bool = False

    def publish(self, layer_id, rows):
        lease = self._lease
        state = self._entry.state_ref()
        if (state is None or getattr(state, "engram_round_schedule", None) is not self
                or not lease.active or lease.schedule is not self or not self._entry.current(bound=False)
                or len(lease.publications) >= 2 or layer_id != self.layers[len(lease.publications)]
                or not _plain(rows) or rows.dtype != torch.int64 or rows.device != self.device
                or rows.shape != (self.local_flat_rows // 24, 24)
                or any(torch._C._is_alias_of(rows, value) for value in self._entry.owners[-2:])):
            raise RuntimeError("CP8 Engram hash publication differs from its admitted producer")
        lease.publications[layer_id] = rows, engram._engram_host_tensor_key(rows)
        lease.evidence.setdefault("published_layers", []).append(layer_id)

    def claim(self, adapter, shard, layout, rows, layer_id):
        lease = self._lease
        state = self._entry.state_ref()
        if (state is None or getattr(state, "engram_round_schedule", None) is not self
                or not lease.active or lease.schedule is not self or not self._entry.current(bound=True)
                or len(lease.records) >= 2 or layer_id != self.layers[len(lease.records)]
                or type(adapter) is not engram.DistributedEnpmGroup
                or adapter.pg is not self.group or self.group is not dist.group.WORLD
                or not dist.is_initialized() or adapter.size != 8 or adapter.rank != self.rank
                or shard.rank != self.rank or not shard.pinned
                or not shard.payload.is_pinned() or not shard.scale.is_pinned()
                or self.local_flat_rows != self.group_max_rows):
            raise RuntimeError("CP8 Engram uniform schedule binding/lifecycle changed after admission")
        published = lease.publications.get(layer_id)
        if (published is None or published[0] is not rows or not _plain(rows)
                or not _device_rows_supported(rows, self.device)
                or rows.shape != (self.local_flat_rows // 24, 24) or rows.dtype != torch.int64
                or published[1] != engram._engram_host_tensor_key(rows)):
            raise RuntimeError("CP8 Engram hash owner changed after publication")
        bound = self._bindings[len(lease.records)]
        fields = (layout.num_embeddings, layout.padded_rows, layout.num_per_part, layout.enp_size, shard.dim)
        if (bound[0] != layer_id or bound[1] is not shard or bound[2] is not layout or bound[3] is not adapter
                or bound[4] != fields
                or bound[5] != tuple(engram._engram_host_tensor_key(value) for value in (shard.payload, shard.scale))):
            raise RuntimeError("CP8 Engram pinned host binding changed after admission")
        record = dict(layer=layer_id, local_rows=rows.numel(), group_max_rows=self.group_max_rows,
                      rounds=[], reduce_scatter=True, uniform_rank_rows=True, complete=False, contribution_bytes=0)
        lease.records.append(record)
        lease.evidence["native_birth_claimed"] = True
        lease.evidence.setdefault("native_birth_layers", []).append(layer_id)
        return record

    def close(self):
        lease = self._lease
        if lease.active and lease.schedule is not self:
            raise RuntimeError("CP8 Engram lease belongs to another schedule")
        complete = (lease.active and len(lease.records) == 2 and all(record["complete"] for record in lease.records))
        lease.active, lease.schedule = False, None
        lease.publications.clear()
        return complete


def prepare_cp8_prefill_engram_schedule(runtime, state, *, epoch, evidence):
    """One WORLD MAX, with exactly the existing DP packet's29-int64 schema.

    CP replaces the last dimension pair by a tag, backend bit and flat count.
    The binding capture still checks dimension 256 locally. Mixed modes,
    mixed backends and unequal counts reject before any layer lookup.
    The packet does not claim agreement on request boundaries across ranks.
    """
    model, device = runtime.model, runtime.device
    rank = dist.get_rank(dist.group.WORLD)
    flag = getattr(runtime, "_cp_engram_requested", False)
    flag_valid = getattr(runtime, "_cp_engram_flag_valid", True)
    ops = model._shared_ops
    bindings = getattr(ops, "_engram", {})
    captured, geometry, binding_bad = engram._capture_local_engram_bindings(bindings, rank)
    group_bad = getattr(getattr(ops, "_enpm_group", None), "pg", None) is not dist.group.WORLD
    snapshot = _entry_snapshot(runtime, state)
    count = state.inputs.input_ids.numel() * 24
    count_valid = type(count) is int and 0 <= count < CP8_MAX_FLAT_ROWS and count % 24 == 0
    mapped_layers = sum(item[1]._peer_table is not None for item in captured)
    peer_shards = mapped_layers == 2
    protocol = CP8_MODE_TAG + (CP8_PEER_SHARDS_TAG if peer_shards else 0) + (count if count_valid else 0)
    geometry = (*geometry[:9], protocol)
    topology = cp8_engram_topology(model, rank)
    business = (state.prefill and not state.idle and not state.compacted and state.cache is not None
                and state.cache._is_prefill and state.cache.sharded
                and not getattr(model, "_is_speculative", False)
                and not any(getattr(state, name, False) for name in ("is_graph", "is_graph_commit", "verify", "proposal"))
                and not getattr(state.attn, "is_cuda_graph", False)
                and not torch.is_grad_enabled()
                and cp8_engram_prefill_profile(runtime.policy))
    pinned = bool(captured) and all(item[1].pinned and item[1].payload.is_pinned()
                                  and item[1].scale.is_pinned() for item in captured)
    supported = bool(topology and not binding_bad and not group_bad and pinned and business and snapshot is not None
                     and _device_supported(device)
                     and tuple(runtime.geometry.engram_layer_ids) == (1, 14))
    native_bad = False
    if supported:
        try:
            supported = engram._triton_kernels() is not None
        except Exception:
            supported, native_bad = False, True
    on = int(flag is True or peer_shards)
    rejected = (not flag_valid or not count_valid or type(flag) is not bool or mapped_layers == 1
                or type(epoch) is not int or not 1 <= epoch < 2**62
                or runtime._engram_incomplete or not runtime._engram_flag_valid or runtime._engram_requested
                or (on and (not topology or binding_bad or group_bad or native_bad)))
    generation = epoch if type(epoch) is int and 0 <= epoch < 2**62 else 0
    dimensions = [item for value in geometry for item in (value, -value)]
    cap = engram.LOCAL_PREFILL_ENGRAM_ROUND_CAP
    packet = torch.tensor([on, -on, cap, -cap, generation, -generation,
                           int(not supported), int(bool(rejected)), *dimensions,
                           count if count_valid else 0], dtype=torch.int64, device=device)
    dist.all_reduce(packet, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
    values = packet.tolist()
    evidence.clear()
    evidence.update(decision="rejected", protocol="cp8_uniform", epoch=generation,
                    round_cap=cap, packet=values, rank=rank, local_flat_rows=count,
                    group_max_rows=values[-1], consumed_layers=[],
                    entry_source_proof=snapshot is not None, native_birth_claimed=False,
                    cross_rank_agreement="protocol_and_flat_count; request_boundaries_checked_locally")
    mismatch = any(values[i] != -values[i + 1] for i in (0, 2, 4, *range(8, 28, 2)))
    if values[7] or mismatch or values[-1] != count:
        raise RuntimeError(f"CP8 Engram configuration/lifecycle disagreement: {values!r}")
    if not values[0] or values[6]:
        evidence["decision"] = "disabled" if not values[0] else "fallback"
        return None
    entry = _CPEntry(runtime, weakref.ref(state), generation, snapshot[0], snapshot[1], snapshot[2], count // 24)
    lease = _CPLease(evidence=evidence)
    schedule = CP8PrefillEngramSchedule(dist.group.WORLD, rank, generation, device, cap,
                                      count, count, entry, captured, lease, peer_shards=peer_shards)
    lease.schedule = schedule
    evidence.update(decision="adopted", consumed_layers=lease.records)
    return schedule
