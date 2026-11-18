"""Bind-time packing of the existing PTPC Q-a and KV-a projections.

Only the output-channel layout changes: resident E4M3 bytes and FP32 channel
scales are concatenated once, then the existing PTPC linear launches one GEMM.
The caller controls Prefill opt-in. The deferred CP140 helper also checks
the shared projection/MegaAttention switches before selecting its protocol.

Model weights are immutable after binding. Ordinary tensor writes, rebinding
and metadata changes invalidate this helper before activation quantization.
Inference tensors have no version counter, so their immutable-weight contract
also applies to writes through aliases and to captured Graph replays.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.ppu.linear import (
    PPUPTPCLinear,
    prepare_shared_ptpc_input,
)


def _physical_pair_supported(first, second):
    return first.weight.is_cuda and second.weight.is_cuda


def _linear_supported(op):
    if type(op) is not PPUPTPCLinear:
        return False
    weight, scale = getattr(op, "weight", None), getattr(op, "scale", None)
    return (
        isinstance(weight, torch.Tensor)
        and isinstance(scale, torch.Tensor)
        and weight.layout == scale.layout == torch.strided
        and weight.dtype == torch.float8_e4m3fn
        and weight.ndim == 2
        and min(weight.shape) > 0
        and scale.dtype == torch.float32
        and tuple(scale.shape) == (weight.shape[0], 1)
        and weight.device == scale.device
        and weight.is_contiguous()
        and scale.is_contiguous()
        and not weight.is_conj()
        and not weight.is_neg()
        and not scale.is_conj()
        and not scale.is_neg()
        and not weight.requires_grad
        and not scale.requires_grad
        and getattr(op, "output_dtype", None) == torch.bfloat16
        and getattr(op, "quantization", None) == "fp8_per_channel_per_token"
        and (
            not weight.is_cuda
            or callable(getattr(op, "_gemm", None))
        )
    )


def _tensor_key(tensor):
    # Inference tensors deliberately have no counter. Reading it is a
    # metadata-only RuntimeError, not a reason to scan/synchronize the weight.
    version = None if torch.is_inference(tensor) else tensor._version
    return (
        id(tensor), tensor.untyped_storage()._cdata, tensor.data_ptr(),
        tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(),
        tensor.dtype, tensor.device, tensor.layout, tensor.requires_grad,
        tensor.is_conj(), tensor.is_neg(), version,
    )


def _linear_key(op):
    return (
        _tensor_key(op.weight), _tensor_key(op.scale),
        op.output_dtype, op.quantization, op._gemm,
    )


class PreparedQKV:
    """A frozen packed PTPC linear retaining both original weight owners."""

    def __init__(self, first, second, packed):
        self._sources = (first, second)
        self._source_keys = tuple(_linear_key(op) for op in self._sources)
        self._linear = packed
        self._packed_key = _linear_key(packed)
        self._q_width = first.weight.shape[0]

    @classmethod
    def try_prepare(cls, wq_a, wkv):
        """Pack compatible resident weights once, or decline using metadata.

        No quantization, dequantization or scale conversion is introduced.
        Initialization/allocation failures propagate instead of being hidden
        behind a second provider.
        """
        if not (_linear_supported(wq_a) and _linear_supported(wkv)):
            return None
        if (
            wq_a.weight.device != wkv.weight.device
            or wq_a.weight.shape[1] != wkv.weight.shape[1]
            or wq_a._gemm is not wkv._gemm
            or not _physical_pair_supported(wq_a, wkv)
        ):
            return None
        # Some CPU Torch builds do not implement cat for float8. Concatenating
        # its byte view preserves every already-quantized E4M3 bit pattern.
        weight = torch.cat(
            (wq_a.weight.view(torch.uint8), wkv.weight.view(torch.uint8)), dim=0
        ).view(torch.float8_e4m3fn)
        scale = torch.cat((wq_a.scale, wkv.scale), dim=0)
        packed = PPUPTPCLinear(weight, scale)
        packed._gemm = wq_a._gemm
        return cls(wq_a, wkv, packed)

    def _current(self):
        return (
            all(_linear_supported(op) for op in self._sources)
            and tuple(_linear_key(op) for op in self._sources) == self._source_keys
            and _linear_supported(self._linear)
            and _linear_key(self._linear) == self._packed_key
        )

    def _input_supported(self, value):
        return (
            isinstance(value, torch.Tensor)
            and value.layout == torch.strided
            and value.ndim == 2
            and value.dtype == torch.bfloat16
            and value.shape[-1] == self._linear.weight.shape[1]
            and value.device == self._linear.weight.device
            and value.is_contiguous()
            and not value.is_conj()
            and not value.is_neg()
            and not value.requires_grad
            and not torch.is_grad_enabled()
        )

    @staticmethod
    def _pair_supported(pair, value):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            return False
        payload, scale = pair
        return (
            isinstance(payload, torch.Tensor)
            and isinstance(scale, torch.Tensor)
            and payload.layout == scale.layout == torch.strided
            and payload.dtype == torch.float8_e4m3fn
            and scale.dtype == torch.float32
            and tuple(payload.shape) == tuple(value.shape)
            and tuple(scale.shape) == (*value.shape[:-1], 1)
            and payload.device == scale.device == value.device
            and payload.is_contiguous()
            and scale.is_contiguous()
            and not payload.is_conj()
            and not payload.is_neg()
            and not scale.is_conj()
            and not scale.is_neg()
            and not payload.requires_grad
            and not scale.requires_grad
        )

    def project(self, x, *, quantized_input=None):
        """Return two raw BF16 views, or None before any activation/device work.

        A supplied pair belongs to this invocation. No activation or output
        cache survives a call; native quantizer/GEMM failures propagate.
        """
        if not self._current() or not self._input_supported(x):
            return None
        if quantized_input is not None and not self._pair_supported(quantized_input, x):
            return None
        pair = (
            self._linear.quantize_input(x)
            if quantized_input is None else quantized_input
        )
        projected = self._linear.forward_quantized(*pair)
        # Both views retain the fresh GEMM output. Norm accepts this explicit
        # row stride, so no forward-time concatenation or raw-output copy occurs.
        return projected[:, :self._q_width], projected[:, self._q_width:]


@lru_cache(maxsize=64)
def _cp8_selected_rows(lengths):
    """Pure request-major retained rows; no plan or validation is cached."""
    selected, offset = [], 0
    for length in lengths:
        selected.extend(range(offset + max(length - 140, 0), offset + length))
        offset += length
    return tuple(selected)


def _cp8_plan_geometry(plan):
    """Describe only the shared host domain of the existing CP140 planner."""
    original, compact, exchange = plan.original_context, plan.context, plan.exchange
    lengths, prefixes = original.input_lengths_global_host, original.prefix_lengths_host
    chunks = original.chunk_lengths_per_req
    if (
        original.cp_size != 8 or not original.kv_cache_sharded
        or compact.cp_size != original.cp_size or compact.cp_rank != original.cp_rank
        or not compact.kv_cache_sharded or not 0 <= original.cp_rank < 8
        or not lengths or prefixes is None or chunks is None
        or len(lengths) != len(prefixes) or len(lengths) != len(chunks)
        or any(type(n) is not int or n <= 0 for n in lengths)
        or any(type(p) is not int or p < 0 for p in prefixes)
        or sum(lengths) != original.seq_len_full or sum(chunks) != original.chunk_length
        or original.padded_seq_len != original.chunk_length * 8
        or original.gather_restore_positions is not None
        or sum(min(n, 140) for n in lengths) >= sum(lengths)
    ):
        return None
    compact_chunks = tuple((min(n, 140) + 15) // 16 * 2 for n in lengths)
    selected = _cp8_selected_rows(tuple(lengths))
    if (
        compact.input_lengths_global_host != lengths or compact.prefix_lengths_host != prefixes
        or compact.chunk_lengths_per_req != compact_chunks
        or compact.chunk_length != sum(compact_chunks)
        or compact.padded_seq_len != sum(compact_chunks) * 8
        or compact.seq_len_full != original.seq_len_full
        or exchange.original_rows != original.chunk_length
        or exchange.compact_rows != compact.chunk_length
        or len(exchange.send_sizes) != 8 or len(exchange.receive_sizes) != 8
        or exchange.candidate_rows_host is None
        or sum(exchange.send_sizes) != len(exchange.candidate_rows_host)
        or any(type(n) is not int or n < 0 for n in (*exchange.send_sizes, *exchange.receive_sizes))
    ):
        return None
    return (
        lengths, prefixes, chunks, compact_chunks, selected, original.cp_rank,
        original.chunk_length, original.padded_seq_len, original.seq_len_full,
        tuple(exchange.send_sizes), tuple(exchange.receive_sizes),
        exchange.candidate_rows_host, exchange.projection_groups_host,
        exchange.receive_is_identity, id(exchange.group),
        getattr(compact, "swa_replay_start", None),
        getattr(compact, "swa_replay_starts_host", None),
    )


def _cp8_plan_transport(plan):
    compact, exchange = plan.context, plan.exchange
    values = tuple(getattr(compact, name, None) for name in (
        "global_positions", "req_id_per_token", "local_is_real", "unpad_restore",
        "gather_restore_positions", "relative_positions",
    )) + (exchange.send_indices, exchange.receive_positions, plan.cu_seqlens)
    selected = sum(min(n, 140) for n in compact.input_lengths_global_host)
    sizes = ((compact.chunk_length,) * 3 + (selected,) * 2 + (compact.chunk_length,)
             + (sum(exchange.send_sizes), sum(exchange.receive_sizes),
                len(compact.input_lengths_global_host) + 1))
    dtypes = (torch.int64, torch.int32, torch.bool) + (torch.int64,) * 5 + (torch.int32,)
    if any(not isinstance(value, torch.Tensor) or value.shape != (size,) or value.dtype != dtype
           for value, size, dtype in zip(values, sizes, dtypes)):
        return None
    device = values[0].device
    if any(value.device != device for value in values):
        return None
    if any(len(group) != 2 or any(not isinstance(value, torch.Tensor) or value.dtype != torch.int64
                                 or value.device != device or value.ndim != 1 for value in group)
           or group[0].shape != group[1].shape for group in plan.indexer_groups):
        return None
    values += tuple(value for group in plan.indexer_groups for value in group)
    host = getattr(exchange, "_layout_host_buffer", None)
    if host is not None:
        values += (host,)
    # These are private, construction-owned replacement transport tensors.
    # Unknown inference-only metadata must not be certified by pointer equality.
    if any(not isinstance(value, torch.Tensor) or value.layout != torch.strided
           or value.is_inference() or value.is_neg() or value.is_conj()
           or value.requires_grad or not value.is_contiguous() for value in values):
        return None
    return values


class CP8QueryPlanBirth:
    """A versioned private CP140 plan, minted only by bounded_plan's factory.

    Original native metadata retains the framework's read-only-within-forward
    contract. Its strong owners and available version counters are checked;
    retained-row values are never inferred from versionless device metadata.
    The actual replacement collective/position tensors are all versioned.
    """

    def __init__(self, plan):
        geometry, transport = _cp8_plan_geometry(plan), _cp8_plan_transport(plan)
        if geometry is None or transport is None:
            raise ValueError("CP140 query plan requires construction-owned versioned transport")
        self.plan, self.original, self.compact, self.exchange = (
            plan, plan.original_context, plan.context, plan.exchange,
        )
        self.geometry, self.transport = geometry, transport
        self.keys = tuple(_tensor_key(value) for value in transport)
        self.original_sources = tuple(getattr(self.original, name, None) for name in (
            "global_positions", "req_id_per_token", "local_is_real", "unpad_restore",
            "relative_positions", "prefix_lengths", "input_lengths_global", "cu_seqlens_global",
        ))
        self.original_keys = tuple(_tensor_key(value) if value is not None else None
                                   for value in self.original_sources)
        self.borrowed_names = ("prefix_lengths", "input_lengths_global", "cu_seqlens_global")
        self.borrowed = tuple(getattr(self.compact, name, None) for name in self.borrowed_names)
        if any(value is not getattr(self.original, name, None)
               for name, value in zip(self.borrowed_names, self.borrowed)):
            raise ValueError("CP140 query context changed original source metadata")

    def current(self):
        plan = self.plan
        transport = _cp8_plan_transport(plan)
        original = tuple(getattr(self.original, name, None) for name in (
            "global_positions", "req_id_per_token", "local_is_real", "unpad_restore",
            "relative_positions", "prefix_lengths", "input_lengths_global", "cu_seqlens_global",
        ))
        return (
            plan.original_context is self.original and plan.context is self.compact
            and plan.exchange is self.exchange and _cp8_plan_geometry(plan) == self.geometry
            and transport is not None and len(transport) == len(self.transport)
            and all(old is new for old, new in zip(self.transport, transport))
            and tuple(_tensor_key(value) for value in transport) == self.keys
            and all(old is new for old, new in zip(self.original_sources, original))
            and tuple(_tensor_key(value) if value is not None else None for value in original)
            == self.original_keys
            and all(value is getattr(self.compact, name, None)
                    and value is getattr(self.original, name, None)
                    for name, value in zip(self.borrowed_names, self.borrowed))
        )


def _cp8_l20_host_domain(state):
    if state is None:
        return False
    policy, model = state.runtime.policy, state.runtime.model
    original = state.original_context
    return (
        state.prefill and not state.idle and not state.compacted
        and not torch.is_grad_enabled()
        and not any(getattr(owner, name, False) for owner in (state, state.inputs, state.attn)
                    for name in ("is_graph", "is_graph_commit", "verify", "proposal", "is_cuda_graph",
                                 "is_target_verify", "is_packed_target_verify"))
        and not getattr(model, "_is_decode_role", False)
        and not getattr(model, "_is_speculative", False)
        and getattr(model, "_prefill_cp_size", None) == 8
        and getattr(getattr(model, "parallelism_config", None), "tp_size", None) == 8
        and state.cache is not None and state.cache.sharded
        and original is not None and original.cp_size == 8 and original.kv_cache_sharded
        and state.context is original and state.plan is not None
        and state.plan_reason == "compact140"
        and (policy.mode, policy.keep_tokens, policy.window,
             policy.max_rollback_tokens, policy.encoder_replay) == ("compact", 140, 128, 0, False)
        and original.input_lengths_global_host == state.lengths
        and original.prefix_lengths_host == state.query_prefixes
    )


class CP8L20ProjectionProof:
    """Bind one creation certificate to its exact forward and source owners."""

    def __init__(self, state, birth):
        if not _cp8_l20_host_domain(state) or not isinstance(birth, CP8QueryPlanBirth) or not birth.current():
            raise ValueError("CP L20 projection proof must bind a live CP140 plan birth")
        self.state, self.birth = state, birth
        self.owners = (state.inputs, state.attn, state.cache, state.runtime.model, state.original_context)
        self.input_sources = (state.inputs.input_ids, getattr(state.attn, "cu_seqlens", None))
        self.input_keys = tuple(_tensor_key(value) if value is not None else None for value in self.input_sources)

    def current(self, state):
        sources = (state.inputs.input_ids, getattr(state.attn, "cu_seqlens", None))
        return (
            state is self.state and _cp8_l20_host_domain(state)
            and getattr(state.runtime, "current", None) is state
            and state.plan is self.birth.plan and state.original_context is self.birth.original
            and getattr(state.plan, "_ppu_cp8_query_birth", None) is self.birth
            and all(old is new for old, new in zip(self.owners, (
                state.inputs, state.attn, state.cache, state.runtime.model, state.original_context)))
            and all(old is new for old, new in zip(self.input_sources, sources))
            and tuple(_tensor_key(value) if value is not None else None for value in sources) == self.input_keys
            and self.birth.current()
        )


@dataclass(frozen=True)
class CP8DeferredQueries:
    hidden: torch.Tensor
    latent: torch.Tensor
    query: torch.Tensor
    indexer_query: torch.Tensor
    indexer_weights: torch.Tensor
    quantized_input: object


def _projection_binding_key(attention, mega):
    prepared = getattr(attention, "_local_prefill_qkv", None)
    if (not isinstance(prepared, PreparedQKV) or not prepared._current()
            or not _linear_supported(attention.wq_b) or not _linear_supported(attention.index_wq)
            or mega is None or getattr(mega, "owner", None) is not attention or not mega._current()):
        return None
    return (
        id(prepared), id(mega), _linear_key(attention.wq_b), _linear_key(attention.index_wq),
        _tensor_key(attention.q_norm.weight), attention.q_norm.variance_epsilon,
        _tensor_key(attention.index_weights),
        tuple(getattr(attention.geometry, name, None) for name in (
            "hidden_size", "q_lora_rank", "n_heads", "head_dim", "index_n_heads", "index_head_dim",
        )),
        _tensor_key(attention.runtime.freqs_for(attention.layer_id)),
    )


class DeferredCP8QueryProjection:
    """Invocation-local full producers with late compact QR/Q/Indexer work.

    Host topology/geometry chooses the collective protocol uniformly. Existing
    replicated model bindings already produced joined projections/MegaAttention
    on earlier layers. A changed local binding or epoch is an error after
    admission; it must never fall back into the other collective sequence.
    """

    @classmethod
    def prepare(cls, attention, state, hidden, mega, *, defer_kv_norm=False):
        from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import (
            DENSE_PROJECTIONS, MEGA_ATTENTION, enabled,
        )

        if not enabled(DENSE_PROJECTIONS) or not enabled(MEGA_ATTENTION) or not _cp8_l20_host_domain(state):
            return None
        proof = getattr(state, "_cp_l20_projection_proof", None)
        if not isinstance(proof, CP8L20ProjectionProof) or not proof.current(state):
            raise RuntimeError("CP L20 replacement transport changed or was not certified at birth")
        binding = _projection_binding_key(attention, mega)
        if (binding is None or attention.layer_id != 20 or not attention.is_kv_source
                or not attention.is_index_source or hidden.ndim != 2
                or hidden.shape[0] != proof.birth.original.chunk_length):
            raise RuntimeError("CP L20 admitted model binding is unavailable or changed")
        if hidden.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CP L20 eager projection cannot enter Graph capture")
        joined = attention._local_prefill_qkv.project(hidden)
        if joined is None:
            raise RuntimeError("CP L20 admitted joined projection declined")
        result = cls()
        result.attention, result.state, result.proof, result.mega = attention, state, proof, mega
        result.hidden, result.raw_latent, raw_kv = hidden, joined[0], joined[1]
        result.input_key, result.binding = _tensor_key(hidden), binding
        result.raw_key = _tensor_key(result.raw_latent)
        result.kv = raw_kv if defer_kv_norm else attention.kv_norm(raw_kv, allow_strided=True)
        result.finished = False
        attention._last_local_prefill_qkv_joined = True
        sends = sum(proof.birth.exchange.send_sizes)
        widths = {
            "raw_qr": result.raw_latent.shape[1],
            "full_weight_result": attention.index_weights.shape[0],
            "full_hidden": hidden.shape[1],
        }
        state.shared["cp_l20_projection"] = {
            "original_query_rows": hidden.shape[0], "compact_query_rows": proof.birth.compact.chunk_length,
            "full_weight_rows": hidden.shape[0], "raw_qr_transport": True,
            "raw_qr_width": result.raw_latent.shape[1], "successful": False,
            "query_collectives": 3, "query_send_rows": sends,
            "query_receive_rows": sum(proof.birth.exchange.receive_sizes),
            "query_transport_elements": {name: sends * width for name, width in widths.items()},
            "query_transport_bytes": {
                name: sends * width * hidden.element_size() for name, width in widths.items()
            },
        }
        return result

    def finish(self, *, rope_indexer, producer_complete):
        if self.finished:
            raise RuntimeError("CP L20 deferred query projection cannot be retried")
        if producer_complete is not True:
            raise RuntimeError("CP L20 full producer publication must finish before query transport")
        if (not self.proof.current(self.state) or _tensor_key(self.hidden) != self.input_key
                or _tensor_key(self.raw_latent) != self.raw_key
                or _projection_binding_key(self.attention, self.mega) != self.binding):
            raise RuntimeError("CP L20 deferred producer/binding/transport epoch changed")
        self.finished = True
        attention, exchange = self.attention, self.proof.birth.exchange
        compact_raw = exchange.compact(self.raw_latent)
        latent = attention.q_norm(compact_raw, allow_strided=True)
        pair = prepare_shared_ptpc_input(latent, (attention.wq_b, attention.index_wq))
        query = self.mega.project_q(latent, quantized_input=pair)
        index = (attention.index_wq(latent) if pair is None else
                 attention.project_indexer_query(latent, quantized_input=pair))
        geometry = attention.geometry
        index = index.reshape(-1, geometry.index_n_heads, geometry.index_head_dim)
        positions = self.proof.birth.compact.global_positions
        index = rope_indexer(index, attention.runtime.freqs_for(attention.layer_id)[positions])
        # Keep the original full-M BF16 accumulation and only then transport.
        raw_weights = F.linear(self.hidden, attention.index_weights)
        weights = exchange.compact(raw_weights).float()
        weights = weights * (geometry.index_head_dim * geometry.index_n_heads) ** -0.5
        hidden = exchange.compact(self.hidden)
        if not self.proof.current(self.state):
            raise RuntimeError("CP L20 replacement transport changed during deferred projection")
        self.state.shared["cp_l20_projection"]["successful"] = True
        return CP8DeferredQueries(hidden, latent, query, index, weights, pair)
