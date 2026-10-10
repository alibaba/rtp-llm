"""Read-only native ownership and CP admission before maintained PP execution."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_cache_bindings, dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.cp import build_cp_context_for_forward
from rtp_llm.models_py.modules.dsv4.fp8.attention import bind_attn_cache
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    build_block_tables_batched,
    primary_attention_inputs,
)
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Span,
)

_CP_FIELDS = (
    "prefill_cp_padding_lengths",
    "prefill_cp_chunk_lengths",
    "prefill_shuffle_indices",
    "prefill_qkv_restore_indice",
    "prefill_qkv_padding_mask",
    "prefill_actual_input_lengths_cpu",
)


@dataclass(frozen=True)
class NativePPPrefillDispatch:
    native_inputs: Any  # Opaque C++ leases remain held by this input.
    input_ids: torch.Tensor
    token_windows: torch.Tensor | None
    cp: Any
    spans: tuple[V41L20Span, ...]
    rows: torch.Tensor
    tables: dict[str, torch.Tensor]
    worker_owner: Any

    @property
    def schedule(self):
        return tuple(span.wire_values() for span in self.spans)


def _same_common(value, other, name):
    if value is None or other is None:
        if value is not other:
            raise ValueError(f"V4.1 tagged {name} metadata is incomplete")
    elif (
        not isinstance(value, torch.Tensor)
        or not isinstance(other, torch.Tensor)
        or value.dtype != other.dtype
        or value.shape != other.shape
        or not torch.equal(value.cpu(), other.cpu())
    ):
        raise ValueError(f"V4.1 tagged {name} metadata disagrees")


def _require_cp_wire(fields, spans, rank):
    """Fence malformed indices before the existing builder indexes on CUDA."""
    sizes = [s.count for s in spans]
    chunks = [2 * ((n + 7) // 8) for n in sizes]
    inverse, padding, shuffle = [], [], []
    local_offset = 0
    for size, chunk in zip(sizes, chunks):
        padded, half = 4 * chunk, chunk // 2
        request_inverse = [0] * padded
        for peer in range(4):
            positions = [
                *range(peer * half, (peer + 1) * half),
                *range(padded - (peer + 1) * half, padded - peer * half),
            ]
            for index, position in enumerate(positions):
                request_inverse[position] = peer * sum(chunks) + local_offset + index
            if peer == rank:
                shuffle.extend(positions)
        inverse.extend(request_inverse)
        padding.extend([1] * size + [0] * (padded - size))
        local_offset += chunk
    expected = {
        "prefill_actual_input_lengths_cpu": sizes,
        "prefill_cp_chunk_lengths": chunks,
        "prefill_cp_padding_lengths": [4 * c - n for c, n in zip(chunks, sizes)],
        "prefill_shuffle_indices": shuffle,
        "prefill_qkv_restore_indice": inverse,
        "prefill_qkv_padding_mask": padding,
    }
    for name, values in expected.items():
        value = fields[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.int32
            or value.ndim != 1
            or not value.is_contiguous()
            or value.cpu().tolist() != values
            or (
                name == "prefill_actual_input_lengths_cpu"
                and value.device.type != "cpu"
            )
        ):
            raise ValueError(
                f"V4.1 native CP {name} disagrees with the zigzag dispatch"
            )


def prepare_native_pp_prefill_dispatch(
    worker: Any, inputs: Any
) -> NativePPPrefillDispatch:
    """Validate without cache writes, source publication or model collectives."""
    args, pcfg, cache = worker._v4_args, worker.parallelism_config, worker.kv_cache
    if (
        args.pp_size != 2
        or args.gen_num_per_cycle != 0
        or worker._stage_agreement is None
        or cache is None
        or torch.cuda.is_current_stream_capturing()
        or torch.cuda.current_stream(worker._weight_device()) != worker._worker_stream
    ):
        raise NotImplementedError(
            "V4.1 native PP preflight requires eager gen=0 CEP4PP2 on the worker stream"
        )
    ids = inputs.input_ids
    device = worker._weight_device()
    if (
        not isinstance(ids, torch.Tensor)
        or ids.device != device
        or ids.ndim != 1
        or ids.dtype not in (torch.int32, torch.int64)
        or not ids.is_contiguous()
        or not ids.numel()
        or bool(((ids < 0) | (ids >= args.vocab_size)).any())
    ):
        raise ValueError(
            "V4.1 native PP preflight requires valid padded local token IDs"
        )
    mm = inputs.multimodal_inputs
    if (
        mm.multimodal_features
        or mm.mm_extra_input
        or (mm.mm_features_locs is not None and mm.mm_features_locs.numel())
        or bool((ids == args.v41_config.get("image_token_id", 129264)).any())
    ):
        raise NotImplementedError("V4.1 PP image inputs are not integrated")
    wire = inputs.v41_cache_spans
    if (
        not isinstance(wire, torch.Tensor)
        or wire.device.type != "cpu"
        or wire.dtype != torch.int64
        or wire.ndim != 2
        or wire.shape[1] != 6
        or not 1 <= wire.shape[0] <= min(2, worker._max_generate_batch_size)
        or not wire.is_contiguous()
    ):
        raise ValueError("V4.1 PP preflight requires native one/two-request spans")
    spans = tuple(
        V41L20Span(V41CacheIdentity(*row[:3]), *row[3:]) for row in wire.tolist()
    )
    if (
        len({s.identity.request_id for s in spans}) != len(spans)
        or len({s.identity.generation for s in spans}) != len(spans)
        or any(s.count <= 0 or s.start + s.count > 256 for s in spans)
    ):
        raise ValueError("V4.1 PP preflight requires distinct short request spans")
    # The C++ API compares leases with allocator provenance rather than the
    # projected cache topology; Python receives copies, never lease constructors.
    leased = cache.native_allocation_tables(inputs)
    expected_tags = {
        b.tag for b in dsv41_cache_bindings(args.v41_config, 2, args.pp_rank)
    }
    tagged = inputs.attention_inputs
    if not isinstance(tagged, Mapping) or set(tagged) != expected_tags:
        raise ValueError("V4.1 PP preflight requires every stage-owned tagged input")
    attn = primary_attention_inputs(tagged, cache)
    if attn.context_parallel_info is None:
        raise ValueError("V4.1 PP preflight requires native CP metadata")
    info = attn.context_parallel_info
    cp_fields = {}
    for name in _CP_FIELDS:
        value = getattr(info, name, None)
        cp_fields[name] = value.clone() if isinstance(value, torch.Tensor) else value
    _require_cp_wire(cp_fields, spans, pcfg.tp_rank)
    cp = build_cp_context_for_forward(
        SimpleNamespace(**cp_fields),
        4,
        pcfg.tp_rank,
        ids.numel(),
        device,
        prefix_lengths=attn.prefix_lengths,
        kv_cache_sharded=False,
    )
    lengths = torch.tensor(cp.chunk_lengths_per_req, device=device, dtype=torch.int32)
    prefixes = torch.tensor([s.start for s in spans], device=device, dtype=torch.int32)
    if (
        tuple(cp.input_lengths_global_host) != tuple(s.count for s in spans)
        or tuple(cp.prefix_lengths_host) != tuple(s.start for s in spans)
        or cp.chunk_length != ids.numel()
        or cp.seq_len_full != sum(s.count for s in spans)
        or cp.chunk_lengths_per_req != tuple(2 * ((s.count + 7) // 8) for s in spans)
    ):
        raise ValueError("V4.1 PP CP geometry disagrees with native dispatch spans")
    cu = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cpu().long().cumsum(0)))
    for tag, value in tagged.items():
        if not value.is_prefill or value.is_cuda_graph or value.is_target_verify:
            raise NotImplementedError(
                "V4.1 PP preflight requires eager target prefill inputs"
            )
        for name in (
            "input_lengths",
            "prefix_lengths",
            "sequence_lengths",
            "cu_seqlens",
            "cu_seqlens_device",
            "combo_position_ids",
        ):
            _same_common(getattr(attn, name), getattr(value, name), name)
        for name in _CP_FIELDS:
            other = getattr(value.context_parallel_info, name, None)
            _same_common(cp_fields[name], other, name)
        # Native CP publishes CUDA lengths and CUDA cumulative boundaries.
        # Its CPU cumulative mirror is deliberately empty. Component callers
        # may supply only a populated CPU mirror; validate every supplied copy.
        host_cu, device_cu = value.cu_seqlens, value.cu_seqlens_device
        cumulative = [host_cu] if host_cu is not None and host_cu.numel() else []
        if device_cu is not None:
            cumulative.append(device_cu)
        valid_cu = bool(cumulative) and all(
            boundary.dtype in (torch.int32, torch.int64)
            and boundary.ndim == 1
            and boundary.is_contiguous()
            and torch.equal(boundary.cpu().long(), cu)
            for boundary in cumulative
        )
        if (
            value.input_lengths is None
            or value.input_lengths.dtype not in (torch.int32, torch.int64)
            or value.prefix_lengths is None
            or value.prefix_lengths.dtype not in (torch.int32, torch.int64)
            or not torch.equal(value.input_lengths.cpu().long(), lengths.cpu().long())
            or not torch.equal(value.prefix_lengths.cpu().long(), prefixes.cpu().long())
            or (value.sequence_lengths is not None and value.sequence_lengths.numel())
            or not valid_cu
            or (device_cu is not None and device_cu.device != device)
        ):
            raise ValueError(f"V4.1 {tag} CP request boundaries disagree")
        positions = value.combo_position_ids
        if (
            positions is not None
            and positions.numel()
            and (
                positions.shape != (ids.numel(),)
                or positions.dtype not in (torch.int32, torch.int64)
                or not torch.equal(
                    positions.to(device).long()[cp.local_is_real],
                    cp.global_positions[cp.local_is_real],
                )
            )
        ):
            raise ValueError(f"V4.1 {tag} CP token positions disagree")
    tables = build_block_tables_batched(cache, tagged)
    if set(tables) != expected_tags:
        raise ValueError("V4.1 PP block tables have the wrong native tag roster")
    for tag, table in tables.items():
        expected = [list(row[tag]) for row in leased]
        if (
            table.dtype != torch.int32
            or table.device != device
            or table.shape != (len(spans), 1)
            or not table.is_contiguous()
            or any(len(row[tag]) != 1 for row in leased)
            or table.tolist() != expected
        ):
            raise ValueError(
                f"V4.1 {tag} physical pages disagree with native ownership"
            )
    layouts = dsv41_page_layouts(0)
    bindings = dsv41_cache_bindings(args.v41_config, 2, args.pp_rank)
    expected_layers = {}
    for binding in bindings:
        expected_layers.setdefault(binding.storage_layer_id, set()).add(binding.tag)
        layer = cache.get_layer_cache(binding.storage_layer_id, binding.tag)
        raw = layer.kv_cache_base
        if (
            raw.device != device
            or not raw.is_contiguous()
            or raw.dtype
            != (torch.float32 if binding.tag == "csa_state" else torch.uint8)
            or raw.numel() * raw.element_size()
            != raw.shape[0] * layouts[binding.tag].stride_bytes
            or layer.read_replica_source_layer_id
            != (20 if binding.is_read_replica else None)
        ):
            raise ValueError(
                "V4.1 PP preflight requires exact gen=0 native byte layouts"
            )
    if cache.layer_count != len(worker.v4.layers):
        raise ValueError("V4.1 PP native cache and maintained stage roster disagree")
    for index, layer in enumerate(worker.v4.layers):
        if {g.tag for g in cache.get_layer_cache_groups(index)} != expected_layers[
            index
        ]:
            raise ValueError("V4.1 PP native storage layer has an invalid tag roster")
        with bind_attn_cache(layer.attn, cache, tables, cp_ctx=cp):
            layer.attn._require_qualified_cp_prefill(lengths, prefixes)
    windows = inputs.engram_token_windows
    if windows is None:
        if worker._engram_layers:
            raise ValueError(
                "V4.1 stage-zero PP preflight requires Engram token history"
            )
    elif (
        windows.shape != (ids.numel(), 4)
        or windows.dtype != torch.int32
        or not windows.is_contiguous()
        or bool(((windows < -1) | (windows >= args.vocab_size)).any())
    ):
        raise ValueError("V4.1 PP Engram history has an invalid layout or token ID")
    else:
        windows = windows.to(device)
        if bool((windows[~cp.local_is_real] != -1).any()) or not torch.equal(
            windows[cp.local_is_real, 0].long(), ids[cp.local_is_real].long()
        ):
            raise ValueError(
                "V4.1 PP Engram history disagrees with CP real/padded rows"
            )
    rows = torch.stack(
        (cp.req_id_per_token.long(), cp.global_positions, cp.local_is_real.long()), -1
    ).cpu()
    rows[~rows[:, 2].bool()] = torch.tensor([-1, -1, 0])
    return NativePPPrefillDispatch(
        inputs,
        ids.clone(),
        None if windows is None else windows.clone(),
        cp,
        spans,
        rows,
        {tag: table.clone() for tag, table in tables.items()},
        worker._native_pp_owner,
    )
