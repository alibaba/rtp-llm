"""Paged PPU Attention bound to loaded ModelWeights and native per-tag pools."""

import hashlib
import io
import json
import math
import os
from itertools import accumulate
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.backend import V41QuantizedTensor
from rtp_llm.models_py.modules.dsv41.cp import (
    _cp_restore_gathered_full_2d,
    cp_all_gather_full,
    cp_all_gather_full_varlen,
    cp_swa_replay_starts,
)
from rtp_llm.models_py.modules.dsv41.fp8._indexer_score import has_fp8_fp4_mqa_logits
from rtp_llm.models_py.modules.dsv41.fp8 import (
    _v41_sparse_prefill_indexer as sparse_prefill_indexer,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_indexer import (
    PrefillIndexerKeys,
    logits_chunk_rows,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_topk import (
    write_torch_topk_results,
)
from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import (
    _prefill_sparse_plan,
    mask_candidate_logits,
    select_candidate_blocks,
)
from rtp_llm.models_py.modules.dsv41.ppu._quant import _bytes_to_sf, is_ppu
from rtp_llm.models_py.modules.dsv41.ppu import prefill_token_topk as _token_topk
from rtp_llm.models_py.modules.dsv41.ppu.topk import select_tokens
from rtp_llm.models_py.modules.dsv41.ppu.attention import flash_mla_sparse_fwd_ppu
from rtp_llm.models_py.modules.dsv41.ppu.rope import try_rotate_inplace as try_rope_
from rtp_llm.models_py.modules.dsv41.ppu.grouped_projection import (
    batched_wo_a,
    project_wo_a,
)
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import (
    enabled as optimization_enabled,
)
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    LOCAL_PREFILL_SWA_SEGMENTS_ENV,
    SharedNativeCache,
    _LocalSwaStageRanges,
    _source_tensor_key,
)
from rtp_llm.models_py.modules.dsv41.ppu.prefill import (
    _build_reference_workspace,
    _make_local_producer_rotary_proof,
    _produce_global,
    _quantized_indexer_query,
    _rope_tail,
    _valid_local_producer_rotary_proof,
    build_cp_prefill_workspace,
    build_step_context,
)
from rtp_llm.models_py.modules.dsv41.ppu.layers import (
    PPUAttentionProjections,
    cp8_yoco_operator_eligible,
)
from rtp_llm.models_py.modules.dsv41.ppu.packed_attention import (
    DEPENDENCY_ENV,
    FUSION_ENV,
    STREAMS_ENV,
    enabled,
    ordinary_eager_reason,
    prepare_packed_attention,
    run_packed_attention,
)


OPTIMIZED_DIAGNOSTICS_ENV = "RTP_V41_PPU_OPTIMIZED_DIAGNOSTICS"
OPTIMIZED_LATE_FOCUS_ENV = "RTP_V41_PPU_OPTIMIZED_LATE_FOCUS"
PREFILL_DENSE_FP4_SCORE_ENV = "RTP_V41_PPU_PREFILL_DENSE_FP4_SCORE"
PREFILL_SPARSE_FP4_SCORE_ENV = "RTP_V41_PPU_PREFILL_SPARSE_FP4_SCORE"
CP8_WO_A_BMM_ENV = "RTP_V41_PPU_CP8_WO_A_BMM"
CP8_GROUPED_QO_ROPE_ENV = "RTP_V41_PPU_CP8_GROUPED_QO_ROPE"
CP8_INDEXER_REQUEST_SLICES_ENV = "RTP_V41_PPU_CP8_INDEXER_REQUEST_SLICES"
CP8_INDEXER_TILE_PREP_ENV = "RTP_V41_PPU_CP8_INDEXER_TILE_PREP"
LOCAL_PREFILL_WORKSPACE_ENV = "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE"
LOCAL_PREFILL_GROUPED_QO_ROPE_ENV = "RTP_V41_PPU_LOCAL_PREFILL_GROUPED_QO_ROPE"
LOCAL_PREFILL_DENSE_PROJECTIONS_ENV = "RTP_V41_PPU_LOCAL_PREFILL_DENSE_PROJECTIONS"
_CP8_DENSE_TOPK_BYTES = 1 << 30
_OPTIMIZED_LATE_FOCUS_LAYERS = (16, 20, 20, 20, 21, 20, 22, 20)


def _local_prefill_dense_projections_enabled(state, cp, x, geometry, *, diagnostic, packed_plan, parallel):
    """Keep the joined FP8 ABI within the existing local eager Prefill route."""
    return (
        os.environ.get(LOCAL_PREFILL_DENSE_PROJECTIONS_ENV, "0") == "1"
        and getattr(state, "local_prefill_sg_align", False)
        and state.prefill and not state.idle and not torch.is_grad_enabled()
        and not any(getattr(state, flag, False) for flag in (
            "is_graph", "is_graph_commit", "verify", "proposal",
        ))
        and diagnostic is None and packed_plan is None and not parallel
        and (cp is None or (cp.cp_size == 1 and not cp.kv_cache_sharded))
        and x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 2
        and x.shape[1] == 5120 and x.is_contiguous() and not x.requires_grad
        and (geometry.q_lora_rank, geometry.head_dim) == (1280, 512)
        and not torch.cuda.is_current_stream_capturing()
    )


def _local_prefill_token_topk_enabled(state, cp, *, diagnostic=None, packed_plan=None):
    if (
        not enabled(_token_topk.ENV, default=False)
        or not getattr(state, "prefill", False) or getattr(state, "idle", True)
        or diagnostic is not None or packed_plan is not None or torch.is_grad_enabled()
        or any(getattr(state, flag, False) for flag in (
            "is_graph", "is_graph_commit", "verify", "proposal",
        ))
        or getattr(getattr(state, "attn", None), "is_cuda_graph", False)
    ):
        return False
    cache = getattr(state, "cache", None)
    model = getattr(getattr(state, "runtime", None), "model", None)
    return (
        cache is not None and not getattr(cache, "sharded", True)
        and (cp is None or (
            getattr(cp, "cp_size", 0) == 1 and not getattr(cp, "kv_cache_sharded", True)
        ))
        and model is not None and not getattr(model, "_is_speculative", False)
        and not getattr(model, "_is_decode_role", False)
        and getattr(model, "_prefill_cp_size", None) == 1
        and getattr(getattr(model, "parallelism_config", None), "tp_size", None) == 1
        and is_ppu() and not torch.cuda.is_current_stream_capturing()
    )


def _local_prefill_grouped_qo_rope_enabled(state, cp, *, diagnostic=None):
    if (
        not optimization_enabled(LOCAL_PREFILL_GROUPED_QO_ROPE_ENV, default=False)
        or not getattr(state, "prefill", False) or getattr(state, "idle", True)
        or diagnostic is not None or torch.is_grad_enabled()
        or any(getattr(state, flag, False) for flag in (
            "is_graph", "is_graph_commit", "verify", "proposal",
        ))
        or getattr(getattr(state, "attn", None), "is_cuda_graph", False)
    ):
        return False
    cache = getattr(state, "cache", None)
    model = getattr(getattr(state, "runtime", None), "model", None)
    return (
        cache is not None and not getattr(cache, "sharded", True)
        and (cp is None or (
            getattr(cp, "cp_size", 0) == 1 and not getattr(cp, "kv_cache_sharded", True)
        ))
        and model is not None and not getattr(model, "_is_speculative", False)
        and is_ppu() and not torch.cuda.is_current_stream_capturing()
    )


def _local_qo_rope_owned(value, *live_inputs):
    """Allow a fresh projection/native-output view, never a copied alias."""
    return (
        value.is_cuda and value.dtype == torch.bfloat16 and value.ndim == 3
        and value.shape[1] == 64 and value.shape[2] in (128, 512)
        and value.is_contiguous() and not value.requires_grad
        and value.untyped_storage().data_ptr() not in {
            tensor.untyped_storage().data_ptr() for tensor in live_inputs if tensor is not None
        }
    )


def _local_sg_indexer_counts_supported(counts):
    # The SG dense score/amax ABI reads aligned groups of eight columns.
    # Startup requests can have an odd bank width even in this deployment.
    return bool(counts) and all(8 <= count <= 131072 and count % 8 == 0 for count in counts)


def _local_sg_rope_positions(state, context, rows, frequencies):
    """Use only creation-stamped positions within their original host spans."""
    if not getattr(state, "local_prefill_sg_align", False):
        return None
    if not (
        isinstance(frequencies, torch.Tensor) and frequencies.layout == torch.strided
        and frequencies.dtype == torch.complex64 and frequencies.ndim == 2
        and frequencies.shape[1] == 32 and frequencies.is_contiguous()
        and not frequencies.is_neg() and not frequencies.is_conj()
        and not frequencies.requires_grad and frequencies.device == rows[0].device
    ):
        return None
    if state.local_indexer_request_slices(context, rows, rows[0].numel()) is None:
        return None
    prefixes, lengths = state.query_prefixes, state.lengths
    if (len(prefixes) != len(lengths) or not lengths
            or any(type(value) is not int or value < 0 for value in (*prefixes, *lengths))
            or max(prefix + length for prefix, length in zip(prefixes, lengths)) > frequencies.shape[0]):
        return None
    return rows[0]


def _local_swa_stage_ranges(state, cp, layer, full_pos, full_req, real, positions, requests,
                            packed, *, packed_plan, retain_rows, retain_prefix_tail,
                            bounded_starts, diagnostic=None):
    """Bind real request ranges only to unchanged creation-stamped outputs."""
    def eligible():
        cache = getattr(state, "cache", None)
        model = getattr(getattr(state, "runtime", None), "model", None)
        return (
            os.environ.get(LOCAL_PREFILL_SWA_SEGMENTS_ENV, "0") == "1"
            and packed_plan is None and retain_rows
            and getattr(state, "prefill", False) and not getattr(state, "idle", True)
            and diagnostic is None and not torch.is_grad_enabled()
            and not any(getattr(state, flag, False) for flag in (
                "is_graph", "is_graph_commit", "verify", "proposal",
            ))
            and not getattr(getattr(state, "attn", None), "is_cuda_graph", False)
            and model is not None and not getattr(model, "_is_speculative", False)
            and type(cache) is SharedNativeCache and cache._is_prefill and not cache.sharded
            and (cp is None or (cp.cp_size == 1 and not cp.kv_cache_sharded))
        )

    if not eligible() or not isinstance(real, slice) or real != slice(None):
        return None
    prove = getattr(state, "_local_swa_stage_rows", None)
    ranges = prove(cp, full_pos, full_req) if prove is not None else None
    if ranges is None or (cp is None and not retain_prefix_tail):
        return None
    if cp is not None and (retain_prefix_tail or bounded_starts != tuple(row[1] for row in ranges)):
        return None
    for source, view in ((full_pos, positions), (full_req, requests)):
        if (not torch._C._is_alias_of(source, view) or source.data_ptr() != view.data_ptr()
                or source.storage_offset() != view.storage_offset() or source.shape != view.shape
                or source.stride() != view.stride() or source.dtype != view.dtype or source.device != view.device):
            return None
    cache = state.cache
    pool_key, owners = cache._local_swa_pool_key(layer)
    inputs = positions, requests, packed
    return _LocalSwaStageRanges(
        cache, ranges, inputs, tuple(_source_tensor_key(value) for value in inputs),
        tuple(state.query_prefixes), cache.lengths, retain_prefix_tail, bounded_starts,
        cache.pools[layer, "swa_kv"], pool_key, owners,
        lambda: eligible() and prove(cp, full_pos, full_req) == ranges,
    )


def _local_prefill_workspace_layout(state, cp, geometry, layer_id, *, diagnostic=None):
    """Describe local query chunks without changing the executor's CP context."""
    if (
        not enabled(LOCAL_PREFILL_WORKSPACE_ENV, default=False)
        or not state.prefill
        or state.idle
        or state.cache.sharded
        or diagnostic is not None
        or (cp is not None and cp.cp_size != 1)
    ):
        return None
    if geometry.ratio(layer_id) and geometry.kv_source_for(layer_id) is None:
        return None
    if cp is not None:
        return cp
    chunks, prefixes = tuple(state.lengths), tuple(state.query_prefixes)
    key = "local_prefill_workspace_layout"
    layout = state.shared.get(key)
    if (
        layout is None
        or layout.chunk_lengths_per_req != chunks
        or layout.prefix_lengths_host != prefixes
    ):
        layout = SimpleNamespace(chunk_lengths_per_req=chunks, prefix_lengths_host=prefixes)
        state.shared[key] = layout
    return layout


def _project_wo_a(output, weight, groups, rank, *, batched, pack_groups=False):
    if (
        batched
        and output.is_cuda
        and output.dtype == weight.dtype == torch.bfloat16
        and output.is_contiguous()
        and weight.is_contiguous()
        and not torch.is_grad_enabled()
    ):
        return batched_wo_a(output, weight, groups, rank)
    grouped_output = output.transpose(0, 1)
    if pack_groups:
        grouped_output = grouped_output.contiguous()
    return torch.stack([
        F.linear(grouped_output[i].contiguous(), weight[i * rank:(i + 1) * rank].contiguous())
        for i in range(groups)
    ], dim=1)


def _store_dense_topk(logits, output, candidates, visible, block, candidate_k, *, local_token_topk=False):
    if candidates is not None:
        select_candidate_blocks(logits, visible, block, candidate_k, out=candidates)
    if local_token_topk and output.shape[1] == _token_topk.K:
        selected = select_tokens(logits, visible, contract="local", disposable=True)
        if selected is not None:
            output.copy_(selected)
            return
    scores, ids = logits.topk(output.shape[1], dim=-1)
    if not write_torch_topk_results(scores, ids, output):
        output.copy_(torch.where(torch.isfinite(scores), ids, -1).int())


def _indexer_request_slices(lengths, rows, requests):
    if lengths is None or len(lengths) != requests or sum(lengths) != rows:
        return None
    boundaries = tuple(accumulate(lengths, initial=0))
    return tuple(slice(start, stop) for start, stop in zip(boundaries, boundaries[1:]))


def optimized_diagnostics_enabled():
    value = os.environ.get(OPTIMIZED_DIAGNOSTICS_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(OPTIMIZED_DIAGNOSTICS_ENV + " must be explicitly 0 or 1")
    if enabled(OPTIMIZED_LATE_FOCUS_ENV) and value != "1":
        raise ValueError(OPTIMIZED_LATE_FOCUS_ENV + "=1 requires " + OPTIMIZED_DIAGNOSTICS_ENV + "=1")
    return value == "1"


def _optimized_focus_layers(rank):
    optimized_diagnostics_enabled()
    if not enabled(OPTIMIZED_LATE_FOCUS_ENV):
        return (0, 2)
    if type(rank) is not int or not 0 <= rank < len(_OPTIMIZED_LATE_FOCUS_LAYERS):
        raise ValueError("late focus requires a native rank in [0, 8)")
    # Frozen b06 first same-Q shadow failure, never selected from this run.
    return (_OPTIMIZED_LATE_FOCUS_LAYERS[rank],)


def _optimized_cpu(value):
    # Clone after the CPU transfer: never serialize a view's whole pool/storage.
    return value.detach().cpu().contiguous().clone()


def _optimized_compare(left, right):
    left, right = left.detach().cpu().double(), right.detach().cpu().double()
    finite = torch.isfinite(left) & torch.isfinite(right)
    error = torch.where(finite, (left - right).abs(), torch.zeros_like(left))
    ratio = error / (0.005 + 0.005 * right.abs()).nan_to_num(nan=1.0, posinf=1.0)
    bad = (finite & (ratio > 1)) | (~finite & ~(left == right))
    different = left != right
    return {"exact": bool(torch.equal(left, right)), "max_abs": float(error.max()),
            "max_budget_ratio": float(ratio.max()), "mismatches": int(bad.sum()),
            "first_difference": different.nonzero()[:1].tolist(),
            "first_budget_failure": bad.nonzero()[:1].tolist()}


def _optimized_json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _optimized_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_optimized_json_value(item) for item in value]
    return value


def _optimized_publish(collector, name, payload):
    if Path(name).name != name:
        raise ValueError("diagnostic output must be a basename")
    target = collector["directory"] / name
    temporary = target.with_name("." + name + ".tmp")
    with temporary.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    # Atomic publication with exclusive destination creation, no replacement.
    os.link(temporary, target)
    temporary.unlink()


def _optimized_write_json(collector, name, record):
    payload = json.dumps(_optimized_json_value(record), sort_keys=True, indent=2, allow_nan=False).encode()
    if len(payload) > 512 * 1024:
        raise ValueError("optimized diagnostic JSON exceeds 512 KiB")
    _optimized_publish(collector, name, payload)


def _optimized_tensor_fits(used, size):
    return 0 <= size <= 64 * 1024 * 1024 and 0 <= used <= 64 * 1024 * 1024 - size


def _optimized_bank(pool, indices=None, lengths=None):
    record = {"entries": pool.entries, "span": pool.span, "pool_shape": list(pool.tensor.shape),
              "pool_stride": list(pool.tensor.stride()), "pool_pointer": pool.tensor.data_ptr(),
              "table": pool.table.cpu().tolist()}
    if indices is not None:
        counts = lengths.cpu().tolist()
        record.update(indices_shape=list(indices.shape), indices_stride=list(indices.stride()),
                      lengths=counts,
                      live_indices=[indices[i, 0, :count].cpu().tolist() for i, count in enumerate(counts)])
    return record


def _optimized_partition_probe(attn, state, context, selected, swa_starts, plan, q, packed, arm):
    # This fixture is one fresh request with positions 0..31 and no CP.
    # Keep the native candidate's output, workspace and model write path intact.
    workspace, indices, lengths = _build_reference_workspace(
        ops=attn.ops, cache=state.cache, layer_id=attn.layer_id, geometry=attn.geometry,
        context=context, selected_blocks=selected, window=attn.geometry.window,
        swa_starts=swa_starts,
    )
    q = q.contiguous()
    scale, sink = attn.geometry.sm_scale(), attn.attn_sink
    accepted, _, _ = flash_mla_sparse_fwd_ppu(
        q, workspace, indices, scale, attn_sink=sink, topk_length=lengths,
    )
    rows = q.shape[0]
    positions = context.positions.cpu()
    if (rows != 32 or state.query_prefixes != (0,) or swa_starts
            or not torch.equal(positions, torch.arange(rows))
            or bool(context.request_ids.any()) or not bool(plan.valid_mask.all())):
        raise AssertionError("partition control requires one fresh 32-token request")
    swa_lengths = plan.swa_lengths.cpu()
    global_lengths = plan.global_lengths.cpu() if plan.global_pool is not None else torch.zeros_like(swa_lengths)
    refs = indices.cpu()[:, 0]
    swa_slots = plan.swa_indices.cpu()[:, 0]
    global_slots = plan.global_indices.cpu()[:, 0] if plan.global_pool is not None else None
    gcount = workspace.shape[0] - rows
    swa_base = int(plan.swa_pool.table[0, 0]) * plan.swa_pool.entries
    global_base = int(plan.global_pool.table[0, 0]) * plan.global_pool.entries if global_slots is not None else 0
    if plan.swa_pool.span < rows or plan.swa_pool.entries < rows:
        raise AssertionError("partition control requires one SWA page")
    if global_slots is not None and (plan.global_pool.span < rows or plan.global_pool.entries < gcount):
        raise AssertionError("partition control requires one compressed page")
    for row, (ns, ng, total) in enumerate(zip(swa_lengths.tolist(), global_lengths.tolist(), lengths.tolist())):
        global_refs = refs[row, :ng]
        if (ns != row + 1 or total != ns + ng or ng > (row + 1) // max(plan.ratio, 1)
                or (global_refs < 0).any() or (global_refs >= gcount).any()
                or global_refs.unique().numel() != ng
                or not torch.equal(refs[row, ng:total], torch.arange(gcount, gcount + ns))
                or not torch.equal(swa_slots[row, :ns], torch.arange(swa_base, swa_base + ns))
                or (swa_slots[row, ns:] != -1).any()
                or (refs[row, total:] != -1).any()):
            raise AssertionError(f"packed and BF16 visible keys differ at row {row}")
        if global_slots is not None and (
                not torch.equal(global_slots[row, :ng], global_refs + global_base)
                or (global_slots[row, ng:] != -1).any()):
            raise AssertionError(f"packed compressed key order or mask differs at row {row}")

    grouped = accepted
    if global_slots is not None:
        # HS64 rounds primary-bank lengths up to 128 before the extra bank.
        # Change only the index grouping of the original BF16 workspace path.
        separated = torch.full_like(indices, -1)
        for row, (ns, ng) in enumerate(zip(swa_lengths.tolist(), global_lengths.tolist())):
            separated[row, 0, :ns] = indices[row, 0, ng:ng + ns]
            separated[row, 0, 128:128 + ng] = indices[row, 0, :ng]
        separated_lengths = torch.where(plan.global_lengths > 0, 128 + plan.global_lengths, plan.swa_lengths)
        grouped, _, _ = flash_mla_sparse_fwd_ppu(
            q, workspace, separated, scale, attn_sink=sink, topk_length=separated_lengths,
        )

    evidence = {"same_visible_keys": True,
                "packed_vs_grouped_bf16": _optimized_compare(packed, grouped),
                "grouped_vs_accepted_bf16": _optimized_compare(grouped, accepted)}
    if arm in ("100-packed", "110-combo"):
        # Independently form QK, normalized weights (including the sink), and
        # PV in CPU64, with the native call's actual float32 scale.
        keys = workspace[:, 0].cpu().double()[refs[:, :gcount + rows].clamp_min(0).long()]
        logits = torch.bmm(q.cpu().double(), keys.transpose(1, 2))
        logits *= float(torch.tensor(scale, dtype=torch.float32, device="cpu"))
        logits.masked_fill_(refs[:, None, :gcount + rows] < 0, -math.inf)
        logits = torch.cat((logits, sink.cpu().double()[None, :, None].expand(rows, -1, -1)), -1)
        mathematical = torch.bmm(logits.softmax(-1)[..., :-1], keys)
        evidence["packed_vs_cpu64"] = _optimized_compare(packed, mathematical)
        evidence["accepted_vs_cpu64"] = _optimized_compare(accepted, mathematical)
        evidence["grouped_vs_cpu64"] = _optimized_compare(grouped, mathematical)
        for name in ("packed_vs_grouped_bf16", "packed_vs_cpu64", "accepted_vs_cpu64", "grouped_vs_cpu64"):
            failures = evidence[name]["first_budget_failure"]
            if failures:
                point = tuple(failures[0])
                evidence[name]["first_failure_values"] = {
                    "packed": float(packed[point]), "accepted": float(accepted[point]),
                    "grouped": float(grouped[point]), "cpu64": float(mathematical[point]),
                }

    if arm == "200-partition-only" or (arm == "400-allowed-effects" and global_slots is not None):
        # The combined control uses grouped BF16 here and the qualified
        # single-bank packed accumulation at SWA-only layers.
        output = grouped
        output_kind = "grouped_bf16"
    elif ((arm == "300-swa-residual" and global_slots is not None)
            or (arm == "500-dual-residual" and global_slots is None)):
        output = accepted
        output_kind = "accepted"
    elif arm == "500-dual-residual":
        # Counterfactual: transfer only the same-group backend residual to
        # the accepted output in FP32, then enter the model at its BF16 edge.
        output = (accepted.float() + (packed.float() - grouped.float())).to(accepted.dtype)
        output_kind = "accepted_plus_same_group_residual"
    else:
        output = packed
        output_kind = "packed"
    evidence["authoritative_output"] = output_kind
    return output, (workspace, indices, lengths, accepted, evidence)


def _optimized_record_layer(attn, state, collector, *, x, q_before, kv_before, q, kv, freqs,
                            packed, positions, requests, context, selected, plan, swa_starts,
                            attention_output, result, swa_before, global_before, q_at_call, sink_at_call,
                            native_q_shapes, partition_probe=None):
    layer, arm = attn.layer_id, collector["arm"]
    cache, g = state.cache, attn.geometry
    if x.shape[0] != 32 or len(state.query_prefixes) != 1 or any(state.query_prefixes):
        raise ValueError("N5 diagnostics require the original fresh32-token single request")
    stored = cache._copy_rows((layer, "swa_kv"), 0, positions)
    global_live, global_positions, global_key = None, None, None
    if attn.compress_ratio:
        source = attn.kv_source_layer_id
        tag = "global_kv_2" if attn.compress_ratio == 2 else "global_kv_1"
        global_key = source, tag
        global_positions = torch.arange(cache.lengths[0] // attn.compress_ratio, device=x.device) * attn.compress_ratio
        global_live = cache._copy_rows(global_key, 0, global_positions)
    before_writes = len(cache.writes)
    # Read-only shadow over the SAME Q/sink/current native bytes. It never
    # replaces the authoritative packed output and never repeats a producer.
    if partition_probe is None:
        workspace, indices, lengths = _build_reference_workspace(
            ops=attn.ops, cache=cache, layer_id=layer, geometry=g, context=context,
            selected_blocks=selected, window=g.window, swa_starts=swa_starts,
        )
        shadow = None
        if plan is not None:
            shadow, _, _ = flash_mla_sparse_fwd_ppu(
                q.contiguous(), workspace, indices, g.sm_scale(),
                attn_sink=attn.attn_sink, topk_length=lengths,
            )
    else:
        workspace, indices, lengths, shadow, partition_evidence = partition_probe
    shadow_assertion = None
    if shadow is not None:
        try:
            torch.testing.assert_close(attention_output, shadow, rtol=0.005, atol=0.005)
            shadow_assertion = {"status": "PASS"}
        except AssertionError as error:
            shadow_assertion = {"status": "FAIL", "error": str(error)[:8192]}
    values = {"input": _optimized_cpu(x), "q": _optimized_cpu(q), "kv": _optimized_cpu(kv),
              "attention": _optimized_cpu(attention_output), "result": _optimized_cpu(result)}
    if arm == "000-baseline":
        collector["baseline"][layer] = values
    baseline = collector["baseline"][layer]
    comparisons = {name: _optimized_compare(value, baseline[name]) for name, value in values.items()}
    comparisons.update(
        rope_q=_optimized_compare(q, _rope_tail(q_before, freqs)),
        rope_kv=_optimized_compare(kv, _rope_tail(kv_before, freqs)),
        encoded_vs_stored=_optimized_compare(packed, stored),
        encoded_vs_stored_before_attention=_optimized_compare(packed, swa_before),
        candidate_swa_readonly=_optimized_compare(swa_before, stored),
        candidate_q_readonly=_optimized_compare(q_at_call, q),
        candidate_sink_readonly=_optimized_compare(sink_at_call, attn.attn_sink),
        shadow_swa_readonly=_optimized_compare(stored, cache._copy_rows((layer, "swa_kv"), 0, positions)),
    )
    if shadow is not None:
        comparisons["packed_vs_same_q_accepted"] = _optimized_compare(attention_output, shadow)
    if global_live is not None:
        comparisons["candidate_global_readonly"] = _optimized_compare(global_before, global_live)
        comparisons["shadow_global_readonly"] = _optimized_compare(
            global_live, cache._copy_rows(global_key, 0, global_positions))
    for name, comparison in comparisons.items():
        if not comparison["exact"]:
            collector["first_exact_difference"].setdefault(name, layer)
        if comparison["mismatches"]:
            collector["first_budget_failure"].setdefault(name, layer)
    banks = {"swa": _optimized_bank(cache.pools[layer, "swa_kv"],
                                    plan.swa_indices if plan is not None else None,
                                    plan.swa_lengths if plan is not None else None)}
    if attn.compress_ratio:
        bank = cache.pools[global_key]
        banks["global"] = _optimized_bank(bank, plan.global_indices if plan is not None else None,
                                          plan.global_lengths if plan is not None else None)
    record = {
        "kind": "DIAGNOSTIC_ONLY", "rank": collector["rank"], "arm": arm, "layer": layer,
        "comparisons_vs_baseline": comparisons, "shadow_assertion": shadow_assertion,
        "first_exact_difference": dict(collector["first_exact_difference"]),
        "first_budget_failure": dict(collector["first_budget_failure"]),
        "q_shape": list(q.shape), "q_stride": list(q.stride()), "q_dtype": str(q.dtype),
        "adapter_q_shapes": native_q_shapes,
        "native_kernel_branch": "NOT_OBSERVED; no inference from pool stride",
        "banks": banks, "sink": attn.attn_sink.cpu().tolist(), "sink_dtype": str(attn.attn_sink.dtype),
        "scale": g.sm_scale(), "positions": positions.cpu().tolist(), "requests": requests.cpu().tolist(),
        "global_live_positions": global_positions.cpu().tolist() if global_positions is not None else None,
        "reference_lengths": lengths.cpu().tolist(),
        "reference_live_indices": [indices[i, 0, :count].cpu().tolist()
                                   for i, count in enumerate(lengths.cpu().tolist())],
        "packed_dependency": plan.backend_identity if plan else None,
        "producer_writes_unchanged_by_probe": len(cache.writes) == before_writes,
        "authoritative_output": "packed" if plan is not None else "accepted",
    }
    cpu64_outlier = False
    if partition_probe is not None:
        record["partition_control"] = partition_evidence
        record["authoritative_output"] = partition_evidence["authoritative_output"]
        cpu64_outlier = any(
            metrics["mismatches"]
            for name, metrics in partition_evidence.items()
            if name.endswith("_vs_cpu64")
        )
    if (plan is not None and arm in ("100-packed", "110-combo")
            and (cpu64_outlier or layer in _optimized_focus_layers(collector["rank"]))):
        # Only live rows, never plan.swa_cache/global_cache or their large storages.
        payload = {**values, "q_before_rope": _optimized_cpu(q_before),
                   "baseline": {name: _optimized_cpu(value) for name, value in baseline.items()},
                   "kv_before_rope": _optimized_cpu(kv_before), "freqs": _optimized_cpu(freqs),
                   "sink": _optimized_cpu(attn.attn_sink), "swa_encoded": _optimized_cpu(packed),
                   "q_at_native_call": q_at_call, "sink_at_native_call": sink_at_call,
                   "swa_before_attention": _optimized_cpu(swa_before),
                   "global_before_attention": _optimized_cpu(global_before) if global_before is not None else None,
                   "swa_stored": _optimized_cpu(stored), "global_live": _optimized_cpu(global_live) if global_live is not None else None,
                   "decoded_workspace": _optimized_cpu(workspace), "reference_indices": _optimized_cpu(indices),
                   "reference_lengths": _optimized_cpu(lengths), "accepted_shadow": _optimized_cpu(shadow),
                   "positions": _optimized_cpu(positions), "requests": _optimized_cpu(requests),
                   "global_live_positions": _optimized_cpu(global_positions) if global_positions is not None else None,
                   "selected_blocks": _optimized_cpu(selected) if selected is not None else None,
                   "swa_indices": _optimized_cpu(plan.swa_indices), "swa_lengths": _optimized_cpu(plan.swa_lengths),
                   "global_indices": _optimized_cpu(plan.global_indices) if plan.global_indices is not None else None,
                   "global_lengths": _optimized_cpu(plan.global_lengths) if plan.global_lengths is not None else None}
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        raw = buffer.getvalue()
        if not _optimized_tensor_fits(collector["pt_bytes"], len(raw)):
            record["tensor_evidence"] = {"status": "BUDGET_EXCEEDED", "bytes": len(raw)}
            collector["evidence_errors"].append(f"{arm} layer{layer} PT budget exceeded")
        else:
            name = f"{arm}-layer{layer:02d}.pt"
            _optimized_publish(collector, name, raw)
            collector["pt_bytes"] += len(raw)
            record["tensor_evidence"] = {"file": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    _optimized_write_json(collector, f"{arm}-layer{layer:02d}.json", record)
    collector["layers"].append(record)


class _SwaMeta(NamedTuple):
    slot_mapping: object = None
    slot_compaction: object = None


class _Meta(NamedTuple):
    workspace: object = None
    swa_meta: object = _SwaMeta()


class SharedPPUAttention(PPUAttentionProjections):
    supports_flat_prefill = True

    def __init__(self, views, layer_id, ops):
        super().__init__(views, layer_id)
        self.geometry, self.ops = views.geometry, ops
        self.window_size = self.geometry.window
        self.n_heads = self.geometry.n_heads
        self._kv_cache = self._block_tables_by_type = None
        self._prefill_meta_shared = None
        self.runtime = None
        self._shared_attention = None

    def _set_prefill_meta_shared(self, meta):
        self._prefill_meta_shared = meta

    def _build_shared_prefill_meta(self, *args, **kwargs):
        # The common builder owns lifetime/workspace propagation. Actual PPU
        # slot geometry comes from SharedNativeCache and each tag's table.
        return _Meta()

    def _layer_weights(self):
        return SimpleNamespace(
            compressor_norm=getattr(self, "global_norm", None),
            index_wk=getattr(self, "index_wk", None),
            index_k_norm=getattr(self, "index_k_norm", None),
        )

    def _project_indexer(self, x, qr, positions, shared, *, quantized_input=None,
                         weight_input=None, weight_rows=None):
        g = self.geometry
        if weight_input is not None and not (
            isinstance(weight_rows, slice) and weight_rows.step in (None, 1)
            and type(weight_rows.start) is int and type(weight_rows.stop) is int
            and 0 <= weight_rows.start < weight_rows.stop == weight_input.shape[0]
            and weight_rows.stop - weight_rows.start == x.shape[0]
            and weight_input.shape[1:] == x.shape[1:]
            and weight_input.dtype == x.dtype and weight_input.device == x.device
        ):
            raise ValueError("Indexer weight projection requires the original full-row suffix")
        index_q = (
            self.index_wq(qr) if quantized_input is None
            else self.project_indexer_query(qr, quantized_input=quantized_input)
        ).reshape(-1, g.index_n_heads, g.index_head_dim)
        index_q = _rope_tail(index_q, self.runtime.freqs_for(self.layer_id)[positions])
        projection = shared.get("ced_indexer_projection")
        weight_input = x if weight_input is None else weight_input
        raw = F.linear(weight_input, self.index_weights) if projection is None else projection(weight_input, self.index_weights)
        if weight_rows is not None:
            # BF16 GEMM chooses a different accumulation path at M140. Keep
            # its original full-row launch; only the FP8 Indexer-Q shrinks.
            raw = raw[weight_rows]
        weights = raw.float() * (g.index_head_dim * g.index_n_heads) ** -0.5
        return index_q, weights

    def _select(
        self, x, qr, context, cache, shared, *, prefill=False,
        request_slices=None, allow_tile_prep=True, yoco_small_ops=False, indexer_inputs=None,
        quantized_input=None, local_token_topk=False, local_sg_align=False,
    ):
        g, source = self.geometry, self.kv_source_layer_id
        sg_align = bool(
            local_sg_align and prefill and request_slices is not None
            and g.index_topk == 512 and g.index_n_heads == 32 and g.index_head_dim == 128
        )
        if sg_align:
            from rtp_llm.models_py.modules.dsv41.ppu import local_prefill_sg_paged_indexer as sg_paged
        local_token_topk = bool(local_token_topk and prefill and g.index_topk == _token_topk.K)
        direct_indexer_output = yoco_small_ops and request_slices is not None
        if not self.is_index_source:
            return shared["topk"][self.index_source_layer_id]
        reference_perf = prefill and os.environ.get("RTP_V41_REFERENCE_PERF_DIR")
        ratio = g.ratio(source)
        counts = [n // ratio for n in cache.lengths]
        sg_align = sg_align and _local_sg_indexer_counts_supported(counts)
        dense_fp4 = (
            prefill
            and self.layer_id in (2, 8, 14, 20)
            and enabled(PREFILL_DENSE_FP4_SCORE_ENV, default=True)
            and x.is_cuda
            and g.index_n_heads in (32, 64)
            and g.index_head_dim == 128
            and has_fp8_fp4_mqa_logits()
        )
        out = torch.full((x.shape[0], g.index_topk), -1, dtype=torch.int32, device=x.device)
        candidate_source = int(self.v41_config.get("candidate_source_layer_id", -1))
        publish = self.layer_id == candidate_source
        shared.setdefault("candidates", None)
        block = int(self.v41_config.get("candidate_block_size", 0))
        candidate_k = int(self.v41_config.get("candidate_topk_blocks", 0))
        if publish:
            shared["candidates"] = (
                torch.full((x.shape[0], candidate_k), -1, dtype=torch.int32, device=x.device)
                if block > 0 and candidate_k > 0 and max(counts, default=0) > block * candidate_k else None
            )
        candidates = shared["candidates"]
        rows_per_request = request_slices[0].stop if direct_indexer_output and counts else 0
        # Dense FP4 logits are BF16. Keep at most one GiB of input tiles per selection.
        batch_row_limit = (
            min(8192, _CP8_DENSE_TOPK_BYTES // (2 * max(counts[0], 1))) if counts else 0
        )
        if rows_per_request:
            batch_row_limit -= batch_row_limit % rows_per_request
        batch_dense_topk = (
            not sg_align
            and
            direct_indexer_output
            and dense_fp4
            and (publish or candidates is None)
            and 1 < len(counts)
            and g.index_topk < counts[0] <= 131072
            and all(count == counts[0] for count in counts)
            and 0 < rows_per_request <= batch_row_limit
            and rows_per_request <= logits_chunk_rows(counts[0])
            and all(rows.stop - rows.start == rows_per_request for rows in request_slices)
        )
        sparse_enabled = (
            prefill
            and self.layer_id in (24, 28, 32, 36)
            and source == candidate_source == 20
            and candidates is not None
            and (ratio, block, candidate_k) == (1, 8, 2048)
            and (g.index_n_heads, g.index_head_dim, g.index_topk) == (32, 128, 512)
            and x.is_cuda
            and enabled(PREFILL_SPARSE_FP4_SCORE_ENV, default=True)
            and is_ppu()
            and sparse_prefill_indexer._get_deep_gemm() is not None
        )
        tile_prep = (
            prefill
            and allow_tile_prep
            and request_slices is not None
            and (dense_fp4 or sparse_enabled)
            and enabled(CP8_INDEXER_TILE_PREP_ENV)
        )
        if max(counts, default=0) <= g.index_topk:
            ids = torch.arange(g.index_topk, device=x.device, dtype=torch.int32)
            out = torch.where(ids[None] < ((context.positions + 1) // ratio)[:, None], ids, -1)
        else:
            index_q, weights = (
                self._project_indexer(
                    x, qr, context.positions, shared, quantized_input=quantized_input,
                )
                if indexer_inputs is None else indexer_inputs
            )
            weights_bf16 = weights.to(torch.bfloat16) if tile_prep else None
            dense_tiles = []
            dense_start = 0
            for rid, count in enumerate(counts):
                if request_slices is None:
                    rows = torch.where(context.request_ids == rid)[0]
                    row_count = rows.numel()
                else:
                    rows = request_slices[rid]
                    row_count = rows.stop - rows.start
                if not row_count or not count:
                    continue
                sparse_fp4 = sparse_enabled and block * candidate_k < count <= 1 << 20
                native_fp4 = dense_fp4 or sparse_fp4
                paged_plan = None
                if sg_align and sparse_fp4 and rid == 0 and row_count == x.shape[0]:
                    publication = shared.get("sg_candidate_publication")
                    if publication is not None:
                        paged_plan = sg_paged.prepare(
                            cache, candidates, publication.visible, publication=publication,
                        )
                if paged_plan is not None:
                    keys, chunk_rows = None, row_count
                elif native_fp4:
                    if isinstance(cache, SharedNativeCache):
                        payload, scales = cache.read_indexer_fp4(source, rid, global_entries=count)
                    else:
                        packed = cache.read_source(source, rid, global_entries=count)[1]
                        width = g.index_head_dim // 2
                        payload = packed[:, :width].contiguous().view(torch.int8)
                        scales = _bytes_to_sf(packed[:, width:])
                    keys = V41QuantizedTensor(payload, scales, "fp4_ue8m0")
                    chunk_rows = logits_chunk_rows(
                        block * candidate_k if sparse_fp4 else count
                    )
                    if sg_align and dense_fp4:
                        # Match SG's full 4096-row FP32 tile at this width;
                        # retain a one-GiB score budget for other admitted rows.
                        chunk_rows = max(1, min(4096, row_count, (1 << 30) // (4 * count)))
                else:
                    packed = cache.read_source(source, rid, global_entries=count)[1]
                    decoded = x.new_empty((count, g.index_head_dim))
                    self.ops.decode_kv(packed, pool_tag="indexer_kv", descriptor=cache.descriptors(source)["indexer_kv"], out=decoded)
                    keys = V41QuantizedTensor(decoded, None, "bf16")
                    chunk_rows = row_count
                key_positions = (
                    None if native_fp4 else torch.arange(count, device=x.device) * ratio
                )
                for chunk_index, start in enumerate(range(0, row_count, chunk_rows)):
                    stop = min(start + chunk_rows, row_count)
                    chunk = rows[start:stop] if request_slices is None else slice(
                        rows.start + start, rows.start + stop,
                    )
                    q = _quantized_indexer_query(index_q[chunk])
                    positions = context.positions[chunk]
                    score_weights = (
                        weights_bf16 if native_fp4 and weights_bf16 is not None else weights
                    )
                    if sparse_fp4:
                        if paged_plan is not None:
                            logits = sg_paged.score(
                                q.values, q.scales, score_weights[chunk], paged_plan,
                            )
                            selected = select_tokens(
                                logits, paged_plan.valid_lens, contract="sg_sparse", blocks=paged_plan.blocks, out=out[chunk],
                            )
                            if selected is None:
                                raise RuntimeError("SG paged sparse TopK rejected an admitted tile")
                            if reference_perf:
                                shared.setdefault("reference_sparse_fp4", set()).add(self.layer_id)
                            shared.setdefault("sg_topk_calls", []).append((self.layer_id, "sparse", stop - start))
                            shared.setdefault("sg_paged_layers", []).append(self.layer_id)
                            del logits
                            continue
                        visible = (positions + 1) // ratio
                        plan = _prefill_sparse_plan(
                            shared,
                            (rid, chunk_index, stop - start, count, ratio, block),
                            candidates[chunk],
                            visible,
                            count,
                            block,
                            ppu=True,
                        )
                        logits = sparse_prefill_indexer.score(
                            q.values,
                            q.scales,
                            PrefillIndexerKeys(keys.values, keys.scales),
                            score_weights[chunk],
                            plan,
                        )
                        if logits is None:
                            raise RuntimeError("PPU sparse Prefill scorer rejected a supported tile")
                        if reference_perf:
                            shared.setdefault("reference_sparse_fp4", set()).add(self.layer_id)
                        if sg_align:
                            selected = select_tokens(
                                logits, plan.end, contract="sg_sparse", blocks=plan.sparse_indices, out=out[chunk],
                            )
                            if selected is None:
                                raise RuntimeError("SG sparse TopK rejected an admitted Prefill tile")
                            shared.setdefault("sg_topk_calls", []).append((self.layer_id, "sparse", stop - start))
                            del logits
                            continue
                        columns = torch.arange(plan.sparse_columns, device=x.device)
                        logits.masked_fill_(
                            columns[None, :] >= plan.end[:, None], -torch.inf
                        )
                        selected = (
                            select_tokens(logits, plan.end, contract="local", disposable=True)
                            if local_token_topk else None
                        )
                        if selected is None:
                            selected = logits.topk(g.index_topk, dim=-1).indices
                        if not (tile_prep or direct_indexer_output) or sparse_prefill_indexer.remap(
                            selected, plan, logits=logits, out=out[chunk]
                        ) is None:
                            selected = selected.to(torch.int32)
                            out[chunk] = sparse_prefill_indexer.remap(
                                selected, plan, logits=logits
                            )
                        del logits
                        continue
                    dense_tile_prep = tile_prep and dense_fp4
                    direct_positions = dense_tile_prep or (dense_fp4 and (yoco_small_ops or sg_align))
                    sub = None if direct_positions else build_step_context(
                        positions, context.request_ids[chunk], context.request_offsets
                    )
                    logits = self.ops.index_score(
                        q,
                        keys,
                        score_weights[chunk],
                        layer_id=self.layer_id,
                        candidate_blocks=None,
                        key_positions=key_positions,
                        context=sub,
                        ratio=ratio,
                        dense_fp4=dense_fp4,
                        positions=positions if direct_positions else None,
                        **({"sg_align": True} if sg_align and dense_fp4 else {}),
                    )
                    if reference_perf and dense_fp4:
                        shared.setdefault("reference_dense_fp4", set()).add(self.layer_id)
                    if sg_align and dense_fp4:
                        visible = ((positions + 1) // ratio).clamp_max(count).to(torch.int32)
                        result = select_tokens(
                            logits, visible, contract="sg_dense", out=out[chunk],
                            block_topk=candidate_k if publish and candidates is not None else g.index_topk,
                            blocks_out=candidates[chunk] if publish and candidates is not None else None,
                        )
                        if result is None:
                            raise RuntimeError(
                                f"SG dense TopK rejected an admitted Prefill tile: layer={self.layer_id}, "
                                f"shape={tuple(logits.shape)}, stride={logits.stride()}, dtype={logits.dtype}"
                            )
                        if publish and candidates is not None and start == 0 and stop == x.shape[0]:
                            publication = sg_paged.make_publication(cache, candidates, visible, source=source)
                            if publication is not None:
                                shared["sg_candidate_publication"] = publication
                        shared.setdefault("sg_topk_calls", []).append((self.layer_id, "dense", stop - start))
                        del logits
                        continue
                    if batch_dense_topk:
                        dense_tiles.append(logits)
                        del logits
                        # Contiguous equal-width tiles share a single selection.
                        # Flush at each tile boundary, keeping the temporary slab bounded.
                        if (
                            chunk.stop - dense_start == batch_row_limit
                            or chunk.stop == x.shape[0]
                        ):
                            batch_rows = slice(dense_start, chunk.stop)
                            batch_logits = (
                                dense_tiles[0] if len(dense_tiles) == 1 else torch.cat(dense_tiles)
                            )
                            dense_tiles.clear()
                            _store_dense_topk(
                                batch_logits, out[batch_rows],
                                candidates[batch_rows] if publish else None,
                                (context.positions[batch_rows] + 1) // ratio if publish or local_token_topk else None,
                                block, candidate_k,
                                **({"local_token_topk": True} if local_token_topk else {}),
                            )
                            del batch_logits
                            dense_start = chunk.stop
                        continue
                    if candidates is not None:
                        if publish:
                            visible = (positions + 1) // ratio
                            if direct_positions and request_slices is not None:
                                select_candidate_blocks(
                                    logits, visible, block, candidate_k, out=candidates[chunk]
                                )
                            else:
                                chosen = select_candidate_blocks(logits, visible, block, candidate_k)
                                candidates[chunk, :chosen.shape[-1]] = chosen
                        elif self.layer_id > candidate_source:
                            mask_candidate_logits(logits, candidates[chunk], block)
                    selected = (
                        select_tokens(logits, (positions + 1) // ratio, contract="local", disposable=True)
                        if local_token_topk else None
                    )
                    if selected is not None:
                        # Advanced row indexing is a copy; assign into the parent.
                        out[chunk] = selected
                    else:
                        scores, chosen = logits.topk(min(g.index_topk, count), dim=-1)
                        if not (
                            direct_positions and request_slices is not None
                            and write_torch_topk_results(scores, chosen, out[chunk, :chosen.shape[-1]])
                        ):
                            out[chunk, :chosen.shape[-1]] = torch.where(torch.isfinite(scores), chosen, -1).int()
                    del logits
        shared.setdefault("topk", {})[self.layer_id] = out
        return out

    def _body(self, x, *, compact_after_producer=None):
        state = self.runtime.current
        if state is None:
            raise RuntimeError("shared Attention has no per-forward resources")
        g, cache, shared = self.geometry, state.cache, state.shared
        shape = x.shape
        x = x.reshape(-1, g.hidden_size)
        if state.idle:
            return torch.zeros_like(x).reshape(shape)
        cp = self._cp_ctx if state.prefill else None
        positions, requests, cu, valid = state.rows(cp)
        context = build_step_context(positions, requests, cu, valid_mask=valid)
        diagnostic = None
        if optimized_diagnostics_enabled():
            diagnostic = getattr(self.runtime, "n5_diagnostics", None)
            if (not isinstance(diagnostic, dict) or diagnostic.get("schema") != "n5-native-b05"
                    or not state.prefill or state.idle):
                raise ValueError("optimized diagnostics require the explicit native Prefill fixture collector")
        use_cp_workspace = cache.sharded and cp is not None and cp.cp_size > 1 and diagnostic is None
        full_cp8 = (
            use_cp_workspace and state.prefill and cp.cp_size == 8
            and cp.gather_restore_positions is None
        )
        compact_cp8 = use_cp_workspace and state.use_compact_cp8_ops(cp)
        yoco_small_ops = use_cp_workspace and cp8_yoco_operator_eligible(state)
        grouped_qo_rope = (full_cp8 or yoco_small_ops) and optimization_enabled(
            CP8_GROUPED_QO_ROPE_ENV, default=True,
        )
        # Every candidate is selected before a projection branch, collective
        # or cache write. Runtime failures propagate without a serial retry.
        if cache.sharded:
            plan, packed_reason = None, "cp_sharded_logical_workspace"
        else:
            plan, packed_reason = prepare_packed_attention(self, state, context, x)
        local_grouped_qo_rope = plan is None and _local_prefill_grouped_qo_rope_enabled(
            state, cp, diagnostic=diagnostic,
        )
        if (
            not state.prefill
            and plan is None
            and packed_reason != "disabled"
            and ordinary_eager_reason(state) is None
        ):
            message = f"PPU eager Decode requires packed attention ({packed_reason})."
            if packed_reason == "packed_native_dependency_unqualified":
                message += f" Set {DEPENDENCY_ENV} to the qualified FlashMLA receipt."
            raise RuntimeError(message)
        from rtp_llm.models_py.modules.dsv41.ppu.projection_streams import (
            QKVStreams, is_supported as streams_supported,
        )
        parallel = enabled(STREAMS_ENV) and streams_supported(state, x)
        if parallel and self.runtime.qkv_streams is None:
            self.runtime.qkv_streams = QKVStreams(x.device)
        frequency_table = self.runtime.freqs_for(self.layer_id)
        sg_rope_positions = (
            _local_sg_rope_positions(state, cp, (positions, requests, cu, valid), frequency_table)
            if diagnostic is None and plan is None else None
        )
        freqs = frequency_table[positions] if sg_rope_positions is None else None
        dense_projections = _local_prefill_dense_projections_enabled(
            state, cp, x, g, diagnostic=diagnostic, packed_plan=plan, parallel=parallel,
        )
        query_rows = (
            state.local_l20_query_slice(x.shape[0])
            if dense_projections and self.layer_id == 20 and compact_after_producer is not None
            and self.is_kv_source and self.is_index_source and sg_rope_positions is not None
            else None
        )
        q_rope_positions = sg_rope_positions[query_rows] if query_rows is not None else sg_rope_positions

        def query_branch(value):
            latent = self.query_latent(value)
            return latent, self.wq_b(latent).reshape(-1, g.n_heads, g.head_dim)

        def kv_branch(value):
            return self.kv_norm(self.wkv(value))

        indexer_quantized_input = None
        projection_receipt = None
        if parallel:
            (qr, q), kv = self.runtime.qkv_streams.run(x, query_branch, kv_branch)
        elif state.prefill:
            qr, q, kv, indexer_quantized_input = self.project_query_kv(
                x, reuse_ptpc=True,
                prepare_indexer=bool(self.compress_ratio and self.is_index_source),
                **({"fused_qkv": True, "query_rows": query_rows} if dense_projections else {}),
            )
            projection_receipt = {
                "layer": self.layer_id, "input_rows": x.shape[0],
                "query_rows": q.shape[0], "kv_rows": kv.shape[0],
                "joined": bool(dense_projections and getattr(self, "_last_local_prefill_qkv_joined", False)),
            }
            shared.setdefault("dense_projection_calls", []).append(projection_receipt)
            q = q.reshape(-1, g.n_heads, g.head_dim)
        else:
            qr, q = query_branch(x)
            kv = kv_branch(x)
        from rtp_llm.models_py.modules.dsv41.ppu.rope import (
            joint_supported as fusion_supported, rotate_qk as qk_rope,
        )
        fused = (sg_rope_positions is None and enabled(FUSION_ENV) and bool(x.shape[0]) and ordinary_eager_reason(state) is None
                 and not torch.cuda.is_current_stream_capturing()
                 and fusion_supported(q, kv, freqs))
        if diagnostic is not None:
            diagnostic_q_before, diagnostic_kv_before = q, kv
        rope_backend = "serial_rope"
        if fused:
            q, kv = qk_rope(q, kv, freqs)
            rope_backend = "triton_fused_qk_rope"
        elif sg_rope_positions is not None:
            from rtp_llm.models_py.modules.dsv41.ppu.rope import try_rotate_fresh as try_rope_tail

            rotated_q = (
                try_rope_(q, frequency_table, preserve_signed_zero=True, positions=q_rope_positions,
                          live_inputs=(x, qr, kv))
                if _local_qo_rope_owned(q, x, qr, kv, frequency_table, q_rope_positions) else None
            )
            rotated_kv = try_rope_tail(kv, frequency_table, False, positions=sg_rope_positions)
            if rotated_q is not None and rotated_kv is not None:
                q, kv = rotated_q, rotated_kv
                rope_backend = "local_sg_position_lookup"
                shared.setdefault("sg_rope_calls", []).extend(((self.layer_id, "q"), (self.layer_id, "kv")))
            else:
                if freqs is None:
                    freqs = frequency_table[positions]
                q_freqs = freqs[query_rows] if query_rows is not None else freqs
                q = rotated_q if rotated_q is not None else _rope_tail(q, q_freqs)
                kv = rotated_kv if rotated_kv is not None else _rope_tail(kv, freqs)
        else:
            # Local Q must own storage disjoint from every live projection input.
            rotated_q = (
                try_rope_(q, freqs, preserve_signed_zero=True, live_inputs=(x, qr, kv))
                if grouped_qo_rope or (
                    local_grouped_qo_rope and _local_qo_rope_owned(q, x, qr, kv, freqs)
                ) else None
            )
            if rotated_q is not None:
                q = rotated_q
                rope_backend = "triton_grouped_q_rope"
            else:
                q = _rope_tail(q, freqs)
            kv = _rope_tail(kv, freqs)
        route = {
            "layer": self.layer_id, "phase": "prefill" if state.prefill else "decode",
            "packed": packed_reason, "attention_backend": "flash_mla_sparse_fwd",
            "qk_rope_backend": rope_backend,
            "projection_streams": "q_current_kv_side_joined" if parallel else "serial",
            "rows": x.shape[0],
        }
        state.shared.setdefault("n5_routes", []).append(route)

        def gather(value):
            if cp is not None:
                if cp.cp_size == 1:
                    return _cp_restore_gathered_full_2d(value, cp)
                shared.setdefault("current_row_gathers", []).append({
                    "layer": self.layer_id, "local_rows": value.shape[0],
                    "width": value.shape[1], "cp_size": cp.cp_size,
                })
                receive = getattr(cache, "gather_receive", None)
                return cp_all_gather_full(value.contiguous(), cp, receive=receive)
            return value

        starts = cp_swa_replay_starts(cp)
        swa_starts = (
            {rid: state.query_prefixes[rid] + start for rid, start in enumerate(starts)}
            if starts is not None and self.layer_id >= 21 else None
        )
        if state.prefill and self.runtime.policy.mode == "compact" and self.layer_id >= 21 and swa_starts is None:
            # Bounded decoder SWA pools contain this request's fresh rows only.
            swa_starts = dict(enumerate(state.query_prefixes))

        pack_local_swa = (
            full_cp8 and yoco_small_ops
            and os.environ.get("RTP_V41_PPU_CP8_YOCO_PACK_LOCAL_SWA", "0") == "1"
        )
        swa_rows = self.ops.encode_kv(kv, pool_tag="swa_kv") if pack_local_swa else kv
        local_compact_rows = (
            diagnostic is None and not self.is_kv_source
            and getattr(state, "use_local_compact_rows", lambda context: False)(cp)
        )
        if local_compact_rows:
            # CP1 is local. The varlen helper's TP group may resolve to the
            # DP world, so reuse only its replay-only restore operation.
            full_kv = swa_rows.index_select(0, cp.unpad_restore)
            full_pos, full_req, real = state.full_rows(cp, selected_only=True)
            route["swa_rows_backend"] = "local_compact_selected"
        elif compact_cp8 and starts is not None and not self.is_kv_source:
            # The CUDA bounded path gathers in retained request order.
            full_kv = cp_all_gather_full_varlen(
                swa_rows, cp, replay_only=True, profile_name="dsv41.ppu.prefill.swa_replay",
            )
            full_pos, full_req, real = state.full_rows(cp, selected_only=True)
        else:
            full_kv = gather(swa_rows)
            full_pos, full_req, real = state.full_rows(cp)
        packed = full_kv[real]
        if not pack_local_swa:
            packed = self.ops.encode_kv(packed, pool_tag="swa_kv")
        stage_positions, stage_requests = full_pos[real], full_req[real]
        stage_options = dict(
            retain_rows=plan is None, retain_prefix_tail=swa_starts is None,
            bounded_starts=(tuple(swa_starts.values()) if state.compacted and cp is state.context
                            and starts is not None and swa_starts is not None else None),
        )
        local_ranges = _local_swa_stage_ranges(
            state, cp, self.layer_id, full_pos, full_req, real, stage_positions, stage_requests, packed,
            packed_plan=plan, diagnostic=diagnostic, **stage_options,
        )
        if local_ranges is not None:
            stage_options["local_ranges"] = local_ranges
        local_rotary_proof = (
            _make_local_producer_rotary_proof(
                local_ranges, cache=cache, layer_id=self.layer_id,
                prefixes=state.query_prefixes, lengths=state.lengths,
            ) if self.is_kv_source and local_ranges is not None else None
        )
        cache.stage_swa(self.layer_id, stage_positions, stage_requests, packed, state.query_prefixes, **stage_options)
        if self.is_kv_source:
            values = gather(self.global_wkv(x))
            scores = gather(self.global_wgate(x)) if self.global_wgate is not None else None
            prefill_layout = None
            if state.prefill and (cp is None or cp.gather_restore_positions is None):
                prefill_layout = (state.query_prefixes, state.lengths)
            source_options = {}
            if local_rotary_proof is not None and _valid_local_producer_rotary_proof(
                local_rotary_proof, cache=cache, layer_id=self.layer_id,
                prefill_layout=prefill_layout, positions=stage_positions,
                request_ids=stage_requests,
            ):
                producer_positions, producer_requests = stage_positions, stage_requests
                source_options["local_rotary_proof"] = local_rotary_proof
            else:
                producer_positions, producer_requests = full_pos[real], full_req[real]
            _produce_global(
                ops=self.ops, layer=self._layer_weights(), geometry=g,
                freqs=self.runtime.freqs_for(self.layer_id),
                values=values[real], scores=scores[real] if scores is not None else None,
                positions=producer_positions, request_ids=producer_requests, cache=cache,
                layer_id=self.layer_id, append_starts=state.append_starts,
                prefill_layout=prefill_layout, **source_options,
            )
            shared.setdefault("global", {})[self.layer_id] = source_identity(cache, self.layer_id)
        indexer_inputs = None
        if compact_after_producer is not None:
            # All full-row SWA/Global/Indexer/CSA writes above complete before
            # the existing state callback. Query projections may already be
            # in the plan's proven contiguous retained-row order.
            exchange = state.plan.exchange
            if query_rows is None:
                index_q, weights = self._project_indexer(
                    x, qr, positions, shared, quantized_input=indexer_quantized_input,
                )
                indexer_inputs = exchange.compact(index_q), exchange.compact(weights)
                q = exchange.compact(q)
                x = exchange.compact(x)
            else:
                if state.local_l20_query_slice(x.shape[0]) != query_rows:
                    raise RuntimeError("L20 query transport changed after projection admission")
                producer_x, x = x, x[query_rows]
                index_q, weights = self._project_indexer(
                    x, qr, positions[query_rows], shared, quantized_input=indexer_quantized_input,
                    weight_input=producer_x, weight_rows=query_rows,
                )
                indexer_inputs = index_q, weights
                del producer_x
            if projection_receipt is not None:
                projection_receipt["indexer_rows"] = index_q.shape[0]
            indexer_quantized_input = None
            del index_q, weights
            qr = None
            compact_after_producer()
            cp = state.context
            # L20 reads exact SWA from the full writes above. The bounded SWA
            # start from the compact context is for the subsequent layers.
            positions, requests, cu, valid = state.rows(cp)
            context = build_step_context(positions, requests, cu, valid_mask=valid)
            frequency_table = self.runtime.freqs_for(self.layer_id)
            sg_rope_positions = (
                _local_sg_rope_positions(state, cp, (positions, requests, cu, valid), frequency_table)
                if diagnostic is None and plan is None else None
            )
            freqs = frequency_table[positions] if sg_rope_positions is None else None
            compact_cp8 = True
            shape = x.shape
            route.update(rows=x.shape[0], producer_rows=route["rows"])
        if self.compress_ratio and (not use_cp_workspace or self.is_index_source):
            cache.prepare_source(self.kv_source_layer_id)
        request_slices = None
        local_sg_align = bool(
            getattr(state, "local_prefill_sg_align", False)
            and diagnostic is None and plan is None and self.is_index_source
        )
        if local_sg_align:
            request_slices = state.local_indexer_request_slices(
                cp, (positions, requests, cu, valid), x.shape[0],
            )
            local_sg_align = request_slices is not None
        if (
            request_slices is None
            and
            (full_cp8 or compact_cp8)
            and self.is_index_source
            and enabled(CP8_INDEXER_REQUEST_SLICES_ENV)
        ):
            request_slices = _indexer_request_slices(
                cp.chunk_lengths_per_req, x.shape[0], len(cache.lengths),
            )
        selected = self._select(
            x, qr, context, cache, shared, prefill=state.prefill,
            request_slices=request_slices, allow_tile_prep=not (compact_cp8 or local_sg_align),
            yoco_small_ops=yoco_small_ops, indexer_inputs=indexer_inputs,
            quantized_input=indexer_quantized_input,
            **({"local_sg_align": True} if local_sg_align else {}),
            **({"local_token_topk": True} if (
                self.is_index_source
                and _local_prefill_token_topk_enabled(state, cp, diagnostic=diagnostic, packed_plan=plan)
                and g.index_topk == _token_topk.K
            ) else {}),
        ) if self.compress_ratio else None
        indexer_quantized_input = None
        if diagnostic is not None:
            diagnostic_swa_before = cache._copy_rows((self.layer_id, "swa_kv"), 0, full_pos[real])
            diagnostic_global_before = None
            if self.compress_ratio:
                diagnostic_tag = "global_kv_2" if self.compress_ratio == 2 else "global_kv_1"
                diagnostic_global_positions = torch.arange(
                    cache.lengths[0] // self.compress_ratio, device=x.device) * self.compress_ratio
                diagnostic_global_before = cache._copy_rows(
                    (self.kv_source_layer_id, diagnostic_tag), 0, diagnostic_global_positions)
            diagnostic_q_at_call = _optimized_cpu(q)
            diagnostic_sink_at_call = _optimized_cpu(self.attn_sink)
            native_q_shapes = []
        if plan is not None:
            plan.bind_selection(selected)
            if diagnostic is not None:
                native = plan.native

                def record_native_call(**kwargs):
                    native_q_shapes.append(list(kwargs["q"].shape))
                    return native(**kwargs)

                plan.native = record_native_call
            output = run_packed_attention(
                q, plan, prefill=state.prefill, softmax_scale=g.sm_scale(),
                attn_sink=self.attn_sink,
            )
            route.update(
                attention_backend="flash_mla_with_kvcache_packed528_288",
                packed_dependency=plan.backend_identity,
                bf16_kv_workspace=False,
                packed_swa_snapshot_retained=any(key[0] == self.layer_id for key in cache._fresh_swa),
                swa_native_alias=plan.swa_cache.data_ptr() == plan.swa_pool.tensor.data_ptr(),
                swa_stride=list(plan.swa_cache.stride()),
                global_native_alias=(plan.global_cache.data_ptr() == plan.global_pool.tensor.data_ptr())
                if plan.global_pool is not None else None,
            )
        else:
            if use_cp_workspace:
                # Q8KV8 is an aggressive optimization that may affect model accuracy.
                # Its numerical accuracy has not been validated. Keep it opt-in.
                q8_attention = (
                    full_cp8 and not compact_cp8 and yoco_small_ops
                    and enabled("RTP_V41_PPU_CP8_YOCO_Q8_ATTENTION", default=False)
                )
                kv_rows, indices, lengths = build_cp_prefill_workspace(
                    ops=self.ops, cache=cache, layer_id=self.layer_id, geometry=g, context=context,
                    selected_blocks=selected, window=g.window, swa_starts=swa_starts, cp=cp, shared=shared,
                    q8=q8_attention,
                )
            elif (local_layout := _local_prefill_workspace_layout(
                state, cp, g, self.layer_id, diagnostic=diagnostic,
            )) is not None:
                q8_attention = False
                kv_rows, indices, lengths = build_cp_prefill_workspace(
                    ops=self.ops, cache=cache, layer_id=self.layer_id, geometry=g, context=context,
                    selected_blocks=selected, window=g.window, swa_starts=swa_starts,
                    cp=local_layout, shared=shared, q8=False,
                )
                route["workspace_backend"] = "local_bf16_reuse"
            else:
                q8_attention = False
                kv_rows, indices, lengths = _build_reference_workspace(
                    ops=self.ops, cache=cache, layer_id=self.layer_id, geometry=g, context=context,
                    selected_blocks=selected, window=g.window, swa_starts=swa_starts,
                )
            if q8_attention:
                from rtp_llm.models_py.modules.dsv41.ppu.packed_attention import load_packed_backend

                native, _ = load_packed_backend()
                output, _, _ = native.flash_mla_sparse_quantized_fwd(
                    q, kv_rows, indices, g.sm_scale(), attn_sink=self.attn_sink,
                    topk_length=lengths, compute_aux=False, trusted_full_topk=True,
                )
                route["attention_backend"] = "flash_mla_sparse_quantized_fwd"
            else:
                native_q = q.contiguous()
                provider = getattr(state, "local_attention_provider", None)
                provider_result = (
                    provider.try_run(state, self.layer_id, g, native_q, kv_rows, indices, lengths, self.attn_sink)
                    if provider is not None and diagnostic is None and self.layer_id < 20
                    and route.get("workspace_backend") == "local_bf16_reuse"
                    else None
                )
                if provider_result is None:
                    output, _, _ = flash_mla_sparse_fwd_ppu(
                        native_q, kv_rows, indices, g.sm_scale(),
                        attn_sink=self.attn_sink, topk_length=lengths,
                    )
                else:
                    output, _, _ = provider_result
                    route["attention_backend"] = "local_bf16_sparse_prefill_provider"
            route["bf16_kv_workspace"] = kv_rows.dtype == torch.bfloat16
        partition_probe = None
        if diagnostic is not None:
            if plan is not None and diagnostic.get("partition_control"):
                output, partition_probe = _optimized_partition_probe(
                    self, state, context, selected, swa_starts, plan, q, output, diagnostic["arm"],
                )
            diagnostic_output = output
        if sg_rope_positions is not None:
            rotated_output = (
                try_rope_(output, frequency_table, inverse=True, preserve_signed_zero=True,
                          positions=sg_rope_positions, live_inputs=(x, qr, q, kv, kv_rows))
                if _local_qo_rope_owned(output, x, qr, q, kv, kv_rows, frequency_table, sg_rope_positions)
                else None
            )
            if rotated_output is not None:
                shared.setdefault("sg_rope_calls", []).append((self.layer_id, "o"))
            elif freqs is None:
                freqs = frequency_table[positions]
        else:
            rotated_output = (
                try_rope_(output, freqs, inverse=True, preserve_signed_zero=True,
                          live_inputs=(x, qr, q, kv, kv_rows))
                if grouped_qo_rope or (
                    local_grouped_qo_rope and _local_qo_rope_owned(output, x, qr, q, kv, kv_rows, freqs)
                )
                else None
            )
        output = _rope_tail(output, freqs, inverse=True) if rotated_output is None else rotated_output
        output = output.reshape(-1, g.o_groups, g.n_heads * g.head_dim // g.o_groups)
        def bf16_projection(output, weight):
            return _project_wo_a(
                output, weight, g.o_groups, g.o_lora_rank,
                batched=full_cp8 and optimization_enabled(CP8_WO_A_BMM_ENV),
                pack_groups=compact_cp8,
            )

        projected = project_wo_a(
            output, self.wo_a, prefill=state.prefill,
            bf16_projection=bf16_projection,
        )
        result = self.wo_b(projected.flatten(1))
        if diagnostic is not None:
            _optimized_record_layer(
                self, state, diagnostic, x=x, q_before=diagnostic_q_before,
                kv_before=diagnostic_kv_before, q=q, kv=kv, freqs=freqs,
                packed=packed, positions=full_pos[real], requests=full_req[real],
                context=context, selected=selected, plan=plan, swa_starts=swa_starts,
                attention_output=diagnostic_output, result=result,
                swa_before=diagnostic_swa_before, global_before=diagnostic_global_before,
                q_at_call=diagnostic_q_at_call, sink_at_call=diagnostic_sink_at_call,
                native_q_shapes=native_q_shapes,
                partition_probe=partition_probe,
            )
        cache.release_swa(self.layer_id)
        return result.reshape(shape)

    def forward(self, x, *args, compact_after_producer=None, **kwargs):
        return self._body(x, compact_after_producer=compact_after_producer)

    def forward_decode(self, x, attn_metadata, **kwargs):
        if getattr(self.runtime.current, "is_graph", False):
            from rtp_llm.models_py.modules.dsv41.ppu.graph_attention_layer import forward_graph_attention

            return forward_graph_attention(self, x, self.runtime.current)
        return self._body(x)


def source_identity(cache, source):
    """Shared producer identity, never a duplicated consumer cache."""
    return tuple(cache.pools[source, tag].tensor for tag in ("global_kv_2" if cache.geometry.ratio(source) == 2 else "global_kv_1", "indexer_kv"))
