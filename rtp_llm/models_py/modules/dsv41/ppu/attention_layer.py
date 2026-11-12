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
from rtp_llm.models_py.modules.dsv41.ppu.attention import flash_mla_sparse_fwd_ppu
from rtp_llm.models_py.modules.dsv41.ppu.graph_swa_write import try_rope_
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.prefill import (
    _build_reference_workspace,
    _produce_global,
    _quantized_indexer_query,
    _rope_tail,
    build_cp_prefill_workspace,
    build_step_context,
)
from rtp_llm.models_py.modules.dsv41.ppu.layers import (
    PPUAttentionProjections,
    cp8_yoco_small_ops_enabled,
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
_CP8_DENSE_TOPK_BYTES = 1 << 30
_OPTIMIZED_LATE_FOCUS_LAYERS = (16, 20, 20, 20, 21, 20, 22, 20)


def _project_wo_a(output, weight, groups, rank, *, batched, pack_groups=False):
    if (
        batched
        and output.is_cuda
        and output.dtype == weight.dtype == torch.bfloat16
        and output.is_contiguous()
        and weight.is_contiguous()
        and not torch.is_grad_enabled()
    ):
        projected = output.new_empty((output.shape[0], groups, rank))
        grouped_weight = weight.view(groups, rank, output.shape[-1])
        torch.bmm(
            output.transpose(0, 1), grouped_weight.transpose(1, 2),
            out=projected.transpose(0, 1),
        )
        return projected
    grouped_output = output.transpose(0, 1)
    if pack_groups:
        grouped_output = grouped_output.contiguous()
    return torch.stack([
        F.linear(grouped_output[i].contiguous(), weight[i * rank:(i + 1) * rank].contiguous())
        for i in range(groups)
    ], dim=1)


def _store_dense_topk(logits, output, candidates, visible, block, candidate_k):
    if candidates is not None:
        select_candidate_blocks(logits, visible, block, candidate_k, out=candidates)
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

    def _project_indexer(self, x, qr, positions, shared, *, quantized_input=None):
        g = self.geometry
        index_q = (
            self.index_wq(qr) if quantized_input is None
            else self.project_indexer_query(qr, quantized_input=quantized_input)
        ).reshape(-1, g.index_n_heads, g.index_head_dim)
        index_q = _rope_tail(index_q, self.runtime.freqs_for(self.layer_id)[positions])
        projection = shared.get("ced_indexer_projection")
        raw = F.linear(x, self.index_weights) if projection is None else projection(x, self.index_weights)
        weights = raw.float() * (g.index_head_dim * g.index_n_heads) ** -0.5
        return index_q, weights

    def _select(
        self, x, qr, context, cache, shared, *, prefill=False,
        request_slices=None, allow_tile_prep=True, yoco_small_ops=False, indexer_inputs=None,
        quantized_input=None,
    ):
        g, source = self.geometry, self.kv_source_layer_id
        direct_indexer_output = yoco_small_ops and request_slices is not None
        if not self.is_index_source:
            return shared["topk"][self.index_source_layer_id]
        reference_perf = prefill and os.environ.get("RTP_V41_REFERENCE_PERF_DIR")
        ratio = g.ratio(source)
        counts = [n // ratio for n in cache.lengths]
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
                if native_fp4:
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
                        columns = torch.arange(plan.sparse_columns, device=x.device)
                        logits.masked_fill_(
                            columns[None, :] >= plan.end[:, None], -torch.inf
                        )
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
                    direct_positions = dense_tile_prep or (dense_fp4 and yoco_small_ops)
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
                    )
                    if reference_perf and dense_fp4:
                        shared.setdefault("reference_dense_fp4", set()).add(self.layer_id)
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
                                (context.positions[batch_rows] + 1) // ratio if publish else None,
                                block, candidate_k,
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
        yoco_small_ops = use_cp_workspace and cp8_yoco_small_ops_enabled(state)
        grouped_qo_rope = (full_cp8 or yoco_small_ops) and enabled(
            CP8_GROUPED_QO_ROPE_ENV, default=True,
        )
        # Every candidate is selected before a projection branch, collective
        # or cache write. Runtime failures propagate without a serial retry.
        if cache.sharded:
            plan, packed_reason = None, "cp_sharded_logical_workspace"
        else:
            plan, packed_reason = prepare_packed_attention(self, state, context, x)
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
        freqs = self.runtime.freqs_for(self.layer_id)[positions]

        def query_branch(value):
            latent = self.query_latent(value)
            return latent, self.wq_b(latent).reshape(-1, g.n_heads, g.head_dim)

        def kv_branch(value):
            return self.kv_norm(self.wkv(value))

        indexer_quantized_input = None
        if parallel:
            (qr, q), kv = self.runtime.qkv_streams.run(x, query_branch, kv_branch)
        elif state.prefill:
            qr, q, kv, indexer_quantized_input = self.project_query_kv(
                x, reuse_ptpc=True,
                prepare_indexer=bool(self.compress_ratio and self.is_index_source),
            )
            q = q.reshape(-1, g.n_heads, g.head_dim)
        else:
            qr, q = query_branch(x)
            kv = kv_branch(x)
        from rtp_llm.models_py.modules.dsv41.ppu.fused_ops import (
            is_supported as fusion_supported, qk_rope,
        )
        fused = (enabled(FUSION_ENV) and bool(x.shape[0]) and ordinary_eager_reason(state) is None
                 and not torch.cuda.is_current_stream_capturing()
                 and fusion_supported(q, kv, freqs))
        if diagnostic is not None:
            diagnostic_q_before, diagnostic_kv_before = q, kv
        rope_backend = "serial_rope"
        if fused:
            q, kv = qk_rope(q, kv, freqs)
            rope_backend = "triton_fused_qk_rope"
        else:
            # The Q projection owns its result in the CP workspace path.
            rotated_q = (
                try_rope_(q, freqs, preserve_signed_zero=True) if grouped_qo_rope else None
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
        if compact_cp8 and starts is not None and not self.is_kv_source:
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
        cache.stage_swa(
            self.layer_id, full_pos[real], full_req[real], packed, state.query_prefixes,
            retain_rows=plan is None, retain_prefix_tail=swa_starts is None,
            bounded_starts=(
                tuple(swa_starts.values())
                if state.compacted and cp is state.context
                and starts is not None and swa_starts is not None
                else None
            ),
        )
        if self.is_kv_source:
            values = gather(self.global_wkv(x))
            scores = gather(self.global_wgate(x)) if self.global_wgate is not None else None
            prefill_layout = None
            if state.prefill and (cp is None or cp.gather_restore_positions is None):
                prefill_layout = (state.query_prefixes, state.lengths)
            _produce_global(
                ops=self.ops, layer=self._layer_weights(), geometry=g,
                freqs=self.runtime.freqs_for(self.layer_id),
                values=values[real], scores=scores[real] if scores is not None else None,
                positions=full_pos[real], request_ids=full_req[real], cache=cache,
                layer_id=self.layer_id, append_starts=state.append_starts,
                prefill_layout=prefill_layout,
            )
            shared.setdefault("global", {})[self.layer_id] = source_identity(cache, self.layer_id)
        indexer_inputs = None
        if compact_after_producer is not None:
            # Preserve the full-row projections and KV writes. Only retained
            # queries enter selection, attention and the output projections.
            exchange = state.plan.exchange
            index_q, weights = self._project_indexer(
                x, qr, positions, shared, quantized_input=indexer_quantized_input,
            )
            indexer_quantized_input = None
            indexer_inputs = exchange.compact(index_q), exchange.compact(weights)
            del index_q, weights
            q = exchange.compact(q)
            x = exchange.compact(x)
            qr = None
            compact_after_producer()
            cp = state.context
            # L20 reads exact SWA from the full writes above. The bounded SWA
            # start from the compact context is for the subsequent layers.
            positions, requests, cu, valid = state.rows(cp)
            context = build_step_context(positions, requests, cu, valid_mask=valid)
            freqs = self.runtime.freqs_for(self.layer_id)[positions]
            compact_cp8 = True
            shape = x.shape
            route.update(rows=x.shape[0], producer_rows=route["rows"])
        if self.compress_ratio and (not use_cp_workspace or self.is_index_source):
            cache.prepare_source(self.kv_source_layer_id)
        request_slices = None
        if (
            (full_cp8 or compact_cp8)
            and self.is_index_source
            and enabled(CP8_INDEXER_REQUEST_SLICES_ENV)
        ):
            request_slices = _indexer_request_slices(
                cp.chunk_lengths_per_req, x.shape[0], len(cache.lengths),
            )
        selected = self._select(
            x, qr, context, cache, shared, prefill=state.prefill,
            request_slices=request_slices, allow_tile_prep=not compact_cp8,
            yoco_small_ops=yoco_small_ops, indexer_inputs=indexer_inputs,
            quantized_input=indexer_quantized_input,
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
                output, _, _ = flash_mla_sparse_fwd_ppu(
                    q.contiguous(), kv_rows, indices, g.sm_scale(),
                    attn_sink=self.attn_sink, topk_length=lengths,
                )
            route["bf16_kv_workspace"] = kv_rows.dtype == torch.bfloat16
        partition_probe = None
        if diagnostic is not None:
            if plan is not None and diagnostic.get("partition_control"):
                output, partition_probe = _optimized_partition_probe(
                    self, state, context, selected, swa_starts, plan, q, output, diagnostic["arm"],
                )
            diagnostic_output = output
        rotated_output = (
            try_rope_(output, freqs, inverse=True, preserve_signed_zero=True)
            if grouped_qo_rope
            else None
        )
        output = _rope_tail(output, freqs, inverse=True) if rotated_output is None else rotated_output
        output = output.reshape(-1, g.o_groups, g.n_heads * g.head_dim // g.o_groups)
        if hasattr(self.wo_a, "forward_grouped"):
            projected = (
                self.wo_a.forward_grouped(output, prefill=True)
                if state.prefill else self.wo_a.forward_grouped(output)
            )
        else:
            projected = _project_wo_a(
                output, self.wo_a.weight, g.o_groups, g.o_lora_rank,
                batched=full_cp8 and enabled(CP8_WO_A_BMM_ENV),
                pack_groups=compact_cp8,
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
