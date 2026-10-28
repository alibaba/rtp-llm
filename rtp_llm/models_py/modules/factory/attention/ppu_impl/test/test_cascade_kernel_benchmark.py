"""Manual PPU Cascade accuracy/latency investigation, run through Bazel only.

Fixtures contain post-RoPE Q and already-written current KV. sequence_lengths
are historical lengths: attention reads L + 1 tokens. No model or RoPE is timed.
The legacy output tolerance is a regression gate, not an accuracy acceptance.
"""

import argparse
import contextlib
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import time
import traceback
import unittest
from pathlib import Path

import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.modules.factory.attention.cuda_impl.beam_search_cascade import (
    BeamSearchCascadeAttention,
    merge_state_in_place,
)

HEADS, KV_HEADS, DIM, PAGE = 16, 8, 128, 64
LEGACY_ATOL, LEGACY_RTOL = 0.006, 0.03
OPTIONS = None


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def comma_ints(value):
    return [int(item) for item in value.split(",") if item.strip()]


def parse_options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--histories", default=os.environ.get("PPU_CASCADE_BENCH_HISTORIES", "")
    )
    parser.add_argument("--beams", type=int, default=500)
    parser.add_argument("--bucket", type=int, default=0, help="0 means exact beams")
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--max-requests", type=int, default=1)
    parser.add_argument("--fixture", action="append", default=[])
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--graph-repeat", type=int, default=32)
    parser.add_argument("--eager-repeat", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reference-rows", type=int, default=4)
    parser.add_argument("--reference-heads", default="0,1,7,15")
    parser.add_argument("--diagnose-stages", action="store_true")
    parser.add_argument("--diagnostic-timeline", action="store_true")
    parser.add_argument("--diagnose-online-proxy", action="store_true")
    parser.add_argument("--diagnostic-lse-worst-pairs", type=int, default=8)
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument(
        "--candidate",
        default=os.environ.get("PPU_CASCADE_BENCH_CANDIDATE", ""),
        help="module:factory or /absolute/file.py:factory; test-only extension",
    )
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output-name", default="cascade_kernel_benchmark.json")
    args, remaining = parser.parse_known_args()
    for name in (
        "beams",
        "groups",
        "max_requests",
        "trials",
        "graph_repeat",
        "eager_repeat",
        "warmup",
        "reference_rows",
        "workspace_mib",
        "diagnostic_lse_worst_pairs",
    ):
        require(getattr(args, name) > 0, f"{name} must be positive")
    require(args.bucket == 0 or args.bucket >= args.beams, "bucket below beams")
    require(args.groups <= args.beams, "groups exceed beams")
    require(
        not args.diagnostic_timeline or args.diagnose_stages,
        "--diagnostic-timeline requires --diagnose-stages",
    )
    require(
        not args.diagnose_online_proxy or args.diagnostic_timeline,
        "--diagnose-online-proxy requires --diagnostic-timeline and --diagnose-stages",
    )
    args.reference_heads = comma_ints(args.reference_heads)
    require(
        bool(args.reference_heads)
        and all(0 <= head < HEADS for head in args.reference_heads),
        "invalid reference heads",
    )
    require(
        Path(args.output_name).name == args.output_name,
        "output-name must be a basename",
    )
    return args, remaining


def metric(actual, reference, per_row=True):
    """CPU-only summaries; ratios are not mismatch fractions or pass gates."""
    require(actual.shape == reference.shape, "metric shape mismatch")
    a, b = actual.double(), reference.double()
    require(bool(torch.isfinite(a).all() & torch.isfinite(b).all()), "nonfinite result")
    difference = a - b
    absolute = difference.abs()
    rmse = float(difference.square().mean().sqrt())
    scale = float(b.square().mean().sqrt())
    result = {
        "shape": list(a.shape),
        "elements": a.numel(),
        "numeric_mismatch_count": int((a != b).sum()),
        "max_abs": float(absolute.max()),
        "rmse": rmse,
        "normalized_rmse": rmse / scale if scale else None,
        "absolute_error_quantiles": dict(
            zip(
                ("p50", "p90", "p99", "p999", "max"),
                absolute.flatten()
                .quantile(
                    torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0], dtype=torch.float64)
                )
                .tolist(),
            )
        ),
    }
    edges = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.006, 0.01, 0.1, 1.0]
    result["absolute_error_histogram"] = {
        "zero": int((absolute == 0).sum()),
        "intervals_open_left_closed_right": edges,
        "counts": [
            int(((absolute > lo) & (absolute <= hi)).sum())
            for lo, hi in zip(edges, edges[1:])
        ],
        "above_last": int((absolute > edges[-1]).sum()),
    }
    if actual.dtype == reference.dtype:
        width = actual.element_size()
        changed = (
            actual.contiguous().view(torch.uint8).reshape(-1, width)
            != reference.contiguous().view(torch.uint8).reshape(-1, width)
        ).any(1)
        result["bitwise_mismatch_count"] = int(changed.sum())
    if actual.dtype == reference.dtype and actual.dtype in (
        torch.bfloat16,
        torch.float32,
    ):
        integer, bits = (
            (torch.int16, 16) if actual.dtype == torch.bfloat16 else (torch.int32, 32)
        )
        mask, sign = (1 << bits) - 1, 1 << (bits - 1)

        def ordered(value):
            value = value.contiguous().view(integer).to(torch.int64) & mask
            return torch.where((value & sign) != 0, (~value) & mask, value | sign)

        distance = (ordered(actual) - ordered(reference)).abs()
        values, counts = torch.unique(distance, return_counts=True)
        result["ulp"] = {
            "dtype": str(actual.dtype),
            "max": int(distance.max()),
            "histogram": {str(int(v)): int(n) for v, n in zip(values, counts)},
            "signed_zero_note": "opposite signed zeros are adjacent bit patterns",
        }
    if per_row and a.ndim >= 2:
        rows_a = difference.reshape(a.shape[0], -1)
        rows_b = b.reshape(a.shape[0], -1)
        rms = rows_a.square().mean(1).sqrt()
        ref_rms = rows_b.square().mean(1).sqrt()
        result["per_row"] = {
            "max_abs": rows_a.abs().max(1).values.tolist(),
            "rmse": rms.tolist(),
            "normalized_rmse": [
                float(x / y) if y else None for x, y in zip(rms, ref_rms)
            ],
        }
    return result


def pair_metric(actual, reference):
    return {
        "output": metric(actual[0], reference[0]),
        "lse_log2": metric(actual[1], reference[1]),
    }


def cpu_pair(pair, active):
    # Copy only outside captures and timing; immutable snapshots survive later runs.
    return tuple(value[:active].detach().cpu().clone() for value in pair)


def reference_rows(active, table, count):
    count = min(active, count)
    rows = {0, active - 1, active // 2}
    boundaries = torch.nonzero(table[1:, 0] != table[:-1, 0]).flatten().tolist()
    for index in boundaries:
        rows.update((index, index + 1))
    ordered = sorted(rows)
    if len(ordered) > count:
        ordered = (
            [ordered[round(i * (len(ordered) - 1) / (count - 1))] for i in range(count)]
            if count > 1
            else [0]
        )
    rows = set(ordered)
    # Fill the largest unsampled intervals, rather than silently returning only
    # first/middle/last when the caller requested more representative rows.
    while len(rows) < count:
        ordered = sorted(rows)
        left, right = max(zip(ordered, ordered[1:]), key=lambda pair: pair[1] - pair[0])
        require(right - left > 1, "cannot fill representative row count")
        rows.add((left + right) // 2)
    return sorted(rows)


def diagnostic_pairs(state, outputs, options):
    """All legacy-failing row/head pairs, plus LSE worst and sample pairs."""
    selected, failed = set(), set()
    failure_records, worst_records, absolute_records = {}, {}, {}
    for mode in ("non_graph_eager", "graph_safe_eager", "replay"):
        actual, baseline = outputs["on:" + mode], outputs["off:" + mode]
        # Match torch.testing._comparison's original-dtype isclose exactly.
        bad = ~torch.isclose(actual[0], baseline[0], atol=LEGACY_ATOL, rtol=LEGACY_RTOL)
        coordinates = torch.nonzero(bad, as_tuple=False).tolist()
        pairs = {(row, head) for row, head, _ in coordinates}
        failed.update(pairs)
        failure_records[mode] = {
            "element_count": len(coordinates),
            "pair_count": len(pairs),
            "coordinates": coordinates,
        }
        absolute = (actual[0].double() - baseline[0].double()).abs()
        flat = int(absolute.argmax())
        row, remainder = divmod(flat, HEADS * DIM)
        head, column = divmod(remainder, DIM)
        selected.add((row, head))
        absolute_records[mode] = {
            "coordinate": [row, head, column],
            "max_abs": float(absolute.max()),
        }
        error = (actual[1].double() - baseline[1].double()).abs()
        count = min(options.diagnostic_lse_worst_pairs, error.numel())
        values, indices = torch.topk(error.flatten(), count)
        worst_records[mode] = []
        for value, index in zip(values.tolist(), indices.tolist()):
            row, head = divmod(index, HEADS)
            selected.add((row, head))
            worst_records[mode].append(
                {"row": row, "head": head, "abs_lse_log2": value}
            )
    selected.update(failed)
    sampled_rows = reference_rows(
        state["q"].shape[0], state["block_table"], options.reference_rows
    )
    sampled_pairs = {
        (row, head) for row in sampled_rows for head in options.reference_heads
    }
    selected.update(sampled_pairs)
    require(failed <= selected, "diagnostic coverage dropped a failing row/head pair")
    pairs = sorted(selected)
    unique_coordinates = sorted(
        {
            tuple(coordinate)
            for record in failure_records.values()
            for coordinate in record["coordinates"]
        }
    )
    return pairs, {
        "pairs": [list(pair) for pair in pairs],
        "legacy_failures": failure_records,
        "all_failed_pairs": [list(pair) for pair in sorted(failed)],
        "all_failed_pairs_covered": True,
        "oracle_covered_failed_elements": sum(
            value["element_count"] for value in failure_records.values()
        ),
        "covered_count_scope": "sum over three modes, not unique elements",
        "unique_failed_element_count": len(unique_coordinates),
        "unique_failed_coordinates": [list(value) for value in unique_coordinates],
        "uncovered_failed_elements": 0,
        "lse_worst_pairs": worst_records,
        "global_max_abs_pairs": absolute_records,
        "representative_pairs": [list(pair) for pair in sorted(sampled_pairs)],
        "requested_representative_rows": options.reference_rows,
        "selected_representative_rows": sampled_rows,
        "pair_count": len(pairs),
        "failed_pair_count": len(failed),
        "no_failure_pair_cap": True,
    }


def select_pairs(value, pairs):
    return torch.stack([value[row, head] for row, head in pairs])


def workspace_values(workspace, offset, count, dtype):
    """Read typed values from a byte offset; never interpret the offset as data."""
    require(workspace.is_contiguous(), "noncontiguous diagnostic workspace")
    size = torch.empty((), dtype=dtype).element_size()
    require(
        offset >= 0 and count >= 0 and offset % size == 0,
        "invalid workspace slice alignment",
    )
    raw = workspace.view(torch.uint8).flatten()
    require(offset + count * size <= raw.numel(), "workspace slice exceeds allocation")
    return raw[offset : offset + count * size].detach().cpu().clone().view(dtype)


def prefix_schedule_snapshot(cascade):
    """Decode the installed FA2 PrefillPlanInfo schema and valid partial storage."""
    wrapper = cascade.prefix
    names = (
        "padded_batch_size",
        "total_num_rows",
        "total_num_rows_offset",
        "cta_tile_q",
        "request_indices_offset",
        "qo_tile_indices_offset",
        "kv_tile_indices_offset",
        "merge_indptr_offset",
        "o_indptr_offset",
        "kv_chunk_size_ptr_offset",
        "v_offset",
        "s_offset",
        "block_valid_mask_offset",
        "enable_cuda_graph",
        "split_kv",
    )
    values = [int(value) for value in wrapper._plan_info]
    require(
        len(values) == len(names),
        "unsupported PrefillPlanInfo schema: expected 15 fields",
    )
    require(
        wrapper._paged_kv_indptr_buf.dtype == torch.int32,
        "diagnostic schema requires int32 IdType",
    )
    info = dict(zip(names, values))
    workspace = wrapper._int_workspace_buffer

    def integers(field, count):
        return workspace_values(workspace, info[field], count, torch.int32).tolist()

    rows = info["total_num_rows"]
    if info["enable_cuda_graph"]:
        rows = int(
            workspace_values(workspace, info["total_num_rows_offset"], 1, torch.int32)[
                0
            ]
        )
    require(0 <= rows <= info["total_num_rows"], "invalid live prefix row count")
    require(rows == int(cascade.qo[-1]), "prefix live rows disagree with qo indptr")
    chunk = integers("kv_chunk_size_ptr_offset", 1)[0]
    require(chunk > 0, "invalid actual kv_chunk_size value")
    padded = info["padded_batch_size"]
    if info["split_kv"] and info["enable_cuda_graph"]:
        valid = workspace_values(
            workspace, info["block_valid_mask_offset"], padded, torch.bool
        ).tolist()
    else:
        valid = [True] * padded
    request = integers("request_indices_offset", padded)
    qo_tile = integers("qo_tile_indices_offset", padded)
    kv_tile = integers("kv_tile_indices_offset", padded)
    launches = [
        {
            "launch_index": index,
            "request_index": int(request[index]),
            "qo_tile_index": int(qo_tile[index]),
            "kv_tile_index": int(kv_tile[index]),
        }
        for index in range(padded)
        if valid[index]
    ]
    for item in launches:
        require(
            0 <= item["request_index"] < cascade.slots
            and item["qo_tile_index"] >= 0
            and item["kv_tile_index"] >= 0,
            "invalid active prefix launch index",
        )
    output_ptr = integers("o_indptr_offset", cascade.slots + 1)
    record = {
        "plan_info_fields": info,
        "offset_unit": "bytes",
        "id_dtype": "int32",
        "partial_output_dtype": "bfloat16",
        "partial_lse_dtype": "float32",
        "actual_total_num_rows": rows,
        "actual_kv_chunk_size_tokens": int(chunk),
        "o_indptr": output_ptr,
        "block_valid_mask": valid,
        "valid_launches": launches,
        "valid_launch_count": len(launches),
        "backend": str(getattr(wrapper, "_backend", "unknown")),
        "split_kv": bool(info["split_kv"]),
    }
    partial = None
    if info["split_kv"]:
        indptr = integers("merge_indptr_offset", rows + 1)
        require(
            indptr[0] == 0
            and all(end >= begin for begin, end in zip(indptr, indptr[1:])),
            "invalid live merge indptr",
        )
        count = indptr[-1]
        require(output_ptr[-1] == count, "o_indptr and merge_indptr disagree")
        record["merge_indptr"] = indptr
        record["merge_counts_per_live_row"] = [
            end - begin for begin, end in zip(indptr, indptr[1:])
        ]
        record["effective_partial_rows"] = count
        # Prefix and tail share this float workspace. Copy immediately after
        # prefix execution, before tail can overwrite any partial values.
        float_workspace = wrapper._float_workspace_buffer
        partial_o = workspace_values(
            float_workspace, info["v_offset"], count * HEADS * DIM, torch.bfloat16
        ).reshape(count, HEADS, DIM)
        partial_s = workspace_values(
            float_workspace, info["s_offset"], count * HEADS, torch.float32
        ).reshape(count, HEADS)
        partial = (partial_o, partial_s)
        record["partial_snapshot_stage"] = (
            "after_prefix_before_tail_shared_workspace_reuse"
        )
    else:
        record["partial_snapshot_stage"] = "NOT_APPLICABLE_NO_SPLIT_WORKSPACE"
    return record, partial


def parse_prefix_kernel_traits(name):
    if "BatchPrefillWithPagedKVCacheKernel" not in name or "KernelTraits<" not in name:
        return None
    fields = name.split("KernelTraits<", 1)[1].split(",")
    if (
        len(fields) < 9
        or "MaskMode" not in fields[0]
        or "PosEncodingMode" not in fields[8]
    ):
        return None
    numbers = []
    for field in fields[1:8]:
        match = re.fullmatch(r"\s*(\d+)[uUlL]*\s*", field)
        if match is None:
            return None
        numbers.append(int(match.group(1)))
    keys = (
        "CTA_TILE_Q",
        "NUM_MMA_Q",
        "NUM_MMA_KV",
        "NUM_MMA_D_QK",
        "NUM_MMA_D_VO",
        "NUM_WARPS_Q",
        "NUM_WARPS_KV",
    )
    result = dict(zip(keys, numbers))
    result["CTA_TILE_KV"] = result["NUM_MMA_KV"] * result["NUM_WARPS_KV"] * 16
    result["HEAD_DIM_QK"] = result["NUM_MMA_D_QK"] * 16
    result["HEAD_DIM_VO"] = result["NUM_MMA_D_VO"] * 16
    return result


def diagnostic_trace_record(path, expected_cta_q):
    trace = json.loads(path.read_text())
    kernels = [
        event for event in trace.get("traceEvents", []) if event.get("cat") == "kernel"
    ]
    prefixes = []
    for event in kernels:
        if "BatchPrefillWithPagedKVCacheKernel" not in event.get("name", ""):
            continue
        arguments = event.get("args", {})
        prefixes.append(
            {
                "name": event["name"],
                "traits": parse_prefix_kernel_traits(event["name"]),
                "grid": arguments.get("grid"),
                "block": arguments.get("block"),
                "correlation": arguments.get("correlation"),
                "external_id": arguments.get("External id"),
            }
        )
    matching = [item["traits"] for item in prefixes if item["traits"] is not None]
    proven = (
        bool(prefixes)
        and len(matching) == len(prefixes)
        and all(item == matching[0] for item in matching)
        and matching[0]["CTA_TILE_Q"] == expected_cta_q
        and matching[0]["HEAD_DIM_QK"] == DIM
        and matching[0]["HEAD_DIM_VO"] == DIM
    )
    return {
        "path": path.name,
        "sha256": digest_file(path),
        "kernel_count": len(kernels),
        "prefix_kernels": prefixes,
        "actual_template_proven": proven,
        "actual_traits": matching[0] if proven else None,
        "status": "TEMPLATE_PARSED_FROM_THIS_CALL" if proven else "TEMPLATE_NOT_PROVEN",
        "timing_use": "DIAGNOSTIC_ONLY_EXCLUDED_FROM_CLEAN_SAMPLES",
    }


def flashinfer_source_identity():
    module = importlib.import_module("flashinfer")
    root = Path(module.__file__).parent
    relative = (
        "data/include/flashinfer/attention/scheduler.cuh",
        "data/include/flashinfer/attention/prefill.cuh",
        "data/csrc/batch_prefill.cu",
    )
    return {
        "version": str(getattr(module, "__version__", "unknown")),
        "sources": {
            name: {"path": str(root / name), "sha256": digest_file(root / name)}
            for name in relative
            if (root / name).is_file()
        },
        "compiled_macros": "NOT_OBSERVED; source hash is not a compiled-macro trace",
    }


def read_cascade_parts(runner, active, trace_path=None):
    """Diagnostic GPU execution only; never invoked in capture or timing."""
    require(
        not torch.cuda.is_current_stream_capturing(), "stage diagnosis during capture"
    )
    require(
        isinstance(runner.backend, CascadeBackend),
        "stage diagnosis requires production Cascade",
    )
    torch.cuda.synchronize()
    cascade = runner.backend.impl
    profiler = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        )
        if trace_path is not None
        else contextlib.nullcontext()
    )
    with profiler as profile:
        with torch.profiler.record_function("cascade_diagnostic.prefix"):
            prefix = cascade.prefix.run(runner.q, runner.kv, return_lse=True)
        prefix_before_merge = cpu_pair(prefix, active)
        schedule, partial = prefix_schedule_snapshot(cascade)
        with torch.profiler.record_function("cascade_diagnostic.tail"):
            tail = cascade.tail.run(runner.q, runner.kv, return_lse=True)
        tail_before_merge = cpu_pair(tail, active)
        with torch.profiler.record_function("cascade_diagnostic.final_merge"):
            merge_state_in_place(prefix[0], prefix[1], tail[0], tail[1])
        merged = cpu_pair(prefix, active)
    trace = {"status": "NOT_REQUESTED", "actual_template_proven": False}
    if trace_path is not None:
        profile.export_chrome_trace(str(trace_path))
        trace = diagnostic_trace_record(
            trace_path, schedule["plan_info_fields"]["cta_tile_q"]
        )
    return {
        "prefix": prefix_before_merge,
        "tail": tail_before_merge,
        "merged": merged,
        "schedule": schedule,
        "partial": partial,
        "trace": trace,
        "source_identity": flashinfer_source_identity(),
        "plan": {
            "qo": cascade.qo.tolist(),
            "prefix_ptr": cascade.prefix_ptr.tolist(),
            "prefix_idx": cascade.prefix_idx.tolist(),
            "prefix_last": cascade.prefix_last.tolist(),
            "tail_ptr": cascade.tail_ptr.tolist(),
            "tail_idx": cascade.tail_idx.tolist(),
            "tail_last": cascade.tail_last.tolist(),
        },
    }


def split_tokens_for_row(state, plan, row):
    """Validate the actual planned partition; do not infer it from length alone."""
    slot = next(
        index
        for index, (start, end) in enumerate(zip(plan["qo"], plan["qo"][1:]))
        if start <= row < end
    )
    pbegin, pend = plan["prefix_ptr"][slot : slot + 2]
    tbegin, tend = plan["tail_ptr"][row : row + 2]
    prefix_pages = plan["prefix_idx"][pbegin:pend]
    tail_pages = plan["tail_idx"][tbegin:tend]
    prefix_tokens = (
        (len(prefix_pages) - 1) * PAGE + plan["prefix_last"][slot]
        if prefix_pages
        else 0
    )
    tail_tokens = (
        (len(tail_pages) - 1) * PAGE + plan["tail_last"][row] if tail_pages else 0
    )
    total = int(state["sequence_lengths"][row]) + 1
    full_pages = state["block_table"][row, : (total + PAGE - 1) // PAGE].tolist()
    require(
        prefix_pages + tail_pages == full_pages,
        "planned prefix/tail pages do not cover full KV in order",
    )
    require(
        prefix_tokens % PAGE == 0 and prefix_tokens + tail_tokens == total,
        "planned prefix/tail token partition has overlap or a gap",
    )
    require(tail_tokens > 0, "current token missing from tail")
    return prefix_tokens


def softmax64(scores, values):
    if scores.numel() == 0:
        return torch.zeros(DIM, dtype=torch.float64), torch.tensor(
            -math.inf, dtype=torch.float64
        )
    return scores.softmax(-1) @ values, torch.logsumexp(scores, dim=-1) / math.log(2)


def merge64(first, second):
    """Merge actual or ideal log2 states, explicitly ignoring empty branches."""
    a, sa = first[0].double(), first[1].double()
    b, sb = second[0].double(), second[1].double()
    require(
        bool((torch.isfinite(sa) | torch.isneginf(sa)).all()), "invalid first merge LSE"
    )
    require(
        bool((torch.isfinite(sb) | torch.isneginf(sb)).all()),
        "invalid second merge LSE",
    )
    valid_a, valid_b = torch.isfinite(sa), torch.isfinite(sb)
    require(bool((valid_a | valid_b).all()), "cannot merge two empty states")
    a = torch.where(valid_a[:, None], a, torch.zeros_like(a))
    b = torch.where(valid_b[:, None], b, torch.zeros_like(b))
    maximum = torch.maximum(sa, sb)
    wa, wb = torch.exp2(sa - maximum), torch.exp2(sb - maximum)
    denominator = wa + wb
    return (
        (wa[:, None] * a + wb[:, None] * b) / denominator[:, None],
        maximum + torch.log2(denominator),
    )


def online_proxy_contract(parts):
    """Require this call's geometry; never infer a missing template from page size."""
    trace, schedule = parts["trace"], parts["schedule"]
    reasons = []
    if not trace.get("actual_template_proven"):
        reasons.append("actual prefix template not proven by this diagnostic trace")
    traits = trace.get("actual_traits") or {}
    if traits.get("NUM_WARPS_KV") != 1:
        reasons.append("multi-warp KV recurrence is outside this proxy contract")
    if not schedule.get("split_kv") or set(
        schedule.get("merge_counts_per_live_row", [])
    ) != {1}:
        reasons.append("requires exactly one stored prefix partial per live row")
    if any(item["kv_tile_index"] != 0 for item in schedule["valid_launches"]):
        reasons.append("multiple prefix KV chunks are outside this proxy contract")
    type_pattern = (
        r"PosEncodingMode\)0,\s*__nv_bfloat16,\s*__nv_bfloat16,\s*__nv_bfloat16,"
        r"\s*float,\s*int,\s*flashinfer::DefaultAttention<false,\s*false,\s*false,\s*false>"
    )
    for kernel in trace.get("prefix_kernels", []):
        if (
            "MaskMode)0," not in kernel["name"]
            or re.search(type_pattern, kernel["name"]) is None
        ):
            reasons.append(
                "unsupported prefix mask, position, type or attention variant"
            )
    expected_hashes = {
        "data/include/flashinfer/attention/scheduler.cuh": "ef69a1ea29b41cb150c5b1b97d12d63d258026211733c47322cea9c29948fb64",
        "data/include/flashinfer/attention/prefill.cuh": "8a6b826afdaaae377891cdc106926261007b7ad40f19468a1a10ab3d7f017dc3",
        "data/csrc/batch_prefill.cu": "90fcdb0b8ac13ff7ceda33bdf6b3cf5f7805f66933866149db06b116dd74eb8b",
    }
    sources = parts["source_identity"]["sources"]
    if any(
        sources.get(name, {}).get("sha256") != value
        for name, value in expected_hashes.items()
    ):
        reasons.append(
            "installed scheduler/prefill source differs from the reviewed proxy contract"
        )
    return {
        "status": (
            "NOT_SUPPORTED" if reasons else "GEOMETRY_PROVEN_FOR_FP64_MECHANISM_PROXY"
        ),
        "reasons": reasons,
        "tile_tokens": traits.get("CTA_TILE_KV"),
        "chunk_tokens": schedule["actual_kv_chunk_size_tokens"],
        "single_chunk_per_query": not reasons,
        "arithmetic_label": "FP64_ONLINE_BF16_EXP_PROXY_NOT_PPU_INSTRUCTION_EMULATION",
        "unmodelled": [
            "FP32 QK MMA accumulation and scaling",
            "FP32 rowsum/PV reduction and rescaling",
            "approximate exp2/log2/reciprocal",
            "FMA/compiler/layout arithmetic",
        ],
    }


def online_prefix64(scores, values, tile_tokens, round_probability):
    """Geometry-matched recurrence; all arithmetic except explicit cast is FP64."""
    if scores.numel() == 0:
        return softmax64(scores, values), {"tile_count": 0, "running_max_increases": 0}
    scores_log2 = scores / math.log(2)
    maximum = torch.tensor(-math.inf, dtype=torch.float64)
    numerator = torch.zeros(DIM, dtype=torch.float64)
    denominator = torch.tensor(0.0, dtype=torch.float64)
    updates, tile_count = 0, 0
    for start in range(0, scores.numel(), tile_tokens):
        stop = min(start + tile_tokens, scores.numel())
        current = scores_log2[start:stop]
        previous = maximum
        maximum = torch.maximum(previous, current.max())
        updates += int(bool(maximum > previous))
        alpha = torch.exp2(previous - maximum)
        exponent = torch.exp2(current - maximum)
        if round_probability:
            exponent = exponent.to(torch.bfloat16).double()
        numerator = numerator * alpha + exponent @ values[start:stop]
        denominator = denominator * alpha + exponent.sum()
        tile_count += 1
    return (
        (numerator / denominator, maximum + denominator.log2()),
        {"tile_count": tile_count, "running_max_increases": updates},
    )


def stage_references(state, pairs, plan, online_geometry=None):
    """R0 and R3 global-max proxies; each selected pair gets all valid tokens."""
    collected = {
        name: []
        for name in (
            "full",
            "prefix",
            "tail",
            "probability_bf16_both",
            "probability_bf16_numerator_only",
            "normalized_probability_counterexample",
        )
    }
    if online_geometry is not None:
        collected.update(
            {
                "online_probability_bf16_both": [],
                "online_no_probability_cast_control": [],
            }
        )
    online_metadata = []
    prefix_lengths, total_lengths = [], []
    for row, head in pairs:
        total = int(state["sequence_lengths"][row]) + 1
        cut = split_tokens_for_row(state, plan, row)
        ids = state["block_table"][row, : (total + PAGE - 1) // PAGE].long()
        data = state["kv_cache"][ids, :, head // (HEADS // KV_HEADS)]
        data = data.permute(1, 0, 2, 3).reshape(2, -1, DIM)[:, :total].double()
        scores = (data[0] @ state["q"][row, head].double()) / math.sqrt(DIM)
        collected["full"].append(softmax64(scores, data[1]))
        prefix = softmax64(scores[:cut], data[1, :cut])
        collected["prefix"].append(prefix)
        collected["tail"].append(softmax64(scores[cut:], data[1, cut:]))
        if online_geometry is not None:
            require(
                cut <= online_geometry["chunk_tokens"],
                "prefix exceeds the proven single chunk",
            )
            for name, rounding in (
                ("online_probability_bf16_both", True),
                ("online_no_probability_cast_control", False),
            ):
                value, audit = online_prefix64(
                    scores[:cut],
                    data[1, :cut],
                    online_geometry["tile_tokens"],
                    rounding,
                )
                collected[name].append(value)
                if rounding:
                    online_metadata.append(audit)
        if cut:
            maximum = scores[:cut].max()
            exponent = torch.exp(scores[:cut] - maximum)
            rounded = exponent.to(torch.bfloat16).double()
            numerator = rounded @ data[1, :cut]
            denominator = rounded.sum()
            collected["probability_bf16_both"].append(
                (numerator / denominator, (maximum + denominator.log()) / math.log(2))
            )
            collected["probability_bf16_numerator_only"].append(
                (numerator / exponent.sum(), prefix[1])
            )
            normalized = scores[:cut].softmax(-1).to(torch.bfloat16).double()
            collected["normalized_probability_counterexample"].append(
                (normalized @ data[1, :cut], prefix[1])
            )
        else:
            for name in (
                "probability_bf16_both",
                "probability_bf16_numerator_only",
                "normalized_probability_counterexample",
            ):
                collected[name].append(prefix)
        prefix_lengths.append(cut)
        total_lengths.append(total)
    references = {
        name: tuple(torch.stack([value[index] for value in values]) for index in (0, 1))
        for name, values in collected.items()
    }
    references["prefix_lengths"] = prefix_lengths
    references["total_lengths"] = total_lengths
    if online_geometry is not None:
        references["online_metadata"] = online_metadata
    return references


def boundary_errors(actual, reference, valid=None):
    """No raw activation values are serialized. Empty prefix O is undefined."""
    if valid is None:
        valid = torch.ones(reference[1].numel(), dtype=torch.bool)
    result = {
        "valid_pair_indices": torch.nonzero(valid).flatten().tolist(),
        "empty_pair_count": int((~valid).sum()),
    }
    if bool((~valid).any()):
        result["empty_lse_negative_infinity_count"] = int(
            torch.isneginf(actual[1][~valid]).sum()
        )
        require(
            bool(torch.isneginf(actual[1][~valid]).all()),
            "empty prefix LSE is not negative infinity",
        )
    if not bool(valid.any()):
        result["status"] = "EMPTY_PREFIX_OUTPUT_IGNORED"
        return result
    for name, a, b in (
        ("output", actual[0][valid], reference[0][valid]),
        ("lse_log2", actual[1][valid], reference[1][valid]),
    ):
        rounded = b.to(a.dtype)
        result[name] = {
            "against_float64": metric(a, b),
            "against_rounded_reference": metric(a, rounded),
            "rounded_reference_floor": metric(rounded, b),
        }
    result["per_pair_abs_lse_log2"] = (
        (actual[1][valid].double() - reference[1][valid]).abs().tolist()
    )
    return result


def cast_state(pair, cast_lse=True):
    return (
        pair[0].to(torch.bfloat16).double(),
        pair[1].float().double() if cast_lse else pair[1].double(),
    )


def counterfactual_errors(pair, full, actual_on):
    cast = (pair[0].to(torch.bfloat16), pair[1].float())
    return {
        "against_full_fp64_before_final_cast": pair_metric(pair, full),
        "against_full_fp64_after_final_cast": pair_metric(cast, full),
        "actual_on_against_counterfactual_cast": pair_metric(actual_on, cast),
    }


def partial_prefix_errors(parts, pairs, actual, references, valid):
    """Re-merge real stored prefix partials; no partial activation is serialized."""
    if parts["partial"] is None:
        return {"status": "NOT_APPLICABLE_NO_SPLIT_WORKSPACE"}
    schedule = parts["schedule"]
    output, lse = parts["partial"]
    pointer = schedule["merge_indptr"]
    selected_o, selected_s, counts = [], [], []
    for index, (row, head) in enumerate(pairs):
        begin, end = pointer[row : row + 2]
        count = end - begin
        expected = max(
            1,
            math.ceil(
                references["prefix_lengths"][index]
                / schedule["actual_kv_chunk_size_tokens"]
            ),
        )
        require(
            count == expected, "actual merge count disagrees with planned KV chunk size"
        )
        counts.append(count)
        if not bool(valid[index]):
            selected_o.append(torch.zeros(DIM, dtype=torch.float64))
            selected_s.append(torch.tensor(-math.inf, dtype=torch.float64))
            continue
        require(count > 0, "nonempty prefix has no stored partial")
        values, states = output[begin:end, head].double(), lse[begin:end, head].double()
        finite = torch.isfinite(states)
        require(
            bool((finite | torch.isneginf(states)).all()) and bool(finite.any()),
            "invalid stored partial LSE",
        )
        values = torch.where(finite[:, None], values, torch.zeros_like(values))
        maximum = states.max()
        weights = torch.exp2(states - maximum)
        denominator = weights.sum()
        selected_o.append((weights[:, None] * values).sum(0) / denominator)
        selected_s.append(maximum + denominator.log2())
    merged = torch.stack(selected_o), torch.stack(selected_s)
    return {
        "status": "ACTUAL_STORED_PARTIALS_MERGED_ON_CPU_FP64",
        "selected_pair_merge_counts": counts,
        "chunk_counts_checked_against_actual_token_value": True,
        "actual_prefix_vs_partial_merge": boundary_errors(actual, merged, valid),
        "partial_merge_vs_ideal_prefix": boundary_errors(
            merged, references["prefix"], valid
        ),
        "storage_scope": "effective rows only; cloned before tail; activation values not serialized",
    }


def signed_failure_errors(coverage, pairs, outputs, full, counterfactuals):
    """Signed errors at every failing element, retaining the original BF16 gate."""
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    cast_counterfactuals = {
        name: pair[0].to(torch.bfloat16) for name, pair in counterfactuals.items()
    }
    records = []
    for row, head, column in coverage["unique_failed_coordinates"]:
        index = pair_index[(row, head)]
        reference = full[0][index, column]
        floor = reference.to(torch.bfloat16)
        item = {
            "coordinate": [row, head, column],
            "selected_pair_index": index,
            "bf16_reference_rounding_floor_minus_fp64": float(
                floor.double() - reference
            ),
            "counterfactual_error_minus_fp64": {
                name: float(value[index, column].double() - reference)
                for name, value in cast_counterfactuals.items()
            },
            "modes": {},
        }
        for mode in ("non_graph_eager", "graph_safe_eager", "replay"):
            on = outputs["on:" + mode][0][row, head, column]
            off = outputs["off:" + mode][0][row, head, column]

            def fails(actual, expected):
                return not bool(
                    torch.isclose(actual, expected, atol=LEGACY_ATOL, rtol=LEGACY_RTOL)
                )

            item["modes"][mode] = {
                "on_minus_off": float(on.double() - off.double()),
                "on_minus_fp64": float(on.double() - reference),
                "off_minus_fp64": float(off.double() - reference),
                "on_vs_off_exceeds_legacy_gate": fails(on, off),
                "counterfactuals": {
                    name: {
                        "on_minus_counterfactual": float(
                            on.double() - value[index, column].double()
                        ),
                        "counterfactual_minus_off": float(
                            value[index, column].double() - off.double()
                        ),
                        "counterfactual_vs_off_exceeds_legacy_gate": fails(
                            value[index, column], off
                        ),
                        "on_vs_counterfactual_exceeds_legacy_gate": fails(
                            on, value[index, column]
                        ),
                    }
                    for name, value in cast_counterfactuals.items()
                },
            }
        records.append(item)
    require(
        len(records) == coverage["unique_failed_element_count"],
        "failed-coordinate diagnostic truncated",
    )
    return {
        "count": len(records),
        "scope": "all unique legacy-failing output coordinates; no cap",
        "gate": {
            "atol": LEGACY_ATOL,
            "rtol": LEGACY_RTOL,
            "implementation": "torch.isclose on original BF16 operands; asymmetric expected operand",
            "counterfactual_storage": "final BF16 output cast before gate",
        },
        "records": records,
    }


def signed_lse_and_weights(pairs, outputs, actual_a, actual_b, references):
    def weights(first, second):
        maximum = torch.maximum(first.double(), second.double())
        a, b = torch.exp2(first.double() - maximum), torch.exp2(
            second.double() - maximum
        )
        return a / (a + b), b / (a + b)

    ideal_a, ideal_b = references["prefix"], references["tail"]
    actual_weights = weights(actual_a[1], actual_b[1])
    ideal_weights = weights(ideal_a[1], ideal_b[1])
    proxy_names = [
        "probability_bf16_both",
        "probability_bf16_numerator_only",
        "normalized_probability_counterexample",
    ]
    if "online_probability_bf16_both" in references:
        proxy_names.append("online_probability_bf16_both")
    records = []
    for index, (row, head) in enumerate(pairs):
        nonempty = references["prefix_lengths"][index] > 0
        item = {
            "coordinate": [row, head],
            "selected_pair_index": index,
            "prefix_nonempty": nonempty,
            "actual_prefix_minus_fp64_lse_log2": (
                float(actual_a[1][index].double() - ideal_a[1][index])
                if nonempty
                else None
            ),
            "actual_tail_minus_fp64_lse_log2": float(
                actual_b[1][index].double() - ideal_b[1][index]
            ),
            "actual_prefix_minus_proxy_lse_log2": {
                name: (
                    float(actual_a[1][index].double() - references[name][1][index])
                    if nonempty
                    else None
                )
                for name in proxy_names
            },
            "merge_weights": {
                "actual_prefix": float(actual_weights[0][index]),
                "actual_tail": float(actual_weights[1][index]),
                "ideal_prefix": float(ideal_weights[0][index]),
                "ideal_tail": float(ideal_weights[1][index]),
                "actual_minus_ideal_prefix": float(
                    actual_weights[0][index] - ideal_weights[0][index]
                ),
                "actual_minus_ideal_tail": float(
                    actual_weights[1][index] - ideal_weights[1][index]
                ),
            },
            "modes": {},
        }
        for mode in ("non_graph_eager", "graph_safe_eager", "replay"):
            on = outputs["on:" + mode][1][row, head].double()
            off = outputs["off:" + mode][1][row, head].double()
            full = references["full"][1][index]
            item["modes"][mode] = {
                "on_minus_off_lse_log2": float(on - off),
                "on_minus_fp64_lse_log2": float(on - full),
                "off_minus_fp64_lse_log2": float(off - full),
            }
        records.append(item)
    return {
        "scope": "all selected pairs including every failing pair and LSE worst pairs",
        "records": records,
    }


def diagnose_stages(state, runner, outputs, options, trace_path=None):
    start = time.perf_counter_ns()
    pairs, coverage = diagnostic_pairs(state, outputs, options)
    parts = read_cascade_parts(runner, state["q"].shape[0], trace_path)
    online_contract = (
        online_proxy_contract(parts)
        if options.diagnose_online_proxy
        else {"status": "NOT_REQUESTED"}
    )
    online_geometry = (
        online_contract
        if online_contract["status"] == "GEOMETRY_PROVEN_FOR_FP64_MECHANISM_PROXY"
        else None
    )
    observed = {
        name: tuple(select_pairs(value, pairs) for value in parts[name])
        for name in ("prefix", "tail", "merged")
    }
    references = stage_references(state, pairs, parts["plan"], online_geometry)
    full, ideal_a, ideal_b = (references[name] for name in ("full", "prefix", "tail"))
    actual_a, actual_b = observed["prefix"], observed["tail"]
    valid_prefix = torch.tensor([value > 0 for value in references["prefix_lengths"]])
    ideal_merged = merge64(ideal_a, ideal_b)
    # Internal FP64 identity check only; not an attention acceptance threshold.
    for actual, reference in zip(ideal_merged, full):
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    report = {
        "status": "DIAGNOSTIC_COMPLETE",
        "accuracy_acceptance": "NOT_ASSESSED",
        "execution": "graph-safe wrappers run eagerly; outside graph capture and all timing",
        "coverage": coverage,
        "partition": {
            "prefix_tokens_per_pair": references["prefix_lengths"],
            "total_tokens_per_pair": references["total_lengths"],
            "actual_planned_pages_match_full_table": True,
            "token_partition_no_overlap_no_gap": True,
        },
        "merged_vs_normal_pipeline": pair_metric(
            parts["merged"], outputs["on:graph_safe_eager"]
        ),
        "R0": {
            "ideal_partition_merge_vs_direct_full": pair_metric(ideal_merged, full),
            "identity_check": {"atol": 1e-12, "rtol": 1e-12},
            "prefix": boundary_errors(actual_a, ideal_a, valid_prefix),
            "tail": boundary_errors(actual_b, ideal_b),
        },
        "R1": {},
        "R2": {},
        "R3": {},
        "R4_evidence": {
            "status": "METADATA_AND_OPTIONAL_TRACE_ONLY; ONLINE_PROXY_NOT_COMPUTED",
            "prefix_schedule": parts["schedule"],
            "diagnostic_trace": parts["trace"],
            "source_identity": parts["source_identity"],
        },
        "R4": {"contract": online_contract},
        "limitations": [
            "Partial snapshots are stored BF16 O/FP32 LSE, not pre-round accumulators.",
            "Global-max probability models are proxies, not tile-online or PPU instruction oracles.",
            "Do not add RMSE values or interpret them as additive percentage attribution.",
            "No business beam-order acceptance is implied.",
        ],
    }
    for name, pair in outputs.items():
        selected = tuple(select_pairs(value, pairs) for value in pair)
        report["R0"][name] = boundary_errors(selected, full)
    cpu_actual_merge = merge64(actual_a, actual_b)
    counterfactuals = {}
    report["R1"]["actual_operands_merge_residual"] = boundary_errors(
        observed["merged"], cpu_actual_merge
    )
    substitutions = {
        "ideal_prefix_ideal_tail": (ideal_a, ideal_b),
        "actual_prefix_ideal_tail": (actual_a, ideal_b),
        "ideal_prefix_actual_tail": (ideal_a, actual_b),
        "actual_prefix_actual_tail": (actual_a, actual_b),
        "actual_prefix_output_ideal_prefix_lse_actual_tail": (
            (actual_a[0], ideal_a[1]),
            actual_b,
        ),
        "ideal_prefix_output_actual_prefix_lse_actual_tail": (
            (ideal_a[0], actual_a[1]),
            actual_b,
        ),
    }
    for name, (first, second) in substitutions.items():
        simulated = merge64(first, second)
        counterfactuals["R1:" + name] = simulated
        report["R1"][name] = counterfactual_errors(simulated, full, observed["merged"])
    for cast_lse in (False, True):
        name = "partial_bf16_output_" + ("fp32_lse" if cast_lse else "fp64_lse")
        simulated = merge64(
            cast_state(ideal_a, cast_lse), cast_state(ideal_b, cast_lse)
        )
        counterfactuals["R2:" + name] = simulated
        report["R2"][name] = counterfactual_errors(simulated, full, observed["merged"])
    for name in (
        "probability_bf16_both",
        "probability_bf16_numerator_only",
        "normalized_probability_counterexample",
    ):
        proxy = references[name]
        simulated = merge64(cast_state(proxy), actual_b)
        counterfactuals["R3:" + name] = simulated
        report["R3"][name] = {
            "label": "GLOBAL_MAX_PROXY_NOT_MATCHED_TILE_ORACLE",
            "actual_prefix_vs_proxy": boundary_errors(actual_a, proxy, valid_prefix),
            "proxy_vs_ideal_prefix": boundary_errors(proxy, ideal_a, valid_prefix),
            "proxy_prefix_stored_then_merge_actual_tail": counterfactual_errors(
                simulated, full, observed["merged"]
            ),
        }
    report["R4_evidence"]["stored_partial_merge"] = partial_prefix_errors(
        parts, pairs, actual_a, references, valid_prefix
    )
    if online_geometry is not None:
        proxy = references["online_probability_bf16_both"]
        control = references["online_no_probability_cast_control"]
        for actual, reference in zip(control, ideal_a):
            torch.testing.assert_close(
                actual[valid_prefix], reference[valid_prefix], atol=1e-12, rtol=1e-12
            )
        simulated = merge64(cast_state(proxy), actual_b)
        counterfactuals["R4:online_probability_bf16_both"] = simulated
        report["R4"].update(
            {
                "status": "GEOMETRY_MATCHED_MECHANISM_PROXY_COMPLETE_NOT_HARDWARE_EMULATION",
                "per_selected_pair_recurrence": references["online_metadata"],
                "no_probability_cast_control_vs_ideal_prefix": boundary_errors(
                    control, ideal_a, valid_prefix
                ),
                "control_identity_check": {
                    "atol": 1e-12,
                    "rtol": 1e-12,
                    "purpose": "FP64 oracle self-consistency only",
                },
                "actual_prefix_vs_proxy": boundary_errors(
                    actual_a, proxy, valid_prefix
                ),
                "proxy_vs_ideal_prefix": boundary_errors(proxy, ideal_a, valid_prefix),
                "proxy_prefix_stored_then_merge_actual_tail": counterfactual_errors(
                    simulated, full, observed["merged"]
                ),
                "prediction": "Online-max rescaling explains global-proxy residual only if actual-prefix errors decrease; evaluate signed coordinates too",
            }
        )
        report["R4_evidence"][
            "status"
        ] = "ACTUAL_GEOMETRY_USED_FOR_FP64_MECHANISM_PROXY; HARDWARE_ARITHMETIC_NOT_EMULATED"
    report["signed_failed_coordinates"] = signed_failure_errors(
        coverage, pairs, outputs, full, counterfactuals
    )
    report["signed_lse_and_merge_weights"] = signed_lse_and_weights(
        pairs, outputs, actual_a, actual_b, references
    )
    report["diagnostic_wall_seconds_not_kernel_time"] = (
        time.perf_counter_ns() - start
    ) / 1e9
    return report, parts["merged"]


def validate_backend_for_test(runner):
    """Optional candidate coverage check, always outside capture and timing."""
    callback = getattr(runner.backend, "validate_for_test", None)
    if callback is None:
        return {"status": "NOT_PROVIDED"}
    require(
        not torch.cuda.is_current_stream_capturing(),
        "candidate validation during graph capture",
    )
    result = callback(runner.q, runner.kv)
    require(
        isinstance(result, dict), "validate_for_test must return a JSON-safe stats dict"
    )
    json.dumps(result, allow_nan=False)
    return {
        "status": "CALLBACK_COMPLETED",
        "scope": "outside capture/timing; callback restores valid current outputs",
        "stats": result,
    }


def cpu_reference(state, options):
    """Independent FP64 QK -> full softmax -> PV, no FlashInfer or fused SDPA."""
    active = state["q"].shape[0]
    rows = reference_rows(active, state["block_table"], options.reference_rows)
    heads = options.reference_heads
    outputs, lses = [], []
    for row in rows:
        length = int(state["sequence_lengths"][row]) + 1
        pages = (length + PAGE - 1) // PAGE
        ids = state["block_table"][row, :pages].long()
        kv = state["kv_cache"][ids].permute(1, 2, 0, 3, 4)
        kv = kv.reshape(2, KV_HEADS, -1, DIM)[:, :, :length].double()
        selected_kv_heads = [head // (HEADS // KV_HEADS) for head in heads]
        q = state["q"][row, heads].double()
        k, v = kv[0, selected_kv_heads], kv[1, selected_kv_heads]
        score = torch.einsum("hd,htd->ht", q, k) / math.sqrt(DIM)
        outputs.append(torch.einsum("ht,htd->hd", score.softmax(-1), v))
        # FlashInfer attention/state.cuh returns m + log2(d).
        lses.append(torch.logsumexp(score, dim=-1) / math.log(2))
    return {
        "rows": rows,
        "heads": heads,
        "output": torch.stack(outputs),
        "lse": torch.stack(lses),
    }


def against_reference(pair, oracle):
    output = pair[0][oracle["rows"]][:, oracle["heads"]]
    lse = pair[1][oracle["rows"]][:, oracle["heads"]]
    result = {}
    for key, actual, reference in (
        ("output", output, oracle["output"]),
        ("lse_log2", lse, oracle["lse"]),
    ):
        rounded = reference.to(actual.dtype)
        result[key] = {
            "against_float64": metric(actual, reference),
            "against_rounded_reference": metric(actual, rounded),
            "rounded_reference_floor": metric(rounded, reference),
        }
    return result


def validate_state(state):
    tensors = ("q", "kv_cache", "block_table", "sequence_lengths")
    for key in tensors:
        require(
            key in state and isinstance(state[key], torch.Tensor),
            f"missing tensor {key}",
        )
        require(state[key].device.type == "cpu", f"fixture {key} must be on CPU")
        state[key] = state[key].contiguous()
    q, kv, table, lengths = (state[key] for key in tensors)
    active = q.shape[0]
    require(q.shape == (active, HEADS, DIM) and q.dtype == torch.bfloat16, "invalid Q")
    require(
        kv.ndim == 5
        and tuple(kv.shape[1:]) == (2, KV_HEADS, PAGE, DIM)
        and kv.dtype == torch.bfloat16,
        "invalid HND BF16 KV",
    )
    require(
        table.ndim == 2 and table.shape[0] == active and table.dtype == torch.int32,
        "invalid block table",
    )
    require(
        lengths.shape == (active,) and lengths.dtype == torch.int32,
        "invalid history lengths",
    )
    require(bool((lengths >= 0).all()), "negative history length")
    pages = (lengths.long() + PAGE) // PAGE
    require(int(pages.max()) <= table.shape[1], "insufficient page-table width")
    used = table[torch.arange(table.shape[1])[None, :] < pages[:, None]]
    require(
        int(used.min()) >= 0 and int(used.max()) < kv.shape[0],
        "page id outside compact KV",
    )
    require(
        active > 0 and bool(torch.isfinite(q).all()) and bool(torch.isfinite(kv).all()),
        "empty/nonfinite fixture",
    )
    return state


def synthetic_state(active, history, groups, seed, regime="random"):
    generator = torch.Generator().manual_seed(seed)
    shared_pages = history // PAGE
    total_pages = groups * shared_pages + active
    table = torch.zeros(active, max(24, shared_pages + 2), dtype=torch.int32)
    for group in range(groups):
        start, end = group * active // groups, (group + 1) * active // groups
        if shared_pages:
            table[start:end, :shared_pages] = torch.arange(
                group * shared_pages, (group + 1) * shared_pages, dtype=torch.int32
            )
        table[start:end, shared_pages] = (
            torch.arange(start, end, dtype=torch.int32) + groups * shared_pages
        )
    q = torch.randn(active, HEADS, DIM, generator=generator).to(torch.bfloat16)
    kv = torch.randn(total_pages, 2, KV_HEADS, PAGE, DIM, generator=generator).to(
        torch.bfloat16
    )
    if regime == "uniform":
        q.zero_()
    elif regime == "peaked":
        q.mul_(4)
        kv[:, 0].mul_(4)
    return validate_state(
        {
            "q": q,
            "kv_cache": kv,
            "block_table": table,
            "sequence_lengths": torch.full((active,), history, dtype=torch.int32),
            "metadata": {"source": "synthetic", "groups": groups, "regime": regime},
        }
    )


def changed_fixture(state):
    """Valid synthetic perturbation, explicitly not another captured model step."""
    pages = state["kv_cache"].shape[0]
    lengths = (state["sequence_lengths"] - 1).clamp_min(0)
    table = state["block_table"].clone()
    used = (
        torch.arange(table.shape[1])[None, :]
        < ((lengths.long() + PAGE) // PAGE)[:, None]
    )
    table[used] = pages - 1 - table[used]
    return validate_state(
        {
            "q": -state["q"].roll(1, dims=-1),
            "kv_cache": -state["kv_cache"].flip(0),
            "block_table": table,
            "sequence_lengths": lengths,
            "metadata": {
                "source": "fixture_derived_perturbation",
                "is_real_model_step": False,
            },
        }
    )


def case_specs(options):
    specs = []
    if not options.skip_synthetic:
        histories = (
            comma_ints(options.histories)
            if options.histories
            else (
                [63, 64, 265, 665, 714, 1028, 1354]
                if options.matrix == "full"
                else [265, 665, 1354]
            )
        )
        for history in histories:
            specs.append(
                {
                    "name": f"synthetic_b{options.beams}_l{history}_g{options.groups}",
                    "active": options.beams,
                    "bucket": options.bucket or options.beams,
                    "history": history,
                    "groups": options.groups,
                    "max_requests": options.max_requests,
                    "regime": "random",
                }
            )
        if options.matrix == "full" and not options.histories:
            for bucket, groups, max_requests, regime in (
                (512, 1, 1, "random"),
                (500, 2, 2, "random"),
                (500, 3, 1, "random"),
                (500, 1, 1, "uniform"),
                (500, 1, 1, "peaked"),
            ):
                specs.append(
                    {
                        "name": f"extended_b500_p{bucket}_g{groups}_{regime}",
                        "active": 500,
                        "bucket": bucket,
                        "history": 665,
                        "groups": groups,
                        "max_requests": max_requests,
                        "regime": regime,
                    }
                )
    for fixture_ordinal, filename in enumerate(options.fixture):
        path = Path(filename)
        # Do not retain every large fixture in memory while other cases execute.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        require(payload.get("schema_version") == 1, "unsupported fixture schema")
        members = payload.get("cases", [payload])
        for index, member in enumerate(members):
            active = member["q"].shape[0]
            specs.append(
                {
                    "name": f"fixture_{fixture_ordinal}_{path.stem}_{index}",
                    "fixture": str(path),
                    "fixture_ordinal": fixture_ordinal,
                    "fixture_index": index,
                    "active": active,
                    "bucket": options.bucket or active,
                    "max_requests": options.max_requests,
                }
            )
        del members, payload
    require(bool(specs), "no benchmark cases selected")
    return specs[: options.case_limit] if options.case_limit else specs


class DecodeBackend:
    """The existing full-KV FlashInfer Decode baseline, including LSE writes."""

    def __init__(self, *, batch_size, max_pages, use_cuda_graph, workspace, **unused):
        self.batch = batch_size
        self.max_pages = max_pages
        self.use_cuda_graph = use_cuda_graph
        self.wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            "HND",
            use_cuda_graph=use_cuda_graph,
            use_tensor_cores=False,
            paged_kv_indptr_buffer=torch.zeros(
                batch_size + 1, dtype=torch.int32, device="cuda"
            ),
            paged_kv_indices_buffer=torch.zeros(
                batch_size * max_pages, dtype=torch.int32, device="cuda"
            ),
            paged_kv_last_page_len_buffer=torch.zeros(
                batch_size, dtype=torch.int32, device="cuda"
            ),
        )
        require(
            self.wrapper.is_cuda_graph_enabled == use_cuda_graph,
            "wrong baseline graph mode",
        )
        self.signature = None

    def plan(self, table, lengths):
        counts = (lengths.long() + PAGE) // PAGE
        self.ptr = torch.cat(
            (torch.zeros(1, dtype=torch.int32), counts.cumsum(0).int())
        ).pin_memory()
        self.indices = table[
            torch.arange(table.shape[1])[None, :] < counts[:, None]
        ].pin_memory()
        self.last = (lengths % PAGE + 1).pin_memory()
        self.wrapper.plan(
            self.ptr,
            self.indices,
            self.last,
            HEADS,
            KV_HEADS,
            DIM,
            PAGE,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            non_blocking=True,
        )
        signature = tuple(self.wrapper._plan_info)
        if self.use_cuda_graph and self.signature is not None:
            require(signature == self.signature, "baseline graph plan changed geometry")
        self.signature = signature

    def run(self, q, kv_cache):
        return self.wrapper.run(q, kv_cache, return_lse=True)

    def describe(self):
        return {
            "class": type(self.wrapper).__name__,
            "use_cuda_graph": self.wrapper.is_cuda_graph_enabled,
            "use_tensor_cores": False,
            "plan_info": (
                None
                if self.signature is None
                else [int(value) for value in self.signature]
            ),
            "metadata_pointers": [
                getattr(self.wrapper, name).data_ptr()
                for name in (
                    "_paged_kv_indptr_buf",
                    "_paged_kv_indices_buf",
                    "_paged_kv_last_page_len_buf",
                )
            ],
        }


class CascadeBackend:
    """Observe the production run's merged LSE without reproducing its math."""

    def __init__(self, **kwargs):
        self.impl = BeamSearchCascadeAttention(**kwargs)
        require(
            self.impl.prefix.is_cuda_graph_enabled == kwargs["use_cuda_graph"]
            and self.impl.tail.is_cuda_graph_enabled == kwargs["use_cuda_graph"],
            "wrong Cascade graph mode",
        )
        original = self.impl.prefix.run

        def observe_prefix(*args, **options):
            pair = original(*args, **options)
            self.lse = pair[1]
            return pair

        self.impl.prefix.run = observe_prefix
        self.lse = None

    def plan(self, table, lengths):
        self.impl.plan(table, lengths)

    def run(self, q, kv_cache):
        output = self.impl.run(q, kv_cache)
        # merge_state_in_place mutates the same FP32 LSE storage.
        return output, self.lse

    def describe(self):
        return {
            "class": type(self.impl).__name__,
            "use_cuda_graph": self.impl.use_cuda_graph,
            # NumPy group boundaries make shared_tokens a NumPy integer.
            # Normalize report scalars only; preserve the implementation state.
            "shared_tokens": int(self.impl.shared_tokens),
            "slots": int(self.impl.slots),
            "plan_info": (
                None
                if self.impl._signature is None
                else [[int(value) for value in plan] for plan in self.impl._signature]
            ),
            "metadata_pointers": [
                getattr(wrapper, name).data_ptr()
                for wrapper in (self.impl.prefix, self.impl.tail)
                for name in (
                    "_paged_kv_indptr_buf",
                    "_paged_kv_indices_buf",
                    "_paged_kv_last_page_len_buf",
                )
            ]
            + [self.impl.prefix._qo_indptr_buf.data_ptr()],
        }


def candidate_factory(specification):
    module_name, separator, function = specification.rpartition(":")
    require(separator and module_name and function, "candidate must be module:factory")
    if module_name.endswith(".py"):
        path = Path(module_name)
        require(
            path.is_absolute() and path.is_file(),
            "candidate file must be an absolute file",
        )
        spec = importlib.util.spec_from_file_location(
            "ppu_cascade_benchmark_candidate", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, function)


class Runner:
    def __init__(
        self, factory, batch, max_pages, pages, max_requests, graph_safe, options
    ):
        self.q = torch.zeros(batch, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
        self.kv = torch.empty(
            pages, 2, KV_HEADS, PAGE, DIM, dtype=torch.bfloat16, device="cuda"
        )
        self.batch, self.max_pages = batch, max_pages
        self.workspace = torch.zeros(
            options.workspace_mib * 1024**2, dtype=torch.uint8, device="cuda"
        )
        self.backend = factory(
            batch_size=batch,
            heads=HEADS,
            kv_heads=KV_HEADS,
            head_dim=DIM,
            page_size=PAGE,
            max_requests=max_requests,
            max_pages=max_pages,
            dtype=torch.bfloat16,
            use_cuda_graph=graph_safe,
            workspace=self.workspace,
        )
        self.graph_safe = graph_safe
        self.graph = None
        self.input_pointers = (self.q.data_ptr(), self.kv.data_ptr())

    def update(self, state):
        # Caller completes old replay before replacing host/device metadata.
        torch.cuda.synchronize()
        active, pages = state["q"].shape[0], state["kv_cache"].shape[0]
        require(
            active <= self.batch and pages + 2 <= self.kv.shape[0],
            "runner capacity exceeded",
        )
        self.q.zero_()
        self.q[:active].copy_(state["q"])
        self.kv.fill_(19)
        self.kv[:pages].copy_(state["kv_cache"])
        scratch = self.kv.shape[0] - 2
        self.kv[scratch].zero_()
        self.table = torch.full(
            (self.batch, self.max_pages), scratch, dtype=torch.int32
        ).pin_memory()
        width = state["block_table"].shape[1]
        self.table[:active, :width].copy_(state["block_table"])
        self.lengths = torch.zeros(self.batch, dtype=torch.int32).pin_memory()
        self.lengths[:active].copy_(state["sequence_lengths"])
        torch.cuda.synchronize()
        sample = self.prepare()
        require(
            self.input_pointers == (self.q.data_ptr(), self.kv.data_ptr()),
            "input addresses changed",
        )
        if self.graph is not None and self.captured_metadata_pointers is not None:
            require(
                self.describe().get("metadata_pointers")
                == self.captured_metadata_pointers,
                "captured metadata addresses changed",
            )
        return sample

    def prepare(self):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        self.backend.plan(self.table, self.lengths)
        returned = time.perf_counter_ns()
        torch.cuda.synchronize()
        ready = time.perf_counter_ns()
        return {
            "host_return_us": (returned - start) / 1000,
            "device_ready_us": (ready - start) / 1000,
        }

    def run(self):
        pair = self.backend.run(self.q, self.kv)
        require(
            isinstance(pair, (tuple, list)) and len(pair) == 2,
            "backend must return output,LSE",
        )
        require(
            pair[0].shape == self.q.shape and pair[0].dtype == torch.bfloat16,
            "bad output contract",
        )
        require(
            pair[1].shape == self.q.shape[:2] and pair[1].dtype == torch.float32,
            "bad LSE contract",
        )
        return pair

    def capture(self, repeats, warmup):
        require(self.graph_safe, "cannot capture the non-graph backend")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self.run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.captured_metadata_pointers = self.describe().get("metadata_pointers")
        with torch.cuda.graph(self.graph, stream=stream):
            for _ in range(repeats):
                self.graph_output = self.run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

    def replay_pair(self):
        require(self.graph is not None, "no captured graph")
        self.graph.replay()
        torch.cuda.synchronize()
        return self.graph_output

    def describe(self):
        return (
            self.backend.describe()
            if hasattr(self.backend, "describe")
            else {
                "class": f"{type(self.backend).__module__}.{type(self.backend).__qualname__}",
                "use_cuda_graph": self.graph_safe,
            }
        )


def timing(runners, options):
    """Events bracket graph-of-N-runs; clones, update, oracle and plan excluded."""
    samples = {
        name: {"graph_us": [], "eager_wall_us": [], "prepare": []} for name in runners
    }
    for runner in runners.values():
        for _ in range(options.warmup):
            runner.graph.replay()
    torch.cuda.synchronize()
    names = list(runners)
    order = []
    for trial in range(options.trials):
        selected = names if trial % 2 == 0 else list(reversed(names))
        order.append(list(selected))
        for name in selected:
            runner = runners[name]
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            begin.record()
            runner.graph.replay()
            end.record()
            end.synchronize()
            samples[name]["graph_us"].append(
                begin.elapsed_time(end) * 1000 / options.graph_repeat
            )
        for name in selected:
            runner = runners[name]
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            for _ in range(options.eager_repeat):
                runner.backend.run(runner.q, runner.kv)
            torch.cuda.synchronize()
            samples[name]["eager_wall_us"].append(
                (time.perf_counter_ns() - start) / 1000 / options.eager_repeat
            )
            samples[name]["prepare"].append(runner.prepare())
    summaries = {}
    for name, values in samples.items():
        summaries[name] = {}
        for key in ("graph_us", "eager_wall_us"):
            data = values[key]
            summaries[name][key] = {
                "min": min(data),
                "median": statistics.median(data),
                "max": max(data),
                "mean": statistics.mean(data),
                "stdev": statistics.stdev(data) if len(data) > 1 else 0.0,
            }
    return {
        "raw_samples": samples,
        "summary": summaries,
        "trial_order": order,
        "graph_repeat": options.graph_repeat,
        "warmup": options.warmup,
        "eager_repeat": options.eager_repeat,
        "timed_graph_nodes": "backend.run only",
        "eager_wall_scope": "graph-safe wrapper run + host launch + synchronization; no plan",
        "hardware_peak": "NOT_PROVIDED: no verified PPU peak assumed",
    }


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


class CascadeKernelBenchmark(unittest.TestCase):
    def test_accuracy_graph_fidelity_and_latency(self):
        options = OPTIONS
        require(options is not None, "invoke the Bazel test entry point")
        require(get_device_type() == DeviceType.Ppu, "requires PPU; never skip as PASS")
        torch.set_num_threads(4)
        torch.manual_seed(options.seed)
        require(torch.cuda.is_available(), "PPU CUDA compatibility runtime unavailable")
        output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = output_dir / options.output_name
        self.report = {
            "schema_version": 1,
            "status": "RUNNING",
            "evidence": "RTP_MEASURED",
            "accuracy_acceptance": "NOT_ASSESSED: legacy tolerance is only regression",
            "options": vars(options),
            "cpu_threads": torch.get_num_threads(),
            "device": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "lse_unit": "log2",
            "history_semantics": "valid KV tokens = sequence_lengths + 1",
            "legacy_output_tolerance": {"atol": LEGACY_ATOL, "rtol": LEGACY_RTOL},
            "cases": [],
            "failures": [],
        }
        factories = {"off": DecodeBackend, "on": CascadeBackend}
        if options.candidate:
            factories["candidate"] = candidate_factory(options.candidate)
            if options.candidate.split(":")[0].endswith(".py"):
                self.report["candidate_sha256"] = digest_file(
                    options.candidate.rsplit(":", 1)[0]
                )
        self.write_report()
        try:
            for index, spec in enumerate(case_specs(options)):
                print(f"CASCADE_BENCH case={spec['name']} stage=begin", flush=True)
                self.run_case(spec, factories, options, index)
                gc.collect()
                self.write_report()
            self.report["status"] = (
                "REGRESSION_PASS" if not self.report["failures"] else "REGRESSION_FAIL"
            )
        except Exception:
            self.report["status"] = "ERROR"
            self.report["exception"] = traceback.format_exc()
            try:
                self.write_report()
            except Exception:
                # Preserve the first failure even if recording it also fails.
                print(
                    "CASCADE_BENCH secondary report-write failure:\n"
                    + traceback.format_exc(),
                    file=sys.stderr,
                    flush=True,
                )
            raise
        else:
            self.write_report()
        self.assertFalse(
            self.report["failures"],
            "see cascade benchmark JSON for regression failures",
        )

    def write_report(self):
        self.report_path.write_text(
            json.dumps(self.report, indent=2, allow_nan=False) + "\n"
        )

    def gate_close(self, actual, expected, label):
        try:
            torch.testing.assert_close(
                actual, expected, atol=LEGACY_ATOL, rtol=LEGACY_RTOL, check_dtype=False
            )
        except AssertionError as error:
            self.report["failures"].append({"check": label, "error": str(error)[:2000]})

    def gate_exact(self, actual, expected, label):
        for field, a, b in zip(("output", "lse_log2"), actual, expected):
            if not torch.equal(
                a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
            ):
                self.report["failures"].append(
                    {"check": label + ":" + field, "error": "bitwise mismatch"}
                )

    def run_case(self, spec, factories, options, index):
        if "fixture" in spec:
            payload = torch.load(spec["fixture"], map_location="cpu", weights_only=True)
            initial = validate_state(
                payload.get("cases", [payload])[spec["fixture_index"]]
            )
            changed = changed_fixture(initial)
            spec["fixture_sha256"] = digest_file(spec["fixture"])
        else:
            initial = synthetic_state(
                spec["active"],
                spec["history"],
                spec["groups"],
                options.seed + 2 * index,
                spec["regime"],
            )
            changed_history = (
                spec["history"] + 1
                if spec["history"] % PAGE == PAGE - 1
                else max(0, spec["history"] - 1)
            )
            changed = synthetic_state(
                spec["active"],
                changed_history,
                spec["groups"],
                options.seed + 2 * index + 1,
                spec["regime"],
            )
            # Remap all physical pages; gathered logical data remain the new data.
            count = changed["kv_cache"].shape[0]
            changed["kv_cache"] = changed["kv_cache"].flip(0)
            changed["block_table"] = count - 1 - changed["block_table"]
        require(
            spec["bucket"] >= spec["active"], "fixture active exceeds selected bucket"
        )
        states = (initial, changed)
        width = max(state["block_table"].shape[1] for state in states)
        pages = max(state["kv_cache"].shape[0] for state in states) + 2
        result = {"spec": spec, "states": [], "timing": {}, "graph_capture_count": 1}
        self.report["cases"].append(result)
        runners = {
            name: Runner(
                factory,
                spec["bucket"],
                width,
                pages,
                spec["max_requests"],
                True,
                options,
            )
            for name, factory in factories.items()
        }
        for name, runner in runners.items():
            runner.update(initial)
            result.setdefault("backend_validation", {})[name + ":before_capture"] = (
                validate_backend_for_test(runner)
            )
            runner.capture(options.graph_repeat, options.warmup)
            result.setdefault("capture", {})[name] = runner.describe()
        for state_index, state in enumerate(states):
            state_result = {
                "state_index": state_index,
                "metadata": state.get("metadata", {}),
                "length_min": int(state["sequence_lengths"].min()),
                "length_max": int(state["sequence_lengths"].max()),
                "comparisons": {},
                "reference": {},
                "backends": {},
                "backend_validation": {},
            }
            result["states"].append(state_result)
            outputs = {}
            # Same bucket in all three modes separates graph planning from padding.
            for name, factory in factories.items():
                plain = Runner(
                    factory,
                    spec["bucket"],
                    width,
                    pages,
                    spec["max_requests"],
                    False,
                    options,
                )
                plain.update(state)
                state_result["backend_validation"][name + ":non_graph_eager"] = (
                    validate_backend_for_test(plain)
                )
                for _ in range(options.warmup):
                    plain.run()
                outputs[name + ":non_graph_eager"] = cpu_pair(
                    plain.run(), spec["active"]
                )
                state_result["backends"][name + ":non_graph_eager"] = plain.describe()
                del plain
                runner = runners[name]
                state_result["backends"][name + ":prepare"] = runner.update(state)
                state_result["backend_validation"][name + ":graph_safe_eager"] = (
                    validate_backend_for_test(runner)
                )
                outputs[name + ":graph_safe_eager"] = cpu_pair(
                    runner.run(), spec["active"]
                )
                outputs[name + ":replay"] = cpu_pair(
                    runner.replay_pair(), spec["active"]
                )
                again = cpu_pair(runner.replay_pair(), spec["active"])
                self.gate_exact(
                    outputs[name + ":replay"],
                    again,
                    spec["name"] + f":state{state_index}:{name}:repeat",
                )
                self.gate_exact(
                    outputs[name + ":replay"],
                    outputs[name + ":graph_safe_eager"],
                    spec["name"] + f":state{state_index}:{name}:graph_fidelity",
                )
                state_result["backends"][name + ":graph_safe"] = runner.describe()
                require(
                    bool((runner.kv[-1] == 19).all()),
                    "unreferenced KV guard page modified",
                )
            for mode in ("non_graph_eager", "graph_safe_eager", "replay"):
                baseline = outputs["off:" + mode]
                if state_index == 0 and "original_output" in initial:
                    captured = initial["original_output"]
                    require(
                        captured.shape == baseline[0].shape
                        and captured.dtype == torch.bfloat16,
                        "invalid original_output in captured fixture",
                    )
                    state_result["comparisons"]["off_vs_captured_original:" + mode] = (
                        metric(baseline[0], captured)
                    )
                    self.gate_close(
                        baseline[0],
                        captured,
                        spec["name"] + ":captured_original:" + mode,
                    )
                for name in factories:
                    if name == "off":
                        continue
                    key = name + "_vs_off:" + mode
                    state_result["comparisons"][key] = pair_metric(
                        outputs[name + ":" + mode], baseline
                    )
                    self.gate_close(
                        outputs[name + ":" + mode][0],
                        baseline[0],
                        spec["name"] + ":" + key,
                    )
            for name in factories:
                state_result["comparisons"][name + ":graph_plan_vs_non_graph"] = (
                    pair_metric(
                        outputs[name + ":graph_safe_eager"],
                        outputs[name + ":non_graph_eager"],
                    )
                )
                state_result["comparisons"][name + ":replay_vs_graph_safe_eager"] = (
                    pair_metric(
                        outputs[name + ":replay"], outputs[name + ":graph_safe_eager"]
                    )
                )
            print(
                f"CASCADE_BENCH case={spec['name']} state={state_index} stage=oracle",
                flush=True,
            )
            oracle = cpu_reference(state, options)
            state_result["reference_scope"] = {
                "rows": oracle["rows"],
                "heads": oracle["heads"],
                "requested_rows": options.reference_rows,
                "selected_rows": len(oracle["rows"]),
                "math": "CPU FP64 full QK/softmax/PV; LSE=logsumexp/ln(2)",
            }
            for key, pair in outputs.items():
                state_result["reference"][key] = against_reference(pair, oracle)
                self.gate_close(
                    pair[0][oracle["rows"]][:, oracle["heads"]],
                    oracle["output"],
                    spec["name"] + ":cpu_reference:" + key,
                )
            if options.diagnose_stages:
                print(
                    f"CASCADE_BENCH case={spec['name']} state={state_index} stage=diagnose",
                    flush=True,
                )
                trace_path = None
                if options.diagnostic_timeline:
                    trace_name = re.sub(r"[^A-Za-z0-9_.-]", "_", spec["name"])
                    trace_path = (
                        self.report_path.parent
                        / f"cascade_diagnostic_{trace_name}_state{state_index}.json"
                    )
                diagnosis, diagnosed_merge = diagnose_stages(
                    state, runners["on"], outputs, options, trace_path
                )
                state_result["stage_diagnostics"] = diagnosis
                self.gate_exact(
                    diagnosed_merge,
                    outputs["on:graph_safe_eager"],
                    spec["name"] + f":state{state_index}:diagnostic_pipeline",
                )
            # All copies and FP64 work finish before any event measurement.
            self.write_report()
            del outputs, oracle
        # Restore the actual initial fixture for timing, not its perturbation.
        for name, runner in runners.items():
            runner.update(initial)
            result["backend_validation"][name + ":before_timing"] = (
                validate_backend_for_test(runner)
            )
        print(f"CASCADE_BENCH case={spec['name']} stage=timing", flush=True)
        result["timing"] = timing(runners, options)
        result["timing_state_index"] = 0
        print(f"CASCADE_BENCH case={spec['name']} stage=done", flush=True)


if __name__ == "__main__":
    OPTIONS, unittest_args = parse_options()
    unittest.main(argv=[sys.argv[0]] + unittest_args)
