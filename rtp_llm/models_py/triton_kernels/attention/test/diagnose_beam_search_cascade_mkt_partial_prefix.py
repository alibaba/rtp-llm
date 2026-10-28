"""External test draft: captured 704/714 prefix diagnosis, no production wiring.

This module is not a test runner. A Bazel-owned test supplies already validated
fixtures and the live module objects in Bindings; it must not allocate an arena
first. Nothing here imports a second copy of the real-fixture test module.
No timing, Graph experiment, scheduler claim, tolerance change, or new kernel.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable
from unittest import mock

import torch

ATOL, RTOL, L2_LIMIT = 0.005, 0.03, 0.01
ORACLE_ATOL, ORACLE_RTOL = 2e-5, 2e-5
HEADS, KV_HEADS, DIM, PAGE, ROWS = 16, 8, 128, 16, 500


@dataclass(frozen=True)
class Bindings:
    """Pass existing objects from the current test module, before any patch."""

    cascade_module: object
    generic_tail: object
    best_tail: object  # Existing layout_w1 adapter, not shared-K or new math.
    tail_recorder_factory: Callable
    prefix_reference: Callable
    full_reference: Callable
    loaded_native_modules: Callable


class PartialPrefixDiagnosticFailure(AssertionError):
    """All safe diagnostic arms were emitted before this numerical failure."""


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _hash(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _same_bits(left, right):
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
        )
    )


def _finite(tensor):
    return {
        "numel": tensor.numel(),
        "nonfinite": int((~torch.isfinite(tensor)).sum()),
        "nan": int(torch.isnan(tensor).sum()),
        "inf": int(torch.isinf(tensor).sum()),
    }


def _compare(actual, expected, *, oracle=False):
    """Aggregate only: never report Q/K/V/output values or token data."""
    atol, rtol, l2 = (
        (ORACLE_ATOL, ORACLE_RTOL, None) if oracle else (ATOL, RTOL, L2_LIMIT)
    )
    actual = actual.to(expected.dtype)
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    close = torch.isclose(actual, expected, atol=atol, rtol=rtol) & finite
    mismatches = int((~close).sum())
    metrics = None
    if bool(finite.all()):
        difference = actual.double() - expected.double()
        metrics = {
            "max_abs": float(difference.abs().max()),
            "relative_l2": float(
                difference.norm() / expected.double().norm().clamp_min(1e-30)
            ),
            "rms": float(difference.square().mean().sqrt()),
        }
    passed = mismatches == 0 and (
        l2 is None or metrics is not None and metrics["relative_l2"] <= l2
    )
    return {
        "thresholds": {"atol": atol, "rtol": rtol, "relative_l2": l2},
        "thresholds_pass": passed,
        "mismatch_count": mismatches,
        "metrics": metrics,
        "actual_finite": _finite(actual),
        "expected_finite": _finite(expected),
    }


def _unpack_cpu(packed):
    """Exact inverse of RealCase._unpack_pages; BF16 vector layout, not plain V."""
    _require(packed.device.type == "cpu", "poison payload must be CPU-owned")
    count = packed.shape[0]
    k = packed[:, 0].contiguous().view(count, KV_HEADS, DIM // 8, PAGE, 8)
    k = k.permute(0, 1, 3, 2, 4).reshape(count, KV_HEADS, PAGE, DIM)
    v = packed[:, 1].contiguous().view(count, KV_HEADS, PAGE // 8, DIM, 8)
    v = v.permute(0, 1, 2, 4, 3).reshape(count, KV_HEADS, PAGE, DIM)
    return k, v


def _current_token_hashes(packed, tail_logical):
    k, v = _unpack_cpu(packed)
    return {
        "K714": _hash(k[tail_logical.long(), :, 10]),
        "V714": _hash(v[tail_logical.long(), :, 10]),
    }


def _poison_payload(clean, tail_logical, first_token, components):
    """Mutate only selected packed positions in an independent CPU clone."""
    _require(first_token in (714, 715), "unsupported poison interval")
    _require(components in ("K", "V", "KV"), "unsupported poison components")
    _require(
        clean.dtype == torch.bfloat16 and tuple(clean.shape[1:]) == (2, 8, 16, 128),
        "bad packed layout",
    )
    ids = tail_logical.long().unique()
    result = clean.clone()

    def change(target, source=None):
        # These .view calls retain the stride across K/V planes exactly as the
        # production native entry does; no contiguous clone hides the writes.
        k = target[:, 0].view(target.shape[0], KV_HEADS, DIM // 8, PAGE, 8)
        v = target[:, 1].view(target.shape[0], KV_HEADS, PAGE // 8, DIM, 8)
        sk = None if source is None else source[:, 0].view_as(k)
        sv = None if source is None else source[:, 1].view_as(v)
        for token in range(first_token, 720):
            offset = token - 704
            if "K" in components:
                k[ids, :, :, offset, :] = (
                    float("nan") if sk is None else sk[ids, :, :, offset, :]
                )
            if "V" in components:
                v[ids, :, offset // 8, :, offset % 8] = (
                    float("nan")
                    if sv is None
                    else sv[ids, :, offset // 8, :, offset % 8]
                )

    change(result)
    restored = result.clone()
    change(restored, clean)
    _require(
        _same_bits(restored, clean), "poison changed bytes outside its declared slots"
    )
    k, v = _unpack_cpu(result)
    offset = first_token - 704
    for component, data in (("K", k), ("V", v)):
        if component in components:
            _require(
                bool(torch.isnan(data[ids, :, offset:16]).all()),
                "poison did not reach all intended packed slots",
            )
    if first_token == 715:
        _require(
            _current_token_hashes(result, ids) == _current_token_hashes(clean, ids),
            "full attention token714 was poisoned",
        )
    return result


def _validate_lineage(fixture, lineage):
    """Only a bounded, caller-declared captured-instance claim is accepted."""
    m, t = fixture.manifest, fixture.tensors
    required = {
        "kind": "explicit_captured_fixture_lineage_not_production_metadata",
        "manifest_sha256": fixture.manifest_sha256,
        "request_ordinal": m["request_ordinal"],
        "request_boundaries": [0, ROWS],
        "original_prompt_length": 714,
        "valid_for_captured_instance_only": True,
    }
    _require(lineage == required, "missing or ambiguous explicit fixture lineage")
    _require(
        m["rows"] == ROWS and m["request_boundaries"] == [0, ROWS],
        "wrong request geometry",
    )
    _require(
        bool((t["history"] == 714).all()) and bool((t["seq_lens"] == 715).all()),
        "wrong history/current-token boundary",
    )
    _require(
        all(_hash(value) == m["tensors"][name]["sha256"] for name, value in t.items()),
        "CPU fixture tensors changed after fixture load",
    )
    table, logical, ids = (
        t["block_table"],
        t["logical_block_table"],
        t["physical_page_ids"],
    )
    _require(
        torch.equal(table[:, :44], table[:1, :44].expand(ROWS, -1)),
        "704 control is not physically shared",
    )
    tail = logical[:, 44].long()
    _require(
        tail.unique().numel() == ROWS,
        "this bounded fixture expects 500 separate tail pages",
    )
    _require(
        torch.equal(ids[tail].to(table.dtype), table[:, 44]),
        "tail physical/logical mapping differs",
    )
    k, v = _unpack_cpu(t["kv_pages"])
    _require(
        _same_bits(k, fixture.key) and _same_bits(v, fixture.value),
        "canonical oracle pages differ from payload",
    )
    history_hashes = {}
    for component, data in (("K", k), ("V", v)):
        values = data[tail, :, :10]
        _require(
            _same_bits(values, values[:1].expand_as(values)),
            "captured partial prompt is not bitwise identical",
        )
        history_hashes[component] = _hash(values)
    return tail, {
        "declaration": lineage,
        "captured_partial_history_sha256": history_hashes,
        "production_lineage_contract": "MISSING_NOT_IMPLEMENTED",
        "scope": "caller-declared captured single request; equality validates this instance only",
    }


def _metadata_hashes(op):
    return {
        name: _hash(getattr(op, name))
        for name in (
            "table",
            "table_host",
            "lengths",
            "lengths_host",
            "prefix_qptr",
            "qptr_host",
            "prefix_table",
            "prefix_table_host",
            "prefix_indptr",
        )
    }


@contextmanager
def _manual_cut(ab, cut):
    _require(cut in (704, 714), "unsupported manual prefix")
    op = ab.on_op
    torch.cuda.synchronize()
    _require(
        op.BEAMS_PER_TILE == 128 and op.splits == 1 and op.tiles == 4,
        "different native constructor geometry",
    )
    _require(
        op.qptr_host.tolist() == [0, 128, 256, 384, 500],
        "unexpected captured request tiles",
    )
    old = {name: getattr(op, name).clone() for name in ("cuts_host", "row_cuts_host")}
    _require(
        bool((old["cuts_host"] == 704).all())
        and bool((old["row_cuts_host"] == 704).all()),
        "unmodified planner must produce704",
    )
    metadata, shared = _metadata_hashes(op), op.shared_tokens
    try:
        op.cuts_host.fill_(cut)
        op.row_cuts_host.fill_(cut)
        op.cuts.copy_(op.cuts_host)
        op.row_cuts.copy_(op.row_cuts_host)
        op.shared_tokens = ROWS * cut
        yield
    finally:
        for name, value in old.items():
            getattr(op, name).copy_(value)
        op.cuts.copy_(op.cuts_host)
        op.row_cuts.copy_(op.row_cuts_host)
        op.shared_tokens = shared
        torch.cuda.synchronize()
        _require(
            metadata == _metadata_hashes(op), "manual cut changed unrelated metadata"
        )
        _require(
            _same_bits(op.cuts.cpu(), old["cuts_host"])
            and _same_bits(op.row_cuts.cpu(), old["row_cuts_host"]),
            "cut restoration failed",
        )


@contextmanager
def _installed_payload(case, payload, clean_hash):
    """Only the helper-owned arena is modified. Restore all live pages on exit."""
    ids = case.physical_ids.to(device=case.q.device, dtype=torch.long)
    _require(
        _hash(case.kv.index_select(0, ids)) == clean_hash,
        "arena was not clean before poison",
    )
    try:
        uploaded = payload.to(device=case.q.device)
        case.kv.index_copy_(0, ids, uploaded)
        torch.cuda.synchronize()
        _require(
            _hash(case.kv.index_select(0, ids)) == _hash(payload),
            "installed poison payload differs",
        )
        yield
    finally:
        case.kv.index_copy_(0, ids, case.packed)
        torch.cuda.synchronize()
        _require(
            _hash(case.kv.index_select(0, ids)) == clean_hash,
            "clean live-page restoration failed",
        )


def _native_call_metadata(
    args, kwargs, returned, ab, expected_cut, bindings, expected_sha
):
    op, case = ab.on_op, ab.case
    output, lse = returned
    _require(
        len(args) == 8 and args[0].data_ptr() == case.q.data_ptr(),
        "unexpected native call signature/Q",
    )
    _require(
        args[6] == 128 and args[7] == op.max_pages * PAGE, "native geometry changed"
    )
    _require(
        kwargs.get("causal") is False and kwargs.get("return_lse") is True,
        "native mode changed",
    )
    _require(
        kwargs["out"].data_ptr() == op.prefix_output.data_ptr() == output.data_ptr(),
        "native did not return the provided FP32 storage",
    )
    _require(
        output.dtype == torch.float32 and output.shape == case.q.shape,
        "native FP32 output contract failed",
    )
    _require(
        lse.dtype == torch.float32
        and tuple(lse.shape) == (HEADS, ROWS)
        and lse.is_contiguous(),
        "native natural-log LSE layout changed",
    )
    _require(
        kwargs["seqlen_k"].data_ptr() == op.cuts.data_ptr(),
        "native cut pointer differs",
    )
    _require(
        kwargs["block_table"].data_ptr() == op.prefix_table.data_ptr(),
        "native prefix table pointer differs",
    )
    _require(
        args[3].data_ptr() == op.prefix_qptr.data_ptr()
        and args[4].data_ptr() == op.prefix_indptr.data_ptr(),
        "native qptr/indptr differ",
    )
    _require(
        args[5].data_ptr() == op.prefix_table.data_ptr(), "native flat table differs"
    )
    _require(
        args[1].data_ptr() == case.kv.data_ptr(), "native K is not the helper-owned KV"
    )
    _require(
        args[2].data_ptr() == case.kv[:, 1].data_ptr(),
        "native V is not the helper-owned KV",
    )
    _require(
        bool((kwargs["seqlen_k"].cpu() == expected_cut).all()),
        "actual native cut differs",
    )
    _require(bool((op.row_cuts.cpu() == expected_cut).all()), "actual row cut differs")
    modules = bindings.loaded_native_modules()
    _require(
        len(modules) == 1 and modules[0]["sha256"] == expected_sha,
        "actual mapped native library differs",
    )
    return {
        "actual_cuts": kwargs["seqlen_k"].cpu().tolist(),
        "actual_qptr": args[3].cpu().tolist(),
        "actual_indptr_sha256": _hash(args[4]),
        "actual_prefix_table_sha256": _hash(kwargs["block_table"]),
        "max_seqlen_q": args[6],
        "max_seqlen_k_capacity": args[7],
        "returned_O_dtype": str(output.dtype),
        "returned_LSE_dtype": str(lse.dtype),
        "prefix_O_finite": _finite(output),
        "prefix_LSE_finite": _finite(lse),
        "prefix_O_sha256": _hash(output),
        "prefix_LSE_sha256": _hash(lse),
        "actual_K_view": {
            "shape": list(args[1].shape),
            "stride": list(args[1].stride()),
            "storage_offset": args[1].storage_offset(),
        },
        "actual_V_view": {
            "shape": list(args[2].shape),
            "stride": list(args[2].stride()),
            "storage_offset": args[2].storage_offset(),
        },
        "returned_storage_is_provided_FP32_workspace": True,
        "test_side_output_cast": False,
        "loaded_native_modules": modules,
        "actual_prefix_route": "aiter_native_API",
        "actual_CK_mangled_dispatch": "NOT_COLLECTED_BY_THIS_DIAGNOSTIC",
    }


def _run_full(ab, kernel, cut, bindings, expected_sha, expected_warps):
    import aiter

    native = aiter.mha_batch_prefill_func
    calls = []
    recorder = bindings.tail_recorder_factory(kernel)

    def observe(*args, **kwargs):
        result = native(*args, **kwargs)
        meta = _native_call_metadata(
            args, kwargs, result, ab, cut, bindings, expected_sha
        )
        calls.append(
            (native, args, dict(kwargs), result[0].clone(), result[1].clone(), meta)
        )
        return result

    ab.on_op.prefix_output.fill_(float("nan"))
    ab.on_op.output.fill_(float("nan"))
    with mock.patch.object(
        aiter, "mha_batch_prefill_func", side_effect=observe
    ), mock.patch.object(bindings.cascade_module, "_private_tail_merge", recorder):
        output = ab.on().clone()
    _require(len(calls) == 1, "expected exactly one native prefix invocation")
    params = recorder.parameters()
    _require(
        recorder.kernel is kernel and params["num_warps"] == expected_warps,
        "actual tail route/warp differs",
    )
    _require(
        params["NATIVE_PREFIX"] and params["SPLITS"] == 1,
        "tail did not consume native prefix",
    )
    _require(
        params["KV"].data_ptr() == ab.case.kv.data_ptr(),
        "tail uses a different KV view",
    )
    _require(
        bool((params["CUTS"].cpu() == cut).all())
        and bool((params["LENGTHS"].cpu() == 715).all()),
        "actual tail interval differs",
    )
    _require(params["PART_OUT"].dtype == torch.float32, "tail prefix precision differs")
    meta = {
        "native": calls[0][5],
        "tail": {
            "kernel": kernel.__name__,
            "is_original_generic_kernel": kernel is bindings.generic_tail,
            "is_existing_best_layout_w1": kernel is bindings.best_tail,
            "num_warps": params["num_warps"],
            "grid": list(recorder.records[0][0]),
            "tail_begin": cut,
            "tail_end_exclusive": 715,
            "binding": "run resolves module-global kernel; patch scoped to this call",
        },
    }
    return output, calls[0], meta


def _run_prefix(invocation, ab, cut, bindings, expected_sha):
    native, args, kwargs = invocation[:3]
    ab.on_op.prefix_output.fill_(float("nan"))
    result = native(*args, **kwargs)
    meta = _native_call_metadata(args, kwargs, result, ab, cut, bindings, expected_sha)
    return result[0].clone(), result[1].clone(), meta


def run_partial_prefix_diagnostic(
    fixtures,
    *,
    lineage_by_fixture,
    expected_native_sha256,
    emit_record,
    case_factory,
    ab_factory,
    bindings,
):
    """Emit every safe arm for all 3 layers, then fail if any numeric/NaN gate fails.

    Caller owns Bazel lifecycle/lock. It supplies CPU fixtures, not an arena or
    an existing GPU case. Exactly one helper-owned full backing arena is used.
    """
    _require(
        os.getenv("MKT_CASCADE_NATIVE_FP32_O") == "1",
        "explicit native FP32 O is required",
    )
    _require(
        os.getenv("MKT_CASCADE_TAIL_VARIANT", "production") == "production",
        "diagnostic owns its scoped tail selection",
    )
    _require("AITER_AOT_IMPORT" not in os.environ, "AITER_AOT_IMPORT must be unset")
    oracle_settings = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    _require(
        oracle_settings == {"float32_matmul_precision": "highest", "allow_tf32": False},
        "MktCascadeTestCase highest/TF32-disabled oracle settings are required",
    )
    _require(
        len(expected_native_sha256) == 64, "explicit expected library SHA is required"
    )
    _require(
        sorted(f.manifest["layer_idx"] for f in fixtures) == [0, 13, 27]
        and len(fixtures) == 3,
        "exactly the three captured layers are required",
    )
    _require(
        all(f.manifest["request_ordinal"] == 0 for f in fixtures),
        "only the captured single request is supported",
    )
    _require(
        set(lineage_by_fixture) == {f.name for f in fixtures},
        "explicit lineage must cover every fixture",
    )
    _require(
        bindings.generic_tail is not bindings.best_tail,
        "generic and best tail objects must differ",
    )
    backing = {f.manifest["paged_kv_metadata"]["storage_nbytes"] for f in fixtures}
    _require(backing == {34358689792}, "unexpected captured arena size")
    validated = {
        f.name: _validate_lineage(f, lineage_by_fixture[f.name]) for f in fixtures
    }
    failures, records = [], []
    original_tail = bindings.cascade_module._private_tail_merge

    def emit(name, payload):
        full = {
            "evidence": "RTP_MEASURED",
            "classification": "partial_prefix_captured_fixture_diagnostic",
            "production_partial_lineage": "NOT_IMPLEMENTED",
            "implementation_acceptance": "NOT_CLAIMED",
            "timing": "NOT_RUN",
            "oracle_settings": oracle_settings,
            **payload,
        }
        emit_record(name, full)
        records.append(name)
        if (
            payload.get("diagnostic_gate") == "FAIL"
            and name != "partial_prefix_diagnostic_summary"
        ):
            failures.append(name)

    def arm_report(
        case,
        name,
        output,
        expected32,
        expected64,
        meta,
        *,
        clean=None,
        lse=None,
        lse32=None,
        lse64=None,
        clean_lse=None,
        poison=None,
        extra_controls=None,
    ):
        checks = {
            "vs_FP32": _compare(output, expected32),
            "vs_FP64": _compare(output, expected64),
        }
        passed = all(v["thresholds_pass"] for v in checks.values())
        gate_parts = {"O_existing_fixed_tolerance_gate": passed}
        hashes = {"output": _hash(output)}
        if lse is not None:
            checks["LSE_vs_FP32"] = _compare(lse.T, lse32, oracle=True)
            checks["LSE_vs_FP64"] = _compare(lse.T, lse64, oracle=True)
            passed &= all(
                checks[k]["thresholds_pass"] for k in ("LSE_vs_FP32", "LSE_vs_FP64")
            )
            gate_parts["LSE_new_strict_diagnostic_gate_not_existing_final_O_gate"] = (
                all(
                    checks[k]["thresholds_pass"] for k in ("LSE_vs_FP32", "LSE_vs_FP64")
                )
            )
            hashes["natural_log_LSE"] = _hash(lse)
        exact = None
        if clean is not None:
            exact = {"output_bitwise_equal_clean": _same_bits(output, clean)}
            if lse is not None:
                exact["LSE_bitwise_equal_clean"] = _same_bits(lse, clean_lse)
            passed &= all(exact.values())
            gate_parts["poison_vs_clean_bitwise_gate"] = all(exact.values())
        if extra_controls:
            passed &= all(extra_controls.values())
        emit(
            case.fixture.name + "_" + name,
            {
                **case.describe(),
                "arm": name,
                "diagnostic_gate": "PASS" if passed else "FAIL",
                "comparisons": checks,
                "diagnostic_gate_components": gate_parts,
                "extra_controls": extra_controls,
                "exact_clean_controls": exact,
                "output_sha256": hashes,
                "execution": meta,
                "poison": poison,
                "lineage": validated[case.fixture.name][1],
            },
        )

    arena = torch.empty(next(iter(backing)) // 2, device="cuda", dtype=torch.bfloat16)
    try:
        for fixture in fixtures:
            case = case_factory(fixture, arena)
            ab = ab_factory(case)
            _require(
                ab.prefix_path() == "aiter_native",
                "fixture did not select native prefix",
            )
            _require(
                ab.on_op.prefix_output.dtype == torch.float32,
                "base helper did not allocate FP32 workspace",
            )
            tail_logical, _ = validated[fixture.name]
            clean_cpu = fixture.tensors["kv_pages"]
            clean_hash = _hash(clean_cpu)
            canonical_hashes = (_hash(case.k), _hash(case.v), _hash(case.q))
            full32 = bindings.full_reference(case, dtype=torch.float32)
            full64 = bindings.full_reference(case, dtype=torch.float64)
            full_oracle = _compare(full32, full64, oracle=True)
            emit(
                fixture.name + "_full_oracle",
                {
                    "oracle": full_oracle,
                    "diagnostic_gate": (
                        "PASS" if full_oracle["thresholds_pass"] else "FAIL"
                    ),
                },
            )
            _require(
                full_oracle["thresholds_pass"],
                "independent full FP32/FP64 oracle disagrees",
            )
            with _manual_cut(ab, 704):
                full704, invocation, meta704 = _run_full(
                    ab, bindings.best_tail, 704, bindings, expected_native_sha256, 1
                )
                arm_report(
                    case,
                    "full_clean704_best_layout_w1",
                    full704,
                    full32,
                    full64,
                    meta704,
                )
            for cut in (704, 714):
                cuts = torch.full((ROWS,), cut, dtype=torch.int32)
                prefix32, lse32 = bindings.prefix_reference(case, cuts, torch.float32)
                prefix64, lse64 = bindings.prefix_reference(case, cuts, torch.float64)
                oracle_o = _compare(prefix32, prefix64, oracle=True)
                oracle_lse = _compare(lse32, lse64, oracle=True)
                valid = oracle_o["thresholds_pass"] and oracle_lse["thresholds_pass"]
                emit(
                    fixture.name + f"_prefix{cut}_oracle",
                    {
                        "O": oracle_o,
                        "natural_log_LSE": oracle_lse,
                        "diagnostic_gate": "PASS" if valid else "FAIL",
                    },
                )
                _require(valid, "independent prefix FP32/FP64 oracle disagrees")
                with _manual_cut(ab, cut):
                    clean_o, clean_lse, native_meta = _run_prefix(
                        invocation, ab, cut, bindings, expected_native_sha256
                    )
                    controls = (
                        {
                            "prefix_only704_matches_actual_call_O": _same_bits(
                                clean_o, invocation[3]
                            ),
                            "prefix_only704_matches_actual_call_LSE": _same_bits(
                                clean_lse, invocation[4]
                            ),
                        }
                        if cut == 704
                        else None
                    )
                    arm_report(
                        case,
                        f"prefix{cut}_clean",
                        clean_o,
                        prefix32,
                        prefix64,
                        native_meta,
                        lse=clean_lse,
                        lse32=lse32,
                        lse64=lse64,
                        extra_controls=controls,
                    )
                    for components in ("K", "V", "KV"):
                        payload = _poison_payload(
                            clean_cpu, tail_logical, 714, components
                        )
                        poison = {
                            "components": components,
                            "tokens": [714, 720],
                            "scope": "prefix_only",
                            "current714_is_invalid_only_in_this_prefix_subproblem": True,
                            "payload_sha256": _hash(payload),
                            "clean_payload_sha256": clean_hash,
                            "tail_or_full_attention_is_not_called_with_this_payload": True,
                        }
                        with _installed_payload(case, payload, clean_hash):
                            o, lse, native_meta = _run_prefix(
                                invocation, ab, cut, bindings, expected_native_sha256
                            )
                            arm_report(
                                case,
                                f"prefix{cut}_invalid_{components}",
                                o,
                                prefix32,
                                prefix64,
                                native_meta,
                                clean=clean_o,
                                lse=lse,
                                lse32=lse32,
                                lse64=lse64,
                                clean_lse=clean_lse,
                                poison=poison,
                            )
                        del payload, o, lse
                del prefix32, prefix64, lse32, lse64, clean_o, clean_lse
            with _manual_cut(ab, 714):
                clean_full, full_call, full_meta = _run_full(
                    ab, bindings.generic_tail, 714, bindings, expected_native_sha256, 2
                )
                del full_call
                _require(
                    full_meta["tail"]["is_original_generic_kernel"],
                    "full714 did not use the original generic tail",
                )
                arm_report(
                    case, "full_clean714_generic", clean_full, full32, full64, full_meta
                )
                for components in ("K", "V", "KV"):
                    payload = _poison_payload(clean_cpu, tail_logical, 715, components)
                    poison = {
                        "components": components,
                        "tokens": [715, 720],
                        "scope": "full_attention",
                        "valid_current714_preserved": True,
                        "current714_sha256": _current_token_hashes(
                            payload, tail_logical
                        ),
                        "payload_sha256": _hash(payload),
                        "clean_payload_sha256": clean_hash,
                    }
                    with _installed_payload(case, payload, clean_hash):
                        output, full_call, meta = _run_full(
                            ab,
                            bindings.generic_tail,
                            714,
                            bindings,
                            expected_native_sha256,
                            2,
                        )
                        del full_call
                        arm_report(
                            case,
                            f"full714_invalid_{components}_generic",
                            output,
                            full32,
                            full64,
                            meta,
                            clean=clean_full,
                            poison=poison,
                        )
                    del payload, output
            _require(
                canonical_hashes == (_hash(case.k), _hash(case.v), _hash(case.q)),
                "clean oracle data/Q mutated",
            )
            _require(_hash(clean_cpu) == clean_hash, "CPU fixture payload mutated")
            _require(
                _hash(
                    case.kv.index_select(
                        0, case.physical_ids.to(device=case.q.device, dtype=torch.long)
                    )
                )
                == clean_hash,
                "final private live pages not restored",
            )
            _require(
                bindings.cascade_module._private_tail_merge is original_tail,
                "global tail patch leaked",
            )
            emit(
                fixture.name + "_restoration",
                {
                    "diagnostic_gate": "PASS",
                    "live_pages_sha256": clean_hash,
                    "canonical_oracle_and_Q_unchanged": True,
                    "metadata_and_tail_symbol_restored": True,
                    "independent_helper_owned_arena": True,
                },
            )
            del ab, case, invocation, full704, clean_full, full32, full64
    except Exception as error:
        emit(
            "partial_prefix_integrity_or_execution_error",
            {
                "diagnostic_gate": "FAIL",
                "error_type": type(error).__name__,
                "message": "Stopped; inspect the test traceback. No tensor values are emitted here.",
                "completed_records": list(records),
            },
        )
        raise
    finally:
        # Context managers restore before reaching here; caller-held fixtures
        # and external arenas have never been written by this helper.
        torch.cuda.synchronize()
        del arena
    emit(
        "partial_prefix_diagnostic_summary",
        {
            "diagnostic_gate": "FAIL" if failures else "PASS",
            "all_3_layers_collected": True,
            "failed_arms": list(failures),
            "completed_records": list(records),
            "scope": "704 control; explicit-fixture714; prefix-only714..719 poison; full-only715..719 poison",
            "production_decision": "NO_PROMOTION_FROM_THIS_DIAGNOSTIC",
        },
    )
    if failures:
        raise PartialPrefixDiagnosticFailure(
            f"{len(failures)} partial-prefix diagnostic records failed; evidence emitted before assertion"
        )
    return {
        "diagnostic_failures": [],
        "implementation_acceptance": "NOT_CLAIMED",
        "completed_records": records,
    }
