"""External, test-only dynamic Graph gate for explicitly declared real fixtures.

The Bazel owner injects the already loaded 9dee partial helper and live test
objects. This file owns one arena, never imports the real test module, and does
not run on import. No performance measurement or production-lineage claim.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest import mock

import torch

# The exact audited helper digest (kept separate from any module import name).
PARTIAL_HELPER_SHA256 = (
    "9dee412e2cb7d698b8b9bb6fb292842b89fcc283f6f893b1827ddbafc76f79ed"
)
NATIVE_SHA256 = "b2cd92dab0ce55400f80194d956d0b0f48cf86e4250068b6d9d2775c867a2f74"
BACKING_BYTES = 34358689792
MAX_COORDINATES = 16


class PartialPrefixGraphFailure(AssertionError):
    """All safe numerical arms were emitted; original gates remain failing."""


class GraphInvariantError(AssertionError):
    """A fixed, data-free assertion message from this helper only."""


def _require(condition, message):
    if not condition:
        raise GraphInvariantError(message)


def _steps(fixtures):
    by_layer = {f.manifest["layer_idx"]: f for f in fixtures}
    _require(
        len(fixtures) == 3 and set(by_layer) == {0, 13, 27},
        "requires three captured layers",
    )
    return [
        (by_layer[layer], permute, cut)
        for layer, permute, cut in (
            (0, False, 704),
            (13, False, 714),
            (27, False, 714),
            (13, True, 704),
            (0, True, 714),
            (27, False, 704),
        )
    ]


def _comparison(h, actual, expected, *, oracle=False):
    result = h._compare(actual, expected, oracle=oracle)
    a = actual.to(expected.dtype)
    thresholds = result["thresholds"]
    bad = (
        ~torch.isfinite(a)
        | ~torch.isfinite(expected)
        | ~torch.isclose(a, expected, atol=thresholds["atol"], rtol=thresholds["rtol"])
    )
    # Bounded coordinates only: no actual/reference values and no full row list.
    result["mismatch_coordinate_sample"] = (
        bad.nonzero()[:MAX_COORDINATES].cpu().tolist()
    )
    result["mismatch_coordinate_sample_limit"] = MAX_COORDINATES
    result["mismatch_coordinate_sample_truncated"] = (
        result["mismatch_count"] > MAX_COORDINATES
    )
    return result


def _addresses(ab, output, lse):
    op, case = ab.on_op, ab.case
    tensors = {
        "Q": case.q,
        "KV": case.kv,
        "output": output,
        "prefix_O": op.prefix_output,
        "captured_native_LSE": lse,
        "case_table": case.table,
        "case_history": case.history,
    }
    for name in (
        "table",
        "table_host",
        "lengths",
        "lengths_host",
        "cuts",
        "cuts_host",
        "row_cuts",
        "row_cuts_host",
        "prefix_qptr",
        "qptr_host",
        "prefix_indptr",
        "prefix_table",
        "prefix_table_host",
    ):
        tensors[name] = getattr(op, name)
    return {
        name: {
            "ptr": t.data_ptr(),
            "shape": list(t.shape),
            "stride": list(t.stride()),
            "storage_offset": t.storage_offset(),
            "dtype": str(t.dtype),
        }
        for name, t in tensors.items()
    }


def _check_cut_state(ab, cut):
    op = ab.on_op
    for gpu, host in (("cuts", "cuts_host"), ("row_cuts", "row_cuts_host")):
        actual, mirror = getattr(op, gpu).cpu(), getattr(op, host)
        _require(
            torch.equal(actual, mirror) and bool((actual == cut).all()),
            "CPU/GPU cut differs",
        )
    _require(op.shared_tokens == 500 * cut, "shared token accounting differs")
    _require(torch.equal(op.table.cpu(), op.table_host), "CPU/GPU table differs")
    _require(
        torch.equal(op.table_host[:, : ab.case.table.shape[1]], ab.case.table),
        "current case table not staged",
    )
    _require(torch.equal(op.lengths.cpu(), op.lengths_host), "CPU/GPU lengths differ")
    _require(
        torch.equal(op.lengths_host, ab.case.history + 1),
        "current case lengths not staged",
    )
    _require(bool((op.lengths_host == 715).all()), "full attention length differs")
    _require(torch.equal(op.prefix_qptr.cpu(), op.qptr_host), "CPU/GPU qptr differs")
    _require(
        torch.equal(op.prefix_table.cpu(), op.prefix_table_host),
        "CPU/GPU prefix table differs",
    )


def _check_capture_links(ab, state, bindings, output):
    """Pointer-only checks; CPU tensor reads are deliberately elsewhere."""
    args, kwargs, returned = state["native"]
    recorder = state["tail"]
    p = recorder.parameters()
    _require(
        recorder.kernel is bindings.generic_tail,
        "capture did not use original generic tail",
    )
    _require(
        p["num_warps"] == 2 and p["NATIVE_PREFIX"] and p["SPLITS"] == 1,
        "capture tail geometry differs",
    )
    _require(tuple(recorder.records[0][0]) == (500, 8), "capture tail grid differs")
    _require(
        p["PART_OUT"].data_ptr()
        == returned[0].data_ptr()
        == ab.on_op.prefix_output.data_ptr(),
        "tail did not consume this invocation's prefix O",
    )
    _require(
        p["PART_LSE"].data_ptr() == returned[1].data_ptr(),
        "tail did not consume this invocation's native LSE",
    )
    _require(
        p["PART_OUT"].dtype == returned[0].dtype == torch.float32,
        "FP32 prefix contract differs",
    )
    _require(
        p["OUT"].data_ptr() == output.data_ptr() == ab.on_op.output.data_ptr(),
        "output storage differs",
    )
    _require(
        p["Q"].data_ptr() == ab.case.q.data_ptr() == args[0].data_ptr(),
        "capture Q differs",
    )
    _require(p["KV"].data_ptr() == ab.case.kv.data_ptr(), "capture KV differs")
    _require(p["TABLE"].data_ptr() == ab.on_op.table.data_ptr(), "tail table differs")
    _require(
        p["LENGTHS"].data_ptr() == ab.on_op.lengths.data_ptr(), "tail lengths differ"
    )
    _require(p["CUTS"].data_ptr() == ab.on_op.row_cuts.data_ptr(), "tail cuts differ")
    _require(
        kwargs["seqlen_k"].data_ptr() == ab.on_op.cuts.data_ptr(), "native cuts differ"
    )
    return {
        "kernel": recorder.kernel.__name__,
        "grid": list(recorder.records[0][0]),
        "num_warps": p["num_warps"],
        "PART_OUT_dtype": str(p["PART_OUT"].dtype),
        "tail_consumes_same_invocation_native_O": True,
        "tail_consumes_same_invocation_native_LSE": True,
        "native_LSE_retained_until_graph_destruction": True,
    }


def run_partial_prefix_graph_gate(
    fixtures,
    *,
    lineage_by_fixture,
    expected_native_sha256,
    emit_record,
    case_factory,
    ab_factory,
    bindings,
    partial_helpers,
    capture_fn,
):
    """One graph, six real updates, 24 clean/K/V/KV Graph-versus-eager pairs.

    Caller must not allocate an arena first. Identity/route/oracle errors stop
    immediately; safe numerical failures are all recorded before final failure.
    Original off tests and their failures are not executed or reclassified.
    """
    h = partial_helpers
    _require(
        hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest()
        == PARTIAL_HELPER_SHA256,
        "injected partial helper is not the audited 9dee source",
    )
    _require(
        expected_native_sha256 == NATIVE_SHA256,
        "this gate requires the reviewed V-safe library",
    )
    _require(
        os.getenv("MKT_CASCADE_NATIVE_FP32_O") == "1", "explicit native FP32 O required"
    )
    _require(
        os.getenv("MKT_CASCADE_TAIL_VARIANT", "production") == "production",
        "gate owns generic tail patch",
    )
    _require("AITER_AOT_IMPORT" not in os.environ, "AITER_AOT_IMPORT must be unset")
    settings = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    _require(
        settings == {"float32_matmul_precision": "highest", "allow_tf32": False},
        "oracle settings differ",
    )
    steps = _steps(fixtures)
    _require(
        len({f.graph_key() for f in fixtures}) == 1,
        "real fixtures require separate graphs; unsupported here",
    )
    _require(
        {f.manifest["paged_kv_metadata"]["storage_nbytes"] for f in fixtures}
        == {BACKING_BYTES},
        "arena differs",
    )
    _require(
        set(lineage_by_fixture) == {f.name for f in fixtures},
        "lineage coverage differs",
    )
    validated = {
        f.name: h._validate_lineage(f, lineage_by_fixture[f.name]) for f in fixtures
    }
    failures, completed = [], []
    original_tail = bindings.cascade_module._private_tail_merge
    original_cpu_hashes = {
        f.name: {n: h._hash(t) for n, t in f.tensors.items()} for f in fixtures
    }
    anchor = max(
        fixtures, key=lambda f: f.manifest["paged_kv_metadata"]["storage_offset"]
    )
    arena = case = ab = graph = output = state = None
    stage = "allocate"

    def emit(name, payload):
        emit_record(
            name,
            {
                "evidence": "RTP_MEASURED",
                "classification": "partial_prefix_dynamic_graph_fixture_gate",
                "production_partial_lineage": "NOT_IMPLEMENTED",
                "production_acceptance": "NOT_CLAIMED",
                "timing": "NOT_RUN",
                "oracle_settings": settings,
                "off_scope": "NOT_RUN; original off failure classification unchanged",
                **payload,
            },
        )
        completed.append(name)
        if payload.get("gate") == "FAIL" and name != "partial_graph_summary":
            failures.append(name)

    try:
        arena = torch.empty(BACKING_BYTES // 2, dtype=torch.bfloat16, device="cuda")
        case = case_factory(anchor, arena)
        ab = ab_factory(case)
        _require(
            ab.prefix_path() == "aiter_native"
            and ab.on_op.prefix_output.dtype == torch.float32,
            "candidate requires actual native FP32 O",
        )
        stage = "capture"
        # Only object references and Python launch metadata are observed inside
        # capture. No .cpu(), hash, synchronization, or output clone is added.
        import aiter

        original_native = aiter.mha_batch_prefill_func
        state = {}

        def capture_invoke():
            calls = []
            recorder = bindings.tail_recorder_factory(bindings.generic_tail)

            def observe(*args, **kwargs):
                returned = original_native(*args, **kwargs)
                calls.append((args, dict(kwargs), returned))
                return returned

            with mock.patch.object(
                aiter, "mha_batch_prefill_func", side_effect=observe
            ), mock.patch.object(
                bindings.cascade_module, "_private_tail_merge", recorder
            ):
                result = ab.on()
            _require(len(calls) == 1, "capture requires one native invocation")
            state.clear()
            state.update(native=calls[0], tail=recorder)
            return result

        with h._manual_cut(ab, 704):
            graph, output = capture_fn(capture_invoke)
            _check_cut_state(ab, 704)
            captured_route = _check_capture_links(ab, state, bindings, output)
            args, kwargs, returned = state["native"]
            # This check is after capture/replay, and retains the actual LSE.
            h._native_call_metadata(
                args, kwargs, returned, ab, 704, bindings, expected_native_sha256
            )
        original_addresses = _addresses(ab, output, returned[1])
        anchor_offset = case.kv.storage_offset()
        previous_hashes = {
            "Q": h._hash(case.q),
            "KV": h._hash(case.packed),
            "table": h._hash(case.table),
        }

        for step, (fixture, permute, cut) in enumerate(steps):
            stage = f"step{step}_update"
            new = case_factory(fixture, arena, permute=permute)
            requested_hashes = {
                "Q": h._hash(new.q),
                "KV": h._hash(new.packed),
                "table": h._hash(new.table),
            }
            ab.update(new)
            del new
            _require(
                _addresses(ab, output, returned[1]) == original_addresses,
                "update changed graph addresses/layout",
            )
            _require(
                case.kv.storage_offset() == anchor_offset, "graph anchor offset changed"
            )
            clean_cpu = fixture.tensors["kv_pages"]
            clean_hash = h._hash(clean_cpu)
            ids = case.physical_ids.to(device=case.q.device, dtype=torch.long)
            _require(
                h._hash(case.kv.index_select(0, ids)) == clean_hash,
                "updated live KV differs",
            )
            current_hashes = {
                "Q": h._hash(case.q),
                "KV": h._hash(case.packed),
                "table": h._hash(case.table),
            }
            _require(
                current_hashes == requested_hashes,
                "update did not install requested Q/KV/table values",
            )
            changed = {
                name: current_hashes[name] != previous_hashes[name]
                for name in current_hashes
            }
            # Consecutive layers can have the same physical page IDs, including
            # after the same permutation. Validate against this fixture's
            # original table; report previous-step changes without requiring them.
            permutation_changed_source_table = (
                current_hashes["table"]
                != fixture.manifest["tensors"]["block_table"]["sha256"]
            )
            _require(
                not permute or permutation_changed_source_table,
                "requested permutation did not change its source fixture table",
            )
            canonical_hashes = (h._hash(case.k), h._hash(case.v), h._hash(case.q))
            metadata = h._metadata_hashes(ab.on_op)
            tail_logical = case.logical_table[:, 44].long()
            _require(
                tail_logical.unique().numel() == 500,
                "permutation lost independent tail pages",
            )
            _require(
                torch.equal(
                    case.physical_ids[tail_logical].to(case.table.dtype),
                    case.table[:, 44],
                ),
                "updated logical/physical tail mapping differs",
            )
            expected32 = bindings.full_reference(case, dtype=torch.float32)
            expected64 = bindings.full_reference(case, dtype=torch.float64)
            oracle = _comparison(h, expected32, expected64, oracle=True)
            emit(
                f"partial_graph_step{step}_oracle",
                {
                    "gate": "PASS" if oracle["thresholds_pass"] else "FAIL",
                    "layer": fixture.manifest["layer_idx"],
                    "cut": cut,
                    "FP32_vs_FP64": oracle,
                },
            )
            _require(
                oracle["thresholds_pass"], "independent full attention oracles disagree"
            )

            with h._manual_cut(ab, cut):
                _check_cut_state(ab, cut)
                clean_output = None

                def run_pair(component):
                    nonlocal clean_output
                    _check_cut_state(ab, cut)
                    _require(
                        _addresses(ab, output, returned[1]) == original_addresses,
                        "replay addresses changed",
                    )
                    ab.on_op.prefix_output.fill_(float("nan"))
                    output.fill_(float("nan"))
                    graph.replay()
                    replay = output.clone()
                    graph_meta = h._native_call_metadata(
                        args,
                        kwargs,
                        returned,
                        ab,
                        cut,
                        bindings,
                        expected_native_sha256,
                    )
                    graph_route = _check_capture_links(ab, state, bindings, output)
                    eager, invocation, eager_meta = h._run_full(
                        ab,
                        bindings.generic_tail,
                        cut,
                        bindings,
                        expected_native_sha256,
                        2,
                    )
                    del invocation
                    checks = {
                        "graph_vs_FP32": _comparison(h, replay, expected32),
                        "graph_vs_FP64": _comparison(h, replay, expected64),
                        "eager_vs_FP32": _comparison(h, eager, expected32),
                        "eager_vs_FP64": _comparison(h, eager, expected64),
                    }
                    exact = h._same_bits(replay, eager)
                    poison_exact = component == "clean" or h._same_bits(
                        replay, clean_output
                    )
                    controls = {
                        "graph_vs_own_eager_bitwise": exact,
                        "poison_vs_same_cut_clean_bitwise": poison_exact,
                        "graph_addresses_and_layout_stable": _addresses(
                            ab, output, returned[1]
                        )
                        == original_addresses,
                        "captured_tail_route_unchanged": graph_route == captured_route,
                        "native_prefix_O_finite": graph_meta["prefix_O_finite"][
                            "nonfinite"
                        ]
                        == 0,
                        "native_prefix_LSE_finite": graph_meta["prefix_LSE_finite"][
                            "nonfinite"
                        ]
                        == 0,
                        "CPU_GPU_cut_equal": True,
                        "prefix_and_output_poisoned_before_replay": True,
                    }
                    passed = all(c["thresholds_pass"] for c in checks.values()) and all(
                        controls.values()
                    )
                    if component == "clean":
                        clean_output = replay.clone()
                    emit(
                        f"partial_graph_step{step}_{component}",
                        {
                            **case.describe(),
                            "step": step,
                            "cut": cut,
                            "gate": "PASS" if passed else "FAIL",
                            "component": component,
                            "comparisons": checks,
                            "controls": controls,
                            "hashes": {
                                "graph_output": h._hash(replay),
                                "own_eager_output": h._hash(eager),
                                "clean_output": h._hash(clean_output),
                                "inputs": current_hashes,
                            },
                            "actual_changes_since_previous_step": changed,
                            "permutation_changed_source_fixture_table": permutation_changed_source_table,
                            "graph_anchor_layer": anchor.manifest["layer_idx"],
                            "graph_anchor_storage_offset": anchor_offset,
                            "source_fixture_storage_offset": fixture.manifest[
                                "paged_kv_metadata"
                            ]["storage_offset"],
                            "graph_native": graph_meta,
                            "graph_tail": graph_route,
                            "eager_execution": eager_meta,
                            "poison": (
                                None
                                if component == "clean"
                                else {
                                    "tokens_half_open": [715, 720],
                                    "components": component,
                                    "legal714_preserved": True,
                                    "payload_sha256": h._hash(payload),
                                    "legal714_sha256": h._current_token_hashes(
                                        payload, tail_logical
                                    ),
                                }
                            ),
                            "lineage": validated[fixture.name][1],
                            "history_update_scope": "actual history714 copied; no fabricated history variation",
                        },
                    )

                stage = f"step{step}_clean"
                run_pair("clean")
                for component in ("K", "V", "KV"):
                    stage = f"step{step}_{component}"
                    payload = h._poison_payload(clean_cpu, tail_logical, 715, component)
                    _require(
                        h._current_token_hashes(payload, tail_logical)
                        == h._current_token_hashes(clean_cpu, tail_logical),
                        "legal token714 changed",
                    )
                    with h._installed_payload(case, payload, clean_hash):
                        run_pair(component)
                    del payload
                del clean_output

            stage = f"step{step}_restore"
            _require(
                h._hash(case.kv.index_select(0, ids)) == clean_hash,
                "live pages not restored",
            )
            _require(
                canonical_hashes == (h._hash(case.k), h._hash(case.v), h._hash(case.q)),
                "canonical/Q changed",
            )
            _require(metadata == h._metadata_hashes(ab.on_op), "metadata not restored")
            _check_cut_state(ab, 704)
            _require(
                bindings.cascade_module._private_tail_merge is original_tail,
                "tail symbol not restored",
            )
            _require(
                _addresses(ab, output, returned[1]) == original_addresses,
                "restoration changed graph addresses",
            )
            emit(
                f"partial_graph_step{step}_restoration",
                {
                    "gate": "PASS",
                    "live_pages_sha256": clean_hash,
                    "metadata_and_cut704_restored": True,
                    "canonical_and_Q_unchanged": True,
                    "graph_addresses_stable": _addresses(ab, output, returned[1])
                    == original_addresses,
                },
            )
            previous_hashes = current_hashes
            del expected32, expected64

        for fixture in fixtures:
            _require(
                {n: h._hash(t) for n, t in fixture.tensors.items()}
                == original_cpu_hashes[fixture.name],
                "CPU fixture payload changed",
            )
        emit(
            "partial_graph_summary",
            {
                "gate": "FAIL" if failures else "PASS",
                "pair_count": 24,
                "graph_count": 1,
                "step_count": 6,
                "captured_tail": captured_route,
                "failed_records": list(failures),
                "completed_record_count_before_summary": len(completed),
                "addresses": original_addresses,
                "all_CPU_fixtures_unchanged": True,
                "scope": "real three layers, manual fixture lineage, generic2wave, cut704/714, invalid715..719 only",
                "high_address_gload_and_synthetic_remainders": "NOT_RUN",
            },
        )
    except Exception as error:
        # Do not print exception repr: upstream library messages may contain
        # tensor values. The fixed stage/code is enough to find the raw failure.
        emit(
            "partial_graph_integrity_error",
            {
                "gate": "FAIL",
                "stage": stage,
                "code": "IDENTITY_ROUTE_ORACLE_EXECUTION_OR_RESTORATION_ERROR",
                "error_class": type(error).__name__,
                # Every call to this module's _require uses a literal message.
                # Keep external/helper exceptions opaque; their repr may hold data.
                "local_invariant": (
                    str(error) if type(error) is GraphInvariantError else None
                ),
            },
        )
        raise PartialPrefixGraphFailure(
            f"partial-prefix Graph integrity failure at {stage}"
        ) from None
    finally:
        _require(
            bindings.cascade_module._private_tail_merge is original_tail,
            "scoped tail patch escaped",
        )
        # Drop Graph before its captured native result/LSE and arena references.
        graph = None
        state = output = ab = case = arena = None
    if failures:
        raise PartialPrefixGraphFailure(
            f"{len(failures)} partial-prefix Graph gate records failed"
        )
