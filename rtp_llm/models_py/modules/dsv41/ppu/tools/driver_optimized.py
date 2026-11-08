"""N5 comparisons through the real initialized public full40 model loops.

Only the isolated fixture calls this driver. It reuses N4 native page and
snapshot helpers; it has no model loop, checkpoint loader or private cache.
"""

import hashlib
import inspect
import os
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv41.ppu.cache import CacheGeometry
from rtp_llm.models_py.modules.dsv41.ppu.engram import build_token_windows
from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import NativeGraphCache
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.attention_layer import (
    optimized_diagnostics_enabled, _optimized_compare, _optimized_write_json,
)
from rtp_llm.models_py.modules.dsv41.ppu.packed_attention import (
    FUSION_ENV, PACKED_ENV, STREAMS_ENV,
    load_packed_backend,
)
from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_graph_fidelity import (
    _coordinates, _hidden_comparison, _inputs, _restore, _snapshot,
    _state_mismatches, _tagged_inputs,
)
from rtp_llm.models_py.modules.dsv41.transformer import V4Transformer


@contextmanager
def _selection(packed, fused, streams):
    choices = {PACKED_ENV: str(packed), FUSION_ENV: str(fused), STREAMS_ENV: str(streams)}
    saved = {key: os.environ.get(key) for key in choices}
    os.environ.update(choices)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _forward(model, inputs, *, prefill):
    role = model._is_decode_role
    before = [(layer.attn_norm.calls, layer.ffn_norm.calls, layer.ffn.calls)
              for layer in model.v4.layers]
    mega_before = sum(model._shared_ops.mhc_execution_stats()["calls"].values())
    try:
        model._is_decode_role = not prefill
        output = model.forward(inputs).hidden_states.clone()
    finally:
        model._is_decode_role = role
    counts = [[layer.attn_norm.calls - old[0], layer.ffn_norm.calls - old[1],
               layer.ffn.calls - old[2]] for layer, old in zip(model.v4.layers, before)]
    mega_calls = sum(model._shared_ops.mhc_execution_stats()["calls"].values()) - mega_before
    if (any(a != 1 or f not in (0, 1) or m != 1 for a, f, m in counts)
            or sum(f for _, f, _ in counts) + mega_calls != 40):
        raise AssertionError("N5 must normalize each layer once, including accepted Mega transitions")
    if not bool(torch.isfinite(output).all()):
        raise AssertionError("N5 model hidden contains nonfinite values")
    return output, list(model.shared_runtime.last_n5_routes), {
        "per_layer_attention_ffn_norm_moe_calls": counts,
        "mega_normalized_transitions": mega_calls, "ffn_normalized_once": True,
    }


def _swa_only_layers(model):
    geometry = model.shared_weight_views.geometry
    return {layer for layer in range(geometry.num_layers) if not geometry.ratio(layer)}


def _assert_routes(routes, *, prefill, selected):
    if len(routes) != 40 or [r["layer"] for r in routes] != list(range(40)):
        raise AssertionError("missing actual shared Attention route coverage")
    for route in routes:
        expected = "flash_mla_with_kvcache_packed528_288" if selected else "flash_mla_sparse_fwd"
        if route["attention_backend"] != expected:
            raise AssertionError(route)
        if selected:
            _, expected_dependency = load_packed_backend()
            if route.get("packed_dependency") != expected_dependency:
                raise AssertionError("packed route did not use the qualified native dependency")
            if route["packed"] != "eligible" or not route["swa_native_alias"]:
                raise AssertionError(route)
            if route["bf16_kv_workspace"] or route["packed_swa_snapshot_retained"]:
                raise AssertionError("direct packed route retained a KV workspace")
            if route["global_native_alias"] is False:
                raise AssertionError(route)
            if route["qk_rope_backend"] != "triton_fused_qk_rope":
                raise AssertionError(route)
            stream_route = "serial" if prefill else "q_current_kv_side_joined"
            if route["projection_streams"] != stream_route:
                raise AssertionError(route)


def _diagnostic_directory():
    if not optimized_diagnostics_enabled():
        return None
    root = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", ""))
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("optimized diagnostics require existing absolute TEST_UNDECLARED_OUTPUTS_DIR")
    directory = root.resolve() / f"n5-optimized-b05-rank{dist.get_rank()}"
    directory.mkdir(exist_ok=False)
    return directory


def _diagnostic_input_identity(inputs):
    # Metadata and real input copies, not weights or whole native pool storage.
    result = {"ids": inputs.input_ids.cpu().tolist(), "windows": inputs.engram_token_windows.cpu().tolist(),
              "positions": inputs.combo_position_ids.cpu().tolist(), "tags": {}}
    for tag, meta in inputs.attention_inputs.items():
        result["tags"][tag] = {name: getattr(meta, name).cpu().tolist() for name in
                               ("input_lengths", "prefix_lengths", "sequence_lengths",
                                "kv_cache_block_id_device", "kv_cache_kernel_block_id_device")}
    return result


def _run_prefill_diagnostics(model, inputs, cache, coords, seed, directory, packed_identity):
    runtime = model.shared_runtime
    if getattr(runtime, "n5_diagnostics", None) is not None:
        raise ValueError("optimized diagnostic collector is already active")
    # The existing runner already forwards this explicit opt-in. Here it
    # requests the packed partition controls only inside the full40 fixture.
    partition_control = os.environ.get("RTP_V41_PPU_PACKED_DIAGNOSTICS", "0") == "1"
    collector = {"schema": "n5-native-b05", "directory": directory, "rank": dist.get_rank(),
                 "baseline": {}, "pt_bytes": 0, "evidence_errors": [],
                 "partition_control": partition_control}
    # This is run-owned data, never a callback/method/registry replacement.
    runtime.n5_diagnostics = collector
    input_identity = _diagnostic_input_identity(inputs)
    arms = (("000-baseline", 0, 0), ("000-repeat", 0, 0), ("010-fusion", 0, 1),
            ("100-packed", 1, 0), ("110-combo", 1, 1))
    if partition_control:
        arms += (("200-partition-only", 1, 0), ("300-swa-residual", 1, 0),
                 ("400-allowed-effects", 1, 0), ("500-dual-residual", 1, 0))
    swa_only_layers = _swa_only_layers(model)
    outputs, reports = {}, []
    baseline_state = None
    baseline_writes = None
    try:
        for name, packed, fused in arms:
            _restore(cache, coords, seed)
            restored = _snapshot(cache, coords)
            restored_differences = _state_mismatches(seed, restored)
            collector.update(arm=name, layers=[], first_exact_difference={}, first_budget_failure={})
            before_inputs = _diagnostic_input_identity(inputs)
            with _selection(packed, fused, 0):
                output, routes, counts = _forward(model, inputs, prefill=True)
            outputs[name] = output
            state = _snapshot(cache, coords)
            writes = list(runtime.last_n5_writes)
            if name == "000-baseline":
                baseline_state, baseline_writes = state, writes
            expected_rope = "triton_fused_qk_rope" if fused else "serial_rope"
            route_ok = (len(routes) == 40 and [r["layer"] for r in routes] == list(range(40))
                        and all(r["qk_rope_backend"] == expected_rope
                                and r["projection_streams"] == "serial" for r in routes))
            for route in routes:
                if packed:
                    expected_global_alias = None if route["layer"] in swa_only_layers else True
                    route_ok = route_ok and (
                        route["attention_backend"] == "flash_mla_with_kvcache_packed528_288"
                        and route.get("packed_dependency") == packed_identity and route.get("swa_native_alias")
                        and route.get("global_native_alias") is expected_global_alias and not route.get("bf16_kv_workspace")
                        and not route.get("packed_swa_snapshot_retained"))
                else:
                    route_ok = route_ok and (
                        route["attention_backend"] == "flash_mla_sparse_fwd" and route.get("bf16_kv_workspace"))
            hidden = _optimized_compare(output, outputs["000-baseline"])
            try:
                torch.testing.assert_close(output, outputs["000-baseline"], rtol=0.005, atol=0.005)
                assertion = {"status": "PASS"}
            except AssertionError as error:
                assertion = {"status": "FAIL", "error": str(error)[:8192]}
            report = {"kind": "DIAGNOSTIC_ONLY", "arm": name, "rank": dist.get_rank(),
                      "seed_restore_mismatches": restored_differences,
                      "seed_restore_exact": not any(restored_differences.values()),
                      "input_before_exact": before_inputs == input_identity,
                      "input_after_exact": _diagnostic_input_identity(inputs) == input_identity,
                      "hidden_vs_baseline": hidden, "same_budget_assertion": assertion,
                      "routes_expected": route_ok, "routes": routes, "full40_counts": counts,
                      "write_coverage_exact": writes == baseline_writes,
                      "state_vs_baseline": _state_mismatches(baseline_state, state),
                      "first_exact_difference": dict(collector["first_exact_difference"]),
                      "first_budget_failure": dict(collector["first_budget_failure"]),
                      "layers_collected": len(collector["layers"]),
                      "pt_total_bytes": collector["pt_bytes"]}
            reports.append(report)
            _optimized_write_json(collector, name + ".json", report)
        repeat = reports[1]
        control_ok = (repeat["hidden_vs_baseline"]["exact"]
                      and not any(repeat["state_vs_baseline"].values())
                      and not repeat["first_exact_difference"]
                      and repeat["write_coverage_exact"] and repeat["routes_expected"]
                      and all(r["seed_restore_exact"] and r["input_before_exact"]
                              and r["input_after_exact"] for r in reports))
        combo = next(report for report in reports if report["arm"] == "110-combo")
        summary = {"kind": "DIAGNOSTIC_ONLY", "SMOKE": "NOT_QUALIFIED", "rank": dist.get_rank(),
                   "arms": [r["arm"] for r in reports], "prefill_forwards": len(arms),
                   "serial_repeat_control_pass": control_ok,
                   "candidate_attribution_allowed": control_ok,
                   "first_exact_difference": {r["arm"]: r["first_exact_difference"] for r in reports},
                   "first_budget_failure": {r["arm"]: r["first_budget_failure"] for r in reports},
                   "hidden_comparisons": {r["arm"]: r["hidden_vs_baseline"] for r in reports},
                   "evidence_errors": collector["evidence_errors"], "pt_total_bytes": collector["pt_bytes"],
                   "input_identity": input_identity,
                   "rtol": 0.005, "atol": 0.005,
                   "original_combo_assertion": combo["same_budget_assertion"],
                   "root_cause": "UNDETERMINED; inspect control and same-Q layer evidence"}
        if partition_control:
            summary["packed_vs_partition_only"] = _optimized_compare(
                outputs["100-packed"], outputs["200-partition-only"],
            )
            summary["candidate_vs_allowed_effects"] = {
                arm: _optimized_compare(outputs[arm], outputs["400-allowed-effects"])
                for arm in ("100-packed", "110-combo")
            }
        _optimized_write_json(collector, "summary.json", summary)
        # Write the control evidence before executing the original hidden gate.
        actual_p, ref_p = outputs["110-combo"], outputs["000-baseline"]
        torch.testing.assert_close(actual_p, ref_p, rtol=0.005, atol=0.005)
        if not control_ok or any(not r["routes_expected"] or not r["write_coverage_exact"]
                                 or r["layers_collected"] != 40 for r in reports):
            raise AssertionError("N5 diagnostic control/route/write evidence failed; no candidate attribution")
        raise AssertionError("N5 DIAGNOSTIC_ONLY collected; run default0 for qualification")
    finally:
        del runtime.n5_diagnostics


@torch.inference_mode()
def run_optimized(model):
    diagnostic_directory = _diagnostic_directory()
    packed_module, packed_identity = load_packed_backend()
    flash_mla_with_kvcache = packed_module.flash_mla_with_kvcache

    g, native, runtime = model.shared_weight_views.geometry, model.kv_cache, model.shared_runtime
    if (native is None or type(native).__name__ != "KVCache"
            or (dist.get_world_size(), model._v4_args.dp_size, g.num_layers) != (8, 8, 40)
            or runtime.policy.mode != "full" or model._is_speculative):
        raise ValueError("N5 requires actual full40/DP8 native non-speculative full-policy model")
    if type(model.v4).forward_decode is not V4Transformer.forward_decode:
        raise AssertionError("N5 must inherit the public full40 decode loop")
    model.shared_graph_runtime_keys = CacheGeometry.from_config(model.shared_weight_views.config).pool_keys()
    device = model.v4.embed.weight.device
    history = [(23 + 31 * dist.get_rank() + i * 17) % g.vocab_size for i in range(32)]
    ids = torch.tensor(history, dtype=torch.int64, device=device)
    windows = build_token_windows(
        ids, torch.tensor([0, 32], dtype=torch.int64, device=device),
        torch.full((1, g.engram_max_ngram_size - 1), -1, dtype=torch.int64, device=device),
    ).to(device)
    tagged = _tagged_inputs(model, 32, prefill=True)
    prefill_inputs = _inputs(ids, tagged, windows)
    cache = NativeGraphCache(model.shared_weight_views.config, native, tagged, device=device)
    coords = _coordinates(cache, 33)
    pointers = {str(key): pool.data_ptr() for key, pool in cache._pools.items()}
    seed = _snapshot(cache, coords)
    if diagnostic_directory is not None:
        return _run_prefill_diagnostics(
            model, prefill_inputs, cache, coords, seed, diagnostic_directory, packed_identity,
        )
    records = []
    with _selection(0, 0, 0):
        ref_p, ref_routes, ref_counts = _forward(model, prefill_inputs, prefill=True)
    ref_p_writes = list(runtime.last_n5_writes)
    _assert_routes(ref_routes, prefill=True, selected=False)
    ref_p_state = _snapshot(cache, coords)
    _restore(cache, coords, seed)
    with _selection(1, 1, 1):
        actual_p, routes, counts = _forward(model, prefill_inputs, prefill=True)
    _assert_routes(routes, prefill=True, selected=True)
    torch.testing.assert_close(actual_p, ref_p, rtol=0.005, atol=0.005)
    if runtime.last_n5_writes != ref_p_writes or not ref_p_writes:
        raise AssertionError("N5 Prefill changed actual native producer write coverage")
    records.append({
        "phase": "prefill", "tokens": 32, "hidden": _hidden_comparison(actual_p, ref_p),
        "routes": routes, "full40_counts": counts, "reference_counts": ref_counts,
        "exact_native_write_coverage": ref_p_writes,
        "downstream_state_byte_diagnostics": _state_mismatches(ref_p_state, _snapshot(cache, coords)),
    })
    # Both Decode arms begin with precisely the SAME real Prefill-produced
    # bytes; no recomputed/private KV bank is used by the model candidate.
    _restore(cache, coords, ref_p_state)
    source = SharedNativeCache(model.shared_weight_views.config, g, native, tagged, (32,), device=device)
    fingerprints = source.fingerprint((32,))
    if not fingerprints or not any(v["nonzero_bytes"] for v in fingerprints.values()):
        raise AssertionError("N5 real request did not populate native state")
    token = (history[-1] + 101) % g.vocab_size
    decode_tagged = _tagged_inputs(model, 32, prefill=False)
    width = g.engram_max_ngram_size
    decode_inputs = _inputs(
        torch.tensor([token], dtype=torch.int32, device=device), decode_tagged,
        torch.tensor([[token] + history[-(width - 1):][::-1]], dtype=torch.int32, device=device),
    )
    with _selection(0, 0, 0):
        ref_d, ref_routes, ref_counts = _forward(model, decode_inputs, prefill=False)
    ref_d_writes = list(runtime.last_n5_writes)
    _assert_routes(ref_routes, prefill=False, selected=False)
    ref_d_state = _snapshot(cache, coords)
    _restore(cache, coords, ref_p_state)
    with _selection(0, 1, 1):
        exact_d, exact_routes, exact_counts = _forward(model, decode_inputs, prefill=False)
    exact_state = _snapshot(cache, coords)
    exact_differences = _state_mismatches(ref_d_state, exact_state)
    if (not torch.equal(exact_d, ref_d) or any(exact_differences.values())
            or runtime.last_n5_writes != ref_d_writes):
        raise AssertionError({"fusion_streams_hidden_exact": bool(torch.equal(exact_d, ref_d)),
                              "fusion_streams_cache_mismatches": exact_differences})
    if any(r["qk_rope_backend"] != "triton_fused_qk_rope"
           or r["projection_streams"] != "q_current_kv_side_joined"
           or r["attention_backend"] != "flash_mla_sparse_fwd" for r in exact_routes):
        raise AssertionError("exact fusion/streams arm missed an actual backend")
    _restore(cache, coords, ref_p_state)
    with _selection(1, 1, 1):
        actual_d, routes, counts = _forward(model, decode_inputs, prefill=False)
    _assert_routes(routes, prefill=False, selected=True)
    torch.testing.assert_close(actual_d, ref_d, rtol=0.005, atol=0.005)
    if runtime.last_n5_writes != ref_d_writes or not ref_d_writes:
        raise AssertionError("N5 Decode changed actual native producer write coverage")
    torch.cuda.synchronize(device)
    records.append({
        "phase": "decode", "tokens": 1, "position": 32,
        "hidden": _hidden_comparison(actual_d, ref_d), "routes": routes,
        "full40_counts": counts, "reference_counts": ref_counts,
        "exact_native_write_coverage": ref_d_writes,
        "fusion_streams_exact_arm": {"hidden_exact": True, "cache_mismatches": exact_differences,
                                     "full40_counts": exact_counts, "routes": exact_routes},
        "downstream_state_byte_diagnostics": _state_mismatches(ref_d_state, _snapshot(cache, coords)),
    })
    if pointers != {str(key): pool.data_ptr() for key, pool in cache._pools.items()}:
        raise AssertionError("N5 moved framework pools")
    if runtime.qkv_streams is None or runtime.qkv_streams.calls != 80:
        raise AssertionError("N5 did not execute exactly80 bounded multistream branches")
    if any(n != 1 for n in model.shared_weight_views.conversions.values()):
        raise AssertionError("N5 caused a second weight conversion")
    enp_sizes = {str(layer): int(binding[2].size)
                 for layer, binding in model._shared_ops._engram.items()}
    if not enp_sizes or any(size != 8 for size in enp_sizes.values()):
        raise AssertionError("N5 requires actual ENP8 bindings")
    wrapper = Path(inspect.getfile(flash_mla_with_kvcache))
    return {
        "ok": True, "rank": dist.get_rank(), "layers": 40, "dp": 8, "ep": 8, "enp": 8,
        "scope": "D native single-role; ordinary eager32-token Prefill then1-token Decode",
        "model_forwards": 5, "records": records, "native_pool_type": type(native).__name__,
        "prefill_native_fingerprints": fingerprints, "fixed_pool_addresses": pointers,
        "installed_flash_mla_wrapper": {"path": str(wrapper), "sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest()},
        "packed_dependency": packed_identity,
        "streams": {"calls": runtime.qkv_streams.calls, "side": int(runtime.qkv_streams.side_stream.cuda_stream),
                    "default": int(torch.cuda.default_stream(device).cuda_stream)},
        "mega_mhc": model._shared_ops.mhc_execution_stats(),
        "enp_binding_sizes": enp_sizes,
        "moe": model._shared_ops.moe_engine.execution_stats(),
        "cache_contract": "packed read-only/scale-byte identity and producer store exact in boundary target; full40 write coverage exact; full40 fusion/streams output+cache exact; packed downstream byte differences diagnostic",
        "FULL_VALIDATION": "DEFERRED_TO_N10", "CP8_to_DP8_RDMA": "NOT_RUN",
    }


def _functional_restore(cache, coords, expected, label):
    _restore(cache, coords, expected)
    differences = _state_mismatches(expected, _snapshot(cache, coords))
    if any(differences.values()):
        raise AssertionError({"functional_restore": label, "mismatches": differences})
    return {"source": label, "exact": True, "bounded_state_mismatches": differences}


def _functional_row_domain(tag, length, window, entries, ratio):
    if length not in (32, 33) or window <= 0 or entries <= 0:
        raise ValueError("functional fixture requires the original bounded32/33 domain")
    if tag in ("global_kv_1", "global_kv_2", "indexer_kv"):
        if ratio not in (1, 2):
            raise ValueError("functional native compressed ratio must be1/2")
        positions = list(range(0, length - length % ratio, ratio))
        return positions, positions
    if tag == "swa_kv":
        positions = list(range(max(0, length - window), length))
        return positions, positions
    if tag == "csa_state":
        # Both ring rows are real payload; only an odd suffix consumes carry.
        return list(range(max(0, length - entries), length)), ([length - 1] if length % 2 else [])
    raise ValueError("unexpected functional native tag")


def _functional_native_state(model, tagged, cache, length):
    """Read only initialized live rows through independent native bridges."""
    from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import decode_kv

    g = model.shared_weight_views.geometry
    bridge = SharedNativeCache(model.shared_weight_views.config, g, model.kv_cache, tagged, (length,),
                               device=cache.device)
    if set(bridge.pools) != set(cache._pools):
        raise AssertionError("functional native pool domain changed")
    records = {}
    for key, pool in bridge.pools.items():
        owner, tag = key
        full = tag in ("global_kv_1", "global_kv_2", "indexer_kv")
        ratio = g.ratio(owner) if full else 1
        expected, logical = _functional_row_domain(tag, length, g.window, pool.entries, ratio)
        consumed = bridge._positions(key, length).tolist()
        if consumed != logical or not expected or any(p < 0 or p >= length for p in expected):
            raise AssertionError({"native_live_domain": str(key), "positions": consumed})
        positions = torch.tensor(expected, device=cache.device, dtype=torch.int64)
        table = pool.table[0].cpu().tolist()
        host_slots = [table[p // pool.span] * pool.entries +
                      ((p % pool.span) // ratio if full else p % pool.entries) for p in expected]
        slots = cache.slots(owner, tag, positions[None])
        if slots.reshape(-1).cpu().tolist() != host_slots or any(s < 0 for s in host_slots):
            raise AssertionError("functional native physical positions disagree")
        alias = (pool.tensor.data_ptr() == pool.handle.kv_cache_base.data_ptr()
                 == cache.pool(owner, tag).data_ptr())
        if not alias:
            raise AssertionError("functional live rows lost native pool alias")
        rows = bridge._copy_rows(key, 0, positions)
        independent = cache.read_rows(owner, tag, slots).reshape_as(rows)
        if not torch.equal(rows.view(torch.uint8), independent.view(torch.uint8)):
            raise AssertionError("functional independent native row reads differ")
        dim = 1024 if tag == "csa_state" else 128 if tag == "indexer_kv" else 512
        if tag == "csa_state":
            decoded = rows
        else:
            decoded = torch.empty((len(expected), dim), dtype=torch.float32, device=cache.device)
            if decode_kv(rows, pool_tag=tag, out=decoded) is not decoded:
                raise AssertionError("functional codec changed caller-owned output")
        if (tuple(decoded.shape) != (len(expected), dim) or decoded.dtype != torch.float32
                or not bool(torch.isfinite(decoded).all())):
            raise AssertionError("functional decoded native live state is invalid/nonfinite")
        records[f"{owner}:{tag}"] = {
            "positions": expected, "consumed_positions": consumed, "rows": len(expected),
            "row_dim": dim, "finite": True, "native_alias": alias,
            "independent_row_read_exact": True,
            "sha256": hashlib.sha256(rows.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
    fingerprints = bridge.fingerprint((length,))
    if not fingerprints or not any(v["nonzero_bytes"] for v in fingerprints.values()):
        raise AssertionError("functional request did not populate native live state")
    return {"logical_length": length, "pools": records, "fingerprints": fingerprints}


def _functional_forward(model, inputs, identity, choice, *, prefill, cache, coords, length, pointers):
    from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import SharedForwardState

    device = cache.device
    if (torch.cuda.is_current_stream_capturing()
            or torch.cuda.current_stream(device) != torch.cuda.default_stream(device)):
        raise AssertionError("functional smoke requires the default eager stream")
    if _diagnostic_input_identity(inputs) != identity:
        raise AssertionError("functional input changed before forward")
    runtime = model.shared_runtime
    attn = next(iter(inputs.attention_inputs.values()))
    prefixes = tuple(map(int, (attn.prefix_lengths if prefill else attn.sequence_lengths).tolist()))
    lengths = tuple(map(int, attn.input_lengths.tolist())) if prefill else (1,) * len(prefixes)
    if (prefixes, lengths) != (((0,), (32,)) if prefill else ((32,), (1,))):
        raise AssertionError("functional continuation has wrong actual prefix/query lengths")
    if length != prefixes[0] + lengths[0] or any(
        bool(meta.is_prefill) != prefill
        or meta.sequence_lengths.tolist() != [32]
        or meta.sequence_lengths_plus_1_device.tolist() != [33]
        for meta in inputs.attention_inputs.values()
    ):
        raise AssertionError("functional per-tag sequence domain changed")
    state = SharedForwardState(runtime, inputs, attn, None, prefill=prefill, idle=False,
                               prefixes=prefixes, lengths=lengths)
    positions, requests, _, valid = state.rows()
    expected_positions = list(range(32)) if prefill else [32]
    if (positions.tolist() != expected_positions or requests.tolist() != [0] * len(expected_positions)
            or not bool(valid.all())):
        raise AssertionError("functional continuation has wrong positions/request domain")
    streams_before = runtime.qkv_streams.calls if runtime.qkv_streams is not None else 0
    with _selection(*choice):
        output, routes, counts = _forward(model, inputs, prefill=prefill)
    rows = 32 if prefill else 1
    if (tuple(output.shape) != (rows, model.shared_weight_views.geometry.hidden_size)
            or output.dtype != torch.bfloat16 or output.device != device):
        raise AssertionError("functional forward changed the output shape/dtype/device")
    if _diagnostic_input_identity(inputs) != identity:
        raise AssertionError("functional input changed during forward")
    _assert_routes(routes, prefill=prefill, selected=bool(choice[0]))
    stream_route = "q_current_kv_side_joined" if choice[2] and not prefill else "serial"
    rope_route = "triton_fused_qk_rope" if choice[1] else "serial_rope"
    if any(r["phase"] != ("prefill" if prefill else "decode") or r["rows"] != rows
           or r["qk_rope_backend"] != rope_route or r["projection_streams"] != stream_route for r in routes):
        raise AssertionError("functional forward missed a selected actual route")
    streams_after = runtime.qkv_streams.calls if runtime.qkv_streams is not None else 0
    if streams_after - streams_before != (40 if stream_route != "serial" else 0):
        raise AssertionError("functional forward did not execute the exact joined stream branches")
    state = _snapshot(cache, coords)
    live = _functional_native_state(model, inputs.attention_inputs, cache, length)
    if any(_state_mismatches(state, _snapshot(cache, coords)).values()):
        raise AssertionError("functional native validation changed live state")
    if pointers != {str(key): pool.data_ptr() for key, pool in cache._pools.items()}:
        raise AssertionError("functional forward moved framework pools")
    return output, state, {
        "arm": ("P" if prefill else "D") + "".join(map(str, choice)),
        "phase": "prefill" if prefill else "decode", "tokens": rows,
        "positions": positions.tolist(), "output_finite": True,
        "input_unchanged": True, "routes": routes, "full40_counts": counts,
        "native_live_state": live, "native_writes": list(runtime.last_n5_writes),
        "stream_calls": streams_after - streams_before,
    }


@torch.inference_mode()
def run_functional(model):
    """Explicit functional continuation; strict run_optimized stays separate."""
    from rtp_llm.models_py.modules.dsv41.ppu.test.optimized_smoke_role import functional_smoke_enabled
    if not functional_smoke_enabled():
        raise ValueError("run_functional requires explicit functional smoke selection")
    packed_module, packed_identity = load_packed_backend()
    g, native, runtime = model.shared_weight_views.geometry, model.kv_cache, model.shared_runtime
    if (native is None or type(native).__name__ != "KVCache"
            or (dist.get_world_size(), model._v4_args.dp_size, model._v4_args.ep_size,
                model._v4_args.tp_size, g.num_layers) != (8, 8, 8, 1, 40)
            or runtime.policy.mode != "full" or runtime.policy.max_rollback_tokens != 0
            or model._is_speculative):
        raise ValueError("functional smoke requires native full40/DP8/EP8/TP1 full-policy eager model")
    if type(model.v4).forward_decode is not V4Transformer.forward_decode:
        raise AssertionError("functional smoke must inherit the public full40 decode loop")
    if runtime.qkv_streams is not None and runtime.qkv_streams.calls != 0:
        raise AssertionError("functional smoke must own exactly its five forwards")
    conversions = dict(model.shared_weight_views.conversions)
    if not conversions or any(n != 1 for n in conversions.values()):
        raise AssertionError("functional smoke requires one weight conversion")
    model.shared_graph_runtime_keys = CacheGeometry.from_config(model.shared_weight_views.config).pool_keys()
    device = model.v4.embed.weight.device
    history = [(23 + 31 * dist.get_rank() + i * 17) % g.vocab_size for i in range(32)]
    ids = torch.tensor(history, dtype=torch.int64, device=device)
    windows = build_token_windows(
        ids, torch.tensor([0, 32], dtype=torch.int64, device=device),
        torch.full((1, g.engram_max_ngram_size - 1), -1, dtype=torch.int64, device=device),
    ).to(device)
    tagged = _tagged_inputs(model, 32, prefill=True)
    prefill_inputs = _inputs(ids, tagged, windows)
    cache = NativeGraphCache(model.shared_weight_views.config, native, tagged, device=device)
    coords = _coordinates(cache, 33)
    pointers = {str(key): pool.data_ptr() for key, pool in cache._pools.items()}
    seed = _snapshot(cache, coords)
    prefill_identity = _diagnostic_input_identity(prefill_inputs)
    _functional_restore(cache, coords, seed, "original_seed")
    ref_p, _, ref_p_record = _functional_forward(
        model, prefill_inputs, prefill_identity, (0, 0, 0), prefill=True,
        cache=cache, coords=coords, length=32, pointers=pointers,
    )
    _functional_restore(cache, coords, seed, "original_seed")
    actual_p, candidate_p_state, candidate_p_record = _functional_forward(
        model, prefill_inputs, prefill_identity, (1, 1, 1), prefill=True,
        cache=cache, coords=coords, length=32, pointers=pointers,
    )
    if not ref_p_record["native_writes"] or candidate_p_record["native_writes"] != ref_p_record["native_writes"]:
        raise AssertionError("functional Prefill changed exact native producer write coverage")
    token = (history[-1] + 101) % g.vocab_size
    decode_tagged = _tagged_inputs(model, 32, prefill=False)
    width = g.engram_max_ngram_size
    decode_inputs = _inputs(
        torch.tensor([token], dtype=torch.int32, device=device), decode_tagged,
        torch.tensor([[token] + history[-(width - 1):][::-1]], dtype=torch.int32, device=device),
    )
    decode_identity = _diagnostic_input_identity(decode_inputs)
    restores = [_functional_restore(cache, coords, candidate_p_state, "actual_P111")]
    ref_d, ref_d_state, ref_d_record = _functional_forward(
        model, decode_inputs, decode_identity, (0, 0, 0), prefill=False,
        cache=cache, coords=coords, length=33, pointers=pointers,
    )
    restores.append(_functional_restore(cache, coords, candidate_p_state, "actual_P111"))
    exact_d, exact_d_state, exact_d_record = _functional_forward(
        model, decode_inputs, decode_identity, (0, 1, 1), prefill=False,
        cache=cache, coords=coords, length=33, pointers=pointers,
    )
    exact_differences = _state_mismatches(ref_d_state, exact_d_state)
    if (not torch.equal(exact_d, ref_d) or any(exact_differences.values())
            or exact_d_record["native_writes"] != ref_d_record["native_writes"]):
        raise AssertionError({"functional_fusion_streams_hidden_exact": bool(torch.equal(exact_d, ref_d)),
                              "native_state_mismatches": exact_differences})
    resources = runtime.qkv_streams
    resource_ids = tuple(id(v) for v in (resources.side_stream, resources.producer, resources.done))
    restores.append(_functional_restore(cache, coords, candidate_p_state, "actual_P111"))
    actual_d, _, candidate_d_record = _functional_forward(
        model, decode_inputs, decode_identity, (1, 1, 1), prefill=False,
        cache=cache, coords=coords, length=33, pointers=pointers,
    )
    if not ref_d_record["native_writes"] or candidate_d_record["native_writes"] != ref_d_record["native_writes"]:
        raise AssertionError("functional Decode changed exact native producer write coverage")
    if (runtime.qkv_streams is not resources or resources.calls != 80
            or resource_ids != tuple(id(v) for v in (resources.side_stream, resources.producer, resources.done))):
        raise AssertionError("functional smoke did not reuse exactly80 joined stream branches")
    side = int(resources.side_stream.cuda_stream)
    default = int(torch.cuda.default_stream(device).cuda_stream)
    if side == default:
        raise AssertionError("functional side stream aliases the default stream")
    if dict(model.shared_weight_views.conversions) != conversions:
        raise AssertionError("functional smoke caused another weight conversion")
    enp_sizes = {str(layer): int(binding[2].size) for layer, binding in model._shared_ops._engram.items()}
    moe = model._shared_ops.moe_engine.execution_stats()
    if (not enp_sizes or any(size != 8 for size in enp_sizes.values())
            or moe["backend"] != "megamoe" or moe["precision"] != "w4a4"
            or not moe["mega_moe_calls"] or moe["fake_balance_expert"]):
        raise AssertionError("functional smoke missed actual ENP8/MegaMoE W4A4 execution")
    torch.cuda.synchronize(device)
    wrapper = Path(inspect.getfile(packed_module.flash_mla_with_kvcache))
    return {
        "ok": True, "rank": dist.get_rank(), "fixture_mode": "FUNCTIONAL_CONTINUATION",
        "layers": 40, "dp": 8, "ep": 8, "enp": 8, "model_forwards": 5,
        "scope": "D native single-role; actual optimized32-token Prefill then1-token Decode",
        "records": [ref_p_record, candidate_p_record, ref_d_record, exact_d_record, candidate_d_record],
        "candidate_prefill_history_verified": True, "decode_seed_restores": restores,
        "prefill_native_fingerprints": candidate_p_record["native_live_state"]["fingerprints"],
        "fusion_streams_exact_arm": {"hidden_exact": True, "state_mismatches": exact_differences,
                                     "write_coverage_exact": True},
        "numerical_diagnostics": {"qualification_gate": False, "rtol": 0.005, "atol": 0.005,
                                  "prefill": _hidden_comparison(actual_p, ref_p),
                                  "decode": _hidden_comparison(actual_d, ref_d)},
        "numerical_alignment": "FAIL_SEPARATE_STRICT_CHECK", "strict_checker": "run_optimized unchanged",
        "fixed_pool_addresses": pointers, "packed_dependency": packed_identity,
        "installed_flash_mla_wrapper": {"path": str(wrapper), "sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest()},
        "streams": {"calls": resources.calls, "side": side, "default": default, "resources_reused": True},
        "mega_mhc": model._shared_ops.mhc_execution_stats(), "enp_binding_sizes": enp_sizes, "moe": moe,
        "FULL_VALIDATION": "DEFERRED_TO_N10", "CP8_to_DP8_RDMA": "NOT_RUN",
    }
