"""N4 fidelity on a real initialized shared model and its native KVCache.

Called only by the isolated native smoke subclass. No checkpoint loader,
private cache allocator, old full runner or mock model is used. Each arm
restores the same live rows in the SAME engine-owned pools, so page layout,
initial state and numerical comparison boundaries agree exactly.
"""

import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv41.ppu.engram import build_token_windows
from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import NativeGraphCache
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy
from rtp_llm.models_py.modules.dsv41.transformer import V4Transformer


def _tagged_inputs(model, length, *, prefill):
    """Reserve fixture pages before service readiness; use native descriptors."""
    g, native = model.shared_weight_views.geometry, model.kv_cache
    device = model.v4.embed.weight.device
    policy = PrefillPolicy.from_config(model.shared_weight_views.config)
    maximum = 132
    by_tag = {}
    for owner, tag in model.shared_graph_runtime_keys:
        framework_tag = policy.framework_tag(owner, tag, g)
        if framework_tag in by_tag:
            continue
        full = tag in ("global_kv_1", "global_kv_2", "indexer_kv")
        physical_span = native.get_seq_size_per_block(framework_tag)
        kernel_span = native.get_kernel_seq_size_per_block(framework_tag)
        columns = math.ceil(maximum / physical_span)
        factor = physical_span // kernel_span
        if factor < 1 or physical_span % kernel_span:
            raise ValueError("invalid native physical/kernel page geometry")
        physical = torch.arange(1, columns + 1, dtype=torch.int32).reshape(1, -1)
        kernel = (physical[:, :, None] * factor + torch.arange(factor)).reshape(1, -1).int()
        pool = native.get_layer_cache(owner, framework_tag).kv_cache_base
        if int((kernel if full else physical).max()) >= pool.shape[0]:
            raise ValueError("native cache has insufficient reserved fixture pages")
        inputs = NS(
            is_prefill=prefill, is_target_verify=False, context_parallel_info=None,
            input_lengths=torch.tensor([length], dtype=torch.int32),
            prefix_lengths=torch.zeros(1, dtype=torch.int32),
            sequence_lengths=torch.tensor([length], dtype=torch.int32, device=device),
            sequence_lengths_plus_1_device=torch.tensor([length + 1], dtype=torch.int32, device=device),
            cu_seqlens=torch.tensor([0, length], dtype=torch.int32),
            kv_cache_block_id=physical, kv_cache_kernel_block_id=kernel,
            kv_cache_block_id_device=physical.to(device),
            kv_cache_kernel_block_id_device=kernel.to(device),
            cache_store_inputs=None, cache_store_writer=None,
        )
        inputs.input_lengths_device = inputs.input_lengths.to(device)
        inputs.prefix_lengths_device = inputs.prefix_lengths.to(device)
        inputs.cu_seqlens_device = inputs.cu_seqlens.to(device)
        by_tag[framework_tag] = inputs
    return by_tag


def _inputs(ids, tagged, windows):
    return NS(
        input_ids=ids, attention_inputs=tagged, engram_token_windows=windows,
        combo_position_ids=torch.arange(ids.numel(), dtype=torch.int32, device=ids.device),
        expert_token_counts=None, is_fake_stream=False, warmup=False,
        need_all_logits=True, need_all_hidden_states=True,
        multimodal_inputs=NS(multimodal_features=[]),
    )


def _coordinates(cache, maximum):
    """Bounded live row domain; no all-cache-byte or page matrix scan."""
    result = {}
    for owner, tag in cache.geometry.pool_keys():
        if tag in ("global_kv_1", "global_kv_2", "indexer_kv"):
            ratio = cache.geometry.ratio(owner)
            positions = torch.arange(maximum // ratio, device=cache.device) * ratio
        else:
            positions = torch.arange(maximum, device=cache.device)
        # A native ring may alias several raw positions to one physical slot.
        # Restore each slot once, including the two new Decode write slots.
        result[owner, tag] = torch.unique(cache.slots(owner, tag, positions[None])).reshape(1, -1)
    return result


def _snapshot(cache, coords):
    return {key: cache.read_rows(*key, slots).clone() for key, slots in coords.items()}


def _restore(cache, coords, snapshot):
    for key, slots in coords.items():
        cache.write_rows(*key, slots, snapshot[key])


def _state_mismatches(left, right):
    return {f"{key[0]}:{key[1]}": int(torch.count_nonzero(left[key].view(torch.uint8) != right[key].view(torch.uint8)))
            for key in left}


def _hidden_comparison(left, right):
    """Bounded observations only; never replace the fidelity assertions."""
    left, right = left.detach().cpu(), right.detach().cpu()
    max_abs = float((left.float() - right.float()).abs().max())
    return {
        "exact": bool(torch.equal(left, right)),
        "allclose": bool(torch.allclose(left, right, rtol=0.005, atol=0.005)),
        "max_abs": max_abs if math.isfinite(max_abs) else str(max_abs),
        "left_finite": bool(torch.isfinite(left).all()),
        "right_finite": bool(torch.isfinite(right).all()),
        "rtol": 0.005, "atol": 0.005,
    }


def _write_fidelity_diagnostic(step, record):
    """Publish before assertions, outside the parent's rank-report glob."""
    directory = Path(os.environ["RTP_V41_N4_REPORT_DIR"]).parent / "fidelity-diagnostics"
    directory.mkdir(exist_ok=True)
    target = directory / f"rank{dist.get_rank()}-step{step}.json"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("x") as output:
        json.dump(record, output, indent=2, allow_nan=False)
        output.write("\n")
    os.replace(temporary, target)


def _pointers(inputs, impl):
    values = {
        "input_ids": inputs.input_ids, "windows": inputs.engram_token_windows,
        "positions": impl.metadata.positions, "valid_mask": impl.metadata.valid_mask,
        "request_ids": impl.metadata.request_ids,
        "valid_count": impl.cuda_graph_valid_request_count_device,
        "lengths": impl._position_source,
        **{"table:" + tag: table for tag, table in impl.cache.tables.items()},
        **{f"pool:{owner}:{tag}": pool for (owner, tag), pool in impl.cache._pools.items()},
    }
    return {key: value.data_ptr() for key, value in values.items()}


def _update(inputs, impl, history, token):
    device = inputs.input_ids.device
    length = len(history)
    inputs.input_ids.fill_(token)
    width = inputs.engram_token_windows.shape[1]
    window = [token] + history[-(width - 1):][::-1]
    inputs.engram_token_windows.copy_(torch.tensor([window], dtype=torch.int32, device=device))
    for entry in inputs.attention_inputs.values():
        entry.sequence_lengths.fill_(length)
        entry.sequence_lengths_plus_1_device.fill_(length + 1)
    impl.prepare_cuda_graph(inputs.attention_inputs, valid_request_count=1)
    return {"token_ids": [token], "positions": [length], "windows": [window], "live": 1}


@torch.inference_mode()
def run_fidelity(model):
    """Exactly one capture and two real replays, all8 ranks in every arm."""
    from rtp_llm.models_py.modules.dsv41.ppu.cache import CacheGeometry

    g, native = model.shared_weight_views.geometry, model.kv_cache
    if native is None or type(native).__name__ != "KVCache":
        raise TypeError("N4 fidelity requires the real C++ engine KVCache")
    if type(model.v4).forward_decode is not V4Transformer.forward_decode:
        raise AssertionError("full40 public Decode loop must be inherited")
    if (dist.get_world_size(), model._v4_args.dp_size, g.num_layers) != (8, 8, 40):
        raise ValueError("require actual full40/DP8/EP8/ENP8")
    if model.shared_runtime.policy.mode != "full":
        raise ValueError("N4 matched reference uses full real Prefill, no CED approximation")
    model.shared_graph_runtime_keys = CacheGeometry.from_config(model.shared_weight_views.config).pool_keys()
    device = model.v4.embed.weight.device
    # Distinct nonzero real token histories per rank, using the real checkpoint.
    history = [(19 + 31 * dist.get_rank() + i * 17) % g.vocab_size for i in range(129)]
    ids = torch.tensor(history, dtype=torch.int64, device=device)
    cu = torch.tensor([0, 129], dtype=torch.int64, device=device)
    windows = build_token_windows(ids, cu, torch.full((1, g.engram_max_ngram_size - 1), -1, dtype=torch.int64, device=device)).to(device)
    prefill_inputs = _inputs(ids, _tagged_inputs(model, 129, prefill=True), windows)
    # Test-only role switch: native pools, weights and modules stay identical.
    # The ordinary shared public Prefill arm produces every live cache row.
    role = model._is_decode_role
    try:
        model._is_decode_role = False
        prefill = model.forward(prefill_inputs).hidden_states
    finally:
        model._is_decode_role = role
    if not bool(torch.isfinite(prefill).all()):
        raise AssertionError("real shared Prefill is not finite")
    tagged = _tagged_inputs(model, 129, prefill=False)
    inputs = _inputs(torch.zeros(1, dtype=torch.int32, device=device), tagged,
                     torch.empty((1, g.engram_max_ngram_size), dtype=torch.int32, device=device))
    impl = model.prepare_fmha_impl(inputs, is_cuda_graph=True)
    runtime, cache = model.shared_graph_runtime, impl.cache
    coords = _coordinates(cache, 131)
    if any(bool((slots < 0).any()) for slots in coords.values()):
        raise AssertionError("native fixture metadata contains an unallocated slot")
    seed = _snapshot(cache, coords)
    # Prove this was a populated native state, rather than synthetic zero KV.
    source = SharedNativeCache(model.shared_weight_views.config, g, native, tagged, (129,), device=device)
    fingerprints = source.fingerprint((129,))
    if not fingerprints or not any(item["nonzero_bytes"] for item in fingerprints.values()):
        raise AssertionError("real Prefill did not populate native cache")
    tokens = [((history[-1] + 101 + i * 53) % g.vocab_size) for i in range(2)]
    capture_metadata = _update(inputs, impl, history[:-1], history[-1])
    pointers = _pointers(inputs, impl)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        # One bounded warmup. It is deliberately excluded from replay counts.
        model.forward(inputs, impl)
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.synchronize(device)
    before_capture = [(layer.attn_norm.calls, layer.ffn_norm.calls, layer.ffn.calls) for layer in model.v4.layers]
    with torch.cuda.graph(graph, stream=stream):
        graph_output = model.forward(inputs, impl).hidden_states
    torch.cuda.synchronize(device)
    capture_calls = [
        [layer.attn_norm.calls - before[0], layer.ffn_norm.calls - before[1], layer.ffn.calls - before[2]]
        for layer, before in zip(model.v4.layers, before_capture)
    ]
    if capture_calls != [[1, 1, 1]] * 40:
        raise AssertionError("capture must normalize attention/FFN exactly once per layer")
    reports = []
    try:
        for step, token in enumerate(tokens):
            _restore(cache, coords, seed)
            metadata = _update(inputs, impl, history, token)
            runtime.validate()
            mhc_before = runtime.ops.mhc_execution_stats()
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                # Independent N2 dispatch/Attention/ENP, on the SAME nondefault
                # stream as capture. The existing eager Mega-mHC qualifier
                # naturally refuses this stream; no default/backend override.
                independent = model.forward(inputs).hidden_states.clone()
                independent_state = _snapshot(cache, coords)
                mhc_independent = runtime.ops.mhc_execution_stats()
                _restore(cache, coords, seed)
                safe = model.forward(inputs, impl).hidden_states.clone()
                safe_state = _snapshot(cache, coords)
                mhc_safe = runtime.ops.mhc_execution_stats()
                _restore(cache, coords, seed)
                for counter in (runtime.layer_calls, runtime.moe_calls, runtime.engram_calls):
                    counter.zero_()
                stats_before = runtime.ops.moe_engine._mega_recv_stats.clone()
                graph.replay()
            torch.cuda.synchronize(device)
            if stream == torch.cuda.default_stream(device):
                raise AssertionError("independent eager must use the nondefault capture stream")
            independent_refusals = mhc_independent["fallbacks"].get("decode:unqualified_stream", 0) - mhc_before["fallbacks"].get("decode:unqualified_stream", 0)
            graph_refusals = mhc_safe["fallbacks"].get("graph:unqualified_phase", 0) - mhc_independent["fallbacks"].get("graph:unqualified_phase", 0)
            if (mhc_before["backend"] != "mega" or mhc_safe["calls"] != mhc_before["calls"]
                or independent_refusals != 40 or graph_refusals != 40):
                raise AssertionError("mHC reference arms did not use the existing unfused qualification paths")
            actual = graph_output.clone()
            after = _snapshot(cache, coords)
            hidden_exact = bool(torch.equal(safe, actual))
            safe_mismatches = _state_mismatches(safe_state, after)
            ref_mismatches = _state_mismatches(independent_state, after)
            _write_fidelity_diagnostic(step, {
                "schema": "N4-b05-fidelity-diagnostics-v1",
                "rank": dist.get_rank(), "step": step,
                "observation": "after existing replay synchronization; before fidelity assertions",
                "acceptance": "NOT_EVALUATED; original assertions remain authoritative",
                "hidden_shape": list(actual.shape), "hidden_dtype": str(actual.dtype),
                "hidden": {
                    "replay_vs_independent": _hidden_comparison(actual, independent),
                    "graph_safe_vs_independent": _hidden_comparison(safe, independent),
                    "replay_vs_graph_safe": _hidden_comparison(actual, safe),
                },
                "graph_safe_hidden_exact": hidden_exact,
                "graph_safe_cache_mismatches": safe_mismatches,
                "independent_cache_mismatches": ref_mismatches,
                "requested_update": metadata, "capture_requested_update": capture_metadata,
                "actual_metadata": {
                    "input_ids": inputs.input_ids.tolist(),
                    "windows": inputs.engram_token_windows.tolist(),
                    "positions": impl.metadata.positions.tolist(),
                    "request_ids": impl.metadata.request_ids.tolist(),
                    "valid_mask": impl.metadata.valid_mask.tolist(),
                    "valid_count": impl.cuda_graph_valid_request_count_device.tolist(),
                    "position_source": impl._position_source.tolist(),
                    "position_source_is_plus_one": impl._position_is_plus_one,
                    "per_tag": {
                        tag: {"sequence_lengths": entry.sequence_lengths.tolist(),
                              "sequence_lengths_plus_1_device": entry.sequence_lengths_plus_1_device.tolist()}
                        for tag, entry in inputs.attention_inputs.items()
                    },
                },
                "addresses": {"capture": pointers, "current": _pointers(inputs, impl),
                              "unchanged": _pointers(inputs, impl) == pointers},
                "streams": {
                    "capture_and_all_three_arms": int(stream.cuda_stream),
                    "current_after_replay": int(torch.cuda.current_stream(device).cuda_stream),
                    "default": int(torch.cuda.default_stream(device).cuda_stream),
                    "arm_stream_is_nondefault": stream != torch.cuda.default_stream(device),
                },
                "capture_attention_norm_ffn_norm_moe_calls_by_layer": capture_calls,
                "replay_device_counters": {
                    "layers": runtime.layer_calls.tolist(),
                    "moe": runtime.moe_calls.tolist(),
                    "engram": runtime.engram_calls.tolist(),
                    "received_assignments": int((runtime.ops.moe_engine._mega_recv_stats - stats_before).sum()),
                },
                "mhc_reference": {"before": mhc_before, "after_independent": mhc_independent,
                                  "after_graph_safe": mhc_safe,
                                  "independent_refusals": independent_refusals,
                                  "graph_safe_refusals": graph_refusals},
            })
            # Preserve the independent Graph/eager hidden and valid-cache-byte criteria.
            torch.testing.assert_close(actual, independent, rtol=0.005, atol=0.005)
            if not hidden_exact or any(safe_mismatches.values()) or any(ref_mismatches.values()):
                raise AssertionError({"hidden_exact": hidden_exact, "safe_state": safe_mismatches, "independent_state": ref_mismatches})
            if not bool(torch.isfinite(actual).all()):
                raise AssertionError("Graph output is not finite")
            if _pointers(inputs, impl) != pointers:
                raise AssertionError("Graph input/metadata/pool allocation changed")
            if impl.metadata.positions.tolist() != [129 + step] or impl.metadata.valid_mask.tolist() != [True]:
                raise AssertionError("replay did not consume current device metadata")
            layer_calls, moe_calls, engram_calls = (x.tolist() for x in (runtime.layer_calls, runtime.moe_calls, runtime.engram_calls))
            expected_enp = [int(i in g.engram_layer_ids) for i in range(40)]
            if layer_calls != [1] * 40 or moe_calls != [1] * 40 or engram_calls != expected_enp:
                raise AssertionError("replay did not execute full40/MegaMoE/ENP")
            received = int((runtime.ops.moe_engine._mega_recv_stats - stats_before).sum())
            reports.append({
                "step": step, "input_update": metadata, "fixed_addresses": True,
                "graph_replay_calls": 1, "layer_calls": layer_calls,
                "moe_calls": moe_calls, "engram_calls": engram_calls,
                "received_assignments": received, "graph_safe_hidden_exact": hidden_exact,
                "mhc_reference": {"backend_default": mhc_before["backend"],
                                  "stream": int(stream.cuda_stream), "nondefault_capture_stream": True,
                                  "independent_refusal": "decode:unqualified_stream", "independent_refusals": independent_refusals,
                                  "graph_safe_refusal": "graph:unqualified_phase", "graph_safe_refusals": graph_refusals,
                                  "new_mega_mhc_calls": 0},
                "graph_safe_cache_mismatches": safe_mismatches,
                "independent_cache_mismatches": ref_mismatches,
                "independent_hidden_max_abs": float((actual.float() - independent.float()).abs().max()),
                "independent_hidden_tolerance": {"rtol": 0.005, "atol": 0.005},
                "finite": True,
            })
            history.append(token)
            seed = after
        views = model.shared_weight_views
        weights = []
        for layer_id in range(40):
            w13, s13, w2, s2 = runtime.ops.moe_engine._weights[layer_id]
            packed, down = views.routed(layer_id)
            if any(a.data_ptr() != b.data_ptr() for a, b in zip((w13, s13, w2, s2), (*packed, *down))):
                raise AssertionError("MegaMoE is not consuming framework routed views")
            if any(n != 1 for n in views.conversions.values()):
                raise AssertionError("shared weights were converted more than once")
            weights.append({"layer": layer_id, "packed_shape": list(w13.shape),
                            "scale_shape": list(s13.shape), "packed_dtype": str(w13.dtype),
                            "scale_dtype": str(s13.dtype), "framework_view_aliases": True,
                            "native_transform_cached": (layer_id, device) in runtime.ops.moe_engine._processed})
        return {
            "ok": True, "rank": dist.get_rank(), "layers": 40, "dp": 8, "ep": 8, "enp": 8,
            "scope": "D single role; direct torch CUDAGraph over public full40 Decode, real native engine pools",
            "prefill_source": "current shared full Prefill; real checkpoint weights;129 real token IDs",
            "prefill_token_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            "prefill_native_fingerprints": fingerprints, "native_pool_type": type(native).__name__,
            "warmup_forwards": 1, "capture_forwards": 1, "replay_count": 2,
            "capture_input": capture_metadata,
            "capture_attention_norm_ffn_norm_moe_calls_by_layer": capture_calls,
            "mega_mhc": runtime.ops.mhc_execution_stats(),
            "bucket": 1, "sync": True, "pointers": pointers, "weights": weights,
            "moe": runtime.ops.moe_engine.execution_stats(), "steps": reports,
            "enp_bindings": {str(layer): {"size": b.size, "rank": b.rank,
                              "ids_capacity": b.ids.numel(), "gathered_capacity": b.gathered.numel(),
                              "mapped_host_payload_pointer": b.payload_pointer,
                              "mapped_host_scale_pointer": b.scale_pointer}
                             for layer, b in runtime.engram._bindings.items()},
            "FULL_VALIDATION": "DEFERRED_TO_N10", "CP8_to_DP8_RDMA": "NOT_RUN",
        }
    finally:
        torch.cuda.synchronize(device)
        # Destroy the test graph before native teardown releases model memory.
        graph.reset()
