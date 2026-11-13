"""Checkpoint-backed WORLD8 Verify FFN fork/join qualification.

Called after the normal native loader. All forty FFNs read independent M32
inputs; no attention, mHC, KV semantics or model performance is claimed.
The old layer.forward with no plan is the zero-tolerance eager oracle.
"""

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist

from rtp_llm.config.v41_runtime import MEGAMOE_DSPARK_ASYNC_FLAGS
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.layers import PPUSharedGate, PPUSharedExpert
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear
from rtp_llm.models_py.modules.dsv41.ppu.moe import PPUMoE
from rtp_llm.models_py.modules.dsv41.ppu.quant_mode import MXFP4_FP8_PTPC, quant_mode
from rtp_llm.models_py.modules.dsv41.ppu.verify_moe_overlap import VerifyMoEOverlap


REPORT_ENV = "RTP_V41_VERIFY_MOE_REPORT_DIR"
NATIVE_KERNELS = (
    "fp4_mega_moe_pre_dispatch", "fp4_mega_moe_dispatch_gemm1",
    "m_grouped_gemm_fp4_moe_scatter", "ppu_mega_moe_barrier_reduce",
)


def _raw(value):
    return value.detach().contiguous().view(torch.uint8).cpu()


def _digest(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(_raw(value).numpy().tobytes())
    return digest.hexdigest()


def _same(actual, expected, label):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{label}: metadata differs")
    if not torch.equal(_raw(actual), _raw(expected)):
        raise AssertionError(f"{label}: raw bytes differ; no tolerance fallback")


def _coordinated_check(device, error):
    flag = torch.tensor(int(error is not None), dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if int(flag.item()):
        raise AssertionError(error or "another native rank failed this phase")


def _frame(device, seed):
    state = NS(
        is_graph=True, verify=True, prefill=False, proposal=False, idle=False,
        grouped_moe=None, bulk_counters=False, verify_moe_overlap_allowed=True,
        # A metadata-only FFN surface. This is not a native attention fixture.
        packed_attention=NS(scope="FFN-only; attention NOT_RUN"),
        metadata=NS(batch_size=8, q_len_per_req=4,
                    positions=torch.arange(32, dtype=torch.int64, device=device),
                    valid_mask=torch.ones(32, dtype=torch.bool, device=device)),
        moe_calls=torch.arange(40, dtype=torch.int64, device=device) + seed,
        layer_calls=torch.arange(40, dtype=torch.int64, device=device) + seed + 100,
        engram_calls=torch.arange(40, dtype=torch.int64, device=device) + seed + 200,
        routed_rows_device=torch.full((40,), -17, dtype=torch.int64, device=device),
    )
    slab = torch.empty((40, 32, 5120), dtype=torch.bfloat16, device=device)
    loads = torch.arange(40 * 48, dtype=torch.int32, device=device).reshape(40, 48) + seed
    return NS(state=state, slab=slab, loads=loads)


def _pointers(frame):
    s = frame.state
    return tuple(value.data_ptr() for value in (
        frame.slab, frame.loads, s.metadata.positions, s.metadata.valid_mask,
        s.moe_calls, s.routed_rows_device, s.layer_calls, s.engram_calls,
    ))


def _telemetry(frame, engine):
    s = frame.state
    return {"received": engine._mega_recv_stats.clone(), "loads": frame.loads.clone(),
            "moe": s.moe_calls.clone(), "rows": s.routed_rows_device.clone(),
            "layers": s.layer_calls.clone(), "engram": s.engram_calls.clone()}


def _restore(frame, engine, seed):
    engine._mega_recv_stats.copy_(seed["received"])
    frame.loads.copy_(seed["loads"])
    s = frame.state
    for key, destination in (("moe", s.moe_calls), ("rows", s.routed_rows_device),
                             ("layers", s.layer_calls), ("engram", s.engram_calls)):
        destination.copy_(seed[key])


def _update(frame, rank, generation):
    generator = torch.Generator(device="cpu").manual_seed(81017 + rank * 1009 + generation * 131)
    host = (torch.randn((40, 32, 5120), generator=generator) * 0.75).bfloat16()
    frame.slab.copy_(host)
    q = torch.arange(32, device=frame.slab.device) % 4
    req = torch.arange(32, device=frame.slab.device) // 4
    lengths = torch.tensor([4, 3, 2, 1, 4, 0, 2, 4], device=frame.slab.device)
    valid = torch.ones_like(q, dtype=torch.bool) if generation % 2 == 0 else q < lengths[req]
    if generation == 2 and rank == 0:
        valid.zero_()
    frame.state.metadata.valid_mask.copy_(valid)
    frame.state.metadata.positions.copy_(64 + generation * 13 + q)


@contextmanager
def _bound(native, frame, plan):
    if native.current is not None:
        raise RuntimeError("FFN fixture cannot overlap a native transaction")
    frame.state.verify_moe_overlap = plan
    native.current = frame.state
    try:
        yield
    finally:
        native.current = None


def _forward(owners, frame, native, plan, routes):
    routes.clear()
    with _bound(native, frame, plan):
        outputs = tuple(owner(
            frame.slab[layer].view(8, 4, 5120),
            expert_load=NS(expert_token_counts=frame.loads[layer]),
        ) for layer, owner in enumerate(owners))
    if len(routes) != 40 or [item[0] for item in routes] != list(range(40)):
        raise AssertionError("all forty real router modules must execute exactly once")
    return outputs, tuple((ids, weights) for _, ids, weights in routes)


def _snapshot(outputs, routes, frame, engine):
    return NS(outputs=tuple(value.clone() for value in outputs),
              routes=tuple((ids.clone(), weights.clone()) for ids, weights in routes),
              telemetry=_telemetry(frame, engine))


def _assert_snapshot(actual, expected, frame, seed):
    for layer, (left, right) in enumerate(zip(actual.outputs, expected.outputs)):
        _same(left, right, f"layer{layer} FFN")
        if not bool(torch.isfinite(left).all()):
            raise AssertionError(f"layer{layer} output is not finite")
    for layer, (left, right) in enumerate(zip(actual.routes, expected.routes)):
        _same(left[0], right[0], f"layer{layer} route IDs")
        _same(left[1], right[1], f"layer{layer} route weights")
    for key in expected.telemetry:
        _same(actual.telemetry[key], expected.telemetry[key], key)
    valid = int(frame.state.metadata.valid_mask.sum().item())
    _same(actual.telemetry["moe"], seed["moe"] + 1, "one MoE call per layer")
    _same(actual.telemetry["rows"], torch.full_like(seed["rows"], valid), "live routed rows")
    _same(actual.telemetry["layers"], seed["layers"], "FFN-only layer counters")
    _same(actual.telemetry["engram"], seed["engram"], "FFN-only Engram counters")
    # The shared branch still computes invalid rows, as the original serial
    # layer does. Do not replace its outputs with zeros to make an oracle pass.
    for ids, weights in actual.routes:
        dead = ~frame.state.metadata.valid_mask
        if bool((ids[dead] != -1).any()) or bool((weights[dead] != 0).any()):
            raise AssertionError("invalid origin rows retained business routes")


def _profile_evidence(path, repeats):
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    counts = Counter(event["name"] for event in kernels)
    if repeats != 3:
        raise ValueError("the native qualifier profiles exactly A/B/A")
    expected = 40 * repeats
    for name in NATIVE_KERNELS:
        if counts[name] != expected:
            raise AssertionError(f"{name}: {counts[name]} != {expected}; full native path required")
    router = [event for event in kernels if "dtypeBF16xBF16xFP32xFP32xFP32" in event["name"]]
    if counts["fp8_dense_gemm"] != 2 * expected or len(router) != expected:
        raise AssertionError("all three launches require exactly eighty shared and forty router GEMMs each")
    launches = {}
    for event in events:
        if (event.get("ph") == "X" and "GraphLaunch" in event.get("name", "")
            and event.get("cat") in ("cuda_runtime", "cuda_driver")):
            correlation = event.get("args", {}).get("correlation")
            if correlation is None:
                raise AssertionError("GraphLaunch is missing its device correlation")
            launches.setdefault(correlation, event)
    if len(launches) != repeats:
        raise AssertionError("exactly three correlated A/B/A GraphLaunch groups required")
    groups, signatures = [], []
    all_native_streams, all_shared_streams = set(), set()
    for correlation, launch in sorted(launches.items(), key=lambda item: item[1]["ts"]):
        group = [event for event in kernels
                 if event.get("args", {}).get("correlation") == correlation]
        group_counts = Counter(event["name"] for event in group)
        if any(group_counts[name] != 40 for name in NATIVE_KERNELS):
            raise AssertionError("each GraphLaunch must execute all four native kernels forty times")
        group_router = [event for event in group if "dtypeBF16xBF16xFP32xFP32xFP32" in event["name"]]
        if group_counts["fp8_dense_gemm"] != 80 or len(group_router) != 40:
            raise AssertionError("each GraphLaunch requires eighty shared and forty real router GEMMs")
        streams = {event.get("args", {}).get("stream") for event in group
                   if event["name"] in NATIVE_KERNELS}
        shared = {event.get("args", {}).get("stream") for event in group
                  if event["name"] == "fp8_dense_gemm"}
        route_streams = {event.get("args", {}).get("stream") for event in group_router}
        if (None in streams or None in shared or len(streams) != 1 or len(shared) != 1
            or streams & shared or route_streams != streams):
            raise AssertionError("within each GraphLaunch, native MoE/router and shared PTPC require distinct fixed streams")
        # CUDA Graphs expose virtual profiler stream IDs. Graph A and B may
        # have different routed IDs despite recording the same side stream.
        # Compare inside a launch, then require repeated A to retain its DAG.
        signatures.append((tuple(sorted(streams)), tuple(sorted(shared))))
        all_native_streams.update(streams)
        all_shared_streams.update(shared)
        groups.append({"correlation": correlation, "host_launch_us": launch["ts"],
                       "native_calls": {name: group_counts[name] for name in NATIVE_KERNELS},
                       "shared_ptpc_gemms": group_counts["fp8_dense_gemm"],
                       "router_gemms": len(group_router), "native_trace_streams": sorted(streams),
                       "shared_trace_streams": sorted(shared), "router_trace_streams": sorted(route_streams)})
    if signatures[0] != signatures[2]:
        raise AssertionError("the first and final A replay must retain the same profiler stream signature")
    return {"native_calls": {name: counts[name] for name in NATIVE_KERNELS},
            "shared_ptpc_gemms": counts["fp8_dense_gemm"], "router_gemms": len(router),
            "native_trace_streams": sorted(all_native_streams), "shared_trace_streams": sorted(all_shared_streams),
            "per_graph_launch": groups, "repeated_A_stream_signature_equal": True,
            "kernel_count": len(kernels), "trace": str(path)}


@torch.inference_mode()
def run_native(model):
    """Two captured overlap graphs, unchanged eager oracle, real WORLD8 math."""
    if os.environ.get("RTP_V41_TEST_VERIFY_MOE_OVERLAP") != "1":
        raise ValueError("explicit native Verify overlap selector required")
    rank, device = dist.get_rank(), model.v4.embed.weight.device
    g, engine, native = model.shared_weight_views.geometry, model._shared_ops.moe_engine, model.shared_runtime
    if (not is_ppu() or dist.get_world_size() != 8 or device.type != "cuda"
        or torch.cuda.get_device_capability(device) != (8, 9)
        or (g.num_layers, g.hidden_size, g.moe_inter_dim) != (40, 5120, 2304)
        or (model._v4_args.dp_size, model._v4_args.ep_size, model._v4_args.tp_size) != (8, 8, 1)
        or quant_mode() != MXFP4_FP8_PTPC or not isinstance(engine, PPUMoE)
        or model._v4_args.fake_balance_expert):
        raise ValueError("requires loaded Flash1017 PTPC target40, actual PPU DP8/EP8/TP1")
    owners = tuple(layer.ffn for layer in model.v4.layers)
    if len(owners) != 40 or any(not isinstance(owner.gate, PPUSharedGate)
                              or not isinstance(owner.shared_expert, PPUSharedExpert)
                              or not isinstance(owner.shared_expert.w13, PPUPTPCLinear)
                              or not isinstance(owner.shared_expert.w2, PPUPTPCLinear) for owner in owners):
        raise AssertionError("actual checkpoint router and shared PTPC modules required")
    if any(os.environ.get(flag) != "0" for flag in MEGAMOE_DSPARK_ASYNC_FLAGS):
        raise ValueError("native qualifier requires all four async flags explicitly zero")
    directory = Path(os.environ[REPORT_ENV])
    trace_dir = directory.parent / "traces"
    trace_dir.mkdir(exist_ok=True)
    progress_path = directory / f"rank{rank}.progress.json"
    report = {"rank": rank, "ok": False, "scope": "checkpoint FFN40-only; attention/mHC/model-performance NOT_RUN",
              "phases": [], "native_workspace_closed": False, "graph_cleanup": False,
              "effective_async_flags": {flag: os.environ[flag] for flag in MEGAMOE_DSPARK_ASYNC_FLAGS}}

    def progress(phase):
        report["phase"] = phase
        progress_path.write_text(json.dumps(report, indent=2) + "\n")

    frames, graphs, handles = {}, {}, []
    plan = None
    previous = tuple((getattr(owner, "runtime", None), owner.calls) for owner in owners)
    caller = torch.cuda.Stream(device=device)
    caller.wait_stream(torch.cuda.current_stream(device))
    observations = []
    try:
        progress("prepare-real-native-resources")
        engine.prepare_shared_graph(torch.empty((32, 5120), dtype=torch.bfloat16, device=device), range(40))
        if engine._mega_capacity < 32 or len([key for key in engine._processed if key[1] == device]) < 40:
            raise AssertionError("all forty native weight transforms and fixed workspace required")
        plan = VerifyMoEOverlap(device, engine, target_layers=40, max_rows=32)
        plan.bind_caller(caller)
        signature = plan.storage_signature()
        for layer, owner in enumerate(owners):
            owner.runtime = native
            def observe(module, args, output, layer=layer):
                observations.append((layer, output[0], output[1]))
            handles.append(owner.gate.register_forward_hook(observe))
        recv_seed = torch.arange(48, dtype=torch.int32, device=device) + 11
        seeds, pointers = {}, {}
        # Preparation and the seed allocation occurred on the initialization
        # stream after caller was created; order that new work explicitly.
        caller.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(caller):
            for index, name in enumerate(("A", "B")):
                frames[name] = frame = _frame(device, 19 + index * 7)
                _update(frame, rank, index)
                engine._mega_recv_stats.copy_(recv_seed)
                seeds[name] = _telemetry(frame, engine)
                pointers[name] = _pointers(frame)
                # Both unchanged eager and new fork/join kernels/JIT are warm
                # before capture. Native resources are never created in it.
                _forward(owners, frame, native, None, observations)
                _restore(frame, engine, seeds[name])
                _forward(owners, frame, native, plan, observations)
                caller.synchronize()
                dist.barrier()
                _restore(frame, engine, seeds[name])
                graph = torch.cuda.CUDAGraph()
                progress("capture-" + name)
                before = tuple(owner.calls for owner in owners)
                with torch.cuda.graph(graph, stream=caller):
                    outputs, routes = _forward(owners, frame, native, plan, observations)
                if tuple(owner.calls - count for owner, count in zip(owners, before)) != (1,) * 40:
                    raise AssertionError("capture must call each real FFN exactly once")
                graphs[name] = NS(graph=graph, outputs=outputs, routes=routes)
        torch.cuda.synchronize(device)
        last_refs, route_digests = {}, {}
        for name, generation in (("A", 0), ("B", 1), ("A", 2), ("B", 3)):
            progress(f"compare-{name}-live-input-{generation}")
            frame, captured, seed = frames[name], graphs[name], seeds[name]
            with torch.cuda.stream(caller):
                _update(frame, rank, generation)
                _restore(frame, engine, seed)
                outputs, routes = _forward(owners, frame, native, None, observations)
                expected = _snapshot(outputs, routes, frame, engine)
                _restore(frame, engine, seed)
                before = tuple(owner.calls for owner in owners)
                captured.graph.replay()
                actual = _snapshot(captured.outputs, captured.routes, frame, engine)
            caller.synchronize()
            error = None
            try:
                _assert_snapshot(actual, expected, frame, seed)
                if _pointers(frame) != pointers[name] or plan.storage_signature() != signature:
                    raise AssertionError("replay changed fixed input/native/plan storage")
                if tuple(owner.calls for owner in owners) != before:
                    raise AssertionError("replay ran Python FFN code instead of its captured DAG")
            except Exception as caught:
                error = repr(caught)
            _coordinated_check(device, error)
            digest = _digest(tuple(value for pair in actual.routes for value in pair))
            old_digest = route_digests.get(name)
            if old_digest is not None:
                _coordinated_check(device, "changed live input did not change real routes" if digest == old_digest else None)
            route_digests[name] = digest
            received = int((actual.telemetry["received"] - seed["received"]).sum().item())
            valid_rows = int(frame.state.metadata.valid_mask.sum().item())
            world_received = torch.tensor([received, valid_rows], device=device, dtype=torch.int64)
            dist.all_reduce(world_received)
            totals = world_received.tolist()
            error = None
            if totals[0] <= 0 or totals[0] != 40 * 6 * totals[1]:
                error = "native received assignments differ from forty layers of six real routes per valid row"
            if generation == 2 and rank == 0 and received <= 0:
                error = "empty-origin rank0 did not execute experts for active peers"
            _coordinated_check(device, error)
            last_refs[name] = expected
            report["phases"].append({"graph": name, "generation": generation, "raw_bytes_equal": True,
                "finite": True, "valid_rows": valid_rows,
                "positions": frame.state.metadata.positions.tolist(), "route_sha256": digest,
                "received_assignments": received, "world_received_assignments": totals[0],
                "world_valid_rows": totals[1],
                "fixed_addresses": True, "replay_python_calls": 0})
            progress("comparison-passed")
        progress("profile-two-graphs-ordered-A-B-A")
        with torch.cuda.stream(caller):
            _restore(frames["A"], engine, seeds["A"])
            _restore(frames["B"], engine, seeds["B"])
        caller.synchronize()
        dist.barrier()
        plan_before = plan.receipt()
        profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                     torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
        with profile, torch.cuda.stream(caller):
            for name in ("A", "B", "A"):
                graphs[name].graph.replay()
            caller.synchronize()
        trace_path = trace_dir / f"verify-moe-overlap-rank{rank}.trace.json"
        profile.export_chrome_trace(str(trace_path))
        error = None
        try:
            evidence = _profile_evidence(trace_path, 3)
            for name, repeats in (("A", 2), ("B", 1)):
                frame, expected, seed = frames[name], last_refs[name], seeds[name]
                for actual, reference in zip(graphs[name].outputs, expected.outputs):
                    _same(actual, reference, name + " interleaved graph output")
                for actual, reference in zip(graphs[name].routes, expected.routes):
                    _same(actual[0], reference[0], name + " interleaved route IDs")
                    _same(actual[1], reference[1], name + " interleaved route weights")
                _same(frame.state.moe_calls, seed["moe"] + repeats, name + " cumulative calls")
                _same(frame.loads, seed["loads"] + repeats * (expected.telemetry["loads"] - seed["loads"]), name + " cumulative load")
                _same(frame.state.routed_rows_device, expected.telemetry["rows"], name + " live rows")
                _same(frame.state.layer_calls, seed["layers"], name + " layer counters")
                _same(frame.state.engram_calls, seed["engram"], name + " Engram counters")
                if _pointers(frame) != pointers[name]:
                    raise AssertionError("interleaved replay changed fixed frame storage")
            expected_received = recv_seed + 2 * (last_refs["A"].telemetry["received"] - recv_seed) + (last_refs["B"].telemetry["received"] - recv_seed)
            _same(engine._mega_recv_stats, expected_received, "two graphs share real native recv state")
            if plan.receipt()["python_calls"] != plan_before["python_calls"]:
                raise AssertionError("interleaved graph replay called Python plan")
        except Exception as caught:
            error = repr(caught)
        _coordinated_check(device, error)
        report.update(ok=True, topology={"dp": 8, "ep": 8, "tp": 1, "layers": 40,
                      "rows": 32, "batch": 8, "query_width": 4},
                      actual_modules=True, checkpoint_weight_changes=False,
                      eager_oracle="unchanged real layer.forward with plan=None; raw bytes, no tolerance",
                      graph_count=2, sequential_profile_replays=["A", "B", "A"],
                      trace=evidence, plan=plan.receipt(), backend=engine.execution_stats())
        progress("passed-before-native-cleanup")
    except BaseException:
        progress("failed-before-cleanup")
        raise
    finally:
        # Publish progress before any potentially blocking native drain. The
        # controller has an outer bounded watchdog and owns only its children.
        for handle in handles:
            handle.remove()
        torch.cuda.synchronize(device)
        for captured in graphs.values():
            captured.graph.reset()
        if plan is not None:
            plan.close()
            report["plan_closed"] = plan.receipt()["closed"]
        for owner, (runtime, calls) in zip(owners, previous):
            owner.runtime, owner.calls = runtime, calls
        native.current = None
        report["graph_cleanup"] = True
        # Normal model shutdown owns MegaMoE destruction; releasing its live
        # workspace here would invalidate an initialized native model.
        progress("graphs-drained-native-shutdown-pending")
    return report
