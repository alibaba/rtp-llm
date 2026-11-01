"""Isolated CUDA-Graph Engram latency experiment, independent of RTP runtime.

The caller owns allocations, external views and lifetime guards for every path.
The caller also verifies arithmetic before and after this experiment. No handles,
transport or checkpoint loading belong here.

Example::

    result = run_bench(
        {"raw_local": local_raw, "raw_remote": remote_raw,
         "canonical_local": local_canonical, "canonical_remote": remote_canonical},
        update_ids, case_metadata={"M": 1, "layers": 1, ...},
        samples=1000, epochs=5, output_dir=Path("results/case"))

``update_ids(pair_id, distribution)`` must enqueue input-only work in the current
stream, using caller-preallocated buffers. It must not access the table. Each
paired set sees the same IDs, and different paths execute in balanced order.
Graph addresses and buffer shapes must remain fixed. The paths take no arguments.

Timing events are EXTERNAL event nodes captured inside each graph. Therefore
GPU elapsed time excludes the CPU gap before graph submission. The same replay
also records CPU submit-to-event-completion wall time. Input update and optional
L2 disturbance finish before both clocks. This intentionally synchronized
single-request test is not a sustained-load/queueing or full-model benchmark.
"""

import collections
import gc
import json
import math
import random
import statistics
import time
import warnings
from pathlib import Path


DEFAULT_COMPARISONS = {
    "raw_remote_minus_raw_local": ("raw_remote", "raw_local"),
    "canonical_remote_minus_canonical_local": ("canonical_remote", "canonical_local"),
    "canonical_remote_minus_raw_local": ("canonical_remote", "raw_local"),
}


def _percentile(sorted_values, fraction):
    """Nearest-rank quantile; no interpolation which could hide tail samples."""
    if not sorted_values:
        raise ValueError("percentile requires samples")
    return sorted_values[max(0, math.ceil(len(sorted_values) * fraction) - 1)]


def summarize(values, threshold_us=100.0):
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("finite non-empty timings required")
    ordered = sorted(values)
    exceed = sum(v > threshold_us for v in values)
    return {
        "n": len(values), "min_us": ordered[0], "mean_us": statistics.mean(values),
        "p50_us": _percentile(ordered, .50), "p95_us": _percentile(ordered, .95),
        "p99_us": _percentile(ordered, .99), "p99_9_us": _percentile(ordered, .999),
        "max_us": ordered[-1], "stdev_us": statistics.pstdev(values),
        "threshold_us": threshold_us, "over_threshold_count": exceed,
        "over_threshold_fraction": exceed / len(values),
        "quantile_method": "nearest_rank",
    }


def balanced_order(names, pair_index, seed=4101604):
    """Rotate and reverse a seeded path order; never overlap paired paths."""
    names = list(names)
    if not names:
        raise ValueError("at least one path required")
    random.Random(seed).shuffle(names)
    cycle, rotation = divmod(pair_index, len(names))
    if cycle % 2:
        names.reverse()
    return names[rotation:] + names[:rotation]


def _validate_metadata(value, location="case_metadata"):
    """Catch accidental capability logging. Caller must provide public metadata."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            if key.lower() in {"handles", "handle", "fabric_handle", "fabric_handles",
                               "private_handles", "capabilities", "private_capabilities",
                               "capability_token"}:
                raise ValueError("private capability field forbidden: " + location + "." + key)
            _validate_metadata(item, location + "." + key)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_metadata(item, location)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite metadata")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError("metadata must be public JSON values")


def _epoch_bootstrap_p99(blocks, iterations, seed):
    """Resample contiguous paired sample blocks, not independent experiments.

    The run_bench 'epoch' label only segments one trace/run. This preserves
    within-block autocorrelation but does not establish block independence.
    Do not present it as a production guarantee or coverage of system drift.
    """
    if iterations <= 0 or len(blocks) < 2:
        return {"status": "NOT_COMPUTED", "reason": "requires >=2 epochs and iterations>0"}
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sample = []
        for _ in blocks:
            sample.extend(blocks[rng.randrange(len(blocks))])
        estimates.append(_percentile(sorted(sample), .99))
    estimates.sort()
    return {
        "status": "EXPLORATORY", "method": "paired_epoch_block_percentile_bootstrap",
        "iterations": iterations, "epoch_count": len(blocks), "seed": seed,
        "p99_lower_95_us": _percentile(estimates, .025),
        "p99_upper_95_us": _percentile(estimates, .975),
        "limitation": "contiguous blocks from one trace/run, not independent repeats; "
                      "block independence unproven; no production-tail guarantee",
    }


def summarize_records(records, comparisons, *, threshold_us=100.0, bootstrap_iterations=200,
                      seed=4101604):
    grouped = collections.defaultdict(list)
    by_pair = collections.defaultdict(dict)
    for row in records:
        grouped[(row["distribution"], row["clock"], row["path"])].append(row)
        key = (row["distribution"], row["clock"], row["epoch"], row["pair_id"])
        if row["path"] in by_pair[key]:
            raise ValueError("duplicate path in paired sample")
        by_pair[key][row["path"]] = row["duration_us"]
    absolute = []
    for (distribution, clock, path), rows in sorted(grouped.items()):
        epoch_values = collections.defaultdict(list)
        for row in rows:
            epoch_values[row["epoch"]].append(row["duration_us"])
        absolute.append({
            "distribution": distribution, "clock": clock, "path": path,
            "stats": summarize([r["duration_us"] for r in rows], threshold_us),
            "epochs": {str(e): summarize(v, threshold_us) for e, v in sorted(epoch_values.items())},
        })
    delta_groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for (distribution, clock, epoch, _), values in sorted(by_pair.items()):
        for label, (candidate, baseline) in comparisons.items():
            if candidate not in values or baseline not in values:
                raise ValueError("incomplete pair for " + label)
            delta_groups[(distribution, clock, label)][epoch].append(values[candidate] - values[baseline])
    absolute_lookup = {(r["distribution"], r["clock"], r["path"]): r["stats"] for r in absolute}
    deltas = []
    for index, ((distribution, clock, label), epochs) in enumerate(sorted(delta_groups.items())):
        candidate, baseline = comparisons[label]
        blocks = [values for _, values in sorted(epochs.items())]
        stats = summarize([v for block in blocks for v in block], threshold_us)
        deltas.append({
            "distribution": distribution, "clock": clock, "comparison": label,
            "candidate": candidate, "baseline": baseline, "paired_delta_stats": stats,
            "difference_of_p99_us": absolute_lookup[(distribution, clock, candidate)]["p99_us"]
                                    - absolute_lookup[(distribution, clock, baseline)]["p99_us"],
            "epochs": {str(e): summarize(v, threshold_us) for e, v in sorted(epochs.items())},
            "p99_ci": _epoch_bootstrap_p99(blocks, bootstrap_iterations, seed + index),
            "threshold_observation": ("FAIL_OBSERVED" if stats["over_threshold_count"]
                                      else "NO_OBSERVED_VIOLATION"),
        })
    return {"absolute": absolute, "paired_deltas": deltas,
            "acceptance_status": "COMPONENT_MEASUREMENT_ONLY", "model_slo": "NOT_RUN"}


def _l2_bytes(torch, device):
    props = torch.cuda.get_device_properties(device)
    value = getattr(props, "l2_cache_size", None)
    if value is not None and int(value) > 0:
        return int(value)
    try:
        from cuda.bindings import runtime
        status, value = runtime.cudaDeviceGetAttribute(
            runtime.cudaDeviceAttr.cudaDevAttrL2CacheSize, device)
        if int(status) == 0 and int(value) > 0:
            return int(value)
    except (ImportError, AttributeError):
        pass
    return None


def _thread_switches_reader():
    """Optional Linux thread counters; do not replace with process-wide counts."""
    try:
        import resource
        who = resource.RUSAGE_THREAD
        resource.getrusage(who)
    except (ImportError, AttributeError, ValueError, OSError):
        return None

    def read():
        usage = resource.getrusage(who)
        return (int(usage.ru_nvcsw), int(usage.ru_nivcsw))
    return read


def _wall_parts(begin_ns, replay_return_ns, completion_ns, switches_before=None, switches_after=None):
    """CPU-side call partitions; completion sync duration is NOT pure GPU wait."""
    if not begin_ns <= replay_return_ns <= completion_ns:
        raise ValueError("nonmonotonic CPU timestamps")
    result = {
        "cpu_graph_replay_call_us": (replay_return_ns - begin_ns) / 1000.0,
        "cpu_completion_sync_call_us": (completion_ns - replay_return_ns) / 1000.0,
    }
    if switches_before is not None and switches_after is not None:
        voluntary = switches_after[0] - switches_before[0]
        involuntary = switches_after[1] - switches_before[1]
        if voluntary < 0 or involuntary < 0:
            raise ValueError("thread context-switch counters decreased")
        result.update(thread_nvcsw_delta=voluntary, thread_nivcsw_delta=involuntary)
    else:
        result.update(thread_nvcsw_delta=None, thread_nivcsw_delta=None)
    return result


def calibrate_timing(samples=100, *, warmup=10):
    """Measure instrumentation-only controls; never subtract these samples.

    event_nodes_only has the SAME timing/completion boundaries as run_bench,
    with no kernels between captured external start/end event nodes.
    empty_graph_wall has genuinely no nodes, so completion uses stream sync
    rather than an end event. It is a separate wall-clock diagnostic, not a
    directly subtractable estimate of event overhead.
    """
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError("calibration samples must be a positive integer")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 1:
        raise ValueError("calibration warmup must be a positive integer")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for instrumentation calibration")
    device = torch.cuda.current_device()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    event_graph = None
    empty_graph = None
    begin = end = None
    event_gpu = []
    event_wall = []
    empty_wall = []
    capture_warnings = []
    records = []
    thread_usage = _thread_switches_reader()
    try:
        with torch.inference_mode(), torch.cuda.stream(stream):
            begin = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            event_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(event_graph, stream=stream):
                begin.record(stream)
                end.record(stream)
            empty_graph = torch.cuda.CUDAGraph()
            # PyTorch may warn that this graph is empty, which is intentional.
            # Preserve the warning in the result instead of hiding it.
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                with torch.cuda.graph(empty_graph, stream=stream):
                    pass
            capture_warnings = [str(w.message) for w in observed]
            stream.synchronize()
            for _ in range(warmup):
                event_graph.replay()
                end.synchronize()
                empty_graph.replay()
                stream.synchronize()
            # Alternating controls prevents a fixed warmup/order comparison.
            for sample_id in range(samples):
                order = ("event_nodes_only", "empty_graph_wall") if sample_id % 2 == 0 else (
                    "empty_graph_wall", "event_nodes_only")
                for control in order:
                    stream.synchronize()
                    switches_before = thread_usage() if thread_usage is not None else None
                    wall_begin = time.perf_counter_ns()
                    if control == "event_nodes_only":
                        event_graph.replay()
                        replay_return = time.perf_counter_ns()
                        end.synchronize()
                    else:
                        empty_graph.replay()
                        replay_return = time.perf_counter_ns()
                        stream.synchronize()
                    completion = time.perf_counter_ns()
                    wall_us = (completion - wall_begin) / 1000.0
                    switches_after = thread_usage() if thread_usage is not None else None
                    parts = _wall_parts(wall_begin, replay_return, completion,
                                        switches_before, switches_after)
                    if not math.isfinite(wall_us) or wall_us <= 0:
                        raise RuntimeError("invalid calibration wall-clock sample")
                    if control == "event_nodes_only":
                        gpu_us = begin.elapsed_time(end) * 1000.0
                        # Zero is valid for an empty interval at timer resolution.
                        if not math.isfinite(gpu_us) or gpu_us < 0:
                            raise RuntimeError("invalid calibration GPU-event sample")
                        event_gpu.append(gpu_us)
                        event_wall.append(wall_us)
                        records.append({"sample_id": sample_id, "control": control,
                                        "gpu_graph_event_us": gpu_us,
                                        "cpu_submit_to_event_complete_us": wall_us, **parts})
                    else:
                        empty_wall.append(wall_us)
                        records.append({"sample_id": sample_id, "control": control,
                                        "cpu_submit_to_stream_complete_us": wall_us, **parts})
    finally:
        stream.synchronize()
        if event_graph is not None:
            event_graph.reset()
        if empty_graph is not None:
            empty_graph.reset()
    return {
        "status": "MEASURED_DIAGNOSTIC_ONLY", "samples_per_control": samples,
        "warmup": warmup, "subtracted_from_path_timings": False,
        "included_in_path_comparisons": False,
        "thread_context_switch_counters": "RUSAGE_THREAD" if thread_usage is not None else "UNAVAILABLE",
        "event_nodes_only": {
            "description": "no kernels; graph contains only external start/end timing event nodes",
            "gpu_graph_event": summarize(event_gpu),
            "cpu_submit_to_event_complete": summarize(event_wall),
            "zero_gpu_timestamp_samples": sum(value == 0 for value in event_gpu),
        },
        "empty_graph_wall": {
            "description": "genuinely empty graph; stream completion differs from event completion",
            "cpu_submit_to_stream_complete": summarize(empty_wall),
            "capture_warnings": capture_warnings,
        },
        "records": records,
        "interpretation": "instrumentation can perturb small-shape timing; these distributions "
                          "are not constant overhead, and tail quantiles must not be subtracted",
    }


def run_bench(paths, update_ids_fn, *, case_metadata, samples=1000, epochs=5, warmup=20,
              distributions=("hot", "random"), l2_disturb=False, scratch_bytes=None,
              comparisons=None, threshold_us=100.0, bootstrap_iterations=200,
              order_seed=4101604, output_dir=None, calibration_samples=100):
    """Return raw records and summaries from isolated, paired Graph replays.

    ``samples`` is the total PAIR count per distribution (not per epoch).
    Every replay produces separate GPU-chain and CPU-wall records.
    For smoke tests samples=100 is accepted and explicitly SCREEN_ONLY.
    'epochs' are contiguous subdivisions of this single trace/run, not fresh
    processes or independently seeded repeats. Instrumentation calibration
    runs after measurement and does not alter or subtract any recorded value.
    Caller must keep owner/imports/views alive until this function returns.
    If capture or execution fails, the function drains and resets all created
    graphs before propagating; it never swallows a CUDA failure as a pass.
    """
    if not paths or any(not isinstance(k, str) or not callable(v) for k, v in paths.items()):
        raise ValueError("nonempty named callables required")
    if not callable(update_ids_fn):
        raise ValueError("update_ids_fn required")
    if not isinstance(samples, int) or samples < 1 or not isinstance(epochs, int) or not 1 <= epochs <= samples:
        raise ValueError("require samples>=epochs>=1")
    if warmup < 1 or not distributions or len(set(distributions)) != len(distributions):
        raise ValueError("positive warmup and distinct distributions required")
    if any(d not in ("hot", "random") for d in distributions):
        raise ValueError("supported distributions: hot/random; metadata must describe their actual trace")
    if not math.isfinite(threshold_us) or threshold_us <= 0:
        raise ValueError("positive finite threshold required")
    if not isinstance(calibration_samples, int) or isinstance(calibration_samples, bool) or calibration_samples < 0:
        raise ValueError("calibration_samples must be a nonnegative integer")
    _validate_metadata(case_metadata)
    comparisons = (dict(comparisons) if comparisons is not None else
                   {k: v for k, v in DEFAULT_COMPARISONS.items() if all(x in paths for x in v)})
    for label, pair in comparisons.items():
        if len(pair) != 2 or any(name not in paths for name in pair):
            raise ValueError("unknown path in comparison " + label)
    result_dir = Path(output_dir) if output_dir is not None else None
    if result_dir is not None:
        result_dir.mkdir(parents=True, exist_ok=True)
        if any((result_dir / name).exists() for name in ("samples.jsonl", "summary.json")):
            raise FileExistsError("benchmark outputs already exist; use a new run directory")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU timings cannot stand in for the GPU benchmark")
    device = torch.cuda.current_device()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    l2 = _l2_bytes(torch, device)
    if l2_disturb:
        if l2 is None:
            raise RuntimeError("runtime L2 size unknown; refuse to guess a disturbance size")
        requested = int(scratch_bytes) if scratch_bytes is not None else 2 * l2
        if requested < 2 * l2 or requested > (1 << 30):
            raise ValueError("scratch must be >=2*runtime L2 and <=1 GiB")
    elif scratch_bytes is not None:
        raise ValueError("scratch_bytes requires l2_disturb=True")
    else:
        requested = 0

    graphs = {}
    outputs = {}
    events = {}
    records = []
    scratch = None
    thread_usage = _thread_switches_reader()
    try:
        with torch.inference_mode(), torch.cuda.stream(stream):
            if requested:
                scratch = torch.zeros((requested,), dtype=torch.uint8, device=device)
            # Precreate external events. Fail clearly on an unsupported PyTorch;
            # silently moving events outside capture would change the metric.
            try:
                events = {name: (torch.cuda.Event(enable_timing=True, external=True),
                                 torch.cuda.Event(enable_timing=True, external=True)) for name in paths}
            except TypeError as exc:
                raise RuntimeError("PyTorch external CUDA event nodes are required") from exc
            for iteration in range(warmup):
                update_ids_fn(iteration, distributions[iteration % len(distributions)])
                for name in balanced_order(paths, iteration, order_seed):
                    outputs[name] = paths[name]()
            stream.synchronize()
            for name, path in paths.items():
                graph = torch.cuda.CUDAGraph()
                graphs[name] = graph
                begin, end = events[name]
                with torch.cuda.graph(graph, stream=stream):
                    begin.record(stream)
                    outputs[name] = path()
                    end.record(stream)
            stream.synchronize()
            # Exercise replay and event completion before taking any sample.
            for name in paths:
                graphs[name].replay()
                events[name][1].synchronize()
            for distribution in distributions:
                for pair_id in range(samples):
                    epoch = min(pair_id * epochs // samples, epochs - 1)
                    update_ids_fn(pair_id, distribution)
                    order = balanced_order(paths, pair_id, order_seed)
                    for position, name in enumerate(order):
                        if scratch is not None:
                            scratch.add_(1)  # Controlled read/write, not a guarantee of globally cold caches.
                        # Exclude input update/disturbance and previous request.
                        stream.synchronize()
                        begin, end = events[name]
                        switches_before = thread_usage() if thread_usage is not None else None
                        wall_begin = time.perf_counter_ns()
                        graphs[name].replay()
                        replay_return = time.perf_counter_ns()
                        end.synchronize()
                        completion = time.perf_counter_ns()
                        wall_us = (completion - wall_begin) / 1000.0
                        switches_after = thread_usage() if thread_usage is not None else None
                        parts = _wall_parts(wall_begin, replay_return, completion,
                                            switches_before, switches_after)
                        gpu_us = begin.elapsed_time(end) * 1000.0
                        if not math.isfinite(gpu_us) or gpu_us <= 0 or not math.isfinite(wall_us) or wall_us <= 0:
                            raise RuntimeError("nonpositive/nonfinite event timestamp; invalid timing sample")
                        base = {"distribution": distribution, "epoch": epoch, "pair_id": pair_id,
                                "path": name, "order_index": position, "path_order": order,
                                "wall_begin_perf_counter_ns": wall_begin,
                                "wall_end_perf_counter_ns": completion, **parts}
                        records.append(dict(base, clock="gpu_graph_event", duration_us=gpu_us))
                        records.append(dict(base, clock="cpu_submit_to_event_complete", duration_us=wall_us))
    finally:
        # Keep views and owners alive in the caller until the submitted work is
        # drained, then destroy graph state before dropping capture outputs.
        # A drain/reset error propagates instead of generating successful stats.
        stream.synchronize()
        for graph in graphs.values():
            graph.reset()
        graphs.clear()
        outputs.clear()
        events.clear()
        scratch = None
        gc.collect()

    props = torch.cuda.get_device_properties(device)
    summary = summarize_records(records, comparisons, threshold_us=threshold_us,
                                bootstrap_iterations=bootstrap_iterations, seed=order_seed)
    summary.update({
        "case_metadata": case_metadata,
        "protocol": {"samples_per_distribution": samples, "epochs": epochs, "warmup": warmup,
                     "epoch_semantics": "contiguous sample blocks; single caller trace/run; not independent repeats",
                     "distributions": list(distributions), "order_seed": order_seed,
                     "graph": True, "timing_events": "captured_external_event_nodes",
                     "input_preparation_timed": False, "per_request_sync": True,
                     "cpu_wall_partition": "replay call then completion synchronization call; "
                                           "includes midpoint timestamp overhead; neither is exclusive GPU wait",
                     "thread_context_switch_counters": "RUSAGE_THREAD" if thread_usage is not None else "UNAVAILABLE",
                     "context_switch_sampling_envelope": "outside CPU timestamps, immediately before/after replay+sync; "
                                                          "correlation only, not scheduler attribution",
                     "measurement_scope": "isolated_component; not sustained queue/whole model",
                     "screening": samples < 10000,
                     "l2_disturb": l2_disturb, "runtime_l2_bytes": l2,
                     "scratch_bytes": requested,
                     "cache_claim": "controlled_scratch_disturbance_not_guaranteed_cold" if l2_disturb
                                    else "uncontrolled_cache_state; distribution defined by caller"},
        "runtime": {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                    "device": device, "gpu_name": props.name,
                    "capability": list(torch.cuda.get_device_capability(device))},
    })
    summary["instrumentation_control"] = (
        dict(calibrate_timing(calibration_samples), phase="post_measurement")
        if calibration_samples else {"status": "NOT_RUN", "reason": "calibration_samples=0"})
    if result_dir is not None:
        with (result_dir / "samples.jsonl").open("x") as file:
            for row in records:
                file.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        with (result_dir / "summary.json").open("x") as file:
            json.dump(summary, file, sort_keys=True, indent=2, allow_nan=False)
            file.write("\n")
    return {"summary": summary, "records": records}
