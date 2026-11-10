"""Same-global-batch K3.1 DSA layer benchmark: DCP, DP or DP x smaller DCP domains.

Run each mode in a fresh eight-process invocation. DCP writes reference outputs;
Other topologies check those outputs by global request ID before timing. All enter the
real KimiK31DSA layer, with the same logical weights, history and hidden states.

  <native distributed launcher> kimi_k31_dsa_layer_bench.py \
      --config-json <checkpoint>/config.json --output-dir <evidence>/dcp
  <same launcher> kimi_k31_dsa_layer_bench.py --parallel-mode dp \
      --config-json <checkpoint>/config.json --reference-dir <evidence>/dcp \
      --output-dir <evidence>/dp
  <same launcher> kimi_k31_dsa_layer_bench.py --tp-size 4 \
      --config-json <checkpoint>/config.json --reference-dir <evidence>/dcp \
      --output-dir <evidence>/dp2dcp4

Latency is the median of rank-max, flush-separated body spans from one profiled
CUDA Graph burst. Outer CUDA-event samples are a separate flush-inclusive check.
This is a single Attention layer comparison, not service or accepted-token TPS.
"""

import argparse
import faulthandler
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    page_rr_mla_metadata,
)

try:
    from rtp_llm.test import kimi_k31_dsa_layer_fixture as fx
except ImportError:  # direct native script with the benchmark's pinned runfiles
    import kimi_k31_dsa_layer_fixture as fx

CAPTURE = (1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128)
K3_CASES = [
    [4, b, c]
    for b, c in (
        (32, 8 << 10),
        (128, 8 << 10),
        (12, 16 << 10),
        (13, 16 << 10),
        (32, 16 << 10),
        (68, 16 << 10),
        (128, 16 << 10),
        (32, 32 << 10),
        (34, 32 << 10),
        (128, 32 << 10),
        (17, 64 << 10),
        (32, 64 << 10),
        (128, 64 << 10),
        (8, 128 << 10),
        (32, 128 << 10),
        (68, 128 << 10),
        (4, 256 << 10),
        (8, 256 << 10),
        (34, 256 << 10),
        (2, 512 << 10),
        (17, 512 << 10),
        (1, 1 << 20),
        (8, 1 << 20),
    )
]


def parse_case(case):
    if isinstance(case, dict):
        return case["queries"], list(case["contexts"])
    queries, batch, context = case
    return queries, [context] * batch


def bucket_of(batch):
    for size in CAPTURE:
        if size >= batch:
            return size
    return math.ceil(batch / 16) * 16


def distribute_requests(contexts, owners):
    """Balance requests across DP owners; never split a request's Q rows between domains."""
    assignments = [[] for _ in range(owners)]
    loads = [0] * owners
    for request in sorted(range(len(contexts)), key=lambda i: (-contexts[i], i)):
        rank = min(range(owners), key=lambda r: (len(assignments[r]), loads[r], r))
        assignments[rank].append(request)
        loads[rank] += contexts[request]
    return [sorted(ids) for ids in assignments]


def write_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@triton.jit
def _k31_bench_flush_l2(Buffer, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Buffer + offset, 0, offset < N)


def parse_burst(path, inner, discard=0):
    """The dedicated flush kernel delimits each logical iteration, including side streams."""
    events = json.loads(path.read_text())["traceEvents"]
    gpu = sorted(
        (
            e
            for e in events
            if e.get("ph") == "X"
            and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
        ),
        key=lambda e: e["ts"],
    )
    flushes = [e for e in gpu if "_k31_bench_flush_l2" in e["name"]]
    if len(flushes) != inner:
        raise RuntimeError(f"{path}: expected {inner} flushes, got {len(flushes)}")
    samples, kernels = [], {}
    # CUPTI reported up to 0.384 us apparent overlap for ordered same-stream kernels.
    # This guards exported timestamps; it never adjusts the measured body spans.
    timestamp_tolerance_us = 1.0
    for i, flush in enumerate(flushes):
        start = flush["ts"] + flush["dur"]
        stop = flushes[i + 1]["ts"] if i + 1 < inner else float("inf")
        body = [e for e in gpu if flush["ts"] <= e["ts"] < stop and e not in flushes]
        if (
            not body
            or body[0]["ts"] < start - timestamp_tolerance_us
            or any(e["ts"] + e["dur"] > stop + timestamp_tolerance_us for e in body)
        ):
            raise RuntimeError(
                f"{path}: incomplete or overlapping body at iteration {i}"
            )
        samples.append(
            {
                "iteration": i,
                "body_us": max(e["ts"] + e["dur"] for e in body) - body[0]["ts"],
                "flush_us": flush["dur"],
                "gpu_events": len(body),
            }
        )
        if i >= discard:
            for event in body:
                kernels[event["name"]] = kernels.get(event["name"], 0.0) + event[
                    "dur"
                ] / (inner - discard)
    if len({s["gpu_events"] for s in samples}) != 1:
        raise RuntimeError(
            f"{path}: fixed graph iterations have different GPU event counts"
        )
    measured_events = [e for e in gpu if e["ts"] >= flushes[0]["ts"]]
    if sum(s["gpu_events"] for s in samples) + len(flushes) != len(measured_events):
        raise RuntimeError(f"{path}: GPU events were lost between iteration boundaries")
    return samples, sorted(
        ((value, name) for name, value in kernels.items()), reverse=True
    )


def measure(body, flush, inner, replays, trace_path, discard):
    """All ranks participate in control collectives; idle DP ranks launch no timed work."""
    from torch.profiler import ProfilerActivity, profile

    graph = torch.cuda.CUDAGraph() if body is not None else None
    dist.barrier()
    torch.cuda.synchronize()
    if graph is not None:
        # JIT outside capture, and use a distinct kernel so trace boundaries are unambiguous.
        _k31_bench_flush_l2[(triton.cdiv(flush.numel(), 8192),)](
            flush, flush.numel(), 8192
        )
        with torch.cuda.graph(graph):
            for _ in range(inner):
                _k31_bench_flush_l2[(triton.cdiv(flush.numel(), 8192),)](
                    flush, flush.numel(), 8192
                )
                body()
        for _ in range(5):
            graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    bare = []
    for _ in range(replays):
        dist.barrier()
        if graph is not None:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            bare.append(begin.elapsed_time(end) * 1000)
        else:
            bare.append(0.0)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        dist.barrier()
        torch.cuda.synchronize()
        if graph is not None:
            graph.replay()
        torch.cuda.synchronize()
    if graph is not None:
        prof.export_chrome_trace(str(trace_path))
        graph.reset()
        samples, kernels = parse_burst(trace_path, inner, discard)
    else:
        samples = [
            {"iteration": i, "body_us": 0.0, "flush_us": 0.0, "gpu_events": 0}
            for i in range(inner)
        ]
        kernels = []
    local = {
        "rank": dist.get_rank(),
        "active": body is not None,
        "samples": samples,
        "bare_graph_us": bare,
        "kernels": kernels[:25],
    }
    write_json(trace_path.with_suffix(".samples.json"), local)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    critical = []
    for i in range(discard, inner):
        rank = max(gathered, key=lambda r: r["samples"][i]["body_us"])
        critical.append(
            {
                "iteration": i,
                "rank": rank["rank"],
                "body_us": rank["samples"][i]["body_us"],
            }
        )
    values = [s["body_us"] for s in critical]
    bare_max = [max(r["bare_graph_us"][i] for r in gathered) for i in range(replays)]
    return {
        "layer_us": statistics.median(values),
        "p90_us": sorted(values)[math.ceil(0.9 * len(values)) - 1],
        "max_us": max(values),
        "critical_samples": critical,
        "bare_graph_us": bare_max,
        "bare_graph_per_inner_us": statistics.median(bare_max) / inner,
        "per_rank": gathered,
    }


def assemble_output(output, ids, queries, tp, batch):
    local = (
        output.detach().cpu()
        if output is not None
        else torch.empty(0, 0, dtype=torch.bfloat16)
    )
    payload = {"ids": ids, "output": local}
    parts = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(payload, parts, dst=0)
    if parts is None:
        return None
    hidden = next(part["output"].shape[-1] for part in parts if part["ids"])
    full = torch.empty(batch, queries, hidden, dtype=torch.bfloat16)
    for first in range(0, len(parts), tp):
        domain = parts[first : first + tp]
        request_ids = domain[0]["ids"]
        if any(part["ids"] != request_ids for part in domain):
            raise RuntimeError(
                "ranks in one DCP domain have different request assignments"
            )
        if request_ids:
            domain_output = torch.cat([part["output"] for part in domain])
            full[request_ids] = domain_output[: len(request_ids) * queries].view(
                -1, queries, hidden
            )
    return full.flatten(0, 1)


def configure_collectives(layer, backend, capacity, hidden_size):
    """Benchmark existing BF16 collectives without changing projection implementations."""
    if backend == "nccl":
        return
    if layer.prologue == "dev" or layer._fp8_collective is not None:
        raise ValueError("collective comparison requires the unchanged DCP prologue/output GEMM")
    collectives = fx.collective.Bf16TpCollectives(
        torch.device("cuda", torch.cuda.current_device()),
        max_tokens=capacity,
        hidden_size=hidden_size,
    )
    if backend in ("custom_ag", "custom_ag_rs"):
        if collectives.gather is None:
            raise RuntimeError("requested CustomAllGather is unavailable; refusing NCCL fallback")

        def gather(state):
            state.full = collectives.all_gather(state.hidden)

        # The same DAG nodes also appear in the finalized capture order.
        for node in layer._dag._nodes:
            if node.name == "gather":
                object.__setattr__(node, "fn", gather)
    if backend in ("push_rs", "custom_ag_rs"):
        if collectives.scatter is None:
            raise RuntimeError("requested PushReduceScatter is unavailable; refusing NCCL fallback")
        # Only the final BF16 RS uses this owner on the multistream path.
        # Do not install Fp8CollectiveProjection: that would also change GEMM.
        layer._mtp_bf16_collectives = collectives


def validate_output(output, signature, case_dir, reference_dir, tolerance, l2_tolerance):
    report = {"finite": True, "parity": "REFERENCE", "row_rel_max": 0.0, "rel_l2": 0.0}
    error = None
    if dist.get_rank() == 0:
        if not torch.isfinite(output).all():
            error = "layer output is not finite"
            report["finite"] = False
        if reference_dir is not None:
            reference = torch.load(
                reference_dir / "output.pt", map_location="cpu", weights_only=True
            )
            if reference["signature"] != signature:
                error = "reference input/configuration signature differs"
            else:
                expected = reference["output"].float()
                actual = output.float()
                relative = (actual - expected).abs().amax(1) / expected.abs().amax(
                    1
                ).clamp(min=1e-6)
                report.update(
                    parity="PASS",
                    row_rel_max=float(relative.max()),
                    rel_l2=float(
                        (actual - expected).norm() / expected.norm().clamp(min=1e-30)
                    ),
                )
                if not torch.isfinite(relative).all() or relative.max() > tolerance:
                    error = f"DP/DCP output row relative error {float(relative.max()):.6g} > {tolerance}"
                elif not math.isfinite(report["rel_l2"]) or report["rel_l2"] > l2_tolerance:
                    error = f"output global relative L2 {report['rel_l2']:.6g} > {l2_tolerance}"
        if error:
            report["parity"] = "FAIL"
            report["error"] = error
        write_json(case_dir / "accuracy.json", report)
        if not error:
            temporary = case_dir / "output.pt.tmp"
            torch.save({"signature": signature, "output": output}, temporary)
            temporary.replace(case_dir / "output.pt")
    message = [report if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(message, src=0)
    if message[0]["parity"] == "FAIL":
        raise RuntimeError(message[0]["error"])
    return message[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--cases", default=json.dumps(K3_CASES))
    parser.add_argument("--parallel-mode", choices=("dcp", "dp"), default="dcp")
    parser.add_argument(
        "--tp-size",
        type=int,
        help="DCP domain size; default world, or 1 for --parallel-mode dp",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--max-seq-len", type=int, default=1 << 20)
    parser.add_argument("--inner", type=int, default=16)
    parser.add_argument(
        "--discard",
        type=int,
        default=4,
        help="initial profiled iterations excluded from statistics",
    )
    parser.add_argument(
        "--replays",
        type=int,
        default=5,
        help="unprofiled flush-inclusive sanity samples",
    )
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--parity-tolerance", type=float, default=2e-2)
    parser.add_argument("--global-l2-tolerance", type=float, default=2e-2)
    parser.add_argument(
        "--nccl-max-ctas", type=int,
        help="diagnostic initialization override; otherwise retain the fixture default",
    )
    parser.add_argument(
        "--collective-backend", default="nccl",
        choices=("nccl", "custom_ag", "push_rs", "custom_ag_rs"),
        help="existing BF16 AG/RS only; projection and attention implementations stay fixed",
    )
    parser.add_argument(
        "--kernels",
        action="store_true",
        help="print rank 0 kernel durations from the same trace",
    )
    parser.add_argument(
        "--fp8-gemm", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fp8-kv-cache", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    cases = [parse_case(case) for case in json.loads(args.cases)]
    if not 0 <= args.discard < args.inner or args.replays < 1:
        parser.error("need 0 <= discard < inner and replays >= 1")
    if not cases or any(
        not c or q < 1 or min(c) < q or max(c) > args.max_seq_len for q, c in cases
    ):
        parser.error("cases must contain requests with Q <= length <= max-seq-len")
    world = int(os.environ["WORLD_SIZE"])
    tp_size = (
        args.tp_size
        if args.tp_size is not None
        else (1 if args.parallel_mode == "dp" else world)
    )
    if tp_size < 1 or world % tp_size:
        parser.error("DCP domain size must be a positive divisor of WORLD_SIZE")
    if args.parallel_mode == "dp" and tp_size != 1:
        parser.error("--parallel-mode dp requires --tp-size 1")
    if tp_size != world and args.reference_dir is None:
        parser.error(
            "partitioned requests require --reference-dir from a matching full-DCP run"
        )
    if args.collective_backend != "nccl" and tp_size == 1:
        parser.error("custom collective comparison requires a DCP domain larger than one")
    if args.nccl_max_ctas is not None:
        if args.nccl_max_ctas < 1:
            parser.error("--nccl-max-ctas must be positive")
        os.environ["NCCL_MAX_CTAS"] = str(args.nccl_max_ctas)
    parallel = fx.init_distributed(tp_size)
    tp, rank, world_rank = parallel.tp_size, parallel.tp_rank, dist.get_rank()
    domains, domain_rank = world // tp, world_rank // tp
    mode = "dp" if tp == 1 else ("dcp" if domains == 1 else f"dp{domains}dcp{tp}")
    actual_tp = dist.get_world_size(
        fx.collective.get_process_group(fx.collective.Group.TP)
    )
    if actual_tp != tp:
        raise RuntimeError(f"Attention TP group has {actual_tp} ranks, expected {tp}")
    if world_rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    manifest = {
        "mode": mode,
        "world": world,
        "tp": tp,
        "dp": world // tp,
        "cases": [{"q": q, "contexts": c} for q, c in cases],
        "seed": args.seed,
        "max_seq_len": args.max_seq_len,
        "fp8_gemm": args.fp8_gemm,
        "fp8_kv_cache": args.fp8_kv_cache,
        "parity_tolerance": args.parity_tolerance,
        "global_l2_tolerance": args.global_l2_tolerance,
        "collective_backend": args.collective_backend,
        "nccl_version": torch.cuda.nccl.version(),
        "nccl_environment": {
            name: os.environ.get(name)
            for name in ("NCCL_MAX_CTAS", "NCCL_MIN_CTAS", "NCCL_ALGO", "NCCL_PROTO",
                         "NCCL_GRAPH_REGISTER", "NCCL_NVLS_ENABLE")
        },
        "inner": args.inner,
        "discard": args.discard,
        "replays": args.replays,
        "cache_mode": "cold",
        "scope": "KimiK31DSA.forward",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(fx.__file__),
                Path(fx.collective.__file__),
                Path(page_rr_mla_metadata.__file__),
            )
        },
        "selectors": {
            name: os.environ.get(name)
            for name in ("DSA_DCP_TOPK", "DSA_DCP_COMBINE", "DSA_DCP_PROLOGUE")
        },
    }
    if world_rank == 0:
        write_json(args.output_dir / "manifest.json", manifest)
    flush = torch.empty(
        2 * torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size,
        dtype=torch.uint8,
        device="cuda",
    )
    results = []
    for case_index, (queries, contexts) in enumerate(cases):
        faulthandler.dump_traceback_later(1800, exit=True)
        case_dir = args.output_dir / f"case{case_index:03d}"
        if world_rank == 0:
            case_dir.mkdir()
        dist.barrier()
        assignments = distribute_requests(contexts, domains)
        ids = assignments[domain_rank]
        active = bool(ids)
        config = fx.load_config(
            args.config_json,
            max_seq_len=args.max_seq_len,
            queries=queries,
            fp8_kv_cache=args.fp8_kv_cache,
            fp8_gemm=args.fp8_gemm,
        )
        slot = layer = owner = recorder = weights = weights_by_layer = pools = None
        requests = []
        workload = {
            "rank": world_rank,
            "domain": domain_rank,
            "tp_rank": rank,
            "request_ids": ids,
            "contexts": [contexts[i] for i in ids],
            "active": active,
            "bucket": 0,
            "padded_rows": 0,
            "cache_bytes": {},
            "mega_topk": False,
            "decode_collectives": [],
        }
        if active:
            layer_index = config.k3_runtime_config.dsa.layers[0]
            weights, _ = fx.loader_weights(config, parallel, seed=args.seed)
            layer = fx.build_layer(config, parallel, weights, layer_index)
            weights_by_layer = [None] * config.num_layers
            weights_by_layer[layer_index] = weights
            bucket = bucket_of(len(ids))
            padded = math.ceil(bucket / fx.sp_unit(tp, queries)) * fx.sp_unit(
                tp, queries
            )
            configure_collectives(
                layer, args.collective_backend, padded * queries, config.hidden_size
            )
            # The default backend serves production decode collectives; the comparison
            # backends keep their NCCL-fallback contract and attach nothing.
            attached = (
                fx.attach_decode_collectives(layer, config, parallel)
                if args.collective_backend == "nccl"
                else []
            )
            owner = fx.build_owner(
                config,
                parallel,
                [(layer_index, weights)],
                max_requests=padded,
                queries=queries,
            )
            recorder = fx.Recorder(layer)
            columns = sum(fx.owner_columns(contexts[i], tp, rank) for i in ids)
            pools = {
                tag: fx.Pool(tag, columns + 2, "cuda", fp8_kv_cache=args.fp8_kv_cache)
                for tag in fx.TAGS
            }
            generator = torch.Generator().manual_seed(args.seed + world_rank)
            for i in ids:
                request = fx.Request(
                    args.seed * 1000 + i, contexts[i] - queries, contexts[i]
                )
                if not request.allocate(pools, tp, rank, generator):
                    raise RuntimeError("bench page pool is too small")
                request.write_history(pools, tp, rank, global_history=True)
                requests.append(request)
            slot = fx.GraphSlot(
                config,
                parallel,
                [layer],
                weights_by_layer,
                owner,
                recorder,
                pools,
                queries=queries,
                bucket=bucket,
                max_seq_len=args.max_seq_len,
                pool_handle=torch.cuda.graph_pool_handle(),
            )
            hidden_gen = torch.Generator(device="cuda").manual_seed(args.seed * 13)
            global_hidden = torch.randn(
                len(contexts),
                queries,
                config.hidden_size,
                generator=hidden_gen,
                device="cuda",
            ).bfloat16()
            local_hidden = torch.zeros(
                slot.padded * queries,
                config.hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
            )
            local_hidden[: len(ids) * queries] = global_hidden[ids].flatten(0, 1)
            rows = slot.hidden.shape[0]
            slot.load(requests, local_hidden[rank * rows : (rank + 1) * rows])
            output = slot.replay()[0]
            torch.cuda.synchronize()
            execution = slot.fmha["indexer_kv"].page_rr_params.indexer_execution
            workload.update(
                bucket=bucket,
                decode_collectives=attached,
                padded_rows=slot.padded * queries,
                cache_bytes={
                    tag: pool.cache.numel() * pool.cache.element_size()
                    for tag, pool in pools.items()
                },
                indexer_planned_bytes=execution.required_bytes,
                topk_slots=slot.recorded["selected"].shape[-1],
                owned_topk_mean=float(
                    (slot.recorded["selected"][: len(ids) * queries] >= 0)
                    .sum(-1)
                    .float()
                    .mean()
                ),
                mega_topk=execution.mega_topk,
                mega_a2a=slot.fmha["full"].fmha_impl.mega_a2a is not None,
                direct_indexer_page=execution.direct_native_page_size,
                prologue=layer.prologue,
                kv_blocks=[
                    len(execution.blocks),
                    len(slot.fmha["full"].page_rr_params.sfa_execution.blocks),
                ],
            )
        else:
            output = None
        full_output = assemble_output(output, ids, queries, tp, len(contexts))
        signature = {
            "q": queries,
            "contexts": contexts,
            "seed": args.seed,
            "max_seq_len": args.max_seq_len,
            "fp8_gemm": args.fp8_gemm,
            "fp8_kv_cache": args.fp8_kv_cache,
            "config_sha256": hashlib.sha256(
                Path(args.config_json).read_bytes()
            ).hexdigest(),
            "source_sha256": manifest["source_sha256"],
        }
        accuracy = validate_output(
            full_output,
            signature,
            case_dir,
            (
                args.reference_dir / f"case{case_index:03d}"
                if args.reference_dir
                else None
            ),
            args.parity_tolerance,
            args.global_l2_tolerance,
        )
        metric = measure(
            slot.forward if active else None,
            flush,
            args.inner,
            args.replays,
            case_dir / f"trace_rank{world_rank}.json",
            args.discard,
        )
        workloads = [None] * world
        dist.all_gather_object(workloads, workload)
        result = {
            "case": case_index,
            "mode": mode,
            "q": queries,
            "batch": len(contexts),
            "contexts": contexts,
            "accuracy": accuracy,
            "workloads": workloads,
            **metric,
        }
        result["query_rows_per_s"] = len(contexts) * queries * 1e6 / metric["layer_us"]
        if world_rank == 0:
            write_json(case_dir / "result.json", result)
            results.append(result)
            print(
                "BENCH "
                + json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "case",
                            "mode",
                            "q",
                            "batch",
                            "layer_us",
                            "p90_us",
                            "query_rows_per_s",
                            "accuracy",
                        )
                    }
                ),
                flush=True,
            )
            if args.kernels:
                print(
                    "KERNELS " + json.dumps(metric["per_rank"][0]["kernels"]),
                    flush=True,
                )
        if slot is not None:
            slot.graph.reset()
        for request in requests:
            request.release(pools)
        del (
            slot,
            layer,
            owner,
            recorder,
            weights,
            weights_by_layer,
            pools,
            requests,
            output,
            full_output,
        )
        torch.cuda.empty_cache()
    faulthandler.cancel_dump_traceback_later()
    if world_rank == 0:
        write_json(args.output_dir / "results.json", results)
    dist.barrier()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
