"""Compare CUDA SP padding primitives, excluding allocation and model execution.

Run with the libcuda_graph_prepare.so built from the same source checkout.
The baseline reproduces ATen copy/fill operations in Python; the candidate calls
the production CUDA primitive through ctypes. Neither runs PyWrappedModel.
CUDA graph replay amortizes Python submission costs equally for both paths.
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
import pathlib
import platform
import statistics
import subprocess
import sys

import torch
from torch.profiler import ProfilerActivity, profile


class Region(ctypes.Structure):
    _fields_ = [
        ("source", ctypes.c_void_p),
        ("destination", ctypes.c_void_p),
        ("source_group_bytes", ctypes.c_int64),
        ("destination_group_bytes", ctypes.c_int64),
        ("groups", ctypes.c_int64),
        ("fill_bits", ctypes.c_uint64),
        ("element_bytes", ctypes.c_int32),
    ]


class Params(ctypes.Structure):
    _fields_ = [("region_count", ctypes.c_int32), ("regions", Region * 32)]


def load_primitive(path):
    symbols = subprocess.check_output(["nm", "-D", str(path)], text=True)
    symbol = next(
        line.split()[-1]
        for line in symbols.splitlines()
        if " T " in line and "invokeSequenceParallelPad" in line
    )
    torch_lib = pathlib.Path(torch.__file__).parent / "lib"
    for name in ("libc10.so", "libtorch_cpu.so"):
        ctypes.CDLL(str(torch_lib / name), mode=ctypes.RTLD_GLOBAL)
    library = ctypes.CDLL(str(path))
    invoke = getattr(library, symbol)
    invoke.argtypes = [Params, ctypes.c_void_p]
    invoke.restype = None
    return library, invoke


def make_case(batch, physical, query, hidden, invoke):
    # Synthetic GPU-resident fields with the production contiguous layouts.
    sources = [
        torch.arange(batch * query, dtype=torch.int32, device="cuda"),
        torch.randn(batch * query, hidden, dtype=torch.bfloat16, device="cuda"),
        torch.full((batch,), query, dtype=torch.int32, device="cuda"),
        torch.full((batch,), 143634, dtype=torch.int32, device="cuda"),
        torch.ones(2, batch, 9, dtype=torch.int32, device="cuda"),
        torch.ones(2, batch, 144, dtype=torch.int32, device="cuda"),
    ]
    dims = [0, 0, 0, 0, 1, 1]
    extras = [(physical - batch) * query] * 2 + [physical - batch] * 4
    fills = [0, 0, query, 0, 0, 0]
    outputs = []
    params = Params()
    params.region_count = len(sources)
    for i, (src, dim, extra, fill) in enumerate(zip(sources, dims, extras, fills)):
        shape = list(src.shape)
        shape[dim] += extra
        dst = torch.empty(shape, dtype=src.dtype, device=src.device)
        outputs.append(dst)
        groups = math.prod(src.shape[:dim])
        row_bytes = src.numel() // groups // src.size(dim) * src.element_size()
        params.regions[i] = Region(
            src.data_ptr(),
            dst.data_ptr(),
            src.size(dim) * row_bytes,
            dst.size(dim) * row_bytes,
            groups,
            fill,
            src.element_size(),
        )

    def baseline():
        for src, dst, dim, extra, fill in zip(sources, outputs, dims, extras, fills):
            dst.narrow(dim, 0, src.size(dim)).copy_(src)
            dst.narrow(dim, src.size(dim), extra).fill_(fill)

    def candidate():
        invoke(params, torch.cuda.current_stream().cuda_stream)

    baseline()
    torch.cuda.synchronize()
    reference = [tensor.clone() for tensor in outputs]
    candidate()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(outputs, reference))
    # Keep all storage alive through capture, timing, and profiling.
    return sources, outputs, baseline, candidate


def measure(fn, calls, samples):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(samples):
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end) * 1000 / calls)
    return {"graph_replay_us_per_call": values, "p50_us": statistics.median(values)}


def profile_path(fn, output):
    iterations = 10
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(output))
    events = json.loads(output.read_text())["traceEvents"]
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy")]
    return {
        "trace": str(output),
        "kernels_per_call": sum(e["cat"] == "kernel" for e in gpu) / iterations,
        "memcopies_per_call": sum(e["cat"] == "gpu_memcpy" for e in gpu) / iterations,
        "profiled_gpu_sum_us_per_call": sum(e.get("dur", 0) for e in gpu) / iterations,
        "names": sorted({e["name"] for e in gpu}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=pathlib.Path, required=True)
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--batches", type=int, nargs="+", default=[13, 14, 28, 32])
    parser.add_argument("--queries", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--calls-per-graph", type=int, default=32)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    assert min(args.tp, args.hidden, args.calls_per_graph, args.samples) > 0
    assert min(args.batches + args.queries) > 0
    args.result_dir.mkdir(parents=True, exist_ok=False)
    library, invoke = load_primitive(args.library)
    torch.manual_seed(20261015)
    rows = []
    for query in args.queries:
        unit = args.tp // math.gcd(args.tp, query)
        for batch in args.batches:
            physical = math.ceil(batch / unit) * unit
            row = {"batch": batch, "query": query, "physical_batch": physical}
            rows.append(row)
            if physical == batch:
                row.update(
                    status="no_padding",
                    reason="Production returns before launching padding",
                )
                continue
            sources, outputs, baseline, candidate = make_case(
                batch, physical, query, args.hidden, invoke
            )
            row.update(
                status="measured", correctness="bitwise_equal", runs=[], profile={}
            )
            for order in (("baseline", "candidate"), ("candidate", "baseline")):
                run = {"order": order}
                for name in order:
                    fn = baseline if name == "baseline" else candidate
                    run[name] = measure(fn, args.calls_per_graph, args.samples)
                row["runs"].append(run)
            for name, fn in (("baseline", baseline), ("candidate", candidate)):
                row["profile"][name] = profile_path(
                    fn, args.result_dir / f"b{batch}_q{query}_{name}.json"
                )
            print(json.dumps(row), flush=True)
            del sources, outputs, baseline, candidate
    root = pathlib.Path(__file__).resolve().parents[3]
    report = {
        "command": sys.argv,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ),
        "script_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()
        ).hexdigest(),
        "library": str(args.library.resolve()),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "params_bytes": ctypes.sizeof(Params),
        "hidden": args.hidden,
        "tp_alignment_only": args.tp,
        "calls_per_graph": args.calls_per_graph,
        "boundary": "Six GPU padding fields; allocation, parameter packing, H2D, model and collectives excluded",
        "limitation": "Single GPU primitive comparison; neither path runs production PyWrappedModel; graph timing is not pure kernel time",
        "aggregation": "Single rank; p50 of replay CUDA-event time divided by calls per graph; AB then BA independent captures",
        "results": rows,
    }
    (args.result_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    del library


if __name__ == "__main__":
    main()
