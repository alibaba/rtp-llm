"""Compare paged KDA target verification with a supplied vLLM source file.

Example (run under a GPU lock):
    python benchmark/benchmark_kimi_kda_short_conv.py \
        --vllm-source /path/to/vllm/model_executor/layers/mamba/ops/causal_conv1d.py \
        --output /path/to/results

Only the selected functions are imported, without either engine's package
initialization. Kernel bodies are copied verbatim into recorded modules.
Allocation, compilation, correctness checks and graph capture are not timed.
CUDA events measure amortized graph stream latency, including native PDL
overlap. Serialized profiler replays provide isolated kernel durations.
Timings use hot caches and repeated in-place updates.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import triton

RTP_KERNEL = "_kimi_kda_short_conv_paged_target_verify_kernel"
VLLM_KERNEL = "_causal_conv1d_update_kernel"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_functions(path: Path, names: list[str], output: Path):
    """Load trusted local functions without importing the surrounding engine."""
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    nodes = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    segments = []
    for name in names:
        node = nodes[name]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        segments.append("".join(lines[start - 1 : node.end_lineno]))
    output.write_text(
        "from __future__ import annotations\n"
        "import torch\nimport triton\nimport triton.language as tl\n"
        "NULL_BLOCK_ID = 0\nPAD_SLOT_ID = -1\n\n" + "\n\n".join(segments)
    )
    spec = importlib.util.spec_from_file_location(output.stem, output)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.current_platform = SimpleNamespace(
        is_arch_support_pdl=lambda: torch.cuda.get_device_capability()[0] >= 9
    )
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def preallocated_rtp_launcher(source: Path, output: Path, module):
    """Change only wrapper allocation; retain its launch arguments verbatim."""
    tree = ast.parse(source.read_text())
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "kimi_kda_short_conv_paged_target_verify"
    )
    assignments = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "output" for t in node.targets)
    ]
    if len(assignments) != 1 or ast.unparse(assignments[0].value.func) != "torch.empty":
        raise ValueError("RTP wrapper allocation changed; review the adapter")
    assignments[0].value = ast.Name(id="preallocated_output", ctx=ast.Load())
    wrapper.args.kwonlyargs.append(ast.arg(arg="preallocated_output"))
    wrapper.args.kw_defaults.append(None)
    wrapper.decorator_list = []
    wrapper.name = "launch_rtp"
    ast.fix_missing_locations(wrapper)
    output.write_text(ast.unparse(wrapper) + "\n")
    exec(compile(output.read_text(), str(output), "exec"), module.__dict__)
    return module.launch_rtp


def stats(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No timing samples")
    return {
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.mean(ordered),
        "samples": len(ordered),
    }


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = (actual.float() - expected.float()).abs()
    tolerance = 2**-10 + 2**-7 * expected.float().abs()
    finite = torch.isfinite(actual).all() & torch.isfinite(expected).all()
    return {
        "passed": bool(finite.item()) and bool((difference <= tolerance).all().item()),
        "max_abs": float(difference.max().item()),
        "rms": float(difference.square().mean().sqrt().item()),
        "bitwise_equal_fraction": float((actual == expected).float().mean().item()),
        "rtol": 2**-7,
        "atol": 2**-10,
    }


class Inputs:
    def __init__(self, batch: int, steps: int, dim: int, layout: str, boundary: str):
        self.batch, self.steps, self.dim = batch, steps, dim
        self.page_size = 4096
        channels, width = 3 * dim, 4
        # Include the other projection fields in the physical token row.
        token_stride = 4 * dim + 128 + max(1, dim // 128)
        packed = torch.randn(
            batch, steps, token_stride, device="cuda", dtype=torch.bfloat16
        )
        self.x = packed[..., :channels]
        self.q, self.k, self.v = self.x.split(dim, dim=-1)
        self.weight = torch.randn(channels, width, device="cuda", dtype=torch.float32)
        self.history = torch.randn(
            batch, width - 1, channels, device="cuda", dtype=torch.bfloat16
        )
        patterns = {
            "in_page": [12],
            "boundary": [4096, 4097, 4095, 1],
        }
        host_lengths = [
            patterns[boundary][b % len(patterns[boundary])] for b in range(batch)
        ]
        self.lengths = torch.tensor(host_lengths, device="cuda", dtype=torch.int32)
        self.history[self.lengths == 1] = 0
        pages = steps + 3
        self.block_map = torch.arange(
            1, batch * pages + 1, device="cuda", dtype=torch.int32
        )
        self.block_map = self.block_map.reshape(batch, pages).flip(1).contiguous()
        blocks = batch * pages + 1
        # Model the production SSM+conv allocation without executing recurrence.
        ssm_bytes = max(1, (dim + 127) // 128) * 128 * 128 * 4
        block_elements = ssm_bytes // 2 + (width - 1) * channels
        self.storage = torch.zeros(
            blocks, block_elements, device="cuda", dtype=torch.bfloat16
        )
        self.rtp_state = self.storage[:, ssm_bytes // 2 :].view(
            blocks, width - 1, channels
        )
        self.state_len = width + steps - 2
        if layout == "SD":
            state = torch.zeros(
                batch + 1, self.state_len, channels, device="cuda", dtype=torch.bfloat16
            )
            self.vllm_state = state.transpose(1, 2)
        else:
            self.vllm_state = torch.zeros(
                batch + 1, channels, self.state_len, device="cuda", dtype=torch.bfloat16
            )
        self.indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
        self.accepted = torch.ones(batch, device="cuda", dtype=torch.int32)
        self.cu = torch.arange(
            0, (batch + 1) * steps, steps, device="cuda", dtype=torch.int32
        )
        self.rtp_out = torch.empty(
            3, batch, steps, dim, device="cuda", dtype=torch.bfloat16
        )
        self.vllm_out = torch.empty(
            batch * steps, channels, device="cuda", dtype=torch.bfloat16
        )
        self.initial_blocks = self.block_map[
            torch.arange(batch, device="cuda"),
            torch.div(
                self.lengths.to(torch.int64) - 2, self.page_size, rounding_mode="floor"
            ).clamp_min(0),
        ].long()
        self.reset()

    def reset(self):
        self.rtp_state.zero_()
        self.rtp_state[self.initial_blocks] = self.history
        self.vllm_state.zero_()
        self.vllm_state[1:, :, :3] = self.history.transpose(1, 2)

    def oracle(self):
        window = torch.cat((self.history.float(), self.x.float()), dim=1)
        output = []
        checkpoints = []
        for t in range(self.steps):
            acc = (window[:, t : t + 4] * self.weight.T).sum(dim=1)
            output.append(torch.nn.functional.silu(acc).to(torch.bfloat16))
            checkpoints.append(window[:, t + 1 : t + 4].to(torch.bfloat16))
        return torch.stack(output, dim=1), checkpoints

    def check(self, rtp, vllm):
        self.reset()
        expected, checkpoints = self.oracle()
        rtp()
        vllm()
        rtp_out = self.rtp_out.permute(1, 2, 0, 3).reshape_as(expected)
        vllm_out = self.vllm_out.reshape_as(expected)
        result = {
            "rtp_vs_oracle": compare(rtp_out, expected),
            "vllm_vs_oracle": compare(vllm_out, expected),
            "rtp_vs_vllm": compare(rtp_out, vllm_out),
        }
        base = torch.div(
            self.lengths.to(torch.int64) - 1, self.page_size, rounding_mode="floor"
        )
        batch_idx = torch.arange(self.batch, device="cuda")
        # The compact strip is [history1, history2, x0, ..., x(T-1)].
        # Window t:t+3 is exactly the checkpoint after accepting token t.
        strip = self.vllm_state[1:].transpose(1, 2)
        checks = []
        for t, expected_state in enumerate(checkpoints):
            blocks = self.block_map[batch_idx, base + t].long()
            checks.append(bool(torch.equal(self.rtp_state[blocks], expected_state)))
            checks.append(bool(torch.equal(strip[:, t : t + 3], expected_state)))
        result["all_acceptance_windows_bitwise"] = all(checks)
        result["sentinel_unchanged"] = bool((self.rtp_state[0] == 0).all().item())
        result["passed"] = (
            all(item["passed"] for item in result.values() if isinstance(item, dict))
            and all(checks)
            and result["sentinel_unchanged"]
        )
        if not result["passed"]:
            raise AssertionError(json.dumps(result))
        self.reset()
        return result


def profile_graph(graph, kernel_name, path, nodes, repeats, discard, serialize):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(repeats + discard):
            graph.replay()
            if serialize:
                # PDL may overlap successive kernels even on one stream.
                # Synchronizing one-node replays prevents that overlap; the
                # CPU wait is excluded from the CUDA kernel event duration.
                torch.cuda.synchronize()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    kernels = sorted(
        [e for e in trace["traceEvents"] if e.get("cat") == "kernel"],
        key=lambda e: e["ts"],
    )
    expected_count = nodes * (repeats + discard)
    valid = len(kernels) == expected_count and all(
        kernel_name in e["name"] for e in kernels
    )
    if not valid:
        raise AssertionError(
            f"Unexpected dispatch: expected {expected_count} {kernel_name}, got {len(kernels)} {[e['name'] for e in kernels[:8]]}"
        )
    kernels = kernels[discard * nodes :]
    overlap_pairs = sum(
        a["ts"] + a["dur"] - b["ts"] > 0.01 for a, b in zip(kernels, kernels[1:])
    )
    if serialize and overlap_pairs:
        raise AssertionError("Serialized profiler replays still overlap")
    kernel_us = [float(e["dur"]) for e in kernels]
    return {
        "kernel_duration_us": stats(kernel_us),
        "kernel_sum_us_per_iter": statistics.mean(kernel_us),
        "kernel_names": sorted({e["name"] for e in kernels}),
        "launches_per_iter": 1,
        "dispatch_verified": valid,
        "serialized_replays": serialize,
        "overlap_pairs": overlap_pairs,
        "profiled_kernel_count": expected_count,
        "discarded_kernel_count": discard * nodes,
        "graph_replays_in_trace": sum(
            "cudaGraphLaunch" in e.get("name", "") for e in trace["traceEvents"]
        ),
        "trace": str(path),
    }


def measure(fn, expected_kernel: str, args, trace_path: Path):
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.graph_nodes):
            fn()
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    for start, end in zip(starts, ends):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    event_us = [
        start.elapsed_time(end) * 1000 / args.graph_nodes
        for start, end in zip(starts, ends)
    ]
    # Native graph traces diagnose overlap; their summed durations are not
    # isolated latency or throughput when PDL is active.
    native = profile_graph(
        graph,
        expected_kernel,
        trace_path.with_name(trace_path.stem + "_native_graph.json"),
        args.graph_nodes,
        repeats=2,
        discard=1,
        serialize=False,
    )
    isolated_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(isolated_graph):
        fn()
    isolated = profile_graph(
        isolated_graph,
        expected_kernel,
        trace_path,
        nodes=1,
        repeats=args.profile_iterations,
        discard=5,
        serialize=True,
    )
    return {
        **isolated,
        "graph_stream_us_per_op": stats(event_us),
        "native_graph_profile": native,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--rtp-source",
        type=Path,
        default=root / "rtp_llm/models_py/triton_kernels/kimi_kda/short_conv.py",
    )
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batches", type=int, nargs="+", default=[1, 2, 3, 4, 8, 16, 17, 32, 64]
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--dims", type=int, nargs="+", default=[1536])
    parser.add_argument("--layouts", nargs="+", choices=["SD", "DS"], default=["SD"])
    parser.add_argument(
        "--boundaries", nargs="+", choices=["in_page", "boundary"], default=["in_page"]
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile-iterations", type=int, default=30)
    parser.add_argument("--graph-nodes", type=int, default=32)
    args = parser.parse_args()
    if any(
        value <= 0
        for value in args.batches
        + args.steps
        + args.dims
        + [args.warmup, args.iterations, args.profile_iterations, args.graph_nodes]
    ):
        parser.error("Shapes and iteration counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    rtp_module = load_functions(
        args.rtp_source,
        [RTP_KERNEL, "is_kimi_kda_short_conv_paged_decode_supported"],
        args.output / "rtp_functions.py",
    )
    rtp_launch = preallocated_rtp_launcher(
        args.rtp_source, args.output / "rtp_launcher.py", rtp_module
    )
    vllm_module = load_functions(
        args.vllm_source,
        [VLLM_KERNEL, "causal_conv1d_update"],
        args.output / "vllm_functions.py",
    )
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    result = {
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "gpu": props.name,
            "capability": list(torch.cuda.get_device_capability()),
            "sm_count": props.multi_processor_count,
            "host": platform.node(),
        },
        "sources": {
            "rtp": {"path": str(args.rtp_source), "sha256": sha256(args.rtp_source)},
            "vllm": {"path": str(args.vllm_source), "sha256": sha256(args.vllm_source)},
            "benchmark_sha256": sha256(Path(__file__)),
        },
        "contract": {
            "torch_compile": False,
            "allocation_timed": False,
            "jit_warmup_timed": False,
            "cache": "hot; repeated in-place state updates after separate correctness reset",
            "weight_dtype": "float32",
            "activation_state_dtype": "bfloat16",
            "width": 4,
            "vllm_pdl": props.major >= 9,
            "primary_metric": "isolated CUDA kernel duration: serialized one-node graph replays, profiler CPU gaps excluded, first 5 kernels discarded",
            "secondary_metric": "unprofiled native graph stream us/op including PDL overlap, amortized across graph_nodes; throughput, not isolated latency",
        },
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "cases": [],
    }
    try:
        result["source_head"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        result["source_head"] = None
    result_path = args.output / "results.json"
    for dim in args.dims:
        for steps in args.steps:
            for batch in args.batches:
                for layout in args.layouts:
                    for boundary in args.boundaries:
                        torch.manual_seed(20260911 + dim + steps + batch)
                        inputs = Inputs(batch, steps, dim, layout, boundary)
                        case = {
                            "B": batch,
                            "T": steps,
                            "D": dim,
                            "layout": layout,
                            "boundary": boundary,
                        }
                        label = f"b{batch}_t{steps}_d{dim}_{layout}_{boundary}"

                        def rtp():
                            rtp_launch(
                                inputs.q,
                                inputs.k,
                                inputs.v,
                                inputs.weight,
                                inputs.rtp_state,
                                inputs.block_map,
                                inputs.lengths,
                                inputs.page_size,
                                preallocated_output=inputs.rtp_out,
                            )

                        def vllm():
                            vllm_module.causal_conv1d_update(
                                inputs.x.reshape(batch * steps, 3 * dim),
                                inputs.vllm_state,
                                inputs.weight,
                                activation="silu",
                                conv_state_indices=inputs.indices,
                                num_accepted_tokens=inputs.accepted,
                                query_start_loc=inputs.cu,
                                max_query_len=steps,
                                out=inputs.vllm_out,
                            )

                        try:
                            case["correctness"] = inputs.check(rtp, vllm)
                            case["strides"] = {
                                "q": list(inputs.q.stride()),
                                "rtp_state": list(inputs.rtp_state.stride()),
                                "vllm_state": list(inputs.vllm_state.stride()),
                            }
                            variants = [
                                ("rtp", rtp, RTP_KERNEL),
                                ("vllm", vllm, VLLM_KERNEL),
                            ]
                            if len(result["cases"]) % 2:
                                variants.reverse()
                            for name, fn, kernel in variants:
                                inputs.reset()
                                case[name] = measure(
                                    fn,
                                    kernel,
                                    args,
                                    args.output / f"{label}_{name}.json",
                                )
                            case["rtp_over_vllm_kernel_ratio"] = (
                                case["rtp"]["kernel_sum_us_per_iter"]
                                / case["vllm"]["kernel_sum_us_per_iter"]
                            )
                            case["rtp_over_vllm_graph_ratio"] = (
                                case["rtp"]["graph_stream_us_per_op"]["median"]
                                / case["vllm"]["graph_stream_us_per_op"]["median"]
                            )
                            case["status"] = "passed"
                        except Exception as error:
                            case["status"] = "failed"
                            case["error"] = repr(error)
                            result["cases"].append(case)
                            result_path.write_text(json.dumps(result, indent=2) + "\n")
                            raise
                        result["cases"].append(case)
                        result_path.write_text(json.dumps(result, indent=2) + "\n")
                        print(
                            f"{label}: RTP={case['rtp']['kernel_sum_us_per_iter']:.3f} us, vLLM={case['vllm']['kernel_sum_us_per_iter']:.3f} us, ratio={case['rtp_over_vllm_kernel_ratio']:.3f}",
                            flush=True,
                        )
                        del inputs
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
