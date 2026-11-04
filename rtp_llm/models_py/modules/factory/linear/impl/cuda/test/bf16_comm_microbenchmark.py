"""Distributed BF16 projection A/B test; results cover projection boundaries only."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import socket
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
    destroy_distributed_environment,
    get_process_group,
    init_distributed_environment,
    reduce_scatter_bf16,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.bf16_comm_fusion import (
    Bf16CollectiveProjection,
    prepare_bf16_projection,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.ops import NcclCommConfig, ParallelismConfig
from rtp_llm.test.utils import hang_watchdog
from rtp_llm.test.utils.port_util import PortsContext


def _tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _mark(stage):
    hang_watchdog.mark(f"rank{dist.get_rank()}:{stage}")


def _check_all_ranks(checks):
    """Raise tensor-check failures only after every rank has checked its output."""
    failures = {}
    for name, check in checks:
        try:
            check()
        except AssertionError as error:
            failures[name] = str(error)
    all_failures = [None] * dist.get_world_size()
    dist.all_gather_object(all_failures, failures)
    if any(all_failures):
        raise AssertionError(json.dumps({"per_rank_failures": all_failures}))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(per_rank):
    maxima = [max(sample) for sample in zip(*per_rank)]
    return {
        "max_rank_ms": maxima,
        "p50_ms": _percentile(maxima, 0.5),
        "p90_ms": _percentile(maxima, 0.9),
        "min_ms": min(maxima),
        "max_ms": max(maxima),
    }


def _timings(function, args):
    for _ in range(args.warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    for _ in range(args.samples):
        dist.barrier()
        start.record()
        for _ in range(args.repeat):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / args.repeat)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, samples)
    return {"per_rank_ms": gathered, **_summary(gathered)}


def _audit(function, collective, boundary):
    """Observe real backend calls without replacing their outputs."""
    observed = {}
    with ExitStack() as stack:
        calls = {}
        if collective.custom_ag is not None:
            calls["custom_ag"] = stack.enter_context(
                patch.object(
                    collective.custom_ag,
                    "all_gather",
                    wraps=collective.custom_ag.all_gather,
                )
            )
        if collective.push_rs is not None:
            calls["push_rs"] = stack.enter_context(
                patch.object(
                    collective.push_rs,
                    "reduce_scatter",
                    wraps=collective.push_rs.reduce_scatter,
                )
            )
        calls["fused_ag_matmul"] = stack.enter_context(
            patch.object(
                torch.ops.symm_mem,
                "fused_all_gather_matmul",
                wraps=torch.ops.symm_mem.fused_all_gather_matmul,
            )
        )
        if collective.deep_gemm is not None:
            calls["deepgemm_gemm_rs"] = stack.enter_context(
                patch.object(
                    collective.deep_gemm,
                    "bf16_gemm_rs_nn",
                    wraps=collective.deep_gemm.bf16_gemm_rs_nn,
                )
            )
        calls["nccl_ag"] = stack.enter_context(
            patch.object(
                dist, "all_gather_into_tensor", wraps=dist.all_gather_into_tensor
            )
        )
        if hasattr(torch.ops.symm_mem, "multimem_all_gather_out"):
            calls["multimem_ag"] = stack.enter_context(
                patch.object(
                    torch.ops.symm_mem,
                    "multimem_all_gather_out",
                    wraps=torch.ops.symm_mem.multimem_all_gather_out,
                )
            )
        calls["nccl_rs"] = stack.enter_context(
            patch.object(
                dist, "reduce_scatter_tensor", wraps=dist.reduce_scatter_tensor
            )
        )
        function()
        torch.cuda.synchronize()
        observed = {name: call.call_count for name, call in calls.items()}
        if observed.get("custom_ag"):
            observed["staging"] = calls["custom_ag"].call_args.kwargs.get(
                "staging", True
            )
    expected = (
        ("custom_ag", "fused_ag_matmul", "nccl_ag", "multimem_ag")
        if boundary == "input"
        else ("push_rs", "deepgemm_gemm_rs", "nccl_rs")
    )
    _check_all_ranks(
        [
            (
                "dispatch",
                lambda: _require(
                    sum(observed.get(name, 0) for name in expected) == 1,
                    f"ambiguous dispatch: {boundary} {observed}",
                ),
            )
        ]
    )
    return observed


def _correctness(
    baseline,
    candidate,
    inputs,
    boundary,
    *,
    return_input=False,
    output_projection=None,
    label="correctness",
):
    _mark(label + ":baseline")
    expected = _tensor(baseline()).clone()
    _mark(label + ":candidate")
    result = candidate()
    actual = _tensor(result).clone()
    checks = []
    if boundary == "input":
        checks.append(
            (
                "input_AB",
                lambda: torch.testing.assert_close(
                    actual,
                    expected,
                    atol=0.125,
                    rtol=0.02,
                ),
            )
        )
    if return_input:
        _mark(label + ":gathered_input")
        gathered = all_gather(inputs, Group.TP)
        checks.append(
            (
                "gathered_input",
                lambda: torch.testing.assert_close(
                    result[1],
                    gathered,
                    atol=0,
                    rtol=0,
                ),
            )
        )
    error = (actual.float() - expected.float()).abs()
    errors = {"max_abs": error.max().item(), "rms": error.square().mean().sqrt().item()}
    if boundary == "output":
        # Isolate communication rounding: preserve the production BF16 GEMM
        # partials, reduce them in FP32, and round only the final local shard.
        _mark(label + ":reference_gemm")
        partial = output_projection(inputs).float()
        reference_fp32 = partial.new_empty(
            partial.shape[0] // dist.get_world_size(), partial.shape[1]
        )
        _mark(label + ":reference_fp32_rs")
        dist.reduce_scatter_tensor(reference_fp32, partial)
        sum_abs = torch.empty_like(reference_fp32)
        _mark(label + ":reference_abs_fp32_rs")
        dist.reduce_scatter_tensor(sum_abs, partial.abs())
        reference = reference_fp32.bfloat16()
        # At most P-1 BF16 additions: |fl(sum)-sum| <= gamma_(P-1)*sum|x|.
        # Also cover the FP32 reference reduction and its final BF16 rounding.
        # Positive FP32 sum_abs may round down, so divide by 1-gamma_fp32.
        additions = dist.get_world_size() - 1
        unit_bf16, unit_fp32 = 2.0**-8, 2.0**-24
        gamma_bf16 = additions * unit_bf16 / (1 - additions * unit_bf16)
        gamma_fp32 = additions * unit_fp32 / (1 - additions * unit_fp32)
        reference_rounding = (reference_fp32.double() - reference.double()).abs()
        bound = (gamma_bf16 + gamma_fp32) * sum_abs.double() / (
            1 - gamma_fp32
        ) + reference_rounding
        baseline_delta = (expected.double() - reference.double()).abs()
        checks.append(
            (
                "baseline_BF16_forward_error",
                lambda: _require(
                    bool(torch.all(baseline_delta <= bound)),
                    f"NCCL BF16 reduction exceeds forward-error bound; "
                    f"max_error={baseline_delta.max().item()} max_bound={bound.max().item()}",
                ),
            )
        )
        checks.append(
            (
                "candidate_FP32_reference",
                lambda: torch.testing.assert_close(
                    actual,
                    reference,
                    atol=0.03125,
                    rtol=0.02,
                ),
            )
        )
        reference_errors = {}
        _mark(label + ":reference_compare")
        for name, value in (("baseline", expected), ("candidate", actual)):
            delta = (value.float() - reference.float()).abs()
            reference_errors[name] = {
                "max_abs": delta.max().item(),
                "rms": delta.square().mean().sqrt().item(),
            }
        errors["fp32_reduce_reference"] = reference_errors
        errors["baseline_forward_error_bound"] = {
            "bf16_unit_roundoff": unit_bf16,
            "fp32_unit_roundoff": unit_fp32,
            "max_additions": additions,
            "gamma_bf16": gamma_bf16,
            "gamma_fp32": gamma_fp32,
            "formula": "(gamma_bf16+gamma_fp32)*sum_abs_fp32/(1-gamma_fp32)+abs(sum_fp32-round_bf16(sum_fp32))",
            "max_reference_rounding": reference_rounding.max().item(),
            "max_bound": bound.max().item(),
            "max_error_over_bound": (
                baseline_delta / bound.clamp_min(torch.finfo(torch.float64).tiny)
            )
            .max()
            .item(),
        }
    _mark(label + ":check_consensus")
    _check_all_ranks(checks)
    return errors


def _graph(function, inputs, active_graphs, label):
    _mark(label + ":warmup")
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    active_graphs.append(graph)
    _mark(label + ":capture")
    with torch.cuda.graph(graph):
        output = function()
    _mark(label + ":initial_replay")
    graph.replay()
    original = _tensor(output).clone()
    inputs.mul_(0.5).add_(0.125)
    _mark(label + ":changed_input_replay")
    graph.replay()
    replayed = _tensor(output).clone()
    gathered = output[1].clone() if isinstance(output, tuple) else None
    _mark(label + ":eager_reference")
    eager = _tensor(function()).clone()
    _mark(label + ":compare")
    checks = [
        (
            "graph_eager",
            lambda: torch.testing.assert_close(replayed, eager, atol=0, rtol=0),
        )
    ]
    if gathered is not None:
        expected_gathered = all_gather(inputs, Group.TP)
        checks.append(
            (
                "graph_gathered_input",
                lambda: torch.testing.assert_close(
                    gathered,
                    expected_gathered,
                    atol=0,
                    rtol=0,
                ),
            )
        )
    checks.append(
        (
            "graph_changed_input",
            lambda: _require(
                not torch.equal(replayed, original),
                "changed graph input did not change the output",
            ),
        )
    )
    _check_all_ranks(checks)

    def replay():
        graph.replay()
        return output

    return replay, {"changed_input": True, "bitwise_eager_match": True}


def _reset_graphs(active_graphs):
    if active_graphs:
        _mark("graphs:drain")
        torch.cuda.synchronize()
        _mark("graphs:reset")
        for graph in active_graphs:
            graph.reset()
        active_graphs.clear()
        gc.collect()
        _mark("graphs:reset_done")


def _profile(functions, path):
    torch.cuda.synchronize()
    dist.barrier()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        for name, function in functions.items():
            _mark("profile:" + path.name + ":" + name)
            with torch.profiler.record_function(name):
                function()
        torch.cuda.synchronize()
    _mark("profile:export:" + path.name)
    profiler.export_chrome_trace(str(path))


def _rows(args, phase):
    if args.rows:
        return args.rows
    if phase == "decode":
        return sorted({args.world_size, 128, 512 - args.world_size, 512, 1024})
    return sorted(
        {
            16 * args.world_size,
            512 - args.world_size,
            512,
            127 * args.world_size,
            128 * args.world_size,
            4095 * args.world_size,
            4096 * args.world_size,
        }
    )


def _worker(rank, args, port, directory):
    hang_watchdog.install(seconds=0)
    hang_watchdog.mark(f"rank{rank}:device_init")
    torch.cuda.set_device(rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    parallelism = ParallelismConfig()
    parallelism.world_rank = parallelism.local_rank = rank
    parallelism.world_size = parallelism.tp_size = args.world_size
    parallelism.dp_size = 1
    nccl = NcclCommConfig(
        nccl_ip="127.0.0.1",
        tp_nccl_port=port + 9,
        dp_tp_nccl_port=port + 1,
        ffn_tp_nccl_port=port + 6,
    )
    hang_watchdog.mark(f"rank{rank}:distributed_init")
    init_distributed_environment(parallelism, nccl, port)
    records, current_case = [], None
    active_graphs = []
    try:
        device = torch.device("cuda", rank)
        properties = torch.cuda.get_device_properties(device)
        identity = {
            "rank": rank,
            "host": socket.gethostname(),
            "device": properties.name,
            "uuid": str(properties.uuid),
            "capability": [properties.major, properties.minor],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "deep_gemm": importlib.metadata.version("deep_gemm"),
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "FT_DISABLE_CUSTOM_AR",
                    "NCCL_ALGO",
                    "NCCL_PROTO",
                )
            },
        }
        identities = [None] * args.world_size
        dist.all_gather_object(identities, identity)
        heads = args.heads // args.world_size
        widths = {
            "dsa": (
                args.q_rank + args.kv_rank + args.suffix_dim + heads * args.head_dim,
                heads * args.head_dim,
            ),
            "kda": (
                4 * heads * args.head_dim + args.forget_rank + heads,
                heads * args.head_dim,
            ),
        }
        for phase in args.phases:
            modules = ("kda",) if phase == "prefill" else ("dsa", "kda")
            for module in modules:
                input_width, output_width = widths[module]
                input_projection = CudaF16Linear(
                    torch.randn(
                        args.hidden_size,
                        input_width,
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    * 0.02
                )
                output_projection = CudaF16Linear(
                    torch.randn(
                        output_width,
                        args.hidden_size,
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    * 0.02
                )
                prepare_bf16_projection(
                    input_projection, alignment=128 if module == "kda" else 1
                )
                prepare_bf16_projection(output_projection)
                collective = Bf16CollectiveProjection(
                    get_process_group(Group.TP),
                    device,
                    max_m=max(_rows(args, phase)),
                    hidden_size=args.hidden_size,
                    output_projections=[output_projection],
                    decode_staging=phase == "decode",
                )
                if phase == "decode":
                    _check_all_ranks(
                        [
                            (
                                "decode_backends",
                                lambda: _require(
                                    collective.custom_ag is not None
                                    and collective.push_rs is not None,
                                    "Decode candidate requires custom staging AG and Push RS",
                                ),
                            )
                        ]
                    )
                for rows in _rows(args, phase):
                    attention = SimpleNamespace(
                        is_prefill=phase == "prefill", is_mtp_draft_update=False
                    )
                    metadata = SimpleNamespace(is_target_verify=phase == "decode")
                    _check_all_ranks(
                        [
                            (
                                "eligibility",
                                lambda: _require(
                                    collective.eligible(
                                        rows // args.world_size, attention, metadata
                                    ),
                                    f"ineligible production shape: {phase} M={rows}",
                                ),
                            )
                        ]
                    )
                    for boundary in ("input", "output"):
                        current_case = {
                            "phase": phase,
                            "module": module,
                            "boundary": boundary,
                            "M": rows,
                        }
                        label = f"{phase}:{module}:{boundary}:M{rows}"
                        print(
                            json.dumps({"rank": rank, "begin_case": current_case}),
                            flush=True,
                        )
                        shape = (
                            (rows // args.world_size, args.hidden_size)
                            if boundary == "input"
                            else (rows, output_width)
                        )
                        inputs = torch.randn(shape, device=device, dtype=torch.bfloat16)
                        keep_input = module == "dsa" and boundary == "input"
                        if boundary == "input":
                            baseline = lambda: input_projection(
                                all_gather(inputs, Group.TP)
                            )
                            candidate = lambda: collective.all_gather_gemm(
                                inputs, input_projection, return_input=keep_input
                            )
                        else:
                            baseline = lambda: reduce_scatter_bf16(
                                output_projection(inputs), Group.TP
                            )
                            candidate = lambda: collective.gemm_reduce_scatter(
                                inputs, output_projection
                            )
                        errors = _correctness(
                            baseline,
                            candidate,
                            inputs,
                            boundary,
                            return_input=keep_input,
                            output_projection=output_projection,
                            label=label + ":correctness",
                        )
                        all_errors = [None] * args.world_size
                        dist.all_gather_object(all_errors, errors)
                        _mark(label + ":audit_baseline")
                        baseline_dispatch = _audit(baseline, collective, boundary)
                        _mark(label + ":audit_candidate")
                        dispatch = _audit(candidate, collective, boundary)
                        per_rank_dispatch = [None] * args.world_size
                        dist.all_gather_object(per_rank_dispatch, dispatch)
                        if any(item != dispatch for item in per_rank_dispatch):
                            raise AssertionError(
                                f"rank dispatch differs: {per_rank_dispatch}"
                            )
                        record = {
                            "phase": phase,
                            "module": module,
                            "boundary": boundary,
                            "M_global_physical_rows": rows,
                            "m_local_rows": rows // args.world_size,
                            "input_shape": shape,
                            "input_projection_logical_width": input_width,
                            "input_projection_physical_width": input_projection.weight.shape[
                                0
                            ],
                            "output_projection_input_width": output_width,
                            "dtype": "bfloat16",
                            "return_gathered_input": keep_input,
                            "dispatch": dispatch,
                            "per_rank_dispatch": per_rank_dispatch,
                            "baseline_dispatch": baseline_dispatch,
                            "per_rank_error": all_errors,
                            "rounds": [],
                        }
                        torch.cuda.reset_peak_memory_stats()
                        functions = {"baseline": baseline, "candidate": candidate}
                        for round_index, order in enumerate(
                            (tuple(functions), tuple(reversed(functions)))
                        ):
                            measurement = {"round": round_index, "order": order}
                            for name in order:
                                _mark(f"{label}:eager:round{round_index}:{name}")
                                measurement[name] = _timings(functions[name], args)
                            measurement["speedup"] = (
                                measurement["baseline"]["p50_ms"]
                                / measurement["candidate"]["p50_ms"]
                            )
                            record["rounds"].append(measurement)
                        if phase == "decode":
                            graphs, graph_checks = {}, {}
                            for name, function in functions.items():
                                graphs[name], graph_checks[name] = _graph(
                                    function,
                                    inputs,
                                    active_graphs,
                                    label + ":graph:" + name,
                                )
                            # Capture checks mutate inputs; compare A/B again at the final values.
                            _correctness(
                                baseline,
                                candidate,
                                inputs,
                                boundary,
                                return_input=keep_input,
                                output_projection=output_projection,
                                label=label + ":post_graph_correctness",
                            )
                            record["graph_correctness"] = graph_checks
                            record["graph_rounds"] = []
                            for round_index, order in enumerate(
                                (tuple(graphs), tuple(reversed(graphs)))
                            ):
                                measurement = {"round": round_index, "order": order}
                                for name in order:
                                    _mark(
                                        f"{label}:graph_timing:round{round_index}:{name}"
                                    )
                                    measurement[name] = _timings(graphs[name], args)
                                measurement["speedup"] = (
                                    measurement["baseline"]["p50_ms"]
                                    / measurement["candidate"]["p50_ms"]
                                )
                                record["graph_rounds"].append(measurement)
                        record["peak_allocated_bytes"] = (
                            torch.cuda.max_memory_allocated()
                        )
                        if args.profile and rows == (
                            max(_rows(args, phase)) if phase == "prefill" else 512
                        ):
                            trace = Path(
                                directory,
                                f"{phase}_{module}_{boundary}_rank{rank}.json",
                            )
                            profiled = dict(functions)
                            if phase == "decode":
                                profiled.update(
                                    (name + "_graph", function)
                                    for name, function in graphs.items()
                                )
                            _profile(profiled, trace)
                            profiled.clear()
                            record["rank_trace_basename"] = trace.name
                        _reset_graphs(active_graphs)
                        if phase == "decode":
                            graphs.clear()
                        records.append(record)
                        Path(directory, f"bf16_comm_rank{rank}.json").write_text(
                            json.dumps(
                                {"identity": identity, "records": records}, indent=2
                            )
                        )
                        if rank == 0:
                            print(
                                json.dumps(
                                    {
                                        key: record[key]
                                        for key in (
                                            "phase",
                                            "module",
                                            "boundary",
                                            "M_global_physical_rows",
                                            "dispatch",
                                        )
                                    }
                                ),
                                flush=True,
                            )
                del collective, input_projection, output_projection
        if rank == 0:
            result = {
                "contract": {
                    "model": "Kimi K3.1 projection geometry, synthetic random inputs/weights",
                    "geometry_source": args.geometry_source,
                    "geometry": vars(args),
                    "topology": {"tp": args.world_size, "sp": args.world_size, "dp": 1},
                    "baseline": "production all_gather -> CudaF16Linear / CudaF16Linear -> reduce_scatter_bf16",
                    "candidate": "production Bf16CollectiveProjection",
                    "timing": "CUDA events around full projection+collective; milliseconds per invocation",
                    "aggregation": "max rank per sample, then linear-interpolated quantiles",
                    "excluded": [
                        "initialization",
                        "JIT warmup",
                        "capture",
                        "correctness",
                        "rank alignment",
                    ],
                    "not_measured": [
                        "attention core",
                        "Indexer",
                        "query-CP reordering",
                        "serving latency",
                    ],
                    "script_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                },
                "environment": identities,
                "records": records,
            }
            Path(directory, "bf16_comm_microbenchmark.json").write_text(
                json.dumps(result, indent=2)
            )
    except BaseException as error:
        Path(directory, f"bf16_comm_rank{rank}_failure.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "case": current_case,
                    "error": repr(error),
                    "completed_records": records,
                },
                indent=2,
            )
        )
        raise
    finally:
        _mark("teardown:graphs")
        _reset_graphs(active_graphs)
        _mark("teardown:synchronize")
        torch.cuda.synchronize()
        _mark("teardown:process_group")
        destroy_distributed_environment()
        hang_watchdog.mark(f"rank{rank}:teardown:done")


class Bf16CommMicrobenchmark(unittest.TestCase):
    def test_projection_paths(self):
        if torch.cuda.device_count() < ARGS.world_size:
            self.fail(f"requires {ARGS.world_size} allocated CUDA devices")
        directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or tempfile.mkdtemp(
            prefix="bf16-comm-"
        )
        Path(directory).mkdir(parents=True, exist_ok=True)
        with PortsContext(num_ports=12) as ports:
            mp.spawn(_worker, args=(ARGS, ports[0], directory), nprocs=ARGS.world_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2, choices=(2, 4, 8))
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("prefill", "decode"),
        default=["prefill", "decode"],
    )
    parser.add_argument("--rows", nargs="+", type=int)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--q-rank", type=int, default=1536)
    parser.add_argument("--kv-rank", type=int, default=512)
    parser.add_argument("--suffix-dim", type=int, default=64)
    parser.add_argument("--forget-rank", type=int, default=128)
    parser.add_argument(
        "--geometry-source",
        default="parameterized synthetic geometry; not a checkpoint claim",
    )
    parser.add_argument(
        "--source-revision", default=os.environ.get("SOURCE_REVISION", "unrecorded")
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    ARGS, remaining = parser.parse_known_args()
    if ARGS.heads % ARGS.world_size or any(
        row <= 0 or row % ARGS.world_size for row in (ARGS.rows or [])
    ):
        parser.error("heads and all global row counts must be divisible by world size")
    if min(ARGS.warmup, ARGS.samples, ARGS.repeat) <= 0:
        parser.error("warmup, samples and repeat must be positive")
    unittest.main(argv=[sys.argv[0], *remaining])
