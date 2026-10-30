"""Compare replay precision fixes using production kernels and native operators.

The three variants differ only in convolution operand casts and explicit FMA.
CUDA-event samples time captured kernel launches, excluding allocation, input
preparation, compilation, capture, and host validation. Synthetic local-head
workloads do not measure model TPOT or distributed communication.
"""

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton

from rtp_llm.models_py.triton_kernels import linear_replay as replay
from rtp_llm.models_py.triton_kernels.causal_conv1d import causal_conv1d_update
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.models_py.triton_kernels.kda import fused_recurrent_kda

KERNELS = ("_rebuild_state_kernel", "_replay_conv_kernel", "_serial_verify_kernel")
STAGES = ("rebuild", "convolution", "verify", "three_kernels")


class BeforeFix(ast.NodeTransformer):
    def __init__(self):
        self.casts = self.fmas = 0

    def visit_Assign(self, node):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return node
        name, value = node.targets[0].id, node.value
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute):
            return node
        if name in ("sample", "weight") and value.func.attr == "to":
            self.casts += 1
            node.value = value.func.value
        elif name == "state" and ast.unparse(value.func) == "tl.fma":
            self.fmas += 1
            node = ast.AugAssign(
                target=node.targets[0],
                op=ast.Add(),
                value=ast.BinOp(value.args[0], ast.Mult(), value.args[1]),
            )
        return node


def variants(output):
    # Derive the old kernels from the checked-out source rather than maintaining
    # copies. Fail if the production change no longer matches these two fixes.
    parts = [
        "import triton\nimport triton.language as tl\n"
        "from rtp_llm.models_py.triton_kernels.linear_replay "
        "import _replay_row, exp, softplus\n"
    ]
    for name in KERNELS:
        tree = ast.parse(inspect.getsource(getattr(replay, name).fn))
        rewrite = BeforeFix()
        tree = rewrite.visit(tree)
        expected = (2, 0) if name == "_replay_conv_kernel" else (0, 1)
        assert (rewrite.casts, rewrite.fmas) == expected, name
        parts.append(ast.unparse(ast.fix_missing_locations(tree)))
    source = "\n\n".join(parts)
    path = output / "before_precision_fixes.py"
    path.write_text(source + "\n")
    spec = importlib.util.spec_from_file_location("before_precision_fixes", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    old = tuple(getattr(module, name) for name in KERNELS)
    new = tuple(getattr(replay, name) for name in KERNELS)
    return {
        "before_both": old,
        "convolution_fix": (old[0], new[1], old[2]),
        "both_fixes": new,
    }


def make_case(family, batch, steps, accepted, weight_dtype):
    torch.manual_seed(1234)
    h, hv, dim, width = 8, 16 if family == "gdn" else 8, 128, 4
    channels = (2 * h + hv) * dim

    def zeros(shape, dtype=torch.float32):
        return torch.zeros(shape, device="cuda", dtype=dtype)

    def random(shape, dtype=torch.float32):
        return torch.randn(shape, device="cuda", dtype=dtype) * 0.2

    raw = random((batch, steps, channels), torch.bfloat16)
    cache = SimpleNamespace(
        k=random((batch, steps, h, dim)),
        u=random((batch, steps, hv, dim)),
        g=-torch.rand(batch, steps, hv, dim if family == "kda" else 1, device="cuda"),
        conv_inputs=raw.clone(),
        slot_generations=torch.ones(batch, device="cuda", dtype=torch.int64),
        log_epochs=torch.ones(batch, device="cuda", dtype=torch.int64),
        valid_counts=torch.full((batch,), steps, device="cuda", dtype=torch.int32),
        error_flags=zeros((batch,), torch.int32),
    )
    inputs = SimpleNamespace(
        slot_ids=torch.arange(batch, device="cuda", dtype=torch.int32),
        slot_generations=torch.ones(batch, device="cuda", dtype=torch.int64),
        prev_accept_lengths=torch.full(
            (batch,), accepted, device="cuda", dtype=torch.int32
        ),
        history_valid_lengths=torch.full(
            (batch,), steps, device="cuda", dtype=torch.int32
        ),
        history_epochs=torch.ones(batch, device="cuda", dtype=torch.int64),
        verify_epochs=torch.full((batch,), 2, device="cuda", dtype=torch.int64),
        init_kinds=zeros((batch,), torch.int32),
        state_read_block_ids=torch.arange(
            1, batch + 1, device="cuda", dtype=torch.int32
        )[None],
        active_block_ids=torch.arange(
            batch + 1, 2 * batch + 1, device="cuda", dtype=torch.int32
        )[None],
    )
    return SimpleNamespace(
        family=family,
        batch=batch,
        steps=steps,
        h=h,
        hv=hv,
        dim=dim,
        raw=raw,
        gate=random(
            (batch * steps, hv * (dim if family == "kda" else 1)), torch.bfloat16
        ),
        beta=random((batch * steps, hv), torch.bfloat16),
        weights=random((channels, width), weight_dtype),
        alog=random((hv,)),
        bias=random((hv * (dim if family == "kda" else 1),)),
        ssm=random((2 * batch + 1, hv, dim, dim)),
        conv=random((2 * batch + 1, width - 1, channels), torch.bfloat16),
        cache=cache,
        inputs=inputs,
        sink=[torch.empty_like(getattr(cache, key)) for key in ("k", "u", "g")],
    )


class Launch:
    def __init__(self, kernel, grid, args, kwargs):
        self.kernel, self.grid, self.args, self.kwargs = kernel, grid, args, kwargs
        self.compiled = None

    def __call__(self):
        self.compiled = self.kernel[self.grid](*self.args, **self.kwargs)


class Recorder:
    def __init__(self, kernel, calls, sink=None, checkpoints=None):
        self.kernel, self.calls, self.sink = kernel, calls, sink
        self.checkpoints = checkpoints

    def __getitem__(self, grid):
        def record(*args, **kwargs):
            if self.sink is not None:
                args = (*args[:7], *self.sink, *args[10:])
            if self.checkpoints is not None:
                args = (*args[:7], self.checkpoints, *args[7:])
            self.calls.append(Launch(self.kernel, grid, args, kwargs))

        return record


def capture_launches(case, kernels, frozen=False, checkpoints=None):
    calls = []
    with patch.object(replay, KERNELS[0], Recorder(kernels[0], calls)), patch.object(
        replay, KERNELS[1], Recorder(kernels[1], calls)
    ), patch.object(
        replay,
        KERNELS[2],
        Recorder(kernels[2], calls, case.sink if frozen else None, checkpoints),
    ):
        sizes = [case.h * case.dim, case.h * case.dim, case.hv * case.dim]
        q, k, v = torch.split(case.raw.flatten(0, 1), sizes, dim=-1)
        output = replay.linear_serial_replay(
            q,
            k,
            v,
            case.gate,
            case.beta,
            case.weights,
            case.alog,
            case.bias,
            case.ssm,
            case.conv,
            case.cache,
            case.inputs,
            group_id=0,
            vector_gate=case.family == "kda",
            state_v_first=case.family == "gdn",
            lower_bound=-5.0 if case.family == "kda" else None,
        )
    assert len(calls) == 3
    return calls, output


def difference(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    return {
        "different": int((actual != expected).sum()),
        "elements": actual.numel(),
        "max_abs": float(delta.max()),
    }


@lru_cache(None)
def checkpoint_kernel(kernel, output):
    """Save producer checkpoints for correctness only; never time this kernel."""
    tree = ast.parse(inspect.getsource(kernel.fn))
    function = tree.body[0]
    function.name += "_checkpoints"
    function.args.args.insert(7, ast.arg(arg="Checkpoints"))
    loop = next(x for x in function.body if isinstance(x, ast.For))
    for index, statement in enumerate(loop.body):
        explicit = (
            isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Call)
            and ast.unparse(statement.value.func) == "tl.fma"
        )
        implicit = (
            isinstance(statement, ast.AugAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "state"
        )
        if explicit or implicit:
            loop.body.insert(
                index + 1,
                ast.parse(
                    "tl.store(Checkpoints + (slot * CAPACITY + t) * HV * V * K + state_offset, "
                    "state, (ik[None, :] < K) & (iv[:, None] < V))"
                ).body[0],
            )
            break
    else:
        raise AssertionError("producer state update not found")
    source = (
        "import triton\nimport triton.language as tl\n"
        "from rtp_llm.models_py.triton_kernels.linear_replay import _replay_row, exp, softplus\n"
        + ast.unparse(ast.fix_missing_locations(tree))
    )
    name = "producer_" + hashlib.sha256(source.encode()).hexdigest()[:12]
    path = output / (name + ".py")
    path.write_text(source + "\n")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return getattr(module, function.name)


def native_check(family, steps, weight_dtype, kernels, output_dir):
    base = make_case(family, 1, steps, 1, weight_dtype)
    shape = (steps + 3, *base.ssm.shape[1:])
    state = base.ssm.new_zeros(shape)
    conv = base.conv.new_zeros((steps + 3, *base.conv.shape[1:]))
    state[1], conv[1] = base.ssm[1], base.conv[1]
    block_map = torch.arange(1, steps + 2, device="cuda", dtype=torch.int32)[None]
    lengths = torch.full((1,), 468, device="cuda", dtype=torch.int32)
    convolved = causal_conv1d_update(
        base.raw.transpose(1, 2),
        conv.transpose(1, 2),
        base.weights,
        activation="silu",
        block_map=block_map,
        sequence_lengths=lengths,
        seq_size_per_block=2048,
    ).transpose(1, 2)
    q, k, v = torch.split(
        convolved, [base.h * 128, base.h * 128, base.hv * 128], dim=-1
    )
    common = dict(
        q=q.reshape(1, steps, base.h, 128).contiguous(),
        k=k.reshape(1, steps, base.h, 128).contiguous(),
        v=v.reshape(1, steps, base.hv, 128).contiguous(),
        initial_state=state,
        inplace_final_state=True,
        block_map=block_map,
        sequence_lengths=lengths,
        seq_size_per_block=2048,
        use_qk_l2norm_in_kernel=True,
    )
    if family == "gdn":
        g, beta = fused_gdn_gating(base.alog, base.gate, base.beta, base.bias)
        expected, _ = fused_recurrent_gated_delta_rule(
            **common,
            g=g.reshape(1, steps, base.hv),
            beta=beta.reshape(1, steps, base.hv),
        )
    else:
        expected, _ = fused_recurrent_kda(
            **common,
            g=base.gate.reshape(1, steps, base.hv, 128),
            beta=base.beta.reshape(1, steps, base.hv),
            A_log=base.alog,
            dt_bias=base.bias,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            state_v_first=False,
            transpose_state_layout=True,
        )
    results, convolutions = {}, {}
    for name, selected in kernels.items():
        case = make_case(family, 1, steps, 1, weight_dtype)
        for field in ("prev_accept_lengths", "history_valid_lengths", "history_epochs"):
            getattr(case.inputs, field).zero_()
        case.inputs.init_kinds.fill_(1)
        case.inputs.verify_epochs.fill_(1)
        checkpoints = torch.empty((1, steps, case.hv, 128, 128), device="cuda")
        instrumented = (*selected[:2], checkpoint_kernel(selected[2], output_dir))
        calls, output = capture_launches(case, instrumented, checkpoints=checkpoints)
        for call in calls:
            call()
        row = {
            "convolution": difference(
                calls[1].args[6].reshape_as(convolved), convolved
            ),
            "output": difference(output.reshape_as(expected), expected),
        }
        convolutions[name] = calls[1].args[6].clone()
        replay.finalize_linear_replay(case.cache, case.inputs, steps)
        snapshot = {key: value.clone() for key, value in vars(case.cache).items()}
        case.inputs.state_read_block_ids.fill_(2)
        case.inputs.active_block_ids.fill_(1)
        case.inputs.history_valid_lengths.fill_(steps)
        case.inputs.history_epochs.fill_(1)
        case.inputs.verify_epochs.fill_(2)
        case.inputs.init_kinds.zero_()
        row["accepted_prefixes"] = []
        for accepted in range(1, steps + 1):
            for key, value in snapshot.items():
                getattr(case.cache, key).copy_(value)
            case.inputs.prev_accept_lengths.fill_(accepted)
            calls, _ = capture_launches(case, selected)
            for call in calls:
                call()
            row["accepted_prefixes"].append(
                {
                    "accepted": accepted,
                    "state": difference(case.ssm[1], state[accepted]),
                    "producer_state": difference(
                        case.ssm[1], checkpoints[0, accepted - 1]
                    ),
                    "conv": difference(case.conv[1], conv[accepted]),
                }
            )
        results[name] = row
    results["both_fixes"]["convolution_vs_before"] = difference(
        convolutions["both_fixes"], convolutions["before_both"]
    )
    (
        output_dir / f"native_{family}_{str(weight_dtype).split('.')[-1]}.json"
    ).write_text(json.dumps(results, indent=2) + "\n")
    fixed = results["both_fixes"]
    if weight_dtype == torch.bfloat16:
        assert fixed["convolution"]["different"] == 0, results
    else:
        # FP32 is an unchanged-precision control. Record existing differences
        # from the native convolution, and require no difference from baseline.
        assert fixed["convolution_vs_before"]["different"] == 0, results
    assert all(
        x["producer_state"]["different"] == 0 for x in fixed["accepted_prefixes"]
    ), results
    if family == "gdn" and fixed["convolution"]["different"] == 0:
        assert results["both_fixes"]["output"]["different"] == 0
        assert all(
            x["state"]["different"] == 0
            for x in results["both_fixes"]["accepted_prefixes"]
        )
    return results


def graph_for(calls, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            for call in calls:
                call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repeats):
            for call in calls:
                call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return graph


def occupancy():
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not uuid.startswith(("GPU-", "MIG-")):
        uuid = "GPU-" + uuid
    own_pids = {os.getpid()}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("NSpid:"):
            own_pids.update(int(pid) for pid in line.split()[1:])
    proc = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            uuid,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    foreign = [
        int(x.strip())
        for x in proc.stdout.splitlines()
        if x.strip().isdigit() and int(x.strip()) not in own_pids
    ]
    if foreign:
        raise RuntimeError(f"foreign GPU processes on {uuid}: {foreign}")
    return {"gpu_uuid": uuid, "foreign_processes": foreign}


def measure(case, kernels, output, args, index):
    # Freeze previous-round logs while the verify kernel writes current-round
    # logs to separate, equally shaped sinks. Raw convolution inputs are already
    # identical to their log. Thus graph repetitions never advance the workload.
    launches = {
        name: capture_launches(case, selected, frozen=True)[0]
        for name, selected in kernels.items()
    }
    graphs, codegen = {}, {}
    for name, calls in launches.items():
        for stage, group in zip(STAGES, ([calls[0]], [calls[1]], [calls[2]], calls)):
            graphs[name, stage] = graph_for(group, args.inner_repeats)
        codegen[name] = {}
        for stage, call in zip(STAGES, calls):
            ptx = call.compiled.asm["ptx"]
            path = output / f"case_{index}_{name}_{stage}.ptx"
            path.write_text(ptx)
            codegen[name][stage] = {
                "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
                "fma_f32": len(re.findall(r"\bfma(?:\.[a-z]+)*\.f32\b", ptx)),
                "mul_f32": len(re.findall(r"\bmul\.(?:rn\.)?f32\b", ptx)),
                "bf16_conversions": len(re.findall(r"\bcvt\.[^\s;]*bf16[^\s;]*", ptx)),
                "registers": call.compiled.n_regs,
                "spills": call.compiled.n_spills,
            }
    samples = {name: {stage: [] for stage in STAGES} for name in kernels}
    for repeat in range(args.rounds):
        occupancy()
        order = list(kernels) if repeat % 2 == 0 else list(reversed(kernels))
        round_samples = {name: {stage: [] for stage in STAGES} for name in kernels}
        for _ in range(args.samples):
            for name in order:
                for stage in STAGES:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record()
                    graphs[name, stage].replay()
                    end.record()
                    end.synchronize()
                    round_samples[name][stage].append(
                        start.elapsed_time(end) * 1000 / args.inner_repeats
                    )
        for name in kernels:
            for stage in STAGES:
                samples[name][stage].append(round_samples[name][stage])
    occupancy()
    return {
        "samples_us": samples,
        "codegen": codegen,
        "p50_us": {
            name: {
                stage: statistics.median(x for row in samples[name][stage] for x in row)
                for stage in STAGES
            }
            for name in kernels
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,16,128")
    parser.add_argument("--accept-lengths", default="1,5")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--inner-repeats", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.environ.get(
                "TEST_UNDECLARED_OUTPUTS_DIR", "build_logs/linear_replay_microbenchmark"
            )
        ),
    )
    args = parser.parse_args()
    assert args.rounds >= 2 and args.samples > 0 and args.inner_repeats > 0
    batches = [int(x) for x in args.batches.split(",")]
    accepts = [int(x) for x in args.accept_lengths.split(",")]
    assert all(x > 0 for x in batches) and all(1 <= x <= args.steps for x in accepts)
    args.output.mkdir(parents=True, exist_ok=True)
    selected = variants(args.output)
    report = {
        "environment": {
            "public_revision": os.environ.get("RTP_BENCHMARK_REVISION", "unknown"),
            "internal_revision": os.environ.get(
                "RTP_BENCHMARK_INTERNAL_REVISION", "unknown"
            ),
            "host": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "replay_sha256": hashlib.sha256(
                Path(replay.__file__).read_bytes()
            ).hexdigest(),
            "command": sys.argv,
            "occupancy": occupancy(),
        },
        "contract": {
            "input": "synthetic local-head tensors; no collectives",
            "gdn_heads": [8, 16],
            "kda_heads": [8, 8],
            "key_value_dimensions": [128, 128],
            "activation_conv_dtype": "bfloat16",
            "ssm_log_dtype": "float32",
            "conv_width": 4,
            "timing": "CUDA events around graph replay / inner_repeats; warm kernels; microseconds",
            "excluded": [
                "allocation",
                "compilation",
                "capture",
                "host wrapper",
                "finalize headers",
            ],
            "frozen_logs": "verify writes separate sinks; inputs do not advance between repeats",
            "rounds": args.rounds,
            "samples": args.samples,
            "inner_repeats": args.inner_repeats,
        },
        "correctness": [],
        "cases": [],
    }
    destination = args.output / "linear_replay_microbenchmark.json"
    for family in ("gdn", "kda"):
        for dtype in (torch.bfloat16, torch.float32):
            correctness = native_check(family, args.steps, dtype, selected, args.output)
            report["correctness"].append(
                {"family": family, "weight_dtype": str(dtype), "variants": correctness}
            )
            destination.write_text(json.dumps(report, indent=2) + "\n")
            for batch in batches:
                for accepted in accepts:
                    case = make_case(family, batch, args.steps, accepted, dtype)
                    measured = measure(
                        case, selected, args.output, args, len(report["cases"])
                    )
                    row = {
                        "family": family,
                        "batch": batch,
                        "steps": args.steps,
                        "accepted": accepted,
                        "weight_dtype": str(dtype),
                        **measured,
                    }
                    report["cases"].append(row)
                    destination.write_text(json.dumps(report, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "family",
                                    "batch",
                                    "accepted",
                                    "weight_dtype",
                                    "p50_us",
                                )
                            }
                        ),
                        flush=True,
                    )
                    del case
    print(f"Results: {destination}", flush=True)


if __name__ == "__main__":
    main()
