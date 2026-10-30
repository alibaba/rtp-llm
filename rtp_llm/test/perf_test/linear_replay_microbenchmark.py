"""Measure the complete replay path, optionally against a previous source file.

Protocol mode measures KDA convolution/fused replay-verify or the GDN three-kernel
path, including old header finalize when --baseline-source is provided. Every
sample restores producer history before a single step. Precision mode retains
the unfused convolution/FMA comparison. These workloads do not measure TPOT.
"""

import argparse
import ast
import base64
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
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
        "import _replay_row, _replay_trap, _load_conv_replay_history, exp, softplus\n"
    ]
    for name in ("_rebuild_state_tile", *KERNELS):
        tree = ast.parse(inspect.getsource(getattr(replay, name).fn))
        rewrite = BeforeFix()
        tree = rewrite.visit(tree)
        expected = (
            (2, 0)
            if name == "_replay_conv_kernel"
            else (0, 0) if name == "_rebuild_state_kernel" else (0, 1)
        )
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
    )
    inputs = SimpleNamespace(
        slot_ids=torch.arange(batch, device="cuda", dtype=torch.int32),
        prev_accept_lengths=torch.full(
            (batch,), accepted, device="cuda", dtype=torch.int32
        ),
        history_valid_lengths=torch.full(
            (batch,), steps, device="cuda", dtype=torch.int32
        ),
        history_log_banks=zeros((batch,), torch.int32),
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
    def __init__(self, kernel, calls, sink=None, checkpoints=None, replacements=None):
        self.kernel, self.calls, self.sink = kernel, calls, sink
        self.checkpoints = checkpoints
        self.replacements = replacements or {}

    def __getitem__(self, grid):
        def record(*args, **kwargs):
            if self.sink is not None:
                args = (*args[:7], *self.sink, *args[10:])
            args = tuple(self.replacements.get(i, arg) for i, arg in enumerate(args))
            if self.checkpoints is not None:
                args = (*args[:7], self.checkpoints, *args[7:])
            self.calls.append(Launch(self.kernel, grid, args, kwargs))

        return record


def capture_launches(
    case,
    kernels,
    frozen=False,
    checkpoints=None,
    *,
    module=replay,
    finalize=False,
    scratch=None,
):
    calls = []
    with patch.object(module, KERNELS[0], Recorder(kernels[0], calls)), patch.object(
        module,
        KERNELS[1],
        Recorder(kernels[1], calls, replacements={6: scratch[0]} if scratch else None),
    ), patch.object(
        module,
        KERNELS[2],
        Recorder(
            kernels[2],
            calls,
            case.sink if frozen else None,
            checkpoints,
            replacements={0: scratch[0], 6: scratch[1]} if scratch else None,
        ),
    ):
        sizes = [case.h * case.dim, case.h * case.dim, case.hv * case.dim]
        q, k, v = torch.split(case.raw.flatten(0, 1), sizes, dim=-1)
        output = module.linear_serial_replay(
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
    banked = case.family == "kda" and case.cache.k.shape[0] == 2 * case.cache.u.shape[0]
    assert len(calls) == (2 if banked else 3)
    if finalize:
        with patch.object(
            module, "_finalize_kernel", Recorder(module._finalize_kernel, calls)
        ):
            module.finalize_linear_replay(case.cache, case.inputs, case.steps)
    return calls, output


def protocol_modules(baseline_source):
    modules = {"candidate": replay}
    if baseline_source is not None:
        spec = importlib.util.spec_from_file_location(
            "linear_replay_protocol_baseline", baseline_source
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        assert hasattr(module, "finalize_linear_replay"), "baseline must retain headers"
        modules["baseline"] = module
    return modules


def legacy_headers(case):
    batch = case.batch
    for obj, names in (
        (case.cache, ("slot_generations", "log_epochs")),
        (case.inputs, ("slot_generations", "history_epochs", "verify_epochs")),
    ):
        for name in names:
            setattr(obj, name, torch.ones(batch, device="cuda", dtype=torch.int64))
    case.cache.valid_counts = torch.full(
        (batch,), case.steps, device="cuda", dtype=torch.int32
    )
    case.cache.error_flags = torch.zeros(batch, device="cuda", dtype=torch.int32)


def protocol_case(module, family, batch, steps, accepted):
    case = make_case(family, batch, steps, accepted, torch.bfloat16)
    if module is replay and family == "kda":
        # Allocate the second bank without consuming RNG: both variants receive
        # the same logical old history, activations, weights and checkpoint.
        for name in ("k", "g"):
            old = getattr(case.cache, name)
            setattr(case.cache, name, torch.cat((old, torch.zeros_like(old))))
    if hasattr(module, "finalize_linear_replay"):
        legacy_headers(case)
    return case


def produced_records(case):
    banked = case.cache.k.shape[0] == 2 * case.batch
    offset = case.batch if banked else 0
    return {
        "k": case.cache.k[offset : offset + case.batch],
        "u": case.cache.u,
        "g": case.cache.g[offset : offset + case.batch],
        "conv_inputs": case.cache.conv_inputs,
    }


def protocol_launches(
    module, case, *, frozen=False, scratch=None, fused_verify_warps=None
):
    calls, output = capture_launches(
        case,
        tuple(getattr(module, kernel) for kernel in KERNELS),
        module=module,
        finalize=hasattr(module, "finalize_linear_replay"),
        frozen=frozen,
        scratch=scratch,
    )
    banked = case.family == "kda" and case.cache.k.shape[0] == 2 * case.batch
    stages = ["convolution", "fused_replay_verify"] if banked else list(STAGES[:3])
    if hasattr(module, "finalize_linear_replay"):
        stages.append("finalize")
    if banked:
        parameters = dict(
            zip(inspect.signature(calls[1].kernel.fn).parameters, calls[1].args)
        )
        assert parameters["FUSE_REBUILD"] is True, "benchmark must use fused KDA"
        calls[1].production_num_warps = calls[1].kwargs["num_warps"]
        if module is replay and fused_verify_warps is not None:
            calls[1].kwargs["num_warps"] = fused_verify_warps
    return calls, output, stages


def single_step_graph(calls, prepare):
    # A fused verify overwrites U after consuming it. Capture one step only;
    # every warmup and measured invocation needs a fresh producer history.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepare()
        for call in calls:
            call()
        prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for call in calls:
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return graph


def measure_protocol(modules, family, batch, steps, accepted, args):
    outputs, correctness_configs = {}, {}
    for name, module in modules.items():
        case = protocol_case(module, family, batch, steps, accepted)
        calls, output, _ = protocol_launches(
            module, case, fused_verify_warps=args.fused_verify_warps
        )
        correctness_configs[name] = launch_configuration(calls)
        for call in calls:
            call()
        outputs[name] = {
            "output": output.clone(),
            "ssm": case.ssm,
            "conv": case.conv,
            **produced_records(case),
        }
    reference = outputs["candidate"]
    correctness = protocol_check(
        outputs,
        reference,
        args,
        family,
        batch,
        steps,
        accepted,
        "independent",
        correctness_configs,
    )

    common = protocol_case(replay, family, batch, steps, accepted)
    banked = family == "kda"
    produced = produced_records(common)
    common.sink = [produced["k"], common.cache.u, produced["g"]]
    if not banked:
        common.sink = [
            torch.empty_like(common.cache.k),
            common.cache.u,
            torch.empty_like(common.cache.g),
        ]
    scratch = (
        torch.empty_like(common.raw.flatten(0, 1)),
        common.raw.new_empty((batch * steps, common.hv, common.dim)),
    )
    mutable = [
        common.ssm,
        common.conv,
        common.cache.k,
        common.cache.u,
        common.cache.g,
        common.cache.conv_inputs,
    ]
    snapshots = [tensor.clone() for tensor in mutable]

    def reset():
        for tensor, snapshot in zip(mutable, snapshots):
            tensor.copy_(snapshot)

    launches, stages, graphs, bindings = {}, {}, {}, {}
    samples = {
        name: {
            mode: {"cuda_us": [], "host_enqueue_us": []} for mode in ("eager", "graph")
        }
        for name in modules
    }
    stage_samples, codegen, stage_graphs = {}, {}, {}
    for name, module in modules.items():
        case = SimpleNamespace(**vars(common))
        case.cache = SimpleNamespace(**vars(common.cache))
        case.inputs = SimpleNamespace(**vars(common.inputs))
        if name != "candidate" and banked:
            case.cache.k, case.cache.g = common.cache.k[:batch], common.cache.g[:batch]
        if hasattr(module, "finalize_linear_replay"):
            legacy_headers(case)
        launches[name], _, stages[name] = protocol_launches(
            module,
            case,
            frozen=not (name == "candidate" and banked),
            scratch=scratch,
            fused_verify_warps=args.fused_verify_warps,
        )
        # Logical old/new views alias even though fused KDA receives both KG
        # banks through one pointer and the baseline writes through sink views.
        bindings[name] = {
            key: tensor.data_ptr()
            for key, tensor in {
                "raw": case.raw,
                "gate": case.gate,
                "beta": case.beta,
                "weights": case.weights,
                "ssm": case.ssm,
                "conv": case.conv,
                "old_k": case.cache.k[:batch],
                "old_u": case.cache.u,
                "old_g": case.cache.g[:batch],
                "conv_inputs": case.cache.conv_inputs,
                "new_k": common.sink[0],
                "new_u": common.sink[1],
                "new_g": common.sink[2],
                "convolved": scratch[0],
                "output": scratch[1],
                **vars(common.inputs),
            }.items()
        }
        assert (
            bindings[name] == bindings["candidate"]
        ), "logical timing buffers must alias"
        graphs[name] = single_step_graph(launches[name], reset)
        reset()
        graphs[name].replay()
        protocol_check(
            {
                name: {
                    "output": scratch[1],
                    "ssm": common.ssm,
                    "conv": common.conv,
                    "k": common.sink[0],
                    "u": common.sink[1],
                    "g": common.sink[2],
                    "conv_inputs": common.cache.conv_inputs,
                }
            },
            reference,
            args,
            family,
            batch,
            steps,
            accepted,
            f"shared_{name}",
            {name: launch_configuration(launches[name])},
        )
        if args.profile_stages:
            stage_samples[name], codegen[name] = {}, {}
            for index, (stage, call) in enumerate(zip(stages[name], launches[name])):

                def prepare_prefix(index=index, name=name):
                    reset()
                    for prefix_call in launches[name][:index]:
                        prefix_call()

                stage_graphs[name, stage] = single_step_graph([call], prepare_prefix)
                codegen[name][stage] = kernel_codegen(
                    call,
                    args.output
                    / f"protocol_{family}_bs{batch}_steps{steps}_accept{accepted}_{name}_{stage}.ptx",
                )
                stage_samples[name][stage] = []

    for repeat in range(args.rounds):
        occupancy()
        order = list(modules) if repeat % 2 == 0 else list(reversed(modules))
        for name in modules:
            for mode in samples[name].values():
                for values in mode.values():
                    values.append([])
            if args.profile_stages:
                for values in stage_samples[name].values():
                    values.append([])
        for _ in range(args.samples):
            for name in order:
                for mode in ("eager", "graph"):
                    reset()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record()
                    begin = time.perf_counter_ns()
                    if mode == "graph":
                        graphs[name].replay()
                    else:
                        for call in launches[name]:
                            call()
                    host_us = (time.perf_counter_ns() - begin) / 1000
                    end.record()
                    end.synchronize()
                    samples[name][mode]["cuda_us"][-1].append(
                        start.elapsed_time(end) * 1000
                    )
                    samples[name][mode]["host_enqueue_us"][-1].append(host_us)
                if args.profile_stages:
                    for index, stage in enumerate(stages[name]):
                        reset()
                        for prefix_call in launches[name][:index]:
                            prefix_call()
                        start, end = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        start.record()
                        stage_graphs[name, stage].replay()
                        end.record()
                        end.synchronize()
                        stage_samples[name][stage][-1].append(
                            start.elapsed_time(end) * 1000
                        )
    result = {
        "family": family,
        "batch": batch,
        "steps": steps,
        "accepted": accepted,
        "banked_kda": banked,
        "capture_steps": 1,
        "fused_verify_warps_override": args.fused_verify_warps,
        "launch_config": {
            name: launch_configuration(calls) for name, calls in launches.items()
        },
        "correctness": correctness,
        "timing_buffer_alias_verified": True,
        "timing_tensor_addresses": bindings,
        "kernel_sequence": {
            name: [call.kernel.fn.__name__ for call in calls]
            for name, calls in launches.items()
        },
        "stages": stages,
        "samples_us": samples,
        "p50_us": {
            name: {
                mode: {
                    metric: statistics.median(x for row in rounds for x in row)
                    for metric, rounds in metrics.items()
                }
                for mode, metrics in modes.items()
            }
            for name, modes in samples.items()
        },
    }
    if args.profile_stages:
        result["profile_stages"] = {
            "samples_us": stage_samples,
            "codegen": codegen,
            "p50_us": {
                name: {
                    stage: statistics.median(x for row in rounds for x in row)
                    for stage, rounds in values.items()
                }
                for name, values in stage_samples.items()
            },
            "pipeline": "measured separately in samples_us; never sum stage times",
        }
    return result


def kernel_codegen(call, path):
    ptx = call.compiled.asm["ptx"]
    path.write_text(ptx)
    return {
        "ptx_file": path.name,
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
        "fma_f32": len(re.findall(r"\bfma(?:\.[a-z]+)*\.f32\b", ptx)),
        "mul_f32": len(re.findall(r"\bmul\.(?:rn\.)?f32\b", ptx)),
        "bf16_conversions": len(re.findall(r"\bcvt\.[^\s;]*bf16[^\s;]*", ptx)),
        "registers": call.compiled.n_regs,
        "spills": call.compiled.n_spills,
        "shared_bytes": call.compiled.metadata.shared,
    }


def difference(actual, expected, *, require_finite=True):
    delta = (actual.float() - expected.float()).abs()
    actual_finite, expected_finite = bool(torch.isfinite(actual).all()), bool(
        torch.isfinite(expected).all()
    )
    if require_finite:
        assert actual_finite and expected_finite
    relative = delta.double() / expected.double().abs()
    relative = torch.where(delta == 0, 0, relative)
    max_abs, max_rel = float(delta.max()), float(relative.max())
    return {
        "dtype": str(actual.dtype),
        "shape": list(actual.shape),
        "expected_dtype": str(expected.dtype),
        "finite": actual_finite and expected_finite,
        "actual_finite": actual_finite,
        "expected_finite": expected_finite,
        "different": int((actual != expected).sum()),
        "elements": actual.numel(),
        "max_abs": max_abs if math.isfinite(max_abs) else str(max_abs),
        "max_rel": max_rel if math.isfinite(max_rel) else str(max_rel),
    }


def launch_configuration(calls):
    return [
        {
            "kernel": call.kernel.fn.__name__,
            "num_warps": call.kwargs.get("num_warps"),
            "production_num_warps": getattr(
                call, "production_num_warps", call.kwargs.get("num_warps")
            ),
        }
        for call in calls
    ]


def protocol_check(
    outputs, reference, args, family, batch, steps, accepted, phase, launch_config
):
    # Finish every comparison before enforcing the strict gate, so a failure
    # leaves the other fields available for numerical diagnosis.
    checks = {
        name: {
            key: difference(actual, reference[key], require_finite=False)
            for key, actual in values.items()
        }
        for name, values in outputs.items()
    }
    path = (
        args.output
        / f"correctness_{family}_bs{batch}_steps{steps}_accept{accepted}_{phase}.json"
    )
    path.write_text(
        json.dumps(
            {
                "family": family,
                "batch": batch,
                "steps": steps,
                "accepted": accepted,
                "phase": phase,
                "fused_verify_warps_override": args.fused_verify_warps,
                "configuration": (
                    "production"
                    if args.fused_verify_warps is None
                    else "counterfactual override"
                ),
                "gate": {"rtol": 0, "atol": 0},
                "launch_config": launch_config,
                "fields": checks,
            },
            indent=2,
        )
        + "\n"
    )
    for name, values in outputs.items():
        for key, actual in values.items():
            assert (
                checks[name][key]["actual_finite"]
                and checks[name][key]["expected_finite"]
            ), f"{phase}: nonfinite {name}.{key}; complete differences: {path}"
            torch.testing.assert_close(
                actual,
                reference[key],
                rtol=0,
                atol=0,
                msg=f"{phase}: {name}.{key}; complete differences: {path}",
            )
    return checks


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
        for field in ("prev_accept_lengths", "history_valid_lengths"):
            getattr(case.inputs, field).zero_()
        case.inputs.init_kinds.fill_(1)
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
        snapshot = {key: value.clone() for key, value in vars(case.cache).items()}
        case.inputs.state_read_block_ids.fill_(2)
        case.inputs.active_block_ids.fill_(1)
        case.inputs.history_valid_lengths.fill_(steps)
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
            codegen[name][stage] = kernel_codegen(
                call, output / f"case_{index}_{name}_{stage}.ptx"
            )
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
    parser.add_argument("--mode", choices=("protocol", "precision"), default="protocol")
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument(
        "--fused-verify-warps",
        type=int,
        choices=(2, 4),
        default=None,
        help="counterfactual candidate fused-verify override; default uses production",
    )
    parser.add_argument("--batches", default="1,16,128")
    parser.add_argument("--accept-lengths", default="1,5")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--inner-repeats",
        type=int,
        default=32,
        help="precision mode only; protocol captures one step",
    )
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
    if args.profile_stages and args.mode != "protocol":
        parser.error("--profile-stages requires --mode=protocol")
    if args.fused_verify_warps is not None and args.mode != "protocol":
        parser.error("--fused-verify-warps requires --mode=protocol")
    assert args.rounds >= 2 and args.samples > 0 and args.inner_repeats > 0
    batches = [int(x) for x in args.batches.split(",")]
    accepts = [int(x) for x in args.accept_lengths.split(",")]
    assert all(x > 0 for x in batches) and all(1 <= x <= args.steps for x in accepts)
    args.output.mkdir(parents=True, exist_ok=True)
    baseline_encoded = os.environ.get("RTP_LINEAR_REPLAY_BASELINE_SOURCE_BASE64")
    if baseline_encoded:
        if args.baseline_source is not None:
            parser.error("pass only one baseline source: path or base64 environment")
        try:
            baseline_bytes = base64.b64decode(
                "".join(baseline_encoded.split()), validate=True
            )
        except ValueError:
            parser.error("RTP_LINEAR_REPLAY_BASELINE_SOURCE_BASE64 is invalid")
        if not baseline_bytes:
            parser.error("baseline source is empty")
        args.baseline_source = args.output / "protocol_baseline.py"
        args.baseline_source.write_bytes(baseline_bytes)
    selected = variants(args.output) if args.mode == "precision" else None
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
            "baseline_sha256": (
                hashlib.sha256(args.baseline_source.read_bytes()).hexdigest()
                if args.baseline_source is not None
                else None
            ),
            "baseline_revision": os.environ.get(
                "RTP_BENCHMARK_BASELINE_REVISION", "unknown"
            ),
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
            "mode": args.mode,
            "precision_scope": "unbanked three-kernel path with generic convolution; excludes fused KDA",
            "profile_stages": args.profile_stages,
            "fused_verify_warps_override": args.fused_verify_warps,
            "fused_verify_configuration": (
                "production"
                if args.fused_verify_warps is None
                else "counterfactual override"
            ),
            "timing": "CUDA events and host enqueue wall time; graph time / inner_repeats; microseconds",
            "excluded": [
                "allocation",
                "compilation",
                "capture",
                "host wrapper",
                "protocol state/record restoration",
            ],
            "baseline_finalize": "included in protocol mode when present",
            "frozen_logs": "protocol restores state and producer records outside events before every one-step sample; fused U may be overwritten",
            "protocol_timing_buffers": "baseline/candidate share all common inputs, state, records, sinks and scratch outputs; correctness uses independent buffers",
            "rounds": args.rounds,
            "samples": args.samples,
            "inner_repeats": args.inner_repeats if args.mode == "precision" else 1,
        },
        "correctness": [],
        "cases": [],
    }
    destination = args.output / "linear_replay_microbenchmark.json"
    if args.mode == "protocol":
        modules = protocol_modules(args.baseline_source)
        report["contract"]["weight_dtype"] = "bfloat16"
        report["contract"]["timing_boundary"] = (
            "production kernel launches including baseline header finalize; "
            "host enqueue excludes wrapper validation and allocation"
        )
        for family in ("gdn", "kda"):
            for batch in batches:
                for accepted in accepts:
                    result = measure_protocol(
                        modules, family, batch, args.steps, accepted, args
                    )
                    report["cases"].append(result)
                    destination.write_text(json.dumps(report, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                key: result[key]
                                for key in ("family", "batch", "accepted", "p50_us")
                            }
                        ),
                        flush=True,
                    )
        print(f"Results: {destination}", flush=True)
        return
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
