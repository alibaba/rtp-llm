"""Full local MoE comparison in isolated original/backport Humming workers.

Normal DeepEP transport is not mocked or claimed covered: this target covers
the local route/gather -> W4A8 w13 -> weighted BF16 SwiGLU -> G128 quant/W4A8
w2 -> exact route reduction chain. Communication acceptance is a later gate.
Only the candidate uses terminal offsets and masked loads for unwritten tails.
No installed package mutation. Parent owns GPU/Bazel and runs workers serially.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from types import SimpleNamespace

import pytest


def _activate():
    assert not any(n == "humming" or n.startswith("humming.") for n in sys.modules)
    filename = Path(os.environ["DSV41_HUMMING_TERMINAL_HELPER"]).resolve(strict=True)
    spec = importlib.util.spec_from_file_location("terminal_source_selection", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.activate_before_import()


def _worker(args):
    activation = _activate()
    import torch
    import humming
    from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_graph import (
        forward_local_static,
    )
    from rtp_llm.models_py.modules.dsv41.moe._sm90_route_kernels import (
        prepare_static_routes,
        routed_swiglu,
        reduce_static_routes,
    )
    from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_humming import (
        _GroupedLocalExperts,
    )
    from rtp_llm.models_py.modules.dsv41.test.sm90_terminal_route_candidate import (
        forward_local_terminal,
        masked_terminal_swiglu,
    )
    from rtp_llm.utils.model_weight import W

    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    assert str(humming.__version__) == "0.1.16"
    assert (
        Path(humming.__file__).resolve().parent
        == Path(activation["selected_site"]) / "humming"
    )
    arm = os.environ["DSV41_HUMMING_TERMINAL_ARM"]
    assert arm in ("baseline", "candidate")
    os.environ["DSV41_SM90_NATIVE_A8"] = "1"
    os.environ["DSV41_SM90_A8_PROMOTION_GROUP"] = "32"
    torch.backends.cuda.matmul.allow_tf32 = False

    def bits(a, b):
        assert a.shape == b.shape and a.dtype == b.dtype
        aa = torch.empty(a.numel(), device=a.device, dtype=a.dtype)
        bb = torch.empty(b.numel(), device=b.device, dtype=b.dtype)
        aa.copy_(a.reshape(-1))
        bb.copy_(b.reshape(-1))
        torch.testing.assert_close(
            aa.view(torch.uint8), bb.view(torch.uint8), rtol=0, atol=0
        )

    def sha(t):
        return hashlib.sha256(
            t.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()

    def capture(fn):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fn()
        return graph, result

    torch.manual_seed(252520)
    original = _GroupedLocalExperts(SimpleNamespace(swiglu_limit=10.0))
    weights = {}
    for wk, sk, n, k in (
        (W.v4_routed_w1_w, W.v4_routed_w1_s, args.inter, args.dim),
        (W.v4_routed_w3_w, W.v4_routed_w3_s, args.inter, args.dim),
        (W.v4_routed_w2_w, W.v4_routed_w2_s, args.dim, args.inter),
    ):
        weights[wk] = torch.randint(
            -128, 128, (48, n, k // 2), device="cuda", dtype=torch.int8
        )
        weights[sk] = torch.randint(
            121, 125, (48, n, k // 32), device="cuda", dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
    weight_hashes = {str(key): sha(value) for key, value in weights.items()}
    original.setup_weights(weights)
    assert (
        original.native_a8
        and original.w13.promotion_group_size == original.w2.promotion_group_size == 32
    )
    assert all(
        str(linear.config.a_dtype) == "float8e4m3"
        and linear.compute_config["use_batch_invariant"]
        and not linear.compute_config["use_f16_accum"]
        for linear in (original.w13, original.w2)
    )

    class PoisonOutput:
        """Real GEMM, with intentionally poisoned scratch allocation only."""

        def __init__(self, linear):
            self.linear = linear
            self.promotion_group_size = linear.promotion_group_size
            self.last = self.last_offsets = self.backing = None

        def __call__(self, x, offsets):
            # Capture the actual device offsets used by each GEMM. No CPU
            # count read or dynamic shape is introduced into this path.
            self.last_offsets = offsets
            self.backing = torch.full(
                (x.shape[0] + 2, self.linear.out_features),
                float("nan"),
                device=x.device,
                dtype=torch.bfloat16,
            )
            self.last = self.backing[1:-1]
            self.linear.prepared(
                *self.linear.quantize_input(x), offsets, outputs=self.last
            )
            return self.last

    local = SimpleNamespace(
        native_a8=True,
        cfg=original.cfg,
        w13=PoisonOutput(original.w13),
        w2=PoisonOutput(original.w2),
    )
    fn = forward_local_static if arm == "baseline" else forward_local_terminal
    records, timings = {}, []

    def update(x, route_weights, indices, step):
        rows = x.shape[0]
        torch.manual_seed(25300 + rows * 10 + step)
        x.copy_(torch.randn_like(x) * 0.125)
        route_weights.copy_(torch.randn_like(route_weights) * 0.2)
        indices.fill_(-1)
        if step == 0:  # duplicate first expert, six real routes plus padding
            indices[:, :6] = 192
        elif step == 1:
            indices[:, :6] = 239
        elif step == 2:
            indices[:, :3] = 192
            indices[:, 3:6] = 239
        elif step == 3:
            flat = indices.reshape(-1)
            count = max(1, flat.numel() // 3)
            flat[:count] = 192 + torch.arange(count, device=x.device) * 7 % 48
            indices[:, -1] = 240  # positive out-of-range IDs are also invalid
        elif step == 4:
            pass  # genuinely empty route set
        else:
            indices.copy_(
                192
                + torch.arange(indices.numel(), device=x.device).reshape_as(indices)
                % 48
            )
        valid = (indices >= 192) & (indices < 240)
        route_weights.masked_fill_(~valid, float("nan"))
        x.masked_fill_(~valid.any(-1)[:, None], float("nan"))

    def verify_metadata(indices, offsets, terminal):
        # Independent counting oracle, no production routing helper.
        flat = indices.flatten()
        counts = torch.stack(
            [(flat == expert).sum(dtype=torch.int32) for expert in range(192, 240)]
        )
        expected = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
        if not terminal:
            expected[-1] = flat.numel()
        bits(offsets, expected)

    for rows in (1, 4, 8, 32):
        x = torch.empty((rows, args.dim), device="cuda", dtype=torch.bfloat16)
        route_weights = torch.empty((rows, 8), device="cuda", dtype=torch.float32)
        indices = torch.empty((rows, 8), device="cuda", dtype=torch.int64)
        update(x, route_weights, indices, 0)
        for output_dtype in (torch.float32, torch.bfloat16):
            call = lambda: fn(
                local, x, route_weights, indices, 192, 240, output_dtype=output_dtype
            )
            graph, captured = capture(call)
            capture_buffers = [
                (linear.last, linear.backing, linear.last_offsets)
                for linear in (local.w13, local.w2)
            ]
            for step in range(6):
                update(x, route_weights, indices, step)
                input_hashes = tuple(sha(t) for t in (x, route_weights, indices))
                valid_total = int(((indices >= 192) & (indices < 240)).sum())
                eager = call()
                for linear in (local.w13, local.w2):
                    verify_metadata(indices, linear.last_offsets, arm == "candidate")
                    assert bool(torch.isnan(linear.backing[:1]).all()) and bool(
                        torch.isnan(linear.backing[-1:]).all()
                    )
                    if arm == "candidate":
                        assert bool(
                            torch.isnan(linear.last[valid_total:]).all()
                        ), "candidate GEMM wrote poison tail"
                graph.replay()
                torch.cuda.synchronize()
                bits(captured, eager)
                for buffer, backing, offsets in capture_buffers:
                    verify_metadata(indices, offsets, arm == "candidate")
                    assert bool(torch.isnan(backing[:1]).all()) and bool(
                        torch.isnan(backing[-1:]).all()
                    )
                    if arm == "candidate":
                        assert bool(torch.isnan(buffer[valid_total:]).all())
                assert bool(
                    torch.isfinite(captured).all()
                ), "invalid poison leaked through GEMM-tail consumers"
                if valid_total == 0:
                    bits(captured, torch.zeros_like(captured))
                assert input_hashes == tuple(
                    sha(t) for t in (x, route_weights, indices)
                )
                key = f"rows{rows}_dtype{output_dtype}_step{step}"
                records[key] = dict(
                    output=captured.cpu(), inputs=input_hashes, valid_total=valid_total
                )

        if args.benchmark:
            # Untainted production allocations: poison wrappers and all oracle
            # checks are excluded, while the candidate's device sum/copy stays.
            for step in (0, 4):
                update(x, route_weights, indices, step)
                graph, output = capture(
                    lambda: fn(
                        original,
                        x,
                        route_weights,
                        indices,
                        192,
                        240,
                        output_dtype=torch.bfloat16,
                    )
                )
                samples = []
                for _ in range(5):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record()
                    for _ in range(100):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000 / 100)
                timings.append(
                    dict(
                        rows=rows,
                        capacity=rows * 8,
                        valid=rows * 6 if step == 0 else 0,
                        us=statistics.median(samples),
                    )
                )

    # Direct poison seams isolate masking semantics from GEMM quantization.
    # Valid NaN must remain NaN, never be globally sanitized to a finite value.
    x = torch.ones((2, args.dim), device="cuda", dtype=torch.bfloat16)
    route_weights = torch.tensor(
        [[1.0, 0.5, float("nan"), 1.0, 1.0, 1.0, 1.0, 1.0]] * 2, device="cuda"
    )
    indices = torch.tensor(
        [[192, 239, -1, -1, -1, -1, -1, -1]] * 2, device="cuda", dtype=torch.int64
    )
    slots, offsets, valid, inverse, _, clean_weights = prepare_static_routes(
        x, route_weights, indices, 192, 240
    )
    grouped_valid = valid[slots]
    gate_up = torch.randn((16, args.inter * 2), device="cuda", dtype=torch.bfloat16)
    gate_up[~grouped_valid] = float("nan")
    for dtype in (torch.float32, torch.bfloat16):
        actual = masked_terminal_swiglu(
            gate_up, clean_weights, slots, valid, clamp_limit=10.0, output_dtype=dtype
        )
        safe = torch.where(grouped_valid[:, None], gate_up, 0)
        old = routed_swiglu(
            safe, clean_weights, slots, clamp_limit=10.0, output_dtype=dtype
        )
        bits(actual, old)
        bits(actual[~grouped_valid], torch.zeros_like(actual[~grouped_valid]))
        gate_up[0, 0] = float("nan")
        nan_output = masked_terminal_swiglu(
            gate_up, clean_weights, slots, valid, clamp_limit=10.0, output_dtype=dtype
        )
        assert bool(torch.isnan(nan_output[0, 0]))
        gate_up[0, 0] = 0
        down = torch.ones((16, args.dim), device="cuda", dtype=torch.bfloat16)
        down[~grouped_valid] = float("nan")
        reduced = reduce_static_routes(down, inverse, valid, 2, 8, output_dtype=dtype)
        bits(reduced, torch.full_like(reduced, 2))
        down[0, 0] = float("nan")
        reduced = reduce_static_routes(down, inverse, valid, 2, 8, output_dtype=dtype)
        assert bool(torch.isnan(reduced[0, 0]))
    # Empty physical input is a separate no-GEMM contract.
    empty = fn(local, x[:0], route_weights[:0], indices[:0], 192, 240)
    assert empty.shape == (0, args.dim) and empty.dtype == torch.float32
    torch.save(
        dict(
            records=records,
            weights=weight_hashes,
            timings=timings,
            activation=activation,
        ),
        args.output,
    )
    print(
        "TERMINAL_CHAIN_WORKER "
        + json.dumps(
            dict(
                arm=arm,
                shape=[48, args.dim, args.inter],
                cases=len(records),
                package=humming.__file__,
                cache=os.environ["HUMMING_CACHE_DIR"],
            )
        ),
        flush=True,
    )


def _child_environment(arm):
    environment = os.environ.copy()
    environment["DSV41_HUMMING_TERMINAL_ARM"] = arm
    paths, seen = [], set()
    for item in [*sys.path, *environment.get("PYTHONPATH", "").split(os.pathsep)]:
        path = Path(item or os.getcwd()).resolve()
        if path.is_dir() and str(path) not in seen:
            paths.append(str(path))
            seen.add(str(path))
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    assert (
        len(environment["PYTHONPATH"].encode()) < 96 * 1024
    ), "Child PYTHONPATH too large; do not silently drop imports"
    return environment


def _compare(tmp_path, *, real):
    import torch

    assert os.environ.get("DSV41_HUMMING_TERMINAL_HELPER")
    name = "real" if real else "small"
    if os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        tmp_path = (
            Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / f"terminal_chain_{name}"
        )
        tmp_path.mkdir(parents=True, exist_ok=False)
    results = {}
    for arm in ("baseline", "candidate"):
        output = tmp_path / f"{arm}.pt"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--dim",
            str(5120 if real else 128),
            "--inter",
            str(2304 if real else 128),
            "--output",
            str(output),
        ]
        if os.getenv("DSV41_SM90_TERMINAL_CHAIN_BENCH") == "1":
            command.append("--benchmark")
        completed = subprocess.run(
            command,
            env=_child_environment(arm),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        log = tmp_path / f"{arm}.log"
        log.write_text(completed.stdout + "\n" + completed.stderr)
        print(completed.stdout, flush=True)
        assert (
            completed.returncode == 0
        ), f"{arm} failed; {log}\n{completed.stderr[-12000:]}"
        results[arm] = torch.load(output, map_location="cpu", weights_only=False)
    old, new = results["baseline"], results["candidate"]
    assert old["activation"]["selected_site"] != new["activation"]["selected_site"]
    assert old["activation"]["cache_paths"] != new["activation"]["cache_paths"]
    assert old["weights"] == new["weights"]
    assert old["records"].keys() == new["records"].keys()
    for key, a in old["records"].items():
        b = new["records"][key]
        assert a["inputs"] == b["inputs"] and a["valid_total"] == b["valid_total"]
        torch.testing.assert_close(
            a["output"].view(torch.uint8),
            b["output"].view(torch.uint8),
            rtol=0,
            atol=0,
            msg=lambda message: f"{key}: {message}",
        )
    for a, b in zip(old["timings"], new["timings"]):
        assert a["rows"] == b["rows"] and a["valid"] == b["valid"]
        print(
            "TERMINAL_CHAIN_BENCH "
            + json.dumps(
                dict(
                    shape=name,
                    **a,
                    candidate_us=b["us"],
                    speedup=a["us"] / b["us"],
                    scope="local full MoE chain incl terminal sum/copy; excludes DeepEP communication/E2E",
                )
            ),
            flush=True,
        )


def test_small_terminal_full_chain_exact_and_poison(tmp_path):
    _compare(tmp_path, real=False)


def test_real_terminal_full_chain_exact_and_poison(tmp_path):
    _compare(tmp_path, real=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--inter", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", action="store_true")
    _worker(parser.parse_args())
