"""Selected GDN prefix states: original-kernel oracle and kernel-only benchmark.

Set PERF_JSON to run the opt-in benchmark after correctness tests. An optional
RTP_GDN_ORIGINAL_SOURCE points to a frozen, pre-change chunk_delta_h.py; CI can
instead use the unchanged no-request specialization as its baseline.
"""

import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from rtp_llm.models_py.triton_kernels.fla.cumsum import chunk_local_cumsum
from rtp_llm.models_py.triton_kernels.fla.solve_tril import solve_tril
from rtp_llm.models_py.triton_kernels.fla.wy_fast import recompute_w_u_fwd


def load_baseline(
    env_name="RTP_GDN_ORIGINAL_SOURCE",
    module_name="gdn_frozen_original",
    optional=False,
):
    source = os.environ.get(env_name)
    if not source:
        if optional:
            return None, None
        return chunk_gated_delta_rule_fwd_h, {
            "kind": "current_no_request_specialization"
        }
    path = Path(source).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.chunk_gated_delta_rule_fwd_h, {
        "kind": (
            "frozen_previous_snapshot_source"
            if optional
            else "frozen_pre_change_source"
        ),
        "source": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


BASELINE, BASELINE_METADATA = load_baseline()
PREVIOUS, PREVIOUS_METADATA = load_baseline(
    "RTP_GDN_PREVIOUS_SOURCE", "gdn_frozen_previous", optional=True
)


def make_inputs(
    lengths,
    heads=4,
    key_heads=2,
    kdim=64,
    vdim=64,
    dtype=torch.bfloat16,
    packed=False,
    initial=True,
):
    """Prepare actual CUDA GDN WY inputs, outside any measured region."""
    torch.manual_seed(1729)
    batch, tokens = (1, sum(lengths)) if packed else (len(lengths), lengths[0])
    assert packed or len(set(lengths)) == 1
    k = F.normalize(
        torch.randn(batch, tokens, key_heads, kdim, device="cuda"), dim=-1
    ).to(dtype)
    v = torch.randn(batch, tokens, heads, vdim, device="cuda", dtype=dtype)
    beta = torch.rand(batch, tokens, heads, device="cuda", dtype=torch.float32)
    # Moderate negative gates preserve measurable state across chunk boundaries.
    raw_g = -torch.rand(batch, tokens, heads, device="cuda") * 0.03
    cu = None
    if packed:
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu = torch.tensor(offsets, device="cuda", dtype=torch.int64)
    g = chunk_local_cumsum(raw_g, chunk_size=64, cu_seqlens=cu)
    a = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g, cu_seqlens=cu, output_dtype=torch.float32
    )
    a = solve_tril(a, cu_seqlens=cu, output_dtype=dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, g_cumsum=g, A=a, cu_seqlens=cu)
    h0 = (
        torch.randn(len(lengths), heads, vdim, kdim, device="cuda") * 0.1
        if initial
        else None
    )
    return dict(
        k=k, w=w, u=u, g=g, initial_state=h0, output_final_state=True, cu_seqlens=cu
    ), (v, beta, raw_g)


def positions_for(lengths):
    # Prefix counts, deliberately unsorted, with duplicates and invalid entries.
    return [
        [length, 0, 1, 31, 63, 64, 65, 127, 128, 129, 257, 64, -1, length + 1]
        for length in lengths
    ]


def selected_buffer(args, positions):
    return torch.full(
        (
            *positions.shape,
            args["u"].shape[-2],
            args["u"].shape[-1],
            args["k"].shape[-1],
        ),
        float("nan"),
        device="cuda",
        dtype=torch.float32,
    )


def run_selected(args, positions, output):
    return chunk_gated_delta_rule_fwd_h(
        **args, state_positions=positions, selected_states=output
    )


def prefix_oracle(args, lengths, requests):
    """Original kernel on contiguous prefix inputs; no new snapshot code."""
    output = torch.zeros(
        len(lengths),
        len(requests[0]),
        args["u"].shape[-2],
        args["u"].shape[-1],
        args["k"].shape[-1],
        device="cuda",
        dtype=torch.float32,
    )
    packed = args["cu_seqlens"] is not None
    offset = 0
    for n, length in enumerate(lengths):
        cache = {}
        for slot, position in enumerate(requests[n]):
            if position < 0 or position > length:
                continue
            if position == 0:
                if args["initial_state"] is not None:
                    output[n, slot].copy_(args["initial_state"][n])
                continue
            if position not in cache:
                prefix = {}
                for key in ("k", "w", "u", "g"):
                    value = args[key]
                    prefix[key] = (
                        value[:, offset : offset + position]
                        if packed
                        else value[n : n + 1, :position]
                    ).contiguous()
                prefix["initial_state"] = (
                    None
                    if args["initial_state"] is None
                    else args["initial_state"][n : n + 1].contiguous()
                )
                cache[position] = BASELINE(**prefix, output_final_state=True)[2][0]
            output[n, slot].copy_(cache[position])
        offset += length
    return output


def recurrent_oracle(args, raw, lengths, requests):
    """Independent FP32 token recurrence, V-first state convention."""
    v, beta, raw_g = (tensor.float() for tensor in raw)
    k = args["k"].float().repeat_interleave(v.shape[-2] // args["k"].shape[-2], dim=-2)
    result = torch.zeros(
        len(lengths),
        len(requests[0]),
        v.shape[-2],
        v.shape[-1],
        k.shape[-1],
        device="cuda",
    )
    offset = 0
    for n, length in enumerate(lengths):
        row, start = (0, offset) if args["cu_seqlens"] is not None else (n, 0)
        state = (
            torch.zeros_like(result[n, 0])
            if args["initial_state"] is None
            else args["initial_state"][n].float().clone()
        )
        for position in range(length + 1):
            for slot, request in enumerate(requests[n]):
                if position == request:
                    result[n, slot].copy_(state)
            if position == length:
                break
            t = start + position
            state = state * raw_g[row, t].exp()[:, None, None]
            residual = v[row, t] - (state * k[row, t, :, None, :]).sum(-1)
            state = (
                state
                + (beta[row, t, :, None] * residual)[:, :, None] * k[row, t, :, None, :]
            )
        offset += length
    return result


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "selected states currently require NVIDIA CUDA",
)
class SelectedStatesTest(unittest.TestCase):
    def assert_outputs_equal(self, expected, actual):
        for original, candidate in zip(expected, actual):
            if original is None:
                self.assertIsNone(candidate)
            else:
                torch.testing.assert_close(candidate, original, rtol=0, atol=0)

    def check_case(self, lengths, **options):
        args, raw = make_inputs(lengths, **options)
        requests = positions_for(lengths)
        positions = torch.tensor(requests, device="cuda", dtype=torch.int64)
        selected = selected_buffer(args, positions)
        baseline = BASELINE(**args)
        self.assert_outputs_equal(baseline, chunk_gated_delta_rule_fwd_h(**args))
        self.assert_outputs_equal(baseline, run_selected(args, positions, selected))
        torch.testing.assert_close(
            selected, prefix_oracle(args, lengths, requests), rtol=0, atol=0
        )
        # The chunk algorithm itself rounds WY intermediates to the input dtype.
        # Require the same established 0.5% RMS envelope against FP32 recurrence.
        reference = recurrent_oracle(args, raw, lengths, requests)
        error = (selected - reference).square().mean().sqrt()
        scale = reference.square().mean().sqrt().clamp_min(1e-8)
        self.assertLess(float(error / scale), 0.005)
        self.assertTrue(torch.isfinite(selected).all().item())

    def test_fixed_and_grouped_heads(self):
        for dtype in (torch.bfloat16, torch.float16):
            for kdim, vdim in ((64, 64), (128, 128), (192, 33), (256, 64)):
                with self.subTest(dtype=dtype, kdim=kdim, vdim=vdim):
                    self.check_case([257, 257], dtype=dtype, kdim=kdim, vdim=vdim)

    def test_packed_variable_lengths(self):
        self.check_case([1, 63, 64, 65, 129, 257], packed=True)
        self.check_case(
            [31, 128, 193], packed=True, dtype=torch.float16, kdim=128, initial=False
        )

    def test_odd_dimensions(self):
        self.check_case([129], kdim=65, vdim=37)

    def test_empty_requests_and_optional_outputs(self):
        args, _ = make_inputs([65], initial=False)
        positions = torch.empty(1, 0, device="cuda", dtype=torch.int32)
        selected = selected_buffer(args, positions)
        self.assert_outputs_equal(
            BASELINE(**args), run_selected(args, positions, selected)
        )
        args.update(output_final_state=False, save_new_value=False)
        positions = torch.tensor([[0, 1, 64, 65, -1]], device="cuda", dtype=torch.int32)
        selected = selected_buffer(args, positions)
        self.assert_outputs_equal(
            BASELINE(**args), run_selected(args, positions, selected)
        )
        torch.testing.assert_close(
            selected, prefix_oracle(args, [65], positions.tolist()), rtol=0, atol=0
        )

    def test_linear_tail_relative_positions(self):
        args, _ = make_inputs([512], heads=2, key_heads=1, kdim=64, vdim=64)
        requests = [236, 492]
        positions = torch.tensor([requests], device="cuda", dtype=torch.int64)
        selected = selected_buffer(args, positions)
        self.assert_outputs_equal(
            BASELINE(**args), run_selected(args, positions, selected)
        )
        torch.testing.assert_close(
            selected, prefix_oracle(args, [512], [requests]), rtol=0, atol=0
        )

    def test_interior_boundary_reuses_main_state(self):
        lengths = [192, 129]
        requests = [[128, 64, 192, 64], [64, 128, 129, 64]]
        args, _ = make_inputs(lengths, packed=True, heads=2, key_heads=1)
        positions = torch.tensor(requests, device="cuda", dtype=torch.int64)
        selected = selected_buffer(args, positions)
        self.assert_outputs_equal(
            BASELINE(**args), run_selected(args, positions, selected)
        )
        torch.testing.assert_close(
            selected, prefix_oracle(args, lengths, requests), rtol=0, atol=0
        )

    def test_dense_complete_boundary_requests(self):
        for slots in (16, 32, 64, 128):
            length = slots * 64
            with self.subTest(slots=slots, length=length):
                args, _ = make_inputs([length], heads=2, key_heads=1, kdim=64, vdim=64)
                requests = [boundary for boundary in range(64, length + 1, 64)]
                positions = torch.tensor([requests], device="cuda", dtype=torch.int64)
                selected = selected_buffer(args, positions)
                self.assert_outputs_equal(
                    BASELINE(**args), run_selected(args, positions, selected)
                )
                torch.testing.assert_close(
                    selected,
                    prefix_oracle(args, [length], [requests]),
                    rtol=0,
                    atol=0,
                )

    def test_state_precision_guard(self):
        args, _ = make_inputs([65])
        positions = torch.tensor([[1]], device="cuda", dtype=torch.int64)
        selected = selected_buffer(args, positions)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                run_selected(dict(args, state_dtype=dtype), positions, selected)
        with self.assertRaises(ValueError):
            run_selected(args, positions, args["initial_state"].unsqueeze(1))

    def test_public_pipeline(self):
        args, raw = make_inputs([129], heads=4, key_heads=2, kdim=64)
        v, beta, raw_g = raw
        q = torch.randn_like(args["k"])
        public = dict(
            q=q,
            k=args["k"],
            v=v,
            beta=beta,
            g=raw_g,
            initial_state=args["initial_state"],
            output_final_state=True,
        )
        positions = torch.tensor([[129, 1, 63, 64, 65, 0]], device="cuda")
        selected = selected_buffer(args, positions)
        expected = chunk_gated_delta_rule(**public)
        actual = chunk_gated_delta_rule(
            **public, state_positions=positions, selected_states=selected
        )
        self.assert_outputs_equal(expected, actual)
        for slot, prefix in enumerate(positions[0].tolist()):
            if prefix == 0:
                reference = args["initial_state"]
            else:
                truncated = {
                    name: value[:, :prefix].contiguous()
                    for name, value in public.items()
                    if name in ("q", "k", "v", "beta", "g")
                }
                reference = chunk_gated_delta_rule(
                    **truncated,
                    initial_state=args["initial_state"],
                    output_final_state=True,
                )[2]
            torch.testing.assert_close(selected[:, slot], reference, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            chunk_gated_delta_rule(
                **public,
                state_positions=positions,
                selected_states=selected.transpose(-1, -2),
            )
        with self.assertRaises(ValueError):
            chunk_gated_delta_rule(
                **public,
                state_positions=positions[:, :1].contiguous(),
                selected_states=args["initial_state"].unsqueeze(1),
            )

    def test_cuda_graph_dynamic_inputs_and_positions(self):
        args, _ = make_inputs([129], kdim=128)
        positions = torch.tensor([[0, 63, 64, 129]], device="cuda", dtype=torch.int64)
        selected = selected_buffer(args, positions)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run_selected(args, positions, selected)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run_selected(args, positions, selected)
        args["k"].mul_(0.875)
        args["w"].mul_(0.625)
        args["u"].mul_(0.75)
        args["g"].mul_(1.125)
        args["initial_state"].add_(0.01)
        positions.copy_(torch.tensor([[1, 65, 128, 1 << 40]], device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        captured_selected = selected.clone()
        eager_selected = selected_buffer(args, positions)
        eager = run_selected(args, positions, eager_selected)
        self.assert_outputs_equal(eager, captured)
        self.assert_outputs_equal(BASELINE(**args), captured)
        torch.testing.assert_close(captured_selected, eager_selected, rtol=0, atol=0)
        torch.testing.assert_close(
            captured_selected,
            prefix_oracle(args, [129], positions.tolist()),
            rtol=0,
            atol=0,
        )


def profile_kernel_time(
    invoke, trace_path, launches, warmup=30, iterations=100, launch_roles=None
):
    for _ in range(warmup):
        invoke()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(iterations):
            invoke()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text())["traceEvents"]
    kernels = sorted(
        (event for event in events if event.get("cat") == "kernel"),
        key=lambda event: event["ts"],
    )
    allowed_kernel_names = (
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        "_chunk_gated_delta_rule_selected_kernel",
    )
    names = sorted({event["name"] for event in kernels})
    if not kernels or any(
        not any(expected_name in name for expected_name in allowed_kernel_names)
        for name in names
    ):
        raise AssertionError(f"Unexpected kernel path: {names}")
    if len(kernels) != iterations * launches:
        raise AssertionError(
            f"Expected {iterations * launches} launches; got {len(kernels)}"
        )
    samples = [
        sum(event["dur"] for event in kernels[index : index + launches])
        for index in range(0, len(kernels), launches)
    ]
    ordered = sorted(samples)
    launch_roles = launch_roles or [f"launch_{index}" for index in range(launches)]
    if len(launch_roles) != launches:
        raise AssertionError("Every measured launch must have a documented role")
    components = {}
    for launch, role in enumerate(launch_roles):
        durations = [event["dur"] for event in kernels[launch::launches]]
        sorted_durations = sorted(durations)
        components[role] = {
            "kernel_sum_us_per_iter": statistics.mean(durations),
            "median_us": statistics.median(durations),
            "p90_us": sorted_durations[math.ceil(iterations * 0.9) - 1],
            "min_us": min(durations),
            "max_us": max(durations),
            "samples_us": durations,
            "kernel_names": sorted(
                {event["name"] for event in kernels[launch::launches]}
            ),
            "launch_index_in_iteration": launch,
        }
    return {
        "components": components,
        "component_assignment": "CUDA stream launch order within each iteration",
        "kernel_sum_us_per_iter": statistics.mean(samples),
        "median_us": statistics.median(samples),
        "p90_us": ordered[math.ceil(len(ordered) * 0.9) - 1],
        "min_us": min(samples),
        "max_us": max(samples),
        "kernel_names": names,
        "launches_per_iter": launches,
        "expected_names_and_counts_pass": True,
        "trace": str(trace_path),
        "samples_us": samples,
    }


def benchmark(output_path):
    import random

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    position_mode = os.environ.get("RTP_GDN_PERF_POSITION_MODE", "random")
    if position_mode not in ("random", "interior_boundary"):
        raise ValueError(f"Unknown GDN position mode: {position_mode}")
    # Qwen3.5-27B TP1: 16 key heads, 48 value heads, K=V=128. Include one
    # no-request control per length alongside the 24 requested snapshot cases.
    tokens_values = tuple(
        int(value)
        for value in os.environ.get(
            "RTP_GDN_PERF_TOKENS", "1024,2048,4096,8192,16384,32768"
        ).split(",")
        if value.strip()
    )
    slot_values = tuple(
        int(value)
        for value in os.environ.get("RTP_GDN_PERF_SLOTS", "0,1,2,4,8").split(",")
        if value.strip()
    )
    cases = [(tokens, slots) for tokens in tokens_values for slots in slot_values]
    base_seed = 20261016
    result = {
        "metric": "sum of actual CUDA kernel event durations; excludes CPU gaps",
        "model_shape": "Qwen3.5-27B TP1: B1 Hg16 H48 K128 V128",
        "model_config_sha256": "f8d190c5b89c1521220f935d2567a587d6e291ed69066a45a106560b05a2174c",
        "baseline": BASELINE_METADATA,
        "previous_snapshot_implementation": PREVIOUS_METADATA,
        "torch_compile": False,
        "warmup": 30,
        "iterations": 100,
        "compilation_excluded": True,
        "input_allocation_outside_measurement": True,
        "output_allocation": "wrapper allocates regular outputs; CUDA allocation is not kernel time",
        "position_seed_base": base_seed,
        "position_sampling": position_mode,
        "comparison": "identical prepared k/w/u/g/h0; original computes regular outputs; candidate also returns S states",
        "gpu": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cases": [],
    }
    for index, (tokens, slots) in enumerate(cases):
        args, _ = make_inputs([tokens], heads=48, key_heads=16, kdim=128, vdim=128)
        seed = base_seed + tokens
        if position_mode == "interior_boundary":
            boundaries = range(64, tokens, 64)
            if slots > len(boundaries):
                raise ValueError("Not enough interior boundaries for requested slots")
            requests = random.Random(seed).sample(boundaries, slots)
        else:
            sample_size = max(8, slots)
            requests = random.Random(seed).sample(range(1, tokens), sample_size)[:slots]
        positions = torch.tensor([requests], device="cuda", dtype=torch.int64)
        selected = selected_buffer(args, positions)
        torch.cuda.reset_peak_memory_stats()
        original = BASELINE(**args)
        candidate = run_selected(args, positions, selected)
        errors = {}
        for name, before, after in zip(
            ("h", "v_new", "final_state"), original, candidate
        ):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            errors[name] = {
                "exact_equal": True,
                "max_abs_error": 0.0,
                "normalized_rms_error": 0.0,
            }
        if slots:
            expected_selected = prefix_oracle(args, [tokens], [requests])
            torch.testing.assert_close(selected, expected_selected, rtol=0, atol=0)
            errors["selected_states_vs_original_prefix"] = {
                "exact_equal": True,
                "max_abs_error": 0.0,
                "normalized_rms_error": 0.0,
            }
        if PREVIOUS is not None:
            previous_outputs = PREVIOUS(
                **args, state_positions=positions, selected_states=selected
            )
            for before, previous in zip(original, previous_outputs):
                torch.testing.assert_close(previous, before, rtol=0, atol=0)
            if slots:
                torch.testing.assert_close(selected, expected_selected, rtol=0, atol=0)
            errors["previous_regular_and_selected_outputs_exact"] = True
            del previous_outputs
        peak_allocated_bytes = torch.cuda.max_memory_allocated()
        del original, candidate
        variants = (
            ("baseline", "previous", "candidate")
            if PREVIOUS is not None
            else ("baseline", "candidate")
        )
        offset = index % len(variants)
        order = variants[offset:] + variants[:offset]
        measurements = {}
        for variant in order:
            if variant == "baseline":
                invoke, launches, roles = lambda: BASELINE(**args), 1, ["original"]
            elif variant == "previous":
                invoke = lambda: PREVIOUS(
                    **args, state_positions=positions, selected_states=selected
                )
                launches = 2 if slots else 1
                roles = ["main", "snapshot"] if slots else ["main"]
            else:
                invoke = lambda: run_selected(args, positions, selected)
                launches = 2 if slots else 1
                roles = ["main", "snapshot"] if slots else ["main"]
            measurements[variant] = profile_kernel_time(
                invoke,
                path.with_name(f"{path.stem}_{index}_{variant}.json"),
                launches,
                launch_roles=roles,
            )
        baseline = measurements["baseline"]
        candidate = measurements["candidate"]
        row = {
            "B": 1,
            "T": tokens,
            "H": 48,
            "Hg": 16,
            "K": 128,
            "V": 128,
            "dtype": "bfloat16",
            "S": slots,
            "positions": requests,
            "position_seed": seed,
            "distribution": "random_unique_intermediate",
            "measurement_order": order,
            "selected_states_bytes": selected.numel() * selected.element_size(),
            "capture_workspace_bytes": 0,
            "peak_cuda_memory_allocated_during_correctness_bytes": peak_allocated_bytes,
            "correctness": errors,
            "baseline": baseline,
            "candidate": candidate,
            "speedup": baseline["kernel_sum_us_per_iter"]
            / candidate["kernel_sum_us_per_iter"],
            "additional_kernel_us": candidate["kernel_sum_us_per_iter"]
            - baseline["kernel_sum_us_per_iter"],
            "overhead_percent": (
                candidate["kernel_sum_us_per_iter"] / baseline["kernel_sum_us_per_iter"]
                - 1
            )
            * 100,
        }
        if PREVIOUS is not None:
            previous = measurements["previous"]
            ratio = (
                candidate["kernel_sum_us_per_iter"] / previous["kernel_sum_us_per_iter"]
            )
            row.update(
                previous=previous,
                candidate_vs_previous_ratio=ratio,
                candidate_vs_previous_speedup=1 / ratio,
                candidate_vs_previous_overhead_percent=(ratio - 1) * 100,
            )
        result["cases"].append(row)
        # Write in place to preserve the benchmark wrapper's reserved inode.
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps({key: row[key] for key in ("B", "T", "H", "S", "speedup")}),
            flush=True,
        )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelectedStatesTest)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    if not outcome.wasSuccessful():
        sys.exit(1)
    if os.environ.get("PERF_JSON"):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError("Kernel-time benchmark requires NVIDIA CUDA")
        benchmark(os.environ["PERF_JSON"])
