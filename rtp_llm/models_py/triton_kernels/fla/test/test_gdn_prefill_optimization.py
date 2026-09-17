"""GDN prefill correctness first, then kernel/event measurements (no serving)."""

import collections
import json
import os
import statistics
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.triton_kernels.common.gated_rmsnorm_prefill import (
    gated_rmsnorm_prefill,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import layer_norm_fwd
from rtp_llm.models_py.triton_kernels.fla.exact_qk_norm import fused_l2norm_qk_exact
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.models_py.triton_kernels.fla.gdn_gating_prefill import gdn_gating_prefill
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd

OUT = Path(
    os.environ.get(
        "GDN_BENCH_OUTPUT",
        os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp/gdn-bench"),
    )
)
LENGTHS = [10007, 16384, 24601, 32768, 40009]


def measure(name, fn, shape, minimum_bytes=0):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(100):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000)
    path = OUT / (name + ".trace.json")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    kernels = [
        e
        for e in json.loads(path.read_text())["traceEvents"]
        if e.get("cat") == "kernel"
    ]
    grouped = collections.defaultdict(list)
    for e in kernels:
        grouped[e["name"]].append(e["dur"])
    table = [
        {
            "name": n,
            "launches_per_iter": len(v) / 5,
            "kernel_sum_us_per_iter": sum(v) / 5,
        }
        for n, v in grouped.items()
    ]
    expected = 2 if name.startswith("qk_") and name.endswith("baseline") else 1
    if len(kernels) != 5 * expected:
        raise AssertionError((name, expected, table))
    expected_prefix = (
        "l2norm_fwd_kernel"
        if name.startswith("qk_") and name.endswith("baseline")
        else (
            "_fused_qk_norm_kernel"
            if name.startswith("qk_")
            else (
                "_layer_norm_fwd_1pass_kernel"
                if name.startswith("gated_") and name.endswith("baseline")
                else (
                    "_gated_rmsnorm_rows"
                    if name.startswith("gated_")
                    else (
                        "fused_gdn_gating_kernel"
                        if name.endswith("baseline")
                        else "_gdn_gating_flat"
                    )
                )
            )
        )
    )
    if not all(expected_prefix in e["name"] for e in kernels):
        raise AssertionError((name, table))
    total = sum(e["dur"] for e in kernels) / 5
    result = {
        "name": name,
        "shape": shape,
        "warmup": 30,
        "measure": 100,
        "event_us": {
            "median": statistics.median(samples),
            "p90": sorted(samples)[89],
            "min": min(samples),
            "max": max(samples),
        },
        "kernel_sum_us_per_iter": total,
        "launches_per_iter": len(kernels) / 5,
        "kernels": table,
        "minimum_io_bytes": minimum_bytes,
        "effective_minimum_io_TB_s": minimum_bytes / total / 1e6 if total else 0,
        "io_estimate_fraction_of_7_7TB_s": (
            minimum_bytes / total / 1e6 / 7.7 if total else 0
        ),
        "timing_contract": "allocations of inputs excluded; output allocation included; warmed JIT; no torch.compile; repeated buffers; effective IO estimate is not a DRAM hardware counter",
    }
    print("BENCH", json.dumps(result), flush=True)
    return result


def errors(a, b):
    d = (a.float() - b.float()).abs()
    return {
        "max_abs": d.max().item(),
        "rms": d.square().mean().sqrt().item(),
        "mismatches": (a != b).sum().item(),
    }


class GDNPrefillOptimization(unittest.TestCase):
    @torch.inference_mode()
    def test_norm_correctness_and_performance(self):
        OUT.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(812)
        report = {
            "status": "RUNNING",
            "device": str(torch.cuda.get_device_properties(0)),
            "correctness": [],
            "benchmarks": [],
        }

        def save():
            (OUT / "norms.json").write_text(json.dumps(report, indent=2))

        # Test irregular heads, shared/per-head weights, optional bias, strided rows,
        # both activations, tiny values and non-power-of-two group widths.
        for d, h in [(64, 3), (96, 5), (128, 16), (128, 64), (256, 8)]:
            for shared, bias_on, activation in [
                (True, False, "silu"),
                (False, True, "sigmoid"),
            ]:
                base = torch.randn(37, h * d * 2, device="cuda", dtype=torch.bfloat16)
                x = base[:, : h * d]
                z = torch.randn_like(base)[:, : h * d]
                w = torch.randn(
                    d if shared else h * d, device="cuda", dtype=torch.bfloat16
                )
                bias = torch.randn_like(w) if bias_on else None
                ref = layer_norm_fwd(
                    x,
                    w,
                    bias,
                    1e-6,
                    z=z,
                    group_size=d,
                    is_rms_norm=True,
                    activation=activation,
                )[0]
                for tile in (4,):
                    actual = gated_rmsnorm_prefill(
                        x,
                        z,
                        w,
                        bias,
                        group_size=d,
                        activation=activation,
                        tile_rows=tile,
                    )
                    self.assertTrue(torch.equal(actual, ref))
                    report["correctness"].append(
                        {
                            "kind": "gated_norm_general",
                            "d": d,
                            "h": h,
                            "tile": tile,
                            **errors(actual, ref),
                        }
                    )
        for length in LENGTHS:
            q = torch.randn(1, length, 16, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn_like(q)
            qr, kr = l2norm_fwd(q), l2norm_fwd(k)
            variants = {"baseline": lambda: (l2norm_fwd(q), l2norm_fwd(k))}
            for tile in (4,):
                fn = lambda tile=tile: fused_l2norm_qk_exact(q, k, tile_rows=tile)
                qa, ka = fn()
                metric = {
                    "kind": "qk",
                    "length": length,
                    "tile": tile,
                    "q": errors(qa, qr),
                    "k": errors(ka, kr),
                }
                report["correctness"].append(metric)
                save()
                print("QK_CHECK", metric, flush=True)
                self.assertTrue(torch.equal(qr, qa))
                self.assertTrue(torch.equal(kr, ka))
                variants["fused" + str(tile)] = fn
            for tag, fn in (
                []
                if os.environ.get("GDN_CORRECTNESS_ONLY") == "1"
                else variants.items()
            ):
                report["benchmarks"].append(
                    measure(f"qk_{length}_{tag}", fn, list(q.shape), 4 * q.numel() * 2)
                )
            del q, k, qr, kr, qa, ka, variants, fn
            x = torch.randn(length, 64 * 128, device="cuda", dtype=torch.bfloat16)
            z = torch.randn_like(x)
            w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
            baseline = lambda: layer_norm_fwd(
                x, w, None, 1e-6, z=z, group_size=128, is_rms_norm=True
            )[0]
            ref = baseline()
            variants = {"baseline": baseline}
            for tile in (4,):
                fn = lambda tile=tile: gated_rmsnorm_prefill(x, z, w, tile_rows=tile)
                actual = fn()
                self.assertTrue(torch.equal(actual, ref))
                report["correctness"].append(
                    {
                        "kind": "gated_norm_long",
                        "length": length,
                        "tile": tile,
                        **errors(actual, ref),
                    }
                )
                variants["tiled" + str(tile)] = fn
            for tag, fn in (
                []
                if os.environ.get("GDN_CORRECTNESS_ONLY") == "1"
                else variants.items()
            ):
                report["benchmarks"].append(
                    measure(
                        f"gated_{length}_{tag}", fn, list(x.shape), 3 * x.numel() * 2
                    )
                )
            del x, z, w, ref, actual, baseline, variants, fn
            a = torch.randn(length, 64, device="cuda", dtype=torch.bfloat16)
            b = torch.randn_like(a)
            al = torch.randn(64, device="cuda")
            bias = torch.randn(64, device="cuda", dtype=torch.bfloat16)
            reference = fused_gdn_gating(al, a, b, bias)
            candidate = gdn_gating_prefill(al, a, b, bias)
            for left, right in zip(reference, candidate):
                self.assertTrue(torch.equal(left, right))
            for tag, fn in [
                ("baseline", lambda: fused_gdn_gating(al, a, b, bias)),
                ("vectorized", lambda: gdn_gating_prefill(al, a, b, bias)),
            ]:
                report["benchmarks"].append(
                    measure(f"gate_{length}_{tag}", fn, [length, 64], length * 64 * 10)
                )
            del a, b, al, bias, reference, candidate, fn
            save()
        report["status"] = "PASS"
        save()


if __name__ == "__main__":
    unittest.main()
