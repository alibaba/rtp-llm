"""Manual GB200 microbenchmark for HY4 sparse-Q padding and iHC pre-norm."""

import json
import os
import statistics

import torch
import torch.nn.functional as F

from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act_packed
from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.modules.hy_v4.ihc_triton import (
    maybe_fused_ihc_head,
    maybe_fused_ihc_post,
    maybe_fused_ihc_pre_normed_grouped,
)
from rtp_llm.models_py.triton_kernels.common.attn_output_gate import (
    sigmoid_mul_fp8_quant_fwd,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.pad_query_heads import (
    maybe_pad_query_heads,
)


WARMUP = 30
MEASURE = 100


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def _measure(function) -> dict[str, float]:
    for _ in range(WARMUP):
        result = function()
        del result
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE)]
    for start, end in zip(starts, ends):
        start.record()
        result = function()
        end.record()
        del result
    torch.cuda.synchronize()
    elapsed = [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]
    return {
        "median_us": statistics.median(elapsed),
        "p90_us": _percentile(elapsed, 0.9),
        "min_us": min(elapsed),
        "max_us": max(elapsed),
    }


def _benchmark_query_padding(device: torch.device) -> dict:
    query = torch.randn(13750, 64, 576, dtype=torch.bfloat16, device=device)

    def baseline():
        padded = query.new_zeros((13750, 128, 576))
        padded[:, :64].copy_(query)
        return padded

    def candidate():
        padded = maybe_pad_query_heads(query, 128)
        assert padded is not None
        return padded

    expected = baseline()
    actual = candidate()
    torch.testing.assert_close(actual[:, :64], expected[:, :64], rtol=0, atol=0)
    assert torch.count_nonzero(actual[:, 64:]).item() == 0
    del expected, actual
    return {
        "shape": [13750, 64, 576],
        "padded_heads": 128,
        "baseline_launches": 2,
        "candidate_launches": 1,
        "baseline": _measure(baseline),
        "candidate": _measure(candidate),
    }


def _benchmark_ihc(device: torch.device) -> dict:
    tokens, hc, hidden = 13750, 4, 6144
    channels = torch.randn(
        tokens, hc, hidden, dtype=torch.bfloat16, device=device
    )
    fn_weight = (
        torch.randn(2 * hc, hc * hidden, dtype=torch.float32, device=device)
        * (hc * hidden) ** -0.5
    )
    scale = torch.tensor([0.2, -0.1], dtype=torch.float32, device=device)
    base = torch.randn(2 * hc, dtype=torch.float32, device=device)
    norm_weight = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    kwargs = {
        "magnitude": 2.0,
        "hc_eps": 1e-6,
        "ihc_norm_eps": 1e-5,
        "read_norm_eps": 1e-5,
    }
    norm = RMSNorm(norm_weight, kwargs["read_norm_eps"])

    def baseline():
        reads = []
        post_gates = []
        for chunk in channels.split(4096, dim=0):
            flat = chunk.flatten(1).float()
            rstd = torch.rsqrt(
                flat.square().mean(dim=-1, keepdim=True)
                + kwargs["ihc_norm_eps"]
            )
            mixes = F.linear(flat, fn_weight) * rstd
            pre_raw, post_raw = mixes.split(hc, dim=-1)
            pre_gate = (
                torch.sigmoid(pre_raw * scale[0] + base[:hc])
                + kwargs["hc_eps"]
            )
            post_gate = (
                kwargs["magnitude"]
                * torch.sigmoid(post_raw * scale[1] + base[hc:])
                + kwargs["hc_eps"]
            )
            read = torch.sum(pre_gate.unsqueeze(-1) * chunk.float(), dim=1).to(
                channels.dtype
            )
            reads.append(read)
            post_gates.append(post_gate)
        return norm(torch.cat(reads, dim=0)), torch.cat(post_gates, dim=0)

    def candidate():
        result = maybe_fused_ihc_pre_normed_grouped(
            channels,
            fn_weight,
            scale,
            base,
            norm_weight,
            chunk_size=4096,
            **kwargs,
        )
        assert result is not None
        return result

    expected_read, expected_gate = baseline()
    actual_read, actual_gate = candidate()
    read_abs = (actual_read.float() - expected_read.float()).abs()
    gate_abs = (actual_gate - expected_gate).abs()
    read_tolerance = 2e-2 + 2e-2 * expected_read.float().abs()
    gate_tolerance = 5e-5 + 5e-4 * expected_gate.abs()
    correctness = {
        "read_exact": torch.equal(actual_read, expected_read),
        "read_diff_count": torch.count_nonzero(actual_read != expected_read).item(),
        "read_max_abs": read_abs.max().item(),
        "read_mean_abs": read_abs.mean().item(),
        "read_relative_l2": (
            torch.linalg.vector_norm(read_abs)
            / torch.linalg.vector_norm(expected_read.float()).clamp_min(1e-12)
        ).item(),
        "read_outside_tolerance": torch.count_nonzero(
            read_abs > read_tolerance
        ).item(),
        "gate_exact": torch.equal(actual_gate, expected_gate),
        "gate_diff_count": torch.count_nonzero(actual_gate != expected_gate).item(),
        "gate_max_abs": gate_abs.max().item(),
        "gate_mean_abs": gate_abs.mean().item(),
        "gate_relative_l2": (
            torch.linalg.vector_norm(gate_abs)
            / torch.linalg.vector_norm(expected_gate).clamp_min(1e-12)
        ).item(),
        "gate_outside_tolerance": torch.count_nonzero(
            gate_abs > gate_tolerance
        ).item(),
    }
    print("HY4_IHC_EAGER_DIFF", correctness)
    torch.testing.assert_close(actual_read, expected_read, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=5e-4, atol=5e-5)
    del expected_read, expected_gate, actual_read, actual_gate
    return {
        "shape": [tokens, hc, hidden],
        "baseline_launches_approx": 83,
        "candidate_launches": 4,
        "correctness": correctness,
        "baseline": _measure(baseline),
        "candidate": _measure(candidate),
    }


def _benchmark_ihc_post(device: torch.device) -> dict:
    tokens, hc, hidden = 13750, 4, 6144
    channels = torch.randn(
        tokens, hc, hidden, dtype=torch.bfloat16, device=device
    )
    block_output = torch.randn(
        tokens, hidden, dtype=torch.bfloat16, device=device
    )
    post_gate = torch.randn(tokens, hc, dtype=torch.float32, device=device)

    def baseline():
        return (
            channels.float()
            + post_gate.unsqueeze(-1) * block_output.float().unsqueeze(1)
        ).to(block_output.dtype)

    def candidate():
        result = maybe_fused_ihc_post(block_output, channels, post_gate)
        assert result is not None
        return result

    expected = baseline()
    actual = candidate()
    abs_diff = (actual.float() - expected.float()).abs()
    tolerance = 2e-2 + 2e-2 * expected.float().abs()
    correctness = {
        "exact": torch.equal(actual, expected),
        "diff_count": torch.count_nonzero(actual != expected).item(),
        "max_abs": abs_diff.max().item(),
        "mean_abs": abs_diff.mean().item(),
        "relative_l2": (
            torch.linalg.vector_norm(abs_diff)
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ).item(),
        "outside_tolerance": torch.count_nonzero(abs_diff > tolerance).item(),
    }
    print("HY4_IHC_POST_DIFF", correctness)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    del expected, actual, abs_diff, tolerance
    return {
        "shape": [tokens, hc, hidden],
        "baseline_launches_approx": 5,
        "candidate_launches": 1,
        "correctness": correctness,
        "baseline": _measure(baseline),
        "candidate": _measure(candidate),
    }


def _benchmark_ihc_head(device: torch.device) -> dict:
    tokens, hc, hidden = 13750, 4, 6144
    channels = torch.randn(
        tokens, hc, hidden, dtype=torch.bfloat16, device=device
    )
    fn_weight = (
        torch.randn(hc, hc * hidden, dtype=torch.float32, device=device)
        * (hc * hidden) ** -0.5
    )
    scale = torch.tensor([0.2], dtype=torch.float32, device=device)
    base = torch.randn(hc, dtype=torch.float32, device=device)
    hc_eps, norm_eps = 1e-6, 1e-5

    def baseline():
        outputs = []
        for chunk in channels.split(4096, dim=0):
            flat = chunk.flatten(1).float()
            rstd = torch.rsqrt(
                flat.square().mean(dim=-1, keepdim=True) + norm_eps
            )
            mixes = F.linear(flat, fn_weight) * rstd
            gates = torch.sigmoid(mixes * scale + base) + hc_eps
            output = torch.sum(gates.unsqueeze(-1) * chunk.float(), dim=1)
            outputs.append(output.to(channels.dtype))
        return torch.cat(outputs, dim=0)

    def candidate():
        outputs = []
        for chunk in channels.split(4096, dim=0):
            result = maybe_fused_ihc_head(
                chunk,
                fn_weight,
                scale,
                base,
                hc_eps=hc_eps,
                norm_eps=norm_eps,
            )
            assert result is not None
            outputs.append(result)
        return torch.cat(outputs, dim=0)

    expected = baseline()
    actual = candidate()
    abs_diff = (actual.float() - expected.float()).abs()
    tolerance = 2e-2 + 2e-2 * expected.float().abs()
    correctness = {
        "exact": torch.equal(actual, expected),
        "diff_count": torch.count_nonzero(actual != expected).item(),
        "max_abs": abs_diff.max().item(),
        "mean_abs": abs_diff.mean().item(),
        "relative_l2": (
            torch.linalg.vector_norm(abs_diff)
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ).item(),
        "outside_tolerance": torch.count_nonzero(abs_diff > tolerance).item(),
    }
    print("HY4_IHC_HEAD_DIFF", correctness)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    del expected, actual, abs_diff, tolerance
    return {
        "shape": [tokens, hc, hidden],
        "baseline_launches_approx": 61,
        "candidate_launches_approx": 17,
        "correctness": correctness,
        "baseline": _measure(baseline),
        "candidate": _measure(candidate),
    }


def _benchmark_gated_mla_quant(device: torch.device) -> dict:
    tokens, hidden = 13750, 16384
    attn = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    gate = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)

    def baseline():
        gated = (attn * torch.sigmoid(gate)).to(torch.bfloat16)
        return mxfp8_quant_act_packed(gated)

    def candidate():
        return sigmoid_mul_fp8_quant_fwd(
            attn,
            gate,
            quant_group_size=32,
            scale_ue8m0=True,
            round_scale_to_pow2=True,
            column_major_scales=True,
        )

    expected_fp8, expected_scale = baseline()
    actual_fp8, actual_scale = candidate()
    correctness = {
        "fp8_exact": torch.equal(
            actual_fp8.view(torch.uint8), expected_fp8.view(torch.uint8)
        ),
        "fp8_diff_count": torch.count_nonzero(
            actual_fp8.view(torch.uint8) != expected_fp8.view(torch.uint8)
        ).item(),
        "scale_exact": torch.equal(actual_scale, expected_scale),
        "scale_diff_count": torch.count_nonzero(
            actual_scale != expected_scale
        ).item(),
        "scale_shape_equal": actual_scale.shape == expected_scale.shape,
        "scale_stride_equal": actual_scale.stride() == expected_scale.stride(),
    }
    print("HY4_GATED_MLA_QUANT_DIFF", correctness)
    if (
        not correctness["fp8_exact"]
        or not correctness["scale_exact"]
        or not correctness["scale_shape_equal"]
        or not correctness["scale_stride_equal"]
    ):
        raise AssertionError(correctness)
    del expected_fp8, expected_scale, actual_fp8, actual_scale
    return {
        "shape": [tokens, hidden],
        "baseline_launches": 4,
        "candidate_launches": 1,
        "correctness": correctness,
        "baseline": _measure(baseline),
        "candidate": _measure(candidate),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] != 10:
        raise RuntimeError("this benchmark requires SM100/SM103 DeepGEMM")
    selected = os.environ.get("HY4_BENCH_CASE", "all")
    if selected not in (
        "all",
        "query_padding",
        "ihc_pre_normed",
        "ihc_post",
        "ihc_head",
        "gated_mla_quant",
    ):
        raise ValueError(f"invalid HY4_BENCH_CASE={selected!r}")
    with torch.no_grad():
        results = {
            "timing": "CUDA events; allocations included; warmup excluded",
            "warmup": WARMUP,
            "measure": MEASURE,
        }
        if selected in ("all", "query_padding"):
            results["query_padding"] = _benchmark_query_padding(device)
        if selected in ("all", "ihc_pre_normed"):
            results["ihc_pre_normed"] = _benchmark_ihc(device)
        if selected in ("all", "ihc_post"):
            results["ihc_post"] = _benchmark_ihc_post(device)
        if selected in ("all", "ihc_head"):
            results["ihc_head"] = _benchmark_ihc_head(device)
        if selected in ("all", "gated_mla_quant"):
            results["gated_mla_quant"] = _benchmark_gated_mla_quant(device)
    for case in (
        "query_padding",
        "ihc_pre_normed",
        "ihc_post",
        "ihc_head",
        "gated_mla_quant",
    ):
        if case not in results:
            continue
        baseline = results[case]["baseline"]["median_us"]
        candidate = results[case]["candidate"]["median_us"]
        results[case]["speedup"] = baseline / candidate
    print("HY4_FUSION_BENCH=" + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
