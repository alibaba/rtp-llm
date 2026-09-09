"""Compare producer GPU time with the existing producer plus group128 quantizer."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_fp8_kernels import (
    gather_fp8_prefix,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_prefix_fp8_producer import (
    Fp8MlaPrefixGather,
)
from rtp_llm.models_py.modules.kimi_k3.fp8_producers import KdaOutputNorm
from rtp_llm.models_py.triton_kernels.kimi_kda.attn_res import kimi_k3_attn_res
from rtp_llm.models_py.triton_kernels.kimi_kda.attn_res_fp8 import kimi_k3_attn_res_fp8
from rtp_llm.models_py.triton_kernels.kimi_kda.fp8_producers import (
    kda_output_fp8,
    rmsnorm_fp8,
    sigmoid_gate_fp8,
)


def quant(x):
    return sgl_per_token_group_quant_fp8(
        x.contiguous(),
        128,
        eps=1.0e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )


def cases(m):
    for k, name, retain in ((1536, "mla_q_norm", False), (512, "mla_kv_norm", True)):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(k, device="cuda", dtype=torch.bfloat16)
        norm = RMSNorm(w, 1.0e-6)
        yield name, lambda: quant(norm(x)), lambda: rmsnorm_fp8(
            x, w, 1.0e-6, retain_bf16=retain
        )
    x, g = [torch.randn(m, 1536, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    yield "mla_gate", lambda: quant(x * torch.sigmoid(g)), lambda: sigmoid_gate_fp8(
        x, g
    )
    x, g = [
        torch.randn(1, m, 12, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    for mode in ("prefill", "decode"):
        norm = KdaOutputNorm(w, 1.0e-5)
        yield "kda_" + mode, lambda: quant(
            norm(x, g, mode).reshape(m, 1536)
        ), lambda: kda_output_fp8(x, g, w, 1.0e-5, mode=mode)
    x = torch.randn(m, 7168, device="cuda", dtype=torch.bfloat16)
    bank = torch.randn(m, 2, 7168, device="cuda", dtype=torch.bfloat16)
    w, p, ow = [
        torch.randn(7168, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    kw = dict(output_norm_weight=ow, output_norm_eps=1.0e-5, num_blocks=2)
    yield "attn_res", lambda: quant(
        kimi_k3_attn_res(x, bank, w, p, 1.0e-5, **kw)
    ), lambda: kimi_k3_attn_res_fp8(x, bank, w, p, 1.0e-5, **kw)
    pages_count = (m + 127) // 128
    cache = torch.randn(pages_count, 128, 576, device="cuda").to(torch.float8_e4m3fn)
    pages = torch.arange(pages_count, device="cuda", dtype=torch.int32)
    info = torch.tensor([[0, m, 0, pages_count]], device="cuda", dtype=torch.int32)
    qi = torch.zeros(2, device="cuda", dtype=torch.int32)
    c = torch.empty(0, 512, device="cuda", dtype=torch.bfloat16)
    r = torch.empty(0, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, 512, device="cuda", dtype=torch.bfloat16)
    rope = torch.empty(m, 64, device="cuda", dtype=torch.bfloat16)

    def baseline():
        gather_fp8_prefix(out, rope, c, r, cache, pages, info, qi, 128, scale=1.0)
        return quant(out)

    yield "prefix_gather", baseline, lambda: Fp8MlaPrefixGather()(
        out, rope, c, r, cache, pages, info, qi, 128
    )


def measure(fn, repeats):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            result = fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    warmed_ms = 0.0
    warm_replays = 0
    # Warm every measured group with the same graph for at least 200 ms and
    # at least ten replays, after allocation and compilation have completed.
    while warmed_ms < 200 or warm_replays < 10:
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        warmed_ms += start.elapsed_time(end)
        warm_replays += 10
    samples = []
    for _ in range(10):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    return dict(
        median_us=statistics.median(samples),
        samples_us=samples,
        warm_replays=warm_replays,
        warm_gpu_ms=warmed_ms,
        repeats=repeats,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,4,4096,65536")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(904)
    results = []
    for m in map(int, args.rows.split(",")):
        for name, baseline, fused in cases(m):
            reference = baseline()
            actual = fused()
            torch.testing.assert_close(
                actual.values.float(), reference[0].float(), atol=0, rtol=0
            )
            torch.testing.assert_close(actual.scales, reference[1], atol=0, rtol=0)
            old = measure(baseline, 32 if m <= 4 else 3)
            new = measure(fused, 32 if m <= 4 else 3)
            item = dict(
                producer=name,
                rows=m,
                baseline=old,
                fused=new,
                speedup=old["median_us"] / new["median_us"],
            )
            results.append(item)
            print(json.dumps(item), flush=True)
            Path(args.output).write_text(
                json.dumps(
                    dict(
                        timing="CUDA graph GPU time; excludes Python dispatch",
                        device=torch.cuda.get_device_name(),
                        torch=torch.__version__,
                        results=results,
                    ),
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
