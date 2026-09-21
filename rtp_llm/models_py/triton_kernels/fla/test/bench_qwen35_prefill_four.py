"""Three alternating groups of five prefill operator A/B measurements.

Run with Bazel's GPU lock; PREFILL_FOUR_BENCH_OUTPUT selects the JSON output.
"""

import json
import os
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.triton_kernels.common.prefill_fusion import (
    MROPE,
    prefill_fusion_scope,
)
from rtp_llm.models_py.triton_kernels.fla.flashinfer_prefill import (
    prepare_flashinfer_prefill_metadata,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating_prefill import gdn_gating_prefill
from rtp_llm.models_py.triton_kernels.fla.test.test_qwen35_prefill_four import make_case
from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.sigmoid_mul_fp8_quant import (
    sigmoid_mul_fp8_quant,
)
from rtp_llm.ops.fused_rope_kvcache_op import FusedRopeKVCachePrefillOpQOut


def compare(name, tokens, baseline, fused, reset=lambda: None):
    records = []
    funcs = {"baseline": baseline, "fused": fused}
    for fn in funcs.values():
        for _ in range(5):
            reset()
            fn()
    torch.cuda.synchronize()
    for group in range(3):
        for iteration in range(5):
            for variant in (
                ("baseline", "fused") if group % 2 == 0 else ("fused", "baseline")
            ):
                reset()
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                t0 = time.perf_counter()
                start.record()
                result = funcs[variant]()
                end.record()
                end.synchronize()
                records.append(
                    dict(
                        group=group,
                        iteration=iteration,
                        variant=variant,
                        gpu_us=start.elapsed_time(end) * 1000,
                        wall_us=(time.perf_counter() - t0) * 1e6,
                    )
                )
                del result
    medians = {
        v: statistics.median(r["gpu_us"] for r in records if r["variant"] == v)
        for v in funcs
    }
    report = dict(
        operator=name,
        tokens=tokens,
        medians_us=medians,
        speedup=medians["baseline"] / medians["fused"],
        samples=records,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)
    return report


def main():
    torch.manual_seed(20260921)
    reports = []
    for n in (24601, 49202):
        x = torch.randn(n, 8192, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn_like(x)

        def base_quant():
            return sgl_per_token_group_quant_fp8(
                x * gate.sigmoid(),
                128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )

        reports.append(
            compare(
                "sigmoid_mul_fp8", n, base_quant, lambda: sigmoid_mul_fp8_quant(x, gate)
            )
        )
        del x, gate
        lengths = [24601] * (n // 24601)
        cfg, params, qkv, cache = make_case(
            lengths, [0] * len(lengths), True, True, qh=32, kh=2
        )
        cfg.rope_config.base = 10000000
        original = qkv.clone()
        op = FusedRopeKVCachePrefillOpQOut(cfg)

        def mrope(flag):
            with patch.dict(os.environ, {MROPE: flag}), prefill_fusion_scope(True):
                return op.forward(qkv, cache, params)

        reports.append(
            compare(
                "mrope_pack_fp8_cache",
                n,
                lambda: mrope("0"),
                lambda: mrope("1"),
                lambda: qkv.copy_(original),
            )
        )
        del qkv, original, cache
        a, b = torch.randn(n, 128, device="cuda", dtype=torch.bfloat16).split(64, -1)
        al, dt = torch.zeros(64, device="cuda", dtype=torch.bfloat16), torch.zeros(
            64, device="cuda", dtype=torch.bfloat16
        )

        def base_gate():
            g, beta = gdn_gating_prefill(al, a, b, dt)
            return g.exp(), beta.float()

        reports.append(
            compare(
                "flashinfer_gates",
                n,
                base_gate,
                lambda: gdn_gating_prefill(al, a, b, dt, flashinfer=True),
            )
        )
        cu = torch.tensor(
            [0] + [24601 * i for i in range(1, len(lengths) + 1)],
            device="cuda",
            dtype=torch.int32,
        )

        def base_metadata():
            outputs = []
            for _ in range(45):
                cu64 = cu.to(torch.int64).contiguous()
                counts = (cu64[1:] - cu64[:-1]) // 2048
                starts = torch.cat(
                    (
                        torch.zeros(1, device=cu.device, dtype=torch.int64),
                        counts.cumsum(0),
                    )
                )
                outputs.append((cu64.to(torch.int32), starts.to(torch.int32)))
            return outputs

        reports.append(
            compare(
                "checkpoint_metadata_45_layers",
                n,
                base_metadata,
                lambda: prepare_flashinfer_prefill_metadata(cu, n),
            )
        )
    output = Path(os.environ["PREFILL_FOUR_BENCH_OUTPUT"])
    output.write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(),
                "reports": reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
