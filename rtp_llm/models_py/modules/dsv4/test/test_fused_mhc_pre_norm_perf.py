"""SM103 performance benchmark for DSV4 fused mHC pre + RMSNorm (+ UE8M0 quant)."""

from __future__ import annotations

import os
import time
import unittest
from typing import List

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.modules.dsv4.hc import (
    can_use_fused_mhc_pre_norm,
    fused_mhc_pre_norm,
    fused_mhc_pre_norm_quant,
)
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCUnit
from rtp_llm.models_py.modules.dsv4.test.dsv4_kernel_perf_utils import (
    device_payload,
    ensure_triton_cc,
    env_flag,
    git_commit,
    iters_for_m,
    measure_kernel,
    measurement_payload,
    parse_int_list,
    report_path_from_env,
    trace_dir_from_report,
    write_json_report,
)

ensure_triton_cc()

DEFAULT_MHC_M_SWEEP = [64, 128, 256, 512, 558, 1024, 4096, 12288, 25410, 65536]
FUSED_KERNEL_REGEX = r".*_mhc_pre_big_fuse_rmsnorm_kernel.*"


def _make_inputs(m: int, *, dim: int = 4096, hc: int = 4):
    torch.manual_seed(20261127 + m)
    fn = torch.randn(24, hc * dim, device="cuda", dtype=torch.float32) * 0.02
    base = torch.zeros(24, device="cuda", dtype=torch.float32)
    scale = torch.ones(3, device="cuda", dtype=torch.float32)
    unit = TileLangHCUnit(
        fn,
        base,
        scale,
        dim=dim,
        hc_mult=hc,
        hc_sinkhorn_iters=10,
        norm_eps=1e-6,
        hc_eps=1e-6,
    )
    w = (torch.rand(dim, device="cuda", dtype=torch.bfloat16) + 0.5).contiguous()
    norm = RMSNorm(w, eps=1e-6)
    x = (
        torch.randn(m, hc, dim, device="cuda", dtype=torch.bfloat16) * 0.25
    ).contiguous()
    return unit, norm, x


def _kernel_dram_bytes(m: int, *, dim: int, hc: int, quant_ue8m0: bool) -> int:
    # Read: residual [m, hc, dim] bf16 + weight [dim] bf16.
    read_bytes = m * hc * dim * 2 + dim * 2
    # Write: out_norm [m, dim] bf16 + post_mix [m, 4] f32 + comb_mix [m, 16] f32.
    write_bytes = m * dim * 2 + m * 20 * 4
    if quant_ue8m0:
        write_bytes += m * dim + m * ((dim // 128 + 3) // 4) * 4
    return read_bytes + write_bytes


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedMhcPreNormPerfTest(unittest.TestCase):
    @torch.inference_mode()
    def test_perf_report(self) -> None:
        major, _ = torch.cuda.get_device_capability()
        if major != 10:
            self.skipTest("requires SM100/SM103 for tf32_hc_prenorm_gemm")

        started = time.time()
        m_list = parse_int_list("DSV4_MHC_PRE_NORM_M_LIST", DEFAULT_MHC_M_SWEEP)
        report_path = report_path_from_env(
            "DSV4_MHC_PRE_NORM_JSON", "dsv4_fused_mhc_pre_norm_perf.json"
        )
        trace_dir = trace_dir_from_report(report_path, "DSV4_MHC_PRE_NORM_TRACE_DIR")
        profile_enabled = env_flag("DSV4_MHC_PRE_NORM_PROFILE", True)

        dim = 4096
        hc = 4
        rows: List[dict] = []

        print(
            "\n[_mhc_pre_big_fuse_rmsnorm_kernel SM103] "
            f"device={torch.cuda.get_device_name(0)}"
        )
        print(
            "  {:>7} {:>12} | {:>10} {:>10} {:>9} | {:>10}".format(
                "M",
                "quant_ue8m0",
                "e2e_us",
                "kernel_us",
                "BW_TB/s",
                "unfused_us",
            )
        )

        for m in m_list:
            unit, norm, x = _make_inputs(m, dim=dim, hc=hc)
            if not can_use_fused_mhc_pre_norm(
                x, unit, norm, quant_ue8m0=True, min_tokens=0
            ):
                self.skipTest("tf32_hc_prenorm_gemm unavailable")

            warmup = 10 if m <= 12288 else 5
            iters = max(iters_for_m(m), 10)

            for quant in (False, True):

                def _run_fused(q_flag=quant):
                    if q_flag:
                        return fused_mhc_pre_norm_quant(x, unit, norm)
                    return fused_mhc_pre_norm(x, unit, norm)

                def _run_unfused(q_flag=quant):
                    x_pre, post, comb = unit.pre(x)
                    x_norm = norm(x_pre)
                    if not q_flag:
                        return x_norm, post, comb
                    q, s = sgl_per_token_group_quant_fp8(
                        x_norm,
                        group_size=128,
                        eps=1e-4,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                        scale_ue8m0=True,
                    )
                    return x_norm, post, comb, (q, s)

                fused_m = measure_kernel(
                    _run_fused,
                    label=f"fused_mhc_pre_norm_M{m}_q{int(quant)}",
                    trace_dir=trace_dir,
                    kernel_regex=FUSED_KERNEL_REGEX,
                    warmup=warmup,
                    iters=iters,
                    profile_enabled=profile_enabled,
                )
                unfused_m = measure_kernel(
                    _run_unfused,
                    label=f"unfused_mhc_pre_norm_M{m}_q{int(quant)}",
                    trace_dir=trace_dir,
                    warmup=warmup,
                    iters=iters,
                    profile_enabled=False,
                )
                dram_bytes = _kernel_dram_bytes(m, dim=dim, hc=hc, quant_ue8m0=quant)
                kernel_us = (
                    fused_m.kernel_sum_us
                    if profile_enabled and fused_m.kernel_count > 0
                    else fused_m.event_us
                )
                bw_tbs = dram_bytes / (kernel_us * 1e6)
                print(
                    "  {:>7d} {:>12} | {:>10.2f} {:>10.2f} {:>9.2f} | {:>10.2f}".format(
                        m,
                        str(quant),
                        fused_m.event_us,
                        kernel_us,
                        bw_tbs,
                        unfused_m.event_us,
                    )
                )
                rows.append(
                    {
                        "M": m,
                        "dim": dim,
                        "hc": hc,
                        "quant_ue8m0": quant,
                        "dram_bytes": dram_bytes,
                        "bw_tbs": bw_tbs,
                        "unfused_us": unfused_m.event_us,
                        **measurement_payload(fused_m),
                    }
                )
            del unit, norm, x
            torch.cuda.empty_cache()

        payload = {
            "elapsed_sec": time.time() - started,
            "git_commit": git_commit(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            ),
            "device": device_payload(),
            "m_list": m_list,
            "results": rows,
        }
        write_json_report(report_path, payload)
        undeclared_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if undeclared_dir:
            write_json_report(
                os.path.join(undeclared_dir, os.path.basename(report_path)), payload
            )
        self.assertTrue(rows)


if __name__ == "__main__":
    unittest.main()
