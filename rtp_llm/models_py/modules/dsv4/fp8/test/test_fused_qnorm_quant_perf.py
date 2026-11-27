"""SM103 performance benchmark for fused Q LoRA RMSNorm + UE8M0 FP8 quant."""

from __future__ import annotations

import os
import time
import unittest
from typing import List

import flashinfer
import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_fp8_quant_triton import (
    rmsnorm_fp8_quant_ue8m0,
)
from rtp_llm.models_py.modules.dsv4.fp8._fused_qnorm_quant_triton import (
    fused_rmsnorm_fp8_ue8m0_quant,
)
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

DEFAULT_QNORM_M_SWEEP = [1, 16, 1024, 12288, 25410]
DEFAULT_QNORM_D_SWEEP = [1024, 4096]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedQNormQuantPerfTest(unittest.TestCase):
    @torch.inference_mode()
    def test_perf_report(self) -> None:
        started = time.time()
        m_list = parse_int_list("DSV4_QNORM_QUANT_M_LIST", DEFAULT_QNORM_M_SWEEP)
        d_list = parse_int_list("DSV4_QNORM_QUANT_D_LIST", DEFAULT_QNORM_D_SWEEP)
        report_path = report_path_from_env(
            "DSV4_QNORM_QUANT_JSON", "dsv4_fused_qnorm_quant_perf.json"
        )
        trace_dir = trace_dir_from_report(report_path, "DSV4_QNORM_QUANT_TRACE_DIR")
        profile_enabled = env_flag("DSV4_QNORM_QUANT_PROFILE", True)
        eps = 1.0e-6

        rows: List[dict] = []
        print(
            "\n[fused_rmsnorm_fp8_ue8m0_quant SM103] "
            f"device={torch.cuda.get_device_name(0)}"
        )
        print(
            "  {:>5} {:>7} | {:>10} {:>8} | {:>10} {:>8} | {:>10} {:>8} {:>10} {:>8}".format(
                "D",
                "M",
                "unfused_us",
                "unfusTB",
                "old_us",
                "oldTB",
                "fp8_us",
                "fp8TB",
                "dual_us",
                "speedup",
            )
        )

        for d in d_list:
            w = (
                torch.rand((d,), device="cuda", dtype=torch.bfloat16) + 0.5
            ).contiguous()
            for m in m_list:
                x = (
                    torch.randn((m, d), device="cuda", dtype=torch.bfloat16) * 0.4
                ).contiguous()
                warmup = 20 if m <= 4096 else 8
                iters = max(iters_for_m(m), 20 if m <= 1024 else 10)

                def _unfused():
                    y = flashinfer.norm.rmsnorm(x, w, eps=eps)
                    return sgl_per_token_group_quant_fp8(
                        y,
                        group_size=128,
                        eps=1.0e-4,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                        scale_ue8m0=True,
                    )

                def _old_fused():
                    return rmsnorm_fp8_quant_ue8m0(x, w, eps=eps)

                def _new_fp8_only():
                    return fused_rmsnorm_fp8_ue8m0_quant(x, w, eps=eps, emit_norm=False)

                def _new_dual():
                    return fused_rmsnorm_fp8_ue8m0_quant(x, w, eps=eps, emit_norm=True)

                unfused_m = measure_kernel(
                    _unfused,
                    label=f"qnorm_unfused_D{d}_M{m}",
                    trace_dir=trace_dir,
                    warmup=warmup,
                    iters=iters,
                    profile_enabled=profile_enabled,
                )
                old_m = measure_kernel(
                    _old_fused,
                    label=f"qnorm_old_fused_D{d}_M{m}",
                    trace_dir=trace_dir,
                    warmup=warmup,
                    iters=iters,
                    profile_enabled=profile_enabled,
                )
                fp8_m = measure_kernel(
                    _new_fp8_only,
                    label=f"qnorm_new_fp8_D{d}_M{m}",
                    trace_dir=trace_dir,
                    kernel_regex=r".*_fused_qnorm_fp8_ue8m0_.*",
                    warmup=warmup,
                    iters=iters,
                    profile_enabled=profile_enabled,
                )
                dual_m = measure_kernel(
                    _new_dual,
                    label=f"qnorm_new_dual_D{d}_M{m}",
                    trace_dir=trace_dir,
                    kernel_regex=r".*_fused_qnorm_fp8_ue8m0_.*",
                    warmup=warmup,
                    iters=iters,
                    profile_enabled=profile_enabled,
                )

                scale_bytes = m * ((d // 128 + 3) // 4) * 4
                fp8_only_bytes = 3 * m * d + 2 * d + scale_bytes
                emit_norm_bytes = 5 * m * d + 2 * d + scale_bytes
                unfused_total_bytes = 7 * m * d + 2 * d + scale_bytes

                unfused_us = (
                    unfused_m.kernel_sum_us if profile_enabled else unfused_m.event_us
                )
                old_us = old_m.kernel_sum_us if profile_enabled else old_m.event_us
                fp8_us = fp8_m.kernel_sum_us if profile_enabled else fp8_m.event_us
                dual_us = dual_m.kernel_sum_us if profile_enabled else dual_m.event_us

                unfused_tb = unfused_total_bytes / (unfused_us * 1e6)
                old_tb = emit_norm_bytes / (old_us * 1e6)
                fp8_tb = fp8_only_bytes / (fp8_us * 1e6)
                speedup = unfused_us / fp8_us

                print(
                    "  {:>5d} {:>7d} | {:>10.2f} {:>8.2f} | {:>10.2f} {:>8.2f} | {:>10.2f} {:>8.2f} {:>10.2f} {:>7.2f}x".format(
                        d,
                        m,
                        unfused_us,
                        unfused_tb,
                        old_us,
                        old_tb,
                        fp8_us,
                        fp8_tb,
                        dual_us,
                        speedup,
                    )
                )
                rows.append(
                    {
                        "D": d,
                        "M": m,
                        "unfused_us": unfused_us,
                        "unfused_e2e_us": unfused_m.event_us,
                        "old_fused_us": old_us,
                        "new_fp8_only_us": fp8_us,
                        "new_fp8_only_e2e_us": fp8_m.event_us,
                        "new_dual_us": dual_us,
                        "fp8_only_tb_s": fp8_tb,
                        "speedup_vs_unfused_kernel": speedup,
                        "speedup_vs_unfused_e2e": unfused_m.event_us / fp8_m.event_us,
                        "unfused_measurement": measurement_payload(unfused_m),
                        "new_fp8_measurement": measurement_payload(fp8_m),
                    }
                )
                del x
                torch.cuda.empty_cache()

        payload = {
            "elapsed_sec": time.time() - started,
            "git_commit": git_commit(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
            ),
            "device": device_payload(),
            "m_list": m_list,
            "d_list": d_list,
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
