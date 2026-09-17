"""Validate FlashInfer outputs AND sparse V-first states before serving dispatch."""

import json
import os
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.triton_kernels.fla.block import store_ssm_state_to_block_map
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.flashinfer_prefill import (
    flashinfer_gdn_prefill,
    store_flashinfer_ssm_state,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd


class FlashInferPrefillAdapter(unittest.TestCase):
    @torch.inference_mode()
    def test_output_final_and_checkpoint_states(self):
        out = Path(
            os.environ.get(
                "GDN_BENCH_OUTPUT", os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", ".")
            )
        )
        out.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "RUNNING",
            "criteria": {"relative_rms": 0.01, "atol": 0.01, "rtol": 0.01},
            "cases": [],
        }

        def save():
            (out / "flashinfer.json").write_text(json.dumps(report, indent=2))

        def compare(actual, ref):
            self.assertEqual(actual.shape, ref.shape)
            self.assertTrue(torch.isfinite(actual).all())
            diff = actual.float() - ref.float()
            rms = diff.square().mean().sqrt().item()
            rr = rms / max(ref.float().square().mean().sqrt().item(), 1e-12)
            bad = (diff.abs() > (0.01 + 0.01 * ref.float().abs())).sum().item()
            return {
                "max_abs": diff.abs().max().item(),
                "rms": rms,
                "relative_rms": rr,
                "out_of_tolerance": bad,
                "ok": rr <= 0.01 and bad == 0,
            }

        torch.manual_seed(731)
        for lengths in (
            [63],
            [64],
            [65],
            [10007],
            [16384],
            [24601],
            [32768],
            [40009],
            [37, 4096, 20468],
        ):
            t = sum(lengths)
            hq, hv = (2, 4) if t < 100 else (16, 64)
            for nonzero in (False, True):
                q = (
                    torch.randn(1, t, hq, 128, device="cuda", dtype=torch.bfloat16)
                    * 0.1
                )
                k = torch.randn_like(q)
                v = (
                    torch.randn(1, t, hv, 128, device="cuda", dtype=torch.bfloat16)
                    * 0.1
                )
                a = torch.randn(t, hv, device="cuda", dtype=torch.bfloat16)
                b = torch.randn_like(a)
                g, beta = fused_gdn_gating(
                    torch.randn(hv, device="cuda"),
                    a,
                    b,
                    torch.randn(hv, device="cuda", dtype=torch.bfloat16),
                )
                initial = torch.zeros(len(lengths), hv, 128, 128, device="cuda")
                if nonzero:
                    initial.normal_(std=0.01)
                saved = initial.clone()
                cu = torch.tensor(
                    [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                    device="cuda",
                    dtype=torch.int32,
                )
                expected, h, final = chunk_gated_delta_rule(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state=initial,
                    output_final_state=True,
                    cu_seqlens=cu,
                    use_qk_l2norm_in_kernel=True,
                )
                actual, afinal, cp, starts = flashinfer_gdn_prefill(
                    q, k, v, g, beta, cu, initial
                )
                checks = {
                    "output": compare(actual, expected),
                    "final_state": compare(afinal, final),
                }
                offset = 0
                ci = 0
                checkpoint_refs = []
                for i, l in enumerate(lengths):
                    for end in range(2048, l + 1, 2048):
                        checkpoint_refs.append(
                            final[i] if end == l else h[0, offset + end // 64]
                        )
                    offset += (l + 63) // 64
                if checkpoint_refs:
                    checks["checkpoints"] = compare(
                        cp[: len(checkpoint_refs)], torch.stack(checkpoint_refs)
                    )
                for prefix in (0, 2048, 37):
                    slots = max((prefix + l + 2047) // 2048 for l in lengths) + 1
                    mapping = (
                        torch.arange(
                            1,
                            1 + len(lengths) * slots,
                            device="cuda",
                            dtype=torch.int32,
                        )
                        .flip(0)
                        .reshape(len(lengths), slots)
                        .contiguous()
                    )
                    prefixes = torch.full(
                        (len(lengths),), prefix, device="cuda", dtype=torch.int32
                    )
                    for dtype in (torch.float32, torch.bfloat16):
                        cache = torch.full(
                            (1 + len(lengths) * slots, hv, 128, 128),
                            123.0,
                            device="cuda",
                            dtype=dtype,
                        )
                        expected_cache = cache.clone()
                        store_ssm_state_to_block_map(
                            h,
                            final,
                            prefixes,
                            cu,
                            mapping,
                            expected_cache,
                            2048,
                            chunk_size=64,
                        )
                        store_flashinfer_ssm_state(
                            cp, starts, afinal, prefixes, cu, mapping, cache, 2048, t
                        )
                        touched = expected_cache != 123.0
                        self.assertTrue(
                            torch.equal(cache[~touched], expected_cache[~touched])
                        )
                        checks[f"cache_prefix{prefix}_{dtype}"] = compare(
                            cache[touched], expected_cache[touched]
                        )
                        del cache, expected_cache, touched
                self.assertTrue(torch.equal(saved, initial))
                if t < 100:
                    qn, kn = l2norm_fwd(q)[0].float(), l2norm_fwd(k)[0].float()
                    qn = qn.repeat_interleave(hv // hq, dim=1)
                    kn = kn.repeat_interleave(hv // hq, dim=1)
                    state = initial[0].clone()
                    oo = []
                    for i in range(t):
                        state *= g[0, i].exp()[:, None, None]
                        delta = (
                            v[0, i].float() - torch.einsum("hvk,hk->hv", state, kn[i])
                        ) * beta[0, i].float()[:, None]
                        state += delta[:, :, None] * kn[i, :, None, :]
                        oo.append(
                            torch.einsum("hvk,hk->hv", state, qn[i]) * (128**-0.5)
                        )
                    checks["fp32_recurrence_output"] = compare(
                        actual[0], torch.stack(oo)
                    )
                    checks["fp32_recurrence_state"] = compare(afinal[0], state)
                case = {
                    "lengths": lengths,
                    "nonzero_initial": nonzero,
                    "checks": checks,
                    "all_ok": all(c["ok"] for c in checks.values()),
                }
                report["cases"].append(case)
                save()
                print("FI_CHECK", json.dumps(case), flush=True)
                del (
                    q,
                    k,
                    v,
                    a,
                    b,
                    g,
                    beta,
                    initial,
                    saved,
                    expected,
                    h,
                    final,
                    actual,
                    afinal,
                    cp,
                    starts,
                    checkpoint_refs,
                )
        report["status"] = (
            "PASS" if all(c["all_ok"] for c in report["cases"]) else "FAIL"
        )
        save()
        self.assertEqual(report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
