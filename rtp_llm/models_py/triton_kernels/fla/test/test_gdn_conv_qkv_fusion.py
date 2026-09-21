"""Exact prefill fusion comparisons against native RTP tensor boundaries."""

import json
import os
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.triton_kernels.common.prefill_fusion import prefill_fusion_scope

OUT = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp/gdn-fusion-tests"))
LENGTHS = [10007, 16384, 24601, 32768, 40009]

from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    causal_conv1d_fn,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.common.scatter_qkv import scatter_qkv
from rtp_llm.models_py.triton_kernels.fla.exact_qk_norm import fused_l2norm_qk_exact
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd


def error(a, b):
    af = a.float()
    bf = b.float()
    d = (af - bf).abs()
    return {
        "max_abs": d.max().item(),
        "rms": d.square().mean().sqrt().item(),
        "mismatches": (
            (a.view(torch.uint8) != b.contiguous().view(torch.uint8)).sum().item()
            if a.dtype == torch.float8_e4m3fn
            else (a != b).sum().item()
        ),
    }


def conv_case(
    lengths, hq=16, hv=64, prefix=0, width=4, bias=False, activation="silu", cache=True
):
    t = sum(lengths)
    d = (2 * hq + hv) * 128
    storage = torch.randn(t, d + 32, device="cuda", dtype=torch.bfloat16)
    x = storage[:, :d]
    w = torch.randn(d, width, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(d, device="cuda", dtype=torch.bfloat16) * 0.1 if bias else None
    cu = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
        device="cuda",
        dtype=torch.int32,
    )
    pre = torch.full((len(lengths),), prefix, device="cuda", dtype=torch.int32)
    slots = max((prefix + n + 2047) // 2048 for n in lengths) + 1
    table = torch.arange(
        1, len(lengths) * (slots + 1) + 1, device="cuda", dtype=torch.int32
    ).reshape(len(lengths), slots + 1)[:, :slots]
    state = torch.randn(
        len(lengths) * (slots + 1) + 1,
        width,
        d + 32,
        device="cuda",
        dtype=torch.bfloat16,
    )
    meta = prepare_causal_conv1d_metadata(cu, device=x.device)

    def run(fused, state):
        out = causal_conv1d_fn(
            x.T,
            w,
            b,
            state[:, : width - 1, :d].transpose(1, 2) if cache else None,
            cu,
            table if cache else None,
            pre,
            2048,
            activation=activation,
            metadata=meta,
            fused_qkv_heads=(hq, hv) if fused else None,
        )
        if fused:
            return out
        q, k, v = scatter_qkv(out.T, hq, hv, 128, 128)
        q, k = (
            fused_l2norm_qk_exact(q, k) if t >= 2048 else (l2norm_fwd(q), l2norm_fwd(k))
        )
        return q, k, v

    run.x = x
    run.w = w
    run.cu = cu
    run.pre = pre
    run.table = table
    run.meta = meta
    return run, state


class ConvQkvFusionTest(unittest.TestCase):
    @torch.inference_mode()
    def test_correctness(self):
        if os.getenv("IO_MODE") in ("perf", "ncu"):
            self.skipTest("Separate performance run")
        torch.manual_seed(918)
        OUT.mkdir(parents=True, exist_ok=True)
        report = {"status": "RUNNING", "conv": [], "norm_quant": []}

        def save():
            (OUT / "correctness.json").write_text(json.dumps(report, indent=2))

        cases = [
            ([n], 16, 64, 0, 4, False, "silu", True)
            for n in ([37, 2048, 2053] + LENGTHS)
        ]
        cases += [
            ([37, 4096, 2053], 4, 8, p, w, b, a, c)
            for p, w, b, a, c in [
                (0, 2, True, None, False),
                (37, 3, True, "silu", True),
                (2048, 4, False, "silu", True),
            ]
        ]
        for case in cases:
            run, state = conv_case(*case)
            refstate = state.clone()
            gotstate = state.clone()
            ref = run(False, refstate)
            got = run(True, gotstate)
            record = {
                "case": case,
                "qkv": [error(a, b) for a, b in zip(got, ref)],
                "cache": error(gotstate, refstate),
            }
            report["conv"].append(record)
            save()
            print("CONV", record, flush=True)
            for a, b in zip(got, ref):
                self.assertTrue(torch.equal(a, b), record)
            self.assertTrue(torch.equal(gotstate, refstate), record)
            del run, state, refstate, gotstate, ref, got
        report["status"] = "PASS"
        save()

    @torch.inference_mode()
    def test_dispatch(self):
        if os.getenv("IO_MODE") in ("perf", "ncu"):
            self.skipTest("Separate performance run")
        from types import SimpleNamespace
        from unittest.mock import patch

        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNetPrefill,
        )

        OUT.mkdir(parents=True, exist_ok=True)
        results = []
        torch.manual_seed(919)
        for lengths, prefix in [([2053], 0), ([37, 4096, 2053], 37), ([24601], 2048)]:
            hq, hv = 16, 64
            t = sum(lengths)
            run, cs = conv_case(lengths, hq, hv, prefix)
            a = torch.randn(t, hv, device="cuda", dtype=torch.bfloat16)
            b = torch.randn_like(a)
            args = SimpleNamespace(
                input_lengths=torch.tensor(lengths),
                cu_seqlens_device=run.cu,
                prefix_lengths_device=run.pre,
                kv_cache_kernel_block_id_device=run.table,
            )
            cache = (
                torch.randn(
                    cs.shape[0], hv + 1, 128, 128, device="cuda", dtype=torch.float32
                )
                * 0.001
            )
            obj = SimpleNamespace(
                alog=torch.randn(hv, device="cuda", dtype=torch.bfloat16),
                dt_bias=torch.randn(hv, device="cuda", dtype=torch.bfloat16),
                local_num_v_heads=hv,
                local_num_k_heads=hq,
                head_k_dim=128,
                head_v_dim=128,
                ssm_state_dtype=torch.float32,
                conv_weights=run.w,
                _get_ssm_states=lambda c: c[:, :hv],
                _get_conv_states=lambda _: cs[:, :3, : (2 * hq + hv) * 128],
            )
            with prefill_fusion_scope(True), patch.dict(
                os.environ, {"RTP_QWEN35_FUSED_CONV_QKV_NORM": "0"}
            ):
                mixed = Qwen3NextGatedDeltaNetPrefill._conv1d(
                    obj, run.x, None, 2048, args, metadata=run.meta
                )
            with prefill_fusion_scope(True), patch.dict(
                os.environ, {"RTP_QWEN35_FUSED_CONV_QKV_NORM": "1"}
            ):
                qkv = Qwen3NextGatedDeltaNetPrefill._conv1d(
                    obj, run.x, None, 2048, args, metadata=run.meta
                )
                self.assertIsInstance(qkv, tuple)
            c0 = cache.clone()
            c1 = cache.clone()
            with prefill_fusion_scope(True), patch.dict(
                os.environ,
                {
                    "RTP_QWEN35_GDN_PREFILL_BACKEND": "native",
                },
            ):
                ref = Qwen3NextGatedDeltaNetPrefill._fla(
                    obj, mixed, b, a, c0, 2048, args
                )
                got = Qwen3NextGatedDeltaNetPrefill._fla(
                    obj, run.x, b, a, c1, 2048, args, normalized_qkv=qkv
                )
            record = {
                "lengths": lengths,
                "prefix": prefix,
                "output": error(got, ref),
                "ssm": error(c1, c0),
            }
            results.append(record)
            (OUT / "dispatch.json").write_text(json.dumps(results, indent=2))
            print("DISPATCH", record, flush=True)
            self.assertTrue(torch.equal(ref, got))
            self.assertTrue(torch.equal(c0, c1))
            del run, cs, a, b, args, cache, obj, mixed, qkv, c0, c1, ref, got


if __name__ == "__main__":
    unittest.main()
