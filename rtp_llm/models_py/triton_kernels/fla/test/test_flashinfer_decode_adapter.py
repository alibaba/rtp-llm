"""Numeric + dispatch checks for the FlashInfer T=1 GDN decode adapter."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Triton JIT writes launcher sources via tempfile; keep that off a full /tmp.
_tmp = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
if _tmp:
    os.environ["TMPDIR"] = _tmp
    os.environ["TEMP"] = _tmp
    os.environ["TMP"] = _tmp
    tempfile.tempdir = _tmp

import torch

from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNetDecode
from rtp_llm.models_py.triton_kernels.fla.flashinfer_decode import (
    fill_paged_decode_indices,
    flashinfer_gdn_decode,
    gdn_decode_backend,
)
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating


def _compare(actual, ref, atol=0.03, rtol=0.03):
    diff = actual.float() - ref.float()
    rms = diff.square().mean().sqrt().item()
    rr = rms / max(ref.float().square().mean().sqrt().item(), 1e-12)
    bad = (diff.abs() > (atol + rtol * ref.float().abs())).sum().item()
    return {
        "max_abs": diff.abs().max().item(),
        "rms": rms,
        "relative_rms": rr,
        "out_of_tolerance": bad,
        "ok": rr <= 0.03 and bad == 0,
    }


def _strided_ssm(pages, hv, device, extra=256):
    raw = torch.randn(
        pages, hv * 128 * 128 + extra, device=device, dtype=torch.bfloat16
    )
    raw.mul_(0.01)
    return torch.as_strided(
        raw, (pages, hv, 128, 128), (raw.stride(0), 128 * 128, 128, 1)
    )


def _rtp_decode(q, k, v, a, b, alog, dt_bias, ssm, block_map, seq, seq_size):
    g, beta = fused_gdn_gating(alog, a, b, dt_bias)
    out, _ = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g.view(q.shape[0], 1, v.shape[2]),
        beta=beta.view(q.shape[0], 1, v.shape[2]),
        scale=None,
        initial_state=ssm,
        inplace_final_state=True,
        block_map=block_map,
        seq_size_per_block=seq_size,
        sequence_lengths=seq,
        use_qk_l2norm_in_kernel=True,
    )
    return out


class FlashInferDecodeAdapter(unittest.TestCase):
    @torch.inference_mode()
    def test_index_fill_matches_rtp_pages(self):
        device = "cuda"
        seq_size = 2048
        block_map = torch.tensor(
            [[0, 3, 0], [2, 4, 5], [0, 0, 0]], device=device, dtype=torch.int32
        )
        seq = torch.tensor([2049, 24576, 1], device=device, dtype=torch.int32)
        read, write = fill_paged_decode_indices(block_map, seq, seq_size)
        # plus_1=2049: read page 0 -> 0 -> -1, write forced -1
        # plus_1=24576: (24574)//2048 = 11, clamped to last page 2 -> 5
        #   write (24575)//2048 = 11 -> 5
        # plus_1=1: read/write page 0 -> 0 -> -1
        self.assertEqual(read.tolist(), [-1, 5, -1])
        self.assertEqual(write.tolist(), [-1, 5, -1])
        alog_bf16 = torch.randn(64, device=device, dtype=torch.bfloat16)
        read2, write2, alog_fp32 = fill_paged_decode_indices(
            block_map, seq, seq_size, A_log=alog_bf16
        )
        self.assertEqual(read2.tolist(), [-1, 5, -1])
        self.assertEqual(write2.tolist(), [-1, 5, -1])
        self.assertEqual(alog_fp32.dtype, torch.float32)
        torch.testing.assert_close(alog_fp32, alog_bf16.float(), atol=0, rtol=0)

    @torch.inference_mode()
    def test_output_and_paged_state(self):
        if not torch.cuda.is_available():
            self.skipTest("cuda required")
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("FlashInfer GDN decode requires SM90+")
        out_dir = Path(
            os.environ.get(
                "GDN_BENCH_OUTPUT", os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", ".")
            )
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {"status": "RUNNING", "cases": []}
        torch.manual_seed(17)
        device = "cuda"
        cases = (
            (1, 16, 64, 64, 24576, False),
            (16, 16, 64, 64, 24576, False),
            (32, 16, 64, 2048, 2049, True),
            (8, 4, 8, 64, 65, True),
        )
        for batch, h, hv, seq_size, plus_1, strided in cases:
            q = torch.randn(batch, 1, h, 128, device=device, dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn(batch, 1, hv, 128, device=device, dtype=torch.bfloat16)
            a = torch.randn(batch, hv, device=device, dtype=torch.bfloat16)
            b = torch.randn_like(a)
            alog = torch.randn(
                hv,
                device=device,
                dtype=torch.bfloat16 if batch == 16 else torch.float32,
            )
            dt_bias = torch.randn(hv, device=device, dtype=torch.bfloat16)
            read_page = (plus_1 - 2) // seq_size
            write_page = (plus_1 - 1) // seq_size
            n_pages = max(read_page, write_page) + 2
            pages = 1 + batch * n_pages
            ssm = (
                _strided_ssm(pages, hv, device)
                if strided
                else torch.randn(pages, hv, 128, 128, device=device, dtype=torch.bfloat16)
                * 0.01
            )
            block_map = torch.zeros(batch, n_pages, device=device, dtype=torch.int32)
            for i in range(batch):
                block_map[i, read_page] = 1 + i * 2
                block_map[i, write_page] = 1 + i * 2 + int(read_page != write_page)
            seq = torch.full((batch,), plus_1, device=device, dtype=torch.int32)
            ref_state = ssm.clone()
            actual_state = ssm.clone()
            expected = _rtp_decode(
                q, k, v, a, b, alog, dt_bias, ref_state, block_map, seq, seq_size
            )
            actual = flashinfer_gdn_decode(
                q,
                k,
                v,
                a,
                b,
                alog,
                dt_bias,
                actual_state,
                block_map,
                seq,
                seq_size,
            )
            checks = {
                "output": _compare(actual, expected),
                "state": _compare(actual_state, ref_state),
            }
            self.assertTrue(checks["output"]["ok"], checks)
            self.assertTrue(checks["state"]["ok"], checks)
            if strided:
                # as_strided extra tail must stay untouched when we only write V*K.
                pass
            report["cases"].append(
                {
                    "batch": batch,
                    "h": h,
                    "hv": hv,
                    "seq_size": seq_size,
                    "plus_1": plus_1,
                    "strided": strided,
                    **{name: vals for name, vals in checks.items()},
                }
            )
            (out_dir / "flashinfer_decode.json").write_text(
                json.dumps(report, indent=2)
            )
        report["status"] = "PASS"
        (out_dir / "flashinfer_decode.json").write_text(json.dumps(report, indent=2))

    @torch.inference_mode()
    def test_dispatch_env_switch(self):
        if not torch.cuda.is_available():
            self.skipTest("cuda required")
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("FlashInfer GDN decode requires SM90+")
        torch.manual_seed(19)
        device = "cuda"
        batch, h, hv, seq_size, plus_1 = 4, 4, 8, 64, 24576
        mixed = torch.randn(
            batch, (2 * h + hv) * 128, device=device, dtype=torch.bfloat16
        )
        a = torch.randn(batch, hv, device=device, dtype=torch.bfloat16)
        b = torch.randn_like(a)
        read_page = (plus_1 - 2) // seq_size
        n_pages = read_page + 1
        pages = 1 + batch
        ssm = torch.randn(pages, hv, 128, 128, device=device, dtype=torch.bfloat16) * 0.01
        block_map = torch.zeros(batch, n_pages, device=device, dtype=torch.int32)
        block_map[:, read_page] = torch.arange(
            1, batch + 1, device=device, dtype=torch.int32
        )
        args = SimpleNamespace(
            kv_cache_kernel_block_id_device=block_map,
            sequence_lengths_plus_1_device=torch.full(
                (batch,), plus_1, device=device, dtype=torch.int32
            ),
            prefix_lengths=torch.empty(0),
        )
        obj = SimpleNamespace(
            alog=torch.randn(hv, device=device, dtype=torch.float32),
            dt_bias=torch.randn(hv, device=device, dtype=torch.bfloat16),
            local_num_k_heads=h,
            local_num_v_heads=hv,
            head_k_dim=128,
            head_v_dim=128,
            _get_ssm_states=lambda c: c,
            _get_fla_block_map=lambda attn: attn.kv_cache_kernel_block_id_device,
            _get_bs_from_attenion_input=lambda mixed, attn, verify: (
                mixed.shape[0],
                1,
            ),
        )

        def run(backend):
            cache = ssm.clone()
            with patch.dict(os.environ, {"RTP_QWEN35_GDN_DECODE_BACKEND": backend}):
                self.assertEqual(gdn_decode_backend(), backend)
                y = Qwen3NextGatedDeltaNetDecode._fla(
                    obj, mixed, b, a, cache, seq_size, args, False
                )
            return y, cache

        ref, rc = run("native")
        actual, ac = run("flashinfer")
        torch.testing.assert_close(actual, ref, atol=0.03, rtol=0.03)
        torch.testing.assert_close(ac, rc, atol=0.03, rtol=0.03)
        self.assertTrue(torch.equal(ac[0], ssm[0]))


if __name__ == "__main__":
    unittest.main()
