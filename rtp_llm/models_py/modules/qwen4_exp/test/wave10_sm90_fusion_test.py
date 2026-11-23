"""SM90 bit-exactness and CUDA Graph tests for Wave 10 Qwen4 fusions."""

import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)
from rtp_llm.models_py.modules.hybrid.gated_residual import GatedResidual
from rtp_llm.models_py.modules.qwen4_exp.gdn_decode_triton import (
    _run_packed_decode,
    try_packed_decode,
)
from rtp_llm.models_py.modules.qwen4_exp.gdn_fp8_input_projection import (
    _ba_scale,
    _quantize_ba,
    _quantize_qkvz,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_fp8_input_projection import (
    try_prepare_qsa_triple,
)
from rtp_llm.models_py.triton_kernels.common.gated_residual_mix_triton import (
    fused_mix_reduce,
)
from rtp_llm.models_py.triton_kernels.common.gated_residual_sm90_up_mix_triton import (
    try_sm90_bf16_up_mix,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Wave10Sm90FusionTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 fast paths")
        torch.manual_seed(20261123)

    def test_sm90_bf16_up_mix_bit_exact_and_graph(self):
        bf16_up = (
            torch.randn((10240, 320), device="cuda", dtype=torch.bfloat16) * 0.02
        ).contiguous()
        for rows in (1, 3, 4, 16, 32, 64, 65, 96, 128):
            x = torch.randn((rows, 320), device="cuda", dtype=torch.bfloat16)
            normed = torch.randn((rows, 10240), device="cuda", dtype=torch.bfloat16)
            expected = fused_mix_reduce(F.linear(x, bf16_up), normed, 4)
            actual = try_sm90_bf16_up_mix(x, bf16_up, normed, 4)
            self.assertIsNotNone(actual)
            self.assertTrue(
                torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)),
                f"mismatch at rows={rows}: max_err={(actual.float() - expected.float()).abs().max().item()}",
            )
        for rows in (0, 129, 2048):
            x = torch.empty((rows, 320), device="cuda", dtype=torch.bfloat16)
            normed = torch.empty((rows, 10240), device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(try_sm90_bf16_up_mix(x, bf16_up, normed, 4))

        # Full GatedResidual consumer + CUDA Graph replay
        gamma = torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.02
        down = (torch.randn((320, 10240), device="cuda") * 0.1).to(torch.float8_e4m3fn)
        up = (torch.randn((10240, 320), device="cuda") * 0.1).to(torch.float8_e4m3fn)
        inj = torch.randn((4, 10240), device="cuda", dtype=torch.bfloat16) * 0.02
        s_down = torch.tensor([0.02], device="cuda", dtype=torch.float32)
        s_up = torch.tensor([0.02], device="cuda", dtype=torch.float32)
        model = GatedResidual(
            gamma,
            down,
            up,
            inj,
            hc_mult=4,
            norm_eps=1e-6,
            weight_scales=(s_down, s_up, None),
        )
        for rows in (4, 64, 96, 128, 2048):
            hx = torch.randn((rows, 10240), device="cuda", dtype=torch.bfloat16)
            with patch.dict(os.environ, {"RTP_QWEN4_DISABLE_SM90_BF16_UP_MIX": "1"}):
                ref_out = model(hx)
            got_out = model(hx)
            for a, b in zip(got_out, ref_out):
                self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))

        hx = torch.randn((128, 10240), device="cuda", dtype=torch.bfloat16)
        model(hx)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_out = model(hx)
        for step in range(3):
            hx.normal_(0, 0.5 + 0.3 * step)
            model.norm_gamma.normal_(0, 0.05)
            graph.replay()
            with patch.dict(os.environ, {"RTP_QWEN4_DISABLE_SM90_BF16_UP_MIX": "1"}):
                ref_out = model(hx)
            for a, b in zip(g_out, ref_out):
                self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
        graph.reset()

    def test_qsa_triple_quantization_and_ba_graph_cap(self):
        qkv_w = (torch.randn((2560, 3584), device="cuda") * 0.1).to(
            torch.float8_e4m3fn
        )
        qkv_s = torch.rand((20, 28), device="cuda", dtype=torch.float32) * 0.02 + 1e-3
        qkv_proj = LinearFactory.create_linear(
            weight=qkv_w,
            weight_scales=qkv_s,
            bias=None,
            quant_config=Fp8BlockWiseQuantConfig(is_quanted=True),
        )
        self.assertIs(type(qkv_proj), CudaFp8GEMMLinear)
        gate = Fp8Linear(
            (torch.randn((3072, 2560), device="cuda") * 0.1).to(torch.float8_e4m3fn),
            torch.tensor([0.03125], device="cuda", dtype=torch.float32),
        )
        qk_proj = Fp8Linear(
            (torch.randn((640, 2560), device="cuda") * 0.1).to(torch.float8_e4m3fn),
            torch.tensor([0.0195], device="cuda", dtype=torch.float32),
        )

        for rows in (1, 4, 16, 32, 33, 64, 192, 256, 512, 2048):
            x = torch.randn((rows, 2560), device="cuda", dtype=torch.bfloat16)
            res = try_prepare_qsa_triple(x, qkv_proj, gate, qk_proj)
            self.assertIsNotNone(res)
            gate_out, qk_out, run_qkv = res
            qkv_out = run_qkv()
            self.assertTrue(torch.equal(gate_out, gate(x)))
            self.assertTrue(torch.equal(qk_out, qk_proj(x)))
            self.assertTrue(torch.equal(qkv_out, qkv_proj(x)))

        # CUDA Graph capture at rows = 192 (exercising single-pass _quant_from_partial up to 512)
        gx = torch.randn((192, 2560), device="cuda", dtype=torch.bfloat16)
        g0, qk0, rq0 = try_prepare_qsa_triple(gx, qkv_proj, gate, qk_proj)
        _ = rq0()
        graph = torch.cuda.CUDAGraph()
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.gdn_fp8_input_projection._ba_scale",
            wraps=_ba_scale,
        ) as ba_spy:
            with torch.cuda.graph(graph):
                cg, cqk, crq = try_prepare_qsa_triple(gx, qkv_proj, gate, qk_proj)
                cqkv = crq()
            ba_spy.assert_not_called()
        for gain in (0.02, 0.5, 3.0):
            gx.normal_(0, gain)
            graph.replay()
            self.assertTrue(torch.equal(cg, gate(gx)))
            self.assertTrue(torch.equal(cqk, qk_proj(gx)))
            self.assertTrue(torch.equal(cqkv, qkv_proj(gx)))
        graph.reset()

        # Hooks/overrides fall back cleanly
        hook = gate.register_forward_hook(lambda m, a, r: r)
        try:
            self.assertIsNone(try_prepare_qsa_triple(gx, qkv_proj, gate, qk_proj))
        finally:
            hook.remove()

    def test_gdn_verify_and_decode_heads_first_bit_exact(self):
        alog = torch.randn(24, device="cuda", dtype=torch.float32)
        bias = torch.randn(24, device="cuda", dtype=torch.bfloat16)
        for seq_len in (1, 4):
            for batch in (16, 32, 48, 64):
                tokens = batch * seq_len
                qkv = torch.randn((tokens, 5120), device="cuda", dtype=torch.bfloat16)
                a = torch.randn((tokens, 24), device="cuda", dtype=torch.bfloat16)
                b = torch.randn_like(a)
                mapping = torch.arange(
                    1, 8 * batch + 1, device="cuda", dtype=torch.int32
                ).view(batch, 8)
                lengths = torch.full((batch,), 68, device="cuda", dtype=torch.int32)
                st_init = (
                    torch.randn(
                        (8 * batch + 2, 24, 128, 128),
                        device="cuda",
                        dtype=torch.float32,
                    )
                    * 0.02
                )
                st_ref = st_init.clone()
                ref_out = _run_packed_decode(
                    qkv,
                    a,
                    b,
                    alog,
                    bias,
                    st_ref,
                    mapping,
                    lengths,
                    64,
                    value_tile=8,
                    sequence_length=seq_len,
                    heads_first=False,
                )
                st_act = st_init.clone()
                act_out = try_packed_decode(
                    qkv,
                    a,
                    b,
                    alog,
                    bias,
                    st_act,
                    mapping,
                    lengths,
                    64,
                    sequence_length=seq_len,
                    tune_h20=True,
                )
                self.assertIsNotNone(act_out)
                self.assertTrue(
                    torch.equal(act_out.view(torch.uint8), ref_out.view(torch.uint8))
                )
                self.assertTrue(
                    torch.equal(st_act.view(torch.uint8), st_ref.view(torch.uint8))
                )


if __name__ == "__main__":
    unittest.main()
