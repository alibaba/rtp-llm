"""SM90 bit-exactness and CUDA Graph tests for Wave 10 Qwen4 fusions."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)
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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Wave10Sm90FusionTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 fast paths")
        torch.manual_seed(20261123)

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

    def test_linear_shared_sigmoid_mul_and_l20x_compact_moe(self):
        from unittest.mock import patch
        from rtp_llm.models_py.modules.qwen4_exp.moe.compact_fp8_decode import (
            _H20_COMPACT_EAGER_MAX_ROWS,
            compact_fp8_decode,
        )
        from rtp_llm.models_py.modules.qwen4_exp.moe.shared_gate_triton import (
            try_linear_shared_sigmoid_mul,
            try_shared_sigmoid_mul,
        )

        self.assertGreaterEqual(_H20_COMPACT_EAGER_MAX_ROWS, 6144)
        w = (torch.randn((1, 2560), device="cuda") * 0.05).bfloat16()
        for rows in (1, 2, 4, 8, 16, 64, 192, 512):
            x = torch.randn((rows, 2560), device="cuda", dtype=torch.bfloat16)
            sh = torch.randn((rows, 2560), device="cuda", dtype=torch.bfloat16)
            gate = torch.nn.functional.linear(x, w)
            ref = try_shared_sigmoid_mul(gate, sh)
            act = try_linear_shared_sigmoid_mul(x, w, sh)
            self.assertIsNotNone(act)
            torch.testing.assert_close(act, ref, atol=1e-2, rtol=1e-2)
            out_buf = torch.empty_like(sh)
            act_inplace = try_linear_shared_sigmoid_mul(x, w, sh, out=out_buf)
            self.assertEqual(act_inplace.data_ptr(), out_buf.data_ptr())
            self.assertTrue(torch.equal(out_buf, act))

        # CUDA Graph capture & replay of try_linear_shared_sigmoid_mul
        gx = torch.randn((64, 2560), device="cuda", dtype=torch.bfloat16)
        gsh = torch.randn((64, 2560), device="cuda", dtype=torch.bfloat16)
        gout = torch.empty_like(gsh)
        try_linear_shared_sigmoid_mul(gx, w, gsh, out=gout)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            try_linear_shared_sigmoid_mul(gx, w, gsh, out=gout)
        for _ in range(3):
            gx.copy_(torch.randn_like(gx))
            gsh.copy_(torch.randn_like(gsh))
            graph.replay()
            ref_g = try_shared_sigmoid_mul(torch.nn.functional.linear(gx, w), gsh)
            torch.testing.assert_close(gout, ref_g, atol=1e-2, rtol=1e-2)
        graph.reset()

        # Verify 10K/4800 prefill row count (5200) under simulated 132-SM L20X properties
        from rtp_llm.models_py.modules.qwen4_exp.moe import (
            compact_fp8_decode as compact_mod,
        )

        tokens = 5200
        x_fp8 = torch.randn((tokens, 2560), device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        x_sc = torch.ones((tokens, 20), device="cuda", dtype=torch.float32) * 0.05
        ids = torch.randint(0, 16, (tokens, 10), device="cuda", dtype=torch.int32)
        rw = torch.softmax(
            torch.randn((tokens, 10), device="cuda", dtype=torch.float32), dim=-1
        )
        counts = torch.bincount(ids.view(-1), minlength=16).to(torch.int32)
        up_w = torch.randn((16, 1024, 2560), device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        up_s = torch.ones((16, 8, 20), device="cuda", dtype=torch.float32) * 0.02
        dn_w = torch.randn((16, 2560, 512), device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        dn_s = torch.ones((16, 20, 4), device="cuda", dtype=torch.float32) * 0.02
        out_h20 = compact_fp8_decode(
            x_fp8, x_sc, ids, rw, counts, (up_w, up_s), (dn_w, dn_s)
        )
        with patch.object(compact_mod, "_device_num_sms", return_value=132):
            out_l20x = compact_fp8_decode(
                x_fp8, x_sc, ids, rw, counts, (up_w, up_s), (dn_w, dn_s)
            )
        torch.testing.assert_close(out_l20x, out_h20, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
