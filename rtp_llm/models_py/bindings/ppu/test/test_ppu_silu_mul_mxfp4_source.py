# Copyright 2026 Alibaba Group Holding Limited.
# SPDX-License-Identifier: Apache-2.0
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[5]
PPU = ROOT / "rtp_llm/models_py/bindings/ppu"
KERNEL = PPU / "kernels/ppu_silu_mul_mxfp4.cu"
HEADER = PPU / "kernels/ppu_silu_mul_mxfp4.h"
NOTICE = PPU / "kernels/ppu_silu_mul_mxfp4.NOTICE"
OP = PPU / "PpuSiluMulMxfp4Op.cc"
REGISTER = PPU / "RegisterPpuOps.cc"
KERNEL_BUILD = PPU / "kernels/BUILD"

class PpuSiluMulMxfp4SourceTest(unittest.TestCase):
    def test_frozen_apache_provenance(self):
        text, notice = KERNEL.read_text(), NOTICE.read_text()
        sha = "131bd3ff990cfbf544f954cc7364110058e5e611ece5454842df4eaa176a2c1a"
        self.assertIn("SPDX-License-Identifier: Apache-2.0", text)
        self.assertIn(sha, text)
        self.assertIn(sha, notice)
        self.assertIn("SGLang v0.5.14_release", notice)

    def test_ppu_native_aot_and_stream_contract(self):
        text, op = KERNEL.read_text(), OP.read_text()
        self.assertIn("__ppu_sgmdf", text)
        self.assertIn("cvt.rn.satfinite.e2m1x2.f32", text)
        self.assertIn("const SiluMulMxfp4Params2D params", text)
        self.assertIn("const dim3 grid(hidden_blocks, token_blocks)", text)
        self.assertIn("constexpr int threads = kBlockN / kElemPerThread", text)
        self.assertIn("<kBlockN, kApplySwigluLimit, true>", text)
        self.assertIn("<kBlockN, kApplySwigluLimit, false>", text)
        self.assertIn("kBlocksTargetDefault", text)
        self.assertIn("GET_CURRENT_STREAM()", op)
        self.assertIn("gate_up.stride(0),", op)
        self.assertNotIn("gate_up.stride(0) * gate_up.element_size()", op)
        for banned in ("torch.cuda.synchronize", "cudaDeviceSynchronize",
                       ".cpu(", ".item(", ".tolist(", "sglang", "jit"):
            self.assertNotIn(banned.lower(), op.lower())

    def test_non_ppu_fail_closed(self):
        op, register = OP.read_text(), REGISTER.read_text()
        kernel_build = KERNEL_BUILD.read_text()
        self.assertIn("#ifdef USE_PPU", op)
        self.assertIn("#ifdef USE_PPU", register)
        self.assertIn("PpuSiluAndMulPostQuantMxfp4", register)
        self.assertIn("py::object swiglu_limit", register)
        self.assertIn("!swiglu_limit.is_none()", register)
        self.assertIn("swiglu_limit.cast<double>()", register)
        self.assertIn('py::arg("swiglu_limit") = py::none()', register)
        self.assertNotIn('py::arg("apply_swiglu_limit")', register)
        self.assertIn('"@platforms//:incompatible"', kernel_build)
        self.assertNotIn("fallback", op.lower())

    def test_layout_contract(self):
        op = OP.read_text()
        for token in ("torch::kBFloat16", "torch::kUInt8", "torch::kUInt16",
                      ".transpose(0, 1)", "hidden / 2", "(hidden + 63) / 64"):
            self.assertIn(token, op)
        self.assertIn("hidden_padded / 64", op)
        self.assertNotIn("(hidden + 31) / 32", op)
        self.assertNotIn("hidden_padded / 32", op)

    def test_no_consumer_dependency(self):
        self.assertNotIn("ppu_grouped_fp4", KERNEL.read_text() + OP.read_text())
        self.assertTrue(HEADER.exists())

if __name__ == "__main__":
    unittest.main()
