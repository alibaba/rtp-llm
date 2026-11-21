"""Exercise the V4-owned norm on an actual GPU; never depend on import order."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.new_models.deepseek_v4.cuda_rms import (
    native_rmsnorm,
    use_native_rms,
)


class CudaV4RmsTest(unittest.TestCase):
    def test_v41_target_loaded_norm_binds_native_and_preserves_inplace(self):
        import os
        import tempfile

        from flashinfer.jit.norm import gen_norm_module
        from flashinfer.utils import device_support_pdl
        from safetensors import safe_open
        from safetensors.torch import save_file

        from rtp_llm.models_py.module_base import RtpModule
        from rtp_llm.models_py.new_models.deepseek_v41.backbone import (
            _target_block_norm_factory,
        )

        dev = torch.device("cuda:0")
        owner = RtpModule()
        # Also exercise the ARM route on non-ARM CUDA test workers.
        with patch("platform.machine", return_value="aarch64"), patch.object(
            torch.version, "cuda", "13.0"
        ), patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
            owner.norm = _target_block_norm_factory(dev)(5120, 1e-20, torch.bfloat16)
        self.assertIs(owner.norm._implementation, native_rmsnorm)
        self.assertIs(
            type(_target_block_norm_factory(None)(64, 1e-6, torch.bfloat16)), RMSNorm
        )
        owner.to(dev)
        dest = owner.norm.weight
        rng = torch.Generator().manual_seed(92255105)
        w = torch.randn(5120, generator=rng).bfloat16()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "norm.safetensors")
            save_file({"norm.weight": w}, path)
            with safe_open(path, framework="pt") as ckpt:
                owner.load_weights([("norm.weight", ckpt.get_tensor("norm.weight"))])
        self.assertIs(owner.norm.weight, dest)
        self.assertTrue(torch.equal(dest.cpu(), w))
        vendor = gen_norm_module().build_and_load()
        for n in (392, 136):
            x = torch.randn(n, 5120, generator=rng).bfloat16().to(dev)
            expected = torch.empty_like(x)
            vendor.rmsnorm(expected, x, dest, 1e-20, device_support_pdl(dev))
            self.assertIs(owner.norm(x, output=x), x)
            self.assertTrue(torch.equal(x, expected))

    def test_route_is_only_v4_cuda13_sm100_arm(self):
        with patch("platform.machine", return_value="aarch64"), patch.object(
            torch.version, "cuda", "13.0"
        ), patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 0)
        ) as arch:
            self.assertFalse(use_native_rms("cpu"))
            arch.assert_not_called()
            self.assertTrue(use_native_rms("cuda"))
            with patch("platform.machine", return_value="x86_64"):
                self.assertFalse(use_native_rms("cuda"))
            arch.return_value = (9, 0)
            self.assertFalse(use_native_rms("cuda"))
            arch.return_value = (10, 0)
            with patch.object(torch.version, "cuda", "12.9"):
                self.assertFalse(use_native_rms("cuda"))

    def test_cuda_matches_vendor_native_with_graph_and_alias(self):
        from flashinfer.jit.norm import gen_norm_module
        from flashinfer.utils import device_support_pdl

        dev = torch.device("cuda:0")
        cpu_rng = torch.Generator().manual_seed(4319)

        def sample(*shape):
            return torch.randn(*shape, generator=cpu_rng).bfloat16().to(dev)

        w = torch.nn.Parameter(sample(4096), requires_grad=False)
        model = RMSNorm(w, 1.0e-6, implementation=native_rmsnorm)
        self.assertIs(model.weight, w)
        vendor = gen_norm_module().build_and_load()
        pdl = device_support_pdl(dev)

        def expected(x):
            ref = torch.empty_like(x)
            vendor.rmsnorm(ref, x, w, 1.0e-6, pdl)
            return ref

        x = sample(28, 4096)
        torch.testing.assert_close(model(x), expected(x), rtol=0, atol=0)
        scratch = x.clone()
        self.assertIs(model(scratch, output=scratch), scratch)
        self.assertTrue(torch.equal(scratch, expected(x)))
        graph = torch.cuda.CUDAGraph()
        out = torch.empty_like(x)
        with torch.cuda.graph(graph):
            model(x, output=out)
        for _ in range(3):
            x.copy_(sample(28, 4096))
            graph.replay()
            self.assertTrue(torch.equal(out, expected(x)))


if __name__ == "__main__":
    unittest.main()
