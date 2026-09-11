"""CUDA HC producer/consumer regression, executed with the platform's locked SDK."""

import importlib
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA SM100")
class HCCudaSplitKTest(unittest.TestCase):
    def test_runtime_and_warmup_with_multiple_partial_planes(self):
        from rtp_llm.device.device_type import DeviceType, get_device_type

        if (
            get_device_type() != DeviceType.Cuda
            or torch.cuda.get_device_capability()[0] != 10
        ):
            self.skipTest("requires CUDA SM100; PPU has a different prenorm ABI")
        import deep_gemm

        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import tf32_hc_prenorm_gemm
        from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401
        from rtp_llm.models_py.modules.dsv4.dsv4_kernel_jit_warmup import (
            _launch_dummy_mhc_pre_big_fuse,
            _launch_dummy_mhc_prenorm_gemm,
        )

        print(
            "CUDA HC SDK",
            torch.__version__,
            torch.version.cuda,
            getattr(deep_gemm, "__version__", "unknown"),
        )
        pre = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.pre_big_fuse"
        )
        torch.manual_seed(1396)
        residual = torch.randn(3, 4, 1024, dtype=torch.bfloat16, device="cuda") * 0.2
        fn = torch.randn(24, 4096, dtype=torch.float32, device="cuda") * 0.1
        observed = []

        def producer(x, weight, out, squares, splits):
            self.assertGreater(splits, 1)
            self.assertEqual(out.shape[0], splits)
            self.assertEqual(squares.shape[0], splits)
            tf32_hc_prenorm_gemm(x, weight, out, squares, splits)
            torch.testing.assert_close(
                out.sum(0), x.float() @ weight.t(), rtol=0.02, atol=0.03
            )
            torch.testing.assert_close(
                squares.sum(0), x.float().square().sum(-1), rtol=1e-4, atol=1e-4
            )
            observed.append(splits)

        outputs = pre.mhc_pre_big_fuse(
            residual,
            fn,
            torch.ones(3, device="cuda"),
            torch.zeros(24, device="cuda"),
            1e-6,
            1e-6,
            1e-6,
            2.0,
            20,
            backend="deepgemm",
            prenorm_gemm=producer,
        )
        self.assertEqual(len(observed), 1)
        self.assertTrue(all(torch.isfinite(t).all().item() for t in outputs))
        for launcher in (
            _launch_dummy_mhc_prenorm_gemm,
            _launch_dummy_mhc_pre_big_fuse,
        ):
            launcher(
                key=(24, 4096),
                info={"fn": fn},
                m_value=3,
                num_splits=observed[0],
                device=torch.device("cuda"),
            )
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
