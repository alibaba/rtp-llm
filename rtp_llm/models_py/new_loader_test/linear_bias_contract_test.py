"""Checkpoint-to-linear parity for the GEMM and post-reduction bias contracts."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.layers.linear import ColumnParallelLinear, RowParallelLinear
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.models_py.registry import register_model


@register_model("_test_dense_row_column_bias")
class _Projections(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        settings = dict(
            bias=True, params_dtype=load.compute_dtype, quant_config=load.quant_config
        )
        self.row = RowParallelLinear(64, 128, prefix="row", **settings)
        self.col = ColumnParallelLinear(64, 128, prefix="col", **settings)


class _Fixture(unittest.TestCase):
    def check_load(self, device, swizzled, method):
        weights = (
            torch.randn(128, 64, generator=torch.Generator().manual_seed(17)).half()
            / 16
        )
        bias = torch.arange(128).half() / 64 + 1
        with tempfile.TemporaryDirectory() as root:
            save_file(
                {
                    f"{leaf}.{key}": value.clone()
                    for leaf in ("col", "row")
                    for key, value in (("weight", weights), ("bias", bias))
                },
                str(Path(root) / "model.safetensors"),
            )
            loader = NewModelLoader(
                SimpleNamespace(model_type="_test_dense_row_column_bias"),
                NewLoaderConfig(
                    compute_dtype=torch.float16,
                    device=device,
                    load_method=method,
                    quant_config=QuantizationConfig(
                        "none", hw_kernel_config=SimpleNamespace(use_swizzleA=swizzled)
                    ),
                ),
                model_path=root,
            )
            owner = loader.load()
        self.assertEqual(loader.loaded_method.value, method)
        self.assertIsNone(owner.row.matmul_bias)
        self.assertIs(owner.col.matmul_bias, owner.col.bias)
        x = torch.randn(5, 64, generator=torch.Generator().manual_seed(19)).half()
        expected = torch.nn.functional.linear(
            x.float(), weights.float(), bias.float()
        ).half()
        # Zero input isolates duplicate bias independently of GEMM accuracy.
        for name in ("row", "col"):
            leaf = getattr(owner, name)
            if device == "cuda":
                self.assertEqual(
                    bool(getattr(leaf, "_dense_swizzled", False)), swizzled
                )
                self.assertIs(leaf._dense_runtime_executor.bias, leaf.matmul_bias)
            torch.testing.assert_close(
                leaf(torch.zeros_like(x, device=device)).cpu(),
                bias.expand_as(expected),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                leaf(x.to(device)).cpu(), expected, rtol=2e-3, atol=3e-3
            )


class CPUCheckpointBiasTest(_Fixture):
    def test_scratch_and_fst(self):
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(method=method):
                self.check_load("cpu", False, method)


class ROCmCheckpointBiasTest(_Fixture):
    def test_hipblas_gemm_adds_checkpoint_bias_once(self):
        self.assertIsNotNone(torch.version.hip)
        for swizzled in (False, True):
            with self.subTest(swizzled=swizzled):
                self.check_load("cuda", swizzled, "scratch")


if __name__ == "__main__":
    unittest.main()
