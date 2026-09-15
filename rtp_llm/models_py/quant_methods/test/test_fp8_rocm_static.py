import subprocess
import sys
import unittest

import torch
from rtp_llm.config.quant_config import Fp8PerTensorCompressedQuantConfig
from rtp_llm.models_py.layers.linear import ColumnParallelLinear
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.fp8 import (
    _is_hip_runtime,
    _runtime_fp8_dtype,
    _select_fp8_runtime_backend,
)


class TestFp8RocmImportIsolation(unittest.TestCase):
    def test_swizzle_executor_import_does_not_require_cktile(self):
        script = r"""
import builtins

real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "aiter.ops.gemm_op_a8w8" and "gemm_a8w8_bpreshuffle_cktile" in fromlist:
        raise AssertionError("swizzle executor imported CKTile eagerly")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
from rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear import (
    RocmFp8PTPCLinearWithSwizzle,
)
assert RocmFp8PTPCLinearWithSwizzle is not None
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


@unittest.skipUnless(
    torch.cuda.is_available() and _is_hip_runtime(),
    "requires a ROCm accelerator",
)
class TestFp8RocmStatic(unittest.TestCase):
    def setUp(self):
        try:
            _select_fp8_runtime_backend(torch.device("cuda"), "per_tensor")
        except RuntimeError as error:
            self.skipTest(str(error))
        if _runtime_fp8_dtype() != torch.float8_e4m3fnuz:
            self.skipTest("requires an FNUZ ROCm runtime")

    def test_fnuz_scale_matches_e4m3fn_checkpoint_reference(self):
        n, k, m = 16, 32, 4
        device = torch.device("cuda")
        weight_source = torch.linspace(
            -2.0, 2.0, n * k, dtype=torch.float32, device=device
        ).reshape(n, k)
        input_source = torch.linspace(
            -1.5, 1.5, m * k, dtype=torch.float32, device=device
        ).reshape(m, k)
        weight_scale = weight_source.abs().max().reshape(1) / 448.0
        input_scale = input_source.abs().max().reshape(1) / 448.0
        checkpoint_weight = (weight_source / weight_scale).to(torch.float8_e4m3fn)
        input_bf16 = input_source.to(torch.bfloat16)

        source_config = Fp8PerTensorCompressedQuantConfig(
            is_quanted=True,
            dynamic=False,
        )
        layer = ColumnParallelLinear(
            input_size=k,
            output_size=n,
            quant_config=QuantizationConfig(
                "fp8",
                source_config=source_config,
            ),
            prefix="static_reference",
            params_dtype=torch.bfloat16,
        ).to(device)
        layer.load_weights(
            {
                "static_reference.weight": checkpoint_weight,
                "static_reference.weight_scale": weight_scale,
                "static_reference.input_scale": input_scale,
            }
        )
        layer.validate_weights_loaded()
        layer.process_weights_after_loading()

        torch.testing.assert_close(layer.input_scale, input_scale * 2.0)
        output = layer(input_bf16)

        reference_input = (input_bf16.float() / input_scale).to(
            torch.float8_e4m3fn
        ).float() * input_scale
        reference_weight = (weight_source / weight_scale).to(
            torch.float8_e4m3fn
        ).float() * weight_scale
        reference = torch.nn.functional.linear(reference_input, reference_weight).to(
            output.dtype
        )
        torch.testing.assert_close(output, reference, rtol=0.05, atol=0.05)


if __name__ == "__main__":
    unittest.main()
