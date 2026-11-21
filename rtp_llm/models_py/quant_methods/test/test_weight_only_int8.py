import os
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.layers.linear import RowParallelLinear
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.groupwise_export import cuda_groupwise_layout
from rtp_llm.models_py.registry import MODEL_REGISTRY


class _Model(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.proj = RowParallelLinear(
            input_size=256,
            output_size=192,
            tp_size=load.tp_size,
            tp_rank=load.tp_rank,
            prefix="proj",
            bias=True,
            quant_config=load.quant_config,
            params_dtype=load.compute_dtype,
        )


def load_fixture(device, tp=1, rank=0, method="scratch", dtype=torch.float16):
    gen = torch.Generator().manual_seed(91299458)
    weight = torch.randn((192, 256), generator=gen).to(dtype)
    # Different rank-local max values distinguish shard-before-quantization.
    weight[:, 128:] *= 4
    bias = torch.randn(192, generator=gen).to(dtype)
    with tempfile.TemporaryDirectory() as directory:
        save_file(
            {"proj.weight": weight, "proj.bias": bias},
            os.path.join(directory, "model.safetensors"),
        )
        with patch.dict(MODEL_REGISTRY, {"w8a16_fixture": _Model}):
            loader = NewModelLoader(
                types.SimpleNamespace(model_type="w8a16_fixture"),
                NewLoaderConfig(
                    device=device,
                    compute_dtype=dtype,
                    tp_size=tp,
                    tp_rank=rank,
                    quant_config=QuantizationConfig("INT8"),
                    load_method=method,
                    force_cpu_load_weights=(method == "fastsafetensors"),
                ),
                model_path=directory,
            )
            layer = loader.load().proj
            assert loader.loaded_method.value == method
    local = weight.chunk(tp, dim=1)[rank].T
    scale = local.abs().amax(0).clamp(min=1e-8) / 128
    q = (local / scale).round().clamp(-128, 127).to(torch.int8)
    return layer, q, scale, bias


class WeightOnlyInt8CheckpointTest(unittest.TestCase):
    def test_local_rank_before_quantization_and_native_fst_parity(self):
        for method in ("scratch", "fastsafetensors"):
            for rank in (0, 1):
                with self.subTest(method=method, rank=rank):
                    layer, q, scale, bias = load_fixture("cpu", 2, rank, method=method)
                    expected = cuda_groupwise_layout(q, 8, 80)
                    torch.testing.assert_close(layer.weight, expected, atol=0, rtol=0)
                    torch.testing.assert_close(
                        layer.weight_scale, scale, atol=0, rtol=0
                    )
                    torch.testing.assert_close(layer.bias, bias, atol=0, rtol=0)
                    storage = layer.weight.data_ptr()
                    layer.process_weights_after_loading()
                    self.assertEqual(layer.weight.data_ptr(), storage)

    def test_reject_prequantized_and_unaligned_weights(self):
        args = dict(quant_config=QuantizationConfig("INT8"), prefix="proj")
        with self.assertRaisesRegex(ValueError, "divisible by 64"):
            RowParallelLinear(input_size=96, output_size=192, **args)
        layer = RowParallelLinear(input_size=128, output_size=192, **args)
        with self.assertRaisesRegex(ValueError, "floating checkpoint"):
            layer.load_weights({"weight": torch.zeros(192, 128, dtype=torch.int8)})


if __name__ == "__main__":
    unittest.main()
