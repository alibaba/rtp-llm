import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs


class HeadOnlyModel(GptModelBase):
    def __init__(self):
        nn.Module.__init__(self)
        self.kv_cache = None
        self.nvfp4_lm_head = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.nvfp4_lm_head.weight.copy_(
                torch.arange(128).reshape(16, 8).to(torch.bfloat16) / 128
            )

    def forward(self, inputs, fmha_impl=None):
        return PyModelOutputs(inputs.input_hiddens)


class QuantizedLmHeadOutputTest(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
        "requires SM120 NVFP4",
    )
    def test_real_packed_head_loader_and_projection(self):
        from rtp_llm.config.modelopt_mixed import ModelOptMixedPrecisionConfig
        from rtp_llm.device.device_impl import CudaImpl
        from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
        from rtp_llm.models_py.modules.factory.linear import LinearFactory
        from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity

        config = ModelOptMixedPrecisionConfig(
            {"lm_head": {"quant_algo": "NVFP4", "group_size": 16}}
        )
        loader = WeightModule.create(
            AtomicWeight(W.lm_head, [CkptWeightInfo("lm_head.weight", identity)]),
            config,
        )
        # E2M1 code 7 is exactly 6; each byte packs two weights. Unit block
        # scales give an independent, exact reference for the real FP4 GEMM.
        tensors = {
            "lm_head.weight": torch.full(
                (128, 64), 0x77, dtype=torch.uint8, device="cuda"
            ),
            "lm_head.weight_scale": torch.ones((128, 8), device="cuda").to(
                torch.float8_e4m3fn
            ),
            "lm_head.weight_scale_2": torch.ones((), device="cuda"),
            "lm_head.input_scale": torch.ones((), device="cuda"),
        }
        from types import SimpleNamespace

        source = SimpleNamespace(
            load_tensor=lambda name, dtype: [tensors[name].to(dtype)]
        )
        load_config = SimpleNamespace(
            compute_dtype=torch.bfloat16, exported_device=CudaImpl
        )
        model = HeadOnlyModel()
        with patch.dict(os.environ, {"RTP_LLM_FP4_GEMM_BACKEND": "cutlass"}):
            weights = loader._postprocess(
                loader._load_raw_tensor(source, None, "cuda", load_config),
                "cuda",
                load_config,
            )
            model.nvfp4_lm_head = LinearFactory.create_linear(
                weights[W.lm_head],
                None,
                weights[W.lm_head_s],
                config,
                weight_scale_2=weights[W.lm_head_s2],
                input_scale=weights[W.lm_head_i_s],
            )
        hidden = torch.full((3, 128), 6, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            native = model._compute_quantized_logits_native(hidden)
            public = model.compute_quantized_logits(hidden)
        self.assertEqual(native.dtype, torch.bfloat16)
        self.assertEqual(public.dtype, torch.float32)
        expected = torch.full((3, 128), 128 * 36, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(native.float(), expected, rtol=0, atol=0)
        torch.testing.assert_close(public, expected, rtol=0, atol=0)

    def test_forward_keeps_hidden_only_contract(self):
        model = HeadOnlyModel()
        hidden = torch.arange(48).reshape(6, 8).float() / 16
        inputs = PyModelInputs(input_hiddens=hidden)
        with patch.object(model.nvfp4_lm_head, "forward") as head:
            outputs = model.forward(inputs)
        head.assert_not_called()
        self.assertTrue(torch.equal(outputs.hidden_states, hidden))

    def test_native_output_preserves_fp32_interface(self):
        # Isolate projection dtype from the packed-weight kernel. The on-device
        # loader/projection case above covers the actual NVFP4 GEMM.
        model = HeadOnlyModel()
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for offset in (0.0, 0.25):
                hidden = (torch.arange(24).reshape(3, 8) / 16 + offset).to(dtype)
                expected = nn.functional.linear(
                    hidden.to(torch.bfloat16), model.nvfp4_lm_head.weight
                ).float()
                native = model._compute_quantized_logits_native(hidden)
                public = model.compute_quantized_logits(hidden)
                self.assertEqual(native.dtype, torch.bfloat16)
                self.assertEqual(public.dtype, torch.float32)
                self.assertTrue(torch.equal(native.float(), expected))
                self.assertTrue(torch.equal(public, expected))

    def test_standalone_samples_packed_and_dense_heads(self):
        from rtp_llm.models_py.standalone.auto_model import AutoModel

        model = HeadOnlyModel()
        runtime = AutoModel.__new__(AutoModel)
        runtime.model = model
        hidden = torch.arange(48).reshape(6, 8).float() / 16
        outputs = model.forward(PyModelInputs(input_hiddens=hidden))
        for packed in (False, True):
            with self.subTest(packed=packed):
                runtime.lm_head_weight = (
                    torch.empty((16, 4), dtype=torch.uint8)
                    if packed
                    else model.nvfp4_lm_head.weight
                )
                with torch.no_grad():
                    actual = runtime._sample_next_token(outputs)
                    expected = model.compute_quantized_logits(hidden[-1:]).argmax(-1)
                self.assertTrue(torch.equal(actual, expected))

    def test_uninitialized_head_fails_explicitly(self):
        model = HeadOnlyModel()
        model.nvfp4_lm_head = None
        for compute in (
            model.compute_quantized_logits,
            model._compute_quantized_logits_native,
        ):
            with self.assertRaisesRegex(ValueError, "No quantized LM head"):
                compute(torch.zeros(1, 8))


if __name__ == "__main__":
    unittest.main()
