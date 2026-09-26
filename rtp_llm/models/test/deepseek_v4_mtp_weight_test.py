"""Exercise MTP global FP8 loading, including scale ownership and dtype."""

from types import SimpleNamespace
import unittest

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models.deepseek_v4 import DeepSeekV4MtpWeight
from rtp_llm.utils.model_weight import W


class MtpProjectionWeightTest(unittest.TestCase):
    def test_quantized_globals_preserve_checkpoint_bytes_without_overwrite(self):
        descriptor = DeepSeekV4MtpWeight.__new__(DeepSeekV4MtpWeight)
        descriptor._hidden_size = 256
        descriptor._num_layers = 0
        info = descriptor._get_weight_info().to_quant_weight_info(
            Fp8BlockWiseQuantConfig(is_quanted=True)
        )
        expected = {}
        tensors = {}
        for stem, weight_key, scale_key in (
            ("e_proj", W.v4_mtp_e_proj_w, W.v4_mtp_e_proj_s),
            ("h_proj", W.v4_mtp_h_proj_w, W.v4_mtp_h_proj_s),
        ):
            weight = (torch.arange(256 * 256) % 127).to(torch.uint8)
            weight = weight.reshape(256, 256).view(torch.float8_e4m3fn)
            scale = torch.tensor([[126, 127], [128, 129]], dtype=torch.uint8)
            scale = scale.view(torch.float8_e8m0fnu)
            tensors[f"mtp.0.{stem}.weight"] = expected[weight_key] = weight
            tensors[f"mtp.0.{stem}.scale"] = expected[scale_key] = scale
        reads = []

        def load_tensor(name, dtype):
            reads.append(name)
            return [tensors[name].to(dtype=dtype)]

        source = SimpleNamespace(load_tensor=load_tensor)
        config = SimpleNamespace(
            compute_dtype=torch.bfloat16,
            merge_lora=False,
            tp_size=1,
            dp_size=1,
            ep_size=1,
            use_swizzleA=False,
            exported_device=SimpleNamespace(
                maybe_rewrite_weight_by_key=lambda name, tensor, **kwargs: tensor
            ),
        )
        loaded = {}
        for module in info.weights:
            if module.name not in expected:
                continue
            values = module.load(source, None, "cpu", config)
            self.assertFalse(set(loaded) & set(values), "duplicate global writer")
            loaded.update(values)
        self.assertCountEqual(reads, tensors)
        self.assertEqual(set(loaded), set(expected))
        for name, tensor in expected.items():
            self.assertEqual(loaded[name].dtype, tensor.dtype)
            self.assertTrue(
                torch.equal(loaded[name].view(torch.uint8), tensor.view(torch.uint8))
            )


if __name__ == "__main__":
    unittest.main()
