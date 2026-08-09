import gc
import unittest

import torch

from rtp_llm.models_py.new_models.deepseek_vl2.test.test_deepseek_vl2_load import (
    _load_language,
    _vision_config,
)
from rtp_llm.models_py.new_models.deepseek_vl2.vision import DeepSeekVLV2VisionModel


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA or ROCm GPU")
class DeepSeekVLV2GpuTest(unittest.TestCase):
    def test_mla_load_and_real_siglip_forward(self):
        model_path = None
        model = None
        vision = None
        try:
            model_path, model, _ = _load_language(
                use_mla=True,
                max_seq_len=64,
                device="cuda",
                compute_dtype=torch.float16,
            )
            self.assertEqual(model.cos_sin_cache.device.type, "cuda")
            self.assertEqual(model.cos_sin_cache.dtype, torch.float32)
            self.assertEqual(model.cos_sin_cache.shape[0], 64)

            vision = DeepSeekVLV2VisionModel(_vision_config(), torch.float16).cuda()
            output = vision(
                torch.zeros(
                    1,
                    3,
                    384,
                    384,
                    dtype=torch.float16,
                    device="cuda",
                )
            )
            self.assertEqual(output.shape, (1, 196, 4))
            self.assertTrue(torch.isfinite(output).all())
        finally:
            del vision
            del model
            if model_path is not None:
                model_path.cleanup()
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
