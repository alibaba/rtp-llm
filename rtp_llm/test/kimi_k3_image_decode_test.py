import pickle
from io import BytesIO
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch
from PIL import Image

import rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_vit as kimi_k3_vit
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_vit import (
    KimiK3ImageEmbedding,
)
from rtp_llm.multimodal.multimodal_util import MMUrlType


class KimiK3ImageDecodeTest(TestCase):
    def test_heic_with_decoder_context_result_is_picklable(self):
        encoded = BytesIO()
        Image.new("RGB", (8, 6), (1, 2, 3)).save(encoded, format="HEIF")
        raw = encoded.getvalue()
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)

        class UnpicklableCtxImage:
            def __reduce__(self):
                raise TypeError("cannot pickle 'CtxImage' object")

        decoded = Image.open(BytesIO(raw))
        decoded.info["depth_images"] = [UnpicklableCtxImage()]
        with self.assertRaisesRegex(TypeError, "CtxImage"):
            pickle.dumps(decoded.copy())

        mm_input = SimpleNamespace(
            mm_type=MMUrlType.IMAGE,
            tensor=tensor,
            url="",
        )
        vit_config = VitConfig()
        vit_config.download_headers = '{"X-Test": "value"}'
        with patch.object(kimi_k3_vit.Image, "open", return_value=decoded):
            image = KimiK3ImageEmbedding.preprocess_input([mm_input], vit_config)

        self.assertEqual(image.size, (8, 6))
        self.assertEqual(image.mode, "RGB")
        pickle.dumps(image)


if __name__ == "__main__":
    main()
