"""CPU contracts for the generic DeepSeek-V4.1 multimodal boundary.

Numerical parity remains covered by the GPU preprocess, packed-forward and
real-weight tests.  These regressions prove that frontend/process-engine work
does not expose V4.1 layout metadata or submit CUDA work.
"""

import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import (
    deepseek_v41_vision as vision_module,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
    V41ImageProcessorConfig,
    num_image_tokens,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
    DeepSeekV41VisionEmbedding,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType


class _NoCudaTensor(torch.Tensor):
    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and args:
            candidate = args[0]
            if isinstance(candidate, (str, torch.device)):
                device = candidate
        if device is not None and torch.device(device).type == "cuda":
            raise AssertionError("CPU preprocess hook submitted CUDA work")
        return super().to(*args, **kwargs)


def _make_adapter(hidden=4):
    adapter = object.__new__(DeepSeekV41VisionEmbedding)
    torch.nn.Module.__init__(adapter)
    adapter.processor_config = V41ImageProcessorConfig(vision_min_pixels=1764)
    adapter._packed_forward_enabled = True
    adapter._packed_max_patches = 1061888
    for name, value in (("image_start", -1), ("image_newline", -2), ("image_end", -3)):
        setattr(
            adapter,
            name,
            torch.nn.Parameter(
                torch.full((hidden,), value, dtype=torch.bfloat16),
                requires_grad=False,
            ),
        )
    return adapter.eval()


def _png(mode="RGBA", width=47, height=31):
    image = Image.new(mode, (width, height))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _media():
    return MultimodalInput(
        "https://test.invalid/image.png",
        MMUrlType.IMAGE,
        torch.empty(0, device="cpu"),
        MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 0),
    )


class DeepSeekV41BackendContractTest(unittest.TestCase):
    def setUp(self):
        self.adapter = _make_adapter()
        self.vit_config = SimpleNamespace(
            download_headers='{"x-test": "generic-contract"}',
            mm_image_max_file_size_kb=123,
        )

    def test_cpu_hook_never_submits_cuda_in_either_preprocess_mode(self):
        payload = _png()
        guarded_rgb = (
            torch.arange(47 * 31 * 3, dtype=torch.int64)
            .remainder(256)
            .to(torch.uint8)
            .reshape(31, 47, 3)
            .as_subclass(_NoCudaTensor)
        )

        with patch.object(
            vision_module, "load_image_source", return_value=io.BytesIO(payload)
        ), patch.object(
            vision_module, "_rgb_tensor", return_value=guarded_rgb
        ), patch.object(
            vision_module,
            "_preprocess_rgb_on_device",
            side_effect=AssertionError("CPU hook reached GPU materialization"),
        ) as gpu_preprocess:
            decoded = self.adapter.preprocess_input(
                [_media()],
                self.vit_config,
                processor_config=self.adapter.processor_config,
                gpu_preprocess=True,
            )
        self.assertIsInstance(decoded, vision_module._DecodedImage)
        self.assertEqual(decoded.rgb.device.type, "cpu")
        self.assertEqual(decoded.rgb.dtype, torch.uint8)
        self.assertTrue(decoded.rgb.is_contiguous())
        gpu_preprocess.assert_not_called()

        with patch.object(
            vision_module, "load_image_source", return_value=io.BytesIO(payload)
        ), patch.object(
            vision_module,
            "_preprocess_rgb_on_device",
            side_effect=AssertionError("CPU preprocess submitted CUDA work"),
        ) as gpu_preprocess:
            prepared = self.adapter.preprocess_input(
                [_media()],
                self.vit_config,
                processor_config=self.adapter.processor_config,
                gpu_preprocess=False,
            )
        self.assertIsInstance(prepared, vision_module._RuntimeImage)
        self.assertEqual(prepared.patches.device.type, "cpu")
        self.assertEqual(prepared.patches.dtype, torch.bfloat16)
        gpu_preprocess.assert_not_called()

    def test_decoded_payload_has_no_grid_span_or_token_layout(self):
        decoded = vision_module._DecodedImage(
            torch.zeros((42, 42, 3), dtype=torch.uint8)
        )
        for backend_field in (
            "start",
            "types",
            "token_types",
            "n_vit_h",
            "n_vit_w",
            "llm_h",
            "llm_w",
        ):
            self.assertFalse(hasattr(decoded, backend_field), backend_field)
        self.assertNotIn("_DecodedImage", vision_module.__all__)
        self.assertNotIn("_RuntimeImage", vision_module.__all__)

    def test_batched_embedding_returns_one_complete_owned_tensor_per_image(self):
        decoded = [
            vision_module._DecodedImage(
                torch.full((42, 42, 3), value, dtype=torch.uint8)
            )
            for value in (5, 9)
        ]

        def preprocess(rgb, target_size, patch_size, device):
            self.assertEqual(target_size, (42, 42))
            self.assertEqual(patch_size, 14)
            self.assertEqual(device.type, "cpu")
            return torch.full((9, 3, 14, 14), int(rgb[0, 0, 0]), dtype=torch.bfloat16)

        def packed(images):
            outputs = []
            for image in images:
                self.assertIsInstance(image, vision_module._RuntimeImage)
                ratio = self.adapter.processor_config.vision_downsample_ratio
                llm_h = (image.n_vit_h + ratio - 1) // ratio
                llm_w = (image.n_vit_w + ratio - 1) // ratio
                aligned = torch.full(
                    (llm_h * llm_w, 4),
                    image.patches.flatten()[0],
                    dtype=torch.bfloat16,
                )
                outputs.append(
                    self.adapter._assemble_canonical_span(aligned, llm_h, llm_w)
                )
            return outputs

        with patch.object(
            vision_module, "_preprocess_rgb_on_device", side_effect=preprocess
        ) as materialize, patch.object(
            self.adapter, "encode_images_packed", side_effect=packed
        ):
            results = self.adapter.batched_embedding(
                decoded, [MMUrlType.IMAGE, MMUrlType.IMAGE]
            )

        self.assertEqual(materialize.call_count, 2)
        self.assertEqual(len(results), 2)
        for result, value in zip(results, (5, 9)):
            embedding, position_ids, extra_inputs = result
            self.assertEqual(position_ids, [])
            self.assertEqual(extra_inputs, [])
            self.assertEqual(tuple(embedding.shape), (num_image_tokens(1, 1), 4))
            self.assertTrue(torch.equal(embedding[0], self.adapter.image_start))
            self.assertTrue(
                torch.equal(embedding[1], torch.full((4,), value, dtype=torch.bfloat16))
            )
            self.assertTrue(torch.equal(embedding[2], self.adapter.image_newline))
            self.assertTrue(torch.equal(embedding[3], self.adapter.image_end))
        self.assertNotEqual(results[0][0].data_ptr(), results[1][0].data_ptr())


if __name__ == "__main__":
    unittest.main()
