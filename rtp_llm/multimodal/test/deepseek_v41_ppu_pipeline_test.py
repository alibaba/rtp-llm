"""Explicit real-weight PPU RGB -> packed ViT/projector -> Graph replay gate."""

import concurrent.futures
import os
import threading
import unittest
from pathlib import Path

import torch
from PIL import Image

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.device.device_type import is_ppu
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
    _rgb_tensor,
    num_image_tokens,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
    DeepSeekV41VisionEmbedding,
    _DecodedImage,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_VISION_PPU_DEVICE") == "1", "explicit PPU device run"
)
class RealWeightPPUPipelineTest(unittest.TestCase):
    @torch.inference_mode()
    def test_rgb_mixed_images_graph_live_inputs_and_reload(self):
        self.assertTrue(is_ppu(), "this gate requires a real PPU")
        checkpoint = os.environ["DSV41_VISUAL_CHECKPOINT"]
        model = (
            DeepSeekV41VisionEmbedding(
                V41Config.from_path(checkpoint),
                device="cuda:0",
                max_batch_images=4,
                graph_max_entries=2,
                graph_max_bytes=8 * 1024**3,
                cache_reserved_bytes=0,
            )
            .eval()
            .requires_grad_(False)
        )
        loaded = model.load_checkpoint(checkpoint)
        self.assertGreater(loaded["weight_tensors"], 0)
        self.assertEqual(loaded["fp32_norm_tensors"], 0)
        self.assertTrue(
            all(b.attn._attention_backend == "fa2" for b in model.vision.blocks)
        )
        fixture = Path(__file__).with_name("testdata") / "qwen2_vl/1.jpg"
        with Image.open(fixture) as source:
            images = [
                _DecodedImage(_rgb_tensor(source.resize(size).convert("RGB")))
                for size in ((512, 512), (777, 513))
            ]
        changed = [_DecodedImage((255 - image.rgb).contiguous()) for image in images]
        types = [MMUrlType.IMAGE] * len(images)
        graph = model._packed_graph

        def eager(inputs):
            model._packed_graph = None
            try:
                return [
                    entry[0]
                    for entry in model.batched_embedding(
                        inputs, [MMUrlType.IMAGE] * len(inputs)
                    )
                ]
            finally:
                model._packed_graph = graph

        expected = eager(images)
        expected_changed = eager(changed)
        changed_second = [images[0], changed[1]]
        expected_second = eager(changed_second)
        torch.testing.assert_close(expected_second[0], expected[0], rtol=0, atol=0)
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(expected, expected_changed))
        )
        for image, output in zip(images, expected):
            height, width = model._vit_grid(image)
            ratio = model.processor_config.vision_downsample_ratio
            count = num_image_tokens(
                (height + ratio - 1) // ratio, (width + ratio - 1) // ratio
            )
            self.assertEqual(tuple(output.shape), (count, model.image_start.numel()))
            self.assertEqual(output.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(output[0], model.image_start, rtol=0, atol=0)
            torch.testing.assert_close(output[-1], model.image_end, rtol=0, atol=0)

        for _ in range(3):
            model.batched_embedding(images, types)
            if model.run_embedding_idle_work():
                break
        self.assertGreater(
            graph.captures, 0, "the real GPU-owner idle hook must capture"
        )
        first = [entry[0] for entry in model.batched_embedding(images, types)]
        saved = [value.clone() for value in first]
        second = [entry[0] for entry in model.batched_embedding(changed, types)]
        for got, want, old, original in zip(second, expected_changed, first, saved):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
            torch.testing.assert_close(old, original, rtol=0, atol=0)
        for got, want in zip(first, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        third = [entry[0] for entry in model.batched_embedding(changed_second, types)]
        for got, want in zip(third, expected_second):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        torch.testing.assert_close(third[0], first[0], rtol=0, atol=0)
        self.assertGreaterEqual(graph.replays, 3)

        # Different producer streams share one graph's scratch storage. The
        # cache must wait on its prior completion event and return owned spans.
        barrier = threading.Barrier(2)

        def concurrent_replay(inputs):
            torch.cuda.set_device(0)
            with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
                barrier.wait(timeout=30)
                output = [entry[0] for entry in model.batched_embedding(inputs, types)]
                torch.cuda.current_stream().synchronize()
                return output

        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(concurrent_replay, inputs)
                for inputs in (images, changed_second)
            ]
            parallel = [future.result(timeout=120) for future in futures]
        for outputs, reference in zip(parallel, (expected, expected_second)):
            for got, want in zip(outputs, reference):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
        for old, original in zip(first, saved):
            torch.testing.assert_close(old, original, rtol=0, atol=0)

        # Reverse heterogeneous input order without creating a second ordered
        # shape. Both real patch rows and per-image outputs must follow their
        # own grid, with caller ordering restored after the canonical replay.
        captures, replays = graph.captures, graph.replays
        for inputs in (images[::-1], changed[::-1], changed_second[::-1]):
            reference = eager(inputs)
            outputs = [entry[0] for entry in model.batched_embedding(inputs, types)]
            for got, want in zip(outputs, reference):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
        self.assertEqual(graph.captures, captures)
        self.assertEqual(graph.replays, replays + 3)

        # Equal grids retain stable content order, while arbitrary permutations
        # of the same grid multiset reuse the same four-image graph. Duplicate
        # inputs must still yield independently owned request spans.
        mixed = [images[0], images[1], changed[0], changed[1]]
        mixed_types = [MMUrlType.IMAGE] * len(mixed)
        mixed_reference = eager(mixed)
        model.batched_embedding(mixed, mixed_types)
        self.assertTrue(model.run_embedding_idle_work())
        self.assertEqual(graph.captures, captures + 1)
        captures, replays = graph.captures, graph.replays
        mixed_first = [
            entry[0] for entry in model.batched_embedding(mixed, mixed_types)
        ]
        for got, want in zip(mixed_first, mixed_reference):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        retained_outputs = [(mixed_first, [value.clone() for value in mixed_first])]
        variants = (
            mixed[::-1],
            [changed[0], images[1], images[0], changed[1]],
            [images[1], images[0], images[1], changed[0]],
        )
        for inputs in variants:
            reference = eager(inputs)
            outputs = [
                entry[0] for entry in model.batched_embedding(inputs, mixed_types)
            ]
            for got, want in zip(outputs, reference):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            retained_outputs.append((outputs, [value.clone() for value in outputs]))
        # Rewrite graph scratch once more after publishing the duplicate spans.
        following = [entry[0] for entry in model.batched_embedding(mixed, mixed_types)]
        for got, want in zip(following, mixed_reference):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        self.assertEqual(graph.captures, captures)
        self.assertEqual(graph.replays, replays + 2 + len(variants))
        duplicate_outputs = retained_outputs[-1][0]
        self.assertNotEqual(
            duplicate_outputs[0].data_ptr(), duplicate_outputs[2].data_ptr()
        )
        for outputs, originals in retained_outputs + [(first, saved)]:
            for old, original in zip(outputs, originals):
                torch.testing.assert_close(old, original, rtol=0, atol=0)

        # Exercise actual graph destruction after a hotter different shape
        # replaces the only entry; earlier request outputs must remain owned.
        graph.clear()
        graph.max_entries = 1
        model.batched_embedding(images[:1], types[:1])
        self.assertTrue(model.run_embedding_idle_work())
        retained = model.batched_embedding(images[:1], types[:1])[0][0]
        retained_copy = retained.clone()
        for _ in range(40):
            model.batched_embedding(images[:1], types[:1])
        evictions = graph.evictions
        for _ in range(graph.stats()["observation_window"] // 2 + 1):
            model.batched_embedding(images[1:], types[1:])
        self.assertTrue(model.run_embedding_idle_work())
        self.assertEqual(graph.evictions, evictions + 1)
        self.assertEqual(graph.stats()["entries"], 1)
        replacement = model.batched_embedding(images[1:], types[1:])[0][0]
        torch.testing.assert_close(replacement, expected[1], rtol=0, atol=0)
        torch.testing.assert_close(retained, retained_copy, rtol=0, atol=0)
        model.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(graph.stats()["entries"], 0)
        self.assertEqual(graph.stats()["pending"], 0)


if __name__ == "__main__":
    unittest.main()
