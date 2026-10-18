"""Manual real-weight byte gate for packed layout and exact-shape graph changes.

The reference is a SHA-verified, pre-change packed_vision.py snapshot supplied
by the operator. No golden is written or adjusted. Run only via Bazel GPU lock.
"""

import hashlib
import importlib.util
import json
import os
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import packed_vision
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
    preprocess_image,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
    DeepSeekV41VisionEmbedding,
    _RuntimeImage,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.packed_graph import (
    PackedVisionGraph,
)


class PackedOptimizationTest(unittest.TestCase):
    @torch.inference_mode()
    def test_real_weight_layout_and_changed_input_graph_replay(self):
        self.assertTrue(torch.cuda.is_available(), "this gate must not skip CUDA")
        self.assertEqual(os.environ.get("GPU_COUNT"), "1")
        self.assertEqual(os.environ.get("WORLD_SIZE"), "1")
        torch.cuda.set_device(0)
        os.sched_setaffinity(0, set(range(64, 72)))
        torch.set_num_threads(8)
        snapshot = Path(os.environ["DSV41_PACKED_REFERENCE"])
        digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        self.assertEqual(digest, os.environ["DSV41_PACKED_REFERENCE_SHA256"])
        spec = importlib.util.spec_from_file_location(
            "frozen_packed_reference", snapshot
        )
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        config = V41Config.from_path(os.environ["DSV41_VISUAL_CHECKPOINT"])
        model = (
            DeepSeekV41VisionEmbedding(config, device="cuda:0")
            .eval()
            .requires_grad_(False)
        )
        model.load_checkpoint(os.environ["DSV41_VISUAL_CHECKPOINT"])
        rng = np.random.default_rng(20261015)
        pixels = rng.integers(0, 256, (1440, 2560, 3), dtype=np.uint8)
        images = []
        for shift in (0, 1):
            patches, h, w, _, _ = preprocess_image(
                Image.fromarray(np.roll(pixels, shift, axis=1)), model.processor_config
            )
            images.append(_RuntimeImage(patches.cuda(), h, w))
        patches, h, w, _, _ = preprocess_image(
            Image.fromarray(pixels[:800, :600]), model.processor_config
        )
        images.append(_RuntimeImage(patches.cuda(), h, w))
        rows = []

        def forward(parts, grids):
            return packed_vision.forward_packed(
                model.vision, model.aligner, parts, grids
            )

        def spans(aligned, batch):
            return [
                model._assemble_image_span(value, image)
                for value, image in zip(aligned, batch)
            ]

        def check(actual, expected, label):
            self.assertEqual(len(actual), len(expected))
            errors, mismatches = [], []
            for got, want in zip(actual, expected):
                self.assertEqual(got.shape, want.shape)
                self.assertEqual(got.dtype, want.dtype)
                self.assertTrue(torch.isfinite(got).all())
                errors.append(float((got.float() - want.float()).abs().max()))
                mismatches.append(
                    int(
                        (
                            got.contiguous().view(torch.uint8)
                            != want.contiguous().view(torch.uint8)
                        ).sum()
                    )
                )
            record = {
                "case": label,
                "max_abs_error": max(errors),
                "different_bytes": sum(mismatches),
            }
            rows.append(record)
            print(json.dumps(record), flush=True)
            self.assertEqual(sum(mismatches), 0, label)

        for order in ([0], [0, 1], [0, 2, 1], [0, 1] * 64):
            batch = [images[i] for i in order]
            parts = [image.patches for image in batch]
            grids = [(image.n_vit_h, image.n_vit_w) for image in batch]
            expected = spans(
                reference.forward_packed(model.vision, model.aligner, parts, grids),
                batch,
            )
            actual = spans(forward(parts, grids), batch)
            check(actual, expected, f"layout_{len(order)}")
            if len(order) == 128:
                graph = PackedVisionGraph(128)
                actual_graph = graph.run(
                    parts, grids, forward, lambda values: spans(values, batch)
                )
                check(actual_graph, expected, "graph_capture_full128")
                retained = [value.clone() for value in actual_graph]
                changed = list(reversed(batch))
                changed_parts = [image.patches for image in changed]
                changed_expected = spans(forward(changed_parts, grids), changed)
                other_stream = torch.cuda.Stream()
                other_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(other_stream):
                    replay = graph.run(
                        changed_parts,
                        grids,
                        forward,
                        lambda values: spans(values, changed),
                    )
                torch.cuda.current_stream().wait_stream(other_stream)
                check(replay, changed_expected, "graph_changed_input_cross_stream")
                check(actual_graph, retained, "previous_request_owned_output")
                self.assertFalse(
                    torch.equal(replay[0], actual_graph[0]),
                    "different media must not share stale graph data",
                )
                tail = graph.run(
                    parts[:2],
                    grids[:2],
                    forward,
                    lambda values: spans(values, batch[:2]),
                )
                check(
                    tail, spans(forward(parts[:2], grids[:2]), batch[:2]), "eager_tail"
                )
                self.assertEqual(graph.stats()["captures"], 1)
                self.assertEqual(graph.stats()["replays"], 2)
                self.assertEqual(graph.stats()["eager"], 1)
                print(
                    json.dumps(
                        {
                            "graph": graph.stats(),
                            "peak_bytes": torch.cuda.max_memory_allocated(),
                        }
                    ),
                    flush=True,
                )
        destination = Path(os.environ["DSV41_OPTIMIZATION_GATE_OUTPUT"])
        with destination.open("x") as stream:
            json.dump(
                {"status": "PASS", "reference_sha256": digest, "cases": rows},
                stream,
                indent=2,
            )


if __name__ == "__main__":
    unittest.main()
