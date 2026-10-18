"""New RGB transfer/graph paths vs unchanged eager path; never writes golden."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
    _rgb_tensor,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
    DeepSeekV41VisionEmbedding,
    _DecodedImage,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.packed_graph import (
    PackedVisionGraph,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType


class PipelineGPUExactTest(unittest.TestCase):
    @torch.inference_mode()
    def test_tuning_large_batch_and_grouped_cpu_inputs(self):
        """New 192-image eager/graph paths versus unchanged 126-image chunks."""
        self.assertTrue(torch.cuda.is_available(), "GPU gate must not skip")
        self.assertEqual(os.environ.get("GPU_COUNT"), "1")
        self.assertEqual(os.environ.get("WORLD_SIZE"), "1")
        torch.cuda.set_device(0)
        os.sched_setaffinity(0, set(range(64, 72)))
        torch.set_num_threads(8)
        checkpoint = os.environ["DSV41_VISUAL_CHECKPOINT"]
        model = DeepSeekV41VisionEmbedding(
            V41Config.from_path(checkpoint), device="cuda:0"
        )
        model.eval().requires_grad_(False)
        model.load_checkpoint(checkpoint)
        self.assertTrue(model._gpu_preprocess_enabled)
        self.assertTrue(model._packed_forward_enabled)
        self.assertIsNone(model._packed_graph)
        self.assertEqual(
            (model._packed_max_images, model._packed_max_patches), (128, 1061888)
        )
        paths = json.loads(os.environ["DSV41_JPEG_PATHS"])
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(_DecodedImage(_rgb_tensor(image)))
        self.assertEqual(len(images), 2)
        media = [
            SimpleNamespace(
                url=path,
                mm_type=MMUrlType.IMAGE,
                mm_preprocess_config=SimpleNamespace(crop_positions=[]),
            )
            for path in paths
        ]
        decoded_group = model.preprocess_input(
            media,
            SimpleNamespace(download_headers="", mm_image_max_file_size_kb=-1),
            **model.get_preprocess_params(),
        )
        for got, want in zip(decoded_group, images):
            self.assertTrue(torch.equal(got.rgb, want.rgb))
        count = 192
        sequence = images * (count // 2)
        rows = []

        def baseline(items):
            graph = model._packed_graph
            cap = model._packed_max_images, model._packed_max_patches
            model._packed_graph = None
            model._packed_max_images, model._packed_max_patches = 128, 1061888
            try:
                return [
                    value[0].cpu()
                    for value in model.batched_embedding(
                        items, [MMUrlType.IMAGE] * len(items)
                    )
                ]
            finally:
                model._packed_max_images, model._packed_max_patches = cap
                model._packed_graph = graph

        def check(got, expected, label):
            flattened = []
            for item in got:
                flattened.extend(item[0] if isinstance(item[0], list) else [item[0]])
            self.assertEqual(len(flattened), len(expected))
            different = 0
            for actual, want in zip(flattened, expected):
                actual = actual.cpu()
                self.assertEqual(actual.shape, want.shape)
                self.assertEqual(actual.dtype, want.dtype)
                self.assertTrue(torch.isfinite(actual).all())
                different += int(
                    (
                        actual.contiguous().view(torch.uint8)
                        != want.contiguous().view(torch.uint8)
                    ).sum()
                )
            rows.append({"case": label, "different_bytes": different})
            print(json.dumps(rows[-1]), flush=True)
            self.assertEqual(different, 0, label)

        expected = baseline(sequence)
        model._packed_max_images, model._packed_max_patches = count, count * 8400
        grouped = [decoded_group] * (count // 2)
        staged = model.prepare_embedding_batch(
            grouped, [MMUrlType.IMAGE] * len(grouped)
        )
        self.assertIsNotNone(staged)
        try:
            staged.wait()
            got = model.batched_embedding(
                staged.data_list, [MMUrlType.IMAGE] * len(grouped)
            )
            check(got, expected, "CPU_group2_upload_eager192_vs_old126_chunks")
        finally:
            staged.close()
        del got, staged
        model._packed_graph = PackedVisionGraph(count, count * 8400)
        self.assertEqual(
            model._packed_graph.effective_batch_images(model._vit_grid(images[0])),
            count,
        )
        got = model.batched_embedding(grouped, [MMUrlType.IMAGE] * len(grouped))
        check(got, expected, "CPU_group2_graph192_vs_old126_chunks")
        changed = list(reversed(sequence))
        changed_expected = baseline(changed)
        changed_groups = [list(reversed(decoded_group))] * (count // 2)
        staged = model.prepare_embedding_batch(
            changed_groups, [MMUrlType.IMAGE] * len(changed_groups)
        )
        self.assertIsNotNone(staged)
        try:
            staged.wait()
            replay = model.batched_embedding(
                staged.data_list, [MMUrlType.IMAGE] * len(changed_groups)
            )
            check(replay, changed_expected, "CPU_group2_upload_graph192_changed_order")
            check(got, expected, "prior_graph_outputs_owned")
        finally:
            staged.close()
        self.assertEqual(model.packed_graph_stats()["captures"], 1)
        self.assertEqual(model.packed_graph_stats()["replays"], 2)
        self.assertEqual(model.input_transfer_stats()["budget_fallbacks"], 0)
        with Path(os.environ["DSV41_PIPELINE_GATE_OUTPUT"]).open("x") as stream:
            json.dump(
                {
                    "status": "PASS",
                    "cases": rows,
                    "graph": model.packed_graph_stats(),
                    "transfer": model.input_transfer_stats(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
                stream,
                indent=2,
            )

    @torch.inference_mode()
    def test_real_weights_rgb_handoff_and_patch_limited_graph(self):
        self.assertTrue(torch.cuda.is_available(), "GPU gate must not skip")
        self.assertEqual(os.environ.get("GPU_COUNT"), "1")
        self.assertEqual(os.environ.get("WORLD_SIZE"), "1")
        torch.cuda.set_device(0)
        os.sched_setaffinity(0, set(range(64, 72)))
        torch.set_num_threads(8)
        checkpoint = os.environ["DSV41_VISUAL_CHECKPOINT"]
        model = DeepSeekV41VisionEmbedding(
            V41Config.from_path(checkpoint), device="cuda:0"
        )
        model.eval().requires_grad_(False)
        model.load_checkpoint(checkpoint)
        self.assertTrue(model._gpu_preprocess_enabled)
        self.assertTrue(model._packed_forward_enabled)
        self.assertIsNone(
            model._packed_graph, "start with the unchanged eager reference"
        )
        self.assertEqual(model._packed_max_images, 128)
        self.assertEqual(model._packed_max_patches, 1061888)
        images = []
        for path in json.loads(os.environ["DSV41_JPEG_PATHS"]):
            with Image.open(path) as image:
                images.append(_DecodedImage(_rgb_tensor(image)))
        self.assertEqual(len(images), 2)
        images.append(_DecodedImage(images[0].rgb[:192, :224].contiguous()))
        rows = []

        def reference(batch):
            return [
                value[0].cpu()
                for value in model.batched_embedding(
                    batch, [MMUrlType.IMAGE] * len(batch)
                )
            ]

        def check(got, want, label):
            self.assertEqual(len(got), len(want))
            mismatches = 0
            for actual, expected in zip(got, want):
                actual = actual[0].cpu()
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(torch.isfinite(actual).all())
                mismatches += int(
                    (
                        actual.contiguous().view(torch.uint8)
                        != expected.contiguous().view(torch.uint8)
                    ).sum()
                )
            rows.append({"case": label, "different_bytes": mismatches})
            print(json.dumps(rows[-1]), flush=True)
            self.assertEqual(mismatches, 0, label)

        for order in ([0, 1], [0, 2, 1], [0, 1] * 63):
            batch = [images[index] for index in order]
            expected = reference(batch)
            staged = model.prepare_embedding_batch(
                batch, [MMUrlType.IMAGE] * len(batch)
            )
            self.assertIsNotNone(staged, "new transfer path must actually execute")
            other = torch.cuda.Stream()
            try:
                with torch.cuda.stream(other):
                    staged.wait()
                    got = model.batched_embedding(
                        staged.data_list, [MMUrlType.IMAGE] * len(batch)
                    )
                # close waits H2D, not forward; record_stream protects device inputs.
                staged.close()
                torch.cuda.current_stream().wait_stream(other)
                check(got, expected, f"upload_cross_stream_B{len(batch)}")
            finally:
                staged.close()

        # Cancellation before a consumer waits must safely retire pinned storage.
        cancelled = model.prepare_embedding_batch(images[:2], [MMUrlType.IMAGE] * 2)
        cancelled.close()
        cancelled.close()

        batch = [images[i] for i in ([0, 1] * 63)]
        expected = reference(batch)
        grids = [model._vit_grid(image) for image in batch]
        self.assertTrue(all(grid == grids[0] for grid in grids))
        model._packed_graph = PackedVisionGraph(128, 1061888)
        self.assertEqual(model._packed_graph.effective_batch_images(grids[0]), 126)
        got = model.batched_embedding(batch, [MMUrlType.IMAGE] * len(batch))
        check(got, expected, "graph_patch_limited126")
        retained = [value[0].cpu() for value in got]
        changed = list(reversed(batch))
        # The eager reference remains independently produced, not a new golden.
        graph = model._packed_graph
        model._packed_graph = None
        changed_expected = reference(changed)
        model._packed_graph = graph
        staged = model.prepare_embedding_batch(
            changed, [MMUrlType.IMAGE] * len(changed)
        )
        try:
            staged.wait()
            replay = model.batched_embedding(
                staged.data_list, [MMUrlType.IMAGE] * len(changed)
            )
            check(replay, changed_expected, "upload_plus_graph_changed_input126")
            check(got, retained, "prior_output_not_overwritten")
        finally:
            staged.close()
        model._packed_graph = None
        tail_expected = reference(images[:2])
        model._packed_graph = graph
        tail = model.batched_embedding(images[:2], [MMUrlType.IMAGE] * 2)
        check(tail, tail_expected, "graph_tail_eager")
        stats = model.packed_graph_stats()
        self.assertEqual(stats["captures"], 1)
        self.assertEqual(stats["replays"], 2)
        self.assertEqual(stats["eager"], 1)
        report = {
            "status": "PASS",
            "cases": rows,
            "graph": stats,
            "transfer": model.input_transfer_stats(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        with Path(os.environ["DSV41_PIPELINE_GATE_OUTPUT"]).open("x") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    unittest.main()
