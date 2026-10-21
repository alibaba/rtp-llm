"""Real CUDA/real-weight multi-shape graph byte gate; no golden updates."""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import deepseek_vision
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
    DeepSeekV41VisionEmbedding,
    _DecodedImage,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import _rgb_tensor
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_vision import (
    _vision_cos_sin,
)


class GraphCacheGPUExactTest(unittest.TestCase):
    @torch.inference_mode()
    def test_real_weights_multishape_replay_and_eviction(self):
        self.assertTrue(torch.cuda.is_available(), "real GPU gate must not skip")
        self.assertEqual(os.environ.get("GPU_COUNT"), "1")
        self.assertEqual(os.environ.get("WORLD_SIZE"), "1")
        torch.cuda.set_device(0)
        os.sched_setaffinity(0, set(range(64, 72)))
        torch.set_num_threads(8)
        torch.manual_seed(20261021)
        # Match the installed, pinned optional backends used by the existing
        # production-path benchmark; never replace Bazel's selected torch.
        plugin = os.environ.get("V41_BENCH_OPTIONAL_PLUGIN_PATH")
        if plugin:
            plugin = Path(plugin).resolve(strict=True)
            sys.path.append(str(plugin))
            sys.path.append(str(plugin / "nvidia_cutlass_dsl/dsl_packages"))
            self.assertTrue((plugin / "flash_attn/cute/interface.py").is_file())
            import flash_attn

            for name, module in tuple(sys.modules.items()):
                if name.startswith("flash_attn.cute"):
                    origin = getattr(module, "__file__", None)
                    if origin:
                        self.assertTrue(
                            Path(origin).resolve().is_relative_to(plugin / "flash_attn")
                        )
            flash_attn.__path__ = [str(plugin / "flash_attn")] + list(
                flash_attn.__path__
            )
        quack = os.environ.get("V41_BENCH_QUACK_SOURCE")
        if quack:
            sys.path.insert(0, str(Path(quack).resolve(strict=True)))
        checkpoint = os.environ["DSV41_VISUAL_CHECKPOINT"]
        model = (
            DeepSeekV41VisionEmbedding(
                V41Config.from_path(checkpoint),
                device="cuda:0",
                max_batch_images=256,
                graph_max_entries=1,
                graph_max_bytes=0,
            )
            .eval()
            .requires_grad_(False)
        )
        loaded = model.load_checkpoint(checkpoint)
        self.assertGreater(loaded["weight_tensors"], 0)
        self.assertEqual(loaded["fp32_norm_tensors"], 0)
        self.assertGreater(loaded["bf16_norm_tensors"], 0)
        model.get_batch_work_budget(256)
        cache = model._packed_graph
        context = model._graph_context()
        patch_dim = model.vision.patch_embed.proj.in_features
        print(
            json.dumps(
                {
                    "weights": loaded,
                    "context": context,
                    "device": str(torch.cuda.get_device_properties(0)),
                }
            ),
            flush=True,
        )

        def parts_for(grids):
            return [
                torch.randn(h * w, patch_dim, device="cuda", dtype=torch.bfloat16)
                for h, w in grids
            ]

        def own(values):
            return [v.clone() for v in values]

        def check(got, want):
            self.assertEqual(len(got), len(want))
            for a, b in zip(got, want):
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.dtype, b.dtype)
                self.assertTrue(torch.isfinite(a).all())
                self.assertTrue(
                    torch.equal(
                        a.contiguous().view(torch.uint8),
                        b.contiguous().view(torch.uint8),
                    )
                )

        # A frequently reused full batch must survive a one-off tail even when
        # the exact-shape cache has room for only one graph.
        full_grids = [(3, 3)] * 128
        full_parts = parts_for(full_grids)
        real_fused_helper = deepseek_vision.apply_vision_residual_rmsnorm

        def checked_fused_helper(*args, **kwargs):
            result = real_fused_helper(*args, **kwargs)
            self.assertIsNotNone(result, "supported packed ViT call must be fused")
            return result

        with patch.object(
            deepseek_vision,
            "apply_vision_residual_rmsnorm",
            side_effect=checked_fused_helper,
        ) as fused_helper:
            full_expected = own(model._forward_packed(full_parts, full_grids))
        self.assertGreater(fused_helper.call_count, 0)
        with patch.object(
            deepseek_vision, "apply_vision_residual_rmsnorm", return_value=None
        ):
            unfused = own(model._forward_packed(full_parts, full_grids))
        relative_l2 = []
        cosine = []
        for fused_value, unfused_value in zip(full_expected, unfused):
            fused_fp32 = fused_value.float().flatten()
            unfused_fp32 = unfused_value.float().flatten()
            relative_l2.append(
                torch.linalg.vector_norm(fused_fp32 - unfused_fp32)
                / torch.linalg.vector_norm(unfused_fp32).clamp_min(
                    torch.finfo(torch.float32).tiny
                )
            )
            cosine.append(
                torch.nn.functional.cosine_similarity(
                    fused_fp32, unfused_fp32, dim=0
                )
            )
        relative_l2_values = torch.stack(relative_l2)
        cosine_values = torch.stack(cosine)
        self.assertTrue(torch.isfinite(relative_l2_values).all())
        self.assertTrue(torch.isfinite(cosine_values).all())
        precision = {
            "max_relative_l2": float(relative_l2_values.max()),
            "min_cosine": float(cosine_values.min()),
            "fused_helper_calls": fused_helper.call_count,
        }
        print(json.dumps({"fused_vs_unfused": precision}), flush=True)
        # User-approved acceptance on 2026-10-21: >=99% cosine similarity
        # and <=1% relative L2. This is embedding similarity, not task accuracy.
        self.assertLessEqual(precision["max_relative_l2"], 1e-2)
        self.assertGreaterEqual(precision["min_cosine"], 0.99)
        del unfused, fused_fp32, unfused_fp32
        full_cold = cache.run(
            full_parts,
            full_grids,
            model._forward_packed,
            own,
            context=context,
        )
        check(full_cold, full_expected)
        self.assertTrue(model.run_embedding_idle_work())
        self.assertEqual(cache.stats()["captures"], 1)

        # Replay twice with changed storage/content and after evicting the
        # global rotary LRU. Each graph owns its captured rotary tables.
        for _ in range(2):
            _vision_cos_sin.cache_clear()
            changed = parts_for(full_grids)
            changed_expected = own(model._forward_packed(changed, full_grids))
            replay = cache.run(
                changed,
                full_grids,
                model._forward_packed,
                own,
                context=context,
            )
            check(replay, changed_expected)

        cached_grids = cache.stats()["cached_grids"]
        tail_grids = [(3, 3)] * 4
        tail_parts = parts_for(tail_grids)
        tail_expected = own(model._forward_packed(tail_parts, tail_grids))
        tail = cache.run(
            tail_parts,
            tail_grids,
            model._forward_packed,
            own,
            context=context,
        )
        check(tail, tail_expected)
        self.assertFalse(model.run_embedding_idle_work())
        self.assertEqual(cache.stats()["cached_grids"], cached_grids)
        self.assertEqual(cache.stats()["entries"], 1)
        self.assertEqual(cache.stats()["captures"], 1)
        self.assertEqual(cache.stats()["evictions"], 0)
        self.assertEqual(cache.stats()["replays"], 2)
        self.assertLessEqual(cache.stats()["charged_bytes"], cache.stats()["max_bytes"])
        print(json.dumps({"byte_equal": True, "graph": cache.stats()}), flush=True)
        cache.clear()
        self.assertEqual(cache.stats()["entries"], 0)

        # Real JPEGs, real image grids and complete assembled embeddings.
        # This tests numerical changes at an actual 128-image packed workload;
        # the tiny-grid graph test above only exercises cache behavior.
        paths = json.loads(os.environ["DSV41_JPEG_PATHS"])
        decoded = []
        for path in paths:
            with Image.open(path) as image:
                decoded.append(_DecodedImage(_rgb_tensor(image)))
        runtime = [model._materialize_image(image) for image in decoded]
        self.assertEqual(len(runtime), 2)
        model._packed_graph = None
        for count in (2, 128):
            batch = runtime * (count // 2)
            with patch.object(
                deepseek_vision, "apply_vision_residual_rmsnorm", return_value=None
            ):
                expected = [value.cpu() for value in model.encode_images_packed(batch)]
            with patch.object(
                deepseek_vision, "apply_vision_residual_rmsnorm",
                side_effect=checked_fused_helper,
            ) as real_helper:
                actual = model.encode_images_packed(batch)
            self.assertGreater(real_helper.call_count, 0)
            metrics = []
            for output, reference in zip(actual, expected):
                output = output.cpu()
                self.assertEqual(output.shape, reference.shape)
                self.assertEqual(output.dtype, reference.dtype)
                a, b = output.float().flatten(), reference.float().flatten()
                self.assertTrue(torch.isfinite(a).all())
                self.assertTrue(torch.isfinite(b).all())
                delta = a - b
                error2 = delta.double().square().sum()
                reference2 = b.double().square().sum()
                actual2 = a.double().square().sum()
                metrics.append({
                    "relative_l2": float((error2 / reference2.clamp_min(1e-300)).sqrt()),
                    "cosine": float((a.double() * b.double()).sum() /
                                    (actual2 * reference2).sqrt().clamp_min(1e-300)),
                    "max_abs": float(delta.abs().max()),
                })
            result = {
                "batch_images": count,
                "grids": [(image.n_vit_h, image.n_vit_w) for image in batch],
                "fused_calls": real_helper.call_count,
                "max_relative_l2": max(row["relative_l2"] for row in metrics),
                "min_cosine": min(row["cosine"] for row in metrics),
                "max_abs": max(row["max_abs"] for row in metrics),
            }
            print(json.dumps({"real_jpeg_fused_vs_separate": result}), flush=True)
            self.assertLessEqual(result["max_relative_l2"], 1e-2)
            self.assertGreaterEqual(result["min_cosine"], 0.99)
            del actual, expected, output, reference, a, b, delta


if __name__ == "__main__":
    unittest.main()
