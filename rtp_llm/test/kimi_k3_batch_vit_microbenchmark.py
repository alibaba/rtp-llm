"""Manual checkpoint-backed K3 batched ViT versus per-image microbenchmark."""

import json
import math
import os
import statistics
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.multimodal.multimodal_mixin_factory import MultimodalMixinFactory
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_config import (
    configure_kimi_k3_multimodal,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_vit import (
    mm_projector_forward,
)


def _summary(samples_ms):
    ordered = sorted(samples_ms)
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "samples_ms": samples_ms,
    }


def _measure_pair(serial, batched, *, cuda_event, warmup=4, repeats=16):
    for _ in range(warmup):
        serial()
        batched()
    torch.cuda.synchronize()

    samples = {"serial": [], "batched": []}
    paths = (("serial", serial), ("batched", batched))
    for iteration in range(repeats):
        # Alternate order so clocks, caches, and thermal drift affect both.
        for label, fn in paths if iteration % 2 == 0 else reversed(paths):
            torch.cuda.synchronize()
            if cuda_event:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                elapsed_ms = start.elapsed_time(end)
            else:
                start_ns = time.perf_counter_ns()
                fn()
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
            samples[label].append(elapsed_ms)

    result = {label: _summary(values) for label, values in samples.items()}
    result["median_speedup"] = (
        result["serial"]["median_ms"] / result["batched"]["median_ms"]
    )
    result["median_time_reduction_pct"] = 100 * (
        1 - result["batched"]["median_ms"] / result["serial"]["median_ms"]
    )
    result["warmup"] = warmup
    result["repeats"] = repeats
    return result


def _peak_extra_bytes(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return max(0, torch.cuda.max_memory_allocated() - baseline)


@unittest.skipUnless(torch.cuda.is_available(), "K3 ViT benchmark requires CUDA")
class KimiK3BatchVitMicrobenchmark(unittest.TestCase):
    def test_batch_against_serial(self):
        checkpoint = Path(os.environ["K3_CKPT_PATH"])
        top_config = json.loads((checkpoint / "config.json").read_text())
        model_config = ModelConfig()
        model_config.model_type = "kimi_k3"
        model_config.ckpt_path = str(checkpoint)
        model_config.data_type = "bf16"
        configure_kimi_k3_multimodal(model_config, top_config)
        engine_config = SimpleNamespace(
            load_config=SimpleNamespace(load_method=LoadMethod.AUTO)
        )
        mixin = MultimodalMixinFactory._create_multimodal_mixin(
            model_config, engine_config, VitConfig(), device="cuda:0"
        )
        embedding = mixin.mm_part
        embedding.vision_tower.eval()
        embedding.mm_projector.eval()

        cases = {
            "small_pair": ((28, 28), (56, 28)),
            "medium_pair": ((448, 224), (448, 224)),
        }
        for name, sizes in cases.items():
            with self.subTest(name=name), torch.inference_mode():
                images = [
                    Image.new("RGB", size, (64 + 32 * index, 128, 32))
                    for index, size in enumerate(sizes)
                ]

                def full_serial():
                    return [embedding.image_embedding([image])[0] for image in images]

                def full_batched():
                    return embedding.image_embedding(images)

                serial_outputs = full_serial()
                batched_outputs = full_batched()
                self.assertEqual(len(serial_outputs), len(batched_outputs))
                max_error = 0.0
                mean_errors = []
                for serial_output, batched_output in zip(
                    serial_outputs, batched_outputs
                ):
                    self.assertEqual(serial_output.shape, batched_output.shape)
                    error = (serial_output.float() - batched_output.float()).abs()
                    max_error = max(max_error, error.max().item())
                    mean_errors.append(error.mean().item())
                    torch.testing.assert_close(
                        batched_output, serial_output, rtol=0.02, atol=0.06
                    )

                per_image = [
                    embedding.image_processor.preprocess(
                        {"type": "image", "image": image}, return_tensors="pt"
                    )
                    for image in images
                ]
                batch = embedding.image_processor.preprocess(
                    [{"type": "image", "image": image} for image in images],
                    return_tensors="pt",
                )
                device = embedding._device
                dtype = embedding._data_type
                serial_pixels = [
                    item["pixel_values"].to(device=device, dtype=dtype)
                    for item in per_image
                ]
                batch_pixels = batch["pixel_values"].to(device=device, dtype=dtype)

                def gpu_serial():
                    return [
                        mm_projector_forward(
                            embedding.mm_projector,
                            embedding.vision_tower(pixels, item["grid_thws"]),
                        )[0]
                        for pixels, item in zip(serial_pixels, per_image)
                    ]

                def gpu_batched():
                    return mm_projector_forward(
                        embedding.mm_projector,
                        embedding.vision_tower(batch_pixels, batch["grid_thws"]),
                    )

                for actual, expected in zip(gpu_batched(), batched_outputs):
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.06)

                gpu_timing = _measure_pair(gpu_serial, gpu_batched, cuda_event=True)
                full_timing = _measure_pair(full_serial, full_batched, cuda_event=False)
                print(
                    json.dumps(
                        {
                            "case": name,
                            "image_sizes": sizes,
                            "patch_counts": [
                                int(item["pixel_values"].shape[0])
                                for item in per_image
                            ],
                            "device": torch.cuda.get_device_name(),
                            "torch": torch.__version__,
                            "dtype": str(dtype),
                            "max_abs_error": max_error,
                            "mean_abs_errors": mean_errors,
                            "gpu_forward_projector_cuda_event_ms": gpu_timing,
                            "image_embedding_wall_ms": full_timing,
                            "peak_extra_bytes": {
                                "serial": _peak_extra_bytes(full_serial),
                                "batched": _peak_extra_bytes(full_batched),
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main()
