import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch

from rtp_llm.ops import VitSeparation
from rtp_llm.utils.mm_process_engine import MMProcessEngine


class FakeVision:
    _device = "cpu"

    def __init__(self):
        self.batches = []
        self.legacy_calls = 0

    def preprocess_embedding(self, url, mm_type, download_headers="", configs=None):
        return SimpleNamespace(
            num_patches=2,
            output_tokens=2,
            value=int(url),
            phase=configs.image_block_start_mod4,
        )

    def batch_embedding(self, images):
        self.batches.append([(image.value, image.phase) for image in images])
        return [
            (torch.tensor([[image.value, image.phase]] * 2, dtype=torch.float32), None)
            for image in images
        ]

    def mm_embedding(self, url, **kwargs):
        self.legacy_calls += 1
        return torch.full((2, 2), int(url), dtype=torch.float32), None


class MMProcessEngineBatchTest(unittest.TestCase):
    def engine(self, role, cache_size=0):
        vision = FakeVision()
        model = SimpleNamespace(
            mm_part=vision,
            model_config=SimpleNamespace(
                mm_model_config=SimpleNamespace(mm_position_ids_style=0),
                mm_related_params=SimpleNamespace(support_batch=False),
                max_seq_len=256,
                hidden_size=2,
                compute_dtype=torch.float32,
            ),
        )
        config = SimpleNamespace(
            vit_separation=role,
            download_headers="",
            vit_batch_wait_ms=50,
            vit_max_batch_images=4,
            vit_max_batch_patches=16,
            mm_cache_item_num=cache_size,
        )
        engine = MMProcessEngine(model, config)
        self.addCleanup(engine.stop)
        return engine, vision

    def test_role_batches_requests_and_preserves_each_image_phase(self):
        engine, vision = self.engine(VitSeparation.VIT_SEPARATION_ROLE)
        barrier = threading.Barrier(4)

        def request(i):
            barrier.wait()
            configs = [
                [-1, -1, -1, -1, -1, -1, -1, phase] for phase in (i, (i + 1) % 4)
            ]
            return engine.submit(
                [str(i * 2), str(i * 2 + 1)], preprocess_configs=configs
            )

        with ThreadPoolExecutor(4) as executor:
            results = list(executor.map(request, range(4)))
        for i, result in enumerate(results):
            self.assertIsNone(result.position_ids)
            self.assertEqual(result.embeddings[0][0].tolist(), [i * 2, i])
            self.assertEqual(result.embeddings[1][0].tolist(), [i * 2 + 1, (i + 1) % 4])
            self.assertGreater(result.max_batch_size, 1)
            self.assertGreaterEqual(result.gpu_forwards, 1)
        self.assertEqual(sum(len(batch) for batch in vision.batches), 8)
        self.assertEqual(vision.legacy_calls, 0)

    def test_local_keeps_existing_inline_path(self):
        engine, vision = self.engine(VitSeparation.VIT_SEPARATION_LOCAL)
        result = engine.submit(["1", "2"])
        self.assertIsNone(engine._scheduler)
        self.assertEqual(vision.legacy_calls, 2)
        self.assertEqual(len(result.embeddings), 2)

    def test_legacy_local_config_without_role_remains_supported(self):
        original, vision = self.engine(VitSeparation.VIT_SEPARATION_LOCAL)
        engine = MMProcessEngine(original.model, SimpleNamespace(download_headers=""))
        self.addCleanup(engine.stop)
        self.assertIsNone(engine._scheduler)
        self.assertEqual(len(engine.submit(["1"]).embeddings), 1)
        self.assertEqual(vision.legacy_calls, 1)

    def test_remote_engine_remains_a_lightweight_initialization_object(self):
        engine, _ = self.engine(VitSeparation.VIT_SEPARATION_REMOTE)
        self.assertIsNone(engine._scheduler)

    def test_count_mismatch_fails_before_preprocessing(self):
        engine, vision = self.engine(VitSeparation.VIT_SEPARATION_ROLE)
        with self.assertRaisesRegex(ValueError, "counts do not match"):
            engine.submit(["1", "2"], types=[0])
        self.assertFalse(vision.batches)

    def test_cached_features_preserve_phase_and_avoid_second_gpu_forward(self):
        engine, vision = self.engine(VitSeparation.VIT_SEPARATION_ROLE, cache_size=2)
        phase0 = [[-1] * 7 + [0]]
        phase1 = [[-1] * 7 + [1]]
        first = engine.submit(["7"], preprocess_configs=phase0)
        again = engine.submit(["7"], preprocess_configs=phase0)
        self.assertIs(first.embeddings[0], again.embeddings[0])
        self.assertEqual(again.gpu_forwards, 0)
        self.assertEqual(len(vision.batches), 1)
        changed = engine.submit(["7"], preprocess_configs=phase1)
        self.assertEqual(changed.embeddings[0][0].tolist(), [7, 1])
        self.assertEqual(len(vision.batches), 2)


if __name__ == "__main__":
    unittest.main()
