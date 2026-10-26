"""Real-weight, in-process eager/late acceptance (no embedding RPC serialization)."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeMixin,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType, VitParameters


class Qwen35LateVitAcceptanceTest(unittest.TestCase):
    def test_four_videos_eager_and_late_are_identical(self):
        # A manual target: missing acceptance resources is an error, never a skip/PASS.
        checkpoint = os.environ["RTP_MM_TEST_CHECKPOINT"]
        video_dir = os.environ["RTP_MM_TEST_VIDEO_DIR"]
        self.assertTrue(torch.cuda.is_available())
        torch.cuda.set_device(0)
        params = VitParameters()
        params.config["ckpt_path"] = checkpoint
        params.preprocess_batch_size = 1
        vit = VitConfig()
        vit.use_local_preprocess = True
        vit.disable_access_log = True
        vit.mm_cache_gpu_max_bytes = 512 * 1024 * 1024
        vit.mm_cache_cpu_max_bytes = 512 * 1024 * 1024
        mixin = Qwen3_5MoeMixin(
            torch.bfloat16, "cuda:0", params, LoadMethod.AUTO, vit, checkpoint
        )
        self.assertTrue(mixin.weights, "real checkpoint weights must be loaded")
        mixin.mm_part.visual.eval()
        model_config = SimpleNamespace(
            mm_model_config=SimpleNamespace(mm_position_ids_style=1),
            mm_related_params=params,
        )
        config = [1280, 704, 2500000, 73728000, 6, -1, 180, [], 120000]
        reference = []
        eager = MMProcessEngine(
            mixin.mm_part,
            model_config,
            vit,
            ProfilingDebugLoggingConfig(),
            device="cuda:0",
        )
        names = ("video.mp4", "video_02.mp4", "video_03.mp4", "video_04.mp4")
        try:
            for index, name in enumerate(names):
                result = eager.mm_embedding_cpp(
                    [os.path.join(video_dir, name)],
                    [MMUrlType.VIDEO],
                    [torch.empty(0)],
                    [config],
                    index + 1,
                )
                reference.append(
                    tuple(
                        [
                            value.detach().cpu().clone()
                            for value in (getattr(result, key) or [])
                        ]
                        for key in ("embeddings", "position_ids", "feature_hashes")
                    )
                )
        finally:
            eager.stop()
        late = MMProcessEngine(
            mixin.mm_part,
            model_config,
            vit,
            ProfilingDebugLoggingConfig(),
            device="cuda:0",
        )
        try:
            for index, name in enumerate(names):
                path = os.path.join(video_dir, name)
                item = MultimodalInput(
                    path, MMUrlType.VIDEO, torch.empty(0), MMPreprocessConfig(*config)
                )
                with patch.object(
                    mixin.mm_part.visual, "forward", wraps=mixin.mm_part.visual.forward
                ) as forward:
                    lengths = late.token_lengths([item])
                    self.assertEqual(
                        forward.call_count, 0, "metadata stage must not run ViT"
                    )
                    actual = late.mm_embedding_cpp(
                        [path],
                        [MMUrlType.VIDEO],
                        [torch.empty(0)],
                        [config],
                        100 + index,
                    )
                    self.assertEqual(
                        len(actual.embeddings), 1, "one complete tensor per video"
                    )
                    self.assertEqual(lengths, [actual.embeddings[0].shape[0]])
                    self.assertEqual(forward.call_count, 1)
                    for key, expected in zip(
                        ("embeddings", "position_ids", "feature_hashes"),
                        reference[index],
                    ):
                        values = getattr(actual, key) or []
                        self.assertEqual(len(values), len(expected))
                        for value, saved in zip(values, expected):
                            torch.testing.assert_close(
                                value.cpu(), saved, rtol=0, atol=0
                            )
                    cached = late.mm_embedding_cpp(
                        [path],
                        [MMUrlType.VIDEO],
                        [torch.empty(0)],
                        [config],
                        200 + index,
                    )
                    self.assertEqual(
                        forward.call_count, 1, "cache hit must not rerun ViT"
                    )
                    torch.testing.assert_close(
                        cached.embeddings[0], actual.embeddings[0], rtol=0, atol=0
                    )
                    print(
                        f"REAL_WEIGHT_ACCEPT video={name} rows={lengths[0]} dtype={actual.embeddings[0].dtype} exact=True",
                        flush=True,
                    )
        finally:
            late.stop()


if __name__ == "__main__":
    unittest.main()
