"""CPU contracts for the reduced ViT resource configuration."""

import io
import os
import sys
import unittest
from unittest.mock import Mock, patch

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.server.server_args.server_args import EnvArgumentParser
from rtp_llm.server.server_args.vit_group_args import init_vit_group_args


class VitResourceConfigTest(unittest.TestCase):
    REMOVED = {
        "mm_preprocess_num_threads": "MM_PREPROCESS_NUM_THREADS",
        "gpu_max_batch_images": "VIT_GPU_MAX_BATCH_IMAGES",
        "mm_max_queue_size": "MM_MAX_QUEUE_SIZE",
    }

    def parse(self, args=(), env=None):
        config = VitConfig()
        parser = EnvArgumentParser()
        parser.set_root_config(config)
        init_vit_group_args(parser, config)
        with patch.dict(os.environ, env or {}, clear=True):
            parser.parse_args(list(args))
        return config, parser

    def test_removed_fields_and_arguments_are_absent(self):
        config, parser = self.parse()
        option_names = parser._option_string_actions
        for name in self.REMOVED:
            with self.subTest(name=name):
                self.assertFalse(hasattr(config, name))
                self.assertNotIn("--" + name, option_names)
                self.assertNotIn(name + ":", config.to_string())

    def test_removed_cli_flags_are_rejected(self):
        for name in self.REMOVED:
            with self.subTest(name=name), patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as raised:
                    self.parse(["--" + name, "2"])
                self.assertEqual(raised.exception.code, 2)

    def test_removed_environment_cannot_create_hidden_limits(self):
        defaults, _ = self.parse()
        config, _ = self.parse(env={name: "2" for name in self.REMOVED.values()})
        self.assertEqual(
            config.embedding_scheduler_args(), defaults.embedding_scheduler_args()
        )
        self.assertEqual(
            config.async_compute_executor_args(), defaults.async_compute_executor_args()
        )
        for name in self.REMOVED:
            self.assertFalse(hasattr(config, name))

    def test_default_prefetch_is_on_and_serial_policy_is_preserved(self):
        config, _ = self.parse()
        self.assertEqual(config.gpu_prefetch_batches, 1)
        self.assertEqual(config.gpu_prefetch_batches, VitConfig().gpu_prefetch_batches)
        self.assertEqual(
            config.embedding_scheduler_args(),
            {
                "max_batch_size": 1,
                "max_batch_images": sys.maxsize,
                "batch_wait_ms": 0,
                "max_queue_size": 128,
                "prefetch_batches": 1,
            },
        )

    def test_one_batch_limit_and_derived_queue_via_environment(self):
        config, _ = self.parse(
            env={
                "VIT_GPU_MAX_BATCH_SIZE": "126",
                "VIT_CONCURRENCY": "256",
                "VIT_MAX_QUEUE_SIZE": "64",
                "VIT_GPU_BATCH_WAIT_MS": "1",
            }
        )
        args = config.embedding_scheduler_args()
        self.assertEqual(args["max_batch_size"], 126)
        self.assertEqual(args["max_batch_images"], 126)
        self.assertEqual(args["max_queue_size"], 320)
        self.assertEqual(args["batch_wait_ms"], 1)

    def test_cli_overrides_env_without_independent_gpu_queue_or_image_caps(self):
        config, _ = self.parse(
            [
                "--gpu_max_batch_size",
                "8",
                "--vit_concurrency",
                "4",
                "--vit_max_queue_size",
                "0",
            ],
            {
                "VIT_GPU_MAX_BATCH_SIZE": "126",
                "VIT_CONCURRENCY": "256",
                "VIT_MAX_QUEUE_SIZE": "64",
            },
        )
        args = config.embedding_scheduler_args()
        self.assertEqual(args["max_batch_size"], 8)
        self.assertEqual(args["max_batch_images"], 8)
        self.assertEqual(args["max_queue_size"], 4)
        self.assertEqual(config.async_compute_executor_args()["max_queue_size"], 0)

    def test_zero_extra_async_queue_still_admits_active_tasks_to_gpu(self):
        config = VitConfig()
        config.vit_concurrency = 1
        config.vit_max_queue_size = 0
        self.assertEqual(config.embedding_queue_capacity(), 1)
        # Preserve the engine's established clamping semantics.
        config.vit_concurrency = 0
        config.vit_max_queue_size = -3
        self.assertEqual(config.embedding_queue_capacity(), 1)

    def test_prefetch_can_be_explicitly_disabled(self):
        for args, env in (
            (["--gpu_prefetch_batches", "0"], {}),
            ([], {"GPU_PREFETCH_BATCHES": "0"}),
        ):
            config, _ = self.parse(args, env)
            self.assertEqual(config.gpu_prefetch_batches, 0)
            self.assertNotIn("prefetch_batches", config.embedding_scheduler_args())
        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                self.parse(["--gpu_prefetch_batches", "2"])

    def test_mixin_uses_the_same_resolved_media_budget(self):
        from rtp_llm.config.dsv41_config import V41Config
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import (
            deepseek_v41_mixin as mixin,
        )

        related = Mock(config={"v41_config": V41Config.__new__(V41Config)})
        for batch_size, expected in ((1, sys.maxsize), (8, 8), (128, 128)):
            config = VitConfig()
            config.gpu_max_batch_size = batch_size
            with self.subTest(batch_size=batch_size), patch.object(
                mixin, "DeepSeekV41VisionEmbedding"
            ) as embedding:
                mixin.DeepSeekV41Mixin._get_mm_module(related, config)
                self.assertEqual(
                    embedding.call_args.kwargs["max_batch_images"], expected
                )

    def test_worker_does_not_override_runtime_threads(self):
        from rtp_llm.multimodal import mm_process_engine as engine

        config = VitConfig()
        with patch.object(engine.signal, "signal"), patch.object(
            engine.torch, "set_num_threads"
        ) as threads, patch.object(
            engine.url_data_cache_, "resize_cache"
        ), patch.object(
            engine, "_worker_vit_config"
        ), patch.object(
            engine, "_worker_preprocess_params"
        ), patch.object(
            engine, "_worker_preprocess_func"
        ):
            engine._worker_initializer(config, {}, Mock())
        threads.assert_not_called()


if __name__ == "__main__":
    unittest.main()
