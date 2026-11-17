"""Parse the actual smoke entry configurations with production setup_args.

This checks startup configuration, not execution of GPU chunk/cache paths.
"""

import importlib
import os
import shlex
from unittest import TestCase
from unittest.mock import patch

from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.test.smoke_framework.manifest import get_gpu_count
from rtp_llm.test.smoke_framework.runner import _build_env_args


def cases(suite):
    return importlib.import_module(
        "rtp_llm.test.smoke.suites.test_smoke_" + suite
    ).SMOKE_CASES


class SmokeStartupArgsTest(TestCase):
    def parse(self, config):
        args = config["smoke_args"]
        roles = args if isinstance(args, dict) else {"single": args}
        envs = _build_env_args(args, config.get("envs", []))
        parsed = {}
        for role, arguments in roles.items():
            role_envs = envs[role] if isinstance(envs, dict) else envs
            with patch.dict(
                os.environ, dict(e.split("=", 1) for e in role_envs), clear=True
            ):
                parsed[role] = setup_args(shlex.split(arguments))
        return parsed

    def test_chunked_cases_reach_server_configuration(self):
        for suite, name, gpus in (
            ("h20_dense", "dense_fp8pb_dynamic", 1),
            ("h20_mla", "mla_glm4_moe_lite", 1),
            ("h20_eagle", "eagle_mtp_tp2", 2),
        ):
            with self.subTest(name=name):
                config = cases(suite)[name]
                self.assertEqual(get_gpu_count(config), gpus)
                parsed = self.parse(config)["single"]
                self.assertEqual(parsed.kv_cache_config.seq_size_per_block, 64)
                self.assertEqual(
                    parsed.runtime_config.fifo_scheduler_config.prefill_chunk_size, 64
                )
                if name == "eagle_mtp_tp2":
                    self.assertEqual(parsed.model_args.max_seq_len, 320)

    def test_cache_cases_reach_canonical_configuration(self):
        selected = {
            "cuda_remote_cache": tuple(cases("cuda_remote_cache")),
            "h20_eagle": ("eagle_mtp_reuse", "eagle_remote_cache_tp2"),
            "h20_next": ("next_long_reuse_memcache", "next_long_reuse_remote"),
            "h20_kimi_linear": ("kimi_long_reuse_memcache",),
        }
        for suite, names in selected.items():
            for name in names:
                with self.subTest(name=name):
                    config = cases(suite)[name]
                    self.assertGreaterEqual(config.get("sleep_time_qr", 0), 10)
                    for parsed in self.parse(config).values():
                        kv = parsed.kv_cache_config
                        if suite == "cuda_remote_cache" or name in (
                            "eagle_remote_cache_tp2",
                            "next_long_reuse_remote",
                        ):
                            self.assertTrue(kv.enable_device_cache)
                            self.assertFalse(kv.enable_memory_cache)
                            self.assertFalse(kv.enable_disk_cache)
                            self.assertEqual(kv.test_block_num, 500)
                            self.assertEqual(
                                kv.block_tree_device_evict_low_watermark_ratio, 0.001
                            )
                            self.assertEqual(
                                kv.block_tree_device_evict_high_watermark_ratio, 0.002
                            )
                        if name in (
                            "remote_cache_tp2",
                            "remote_cache_pd",
                            "eagle_remote_cache_tp2",
                        ):
                            self.assertEqual(kv.kvcm_put_timeout_ms, 12000)
                            self.assertEqual(kv.kvcm_get_timeout_ms, 12000)
                            self.assertEqual(kv.kvcm_get_broadcast_timeout, 15000)
                            self.assertEqual(kv.kvcm_put_broadcast_timeout, 15000)
                        elif name == "next_long_reuse_remote":
                            self.assertEqual(kv.kvcm_put_timeout_ms, 17000)
                            self.assertEqual(kv.kvcm_get_timeout_ms, 17000)
                            self.assertEqual(kv.kvcm_get_broadcast_timeout, 20000)
                            self.assertEqual(kv.kvcm_put_broadcast_timeout, 20000)
