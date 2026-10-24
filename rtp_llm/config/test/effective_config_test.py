"""Exercise the owning real parser/finalizer without importing native GPU bindings."""

import argparse
import ast
import json
import logging
import os
import sys
import tempfile
import types
import typing
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from rtp_llm.config.effective_config import (
    merge,
    snapshot,
    vit_snapshot,
    write_snapshot,
)

ROOT = Path(__file__).resolve().parents[3]


def definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    if len(selected) != len(names):
        raise AssertionError(f"missing owning definitions: {names}")
    exec(  # noqa: S102 - execute selected owning definitions without GPU imports
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )


class EffectiveConfigTest(unittest.TestCase):
    def parser(self):
        ns = dict(vars(typing), argparse=argparse, logging=logging, os=os, sys=sys)
        definitions("rtp_llm/server/server_args/util.py", {"str2bool"}, ns)
        definitions(
            "rtp_llm/server/server_args/server_args.py",
            {"ConfigBinding", "EnvArgumentGroup", "EnvArgumentParser"},
            ns,
        )
        definitions(
            "rtp_llm/server/server_args/fifo_scheduler_group_args.py",
            {"init_fifo_scheduler_group_args"},
            ns,
        )
        definitions(
            "rtp_llm/config/engine_config.py", {"finalize_scheduler_config"}, ns
        )
        config = NS()
        parser = ns["EnvArgumentParser"]()
        parser.set_root_config(config)
        ns["init_fifo_scheduler_group_args"](parser, config)
        return parser, config, ns["finalize_scheduler_config"]

    def test_real_cli_env_default_and_derived_zero_semantics(self):
        parser, config, finalize = self.parser()
        with patch.dict(
            os.environ,
            {"MAX_CONTEXT_BATCH_SIZE": "3", "MAX_BATCH_TOKENS_SIZE": "99"},
            clear=True,
        ):
            parser.parse_args(["--max_batch_tokens_size=0"])
        self.assertEqual(config.max_context_batch_size, 3)
        self.assertEqual(config.max_batch_tokens_size, 0)
        finalize(config, 1024)
        self.assertEqual(config.max_batch_tokens_size, 3072)
        parser, config, finalize = self.parser()
        with patch.dict(os.environ, {}, clear=True):
            parser.parse_args([])
        finalize(config, 2048)
        self.assertEqual(config.max_batch_tokens_size, 2048)
        self.assertTrue(config.cp_force_single_prefill)

    def test_real_abbreviated_cli_still_beats_environment(self):
        parser, config, _ = self.parser()
        with patch.dict(os.environ, {"MAX_CONTEXT_BATCH_SIZE": "4"}, clear=True):
            parser.parse_args(["--max_context_batch_s", "2"])
        self.assertEqual(config.max_context_batch_size, 2)

    def test_export_uses_finalized_values_and_preserves_capacity_authority(self):
        parser, fifo, finalize = self.parser()
        with patch.dict(os.environ, {}, clear=True):
            parser.parse_args([])
        finalize(fifo, 4096)
        engine = NS(
            runtime_config=NS(
                fifo_scheduler_config=fifo,
                max_generate_batch_size=32,
                use_batch_decode_scheduler=False,
            ),
            pd_sep_config=NS(
                role_type=NS(name="PREFILL"),
                prefill_retry_times=0,
                prefill_retry_timeout_ms=20,
                prefill_max_wait_timeout_ms=600000,
                decode_retry_interval_ms=1,
                load_cache_timeout_ms=5000,
                max_rpc_timeout_ms=7200000,
                prefill_prepare_resource_pool_size=0,
                prefill_stop_stream_wait_timeout_ms=2000,
                decode_retry_times=100,
                decode_retry_timeout_ms=100,
            ),
            concurrency_config=NS(concurrency_limit=32),
            parallelism_config=NS(
                tp_size=8, dp_size=1, prefill_cp_config=NS(is_enabled=lambda: True)
            ),
            kv_cache_config=NS(
                seq_size_per_block=64,
                reuse_cache=False,
                enable_device_cache=True,
                enable_memory_cache=False,
                enable_disk_cache=False,
                enable_remote_cache=False,
                reserve_block_ratio=5,
                device_eviction_policy="lru",
                memory_eviction_policy="lru",
                block_tree_transfer_worker_count=4,
                block_tree_business_queue_max_size=10000,
                block_tree_transfer_queue_max_size=10000,
                memory_cache_max_descriptors_per_transfer_batch=8,
                memory_cache_sync_timeout_ms=10000,
                block_tree_device_evict_low_watermark_ratio=0.82,
                block_tree_device_evict_high_watermark_ratio=0.90,
                block_tree_memory_evict_low_watermark_ratio=0.90,
                block_tree_memory_evict_high_watermark_ratio=0.94,
            ),
            config_provenance={"reuse_cache": {"value": False, "source": "default"}},
        )
        exported = snapshot(engine, NS(max_seq_len=4096), revision="fixture")
        role = exported["roles"]["prefill"]
        self.assertEqual(role["scheduler"]["max_batch_tokens_size"], 4096)
        self.assertEqual(role["model"]["cp_size"], 8)
        self.assertNotIn("capacity", role)
        self.assertEqual(role["provenance"]["kv_cache.reuse_cache"], "default")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "effective.json"
            write_snapshot(path, exported)
            self.assertEqual(json.loads(path.read_text()), exported)

    def test_vit_exports_resolved_scheduler_not_raw_knobs(self):
        tree = ast.parse((ROOT / "rtp_llm/config/py_config_modules.py").read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "VitConfig"
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "embedding_scheduler_args"
        )
        namespace = dict(vars(typing), sys=sys)
        exec(  # noqa: S102 - execute selected owning definitions without GPU imports
            compile(
                ast.Module(body=[method], type_ignores=[]),
                "VitConfig.embedding_scheduler_args",
                "exec",
            ),
            namespace,
        )
        vit = NS(
            vit_concurrency=0,
            vit_max_queue_size=-1,
            gpu_max_batch_size=1,
            gpu_max_batch_images=200,
            gpu_batch_wait_ms=10,
            mm_max_queue_size=1024,
            mm_cache_gpu_max_bytes=0,
            mm_cache_cpu_max_bytes=1024,
            mm_hash_key_cache_max_bytes=1024,
            mm_remote_cache_enable=False,
            mm_cache_item_num=10,
            url_cache_item_num=100,
            mm_timeout_ms=120000,
            mm_image_max_file_size_kb=16384,
            use_local_preprocess=False,
            mm_preprocess_max_workers=4,
            download_headers="",
            output_transport=NS(mode="grpc"),
        )
        vit.embedding_scheduler_args = types.MethodType(
            namespace["embedding_scheduler_args"], vit
        )
        result = vit_snapshot(vit, revision="fixture")["roles"]["vit"]
        self.assertEqual(result["gpu_batch_wait_ms"], 0)
        self.assertEqual(result["gpu_max_batch_images"], sys.maxsize)
        self.assertEqual(result["vit_concurrency"], 1)
        self.assertEqual(result["vit_max_queue_size"], 0)
        self.assertIsNone(result["work_budget"])
        self.assertIsNone(result["hash_cache_accounting"])
        self.assertFalse(result["greennet_enabled"])
        self.assertTrue(vit_snapshot(vit, revision="fixture", greennet_enabled=True)["roles"]["vit"]["greennet_enabled"])
        accounting = {"tensor_object_bytes": 80}
        self.assertEqual(vit_snapshot(vit, revision="fixture", hash_cache_accounting=accounting)["roles"]["vit"]["hash_cache_accounting"], accounting)
        budget = NS(
            input_patches=8,
            output_tokens=16,
            estimated_workspace_bytes=32,
            max_attention_segment=4,
            attention_work=64,
        )
        actual = vit_snapshot(vit, revision="fixture", work_budget=budget)["roles"][
            "vit"
        ]
        self.assertEqual(actual["work_budget"], vars(budget))

    def test_merge_rejects_revision_drift_and_duplicate_roles(self):
        one = {
            "schema_version": 1,
            "source": {"kind": "real", "revision": "a"},
            "roles": {"prefill": {}},
        }
        two = {
            "schema_version": 1,
            "source": {"kind": "real", "revision": "b"},
            "roles": {"decode": {}},
        }
        with self.assertRaises(ValueError):
            merge([one, two])
        with self.assertRaises(ValueError):
            merge([one, one])
        two["source"]["revision"] = "a"
        self.assertEqual(set(merge([one, two])["roles"]), {"prefill", "decode"})


if __name__ == "__main__":
    unittest.main()
