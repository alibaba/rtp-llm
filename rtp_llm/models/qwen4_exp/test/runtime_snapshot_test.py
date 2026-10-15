"""Configuration ownership across final overrides and target/draft spawn."""

import multiprocessing
import pickle
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.qwen4_exp_config import freeze_runtime_config


def _rebuild_snapshot_in_child(connection, budget, draft):
    try:
        config = RuntimeSnapshotTest().config()
        config.apply_override_args('{"qwen4_ple_prefetch_mb": %d}' % budget)
        config.is_mtp = draft
        if draft:
            config._qwen4_ple_layer_ids = []
        connection.send(
            freeze_runtime_config(config, SimpleNamespace(get_attn_tp_size=lambda: 2))
        )
    finally:
        connection.close()


class RuntimeSnapshotTest(unittest.TestCase):
    def config(self):
        config = ModelConfig()
        config.hidden_size = 2560
        config.hc_mult = 4
        config.is_mtp = False
        config.quant_config = None
        config._qwen4_ple_layer_ids = [1, 5]
        config._qwen4_split_ngram_parts = 128
        config._qwen4_indexer_head_dim = 128
        config._qwen4_indexer_compress_ratio = 4
        config._qwen4_indexer_budget = 1024
        config._qwen4_indexer_kv_heads = 1
        return config

    def test_final_override_snapshot_survives_pickle_and_is_immutable(self):
        config = self.config()
        config.qwen4_ple_prefetch_mb = 512
        snapshot = freeze_runtime_config(
            config, SimpleNamespace(get_attn_tp_size=lambda: 2)
        )
        spawned = pickle.loads(pickle.dumps(snapshot))
        self.assertEqual(spawned, snapshot)
        self.assertEqual(spawned.options.ple_prefetch_mb, 512)
        with self.assertRaises(FrozenInstanceError):
            spawned.options.ple_prefetch_mb = 128
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            config.qwen4_ple_prefetch_mb = 128

    def test_spawn_reconstructs_target_and_draft_after_final_overrides(self):
        context = multiprocessing.get_context("spawn")
        for draft in (False, True):
            with self.subTest(draft=draft):
                parent, child = context.Pipe(duplex=False)
                process = context.Process(
                    target=_rebuild_snapshot_in_child, args=(child, 512, draft)
                )
                process.start()
                child.close()
                try:
                    self.assertTrue(parent.poll(60), "snapshot child timed out")
                    snapshot = parent.recv()
                    process.join(30)
                    self.assertEqual(process.exitcode, 0)
                    self.assertEqual(snapshot.options.ple_prefetch_mb, 512)
                    self.assertEqual(snapshot.attention_tp, 2)
                    self.assertEqual(snapshot.is_mtp_draft, draft)
                    self.assertEqual(
                        snapshot.geometry.ple_layer_ids, () if draft else (1, 5)
                    )
                finally:
                    parent.close()
                    if process.is_alive():
                        process.terminate()
                        process.join(10)
                    process.close()

    def test_draft_snapshot_has_independent_geometry_and_role(self):
        target = self.config()
        draft = self.config()
        draft.is_mtp = True
        draft._qwen4_ple_layer_ids = []
        parallelism = SimpleNamespace(get_attn_tp_size=lambda: 2)
        first = freeze_runtime_config(target, parallelism)
        second = freeze_runtime_config(draft, parallelism)
        self.assertEqual(first.geometry.ple_layer_ids, (1, 5))
        self.assertEqual(second.geometry.ple_layer_ids, ())
        self.assertFalse(first.is_mtp_draft)
        self.assertTrue(second.is_mtp_draft)

    def test_changed_effective_topology_cannot_reuse_frozen_snapshot(self):
        config = self.config()
        freeze_runtime_config(config, SimpleNamespace(get_attn_tp_size=lambda: 2))
        with self.assertRaisesRegex(RuntimeError, "configuration changed"):
            freeze_runtime_config(config, SimpleNamespace(get_attn_tp_size=lambda: 4))

    def test_noninteger_and_nonpositive_resource_overrides_are_rejected(self):
        for value in (True, 0, -1, 1.5, "256"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.config().qwen4_ple_prefetch_mb = value


if __name__ == "__main__":
    unittest.main()
