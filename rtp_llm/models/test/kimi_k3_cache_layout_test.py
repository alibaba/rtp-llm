"""The Decode allocator and PageRR readers must agree before engine creation."""

import pickle
import unittest
from types import SimpleNamespace

from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops import CPRotateMethod, ParallelismConfig, PrefillCPConfig, RoleType


class KimiK3CacheLayoutTest(unittest.TestCase):
    def configure(self, *, k3=True, dsa=True, role=RoleType.DECODE, tp=8, sharded=True):
        parallel = ParallelismConfig()
        parallel.role_type = role
        parallel.tp_size = tp
        cp = parallel.prefill_cp_config
        cp.method = CPRotateMethod.PREFILL_CP
        cp.kv_cache_sharded = sharded
        cp.prefill_cp_size = 8
        model = SimpleNamespace(max_seq_len=4096, model_name="fixture")
        if k3:
            model.k3_runtime_config = SimpleNamespace(dsa=object() if dsa else None)
        engine = SimpleNamespace(
            parallelism_config=parallel,
            runtime_config=SimpleNamespace(
                model_name="", fifo_scheduler_config=SimpleNamespace(
                    max_context_batch_size=2, max_batch_tokens_size=8192
                )
            ),
        )
        ModelFactory.update_engine_config_from_model_config(engine, model)
        return parallel

    def test_target_tp_controls_decode_shards_without_enabling_query_cp(self):
        for tp in (4, 8):
            for dsa in (False, True):
                with self.subTest(tp=tp, dsa=dsa):
                    p = self.configure(tp=tp, dsa=dsa)
                    cp = p.prefill_cp_config
                    self.assertTrue(cp.decode_kv_cache_sharded)
                    self.assertEqual(cp.prefill_cp_size, 8)
                    self.assertFalse(cp.is_enabled())
                    self.assertEqual(p.get_attn_tp_size(), tp)
                    restored = pickle.loads(pickle.dumps(p))
                    self.assertTrue(restored.prefill_cp_config.decode_kv_cache_sharded)
                    self.assertEqual(restored.tp_size, tp)

    def test_replicated_backends_and_disabled_dcp_remain_replicated(self):
        for overrides in ({"k3": False}, {"sharded": False}, {"tp": 1}):
            with self.subTest(overrides=overrides):
                self.assertFalse(self.configure(**overrides).prefill_cp_config.decode_kv_cache_sharded)

    def test_prefill_keeps_its_existing_query_cp_contract(self):
        cp = self.configure(role=RoleType.PREFILL).prefill_cp_config
        self.assertTrue(cp.is_enabled())
        self.assertTrue(cp.preserve_tp_layout)
        self.assertFalse(cp.decode_kv_cache_sharded)

    def test_old_serialized_configs_default_to_replicated_decode(self):
        states = (
            (CPRotateMethod.PREFILL_CP, 512),
            (CPRotateMethod.PREFILL_CP, 512, True, 8),
            (CPRotateMethod.PREFILL_CP, 512, True, 8, True),
        )
        for state in states:
            with self.subTest(state=state):
                config = PrefillCPConfig.__new__(PrefillCPConfig)
                config.__setstate__(state)
                self.assertFalse(config.decode_kv_cache_sharded)
                self.assertEqual(config.method, CPRotateMethod.PREFILL_CP)
                if len(state) == 5:
                    self.assertTrue(config.preserve_tp_layout)


if __name__ == "__main__":
    unittest.main()
