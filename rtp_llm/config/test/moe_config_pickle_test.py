import pickle
import unittest

from rtp_llm.ops import MoeConfig


class MoeConfigPickleTest(unittest.TestCase):
    def test_fp4_backend_survives_process_serialization(self):
        for backend in ("auto", "cutlass", "trtllm", "cutedsl"):
            with self.subTest(backend=backend):
                config = MoeConfig()
                config.moe_strategy = "nvfp4_cutlass"
                config.fp4_moe_op = backend
                config.use_all_gather = False
                config.deep_ep_num_sm = 7
                restored = pickle.loads(pickle.dumps(config))
                self.assertEqual(restored.fp4_moe_op, backend)
                self.assertEqual(restored.moe_strategy, config.moe_strategy)
                self.assertEqual(restored.use_all_gather, False)
                self.assertEqual(restored.deep_ep_num_sm, 7)

    def test_legacy_state_defaults_to_auto(self):
        config = MoeConfig()
        config.moe_strategy = "nvfp4_cutlass"
        state = config.__getstate__()
        self.assertEqual(len(state), 13)
        restored = MoeConfig.__new__(MoeConfig)
        restored.__setstate__(state[:12])
        self.assertEqual(restored.fp4_moe_op, "auto")
        self.assertEqual(restored.moe_strategy, "nvfp4_cutlass")

    def test_invalid_state_size_is_rejected(self):
        state = MoeConfig().__getstate__()
        for bad in (state[:11], state + ("extra",)):
            with self.assertRaises(RuntimeError):
                restored = MoeConfig.__new__(MoeConfig)
                restored.__setstate__(bad)


if __name__ == "__main__":
    unittest.main()
