import pickle
import unittest

from rtp_llm.ops import ParallelismConfig, RoleType


class ParallelismConfigPickleTest(unittest.TestCase):
    def test_round_trip_preserves_pp_and_mainline_fields(self):
        config = ParallelismConfig()
        fields = dict(
            tp_size=2,
            ep_size=4,
            dp_size=2,
            pp_size=3,
            world_size=12,
            world_rank=10,
            local_world_size=4,
            tp_rank=0,
            ep_rank=2,
            dp_rank=1,
            ffn_tp_size=2,
            ffn_tp_rank=0,
            use_ub_comm=True,
            role_type=RoleType.DECODE,
            pp_rank=2,
            pp_stage_layer_counts=[2, 3, 4],
        )
        for name, value in fields.items():
            setattr(config, name, value)
        config.prefill_cp_config.comm_buffer_size = 4096
        config.prefill_cp_config.kv_cache_sharded = True
        config.prefill_cp_config.prefill_cp_size = 2

        restored = pickle.loads(pickle.dumps(config))
        for name, value in fields.items():
            self.assertEqual(getattr(restored, name), value, name)
        self.assertEqual(
            restored.prefill_cp_config.__getstate__(),
            config.prefill_cp_config.__getstate__(),
        )


if __name__ == "__main__":
    unittest.main()
