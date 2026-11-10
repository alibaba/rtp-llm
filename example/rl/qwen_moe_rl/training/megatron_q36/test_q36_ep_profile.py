import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch
from q36_ep_profile import destination, expert_coordinate, validate_model, validate_native_profile


MODEL = dict(num_hidden_layers=40, hidden_size=2048, num_experts=256,
             num_experts_per_tok=8, num_attention_heads=16, num_key_value_heads=2)
NATIVE = dict(world_size=4, tp_size=4, dp_size=1, ep_size=4, pp_size=1,
              ffn_sp_size=4, ffn_tp_size=1, rank=0, model_config=MODEL,
              quantized=False, eplb=False, eager=True, moe_strategy='nccl_bf16_reference')


class ProfileTest(unittest.TestCase):
    def test_native_topology_opt_in_does_not_relax_ep1(self):
        path = Path(__file__).resolve().parents[2] / 'adapter' / 'rtp_topology.py'
        spec = importlib.util.spec_from_file_location('candidate_ep4_topology', path)
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            def topology(ep):
                return module.ReplicaTopology(replica_rank=0, tp_size=4, ep_size=ep,
                    gpu_ids=('0', '1', '2', '3'), rank_hosts=('n',) * 4, start_port=24000)
            with patch.dict('os.environ', {'RTP_LLM_RL_Q36_EP4': '0'}):
                self.assertEqual(topology(1).node_environment(0, 'run')['RTP_EP_SIZE'], '1')
                with self.assertRaises(NotImplementedError): topology(4).node_environment(0, 'run')
            with patch.dict('os.environ', {'RTP_LLM_RL_Q36_EP4': '1'}):
                self.assertEqual(topology(4).node_environment(0, 'run')['RTP_EP_SIZE'], '4')
        finally:
            sys.modules.pop(spec.name, None)

    def test_exact_ep4_profile(self):
        validate_native_profile(**NATIVE)
        for key, value in [('ep_size', 1), ('world_size', 8), ('eplb', True),
                           ('moe_strategy', 'auto'), ('rank', 4)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_native_profile(**dict(NATIVE, **{key: value}))
        with self.assertRaises(ValueError): validate_model(dict(MODEL, hidden_size=8192))

    def test_global_expert_coverage_and_two_to_one(self):
        names, slots = set(), set()
        for source in range(8):
            for local in range(32):
                global_id = source * 32 + local
                coordinate = expert_coordinate(logical_name='mlp.experts', shape=(512, 2048),
                    layer=3, ep_rank=source, local_expert=local,
                    source_global_expert=global_id, tensor_kind='down_proj')
                names.add(coordinate.canonical_name)
                target, target_local = destination(source, local)
                self.assertEqual((target, target_local), (global_id // 64, global_id % 64))
                slots.add((target, target_local))
        self.assertEqual(len(names), 256)
        self.assertEqual(slots, {(target, local) for target in range(4) for local in range(64)})

    def test_reject_misidentified_live_expert(self):
        with self.assertRaisesRegex(ValueError, 'ownership'):
            expert_coordinate(logical_name='mlp.experts', shape=(2, 2), layer=0, ep_rank=3,
                              local_expert=1, source_global_expert=1, tensor_kind='down_proj')


if __name__ == '__main__': unittest.main()
