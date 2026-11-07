import copy
import unittest
from types import SimpleNamespace

from k31_last_n_tier_probe import TierProbeRunner, completed_seed_writes


class IsolatedTierProbeTest(unittest.TestCase):
    def test_lower_tier_requires_coordinator_settlement_key_crc_and_payload(self):
        events = {rank: [
            dict(event='linear_cache_publish_candidates', group_tag='linear',
                 prompt_tokens=12416, target_tier='HOST', time_ns=1,
                 checkpoints=[dict(tokens=point, cache_key=point+rank)
                              for point in range(8192, 12289, 1024)]),
            dict(event='linear_cache_transfer_completed', group_tag='linear',
                 cache_key=10240+rank, target_tier='HOST', target_block=rank+1,
                 time_ns=2, crc_enabled=True, payload_bytes=1024)] for rank in range(8)}
        self.assertEqual(len(completed_seed_writes(events, 'HOST')), 1)
        # Worker ranks do not own request/cache-tree metadata.
        central = {rank: rows if rank == 0 else [] for rank,rows in events.items()}
        self.assertEqual(completed_seed_writes(central, 'HOST'), completed_seed_writes(events, 'HOST'))
        for mutation in (dict(cache_key=-1), dict(crc_enabled=False), dict(payload_bytes=0),
                         dict(time_ns=0), dict(target_tier='DISK')):
            bad = copy.deepcopy(events)
            bad[0][1].update(mutation)
            with self.subTest(mutation=mutation):
                self.assertIsNone(completed_seed_writes(bad, 'HOST'))

    def test_all_tiers_use_identical_two_inputs_and_exact_c10_branch(self):
        inputs = []
        for tier in ('DEVICE', 'HOST', 'DISK'):
            runner = object.__new__(TierProbeRunner)
            runner.args = SimpleNamespace(namespace='fixed', last_n_checkpoints=5, reuse_unit_tokens=1024)
            runner.cache_seed_prompt = lambda tag, answer: 'seed-a'
            runner._last_n_cache_pairs = {'seed-a': dict(probe_prompt='probe-b', probe_tokens=10368, common_tokens=10240)}
            stages, waits = [], []
            runner._required_stage = lambda name, cases: stages.extend(cases)
            runner.wait_seed_write = waits.append
            runner.run_probe(tier)
            self.assertEqual(len(stages), 2)
            self.assertEqual(waits, [tier])
            self.assertEqual((stages[0].expected_input_len, stages[1].expected_reuse_len), (12416, 10240))
            self.assertEqual((stages[0].preparation_only, stages[1].preparation_only), (True, False))
            self.assertTrue(runner.precision_case(stages[1]))
            inputs.append([case.prompt for case in stages])
        self.assertEqual(inputs, [['seed-a', 'probe-b']]*3)


if __name__ == '__main__':
    unittest.main()
