import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from k31_last_n_calibration import CalibrationRunner, audit_calibration
from text_smoke import Case


class CalibrationTest(unittest.TestCase):
    def test_only_six_existing_inputs_and_unbounded_c7_positive_control(self):
        runner = object.__new__(CalibrationRunner)
        runner.args = SimpleNamespace(last_n_checkpoints=5, reuse_unit_tokens=1024, namespace='fixed')
        runner.cache_seed_prompt = lambda tag, value: 'seed-a'
        runner._last_n_cache_pairs = {'seed-a': dict(probe_prompt='probe-b', probe_tokens=10368, common_tokens=10240)}
        runner.linear_c7_cases = lambda: (Case('prefix-A-seed', 'c7-a', '.', 'miss'),
                                        Case('prefix-B-partial', 'c7-b', '.', 'miss', expected_reuse_len=0))
        stages = []
        runner._required_stage = lambda name, cases: stages.append((name, cases))
        runner.run_calibration()
        cases = [case for _, items in stages for case in items]
        self.assertEqual(len(cases), 6)
        self.assertEqual([case.prompt for case in cases[:4]], ['probe-b', 'probe-b', 'seed-a', 'probe-b'])
        self.assertEqual(cases[3].expected_reuse_len, 10240)
        self.assertEqual(cases[5].expected_reuse_len, 7168)
        self.assertEqual(cases[5].reuse, 'partial')
        self.assertEqual([case.preparation_only for case in cases], [True]*4+[False]*2)
        for case in cases[:2]:
            self.assertFalse(runner.prepare_http_payload({}, case)['extra_configs']['reuse_cache'])

    def test_a_last_five_run_cannot_calibrate_the_unbounded_control(self):
        names = ('tier-memory-seed-1', 'tier-witness-seed', 'tier-memory-seed-0',
                 'tier-device-probe-0', 'prefix-A-seed', 'prefix-B-partial')
        result = dict(passed=True, cases=[dict(name=name) for name in names], stages=[],
                      validation_profile='last-n-unbounded-calibration', native_K31_MTP=True,
                      target_only=False, geometry=dict(reuse_cache_tail_blocks=5))
        with self.assertRaisesRegex(ValueError, 'original unbounded path'):
            audit_calibration(result, {}, {}, '/unused')


if __name__ == '__main__':
    unittest.main()
