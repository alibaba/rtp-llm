"""Fail closed when an old scrape-token capture is used as execution TPS."""
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_eval.analysis import compare_ab as ab
from flexlb_eval.analysis import compare_twin as twin


def run(contract="execution_us_v1"):
    meta = {"trace_file_sha256": "same", "duration_s": 10, "prefill_tps_contract": contract}
    return {"meta": meta, "run_meta": None, "label": "fixture", "path": "fixture.json",
            "aggregate": {"meta": meta, "summary": {"test_valid": True}, "mock_tps_ts": []}}


class PrefillTpsContractTest(unittest.TestCase):
    def test_mixed_denominators_rejected(self):
        with self.assertRaisesRegex(ab.PrecheckError, 'TPS contract mismatch'):
            ab.precheck(run(), run("legacy_or_unknown"))

    def test_identical_new_contract_accepted(self):
        details, _ = ab.precheck(run(), run())
        self.assertTrue(details['prefill_tps_contract']['match'])

    def test_comparison_preserves_execution_and_wall_rates_separately(self):
        a = {'mock_tps_ts': [{'t': 1, 'context_tps': 8000, 'context_tps_with_cache': 6000,
                             'context_wall_tps': 800, 'context_wall_tps_with_cache': 1800}]}
        b = copy.deepcopy(a)
        b['mock_tps_ts'][0]['context_wall_tps'] = 400
        rows = []
        ab._collect_cache_tps({}, {}, a, b, 0, 2, rows)
        keyed = {r['name']: r for r in rows}
        self.assertEqual(keyed['mock_tps_steady_context_tps']['a'], 8000)
        self.assertEqual(keyed['mock_tps_steady_context_tps']['b'], 8000)
        self.assertEqual(keyed['mock_tps_steady_context_wall_tps']['b'], 400)

    def test_silent_steps_are_not_zero_or_missing_key_errors(self):
        means, count = ab.steady_mean([
            {'t': 1, 'context_tps_with_cache': 6000},
            {'t': 2, 'context_tps': 8000, 'context_tps_with_cache': 5000},
            {'t': 3, 'context_tps': None},
        ], 0, 4, ('context_tps', 'context_tps_with_cache'))
        self.assertEqual(count, 3)
        self.assertEqual(means, {'context_tps': 8000, 'context_tps_with_cache': 5500})

    def test_twin_does_not_alias_wall_to_execution(self):
        parsed = {'rtp_llm_context_wall_tps': [(0, 1)]}
        self.assertIsNone(twin._find_prom_series(parsed, twin.PROM_SERIES_CANDIDATES['context_tps']))


if __name__ == '__main__':
    unittest.main()
