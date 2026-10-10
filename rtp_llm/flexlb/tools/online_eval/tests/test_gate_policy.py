import unittest
from types import SimpleNamespace

from scenario.actions.master import _client_check
from workload.evidence_analysis import bounded_collection_gaps
from workload.performance_gate import for_profile
from test_performance_gate import evidence


class GatePolicyTest(unittest.TestCase):
    def test_short_gap_requires_success_on_both_sides(self):
        series = {'1/master-B/up/{}': [[0, 1], [1, None], [2, 1]]}
        gap = {'1/master-B/collection': [1]}
        self.assertEqual(bounded_collection_gaps(series, gap, 5), (gap, {}))
        for points in [[[1, None], [2, 1]], [[0, 1], [1, None]],
                       [[0, 1], [1, None], [6, 1]]]:
            self.assertEqual(bounded_collection_gaps(
                {'1/master-B/up/{}': points}, gap, 5), ({}, gap))

    def test_consecutive_failures_use_entire_gap(self):
        gap = {'1/master-B/collection': [1, 2, 3, 4, 5]}
        series = {'1/master-B/up/{}': [[0, 1], [6, 1]]}
        self.assertEqual(bounded_collection_gaps(series, gap, 5), ({}, gap))

    def test_advisory_errors_do_not_waive_sample_coverage_or_single(self):
        rows = [{'status': 'ok'}, {'status': 'error'}]
        ctx = SimpleNamespace(instance={'profile': 'batch-window'},
                              resource=lambda *args: rows)
        deadline = SimpleNamespace(check=lambda: None)
        params = dict(rows='rows', metric='non_ok_count', op='eq', expected=0,
                      min_samples=2, warning_profiles=['batch-window'])
        check = _client_check(ctx, params, deadline).checks[0]
        self.assertEqual((check.status, check.actual, check.expected), ('WARNING', 1, 0))
        self.assertEqual(_client_check(ctx, {**params, 'min_samples': 3}, deadline)
                         .checks[0].status, 'FAIL')
        ctx.instance['profile'] = 'single-nonbatch'
        self.assertEqual(_client_check(ctx, params, deadline).checks[0].status, 'FAIL')

    def test_profile_floors_preserve_batch_and_decode(self):
        criteria = evidence()['criteria']
        criteria['engine_tps'] = dict(rtp_llm_context_tps=50000,
                                     rtp_llm_context_tps_with_cache=100000,
                                     rtp_llm_generate_tps=2230)
        criteria['engine_tps_by_profile'] = {'single-nonbatch': {
            'rtp_llm_context_tps': 45000, 'rtp_llm_context_tps_with_cache': 90000}}
        single = for_profile(criteria, 'single-nonbatch')
        batch = for_profile(criteria, 'batch-window')
        self.assertEqual(single['engine_tps']['rtp_llm_context_tps'], 45000)
        self.assertEqual(single['engine_tps']['rtp_llm_generate_tps'], 2230)
        self.assertEqual(batch['engine_tps']['rtp_llm_context_tps'], 50000)
        self.assertEqual(criteria['engine_tps']['rtp_llm_context_tps'], 50000)
