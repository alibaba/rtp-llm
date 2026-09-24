import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import test_cache_scale_gate as fixtures
from workload.cache_gate import analyze
from workload.cache_gate_ab import compare, main


class CacheExperimentReportTest(unittest.TestCase):
    def evidence(self, hit):
        e = fixtures.CacheGateTest().evidence(hit)
        e['provenance'] = dict(
            instance='cache::step::profile', topology={'prefill': 2}, capacity={},
            performance={}, master_config={}, actual_master_config={'scheduler': 'same'},
            configuration_sha256='same', client_environment={}, trace={'sha256': 'same'},
            files={'/mock/flexlb-mock-engine-test.jar': 'mock'},
            master_artifact={'jar_sha256': 'same'},
        )
        return e

    def test_cli_never_judges_direction_or_missing_events(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, b = root/'a.json', root/'b.json'
            policy = root/'policy.yaml'
            policy.write_text('alignment_event: custom_event\n')
            for hits in ((.8,.8), (.1,.1), (.8,.1), (.1,.8)):
                ea, eb = map(self.evidence, hits)
                # Deliberate control difference is information, not an experiment verdict.
                eb['provenance']['actual_master_config'] = {'scheduler': 'other'}
                a.write_text(json.dumps(ea)); b.write_text(json.dumps(eb))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main([str(a), str(b), '--config', str(policy), '--output', str(root/'report')]), 0)
                result = json.loads((root/'report/reports/comparison/cache-scale-in-ab/analysis.json').read_text())['result']
                self.assertEqual(result['verdicts'], {'A': analyze(ea)['verdict'], 'B': analyze(eb)['verdict']})
                self.assertEqual(result['time_alignment']['status'], 'UNAVAILABLE')
                self.assertEqual(result['identity']['master_artifact'], 'SAME')
                self.assertEqual(result['identity']['master_configuration'], 'DIFFERENT')
                self.assertNotIn('decision', result)
                self.assertFalse((root/'report/old').exists())
                self.assertTrue((root/'report/A').exists())
            eb['errors'] = ['observation failed']
            b.write_text(json.dumps(eb))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(a), str(b), '--output', str(root/'invalid')]), 0)

    def test_declared_event_shifts_curves_without_changing_verdicts(self):
        a, b = self.evidence(.8), self.evidence(.8)
        a['events'] = [{'name': 'custom_event', 't': 10}, {'name': 'withdraw_start', 't': 20}]
        b['events'] = [{'name': 'custom_event', 't': 15}, {'name': 'withdraw_start', 't': 20}]
        original = copy.deepcopy((a,b))
        panel = {'panels': [{'id': 'run', 'caption': '', 'series': [
            {'name': 'P cache hit ratio', 'color': '#1677ff', 'points': [{'x': 5, 'y': .5}, {'x': 25, 'y': .8}]}]}]}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pa,pb = root/'a.json',root/'b.json'
            pa.write_text(json.dumps(a));pb.write_text(json.dumps(b))
            with mock.patch('workload.cache_gate_ab.build_spec', return_value=panel):
                result = compare(pa,pb,root/'report',alignment_event='custom_event')
                plain = compare(pa,pb,root/'plain')
            spec = json.loads((root/'report/reports/comparison/cache-scale-in-ab/report-spec.json').read_text())
            self.assertEqual(spec['events'], [{'name':'custom_event','t':0}])
            self.assertEqual([p['x'] for p in spec['panels'][1]['series'][0]['points']], [-5,15])
            self.assertEqual([p['x'] for p in spec['panels'][2]['series'][0]['points']], [-10,10])
            self.assertEqual(spec['timeAxis']['min'], -10)
            self.assertEqual(result['verdicts'], plain['verdicts'])
            self.assertEqual((a,b), original)
            self.assertEqual(panel['panels'][0]['series'][0]['points'][0]['x'],5)
