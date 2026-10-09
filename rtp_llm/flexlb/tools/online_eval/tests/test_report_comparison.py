"""Frozen reports are the sole source for comparisons across all producers."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from reporting import load_analysis, read_bundle, run_meta, write_bundle
from reporting.comparison import compare, main
from cases.master_performance.publication import publish_performance as report
from cases.master_performance.analysis import analyze as analyze_performance
from cases.cache_scale_in.analysis import analyze as analyze_cache
from cases.cache_scale_in.publication import publish_cache as write_report
from test_performance_gate import evidence
import test_cache_scale_gate as cache_fixtures


class ReportComparisonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def bundle(self, name, *, verdict='PASS', value=10, event=10, configuration=None,
               unit='ms', second_panel=False):
        result = dict(verdict=verdict, metrics=dict(latency=value), criteria=dict(limit=20))
        spec = dict(title=name, subtitle=verdict, timeOriginLabel='run start',
                    timeAxis=dict(min=0, max=30),
                    events=[] if event is None else [dict(name='checkpoint', t=event)],
                    kpis=[dict(label='run verdict', value=verdict)],
                    sections=[dict(type='details', title='missing samples', value=['latency gap'])],
                    panels=[dict(id='latency', title='Latency', type='line', unit=unit, timeX=True,
                                 axes={'y': dict(title=unit)},
                                 series=[dict(name='latency', points=[dict(x=t, y=v) for t,v in zip([5, 10, 25], [value, None, value])], color='#1677ff')])])
        if second_panel:
            spec['panels'].append(dict(id='other', title='Other', type='bar', axes={'y': dict(title='count')},
                                       series=[dict(name='bar', points=[dict(x='x', y=2)])]))
        return write_bundle(self.root/name, 'run', 'fixture', result, spec,
                            meta=run_meta(dict(id=name), configuration=configuration or dict(scheduler='same'),
                                          workload=dict(sha256='same'), environment=dict(fetch=True)),
                            producer='fixture')

    def output(self, name='comparison'):
        return self.root/name

    def spec(self, name='comparison'):
        return json.loads((self.output(name)/'reports/comparison/runs/report-spec.json').read_text())

    def test_three_frozen_runs_are_shown_without_reanalysis_or_verdict(self):
        paths = [self.bundle('a'), self.bundle('b', verdict='FAIL', value=30),
                 self.bundle('c', verdict='INVALID', value=None)]
        before = [{p.name: p.read_bytes() for p in d.iterdir()} for d in paths]
        with mock.patch('cases.master_performance.analysis.analyze', side_effect=AssertionError('reanalysis')), \
             mock.patch('cases.cache_scale_in.analysis.analyze', side_effect=AssertionError('reanalysis')):
            result = compare(paths, self.output())
        self.assertEqual({k: v['verdict'] for k, v in result['runs'].items()},
                         dict(A='PASS', B='FAIL', C='INVALID'))
        self.assertNotIn('verdict', result)
        self.assertNotIn('gate', result)
        self.assertEqual(result['runs']['B']['metrics']['latency'], 30)
        self.assertEqual(before, [{p.name: p.read_bytes() for p in d.iterdir()} for d in paths])
        for token in ('Tier', 'noise_floor', 'rank_score', 'regression', 'gate passed'):
            self.assertNotIn(token, json.dumps(result))
        self.assertEqual(len(self.spec()['panels']), 4)
        self.assertEqual(self.spec()['panels'][0]['timeX'], True)
        styles = [s['dash'] for s in self.spec()['panels'][0]['series']]
        self.assertEqual(len({tuple(style) for style in styles}), 3)
        read_bundle(self.output()/'reports/comparison/runs')

    def test_explicit_event_shifts_only_copies_and_preserves_gaps(self):
        a, b = self.bundle('a', event=10), self.bundle('b', event=15)
        original = json.loads((a/'report-spec.json').read_text())
        result = compare([a, b], self.output(), alignment_event='checkpoint')
        self.assertEqual(result['time_alignment']['status'], 'ALIGNED')
        panels = self.spec()['panels']
        self.assertEqual([p['x'] for p in panels[1]['series'][0]['points']], [-5, 0, 15])
        self.assertEqual([p['x'] for p in panels[2]['series'][0]['points']], [-10, -5, 10])
        self.assertIsNone(panels[0]['series'][0]['points'][1]['y'])
        self.assertEqual(self.spec()['timeAxis']['min'], -15)
        self.assertEqual(json.loads((a/'report-spec.json').read_text()), original)

    def test_differences_missing_panels_and_invalid_runs_are_not_execution_errors(self):
        a = self.bundle('a', second_panel=True)
        b = self.bundle('b', verdict='INVALID', event=None, configuration=dict(scheduler='different'))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(a), str(b), '--output', str(self.output()),
                                   '--alignment-event', 'checkpoint']), 0)
        result = load_analysis(self.output()/'reports/comparison/runs')
        self.assertEqual(result['controls']['comparisons']['B']['status'], 'DIFFERENT')
        self.assertEqual(result['time_alignment']['status'], 'UNAVAILABLE')
        self.assertEqual(result['pairing'][1]['status'], 'SEPARATE')
        self.assertEqual(self.spec()['panels'][1]['series'][0]['points'][0]['x'], 5)
        self.assertIn('latency gap', json.dumps(self.spec()))

    def test_incompatible_units_are_displayed_separately(self):
        result = compare([self.bundle('a'), self.bundle('b', unit='seconds')], self.output())
        self.assertEqual(result['pairing'][0]['status'], 'SEPARATE')
        self.assertEqual(len(self.spec()['panels']), 2)

    def test_missing_controls_are_unknown_even_when_both_missing(self):
        paths = [write_bundle(self.root/name, 'run', name, {}, dict(title=name, panels=[]))
                 for name in ('a', 'b')]
        result = compare(paths, self.output())
        self.assertEqual(result['controls']['comparisons']['B']['status'], 'UNKNOWN')
        self.assertNotIn('verdict', result['runs']['A'])

    def test_corrupt_archive_fails_without_output_and_raw_evidence_is_not_analyzed(self):
        a, b = self.bundle('a'), self.bundle('b')
        (a/'analysis.json').write_text('{}')
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main([str(a), str(b), '--output', str(self.output())]), 2)
        self.assertFalse(self.output().exists())
        raw = self.root/'evidence.json'
        raw.write_text(json.dumps(evidence()))
        with self.assertRaisesRegex(ValueError, 'bundle directory'):
            compare([raw, b], self.output())

    def test_async_series_keep_their_own_sample_times(self):
        paths = []
        for name, times in [('a', [1, 2, 12]), ('b', [1.2, 2.2, 12.2])]:
            spec = dict(title=name, timeOriginLabel='run start', panels=[dict(
                id='metric', title='metric', timeX=True, axes={'y': dict(title='count')},
                series=[dict(name='metric', points=[dict(x=t,y=v) for t,v in zip(times,[3,None,4])])],
            )])
            paths.append(write_bundle(self.root/name, 'run', name, {}, spec))
        compare(paths, self.output())
        series = self.spec()['panels'][0]['series']
        self.assertEqual([p['x'] for p in series[0]['points']], [1, 2, 12])
        self.assertEqual([p['x'] for p in series[1]['points']], [1.2, 2.2, 12.2])
        self.assertIsNone(series[1]['points'][1]['y'])

    def test_origin_mismatch_is_not_silently_overlaid(self):
        paths = []
        for name in ('a', 'b'):
            paths.append(write_bundle(self.root/name, 'run', name, {}, dict(
                title=name, timeOriginLabel=name, panels=[dict(
                    id='metric', timeX=True, axes={'y': dict(title='count')},
                    series=[dict(name='metric', points=[dict(x=1,y=1)])],
                )],
            )))
        result = compare(paths, self.output())
        self.assertEqual(result['pairing'][0]['status'], 'SEPARATE')
        self.assertEqual(len(self.spec()['panels']), 2)

    def test_relative_run_links_are_rebased_without_copying_or_rebuilding_reports(self):
        a, b = self.bundle('a'), self.bundle('b')
        compare([a, b], self.output())
        base = self.output()/'reports/comparison/runs'
        sections = self.spec()['sections']
        targets = [(base/item['href']).resolve() for s in sections if s['type'] == 'links'
                   for item in s['items']]
        self.assertEqual(targets, [(a/'report.html').resolve(), (b/'report.html').resolve()])

    def test_comparison_bundle_cannot_be_reused_as_a_run(self):
        a, b = self.bundle('a'), self.bundle('b')
        compare([a, b], self.output())
        with self.assertRaisesRegex(ValueError, 'run bundle'):
            compare([self.output()/'reports/comparison/runs', a], self.output('nested'))

    def test_performance_outcomes_and_configuration_are_frozen(self):
        a, b = evidence(), evidence()
        b['criteria']['min_output_tps'] = 10000
        b['provenance']['actual_master_config']['dispatcher']['type'] = 'NON_BATCH'
        paths = [report(self.root/name, e, analyze_performance(e)) for name, e in (('a', a), ('b', b))]
        expected = [load_analysis(p) for p in paths]
        a['criteria']['min_output_tps'] = 10000
        with mock.patch('cases.master_performance.analysis.analyze', side_effect=AssertionError('reanalysis')), \
             mock.patch('cases.master_performance.report.write_report', side_effect=AssertionError('rebuild')):
            result = compare(paths, self.output())
        self.assertEqual(list(result['runs'].values()), expected)
        self.assertEqual([v['verdict'] for v in result['runs'].values()], ['PASS', 'FAIL'])
        self.assertTrue(result['controls']['comparisons']['B']['differences'])
        self.assertEqual(len(self.spec()['panels']), 18)
        self.assertEqual({panel['id'] for panel in self.spec()['panels']
                          if panel['id'].startswith('overlay:')},
                         {'overlay:engine-tps', 'overlay:client-qps',
                          'overlay:latency', 'overlay:cache-hit',
                          'overlay:prefill-batch', 'overlay:prefill-state'})

    def test_cache_outcomes_and_prepared_curves_are_frozen(self):
        paths, expected = [], []
        for name, hit in (('a', .1), ('b', .8)):
            e = cache_fixtures.CacheGateTest().evidence(hit)
            e['events'] = [dict(name='checkpoint', t=20)]
            result = analyze_cache(e)
            expected.append(result)
            root = self.root/name
            write_report(root, e, result)
            paths.append(root/'reports/run/cache-scale-in')
        with mock.patch('cases.cache_scale_in.analysis.analyze', side_effect=AssertionError('reanalysis')), \
             mock.patch('cases.cache_scale_in.report.prepare_report', side_effect=AssertionError('rebuild')):
            result = compare(paths, self.output(), alignment_event='checkpoint')
        self.assertEqual(list(result['runs'].values()), expected)
        self.assertEqual([v['verdict'] for v in result['runs'].values()], ['FAIL', 'PASS'])
        reversed_result = compare(paths[::-1], self.output('reversed'))
        self.assertEqual([v['verdict'] for v in reversed_result['runs'].values()], ['PASS', 'FAIL'])


if __name__ == '__main__':
    unittest.main()
