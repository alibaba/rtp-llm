import copy
import json
import re
import unittest
from pathlib import Path

from migrate_metrics import alert_rules, migrate_dashboard, migrate_query


CONFIG = json.loads(Path(__file__).with_name('metric_migration.json').read_text())


class MetricMigrationTest(unittest.TestCase):
    def test_removes_model_only_from_affected_metric_selectors(self):
        query = ('max({__name__="whale-lb.app.cache.total.kv.cache.tokens",model="old",'
                 'role="PREFILL",host=~"${master_host:regex}"}) + '
                 'max({__name__="unrelated.metric",model="keep"})')
        migrated, _ = migrate_query(query, CONFIG)
        self.assertIn('whale-lb.app.cache.total.kv.cache.tokens', migrated)
        self.assertNotIn('model="old"', migrated)
        self.assertIn('model="keep"', migrated)
        self.assertIn('host=~"${master_host:regex}"', migrated)

    def test_legacy_gauge_becomes_a_window_increment_with_its_filters(self):
        query = 'avg({__name__="whale-lb.app.cache.hit.count",role="PREFILL",BIZ_NAME="dash_pd"})'
        migrated, _ = migrate_query(query, CONFIG)
        self.assertEqual('sum by (${group_by:csv}) (increase({__name__="whale-lb.app.cache.hit.count",'
                         'role="PREFILL",BIZ_NAME="dash_pd"}[1m]))', migrated)

    def test_preserves_existing_weighted_ratio_and_rewrites_old_aliases(self):
        query = ('sum(increase({__name__="whale-lb.app.cache.recent.key.hit.count"}[5m])) / '
                 'sum(increase({__name__="whale-lb.app.cache.recent.key.total.count"}[5m]))')
        migrated, _ = migrate_query(query, CONFIG)
        self.assertIn('app.cache.theory.hit.count', migrated)
        self.assertIn('app.cache.theory.total.count', migrated)
        self.assertEqual(2, migrated.count('[5m]'))
        self.assertNotIn('[1m]', migrated)

    def test_ack_filters_match_real_worker_addresses_and_pdfusion(self):
        query = '{__name__="whale-lb.app.flexlb.ack.to.response.time.ms",engineIp="10.0.0.1",role="PREFILL"}'
        migrated, _ = migrate_query(query, CONFIG)
        self.assertIn('engineIp=~' + json.dumps(r'10\.0\.0\.1:[0-9]+(@[0-9]+)?'), migrated)
        self.assertIn('role=~"PREFILL|PDFUSION"', migrated)

    def test_rejects_non_equivalent_settle_miss_alert_migration(self):
        with self.assertRaisesRegex(ValueError, 'not an equivalent signal'):
            migrate_query('{__name__="whale-lb.auto_tpm.inflight_settle_miss.count",kind="yielded"}', CONFIG)

    def test_refuses_to_guess_an_old_multi_counter_formula(self):
        query = ('avg({__name__="whale-lb.app.cache.theory.hit.count"}) / '
                 'avg({__name__="whale-lb.app.cache.theory.total.count"})')
        with self.assertRaisesRegex(ValueError, 'weighted-ratio'):
            migrate_query(query, CONFIG)

    def test_does_not_rewrite_unrelated_metric_name_suffixes(self):
        query = '{__name__="whale-lb.app.cache.hit.count.debug",model="keep"}'
        self.assertEqual((query, []), migrate_query(query, CONFIG))

    def test_preserves_original_export_and_migration_is_idempotent(self):
        dashboard = {
            'uid': CONFIG['dashboardUid'], 'version': 205,
            'panels': [{'id': 1, 'title': 'old', 'type': 'timeseries',
                        'targets': [{'expr': '{__name__="whale-lb.grpc.server.executor.caller.runs"}'}]}],
            'templating': {'list': [{'definition': 'label_values({__name__="whale-lb.app.cache.hit.count"},role)'}]},
        }
        original = copy.deepcopy(dashboard)
        migrated, report = migrate_dashboard(dashboard, CONFIG)
        self.assertEqual(original, dashboard)
        self.assertEqual(CONFIG['dashboardUid'], migrated['uid'])
        self.assertEqual(4, len(report['addedPanels']))
        self.assertEqual('gRPC 任务拒绝数（最近 1 分钟）', migrated['panels'][0]['title'])
        self.assertTrue(migrated['templating']['list'][0]['definition'].startswith('label_values('))
        self.assertEqual(migrated, migrate_dashboard(migrated, CONFIG)[0])

    def test_generated_alerts_have_concrete_biz_scope_and_counter_windows(self):
        rules = alert_rules(CONFIG, 'dash_pd')['groups'][0]['rules']
        for rule in rules:
            self.assertIn('BIZ_NAME="dash_pd"', rule['expr'])
            self.assertIn('increase(', rule['expr'])
            self.assertNotIn('${', rule['expr'])
            self.assertNotIn('preemption.target_invalid', rule['expr'])

    def test_existing_metric_names_are_preserved(self):
        for rule in CONFIG['rules']:
            self.assertEqual(rule['oldName'], rule['newName'])

    def test_keeps_valid_counter_and_worker_capacity_queries_unchanged(self):
        queries = [
            'sum by (role) (increase({ __name__ = "whale-lb.app.cache.hit.count", role = "PREFILL" }[1m]))  ',
            'max by (role,engineIp) ({__name__="whale-lb.app.cache.total.kv.cache.tokens",role="PREFILL"})',
            'sum(increase({__name__="whale-lb.app.cache.theory.hit.count"}[5m])) / '
            'sum(increase({__name__="whale-lb.app.cache.theory.total.count"}[5m]))',
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertEqual((query, []), migrate_query(query, CONFIG))

    def test_labels_decode_reservations_and_variance_units_without_changing_queries(self):
        expressions = [
            '{__name__="whale-lb.app.flexlb.decode.inflight.kv.reserved.tokens"}',
            '{__name__="whale-lb.app.engine.worker.info.running.query.len.var",role="PREFILL"}',
            '{__name__="whale-lb.app.engine.worker.info.running.query.len.var",role="DECODE"}',
            '{__name__="whale-lb.app.engine.worker.info.step.latency.var"}',
        ]
        dashboard = {'panels': [{'id': i, 'title': 'old', 'targets': [{'expr': expr}]}
                                for i, expr in enumerate(expressions, 1)]}
        migrated, _ = migrate_dashboard(dashboard, CONFIG)
        panels = migrated['panels'][:4]
        self.assertEqual(expressions, [p['targets'][0]['expr'] for p in panels])
        self.assertEqual('Decode KV 预留 Tokens（含排队）', panels[0]['title'])
        self.assertEqual(['suffix:work-ms²', 'suffix:count²', 'suffix:ms²'],
                         [p['fieldConfig']['defaults']['unit'] for p in panels[1:]])

    def test_all_retained_metric_names_exist_in_the_production_contract(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / 'flexlb-common/src/main/java/org/flexlb/constant/MetricConstant.java').read_text()
        names = set(re.findall(r'public static final String\s+\w+\s*=\s*"([^"]+)"', source))
        for rule in CONFIG['rules']:
            self.assertIn(rule['newName'], names)
        for alert in CONFIG['alerts']:
            self.assertIn(alert['metric'], names)


if __name__ == '__main__':
    unittest.main()
