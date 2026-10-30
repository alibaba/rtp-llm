"""Dashboard contracts: role isolation, honest coverage, units and collapsed rows."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def dashboard():
    return json.loads((ROOT / 'views/default.json').read_text())


def panels():
    return [panel for row in dashboard()['panels'] for panel in row['panels']]


def test_native_dashboard_has_unique_ids_valid_layout_and_resolved_filter_variables():
    value = dashboard()
    variables = {variable['name'] for variable in value['templating']['list']}
    ids = set()
    for row in value['panels']:
        assert row['type'] == 'row' and row['collapsed'] is True
        assert row['panels'], row['title']
        for panel in [row, *row['panels']]:
            assert panel['id'] not in ids
            ids.add(panel['id'])
            pos = panel['gridPos']
            assert 0 <= pos['x'] < 24 and pos['x'] + pos['w'] <= 24 and pos['h'] > 0
            if panel['type'] == 'row':
                continue
            assert panel['fieldConfig']['defaults']['custom']['spanNulls'] is False
            for target in panel['targets']:
                assert set(re.findall(r'\$\{(\w+)(?::\w+)?\}', target['expr'])) <= variables
                assert ':regex}' not in target['expr']
                assert '127.0.' not in target['expr']
                assert 'or vector(0)' not in target['expr']
                assert 'role=null' not in target['expr']


def test_engine_roles_do_not_mix_compute_queue_or_cache_observations():
    for row in dashboard()['panels']:
        role = 'prefill' if 'Prefill' in row['title'] else 'decode' if 'Decode' in row['title'] else None
        if role is None:
            continue
        expressions = '\n'.join(t['expr'] for p in row['panels'] for t in p['targets'])
        assert f'role="{role}"' in expressions
        assert f'role="{"decode" if role == "prefill" else "prefill"}"' not in expressions
        for p in row['panels']:
            for t in p['targets']:
                assert f'role="{role}"' in t['expr'], (row['title'], p['title'])
        if role == 'decode':
            assert 'rtp_llm_cache_lookup_tokens_total' not in expressions
            assert 'rtp_llm_context_wall_tps{' not in expressions
        else:
            assert 'rtp_llm_generate_tps{' not in expressions
            assert 'rtp_llm_sp_avg_accept_rate{' not in expressions


def test_master_follows_rtpllm_sections_and_preserves_worker_role_filters():
    rows = dashboard()['panels']
    assert len([r for r in rows if r['title'].startswith('FlexLB ·')]) == 7
    assert not any('兼容指标' in r['title'] for r in rows)
    items = {p['id']: p for p in panels()}
    expected = items[410260]
    assert expected['title'] == 'Worker 剩余工作预计还需多久（排队＋运行）'
    assert 'role="PREFILL"' in expected['targets'][0]['expr']
    assert 'role="PDFUSION"' in expected['targets'][1]['expr']
    assert 'flexlb_app_engine_health_check_running_queue_time' in expected['targets'][0]['expr']
    for p in panels():
        for link in p.get('links', []):
            assert '/d/iZ5Q_81nz?' in link['url']
            assert 'fv0CDqzDk' not in link['url']


def test_available_master_gauges_and_timers_keep_distinct_semantics():
    items = {p['id']: p for p in panels()}
    network = items[410219]
    assert all('_seconds_' not in t['expr'] for t in network['targets'])
    ack = items[410225]
    assert '_seconds_sum' in ack['targets'][0]['expr']
    assert '_seconds_max' in ack['targets'][1]['expr']
    assert ack['fieldConfig']['defaults']['unit'] == 'ms'
    pools = next(p for p in panels() if p['title'] == 'Master JVM 内存池 · 已用 / 上限')
    assert 'by(${master_group},area,id)' in pools['targets'][0]['expr']


def test_vit_and_shared_process_resources_are_not_faked():
    rows = dashboard()['panels']
    vit = next(r for r in rows if r['title'] == 'ViT')
    assert len(vit['panels']) >= 20
    assert all('待埋点' in p['title'] for p in vit['panels'])
    assert all('role="vit"' in t['expr'] for p in vit['panels'] for t in p['targets'])
    host = next(r for r in rows if r['title'].startswith('Engine 宿主'))
    assert all(not p['targets'] for p in host['panels'])
    assert all('共享 JVM' in p['description'] for p in host['panels'])
    kv = [p for p in panels() if 'Host KV 容量' in p['title']]
    assert len(kv) == 2
    assert all('rtp_llm_host_kv_tokens' in p['targets'][0]['expr'] for p in kv)
    assert all('JVM 堆' in p['description'] for p in kv)


def test_local_legends_use_prometheus_labels_and_respect_aggregation():
    for panel in panels():
        for target in panel['targets']:
            legend = target.get('legendFormat', '')
            assert '[[tag_' not in legend, panel['title']
            assert not re.search(r'\{\{\s*(host|host_ip|container_ip)\s*\}\}', legend)
            for engine_role in ('prefill', 'decode', 'vit'):
                if '${' + engine_role + '_instance}' not in target['expr']:
                    continue
                for group in ('role', 'instance'):
                    expression = target['expr'].replace('${' + engine_role + '_group}', group)
                    resolved_legend = legend.replace('${' + engine_role + '_group}', group)
                    groups = re.findall(r'\bby\s*\(([^)]*)\)', expression)
                    retained = set(groups[0].split(',')) if groups else set()
                    assert set(re.findall(r'\{\{(\w+)\}\}', resolved_legend)) <= retained
    items = {p['id']: p for p in panels()}
    forward = items[100231]['targets'][0]
    assert forward['legendFormat'] == '{{${prefill_group}}}'
    assert forward['expr'].startswith('avg by(${prefill_group})(')
    decode_forward = items[200231]['targets'][0]
    assert decode_forward['legendFormat'] == '{{${decode_group}}}'
    assert decode_forward['expr'].startswith('avg by(${decode_group})(')
    cache = items[110587]['targets'][0]
    assert 'cache_dependency={{cache_dependency}}' in cache['legendFormat']
    assert 'loading_entered={{loading_entered}}' in cache['legendFormat']
    master = items[410260]['targets'][0]
    assert '{{instance}}' in master['legendFormat']
    assert 'instance' in re.findall(r'\bby\s*\(([^)]*)\)', master['expr'])[0].split(',')


def test_engine_selection_controls_query_grouping_and_legend_together():
    variables = {v['name']: v for v in dashboard()['templating']['list']}
    assert 'engine_instance' not in variables and 'engine_group' not in variables
    for role in ('prefill', 'decode', 'vit'):
        assert variables[role + '_instance']['allValue'] == '.*'
        grouping = variables[role + '_group']
        assert grouping['hide'] == 2 and grouping['type'] == 'query'
        assert '${' + role + '_instance}' in grouping['query']
        assert '"role"' in grouping['query'] and '"instance"' in grouping['query']
    for panel in panels():
        for target in panel['targets']:
            assert '${engine_instance}' not in target['expr']
            roles = re.findall(r'\$\{(prefill|decode|vit)_instance\}', target['expr'])
            if not roles:
                continue
            assert len(set(roles)) == 1
            role = roles[0]
            assert 'role="' + role + '"' in target['expr']
            groups = re.findall(r'\bby\s*\(([^)]*)\)', target['expr'])
            assert groups and all('${' + role + '_group}' in g.split(',') for g in groups)
            assert '{{${' + role + '_group}}}' in target['legendFormat']


def test_real_engine_observations_return_to_rtp_panels_and_mock_rows_only_compare_simulation():
    items = {p['id']: p for p in panels()}
    for offset in (100000, 200000):
        for source_id in (127, 128, 142, 154, 173, 174, 231):
            panel = items[offset + source_id]
            assert '待埋点' not in panel['title']
            assert '尚未导出' not in panel['description']
        assert items[offset + 154]['fieldConfig']['defaults']['unit'] == 'percent'
    rows = dashboard()['panels']
    for role in ('Prefill', 'Decode'):
        quality = next(r for r in rows if r['title'] == role + ' · Mock 模拟质量')
        assert len(quality['panels']) == 3
        assert all('mock_model_forward_planned_us' in t['expr'] or 'rtp_llm_model_forward_seconds_sum' in t['expr']
                   for p in quality['panels'] for t in p['targets'] if 'rtp_llm_model_forward_us' not in t['expr'])
    assert not any('mock_engine_' in t['expr'] for p in panels() for t in p['targets'])
    assert 'rtp_llm_context_tps{' in items[110379]['targets'][0]['expr']
    assert 'rtp_llm_generate_tps{' in items[210369]['targets'][0]['expr']


def test_custom_panels_follow_their_own_address_selection_without_losing_business_dimensions():
    variables = {v['name']: v for v in dashboard()['templating']['list']}
    for kind in ('prefill', 'decode', 'vit', 'master', 'client'):
        group = variables[kind + '_group']
        assert group['hide'] == 2 and group['type'] == 'query'
        assert '${' + kind + '_instance}' in group['query']
    for panel in panels():
        if panel.get('links'):
            continue
        for target in panel['targets']:
            for kind in ('prefill', 'decode', 'vit', 'master', 'client'):
                if '${' + kind + '_instance}' not in target['expr']:
                    continue
                group = '${' + kind + '_group}'
                assert group in target['expr'], panel['title']
                assert '{{' + group + '}}' in target['legendFormat'], panel['title']
    items = {p['id']: p for p in panels()}
    assert 'by(${client_group},le)' in items[5]['targets'][1]['expr']
    assert 'by(${client_group},outcome)' in items[3]['targets'][0]['expr']
    assert 'avg by(${master_group},area)(sum by(instance,application,area)' in items[56]['targets'][0]['expr']
    assert 'by(${master_group},area,id)' in items[190]['targets'][0]['expr']
    assert 'by(${master_group},state)' in items[59]['targets'][0]['expr']
