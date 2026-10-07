#!/usr/bin/env python3
"""Update Grafana queries and alert rules while keeping FlexLB metric names.

Reads exported JSON files and writes local files only. No network requests.
"""

import argparse
import copy
import ipaddress
import json
import re
from pathlib import Path


SELECTOR = re.compile(r'\{(?:[^{}"]|"(?:\\.|[^"\\])*")*\}')
MATCHER = re.compile(r'\s*(\w+)\s*(=~|!~|!=|=)\s*("(?:\\.|[^"\\])*")\s*')
LABELS = re.compile(r'(?:[^,"]|"(?:\\.|[^"\\])*")+')


def migrate_query(query, config, convert_counters=True):
    prefix = config['metricPrefix']
    rules = {prefix + name: rule for rule in config['rules']
             for name in [rule['oldName'], *rule.get('aliases', [])]}
    retired = [prefix + rule['name'] for rule in config['retiredMetrics']]
    if any(name in query for name in retired):
        raise ValueError('Retire the old settle-miss alert separately; lifecycle failures are not an equivalent signal')
    changes = []
    counter_selectors = []
    counter_names = []

    def update_selector(match):
        labels = [label.strip() for label in LABELS.findall(match[0][1:-1])]
        parsed = [MATCHER.fullmatch(label) for label in labels]
        rule = None
        for label in parsed:
            if label and label[1] == '__name__' and label[2] == '=':
                rule = rules.get(json.loads(label[3]))
        if rule is None:
            return match[0]
        updated = []
        selector_changed = False
        for text, label in zip(labels, parsed):
            if not label:
                updated.append(text)
                continue
            name, operator, value = label.groups()
            if name in rule.get('removeLabels', []):
                selector_changed = True
                continue
            if name == '__name__':
                current_name = json.loads(value)
                replacement = prefix + rule['newName']
                if current_name != replacement:
                    value = json.dumps(replacement)
                    selector_changed = True
            if rule.get('includePdfusion') and name == 'role' and operator == '=' and json.loads(value) == 'PREFILL':
                operator, value = '=~', '"PREFILL|PDFUSION"'
                selector_changed = True
            if rule.get('workerAddress') and name == 'engineIp' and operator == '=':
                address = json.loads(value)
                try:
                    ipaddress.IPv4Address(address)
                except ipaddress.AddressValueError:
                    pass
                else:
                    operator = '=~'
                    value = json.dumps(re.escape(address) + r':[0-9]+(@[0-9]+)?')
                    selector_changed = True
            updated.append(name + operator + value)
        selector = '{' + ','.join(updated) + '}' if selector_changed else match[0]
        if selector_changed:
            changes.append(rule['newName'])
        if rule['type'] == 'COUNTER':
            counter_selectors.append(selector)
            counter_names.append(rule['newName'])
        return selector

    migrated = SELECTOR.sub(update_selector, query)
    if convert_counters and counter_selectors and not re.search(r'\b(?:increase|i?rate)\s*\(', migrated):
        if len(counter_selectors) != 1:
            raise ValueError('A legacy multi-counter expression needs an explicit weighted-ratio query')
        migrated = 'sum by (${group_by:csv}) (increase(' + counter_selectors[0] + '[1m]))'
        changes.extend(counter_names)
    return migrated, changes


def panels_in(panels):
    for panel in panels:
        yield panel
        yield from panels_in(panel.get('panels', []))


def migrate_dashboard(export, config):
    result = copy.deepcopy(export)
    dashboard = result.get('dashboard', result)
    changes = []

    def update_queries(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('expr', 'query', 'definition') and isinstance(item, str):
                    value[key], names = migrate_query(item, config, convert_counters=key == 'expr')
                    changes.extend(names)
                else:
                    update_queries(item)
        elif isinstance(value, list):
            for item in value:
                update_queries(item)

    update_queries(dashboard)
    panels = list(panels_in(dashboard['panels']))
    def describe_panel(panel):
        expressions = '\n'.join(target.get('expr', '') for target in panel.get('targets', []))
        for rule in config['rules']:
            if config['metricPrefix'] + rule['newName'] in expressions:
                note = rule.get('note', '')
                description = panel.get('description', '')
                if note and note not in description:
                    panel['description'] = (description + '\n' + note).strip()
        if config['metricPrefix'] + 'grpc.server.executor.caller.runs' in expressions:
            panel['title'] = 'gRPC 任务拒绝数（最近 1 分钟）'
        if 'app.flexlb.decode.inflight.' in expressions and 'kv.reserved.tokens' in expressions:
            panel['title'] = 'Decode KV 预留 Tokens（含排队）'
        if config['metricPrefix'] + 'app.engine.worker.info.running.query.len.var' in expressions:
            if 'role="PREFILL"' in expressions:
                panel['title'] = 'Prefill 负载方差（work-ms²）'
                panel.setdefault('fieldConfig', {}).setdefault('defaults', {})['unit'] = 'suffix:work-ms²'
            elif 'role="DECODE"' in expressions:
                panel['title'] = 'Decode 活动任务数方差（count²）'
                panel.setdefault('fieldConfig', {}).setdefault('defaults', {})['unit'] = 'suffix:count²'
        if config['metricPrefix'] + 'app.engine.worker.info.step.latency.var' in expressions:
            panel['title'] = 'Step 延迟方差（ms²）'
            panel.setdefault('fieldConfig', {}).setdefault('defaults', {})['unit'] = 'suffix:ms²'

    for panel in panels:
        describe_panel(panel)

    titles = {panel.get('title') for panel in panels}
    next_id = max((panel.get('id', 0) for panel in panels), default=0) + 1
    additions = []
    for spec in config['newPanels']:
        if spec['title'] in titles:
            continue
        panel = {
            'id': next_id, 'title': spec['title'], 'description': spec['description'],
            'type': 'timeseries', 'datasource': config['datasource'],
            'gridPos': {'h': 8, 'w': 12, 'x': len(additions) % 2 * 12, 'y': len(additions) // 2 * 8},
            'targets': [{'refId': 'A', 'expr': spec['expr'], 'editorMode': 'code',
                         'range': True, 'instant': False, 'datasource': config['datasource'],
                         'legendFormat': spec.get('legend', ''), 'tenant': 'default'}],
            'fieldConfig': {'defaults': {'unit': spec.get('unit', 'short')}, 'overrides': []},
        }
        describe_panel(panel)
        additions.append(panel)
        next_id += 1
    if additions:
        bottom = max((p.get('gridPos', {}).get('y', 0) + p.get('gridPos', {}).get('h', 0)
                      for p in panels), default=0)
        row = {'id': next_id, 'title': '发现状态与生命周期异常', 'type': 'row',
               'collapsed': True, 'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': bottom}, 'panels': additions}
        for panel in additions:
            panel['gridPos']['y'] += bottom + 1
        dashboard['panels'].append(row)
    return result, {'metrics': sorted(set(changes)), 'addedPanels': [p['title'] for p in additions]}


def alert_rules(config, biz_name):
    rules = []
    for alert in config['alerts']:
        series = '{__name__=' + json.dumps(config['metricPrefix'] + alert['metric']) + ',BIZ_NAME=' + json.dumps(biz_name) + '}'
        rules.append({
            'alert': alert['name'],
            'expr': 'sum by (BIZ_NAME,host,stage) (increase(' + series + '[5m])) > 0',
            'for': '0m', 'labels': {'severity': 'warning'},
            'annotations': {'summary': alert['summary']},
        })
    return {'groups': [{'name': 'flexlb-lifecycle-and-executors', 'rules': rules}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('metric_migration.json'))
    parser.add_argument('--alerts-output', type=Path)
    parser.add_argument('--biz-name', default='dash_pd')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Keep the original export; input and output must differ')
    config = json.loads(args.config.read_text())
    dashboard, report = migrate_dashboard(json.loads(args.input.read_text()), config)
    args.output.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + '\n')
    if args.alerts_output:
        args.alerts_output.write_text(json.dumps(alert_rules(config, args.biz_name), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
