"""Provision a native Grafana dashboard and saved execution annotations in an isolated session."""
import base64
import copy
from datetime import datetime, timezone
import json
from urllib.request import Request, urlopen

import yaml


DATASOURCE = {'type': 'prometheus', 'uid': 'mock-prometheus'}


def dashboard(value):
    if not isinstance(value, dict) or not isinstance(value.get('panels'), list) or not value.get('title'):
        raise ValueError('view must be a native Grafana dashboard JSON object with title and panels')
    result = copy.deepcopy(value)
    result.update(uid='mock-experiment', id=None, editable=False, refresh='')
    def bind(panels):
        for panel in panels:
            if panel.get('type') == 'row':
                bind(panel.get('panels', []))
                continue
            panel['datasource'] = dict(DATASOURCE)
            for target in panel.get('targets', []):
                target['datasource'] = dict(DATASOURCE)
    bind(result['panels'])
    result['annotations'] = {'list': [{'builtIn': 1, 'datasource': {'type': 'grafana', 'uid': '-- Grafana --'},
        'enable': True, 'hide': False, 'iconColor': '#ff9830', 'name': 'Phase / Step / 门禁', 'type': 'dashboard'}]}
    return result


def provision(directory, value, prometheus_url, start, end):
    for path in ('dashboards', 'provisioning/dashboards', 'provisioning/datasources'):
        (directory / path).mkdir(parents=True)
    value = dashboard(value)
    value['time'] = {key: datetime.fromtimestamp(value / 1000, timezone.utc).isoformat()
                     for key, value in [('from', start), ('to', end)]}
    (directory / 'dashboards/default.json').write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    def write(relative, content):
        (directory / relative).write_text(yaml.safe_dump(content), encoding='utf-8')
    write('provisioning/datasources/default.yaml', {'apiVersion': 1, 'datasources': [{
        **DATASOURCE, 'name': 'Experiment Prometheus', 'access': 'proxy', 'url': prometheus_url,
        'isDefault': True, 'jsonData': {'timeInterval': '1s'}}]})
    write('provisioning/dashboards/default.yaml', {'apiVersion': 1, 'providers': [{
        'name': 'experiment', 'options': {'path': str(directory / 'dashboards')}}]})


def annotations(url, password, events, cancelled):
    authorization = 'Basic ' + base64.b64encode(('admin:' + password).encode()).decode()
    for event in events:
        if cancelled.is_set():
            return
        body = {'dashboardUID': 'mock-experiment', 'time': event['time'], 'text': event['text'], 'tags': ['experiment']}
        if event.get('timeEnd') is not None:
            body['timeEnd'] = event['timeEnd']
        request = Request(url + '/api/annotations', json.dumps(body).encode(),
                          {'Content-Type': 'application/json', 'Authorization': authorization})
        with urlopen(request, timeout=3) as response:
            json.load(response)
