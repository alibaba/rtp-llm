import asyncio
from copy import deepcopy
import io
import json
from types import SimpleNamespace

import pytest

from mock_framework.config import Registry
from mock_framework.execution.transport import ExecutionError
from mock_framework.interfaces import master


def ledger():
    return {'scheduler_inflight': 0, 'prefill_endpoints': [{'inflight_requests': 0}],
            'decode_endpoints': [{'master_queued': 0, 'confirmed_running': 0}]}


def test_idle_reads_all_selected_native_masters_and_rejects_any_busy_owner(monkeypatch):
    data = {'http://a': ledger(), 'http://b': ledger()}
    nodes = [dict(id=f'master-{i}', rpc=f'{name}:1', http=f'http://{name}') for i, name in enumerate(('a', 'b'))]
    calls = []
    def request(req, timeout):
        calls.append(req.full_url)
        assert req.get_method() == 'GET' and timeout <= 2
        return io.BytesIO(json.dumps(data[req.full_url.removesuffix('/rtp_llm/inflight_status')]).encode())
    monkeypatch.setattr(master, 'urlopen', request)
    ctx = SimpleNamespace(timeout=3, runtime=SimpleNamespace(topology=SimpleNamespace(active=lambda role: nodes)))
    query = Registry.standard().get('master/idle')
    assert asyncio.run(query.invoke(ctx, {})) is True
    assert calls == ['http://a/rtp_llm/inflight_status', 'http://b/rtp_llm/inflight_status']
    for section, field in [(None, 'scheduler_inflight'), ('prefill_endpoints', 'inflight_requests'),
                           ('decode_endpoints', 'master_queued'), ('decode_endpoints', 'confirmed_running')]:
        data['http://a'] = ledger()
        row = data['http://a'] if section is None else data['http://a'][section][0]
        row[field] = 1
        assert asyncio.run(query.invoke(ctx, {})) is False
        assert asyncio.run(query.invoke(ctx, {'targets': ['b:1']})) is True


@pytest.mark.parametrize('change', [lambda d: d.pop('scheduler_inflight'),
    lambda d: d.update(prefill_endpoints=None), lambda d: d['decode_endpoints'][0].pop('master_queued'),
    lambda d: d.update(scheduler_inflight=True), lambda d: d.update(scheduler_inflight=-1)])
def test_missing_or_invalid_owner_counts_never_become_zero(monkeypatch, change):
    data = deepcopy(ledger()); change(data)
    monkeypatch.setattr(master, 'urlopen', lambda *args, **kwargs: io.BytesIO(json.dumps(data).encode()))
    with pytest.raises(ExecutionError):
        master.inflight('http://master', 1)
