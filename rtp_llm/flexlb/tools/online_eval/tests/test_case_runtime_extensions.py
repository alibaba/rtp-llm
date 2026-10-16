"""Extension trials execute the public contracts, including ownership and cleanup."""
import json
import time
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from case_api.registry import CaseDefinition, CaseRegistry, RegisteredCase, ClientFactory, SnapshotSource
from scenario.context import RuntimeContext
from scenario.contracts import PlanContext
from scenario.actions.observation import validate_snapshot
from adapters.observation_sources import Sources
from runtime.deadline import Deadline


class MemoryClient:
    def __init__(self, value):
        self.value, self.closed = value, False
    def start(self, deadline):
        deadline.check()
    def cleanup(self, deadline):
        deadline.check()
        self.closed = True


class StateReader:
    def __init__(self):
        self.closed = False
    def read(self, deadline):
        deadline.check()
        return {'pending': 7}
    def close(self, deadline):
        self.closed = True


def snapshot_registry(tmp_path):
    definition = CaseDefinition({'default': lambda case: None},
        clients={'stream': ClientFactory(lambda services, params, deadline: MemoryClient(params['value']))},
        snapshot_sources={'state': SnapshotSource(lambda services, params: StateReader())})
    return CaseRegistry.build([RegisteredCase(name, 'fixture.' + name, tmp_path / 'program.py', definition)
                              for name in ('first', 'second')])


def test_local_extension_names_execute_without_common_dispatch_branch(tmp_path):
    with patch('case_api.registry._snapshot', snapshot_registry(tmp_path)):
        ctx = RuntimeContext({'implementation': {'configuration': {'case': 'first'}}}, None,
                             tmp_path, time.monotonic, time.sleep)
        ctx.env = SimpleNamespace(master_http_port=1, mock_http_port=2)
        deadline = Deadline(time.monotonic() + 5)
        handle = ctx.services.client('stream', {'value': 3}, deadline)
        client = ctx.resource(handle, 'client')
        assert client.value == 3
        params = validate_snapshot({'sources': ['state']}, PlanContext('stage', {}, case='first'))
        source = Sources(ctx, params)
        payload = json.loads(source.capture(deadline).encoded)
        assert payload['sources']['state']['data'] == {'pending': 7}
        assert payload['sources']['state']['status'] == 'ok'
        reader = source.custom['state']
        assert all(row['status'] == 'PASS' for row in ctx.cleanup(5))
        assert reader.closed and client.closed


def test_undeclared_source_and_wrong_factory_signature_are_rejected(tmp_path):
    with patch('case_api.registry._snapshot', snapshot_registry(tmp_path)):
        with pytest.raises(ValueError, match='supported source'):
            validate_snapshot({'sources': ['unknown']}, PlanContext('stage', {}, case='first'))
    definition = CaseDefinition({'default': lambda case: None}, clients={'bad': ClientFactory(lambda x: x)})
    with pytest.raises(ValueError, match='3 positional'):
        CaseRegistry.build([RegisteredCase('bad', 'bad.program', tmp_path, definition)])


def test_client_is_owned_before_partial_start_failure(tmp_path):
    class Partial(MemoryClient):
        def start(self, deadline):
            self.acquired = True
            raise RuntimeError('startup interrupted')
    client = Partial(1)
    definition = CaseDefinition({'default': lambda case: None}, clients={
        'stream': ClientFactory(lambda services, params, deadline: client)})
    owner = RegisteredCase('trial', 'trial.program', tmp_path / 'program.py', definition)
    with patch('case_api.registry._snapshot', CaseRegistry.build([owner])):
        ctx = RuntimeContext({'implementation': {'configuration': {'case': 'trial'}}}, None,
                             tmp_path, time.monotonic, time.sleep)
        with pytest.raises(RuntimeError, match='startup interrupted'):
            ctx.services.client('stream', {}, Deadline(time.monotonic()+1))
        assert client.acquired and ctx.resource_count == 1
        assert ctx.cleanup(1)[0]['status'] == 'PASS'
        assert client.closed


def test_client_record_decoder_does_not_relabel_invalid_or_stale_rows():
    from case_api.client_records import ClientRecordAdapter, read_records
    adapter = ClientRecordAdapter(lambda: [{'raw': 1}], lambda row: {
        'client_record_schema_version': 1, 'env_epoch': 2, 'wire_request_id': row['raw'], 'attempt': 1})
    deadline = Deadline(time.monotonic() + 5)
    assert read_records(adapter, deadline, 2)[0]['wire_request_id'] == 1
    with pytest.raises(ValueError, match='stale'):
        read_records(adapter, deadline, 3)


def test_snapshot_timeout_is_preserved_as_incomplete_evidence(tmp_path):
    class Expired:
        def read(self, deadline):
            raise TimeoutError('source budget exhausted')
    definition = CaseDefinition({'default': lambda case: None},
        snapshot_sources={'state': SnapshotSource(lambda services, params: Expired())})
    owner = RegisteredCase('trial', 'trial.program', tmp_path / 'program.py', definition)
    with patch('case_api.registry._snapshot', CaseRegistry.build([owner])):
        ctx = RuntimeContext({'implementation': {'configuration': {'case': 'trial'}}}, None,
                             tmp_path, time.monotonic, time.sleep)
        ctx.env = object()  # A custom source does not require Mock HTTP ports.
        params = validate_snapshot({'sources': ['state']}, PlanContext('stage', {}, case='trial'))
        payload = json.loads(Sources(ctx, params).capture(Deadline(time.monotonic() + 5)).encoded)
        assert payload['status'] == 'partial'
        assert payload['sources']['state']['errors'][0]['type'] == 'TimeoutError'


def test_registered_client_records_use_the_common_snapshot_action(tmp_path):
    class Client(MemoryClient):
        def snapshot_records(self):
            return [dict(client_record_schema_version=1, env_epoch=self.value,
                         wire_request_id=1, attempt=1)]
    definition = CaseDefinition({'default': lambda case: None}, clients={
        'stream': ClientFactory(lambda services, params, deadline: Client(services.epoch))})
    owner = RegisteredCase('trial', 'trial.program', tmp_path / 'program.py', definition)
    with patch('case_api.registry._snapshot', CaseRegistry.build([owner])):
        ctx = RuntimeContext({'implementation': {'configuration': {'case': 'trial'}}}, None,
                             tmp_path, time.monotonic, time.sleep)
        ctx.env = object()
        deadline = Deadline(time.monotonic() + 5)
        handle = ctx.services.client('stream', {}, deadline)
        plan = PlanContext('stage', {'start': {'client': 'client'}}, case='trial')
        params = validate_snapshot({'sources': ['client_records'],
            'client': {'$ref': 'stages.start.output.client'}}, plan)
        params['client'] = handle
        payload = json.loads(Sources(ctx, params).capture(deadline).encoded)
        assert payload['sources']['client_records']['data'][0]['wire_request_id'] == 1
        assert payload['status'] == 'ok'
        ctx.cleanup(5)


def test_selected_report_receives_the_live_render_budget(tmp_path):
    from reporting.workload import write_views
    from case_api.registry import ReportView
    deadline = Deadline(time.monotonic() + 5)
    seen = []
    def render(directory, analysis, presentation, budget):
        budget.check()
        seen.append(budget)
        return directory / 'report.html'
    capability = ReportView(lambda *args: None, lambda *args: None, deadline_renderer=render)
    with patch('reporting.workload.view_capabilities', return_value={'trial.yaml': capability}), \
         patch('reporting.workload.view', return_value=object()):
        assert write_views(tmp_path, {}, ['trial.yaml'], deadline=deadline)['trial.yaml'] == tmp_path / 'report.html'
    assert seen == [deadline]
