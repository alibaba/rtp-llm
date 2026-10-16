"""Reject invalid workloads and distinguish healthy survivors from planned victims."""

import copy
import json
from pathlib import Path

import pytest

from cases.cache_sync_recovery.actions import atomic_write, withdraw
from cases.cache_sync_recovery.health import collateral, converged
from document_io import load_document
from flexlb_cfg import ConfigOverride, render_env
from scenario.program_compiler import configure_program

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_document(ROOT / "config/scenarios/cache_sync_recovery.yaml")


def row(generation=1, age=20):
    return dict(generation=generation, status_age_ms=age, reported_alive=True, role="prefill",
                cache_indexed=True, cache_key_size=306000, cache_version=2, cache_indexed_version=2)


def test_registered_workload_compiles_with_shared_monitoring_and_automatic_reports():
    plan = configure_program(config(), "cache_sync_recovery.yaml")
    assert [v["id"] for v in plan["variants"]] == ["steady", "recovery"]
    a, b = plan["variants"]
    assert all(v["metadata"]["kind"] == "workload" and v["reports"] == ["cache_sync_recovery.yaml"] for v in (a,b))
    assert [s["action"] for s in a["stages"]] == ["setup", "cache_recovery_cycle", "teardown"]
    assert a["environment"] == b["environment"] if "environment" in a else True
    left, right = copy.deepcopy(a["stages"][1]), copy.deepcopy(b["stages"][1])
    assert left["params"].pop("intervention") is False
    assert right["params"].pop("intervention") is True
    assert right.pop("timeout_s")-left.pop("timeout_s") == 20
    assert left == right


@pytest.mark.parametrize("section,key,value", [
    ("procedure", "delta_keys", 400000),
    ("procedure", "expired_workers", 120),
    ("procedure", "measurement_s", 0),
    ("observation", "max_gap_s", 11),
    ("checks", "min_added_keys_per_s", 2500),
])
def test_invalid_parameters_fail_before_launch(section, key, value):
    data = config()
    data["parameters"][section][key] = value
    with pytest.raises(ValueError):
        configure_program(data, "case.yaml")


def test_stale_absent_or_replaced_healthy_worker_is_not_hidden_by_victim_exclusion():
    baseline = {"victim": row(), "healthy": row()}
    assert collateral({"healthy": row()}, baseline, {"victim"}, 10000) == []
    for current in ({}, {"healthy": row(age=10001)}, {"healthy": row(generation=2)}):
        assert collateral(current, baseline, {"victim"}, 10000) == ["healthy"]


@pytest.mark.parametrize("extra", [{"applied_status_age_ms": 10001}, {"scheduling_status_fresh": False}])
def test_fresh_rpc_does_not_hide_stale_application(extra):
    current = dict(row(age=1), **extra)
    baseline = {"worker": row()}
    assert collateral({"worker": current}, baseline, set(), 10000) == ["worker"]
    assert not converged({"worker": current}, {"worker"}, 306000, {"worker": 2}, 10000)


def test_recovery_needs_exact_population_version_index_and_fresh_status():
    assert converged({"a": row()}, {"a"}, 306000, {"a": 2}, 10000)
    for field, value in (("cache_key_size", 305999), ("cache_indexed_version", 1),
                         ("cache_indexed", False), ("reported_alive", False), ("status_age_ms", -1)):
        current = row()
        current[field] = value
        assert not converged({"a": current}, {"a"}, 306000, {"a": 2}, 10000)
    assert not converged({}, {"a"}, 306000, {"a": 2}, 10000)


def test_withdrawal_preserves_other_roles_and_atomic_restore(tmp_path):
    original = json.dumps({"P": ["a", "b"], "D": ["c"]})
    hidden = withdraw(original, {"a"})
    assert json.loads(hidden) == {"P": ["b"], "D": ["c"]}
    with pytest.raises(ValueError):
        withdraw(original, {"missing"})
    path = tmp_path / "discovery.json"
    atomic_write(path, hidden)
    atomic_write(path, original)
    assert path.read_text() == original
    assert list(tmp_path.iterdir()) == [path]


def test_refresh_override_is_explicit_and_heap_parameters_are_frozen():
    default = json.loads(render_env("single-nonbatch"))
    assert "cacheStatus" not in default["workerRegistry"]
    rendered = json.loads(render_env("single-nonbatch", ConfigOverride(
        cache_min_refresh_ms=2400, cache_max_refresh_ms=2400)))
    assert rendered["workerRegistry"]["cacheStatus"] == {
        "minRefreshIntervalMs": 2400, "maxRefreshIntervalMs": 2400}
    with pytest.raises(ValueError):
        render_env("single-nonbatch", ConfigOverride(cache_min_refresh_ms=4000, cache_max_refresh_ms=2400))
    data = copy.deepcopy(config())
    data["environment"]["mock_heap"] = "8g; echo bad"
    with pytest.raises(ValueError):
        configure_program(data, "case.yaml")




def test_status_poll_override_is_distinct_from_rpc_timeout_and_validated():
    rendered = json.loads(render_env("single-nonbatch", ConfigOverride(
        status_poll_interval_ms=125, status_rpc_ms=5000)))
    health = rendered["workerRegistry"]["health"]
    assert health["statusPollIntervalMs"] == 125
    assert health["statusRpcTimeoutMs"] == 5000
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            ConfigOverride(status_poll_interval_ms=invalid)


def test_actual_intensity_uses_response_counters_and_elapsed_time():
    from cases.cache_sync_recovery.telemetry import interval_rates
    fields = ("responses", "full_responses", "unchanged_responses", "entries", "key_payload_bytes",
              "changed_responses", "added_keys", "removed_keys")
    before = {"a": dict.fromkeys(fields, 10), "b": dict.fromkeys(fields, 20)}
    after = {"a": dict.fromkeys(fields, 110), "b": dict.fromkeys(fields, 320)}
    rates = interval_rates(before, after, 20)
    assert rates["responses"] == 20
    assert rates["entries"] == 20
    assert rates["totals"]["added_keys"] == 400
    with pytest.raises(ValueError):
        interval_rates(after, before, 20)


def test_fingerprints_require_actual_membership_and_exact_population():
    from cases.cache_sync_recovery.telemetry import fingerprint_matches
    expected = {"a": dict(keys=5, sum=123, xor=42, version=9)}
    worker = dict(ip_port="a", cache_version=9, fingerprint=dict(indexed_keys=5,
        status_keys=5, sum=123, xor=42, missing_status=0, missing_global=0))
    assert fingerprint_matches(expected, {"prefill_endpoints": [worker]})
    for field in ("missing_global", "missing_status", "sum", "xor", "indexed_keys"):
        bad = copy.deepcopy(worker)
        bad["fingerprint"][field] += 1
        assert not fingerprint_matches(expected, {"prefill_endpoints": [bad]})
    assert not fingerprint_matches(expected, {"prefill_endpoints": []})


def test_report_preserves_gc_events_and_shared_time_origin(tmp_path):
    from cases.cache_sync_recovery.report import report
    instance = tmp_path/'instance'
    cache = instance/'cache-stability'
    cache.mkdir(parents=True)
    logs = instance/'environment/env1'
    logs.mkdir(parents=True)
    stamp = '2026-10-15T10:00:00.000+0000'
    import datetime
    epoch = datetime.datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()
    (logs/'master_gc.log').write_text(
        f'[{stamp}][2.0s][info][gc] GC(9) Pause Full (G1 Compaction Pause) 63G->20G(64G) 180.000ms\n')
    raw = dict(parameters=config()['parameters'], environment={}, errors=[], checks=[dict(
        id='healthy_survivors',status='PASS',actual=0)], intensity={'measurement': dict(
            responses=1,entries=306000,key_payload_bytes=100)}, expiry_scope='discovery',
        events=[dict(id='cache_only_started',epoch_s=epoch-120),
                dict(id='performance_observation_finished',epoch_s=epoch+2),
                dict(id='observation_finished',epoch_s=epoch+5)],
        frames=[dict(epoch_s=e,phase='measurement', master={},mock={},healthy_max_status_age_ms=age)
                for e,age in ((epoch-120,7),(epoch+1,None))])
    (cache/'evidence.json').write_text(json.dumps(raw))
    from cases.cache_sync_recovery.telemetry import gc_events
    from artifacts.gate import freeze_gate
    raw.update(gc=gc_events(logs/'master_gc.log'), intervention=False)
    freeze_gate(instance, 'cache_stability', raw, dict(verdict='PASS', validity='VALID',
                checks=raw['checks'], validity_checks=[]))
    output = tmp_path/'report'
    report(instance,output,'example')
    specs = list(output.rglob('report-spec.json'))
    assert len(specs) == 1
    spec = json.loads(specs[0].read_text())
    assert spec['timeOriginEpochS'] == epoch-120
    assert spec['timeAxis']['max'] == 123
    assert len(spec['panels']) == 2
    assert all(p['title'].startswith('steady · ') for p in spec['panels'])
    full = next(p for p in spec['panels'] if p['id']=='steady-gc')['series'][0]
    assert full['points'] == [dict(x=120,y=180)]
    assert all(len(p['series'])==1 for p in spec['panels'])
    from reporting.comparison import compare
    left = specs[0].parent
    right = report(instance,tmp_path/'second','second')
    compare([left,right],tmp_path/'comparison',alignment_event='cache_only_started')
    compared = json.loads((tmp_path/'comparison/reports/comparison/runs/report-spec.json').read_text())
    gc_overlay = [p for p in compared['panels'] if p['id'].startswith('overlay:')
                  and any(s['metric_id']=='steady.gc' for s in p['series'])]
    assert len(gc_overlay) == 1
    assert len(gc_overlay[0]['series']) == 2


def test_later_collection_error_preserves_positive_collateral_failure(tmp_path, monkeypatch):
    import contextlib
    from types import SimpleNamespace
    from cases.cache_sync_recovery import actions, telemetry
    import traffic.traffic_source as traffic
    engines = [dict(name=f'prefill-{i}',role='prefill',http_addr=f'h{i}',grpc_addr=f'p{i}') for i in range(3)]
    engines.append(dict(name='decode-0',role='decode',http_addr='hd',grpc_addr='pd'))
    discovery = tmp_path/'discovery.json'
    discovery.write_text(json.dumps(dict(P=['h0','h1','h2'],D=['hd'])))
    params = copy.deepcopy(config()['parameters'])
    params['procedure'].update(seed_interval_s=0,expired_workers=1)
    params['observation']['converged_samples'] = 1
    params['intervention'] = False
    time = [0]
    snapshots = []

    def control(ops, endpoint, deadline, body=None):
        if endpoint.startswith('snapshot'):
            return dict(engines=engines)
        return dict(engine=body['engine'],status='ok',fixture=dict(version=2))

    def fetch(url, timeout):
        if url.endswith('prometheus'):
            return b''
        snapshots.append(True)
        if len(snapshots)>2:
            raise TimeoutError('later snapshot unavailable')
        rows = [dict(ip_port=f'h{i}',**row(age=10001 if i==1 and len(snapshots)==2 else 20)) for i in range(3)]
        return json.dumps(dict(prefill_endpoints=rows,decode_endpoints=[dict(ip_port='hd',**row())],scheduler_inflight=0)).encode()

    stub = SimpleNamespace(GetWorkerStatus=lambda request,timeout: SimpleNamespace(alive=True))
    ops = SimpleNamespace(master_http_port=1,pb2=SimpleNamespace(StatusVersionPB=lambda: None),
        pb2_grpc=SimpleNamespace(RpcServiceStub=lambda channel: stub))
    ctx = SimpleNamespace(ops=ops,artifact_dir=tmp_path,env_epoch=1,instance=dict(environment={},source_path=str(tmp_path/'case.yaml')),
        env=SimpleNamespace(spec=SimpleNamespace(n_prefill=3,n_decode=1),discovery_file=discovery,
                            master_management_port=2,master=None,mock=None),
        clock=lambda:time[0],wall_clock=lambda:100+time[0],add_cleanup=lambda *a, **kw:None,
        record_event=lambda identity:dict(id=identity,epoch_s=100+time[0]))
    deadline = SimpleNamespace(clock=lambda:time[0],remaining=lambda:100,check=lambda:None,sleep=lambda value:time.__setitem__(0,time[0]+value))
    monkeypatch.setattr(actions,'control_json',control)
    monkeypatch.setattr(telemetry,'fetch',fetch)
    monkeypatch.setattr(telemetry,'resource',lambda proc:{})
    monkeypatch.setattr(traffic,'materialize',lambda *a,**kw:tmp_path/'traffic.jsonl')
    monkeypatch.setattr(actions.grpc,'insecure_channel',lambda address:contextlib.nullcontext(address))
    from cases.cache_sync_recovery.program import GATE
    ctx.monitor = SimpleNamespace(client=SimpleNamespace(metric_snapshot=lambda *a, **kw: []))
    ctx.services = SimpleNamespace(gate=lambda name: GATE)
    result = actions.execute(ctx,params,deadline)
    from scenario.stage_execution import _validate_output
    _validate_output(ctx,result,{},actions.CHECK_IDS)
    assert next(c.status for c in result.checks if c.id=='healthy_survivors') == 'FAIL'
    assert next(c.status for c in result.checks if c.id=='sampling_complete') == 'ERROR'
    evidence = json.loads((tmp_path/'cache-stability/evidence.json').read_text())
    assert evidence['affected_workers'] == ['h1']
    assert evidence['healthy_violations'][0]['direct_alive'][0]['alive'] is True
    assert evidence['runtime_incomplete'] is True
    assert 'later snapshot unavailable' in evidence['errors'][0]


def test_recovery_report_keeps_two_focused_panels_and_only_distinct_events():
    from cases.cache_sync_recovery.report import compact_report_spec
    data = dict(intervention=True, parameters={'procedure': {'expired_workers': 12}},
        environment=dict(n_prefill=120,n_decode=320), victims=list(range(12)), phase_summary={},
        frames=[dict(epoch_s=430,phase='post',healthy_max_status_age_ms=11000,missing_workers=1)],
        events=[dict(id=k,epoch_s=t) for k,t in (
            ('cache_only_started',0),('measurement_started',100),('measurement_finished',400),
            ('partial_workers_withdrawn',400.01),('intervention_started',400.02),
            ('partial_workers_restored',420),('post_started',420.01),('post_finished',540))])
    spec=compact_report_spec(data,dict(timeOriginEpochS=0,timeAxis={'max':600},run_id='sample'))
    assert len(spec['panels'])==2
    assert all(p['title'].startswith('recovery · ') and len(p['series'])==1 for p in spec['panels'])
    assert spec['panels'][0]['series'][0]['points']==[dict(x=430,y=11)]
    assert spec['panels'][1]['series'][0]['points']==[dict(x=430,y=427)]
    for panel in spec['panels']:
        assert [e['id'] for e in panel['events']]==[
            'measurement_started','partial_workers_withdrawn','partial_workers_restored']
    assert spec['events'][0]['id']=='cache_only_started'


def test_ring_fixture_requires_at_least_three_prefill_workers_before_launch():
    data = config()
    data['environment']['n_prefill'] = 2
    data['parameters']['procedure']['expired_workers'] = 1
    with pytest.raises(ValueError):
        configure_program(data,'case.yaml')


def test_archived_metrics_respect_epoch_labels_age_and_missing_values():
    from cases.cache_sync_recovery.telemetry import archived_metrics
    from types import SimpleNamespace
    store = SimpleNamespace(document={'metrics': {
        'master/worker_status_commits': [
            dict(epoch='1',labels={},points=[[100,2],[105,7]]),
            dict(epoch='2',labels={},points=[[100,999]])],
        'master/heap_bytes': [dict(epoch='1',labels={'area':'nonheap'},points=[[100,123]])]}})
    rows = archived_metrics(store,[dict(epoch_s=t) for t in (90,100,110,116)],epoch=1,maximum_gap_s=10)
    assert [r['worker_status_commits'] for r in rows] == [None,2,7,None]
    assert all(r['heap_bytes'] is None and r['diff_scanned_keys'] is None for r in rows)
    # Aggregation is declared at collection, before high-cardinality results are archived.
    plan = load_document(ROOT/'config/monitoring/cache_sync_recovery.yaml')
    assert all(s['promql'].startswith('sum') for s in plan['sources']['master'].values())


def pressure_evidence(intervention=False):
    params = copy.deepcopy(config()['parameters'])
    params['intervention'] = intervention
    params['procedure']['expired_workers'] = 1
    workers = {f'w{i}': row() for i in range(4)}
    workers['w3']['role'] = 'decode'
    frames = [dict(epoch_s=t, phase='measurement' if t < 14 else 'post', workers=copy.deepcopy(workers),
                   master={}, mock={}, healthy_max_status_age_ms=20, missing_workers=0,
                   process_allocated_bytes=int((t-10)*2**30)) for t in (10, 12, 14, 16)]
    return dict(parameters=params, intervention=intervention, environment=dict(n_prefill=3,n_decode=1),
        provenance=dict(env_epoch=1), frames=frames, worker_addresses=list(workers), affected_workers=[],
        victims=['w0'] if intervention else [], expired_workers=['w0'] if intervention else [],
        rebuilt_workers=['w0'] if intervention else [], convergence_samples=5, max_sample_gap_s=2,
        consistency=dict(consistent=True), client=dict(complete=True), errors=[],
        expiry_scope='discovery withdrawal', healthy_violations=[],
        gc_available=True, gc=[dict(epoch_s=15,full=True,duration_ms=180)],
        events=[dict(id=k,epoch_s=t) for k,t in [('cache_baseline_ready',9),('cache_only_started',10),
            ('measurement_started',10),('performance_observation_finished',17),('observation_finished',18)]],
        intensity={'measurement':dict(responses=50,entries=50*306000,key_payload_bytes=200000000,
            added_keys=2340,totals=dict(responses=15000,full_responses=15000,entries=15000*306000))})


@pytest.mark.parametrize('recovery', [False, True])
def test_core_checks_and_validity_are_separate_without_weakening_recovery(recovery):
    from cases.cache_sync_recovery.analysis import analyze
    evidence = pressure_evidence(recovery)
    result = analyze(evidence)
    assert result['verdict'] == 'PASS' and result['validity'] == 'VALID'
    assert {c['id'] for c in result['checks']} == __import__('cases.cache_sync_recovery.analysis',fromlist=['CHECK_IDS']).CHECK_IDS
    evidence['consistency']['consistent'] = False
    assert analyze(evidence)['verdict'] == 'FAIL'
    if recovery:
        evidence['consistency']['consistent'] = True
        evidence['rebuilt_workers'] = []
        assert analyze(evidence)['verdict'] == 'FAIL'


@pytest.mark.parametrize('failure', ['pressure','shape','mutations','gap','client'])
def test_invalid_measurement_cannot_become_valid_functional_pass(failure):
    from cases.cache_sync_recovery.analysis import analyze
    evidence = pressure_evidence()
    if failure == 'pressure': evidence['intensity']['measurement']['responses'] = 34
    if failure == 'shape': evidence['intensity']['measurement']['totals']['entries'] -= 1
    if failure == 'mutations': evidence['intensity']['measurement']['added_keys'] = 1999
    if failure == 'gap': evidence['max_sample_gap_s'] = 11
    if failure == 'client': evidence['client']['complete'] = False
    result = analyze(evidence)
    assert result['verdict'] == result['validity'] == 'INVALID'
    assert all(c['status'] == 'PASS' for c in result['functional_checks'] if c['status'] != 'SKIP')


@pytest.mark.parametrize('recovery', [False, True])
def test_shared_producer_and_automatic_report_render_frozen_gate(tmp_path, monkeypatch, recovery):
    from cases.cache_sync_recovery.program import GATE
    from monitoring.query_plan import load_plan, definitions
    from monitoring.metric_store import MetricStore
    from reporting.workload import write_views
    from reporting.core import read_bundle, load_analysis
    from test_workload_report_views import payload
    evidence = pressure_evidence(recovery)
    committed = GATE.commit(tmp_path, evidence)
    MetricStore(dict(metrics_schema_version=1, definitions=definitions(load_plan('cache_sync_recovery.yaml')),
                     metrics={},collection_gaps={},errors=[])).save(tmp_path)
    GATE.project(tmp_path)
    stored = MetricStore.read(tmp_path)
    assert stored.document['metrics']['cache_stability/full_gc_pause_ms'][0]['points'] == [[15,180]]
    # Delivery must not analyze, acquire telemetry, or parse mutable live logs.
    monkeypatch.setattr(type(GATE), 'evaluate', lambda *a: pytest.fail('report re-adjudicated gate'))
    run = payload()
    paths = write_views(tmp_path, run, ['cache_sync_recovery.yaml'])
    bundle = read_bundle(paths['cache_sync_recovery.yaml'].parent)
    spec = json.loads((bundle/'report-spec.json').read_text())
    assert len(spec['panels']) == 2
    assert all(p['title'].startswith(('recovery' if recovery else 'steady')+' · ') for p in spec['panels'])
    assert spec['timeOriginEpochS'] == 10
    assert load_analysis(bundle)['outcome'] == dict(gate=committed.result['verdict'],validity='VALID')


def test_cache_suite_stays_out_of_default_ci():
    suites = load_document(ROOT/'config/suites.yaml')
    assert suites['suites']['cache_sync'] == ['cache_sync_recovery::steady','cache_sync_recovery::recovery']
    assert all(not x.startswith('cache_sync_recovery::') for x in suites['suites']['workload'])
    assert all(j['suite'] != 'cache_sync' for j in suites['ci_jobs'].values())
