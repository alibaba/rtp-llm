import ast
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).parent
DEFAULT_SOURCE = ROOT.parents[1]/'navi_sched/adapters/vllm'
if not DEFAULT_SOURCE.is_dir():
    DEFAULT_SOURCE = ROOT/'candidate/navi_sched/adapters/vllm'
SOURCE = Path(__import__('os').environ.get('NAVI_TEST_KV_READINESS_SOURCE', str(DEFAULT_SOURCE)))
spec = importlib.util.spec_from_file_location('readiness_under_test', SOURCE/'VLLMKVLoadReadiness.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
VLLMKVLoadReadiness = module.VLLMKVLoadReadiness
forwarded_execution_load_flags = module.forwarded_execution_load_flags


class KVReadinessTest(unittest.TestCase):
    def fixture(self):
        agent = SimpleNamespace(begin_kv_preparation=mock.Mock(return_value=(71, 9)),
            complete_kv_preparation=mock.Mock(return_value=True),
            cancel_kv_preparation=mock.Mock(return_value=True),
            retire_kv_preparation=mock.Mock(return_value=True))
        request = SimpleNamespace(request_id='external', navi_id=123,
            status=SimpleNamespace(name='WAITING'), cache_known=False)
        scheduler = SimpleNamespace(requests={'external': request})
        events = []
        reader = VLLMKVLoadReadiness(agent, scheduler, lambda req: req.navi_id,
                                    lambda req: req.cache_known, events.append)
        reader.source = SimpleNamespace(_request_states={})
        types = SimpleNamespace(NaviSchedKVPreparation=SimpleNamespace,
            NativeKVExecutionState=SimpleNamespace(PREPARING=0, READY=1, EXECUTING=2, FAILED=3))
        contribution = SimpleNamespace(batch_id=17, prefill_context_tokens=20, remaining_prefill_tokens=80)
        native = SimpleNamespace(request_id='external', engine_id='engine', load_index=1)
        return reader, agent, request, types, contribution, native, events

    def test_predicted_external_hit_begins_before_native_dispatch_and_reuses_key(self):
        r,a,req,types,c,n,events = self.fixture()
        r.admitted(req, False)
        a.begin_kv_preparation.assert_called_once()
        self.assertTrue(r._on_started(n))
        a.begin_kv_preparation.assert_called_once()
        self.assertEqual(9, r.make_record(req,c,True,False,types).load_generation)
        r._on_finished(n, True)
        a.complete_kv_preparation.assert_called_once()
        self.assertEqual((71,123,9,True), a.complete_kv_preparation.call_args.args[:4])
        self.assertEqual(1, r.make_record(req,c,True,False,types).state)
        # A later snapshot still has the ready key while Native says loading.
        r.source._request_states['external'] = SimpleNamespace(load_inflight=True,
            pending_load_meta={}, resolved_match=(20,True))
        self.assertEqual(1, r.make_record(req,c,True,False,types).state)
        a.retire_kv_preparation.assert_not_called()
        self.assertEqual(['kv_load_preparing','kv_load_ready'], [e['event'] for e in events])

    def test_missing_marker_is_not_an_executable_default(self):
        r,a,req,types,c,_,_ = self.fixture()
        r.admitted(req, None)
        a.begin_kv_preparation.assert_not_called()
        self.assertIsNone(r.make_record(req,c,True,None,types))

    def test_hbm_and_no_hit_are_included_without_waiting_for_a_load(self):
        r,a,req,types,c,_,_ = self.fixture()
        self.assertEqual(1, r.make_record(req,c,True,True,types).state)
        a.begin_kv_preparation.assert_not_called()

    def test_native_load_overrides_predicted_hbm_inclusion(self):
        r,a,req,types,c,_,_ = self.fixture()
        r.source._request_states['external'] = SimpleNamespace(load_inflight=True,
            pending_load_meta={}, resolved_match=(20,True))
        record = r.make_record(req,c,True,True,types)
        self.assertEqual(0, record.state)
        self.assertEqual((123,17,20,80,True,9), (record.request_id,record.local_batch_id,
            record.context_tokens,record.compute_tokens,record.running_capacity,record.load_generation))

    def test_external_prediction_resolves_to_no_load_and_is_not_stuck_preparing(self):
        r,a,req,types,c,_,_ = self.fixture()
        r.admitted(req, False)
        req.cache_known = True
        record = r.make_record(req,c,False,False,types)
        self.assertEqual((1,0), (record.state,record.load_generation))
        a.cancel_kv_preparation.assert_called_once_with(71,123,9)
        a.retire_kv_preparation.assert_called_once_with(71,123,9)

    def test_failed_load_is_excluded_until_native_recomputation_runs(self):
        r,a,req,types,c,n,_ = self.fixture()
        r._on_started(n)
        r._on_finished(n,False)
        self.assertEqual(3,r.make_record(req,c,True,False,types).state)
        req.status.name = 'RUNNING'
        self.assertEqual(2,r.make_record(req,c,True,False,types).state)

    def test_old_generation_cannot_complete_replacement(self):
        r,a,req,types,c,n,_ = self.fixture()
        r._on_started(n)
        a.begin_kv_preparation.return_value = (71,10)
        new = SimpleNamespace(request_id='external',engine_id='engine',load_index=2)
        r._on_started(new)
        r._on_finished(n,True)
        a.complete_kv_preparation.assert_not_called()
        r._on_finished(new,True)
        self.assertEqual((71,123,10,True),a.complete_kv_preparation.call_args.args[:4])

    def test_forget_and_close_retire_pending_keys_and_ignore_late_completion(self):
        r,a,req,types,c,n,_ = self.fixture()
        r._on_started(n)
        r.forget('external')
        r._on_finished(n,True)
        a.complete_kv_preparation.assert_not_called()
        a.retire_kv_preparation.assert_called_once_with(71,123,9)
        r.close()
        self.assertFalse(r._on_started(n))

    def test_completion_requires_explicit_bool(self):
        r,*rest = self.fixture()
        for value in (None,1,'yes'):
            with self.assertRaises(TypeError):
                r._on_finished(rest[4],value)

    def test_forward_flags_follow_request_identity_after_partial_abort(self):
        batch = SimpleNamespace(request_ids=[11,22,33],execution_load_included=[False,None,True])
        self.assertEqual({33:True,11:False},forwarded_execution_load_flags(batch,[33,11]))
        self.assertEqual({22:None},forwarded_execution_load_flags(batch,[22]))
        batch.execution_load_included = [True]
        with self.assertRaises(ValueError):forwarded_execution_load_flags(batch,[11])

    def test_no_v6d_does_not_enable_or_modify_executor(self):
        r,a,*_ = self.fixture()
        executor = SimpleNamespace(enable_kv_load_notifications=mock.Mock())
        self.assertFalse(r.bind(executor,SimpleNamespace()))
        executor.enable_kv_load_notifications.assert_not_called()

    def test_normal_discovery_mode_binds_without_native_ingress_or_address_configuration(self):
        r,a,*_ = self.fixture()
        executor = SimpleNamespace(enable_kv_load_notifications=mock.Mock(return_value=True))
        source = SimpleNamespace(_request_states={})
        connector = SimpleNamespace(_connectors=[SimpleNamespace(
            set_load_completion_observer=mock.Mock(),scheduler=source)])
        self.assertTrue(r.bind(executor,connector))
        self.assertIs(r.source,source)
        self.assertIs(r.executor,executor)

    def test_actual_observer_binding_polls_non_pp_without_changing_step_result(self):
        self._observer_binding_test(1)

    def test_actual_observer_binding_keeps_pp_step_function(self):
        self._observer_binding_test(4)

    def _observer_binding_test(self, pp):
        text = (SOURCE/'VLLMEngineObserver.py').read_text()
        cls = next(n for n in ast.parse(text).body if getattr(n,'name','')=='VLLMEngineObserver')
        node = next(n for n in cls.body if getattr(n,'name','')=='_bind_kv_load_readiness')
        # Supply the imported class while executing the original method body.
        node.body = [n for n in node.body if not isinstance(n,ast.ImportFrom)]
        ns = dict(VLLMKVLoadReadiness=VLLMKVLoadReadiness)
        exec('from __future__ import annotations\n'+ast.unparse(node),ns)
        r,a,req,types,c,n,events = self.fixture()
        executor = SimpleNamespace(enable_kv_load_notifications=mock.Mock(return_value=True),
                                   poll_kv_load_notifications=mock.Mock())
        connector = SimpleNamespace(set_load_completion_observer=mock.Mock(),scheduler=r.source)
        scheduler = SimpleNamespace(connector=connector,requests={'external':req})
        step = mock.Mock(return_value=('unchanged',True))
        core = SimpleNamespace(model_executor=executor,scheduler=scheduler,step_fn=step,
            vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=pp)))
        observer = SimpleNamespace(_native_ingress=False,_agent=a,_navi_request_id=lambda req:req.navi_id,
            _native_cache_lookup_complete=lambda req:req.cache_known,
            on_request_lifecycle_event=events.append,_refresh_and_report_workload=mock.Mock())
        ns['_bind_kv_load_readiness'](observer,core)
        self.assertEqual(('unchanged',True),core.step_fn())
        if pp==1:executor.poll_kv_load_notifications.assert_called_once_with()
        else:
            self.assertIs(step,core.step_fn)
            executor.poll_kv_load_notifications.assert_not_called()

    def test_actual_observer_partition_records_all_requests_and_keeps_raw_totals(self):
        text = (SOURCE/'VLLMEngineObserver.py').read_text()
        tree = ast.parse(text)
        r,a,req,types,c,n,_ = self.fixture()
        ns = dict(dataclass=dataclass,field=field,naviSchedModule=lambda:types)
        for name in ['_PrefillWorkloadSlice','_PrefillWorkloadSliceAggregate']:
            node = next(n for n in tree.body if getattr(n,'name','')==name)
            exec('from __future__ import annotations\n'+ast.unparse(node),ns)
        cls = next(n for n in tree.body if getattr(n,'name','')=='VLLMEngineObserver')
        node = next(n for n in cls.body if getattr(n,'name','')=='_partition_prefill_workload')
        exec('from __future__ import annotations\n'+ast.unparse(node),ns)
        requests = [SimpleNamespace(request_id=str(i),navi_id=i,status=SimpleNamespace(name='WAITING'),cache_known=True) for i in (1,2,3)]
        scheduler = SimpleNamespace(requests={q.request_id:q for q in requests},running=requests[:1],
            waiting=requests[1:2],max_num_running_reqs=2)
        r._scheduler = scheduler
        contrib = {q.request_id:SimpleNamespace(batch_id=10+i,prefill_context_tokens=20+i,
            remaining_prefill_tokens=80+i)for i,q in enumerate(requests)}
        observer = SimpleNamespace(_native_request_state_audit_enabled=False,_pp_capacity_unified=True,
            _is_finished_pp_request=lambda req:False,_workload_contributions=contrib,
            _forward_request_state=lambda req:None,_navi_request_id=lambda q:q.navi_id,_kv_load_readiness=r)
        records=[]
        result=ns['_partition_prefill_workload'](observer,scheduler,execution_load_requests=records)
        self.assertEqual([1,2,3],[v.request_id for v in records])
        self.assertEqual([True,True,False],[v.running_capacity for v in records])
        self.assertEqual(161,result[2].compute_tokens)
        self.assertEqual(82,result[3].compute_tokens)
        self.assertEqual(41,result[2].context_tokens)
        self.assertEqual(22,result[3].context_tokens)


    def test_refresh_retires_unforwarded_native_request_even_when_workload_is_unchanged(self):
        r,a,req,types,c,n,_ = self.fixture()
        r._on_started(n)
        r._on_finished(n,True)
        r._scheduler.requests.clear()
        tree=ast.parse((SOURCE/'VLLMEngineObserver.py').read_text())
        cls=next(n for n in tree.body if getattr(n,'name','')=='VLLMEngineObserver')
        node=next(n for n in cls.body if getattr(n,'name','')=='_refresh_and_report_workload')
        ns={}
        exec('from __future__ import annotations\n'+ast.unparse(node),ns)
        observer=SimpleNamespace(_workload_report_lock=threading.RLock(),_workload_lock=threading.RLock(),
            _kv_load_readiness=r,_synchronize_workload_locked=lambda *args:None)
        ns['_refresh_and_report_workload'](observer,r._scheduler,(),'test')
        a.retire_kv_preparation.assert_called_once_with(71,123,9)
        self.assertFalse(r._loads)
        r._on_finished(n,True)
        self.assertEqual(1,a.complete_kv_preparation.call_count)

    def test_pending_admission_view_preserves_tracking_even_when_no_requests_are_pending(self):
        self._pending_view_test(False,False)

    def test_pending_admission_view_carries_each_request_initial_flag(self):
        self._pending_view_test(True,False)
        self._pending_view_test(True,True)
        self._pending_view_test(True,None)

    def _pending_view_test(self,pending,included):
        import math,itertools
        from dataclasses import replace
        r,a,req,types,c,n,_=self.fixture()
        tree=ast.parse((SOURCE/'VLLMEngineObserver.py').read_text())
        ns=dict(dataclass=dataclass,field=field,math=math,itertools=itertools,
            replace=replace,naviSchedModule=lambda:types)
        for name in ['_PrefillWorkloadSlice','_PrefillWorkloadSliceAggregate','_ObservedWorkload',
                     '_WorkloadContribution','_WorkloadAggregate']:
            node=next(n for n in tree.body if getattr(n,'name','')==name)
            exec('from __future__ import annotations\n'+ast.unparse(node),ns)
        cls=next(n for n in tree.body if getattr(n,'name','')=='VLLMEngineObserver')
        node=next(n for n in cls.body if getattr(n,'name','')=='_workload_with_pending_admissions_locked')
        exec('from __future__ import annotations\n'+ast.unparse(node),ns)
        aggregate=ns['_WorkloadAggregate']()
        contribution=ns['_WorkloadContribution'](17,100,20,80,20,80)
        state=SimpleNamespace(execution_load_included={123:included})
        prepared={123:(state,contribution)} if pending else {}
        if pending:aggregate.apply(contribution,1)
        observer=SimpleNamespace(_prepared_workload_lock=threading.RLock(),_native_handoff_batches={},
            _prepared_workload=aggregate,_prepared_contributions=prepared,_pp_capacity_unified=True,
            _kv_load_readiness=r)
        base=ns['_ObservedWorkload'](max_running_request_count=2,execution_load_tracked=True)
        result=ns['_workload_with_pending_admissions_locked'](observer,base)
        self.assertEqual(int(pending),result.request_count)
        self.assertEqual(80 if pending else 0,result.remaining_prefill_tokens)
        if pending and included is None:
            self.assertFalse(result.execution_load_tracked)
            self.assertEqual((),result.execution_load_requests)
        else:
            self.assertTrue(result.execution_load_tracked)
            self.assertEqual(int(pending),len(result.execution_load_requests))
            if pending:
                record=result.execution_load_requests[0]
                self.assertEqual((123,17,20,80,True,0), (record.request_id,record.local_batch_id,
                    record.context_tokens,record.compute_tokens,record.running_capacity,record.load_generation))
                self.assertEqual(1 if included else 0,record.state)


if __name__=='__main__':
    unittest.main(verbosity=2)
