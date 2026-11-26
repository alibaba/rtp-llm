"""Real Observer callbacks and actual compiled binding; controlled queue facts."""
import sys
import unittest
from types import SimpleNamespace
from unittest import mock
import VLLMNativeWorkObserverTest as prior

module=prior.module

class ElapsedObserverIntegrationTest(prior.NativeWorkObserverTest):
    @staticmethod
    def snapshot(clock,count,elapsed):
        part=module._PrefillWorkloadSlice(count,count*1024,count*16,
            float(count*256),float(count*16384))
        value=module._ObservedWorkload(
            batch_count=int(count>0),request_count=count,remaining_prefill_request_count=count,
            running_request_count=count,running_elapsed_time_ms=elapsed,observed_time_ns=clock,
            remaining_prefill_context_tokens=part.context_tokens,remaining_prefill_tokens=part.compute_tokens,
            remaining_prefill_token_square_sum=part.compute_token_square_sum,
            context_remaining_prefill_token_product_sum=part.context_compute_token_product_sum,
            remaining_total_context_tokens=part.context_tokens,remaining_total_tokens=part.compute_tokens,
            remaining_total_token_square_sum=part.compute_token_square_sum,
            context_remaining_total_token_product_sum=part.context_compute_token_product_sum,
            total_prompt_tokens=part.context_tokens+part.compute_tokens,max_running_request_count=64,
            running_capacity_slot_count=count,running_capacity_prefill=part,scheduler_running_prefill=part)
        assert value.valid()
        return value

    def publish(self,observer,snapshot):
        scheduler=SimpleNamespace(requests={})
        with mock.patch.object(observer,'_workload_snapshot',return_value=snapshot), \
             mock.patch.object(observer,'_refresh_workload_contributions'):
            return observer._synchronize_workload_locked(scheduler,())

    def temporal_case(self,ready_ns):
        observer=self.make_work_observer()
        self.publish(observer,self.snapshot(0,0,0.))
        old,oldrequests=self.add_own(observer,81,[('prior',16)],1)
        self.publish(observer,self.snapshot(100_000_000,1,1000.))
        self.step(observer,{'prior':oldrequests[0]},[('prior',0,16,16)],100_000_000,200_000_000)
        self.assertTrue(old.released)
        self.publish(observer,self.snapshot(250_000_000,0,0.))
        with mock.patch.object(module.time,'perf_counter_ns',return_value=300_000_000):
            preceding=observer._capture_arrival_workload()
            target,requests=self.add_own(observer,82,[('target',16)],ready_ns,preceding)
        self.step(observer,{'target':requests[0]},[('target',0,16,16)],310_000_000,400_000_000)
        self.assertTrue(target.released)
        args=self.assert_binding(observer)
        observer.shutdown()
        return args

    def test_receipt_before_completion_keeps_past_elapsed_at_late_handoff(self):
        args=self.temporal_case(150_000_000)
        self.assertEqual(1050.,args[19])
        self.assertTrue(args[-1])
        self.assertEqual((1,2,2,0,32,512.,0.),args[7:14])

    def test_receipt_after_completion_uses_new_queue_population(self):
        args=self.temporal_case(260_000_000)
        self.assertEqual(0.,args[19])
        self.assertTrue(args[-1])
        self.assertEqual((0,1,1,0,16,256.,0.),args[7:14])

    def test_missing_checkpoint_suppresses_learning_without_losing_response(self):
        observer=self.make_work_observer()
        observer._native_work_history=module.NativeWorkHistory()
        target,requests=self.add_own(observer,83,[('target',16)],1)
        self.step(observer,{'target':requests[0]},[('target',0,16,16)],10,20)
        args=self.assert_binding(observer)
        self.assertFalse(args[-1])
        self.assertTrue(target.released)
        self.assertEqual(1,len(observer._bridge.finished))
        observer.shutdown()

    def test_rebuilt_valid_snapshot_is_recorded_without_invalid_predecessor(self):
        observer=self.make_work_observer()
        invalid=module._ObservedWorkload(running_elapsed_time_ms=99.)
        self.assertFalse(invalid.valid())
        value=self.snapshot(100_000_000,1,1000.)
        with mock.patch.object(observer,'_workload_snapshot',side_effect=[invalid,value]), \
             mock.patch.object(observer,'_refresh_workload_contributions'):
            self.assertIs(value,observer._synchronize_workload_locked(SimpleNamespace(requests={}),()))
        window=observer._native_work_history.capture(150_000_000)
        self.assertEqual(1050.,window.admission_elapsed_time_ms)
        observer.shutdown()

    def test_model_identity_separates_corrected_pp_elapsed(self):
        import json,os
        config=self.make_config();config.parallel_config.pipeline_parallel_size=4
        env={'VLLM_NAVI_SCHED_NATIVE_INGRESS':'1','TPU_RAIDEN_JOB_NAME':'prefill',
             'VLLM_NAVI_SCHED_PP_LATENCY_MODE':'native_work19'}
        with mock.patch.dict(os.environ,env):
            observer=module.VLLMEngineObserver(config,self.make_kv_cache_config())
        value=json.loads(observer._agent_options['latency_model_fingerprint'])
        self.assertEqual('cpp_ready_workload_history_v1',value['unified_capacity_elapsed_scope'])
        observer.shutdown()
        config.parallel_config.pipeline_parallel_size=1
        with mock.patch.dict(os.environ,env):
            observer=module.VLLMEngineObserver(config,self.make_kv_cache_config())
        value=json.loads(observer._agent_options['latency_model_fingerprint'])
        self.assertNotIn('unified_capacity_elapsed_scope',value)
        observer.shutdown()

if __name__ == '__main__':
    tests = ['ElapsedObserverIntegrationTest.' + name
             for name in ElapsedObserverIntegrationTest.__dict__
             if name.startswith('test_')]
    unittest.main(defaultTest=tests, verbosity=2)
