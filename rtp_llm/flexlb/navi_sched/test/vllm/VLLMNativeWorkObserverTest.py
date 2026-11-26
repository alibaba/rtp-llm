"""Actual Observer callbacks, measured Worker metadata, and compiled binding."""
from pathlib import Path
import importlib.util
import json
import os
import sys
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).parent
import navi_sched._navi_sched_core as core
import VLLMEngineObserverTest as base
module = __import__('importlib').import_module('navi_sched.adapters.vllm.VLLMEngineObserver')


def geometry(spans):
    tokens = sum(n for _, _, n, _ in spans)
    return {'version': 1, 'chunks': [{
        'scheduled_tokens': tokens, 'pcp_size': 1, 'interleave_size': 1,
        'max_local_tokens': tokens, 'local_padded_tokens': tokens,
        'global_padded_tokens': tokens,
        'request_spans': [dict(request_id=rid, computed_tokens=h,
                               scheduled_tokens=n, prompt_tokens=prompt)
                          for rid, h, n, prompt in spans],
    }]}


class NativeWorkObserverTest(base.VLLMEngineObserverTest):
    def make_work_observer(self):
        observer = self.make_continuous_timing_observer()
        observer._initialize_agent(SimpleNamespace())
        observer._pp_capacity_unified = True
        observer._pp_native_work_training = True
        observer._bridge.forward_admission_watermark_ns = lambda: 0
        observer._bridge.acknowledge_forward_admission = mock.Mock(return_value=True)
        report_step = observer._bridge.report_execution_step
        observer._start_order_reports = []
        def report_with_start_order(elapsed_ms, prefill, decode, overtakes=0):
            observer._start_order_reports.append(overtakes)
            return report_step(elapsed_ms, prefill, decode)
        observer._bridge.report_execution_step = report_with_start_order
        observer.on_request_lifecycle_event = mock.Mock()
        idle = module._ObservedWorkload(observed_time_ns=0)
        with mock.patch.object(observer, '_workload_snapshot', return_value=idle), \
             mock.patch.object(observer, '_refresh_workload_contributions'):
            observer._synchronize_workload_locked(SimpleNamespace(requests={}), ())
        return observer

    def add_own(self, observer, batch_id, specs, ready_ns, preceding=None):
        requests = [self.make_request(prompt_tokens=prompt, computed_tokens=0,
                                     cached_tokens=0, request_id=rid)
                    for rid, prompt in specs]
        state = self.add_forward_batch(observer, batch_id, [r.request_id for r in requests])
        for request in requests:
            observer._admit_request(request)
        observer._set_admission_state(state, requests,
                                     preceding or module._ObservedWorkload(max_running_request_count=64), ready_ns)
        return state, requests

    def step(self, observer, requests, spans, started_ns, completed_ns, *, valid=True, output_first=False):
        future = Future()
        output = SimpleNamespace(num_scheduled_tokens={rid: n for rid, _, n, _ in spans})
        for rid, h, n, _ in spans:
            requests[rid].num_computed_tokens = h + n
        wrapped = observer.wrap_execute_model(lambda unused, non_block: future,
                                              SimpleNamespace(requests=requests))
        with mock.patch.object(module.time, 'perf_counter_ns', return_value=started_ns) as clock:
            wrapped(output, non_block=True)
            clock.return_value = completed_ns
            model_output = SimpleNamespace(execution_geometry=geometry(spans) if valid else None)
            if output_first:
                observer.on_model_output(output, model_output)
                future.set_result('device-output')
            else:
                future.set_result('device-output')
                observer.on_model_output(output, model_output)
        return output

    def assert_binding(self, observer):
        args = observer._bridge.backend_batch_executions[-1]
        stats = observer._bridge.finished[-1][1]
        self.assertFalse(core._VLLMAgentBridge.finish_batch_with_execution(None, stats, *args))
        return args

    def test_partial_long_request_only_counts_through_short_terminal_step(self):
        observer = self.make_work_observer()
        long, lr = self.add_own(observer, 31, [('long', 10000)], 1)
        self.step(observer, {'long': lr[0]}, [('long', 0, 100, 10000)], 10, 20)
        preceding = module._ObservedWorkload(
            max_running_request_count=64, prior_cache_forecasts=(
                module._AdmissionPriorCacheForecast(True, 10000, 100, 9900, 100,
                    next(iter(long.cache_lookups.values())), 31),
            ),
        )
        short, sr = self.add_own(observer, 32, [('short', 16)], 30, preceding)
        refs = {'long': lr[0], 'short': sr[0]}
        self.step(observer, refs, [('long', 100, 4096, 10000), ('short', 0, 16, 16)], 40, 50)
        args = self.assert_binding(observer)
        self.assertEqual((1, 1, 2, 100, 4112, float(4096**2 + 16**2), 409600.), args[7:14])
        self.assertEqual((0, 0, 0, 0., 0.), args[14:19])
        self.assertTrue(args[-1])
        self.assertTrue(short.released)
        self.assertFalse(long.released)
        self.step(observer, refs, [('long', 4196, 5804, 10000)], 60, 70)
        self.assertTrue(long.released)
        self.assertEqual(4112, short.native_work_terminal[1].compute_tokens - short.admission_state.native_work_window[1].begin.compute_tokens)
        observer.shutdown()

    def test_work_between_own_members_is_included_until_last_completion(self):
        observer = self.make_work_observer()
        target, own = self.add_own(observer, 33, [('a', 16), ('b', 32)], 1)
        prior, old = self.add_own(observer, 34, [('late-ready', 64)], 2)
        refs = {r.request_id: r for r in own + old}
        self.step(observer, refs, [('a', 0, 16, 16)], 10, 20)
        self.assertFalse(target.released)
        self.step(observer, refs, [('late-ready', 0, 64, 64)], 30, 40)
        self.assertTrue(prior.released)
        self.step(observer, refs, [('b', 0, 32, 32)], 50, 60, output_first=True)
        args = self.assert_binding(observer)
        self.assertEqual((1, 3, 3, 0, 112, float(16**2 + 32**2 + 64**2), 0.), args[7:14])
        self.assertTrue(target.released)
        self.assertEqual({}, observer._native_work_pending)
        observer.shutdown()

    def test_invalid_geometry_completes_request_and_suppresses_only_training(self):
        observer = self.make_work_observer()
        target, own = self.add_own(observer, 35, [('a', 16)], 1)
        self.step(observer, {'a': own[0]}, [('a', 0, 16, 16)], 10, 20, valid=False)
        args = self.assert_binding(observer)
        self.assertFalse(args[-1])
        self.assertTrue(target.released)
        self.assertEqual(1, len(observer._bridge.finished))
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual({}, observer._pending_forward_batches)
        # Subsequent complete facts start a new measured epoch and can train.
        idle = module._ObservedWorkload(observed_time_ns=21)
        with mock.patch.object(observer, '_workload_snapshot', return_value=idle), \
             mock.patch.object(observer, '_refresh_workload_contributions'):
            observer._synchronize_workload_locked(SimpleNamespace(requests={}), ())
        target, own = self.add_own(observer, 36, [('b', 16)], 30)
        self.step(observer, {'b': own[0]}, [('b', 0, 16, 16)], 40, 50)
        self.assertTrue(self.assert_binding(observer)[-1])
        self.assertTrue(target.released)
        observer.shutdown()

    def test_native_work_mode_isolated_from_legacy_and_non_pp(self):
        config = self.make_config()
        config.parallel_config.pipeline_parallel_size = 4
        env = {'VLLM_NAVI_SCHED_NATIVE_INGRESS': '1', 'TPU_RAIDEN_JOB_NAME': 'prefill',
               'VLLM_NAVI_SCHED_PP_LATENCY_MODE': 'native_work19'}
        with mock.patch.dict(os.environ, env):
            observer = module.VLLMEngineObserver(config, self.make_kv_cache_config())
        self.assertTrue(observer._pp_native_work_training)
        fingerprint = json.loads(observer._agent_options['latency_model_fingerprint'])
        self.assertEqual('completed_native_work_window_v1', fingerprint['unified_capacity_training_scope'])
        self.assertEqual('native_positive_prefill_step_v1', fingerprint['unified_capacity_batch_count_unit'])
        observer.shutdown()
        config.parallel_config.pipeline_parallel_size = 1
        with mock.patch.dict(os.environ, env):
            observer = module.VLLMEngineObserver(config, self.make_kv_cache_config())
        self.assertFalse(observer._pp_native_work_training)
        observer.shutdown()

    def test_actual_binding_default_and_measured_clocks(self):
        old = core._VLLMAgentBridge(None)
        self.assertEqual(0, old.forward_admission_watermark_ns())
        measured = core._VLLMAgentBridge(None, track_native_work=True)
        first = measured.forward_admission_watermark_ns()
        self.assertGreater(first, 0)
        self.assertGreaterEqual(measured.forward_admission_watermark_ns(), first)
        self.assertFalse(measured.acknowledge_forward_admission(None))
        measured.release_forward_batch(None)
        self.assertIn('completed_features_valid', measured.finish_batch_with_execution.__doc__)
        self.assertIn('first_start_overtake_count', measured.report_execution_step.__doc__)
        self.assertFalse(measured.report_execution_step(1., 16, 0, 1))

    def test_later_request_starting_first_reports_one_overtake(self):
        observer = self.make_work_observer()
        early, a = self.add_own(observer, 41, [('early', 16)], 1)
        late, b = self.add_own(observer, 42, [('late', 16)], 2)
        refs = {r.request_id: r for r in a + b}
        self.step(observer, refs, [('late', 0, 16, 16)], 10, 20)
        self.assertEqual([1], observer._start_order_reports)
        event = next(c.args[0] for c in observer.on_request_lifecycle_event.call_args_list
                     if c.args[0].get('event') == 'native_request_start_overtake')
        self.assertEqual(late.request_ids, event['overtaking_request_ids'])
        self.assertEqual(early.request_ids[0], event['earliest_unstarted_request_id'])
        self.step(observer, refs, [('early', 0, 16, 16)], 30, 40)
        self.assertEqual([1, 0], observer._start_order_reports)
        observer.shutdown()

    def test_same_step_first_starts_do_not_report_overtake(self):
        observer = self.make_work_observer()
        _, a = self.add_own(observer, 43, [('early', 16)], 1)
        _, b = self.add_own(observer, 44, [('late', 16)], 2)
        self.step(observer, {r.request_id: r for r in a + b},
                  [('late', 0, 16, 16), ('early', 0, 16, 16)], 10, 20)
        self.assertEqual([0], observer._start_order_reports)
        observer.shutdown()

    def test_abort_before_first_start_does_not_leave_false_early_waiter(self):
        observer = self.make_work_observer()
        early, a = self.add_own(observer, 45, [('cancelled', 16)], 1)
        _, b = self.add_own(observer, 46, [('late', 16)], 2)
        observer._abort_forward_requests(early, early.request_ids)
        self.assertTrue(early.released)
        self.step(observer, {'late': b[0]}, [('late', 0, 16, 16)], 10, 20)
        self.assertEqual([0], observer._start_order_reports)
        observer.shutdown()


if __name__ == '__main__':
    unittest.main(defaultTest=['NativeWorkObserverTest.' + n for n in NativeWorkObserverTest.__dict__
                              if n.startswith('test_')], verbosity=2)
