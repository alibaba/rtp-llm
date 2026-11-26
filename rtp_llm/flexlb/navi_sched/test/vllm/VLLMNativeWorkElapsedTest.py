import ast, dataclasses, importlib.util, logging, math, sys, threading, unittest
from pathlib import Path
from types import SimpleNamespace
from navi_sched.adapters.vllm import VLLMNativeWork as native
from navi_sched.adapters.vllm.VLLMNativeWork import NativeWorkHistory, NativeWorkTracker, NativeRequestProgress

SOURCE=Path(native.__file__).with_name('VLLMEngineObserver.py').read_text()
TREE=ast.parse(SOURCE)
record=next(n for n in ast.walk(TREE)if isinstance(n,ast.FunctionDef)and n.name=='_record_native_work_elapsed')
ns=dict(logger=logging.getLogger('candidate-test'))
exec(compile(ast.Module(body=[record],type_ignores=[]),'candidate-observer','exec'),ns)
record_elapsed=ns['_record_native_work_elapsed']

class ElapsedAdmissionTest(unittest.TestCase):
    def test_late_handoff_does_not_read_future_queue_population(self):
        h=NativeWorkHistory()
        h.record_elapsed(100_000_000,1000.,8)
        h.record_elapsed(200_000_000,200.,3)
        window=h.capture(150_000_000)
        self.assertEqual(1400.,window.admission_elapsed_time_ms)
        self.assertEqual(100_000_000,window.admission_elapsed_observed_ns)
        self.assertEqual(200.,h.capture(200_000_000).admission_elapsed_time_ms)

    def test_existing_window_is_frozen_across_updates_and_pruning(self):
        h=NativeWorkHistory();h.record_elapsed(100,400.,2)
        w=h.capture(200)
        h.record_elapsed(300,40.,1);h.prune(300)
        self.assertEqual(400.0002,w.admission_elapsed_time_ms)
        self.assertEqual(100,w.admission_elapsed_observed_ns)

    def test_unknown_history_remains_explicit(self):
        h=NativeWorkHistory()
        self.assertIsNone(h.capture(100).admission_elapsed_time_ms)
        h.record_elapsed(200,700.,2)
        self.assertIsNone(h.capture(100).admission_elapsed_time_ms)
        h.clear_elapsed()
        self.assertIsNone(h.capture(300).admission_elapsed_time_ms)

    def test_equal_clock_uses_last_scalar_snapshot(self):
        h=NativeWorkHistory();h.record_elapsed(100,7.,2);h.record_elapsed(100,8.,3)
        self.assertEqual((8.,100),h.elapsed_at(100))
        self.assertEqual(1,len(h._elapsed_times))

    def test_pruning_keeps_pre_watermark_anchor_and_future_snapshots(self):
        h=NativeWorkHistory()
        for i in range(1000):h.record_elapsed((i+1)*1000,float(i),2)
        h.prune(900_500)
        self.assertEqual((899.001,900_000),h.elapsed_at(900_500))
        self.assertEqual((999.,1_000_000),h.elapsed_at(1_000_000))
        self.assertEqual((None,0),h.elapsed_at(0))
        # The Native work prefix may be older than the elapsed anchor; an
        # expired elapsed query is explicitly missing rather than backfilled.
        self.assertEqual((None,0),h.elapsed_at(899_999))
        self.assertLessEqual(len(h._elapsed_times),101)

    def test_retired_native_boundary_is_rejected(self):
        h=NativeWorkHistory();h.record_elapsed(100,1.,2)
        h.append(200,native.NativeWorkPrefix(step_count=1),())
        h.prune(200)
        with self.assertRaises(ValueError):h.elapsed_at(199)

    def test_moving_watermark_does_not_retain_lifetime_history(self):
        h=NativeWorkHistory();maximum=0
        for i in range(5000):
            h.record_elapsed(i+1,float(i),2)
            h.prune(max(0,i-8))
            maximum=max(maximum,len(h._elapsed_times))
        self.assertLessEqual(maximum,74)
        self.assertAlmostEqual(4999.,h.elapsed_at(5000)[0])

    def test_bad_input_cannot_partially_change_history(self):
        h=NativeWorkHistory();h.record_elapsed(100,1.,1)
        for args in [(99,1.,1),(101,float('nan'),1),(101,-1.,1),(101,1.,-1),(101,1.,True)]:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):h.record_elapsed(*args)
                self.assertEqual((1.,100),h.elapsed_at(100))

    def test_recording_is_pp_native_work_only(self):
        class Exploding:
            def __enter__(self):raise AssertionError('mode-off took execution lock')
            def __exit__(self,*args):pass
        observer=SimpleNamespace(_pp_native_work_training=False,_execution_lock=Exploding())
        record_elapsed(observer,SimpleNamespace())

    def test_actual_observer_records_workload_clock_and_running_population(self):
        h=NativeWorkHistory()
        observer=SimpleNamespace(_pp_native_work_training=True,_execution_lock=threading.Lock(),_native_work_history=h)
        record_elapsed(observer,SimpleNamespace(observed_time_ns=100_000_000,
            running_elapsed_time_ms=700.,scheduler_running_prefill=SimpleNamespace(request_count=4)))
        self.assertEqual((900.,100_000_000),h.elapsed_at(150_000_000))

    def test_failed_observation_cannot_leave_a_stale_curve_for_training(self):
        h=NativeWorkHistory();h.record_elapsed(100,10.,2)
        observer=SimpleNamespace(_pp_native_work_training=True,_execution_lock=threading.Lock(),_native_work_history=h)
        with self.assertLogs('candidate-test',level='ERROR'):
            record_elapsed(observer,SimpleNamespace(observed_time_ns=99,
                running_elapsed_time_ms=20.,scheduler_running_prefill=SimpleNamespace(request_count=2)))
        self.assertEqual((None,0),h.elapsed_at(200))

    def _sample_block(self,mode=True,known=True):
        h=NativeWorkHistory()
        if known:
            h.record_elapsed(100_000_000,1000.,8)
            h.record_elapsed(200_000_000,200.,3)
        window=h.capture(150_000_000)
        tracker=NativeWorkTracker();end=tracker.dispatch([(NativeRequestProgress(),16,32)])
        events=[]
        values=dict(self=SimpleNamespace(_pp_native_work_training=mode,on_request_lifecycle_event=events.append),
            state=SimpleNamespace(native_work_terminal=(1,end),cancelled_request_ids=set(),batch_id=1,schedule_batch_id=2),
            admission_state=SimpleNamespace(native_work_window=(1,window)),preceding=SimpleNamespace(running_elapsed_time_ms=200.),
            request_stats=[(1,48,16)],completed_time_us=1,admission_started_ns=150_000_000,
            measured_work_valid=True,sample_running_elapsed_time_ms=200.,asdict=dataclasses.asdict,
            _PrefillWorkloadSlice=lambda *args:SimpleNamespace(values=args))
        block=next(n for n in ast.walk(TREE)if isinstance(n,ast.If)
            and ast.unparse(n.test)=='getattr(self, \'_pp_native_work_training\', False)'
            and any(isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='work'for t in x.targets)for x in n.body))
        exec(compile(ast.Module(body=[block],type_ignores=[]),'actual-completion-block','exec'),values)
        return values,events

    def test_actual_completion_block_uses_historical_elapsed_with_real_work(self):
        values,events=self._sample_block()
        self.assertTrue(values['measured_work_valid'])
        self.assertEqual(1400.,values['sample_running_elapsed_time_ms'])
        self.assertEqual(1,values['sample_batch_count'])
        self.assertEqual(32,values['work'].compute_tokens)
        self.assertEqual(1400.,events[0]['admission_elapsed_time_ms'])
        self.assertEqual(100_000_000,events[0]['admission_elapsed_observed_perf_ns'])
        breakdown=next(n for n in ast.walk(TREE)if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name)and t.id=='execution_breakdown'for t in n.targets))
        fields=[ast.unparse(n)for n in breakdown.value.elts]
        self.assertIn('sample_running_elapsed_time_ms',fields)
        self.assertNotIn('preceding.running_elapsed_time_ms',fields)

    def test_missing_historical_elapsed_invalidates_training_without_zero_fallback(self):
        values,events=self._sample_block(known=False)
        self.assertFalse(values['measured_work_valid'])
        self.assertFalse(events[0]['valid'])
        self.assertIsNone(events[0]['admission_elapsed_time_ms'])

    def test_completion_without_mode_preserves_original_elapsed(self):
        values,events=self._sample_block(mode=False)
        self.assertTrue(values['measured_work_valid'])
        self.assertEqual(200.,values['sample_running_elapsed_time_ms'])
        self.assertEqual([],events)

if __name__=='__main__':
    unittest.main(verbosity=2)
