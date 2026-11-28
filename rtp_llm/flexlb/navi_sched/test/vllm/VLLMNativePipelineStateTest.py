import importlib.util
from collections import deque
from concurrent.futures import Future
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest import mock


source = Path(__file__).resolve().parents[2] / "adapters/vllm/VLLMNativePipelineState.py"
spec = importlib.util.spec_from_file_location("native_pipeline_state_under_test", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class NativeStepper:
    def __init__(self):
        self.stages = 4
        self.wave = SimpleNamespace(open=False)
        self.in_flight = deque()
        self.slots = []
        self.first_slot = 0
        self.last_real = -1
        self.measured = 0.125
        self.step_calls = 0
        self.dispatch_calls = 0
        self.dispatch_action = None
        self.step_result = object()
        self.step_error = None
        self.core = SimpleNamespace(
            scheduler=SimpleNamespace(num_waiting_for_streaming_input=0, running=[]),
            step_fn=self.step,
        )

    def step(self):
        self.step_calls += 1
        if self.step_error is not None:
            raise self.step_error
        return self.step_result

    def _dispatch(self):
        self.dispatch_calls += 1
        return self.dispatch_action() if self.dispatch_action else False

    def _push(self):
        raise AssertionError("this closed wave has no push")


class NativePipelineStateTest(unittest.TestCase):
    def setUp(self):
        self.stepper = NativeStepper()
        self.reader = module.NativePipelineState(self.stepper)
        self.errors = []

    def record_error(self):
        self.errors.append(sys.exc_info()[1])

    def test_observation_only_keeps_existing_step_hook(self):
        original_step = self.stepper.core.step_fn
        self.reader.bind(None, self.record_error)
        self.assertIs(original_step, self.stepper.core.step_fn)

    def test_observation_only_remains_available_after_native_steps(self):
        self.reader.bind(None, self.record_error)
        for _ in range(3):
            self.assertIs(self.stepper.step_result, self.stepper.core.step_fn())
        self.assertEqual(3, self.stepper.step_calls)
        self.assertFalse(self.reader.failed)
        self.assertEqual([], self.errors)
        snapshot = self.reader.capture()
        self.assertEqual(4, snapshot.stages)
        self.assertEqual((), snapshot.results)

    def test_observation_only_adds_no_per_step_capture_or_report(self):
        self.reader.bind(None, self.record_error)
        with mock.patch.object(self.reader, "capture", wraps=self.reader.capture) as capture:
            self.stepper.core.step_fn()
            self.stepper.core.step_fn()
            capture.assert_not_called()
            self.reader.capture()
            self.assertEqual(1, capture.call_count)
        self.assertFalse(self.reader.failed)

    def test_observation_only_retains_load_carrier_without_consuming_future(self):
        self.reader.bind(None, self.record_error)
        output = object()
        future = Future()

        def dispatch():
            self.reader.record_decision(id(output), (), load_request_ids=(42,))
            self.stepper.in_flight.append(SimpleNamespace(
                scheduler_output=output, exec_future=future, future=future,
                dispatched=time.monotonic(), slot=None,
            ))
            return True

        self.stepper.dispatch_action = dispatch
        with mock.patch.object(future, "result", side_effect=AssertionError("consumed Future")) as consume:
            self.assertIs(True, self.stepper._dispatch())
            snapshot = self.reader.capture()
            consume.assert_not_called()
        self.assertEqual(1, self.stepper.dispatch_calls)
        self.assertEqual((42,), snapshot.results[0].load_request_ids)
        self.assertFalse(snapshot.results[0].result_ready)
        self.assertFalse(self.reader.failed)
        self.assertEqual([], self.errors)

    def test_explicit_callback_publishes_only_changed_native_state(self):
        report = mock.Mock()
        self.reader.bind(report, self.record_error)
        self.assertIs(self.stepper.step_result, self.stepper.core.step_fn())
        self.stepper.core.step_fn()
        self.assertEqual(1, report.call_count)
        self.stepper.core.scheduler.running.append(object())
        self.stepper.core.step_fn()
        self.assertEqual(2, report.call_count)
        self.assertEqual(3, self.stepper.step_calls)
        self.assertFalse(self.reader.failed)
        self.assertEqual([], self.errors)

    def test_explicit_callback_failure_preserves_native_result_and_reports_once(self):
        report = mock.Mock(side_effect=RuntimeError("report failed"))
        self.reader.bind(report, self.record_error)
        for _ in range(2):
            self.assertIs(self.stepper.step_result, self.stepper.core.step_fn())
        self.assertTrue(self.reader.failed)
        self.assertEqual(1, report.call_count)
        self.assertEqual(1, len(self.errors))
        self.assertIsInstance(self.errors[0], RuntimeError)
        self.assertEqual(2, self.stepper.step_calls)

    def test_native_step_failure_still_propagates_in_both_modes(self):
        for callback in (None, mock.Mock()):
            with self.subTest(observation_only=callback is None):
                stepper = NativeStepper()
                reader = module.NativePipelineState(stepper)
                reader.bind(callback, self.record_error)
                error = RuntimeError("native failed")
                stepper.step_error = error
                with self.assertRaises(RuntimeError) as raised:
                    stepper.core.step_fn()
                self.assertIs(error, raised.exception)
                self.assertFalse(reader.failed)
        self.assertEqual([], self.errors)


if __name__ == "__main__":
    unittest.main()
