"""Exercise physical-window sampling through a real native Agent and model."""

import ctypes
import os
import re
import select
import tempfile
import time
import unittest
from pathlib import Path

from navi_sched.adapters.vllm.VLLMExecutionSample import (
    PrefillExecutionSample,
    PrefillStepRequest,
)
from navi_sched.adapters.vllm.VLLMExecutionWindow import PrefillExecutionWindowAssembler

from navi_sched import NaviSchedAgent
from navi_sched import _navi_sched_core as core


class VLLMExecutionWindowIntegrationTest(unittest.TestCase):
    def test_empty_control_steps_do_not_discard_prefill_samples(self):
        from types import SimpleNamespace

        from navi_sched.adapters.vllm.VLLMEngineObserver import VLLMEngineObserver

        def sample(identity):
            start = identity * 1000000
            return PrefillExecutionSample(
                identity,
                (PrefillStepRequest(identity, 0, 8192, 0, identity),),
                start,
                start + 4000000,
                8192,
            )

        observer = VLLMEngineObserver.__new__(VLLMEngineObserver)
        observer._native_ingress = True
        observer._v6d_scheduler = None
        observer._observed_execution_window_training = True
        assembler = PrefillExecutionWindowAssembler((4,))
        observer._prefill_execution_windows = assembler
        assembler.register(1)
        observer._prepare_execution_step(
            SimpleNamespace(num_scheduled_tokens={}),
            lambda identity: self.fail("empty control step must not read requests"),
            SimpleNamespace(started_ns=2, prefill_tokens=0, batch_progress=[]),
        )
        assembler.register(3)
        windows = list(assembler.complete(2, None))
        windows.extend(assembler.complete(1, sample(1)))
        windows.extend(assembler.complete(3, sample(3)))
        self.assertEqual([1, 3], [s.step_id for w in windows for s in w.steps])
        self.assertEqual(0, assembler.rejected_windows)

    def test_observed_windows_train_without_request_lifecycle_events(self):
        self.assertEqual(1, core.VLLM_SHARED_EXECUTION_SAMPLES_VERSION)
        original = os.getcwd()
        agent = None
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        with tempfile.TemporaryDirectory(prefix="navi-execution-window-") as directory:
            root = Path(directory)
            watched = set()
            try:
                os.chdir(directory)
                policy = core.TPUPrefillExecutionPolicyConfig()
                policy.pcp_size = 8
                policy.interleave_size = 256
                policy.compile_sizes = [512, 1024]
                policy.max_scheduled_tokens = 8192
                policy.long_prefill_token_threshold = 8192
                policy.hash_block_size = 16
                policy.scheduler_block_size = 16
                agent = NaviSchedAgent(
                    "execution_window_test",
                    server_ip="127.0.0.1",
                    server_port=17093,
                    total_blocks=1024,
                    block_size=16,
                    linear_step=1,
                    hybrid_tail_required=False,
                    execution_geometry_enabled=True,
                    execution_cache_source_enabled=True,
                    prefill_execution_policy=policy,
                    runtime_config={"thread_num": 2},
                    name="execution_window_agent",
                    enable_kmon=False,
                    enable_snapshot=False,
                    wait_no_query=False,
                    latency_validation_enabled=True,
                    latency_fit_batch_size=1,
                    latency_fit_interval=0,
                    latency_warmup_sample_count=0,
                    latency_cold_start_fit_epochs=1,
                )
                agent.init()
                bridge = core._VLLMAgentBridge(agent, pd_separated=True)

                def record(requests):
                    return bridge.record_observed_execution_sample(
                        requests, 2000.0, 4, 32768, time.time_ns() // 1000
                    )

                self.assertFalse(record([(10, 32768, 0, 0)]))
                self.assertTrue(bridge.enable_observed_execution_window_training())
                self.assertFalse(record([(10, 32768, 0, -1)]))
                for identity in range(1, 13):
                    self.assertTrue(record([(identity, 32768, 0, 0)]))

                assembler = PrefillExecutionWindowAssembler((4,))
                for identity in (1001, 1002):
                    assembler.register(identity)
                first = PrefillExecutionSample(
                    1001,
                    (PrefillStepRequest(1001, 0, 8192, 0, 1),),
                    1,
                    2000000001,
                    8192,
                )
                second = PrefillExecutionSample(
                    1002,
                    (PrefillStepRequest(1002, 0, 8192, 0, 2),),
                    3000000001,
                    5000000001,
                    8192,
                )
                self.assertEqual((), assembler.complete(1001, first))
                windows = assembler.complete(1002, second)
                self.assertEqual(2, len(windows))
                for window in windows:
                    self.assertEqual(1, window.execution_step_count)
                    self.assertTrue(
                        bridge.record_observed_execution_sample(
                            window.wire_requests(),
                            window.elapsed_ms,
                            window.execution_step_count,
                            window.global_padded_tokens,
                            time.time_ns() // 1000,
                        )
                    )

                deadline = time.monotonic() + 20
                text = ""
                while True:
                    for path in [root, *root.rglob("*")]:
                        if path.is_dir() and path not in watched:
                            result = libc.inotify_add_watch(
                                fd, os.fsencode(path), 0x100 | 0x80 | 0x8
                            )
                            if result < 0:
                                raise OSError(ctypes.get_errno(), str(path))
                            watched.add(path)
                    text = "\n".join(
                        p.read_text(errors="replace")
                        for p in root.rglob("*.log")
                        if p.is_file()
                    )
                    samples = [
                        line
                        for line in text.splitlines()
                        if "[navi-latency-training-sample]" in line
                        and "type=observed_execution_sample " in line
                    ]
                    revisions = [
                        (int(before), int(after))
                        for line in text.splitlines()
                        if "[navi-latency-training-update]" in line
                        for before, after in re.findall(
                            r"revision_before=(\d+) revision_after=(\d+)", line
                        )
                    ]
                    if len(samples) == 14 and any(b > a for a, b in revisions):
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                        self.fail(
                            "native window samples did not train:\n" + text[-4000:]
                        )
                    try:
                        os.read(fd, 65536)
                    except BlockingIOError:
                        pass
            finally:
                if agent is not None:
                    agent.stop()
                os.chdir(original)
                os.close(fd)


class VLLMPipelineBatchSampleTest(unittest.TestCase):
    def make_case(self):
        from collections import deque
        from types import SimpleNamespace

        from navi_sched.adapters.vllm.VLLMEngineObserver import (
            VLLMEngineObserver,
            _AdmissionState,
            _ExecutionStartState,
            _ObservedWorkload,
            _PrefillWorkloadSlice,
            _WorkloadContribution,
        )

        observer = VLLMEngineObserver.__new__(VLLMEngineObserver)
        observer._native_ingress = True
        observer._pp_workload_timing_enabled = True
        observer._vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=4)
        )
        observer._engine_core = SimpleNamespace(batch_queue=deque(maxlen=4))
        events = []
        observer.on_request_lifecycle_event = events.append
        preceding = _ObservedWorkload(
            batch_count=2,
            request_count=3,
            running_elapsed_time_ms=150.0,
            observed_time_ns=999000000,
            running_capacity_prefill=_PrefillWorkloadSlice(2, 300, 20, 200.0, 3000.0),
            waiting_capacity_prefill=_PrefillWorkloadSlice(1, 400, 30, 900.0, 12000.0),
        )
        admission = _AdmissionState(
            preceding_workload=preceding,
            incoming_running_capacity_request_ids=frozenset((1,)),
            active_executions=3,
        )
        state = SimpleNamespace(
            batch_id=11,
            schedule_batch_id=22,
            request_ids=(1, 2),
            cancelled_request_ids=set(),
            admission_started_ns=1000000000,
            execution_started_ns=1300000000,
            execution_step_count=3,
            admission_contributions={
                1: _WorkloadContribution(11, 1000, 100, 900, 100, 900),
                2: _WorkloadContribution(11, 2000, 200, 1800, 200, 1800),
            },
        )
        # Actual cache hits differ after queueing; they must not enter the replay inputs.
        args = (
            state,
            admission,
            _ExecutionStartState(active_executions=3),
            [(1, 1000, 900), (2, 2000, 1900)],
            300.0,
            700.0,
            2000000000,
            1234567,
        )
        return observer, args, events

    def test_records_frozen_arrival_features_and_separate_completion_labels(self):
        observer, args, events = self.make_case()
        observer._record_pipelined_prefill_batch_sample(*args)
        self.assertEqual(1, len(events))
        event = events[0]
        self.assertEqual("prefill_pipeline_batch_sample", event["event"])
        self.assertEqual(
            (11, 22), (event["native_batch_id"], event["native_schedule_batch_id"])
        )
        self.assertEqual(
            150.0, event["features_at_admission"]["running_elapsed_time_ms"]
        )
        self.assertEqual(
            400, event["features_at_admission"]["running"]["context_tokens"]
        )
        self.assertEqual(
            920, event["features_at_admission"]["running"]["compute_tokens"]
        )
        self.assertEqual(
            600, event["features_at_admission"]["waiting"]["context_tokens"]
        )
        self.assertEqual(
            1830, event["features_at_admission"]["waiting"]["compute_tokens"]
        )
        self.assertEqual(2500, event["cache_forecast_error_abs_tokens"])
        self.assertEqual(1000.0, event["observed_total_ms"])
        self.assertEqual(1234567, event["event_time_us"])
        self.assertEqual(3, event["observed_own_step_count"])
        self.assertEqual(
            300, args[1].preceding_workload.running_capacity_prefill.context_tokens
        )

    def test_non_pipeline_and_other_backend_paths_do_not_emit_samples(self):
        for mode in ("non_native", "non_pp", "serial", "diagnostic_disabled"):
            with self.subTest(mode=mode):
                observer, args, events = self.make_case()
                if mode == "non_native":
                    observer._native_ingress = False
                elif mode == "non_pp":
                    observer._vllm_config.parallel_config.pipeline_parallel_size = 1
                elif mode == "serial":
                    observer._engine_core.batch_queue = None
                else:
                    observer._pp_workload_timing_enabled = False
                observer._record_pipelined_prefill_batch_sample(*args)
                self.assertEqual([], events)

    def test_real_batch_completion_preserves_native_report_when_diagnostic_sink_fails(
        self,
    ):
        from types import SimpleNamespace
        from unittest import mock

        from navi_sched.adapters.vllm.VLLMEngineObserver import _ForwardBatchState

        observer, args, events = self.make_case()
        source, admission, start = args[:3]
        state = _ForwardBatchState(
            batch_id=11,
            schedule_batch_id=22,
            schedule_batch_size=2,
            pending_cache_tracked=False,
            native_batch=None,
            request_ids=(1, 2),
            external_request_ids={1: "r1", 2: "r2"},
            requests={
                1: SimpleNamespace(num_prompt_tokens=1000),
                2: SimpleNamespace(num_prompt_tokens=2000),
            },
            handle=object(),
            predicted_cache_hits={1: 100, 2: 200},
            initial_cache_hits={1: 900, 2: 1900},
            admission_started_ns=source.admission_started_ns,
            execution_started_ns=source.execution_started_ns,
            execution_step_count=3,
            admission_state=admission,
            admission_contributions=source.admission_contributions,
            execution_start_state=start,
        )
        observer._execution_geometry_enabled = False
        observer._cleanup_released_forward_batch = mock.Mock()
        observer._bridge = SimpleNamespace(
            finish_batch_with_execution=mock.Mock(return_value=True)
        )

        def sink(event):
            events.append(event)
            if event["event"] == "prefill_pipeline_batch_sample":
                raise RuntimeError("diagnostic sink unavailable")

        observer.on_request_lifecycle_event = sink
        with mock.patch("navi_sched.adapters.vllm.VLLMEngineObserver.logger") as logger:
            observer._finish_batch(
                state, completed_ns=2000000000, completed_time_us=1234567
            )
        observer._bridge.finish_batch_with_execution.assert_called_once()
        observer._cleanup_released_forward_batch.assert_called_once_with(state)
        self.assertTrue(state.released)
        self.assertEqual(
            ["prefill_pipeline_batch_sample", "legacy_native_batch_clock"],
            [event["event"] for event in events],
        )
        logger.exception.assert_called_once()

    def test_cancelled_batch_is_not_used_as_complete_arrival_sample(self):
        observer, args, events = self.make_case()
        args[0].cancelled_request_ids.add(1)
        observer._record_pipelined_prefill_batch_sample(*args)
        self.assertEqual([], events)

    def test_diagnostic_failure_does_not_escape_into_batch_completion(self):
        from unittest import mock

        observer, args, _ = self.make_case()
        observer.on_request_lifecycle_event = mock.Mock(
            side_effect=RuntimeError("sink unavailable")
        )
        with mock.patch("navi_sched.adapters.vllm.VLLMEngineObserver.logger") as logger:
            observer._record_pipelined_prefill_batch_sample(*args)
        logger.exception.assert_called_once()


if __name__ == "__main__":
    unittest.main()
