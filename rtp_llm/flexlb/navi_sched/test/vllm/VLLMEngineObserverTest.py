import asyncio
import contextlib
import gc
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

_native_stub = types.ModuleType("navi_sched._navi_sched_core")
_native_stub.NaviSchedAgent = type("NaviSchedAgent", (), {})
sys.modules.setdefault("navi_sched._navi_sched_core", _native_stub)

from navi_sched.adapters.vllm.VLLMAdapterUtils import make_kmon_tags
from navi_sched.adapters.vllm.VLLMBlockHash import stable_request_id
from navi_sched.adapters.vllm.VLLMNativePipelineState import NativePipelineState
from navi_sched.adapters.vllm.VLLMEngineObserver import (
    VLLMEngineObserver,
    _AdmissionState,
    _AdmissionPriorCacheForecast,
    _AgentConfig,
    _BatchProgress,
    _ExecutionStep,
    _ForwardBatchState,
    _latency_history_fingerprint,
    _latency_model_fingerprint,
    _latency_tuning_defaults,
    _native_pd_latency_model_fingerprint,
    _native_pp_unified_latency_enabled,
    _NaviSchedScheduler,
    _ObservedWorkload,
    _PrefillWorkloadSlice,
    _RequestCacheLookup,
    _ResolvedCacheHit,
    _RoutingOutputQueue,
    _SerialPrefillTestBridge,
    _vllm_binary_fingerprint,
)
from navi_sched_vllm_plugin import register_vllm_plugin


def _install_vllm_test_stubs():
    try:
        import msgspec
    except ImportError:
        sys.modules["msgspec"] = types.ModuleType("msgspec")

    try:
        import vllm.distributed.kv_events

        return
    except ImportError:
        pass

    class CacheEvent:
        def __init__(self, **values):
            self.__dict__.update(values)

    vllm = types.ModuleType("vllm")
    distributed = types.ModuleType("vllm.distributed")
    kv_events = types.ModuleType("vllm.distributed.kv_events")
    kv_events.MEDIUM_GPU = "GPU"
    kv_events.BlockStored = type("BlockStored", (CacheEvent,), {})
    kv_events.BlockRemoved = type("BlockRemoved", (CacheEvent,), {})
    kv_events.AllBlocksCleared = type("AllBlocksCleared", (CacheEvent,), {})
    plugins = types.ModuleType("vllm.plugins")
    plugins.get_engine_adapter = lambda: getattr(plugins, "_engine_adapter", None)
    plugins.register_engine_adapter = lambda adapter: setattr(
        plugins, "_engine_adapter", adapter
    )
    plugins.register_engine_observer_factory = lambda unused: None
    plugins.register_engine_transport_factory = lambda unused: None
    plugins.register_renderer_transport_factory = lambda unused: None
    plugins.register_frontend_request_interceptor = lambda unused: None
    plugins.register_frontend_app_configurator = lambda unused: None
    plugins.register_scheduler_iteration_metrics = lambda: None
    v1 = types.ModuleType("vllm.v1")
    outputs = types.ModuleType("vllm.v1.outputs")

    class AsyncModelRunnerOutput:
        def get_output(self):
            raise NotImplementedError

    outputs.AsyncModelRunnerOutput = AsyncModelRunnerOutput
    core = types.ModuleType("vllm.v1.core")
    kv_cache_interface = types.ModuleType("vllm.v1.kv_cache_interface")
    kv_cache_interface.iter_layer_specs = lambda spec: iter((spec,))
    kv_cache_utils = types.ModuleType("vllm.v1.core.kv_cache_utils")

    def resolve_kv_cache_block_sizes(kv_cache_config, vllm_config):
        assert len(kv_cache_config.kv_cache_groups) == 1
        parallel = vllm_config.parallel_config
        scale = (
            parallel.decode_context_parallel_size
            * parallel.prefill_context_parallel_size
        )
        return (
            kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size * scale,
            vllm_config.cache_config.block_size * scale,
        )

    kv_cache_utils.resolve_kv_cache_block_sizes = resolve_kv_cache_block_sizes
    kv_cache_utils.resolve_dcp_kv_block_size = lambda spec, dcp: spec.block_size * dcp
    core.kv_cache_utils = kv_cache_utils
    engine = types.ModuleType("vllm.v1.engine")
    request = types.ModuleType("vllm.v1.request")

    class EngineOutput:
        def __init__(self, **values):
            self.__dict__.update(values)

    engine.EngineCoreOutput = type("EngineCoreOutput", (EngineOutput,), {})
    engine.EngineCoreOutputs = type("EngineCoreOutputs", (EngineOutput,), {})
    engine.EngineCoreRequestType = SimpleNamespace(ABORT="abort")
    engine.FinishReason = SimpleNamespace(ERROR="error")
    request.RequestStatus = SimpleNamespace(FINISHED_ABORTED="aborted")
    vllm.distributed = distributed
    vllm.plugins = plugins
    vllm.v1 = v1
    distributed.kv_events = kv_events
    v1.outputs = outputs
    v1.engine = engine
    v1.request = request
    sys.modules.update(
        {
            "vllm": vllm,
            "vllm.distributed": distributed,
            "vllm.distributed.kv_events": kv_events,
            "vllm.plugins": plugins,
            "vllm.v1": v1,
            "vllm.v1.outputs": outputs,
            "vllm.v1.core": core,
            "vllm.v1.kv_cache_interface": kv_cache_interface,
            "vllm.v1.core.kv_cache_utils": kv_cache_utils,
            "vllm.v1.engine": engine,
            "vllm.v1.request": request,
        }
    )


_install_vllm_test_stubs()

from vllm.v1.request import RequestStatus


class FakeAgent:
    def __init__(self, **options):
        self.options = options
        self.stopped = False

    def init(self):
        return self

    def stop(self):
        self.stopped = True


class FullAttentionSpec:
    def __init__(self, block_size):
        self.block_size = block_size


class MambaSpec:
    def __init__(self, block_size, mode="align"):
        self.block_size = block_size
        self.mamba_cache_mode = mode


class FakeBlockPool:
    def __init__(self):
        self.listener = None
        self.touched = []
        self.freed = []

    def set_block_lifecycle_listener(self, listener):
        self.listener = listener

    def touch(self, blocks):
        self.touched.extend(blocks)

    def free_blocks(self, blocks):
        self.freed.extend(blocks)


class FakeKVCacheManager:
    def __init__(self):
        self.block_pool = FakeBlockPool()
        self.coordinator = SimpleNamespace(attention_groups=())
        self.promoted_requests = []

    def get_computed_blocks(self, request):
        self.promoted_requests.append(request)
        return SimpleNamespace(blocks=(list(request.block_hashes),)), 0, 0


class FakeHandle:
    def __init__(self, deleted):
        self.deleted = deleted

    def __del__(self):
        self.deleted.append(True)


class FakeForwardBatch:
    def __init__(
        self,
        batch_id,
        request_ids,
        predicted_cache_hits=None,
        *,
        schedule_batch_id=None,
        schedule_batch_size=None,
        pending_cache_tracked=True,
    ):
        self.batch_id = batch_id
        self.schedule_batch_id = (
            batch_id if schedule_batch_id is None else schedule_batch_id
        )
        self.schedule_batch_size = (
            len(request_ids) if schedule_batch_size is None else schedule_batch_size
        )
        self.pending_cache_tracked = pending_cache_tracked
        self.ready_time_ns = time.monotonic_ns()
        self.ready_time_us = time.time_ns() // 1000
        self.payloads = [value.encode("utf-8") for value in request_ids]
        self.request_ids = [stable_request_id(value) for value in request_ids]
        if predicted_cache_hits is None:
            predicted_cache_hits = [0] * len(request_ids)
        self.predicted_cache_hit_token_counts = list(predicted_cache_hits)


class FakeForwardRequestTouch:
    def __init__(self, request_id, payload, accepted_time_ns=None):
        self.request_id = request_id
        self.payload = payload
        self.accepted_time_ns = (
            time.monotonic_ns() if accepted_time_ns is None else accepted_time_ns
        )


class FakeBridge:
    def __init__(self, agent, pd_separated=False, prefill_capacity_split=False):
        self.agent = agent
        self.pd_separated = pd_separated
        self.prefill_capacity_split = prefill_capacity_split
        self.started = []
        self.finished = []
        self.inserted = []
        self.removed = []
        self.cache_updates = []
        self.cache_lifecycle_audits = []
        self.cache_touch_audits = []
        self.deleted = []
        self.released_forward_batches = []
        self.aborted_forward_requests = []
        self.backend_execution_count = 0
        self.backend_execution_events = []
        self.backend_execution_request_counts = []
        self.execution_boundaries = []
        self.resolved_execution_cache = []
        self.execution_steps = []
        self.backend_batch_executions = []
        self.workloads = []
        self.workload_elapsed_request_counts = []
        self.native_cache_query_reports = []
        self.native_pp_reports = []
        self.queued_forward_batches = []
        self.queued_forward_request_touches = []
        self.cancelled_requests = []
        self.published_responses = []
        self.failed_responses = []
        self.take_next_hook = None

    def enable_forwarding(self):
        return True

    def disable_forwarding(self):
        return True

    def take_next_forward_batch(self, unused_timeout_ms):
        if self.take_next_hook is not None:
            self.take_next_hook()
        if not self.queued_forward_batches:
            return None
        return self.queued_forward_batches.pop(0)

    def take_forward_request_touches(self, max_count=0):
        count = max_count or len(self.queued_forward_request_touches)
        values = self.queued_forward_request_touches[:count]
        self.queued_forward_request_touches = self.queued_forward_request_touches[
            count:
        ]
        return values

    def take_cancelled_requests(self):
        values = self.cancelled_requests
        self.cancelled_requests = []
        return values

    def publish_forward_response(self, request_id, frames, terminal):
        self.published_responses.append((request_id, list(frames), terminal))
        return True

    def fail_forward_response(self, request_id, message):
        self.failed_responses.append((request_id, message))
        return True

    def enable_observed_execution_window_training(self):
        self.observed_execution_window_training = True
        return True

    def record_observed_execution_sample(self, *values):
        return True

    def start_forward_batch(self, batch, request_ids, execution_start_time_us=0):
        self.execution_boundaries.append(("batch_start", execution_start_time_us))
        if set(request_ids) != set(batch.request_ids):
            return None
        handle = FakeHandle(self.deleted)
        self.started.append((batch.batch_id, list(request_ids), handle))
        return handle

    def resolve_forward_execution_cache(self, handle, values):
        self.resolved_execution_cache.append((handle, list(values)))
        return True

    def abort_forward_batch_requests(self, batch, request_ids):
        aborted = set(request_ids)
        if (
            not aborted
            or len(aborted) != len(request_ids)
            or not aborted < set(batch.request_ids)
        ):
            return False
        retained = [
            index
            for index, request_id in enumerate(batch.request_ids)
            if request_id not in aborted
        ]
        batch.request_ids = [batch.request_ids[index] for index in retained]
        batch.payloads = [batch.payloads[index] for index in retained]
        batch.predicted_cache_hit_token_counts = [
            batch.predicted_cache_hit_token_counts[index] for index in retained
        ]
        self.aborted_forward_requests.append((batch.batch_id, sorted(request_ids)))
        return True

    def release_forward_batch(self, batch):
        self.released_forward_batches.append(batch.batch_id)

    def start_batch(self, batch_id, requests, prefix_cache_prompt_tokens=None):
        self.prefix_cache_prompt_tokens = prefix_cache_prompt_tokens
        handle = FakeHandle(self.deleted)
        self.started.append((batch_id, requests, handle))
        return handle

    def finish_batch(self, handle, stats):
        self.finished.append((handle, stats))
        return True

    def finish_batch_with_execution(self, handle, stats, *execution_breakdown):
        self.finished.append((handle, stats))
        self.backend_batch_executions.append(execution_breakdown)
        return True

    def begin_backend_execution(self, event_time_us=0, request_count=0):
        self.backend_execution_request_counts.append(request_count)
        self.execution_boundaries.append(("begin", event_time_us))
        self.backend_execution_count += 1
        self.backend_execution_events.append("begin")
        return True

    def end_backend_execution(self, event_time_us=0):
        self.execution_boundaries.append(("end", event_time_us))
        if self.backend_execution_count <= 0:
            return False
        self.backend_execution_count -= 1
        self.backend_execution_events.append("end")
        return True

    def report_cache_inserted(self, hashes):
        self.inserted.append(hashes)

    def report_cache_removed(self, hashes):
        self.removed.append(hashes)

    def report_cache_updated(self, *values):
        self.cache_updates.append(values)

    def log_cache_lifecycle_audit(self, value):
        self.cache_lifecycle_audits.append(value)

    def log_request_lifecycle_audit(self, value):
        if not hasattr(self, "request_lifecycle_audits"):
            self.request_lifecycle_audits = []
        self.request_lifecycle_audits.append(json.loads(value))

    def report_execution_step(self, elapsed_ms, prefill_tokens, decode_tokens, first_start_overtake_count=0):
        self.execution_steps.append((elapsed_ms, prefill_tokens, decode_tokens))

    def log_cache_touch_audit(
        self,
        request_count,
        promoted_block_count,
        last_drain_request_count,
        last_drain_max_lag_ms,
        last_drain_processing_ms,
    ):
        self.cache_touch_audits.append(
            (
                request_count,
                promoted_block_count,
                last_drain_request_count,
                last_drain_max_lag_ms,
                last_drain_processing_ms,
            )
        )

    def report_workload(
        self,
        *values,
        running_elapsed_request_count,
        native_cache_queries=(),
        native_pp_state=None,
        native_pp_observation_age_us=-1,
    ):
        self.workload_elapsed_request_counts.append(running_elapsed_request_count)
        self.native_cache_query_reports.append(native_cache_queries)
        self.native_pp_reports.append((native_pp_state, native_pp_observation_age_us))
        self.workloads.append(values)
        return True


class LifecycleScheduler:
    def __init__(self, chunk_size=None, **unused):
        self.requests = {}
        self.running = []
        self.max_num_running_reqs = 8
        self.max_num_scheduled_tokens = 1024
        self.scheduler_config = SimpleNamespace(
            enable_chunked_prefill=chunk_size is not None,
            long_prefill_token_threshold=0,
        )
        self.kv_cache_manager = FakeKVCacheManager()
        self.chunk_size = chunk_size
        self.finish_calls = []
        self.request_observer = None

    def set_request_lifecycle_listener(self, listener):
        self.lifecycle_listener = listener

    def add_request(self, request):
        self.requests[request.request_id] = request

    def has_requests(self):
        return any(
            request.num_computed_tokens < request.num_prompt_tokens
            for request in self.requests.values()
        )

    def schedule(self):
        scheduled = {}
        for request in self.requests.values():
            remaining = request.num_prompt_tokens - request.num_computed_tokens
            if remaining <= 0:
                continue
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = request.num_computed_tokens
            token_count = (
                remaining
                if self.chunk_size is None
                else min(self.chunk_size, remaining)
            )
            request.num_computed_tokens += token_count
            scheduled[request.request_id] = token_count
        return SimpleNamespace(num_scheduled_tokens=scheduled)

    def update_from_output(self, scheduler_output, model_runner_output):
        return model_runner_output

    def finish_requests(self, request_ids, finished_status):
        self.finish_calls.append((request_ids, finished_status))
        if request_ids is None:
            request_ids = list(self.requests)
        elif isinstance(request_ids, str):
            request_ids = [request_ids]
        results = []
        for request_id in request_ids:
            request = self.requests.pop(request_id, None)
            if request is not None:
                request.status = finished_status
                results.append(request)
        return results


def finished_request_pairs(requests):
    return [(request.request_id, request.client_index) for request in requests]


class VLLMEngineObserverTest(unittest.TestCase):

    @contextlib.contextmanager
    def _assert_async_logs(self, logger, level):
        from navi_sched.adapters.vllm import ExecutionTimingProbe as probe

        # Keep the real bounded producer queue and real delivery implementation.
        # Start delivery after deterministic observer-clock mocks have exited,
        # while the original assertLogs handler is still installed. This avoids
        # racing log assertions or consuming the observer's clock fixture from
        # a separate writer thread; no production log or payload is mocked out.
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(threading.Thread, "start"):
                emitter = probe._AsyncDiagnosticEmitter(directory, capacity=4096)
            with mock.patch.object(probe, "_diagnostic_emitter", emitter):
                with self.assertLogs(logger, level=level) as captured:
                    try:
                        yield captured
                    finally:
                        emitter.close()
                        emitter.thread.start()
                        emitter.thread.join(3)
                        self.assertFalse(emitter.thread.is_alive())
                        self.assertEqual(0, emitter.failed)
                        self.assertFalse(emitter.loss_observed)

    def test_pd_batch_waits_for_model_output_before_finishing(self):
        observer = VLLMEngineObserver.__new__(VLLMEngineObserver)
        observer._native_ingress = True
        observer._pd_output_lock = threading.Lock()
        observer._pd_output_completed_ns = {}
        observer._pd_output_geometry = {}
        observer._pd_ready_finishes = {}
        observer._finish_batch = mock.Mock()
        scheduler_output = object()
        state = object()

        observer._defer_pd_batch_finish(id(scheduler_output), [state])
        observer._finish_batch.assert_not_called()
        observer.on_model_output(scheduler_output)
        observer._finish_batch.assert_called_once()
        self.assertIs(observer._finish_batch.call_args.args[0], state)
        self.assertGreater(observer._finish_batch.call_args.args[1], 0)

    def test_pd_model_output_before_execute_callback_still_finishes(self):
        observer = VLLMEngineObserver.__new__(VLLMEngineObserver)
        observer._native_ingress = True
        observer._pd_output_lock = threading.Lock()
        observer._pd_output_completed_ns = {}
        observer._pd_output_geometry = {}
        observer._pd_ready_finishes = {}
        observer._finish_batch = mock.Mock()
        scheduler_output = object()
        state = object()

        observer.on_model_output(scheduler_output)
        observer._defer_pd_batch_finish(id(scheduler_output), [state])
        observer._finish_batch.assert_called_once()
        self.assertIs(observer._finish_batch.call_args.args[0], state)
        self.assertGreater(observer._finish_batch.call_args.args[1], 0)

    def test_non_pd_model_output_does_not_change_batch_finish(self):
        observer = VLLMEngineObserver.__new__(VLLMEngineObserver)
        observer._native_ingress = False
        observer._finish_batch = mock.Mock()
        observer.on_model_output(object())
        observer._finish_batch.assert_not_called()

    def test_native_pd_local_batch_waits_for_model_output(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
            },
        ):
            observer = VLLMEngineObserver(
                self.make_config(), self.make_kv_cache_config()
            )
        request = self.make_request(
            prompt_tokens=100, computed_tokens=100, cached_tokens=0
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 100})

        self.run_execution_step(observer, output, {request.request_id: request})

        self.assertEqual(1, len(observer._bridge.started))
        self.assertEqual([], observer._bridge.finished)
        observer.on_model_output(output)
        self.assertEqual(1, len(observer._bridge.finished))
        observer.on_model_output(output)
        self.assertEqual(1, len(observer._bridge.finished))

    def test_native_none_result_cleans_local_request_between_prefill_chunks(self):
        for output_before_callback in (False, True):
            with self.subTest(output_before_callback=output_before_callback):
                observer = self.make_continuous_timing_observer()
                request = self.make_request(
                    prompt_tokens=48,
                    computed_tokens=0,
                    cached_tokens=0,
                    block_hashes=[bytes([index]) * 32 for index in range(3)],
                )
                for index in range(3):
                    self.assertEqual(index, len(observer._bridge.finished))
                    request.num_computed_tokens = (index + 1) * 16
                    output = SimpleNamespace(
                        num_scheduled_tokens={request.request_id: 16}
                    )
                    future = Future()
                    wrapped = observer.wrap_execute_model(
                        lambda unused_output, non_block: future,
                        SimpleNamespace(requests={request.request_id: request}),
                    )
                    wrapped(output, non_block=True)
                    self.assertEqual(index + 1, len(observer._bridge.started))
                    if output_before_callback:
                        observer.on_model_output(output)
                        self.assertEqual(index, len(observer._bridge.finished))
                        future.set_result(None)
                    else:
                        future.set_result(None)
                        self.assertEqual(index, len(observer._bridge.finished))
                        observer.on_model_output(output)
                    self.assertEqual(index + 1, len(observer._bridge.finished))
                    self.assertIs(
                        observer._bridge.started[index][2],
                        observer._bridge.finished[index][0],
                    )
                    self.assertEqual(0, observer._active_execution_count)
                    self.assertEqual(0, observer._bridge.backend_execution_count)
                    self.assertEqual({}, observer._pd_ready_finishes)
                    self.assertEqual({}, observer._pd_output_completed_ns)
                    self.assertEqual({}, observer._pd_output_geometry)

    def test_native_pd_local_batch_handles_output_before_future_callback(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
            },
        ):
            observer = VLLMEngineObserver(
                self.make_config(), self.make_kv_cache_config()
            )
        request = self.make_request(
            prompt_tokens=100, computed_tokens=100, cached_tokens=0
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 100})
        future = Future()
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: future,
            SimpleNamespace(requests={request.request_id: request}),
        )
        wrapped(output, non_block=True)
        observer.on_model_output(output)
        self.assertEqual([], observer._bridge.finished)
        future.set_result("model-output")
        self.assertEqual(1, len(observer._bridge.finished))

    def setUp(self):
        self.environment = mock.patch.dict(
            os.environ,
            {
                "PYTHONHASHSEED": "0",
                "VLLM_NAVI_SCHED_BIZ_NAME": "prefill",
                "VLLM_NAVI_SCHED_SERVER_PORT": "8001",
            },
            clear=False,
        )
        self.environment.start()
        os.environ.pop("VLLM_NAVI_SCHED_LATENCY_VALIDATION", None)
        self.agent_patch = mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.NaviSchedAgent",
            FakeAgent,
        )
        self.module_patch = mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.naviSchedModule",
            return_value=SimpleNamespace(
                _VLLMAgentBridge=FakeBridge,
                VLLM_NATIVE_CAPACITY_COMPLETION_FEATURES_VERSION=1,
                VLLM_NATIVE_CACHE_QUERY_INPUT_VERSION=1,
                _VLLMNativeCacheQuery=lambda **fields: SimpleNamespace(**fields),
            ),
        )
        self.agent_patch.start()
        self.module_patch.start()
        # These tests isolate observer state transitions from the wire codec.
        for name in ("forward_outputs_fields", "forward_request_type"):
            patch = mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver." + name,
                side_effect=lambda value: value,
            )
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self.module_patch.stop()
        self.agent_patch.stop()
        self.environment.stop()

    @staticmethod
    def make_config(block_size=16, dcp_size=1, pcp_size=1):
        return SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=block_size,
                enable_prefix_caching=True,
                prefix_caching_hash_algo="sha256",
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=dcp_size,
                prefill_context_parallel_size=pcp_size,
            ),
            scheduler_config=SimpleNamespace(
                enable_chunked_prefill=False,
                max_num_partial_prefills=1,
                max_long_partial_prefills=1,
                long_prefill_token_threshold=0,
                max_num_scheduled_tokens=1024,
            ),
        )

    @staticmethod
    def make_kv_cache_config(num_blocks=1024, specs=None):
        if specs is None:
            specs = [FullAttentionSpec(16)]
        return SimpleNamespace(
            num_blocks=num_blocks,
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in specs],
        )

    @staticmethod
    def make_request(
        *,
        prompt_tokens,
        computed_tokens,
        cached_tokens,
        request_id="internal-request",
        block_hashes=None,
    ):
        return SimpleNamespace(
            request_id=request_id,
            num_prompt_tokens=prompt_tokens,
            num_computed_tokens=computed_tokens,
            num_cached_tokens=cached_tokens,
            max_tokens=0,
            num_preemptions=0,
            block_hashes=([bytes(range(32))] if block_hashes is None else block_hashes),
            client_index=1,
        )

    @staticmethod
    def add_forward_batch(observer, batch_id, request_ids, predicted_cache_hits=None):
        batch = FakeForwardBatch(batch_id, request_ids, predicted_cache_hits)
        state = _ForwardBatchState(
            batch_id=batch_id,
            schedule_batch_id=batch.schedule_batch_id,
            schedule_batch_size=batch.schedule_batch_size,
            pending_cache_tracked=batch.pending_cache_tracked,
            native_batch=batch,
            request_ids=tuple(batch.request_ids),
            external_request_ids=dict(zip(batch.request_ids, request_ids, strict=True)),
            requests={},
            predicted_cache_hits=dict(
                zip(
                    batch.request_ids,
                    batch.predicted_cache_hit_token_counts,
                    strict=True,
                )
            ),
        )
        for request_id in batch.request_ids:
            observer._pending_forward_batches[request_id] = state
        return state

    @staticmethod
    def run_execution_step(observer, scheduler_output, requests):
        future = Future()
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: future,
            SimpleNamespace(requests=requests),
        )
        returned = wrapped(scheduler_output, non_block=True)
        future.set_result("model-output")
        return returned

    def test_engine_stage_audit_is_cached_opt_in_and_keeps_execution_reports(self):
        key = "VLLM_NAVI_SCHED_ENGINE_STAGE_AUDIT_ENABLED"
        for value, expected in ((None, False), ("0", False), ("1", True)):
            with self.subTest(value=value), mock.patch.dict(os.environ):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
                observer = VLLMEngineObserver(
                    self.make_config(), self.make_kv_cache_config()
                )
                self.assertIs(expected, observer._engine_stage_audit_enabled)
                os.environ[key] = "0" if expected else "1"
                self.assertIs(expected, observer._engine_stage_audit_enabled)
                request = self.make_request(
                    prompt_tokens=100, computed_tokens=100, cached_tokens=20
                )
                self.run_execution_step(
                    observer,
                    SimpleNamespace(num_scheduled_tokens={request.request_id: 80}),
                    {request.request_id: request},
                )
                self.assertEqual(
                    [(stable_request_id(request.request_id), 100, 20)],
                    observer._bridge.finished[0][1],
                )

    def test_reports_full_prefill_execution(self):
        config = self.make_config()
        observer = VLLMEngineObserver(config, self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=100,
            computed_tokens=100,
            cached_tokens=20,
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 80})

        self.run_execution_step(observer, output, {request.request_id: request})

        self.assertEqual("sha256_cbor", config.cache_config.prefix_caching_hash_algo)
        self.assertEqual(1024, observer._agent.options["total_blocks"])
        self.assertEqual(16, observer._agent.options["block_size"])
        self.assertEqual(1, observer._agent.options["linear_step"])
        self.assertFalse(observer._agent.options["hybrid_tail_required"])
        _, requests, handle = observer._bridge.started[0]
        request_id = stable_request_id("internal-request")
        self.assertEqual(
            [(request_id, 100, 16, [1736447835066146335])],
            requests,
        )
        self.assertEqual(
            [(handle, [(request_id, 100, 20)])],
            observer._bridge.finished,
        )

    def test_v029_scheduler_config_omits_retired_partial_prefill_fields(self):
        config = self.make_config()
        del config.scheduler_config.max_num_partial_prefills
        del config.scheduler_config.max_long_partial_prefills

        with self._assert_async_logs("vllm.navi_sched.agent", level="INFO") as logs:
            VLLMEngineObserver(config, self.make_kv_cache_config())

        scheduler_log = next(
            message
            for message in logs.output
            if "NaviSched VLLM native scheduler config" in message
        )
        self.assertIn("max_num_partial_prefills=None", scheduler_log)
        self.assertIn("max_long_partial_prefills=None", scheduler_log)

    def test_agent_uses_default_latency_configuration(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())

        self.assertEqual("sched_agent", observer._agent.options["name"])
        self.assertFalse(observer._agent.options["pd_separated"])
        self.assertEqual("unified", observer._agent.options["latency_prediction_mode"])
        self.assertFalse(observer._bridge.pd_separated)

        self.assertFalse(observer._agent.options["wait_no_query"])
        self.assertEqual(
            {
                "batch_timeout_ms": 50,
                "latency_fit_batch_size": 4,
                "latency_fit_interval": 0,
                "latency_learning_rate": 0.05,
                "latency_online_learning_rate_scale": 1.0,
                "latency_current_sample_weight": 4.0,
                "latency_error_clip_ms": 5000.0,
                "queue_learning_rate": 0.002,
                "latency_cold_start_fit_epochs": 100,
                "latency_history_limit": 2000,
                "latency_history_bucket_capacity": 64,
                "latency_persist_interval": 256,
                "latency_persist_interval_ms": 10000,
                "latency_warmup_sample_count": 10,
                "latency_validation_enabled": True,
                "latency_model_state_file": "",
                "residual_learning_rate": 0.0,
                "residual_update_error_floor_ms": 100.0,
                "residual_update_error_ratio": 0.1,
                "residual_idle_decay_start_ms": 120000,
                "residual_idle_decay_end_ms": 300000,
            },
            {
                key: observer._agent.options[key]
                for key in (
                    "batch_timeout_ms",
                    "latency_fit_batch_size",
                    "latency_fit_interval",
                    "latency_learning_rate",
                    "latency_online_learning_rate_scale",
                    "latency_current_sample_weight",
                    "latency_error_clip_ms",
                    "queue_learning_rate",
                    "latency_cold_start_fit_epochs",
                    "latency_history_limit",
                    "latency_history_bucket_capacity",
                    "latency_persist_interval",
                    "latency_persist_interval_ms",
                    "latency_warmup_sample_count",
                    "latency_validation_enabled",
                    "latency_model_state_file",
                    "residual_learning_rate",
                    "residual_update_error_floor_ms",
                    "residual_update_error_ratio",
                    "residual_idle_decay_start_ms",
                    "residual_idle_decay_end_ms",
                )
            },
        )

    def test_dashllm_native_prefill_trains_from_observed_batches(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
            },
        ):
            observer = VLLMEngineObserver(
                self.make_config(), self.make_kv_cache_config()
            )
        self.assertTrue(observer._native_ingress)
        self.assertTrue(observer._agent.options["pd_separated"])
        self.assertEqual("split", observer._agent.options["latency_prediction_mode"])
        self.assertTrue(observer._bridge.pd_separated)
        self.assertEqual(
            "split_native_start_output_complete_v3",
            json.loads(observer._agent.options["latency_model_fingerprint"])[
                "navi_latency_mode"
            ],
        )
        self.assertEqual(
            {"block_size": 16, "linear_step": 1, "hybrid_tail_required": False},
            json.loads(observer._agent.options["latency_model_fingerprint"])[
                "effective_cache_policy"
            ],
        )

    def test_native_pd_cache_geometry_separates_model_and_history_only(self):
        config = self.make_config()
        base = _latency_model_fingerprint(config)
        non_pd_history = _latency_history_fingerprint(config, base)
        simple = SimpleNamespace(
            block_size=768, linear_step=8, hybrid_tail_required=True
        )
        pcp = SimpleNamespace(block_size=6144, linear_step=1, hybrid_tail_required=True)
        native_simple = _native_pd_latency_model_fingerprint(base, simple)
        native_pcp = _native_pd_latency_model_fingerprint(base, pcp)

        self.assertNotEqual(native_simple, native_pcp)
        self.assertNotEqual(
            _latency_history_fingerprint(config, native_simple),
            _latency_history_fingerprint(config, native_pcp),
        )
        self.assertEqual(base, _latency_model_fingerprint(config))
        self.assertEqual(non_pd_history, _latency_history_fingerprint(config, base))

    def test_cache_source_environment_requires_native_geometry(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_EXECUTION_CACHE_SOURCE_ENABLED": "1",
                "VLLM_NAVI_SCHED_EXECUTION_GEOMETRY_ENABLED": "0",
            },
        ):
            with self.assertRaisesRegex(
                ValueError, "requires native execution geometry"
            ):
                VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())

    def test_cache_source_defaults_do_not_use_unified_model_coordinate_rates(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_EXECUTION_CACHE_SOURCE_ENABLED": "1",
            },
        ):
            os.environ.pop(
                "VLLM_NAVI_SCHED_LATENCY_PARAMETER_LEARNING_RATE_SCALES", None
            )
            self.assertEqual(
                [], _AgentConfig.from_env().latency_parameter_learning_rate_scales
            )
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_EXECUTION_CACHE_SOURCE_ENABLED": "0",
            },
        ):
            os.environ.pop(
                "VLLM_NAVI_SCHED_LATENCY_PARAMETER_LEARNING_RATE_SCALES", None
            )
            self.assertEqual(
                19, len(_AgentConfig.from_env().latency_parameter_learning_rate_scales)
            )

    def test_cache_source_parameter_rates_use_exact_native_coordinates(self):
        key = "VLLM_NAVI_SCHED_LATENCY_PARAMETER_LEARNING_RATE_SCALES"
        rates = [1.0] * 10 + [0.0] * 9 + [0.5, 2.0]
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_EXECUTION_CACHE_SOURCE_ENABLED": "1",
                key: json.dumps(rates),
            },
        ):
            self.assertEqual(
                rates, _AgentConfig.from_env().latency_parameter_learning_rate_scales
            )
        invalid = [[1.0] * n for n in (10, 17, 19)]
        for index, value in (
            (0, 0.0),
            (10, 1.0),
            (18, 1.0),
            (19, 0.0),
            (20, -1.0),
            (19, True),
            (20, "bad"),
        ):
            changed = list(rates)
            changed[index] = value
            invalid.append(changed)
        for changed in invalid:
            with (
                self.subTest(rates=changed),
                mock.patch.dict(
                    os.environ,
                    {
                        "VLLM_NAVI_SCHED_EXECUTION_CACHE_SOURCE_ENABLED": "1",
                        key: json.dumps(changed),
                    },
                ),
            ):
                with self.assertRaisesRegex(ValueError, "21 finite numbers"):
                    _AgentConfig.from_env()
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_EXECUTION_CACHE_SOURCE_ENABLED": "0",
                key: json.dumps(rates),
            },
        ):
            with self.assertRaisesRegex(ValueError, "10, 17, or 19"):
                _AgentConfig.from_env()

    def test_cache_source_environment_reaches_deferred_native_agent(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
                "VLLM_NAVI_SCHED_EXECUTION_GEOMETRY_ENABLED": "1",
                "VLLM_NAVI_SCHED_EXECUTION_CACHE_SOURCE_ENABLED": "1",
            },
        ):
            observer = VLLMEngineObserver(
                self.make_config(), self.make_kv_cache_config()
            )
        self.assertTrue(observer._native_ingress)
        self.assertTrue(observer._execution_geometry_enabled)
        self.assertTrue(observer._agent_options["execution_cache_source_enabled"])
        self.assertIsNone(observer._agent)

    def test_pp_latency_mode_keeps_split_by_default(self):
        config = self.make_config()
        config.parallel_config.pipeline_parallel_size = 4
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_native_pp_unified_latency_enabled(config, True))

    def test_pp_unified_mode_is_limited_to_native_pipeline(self):
        with mock.patch.dict(
            os.environ, {"VLLM_NAVI_SCHED_PP_LATENCY_MODE": "unified"}
        ):
            for native, stages, expected in (
                (True, 4, True),
                (True, 1, False),
                (False, 4, False),
                (False, 1, False),
            ):
                with self.subTest(native=native, stages=stages):
                    config = self.make_config()
                    config.parallel_config.pipeline_parallel_size = stages
                    self.assertEqual(
                        expected, _native_pp_unified_latency_enabled(config, native)
                    )

    def test_pp_unified19_selects_capacity_model_and_isolates_history(self):
        parallel = SimpleNamespace(pipeline_parallel_size=4)
        config = SimpleNamespace(parallel_config=parallel)
        with mock.patch.dict(
            os.environ, {"VLLM_NAVI_SCHED_PP_LATENCY_MODE": "unified19"}
        ):
            self.assertTrue(_native_pp_unified_latency_enabled(config, True))
            self.assertFalse(_native_pp_unified_latency_enabled(config, False))
        policy = SimpleNamespace(
            block_size=16, linear_step=1, hybrid_tail_required=False
        )
        split = _native_pd_latency_model_fingerprint(
            "{}",
            policy,
            execution_geometry_enabled=False,
            unified_latency_prediction=True,
        )
        capacity = _native_pd_latency_model_fingerprint(
            "{}",
            policy,
            execution_geometry_enabled=False,
            unified_latency_prediction=True,
            capacity_split_prediction=True,
        )
        self.assertNotEqual(split, capacity)
        self.assertEqual(
            "unified_native_prefill_capacity19_live_requests_v6",
            json.loads(capacity)["navi_latency_mode"],
        )

    def test_pp_capacity_history_distinguishes_complete_batch_training(self):
        policy = SimpleNamespace(
            block_size=16, linear_step=1, hybrid_tail_required=False
        )
        for unified, capacity in ((False, False), (True, False), (True, True)):
            with self.subTest(unified=unified, capacity=capacity):
                fingerprint = json.loads(
                    _native_pd_latency_model_fingerprint(
                        "{}",
                        policy,
                        unified_latency_prediction=unified,
                        capacity_split_prediction=capacity,
                    )
                )
                if unified and capacity:
                    self.assertEqual(
                        "forwarded_batch_start_order_predecessors_v3",
                        fingerprint["unified_capacity_training_scope"],
                    )
                else:
                    self.assertNotIn("unified_capacity_training_scope", fingerprint)

    def test_pp4_pcp8_without_native_stepper_does_not_select_native_timing(self):
        config = self.make_config(pcp_size=8)
        config.parallel_config.pipeline_parallel_size = 4
        config.parallel_config.data_parallel_size = 1
        environment = {
            "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
            "TPU_RAIDEN_JOB_NAME": "prefill",
            "VLLM_NAVI_SCHED_PP_LATENCY_MODE": "unified19",
        }
        with (
            mock.patch.dict(os.environ, environment),
            mock.patch.dict(sys.modules, {
                "vllm_torchtpu.core.pp_lazy_step": SimpleNamespace()
            }),
        ):
            observer = VLLMEngineObserver(config, self.make_kv_cache_config(specs=[FullAttentionSpec(128)]))
            observer._initialize_agent(SimpleNamespace())
        self.assertFalse(observer._native_pp_policy_enabled)
        self.assertTrue(observer._agent.options["pd_unified_capacity_split"])
        self.assertEqual("unified", observer._agent.options["latency_prediction_mode"])
        self.assertNotIn("native_prefill_pipeline_stages", observer._agent.options)
        self.assertNotIn("prefill_execution_policy", observer._agent.options)
        self.assertTrue(observer._bridge.prefill_capacity_split)

    def test_native_pp4_timing_requires_explicit_mode(self):
        config = self.make_config(pcp_size=8)
        config.parallel_config.pipeline_parallel_size = 4
        config.parallel_config.data_parallel_size = 1
        environment = {
            "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
            "TPU_RAIDEN_JOB_NAME": "prefill",
            "VLLM_NAVI_SCHED_PP_LATENCY_MODE": "native_pp4",
        }
        with (
            mock.patch.dict(os.environ, environment),
            mock.patch.dict(sys.modules, {
                "vllm_torchtpu.core.pp_lazy_step": SimpleNamespace(LazyPushStepper=object)
            }),
        ):
            observer = VLLMEngineObserver(config, self.make_kv_cache_config(specs=[FullAttentionSpec(128)]))
            with self.assertRaisesRegex(RuntimeError, "Native PP4 timing training"):
                observer._initialize_agent(SimpleNamespace())
        self.assertTrue(observer._native_pp_policy_enabled)

    def test_pp_v6d_capability_is_captured_after_native_handshake(self):
        module_name = "navi_sched.adapters.vllm.VLLMEngineObserver"
        config_module = "navi_sched.adapters.vllm.VLLMTairKvcmConfig"
        for mode in ("unified19", "split"):
            with self.subTest(mode=mode):
                config = self.make_config(pcp_size=8)
                config.parallel_config.pipeline_parallel_size = 4
                config.parallel_config.data_parallel_size = 1
                source = SimpleNamespace(_client_context=None)
                connector_type = type("V6dObjectConnector", (), {})
                connector = connector_type()
                connector.scheduler = source
                scheduler = SimpleNamespace(
                    connector=connector,
                    kv_cache_manager=SimpleNamespace(
                        coordinator=SimpleNamespace(attention_groups=()),
                        block_pool=FakeBlockPool(),
                    ),
                    scheduler_config=config.scheduler_config,
                )
                remote_config = object()
                captured_contexts = []

                def capture(actual_source, policy, native_module):
                    self.assertIs(source, actual_source)
                    captured_contexts.append(actual_source._client_context)
                    return remote_config if actual_source._client_context is not None else None

                serial_utils = types.ModuleType("vllm.v1.serial_utils")
                serial_utils.MsgpackEncoder = mock.Mock(return_value=object())
                with (
                    mock.patch.dict(os.environ, {
                        "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                        "TPU_RAIDEN_JOB_NAME": "prefill",
                        "VLLM_NAVI_SCHED_PP_LATENCY_MODE": mode,
                    }),
                    mock.patch.dict(sys.modules, {
                        "kvs_connector.v6d_object_connector": SimpleNamespace(
                            V6dObjectConnector=connector_type),
                        "vllm.v1.serial_utils": serial_utils,
                    }),
                    mock.patch(config_module + ".load_tair_kvcm_config", side_effect=capture),
                    mock.patch(module_name + ".get_vllm_full_attention_query_drop_tokens", return_value=0),
                    mock.patch(module_name + "._execution_timing_probe.bind_engine_loop_audit"),
                    mock.patch("navi_sched.adapters.vllm.VLLMExecutionGeometry.capture_execution_policy", return_value={}),
                    mock.patch(module_name + ".threading.Thread"),
                ):
                    observer = VLLMEngineObserver(
                        config, self.make_kv_cache_config(specs=[FullAttentionSpec(128)])
                    )
                    with (
                        mock.patch.object(observer, "_refresh_and_report_workload"),
                        mock.patch.object(observer, "_bind_backend_pipeline_stages"),
                        mock.patch.object(observer, "_enable_pp_kv_block_hashing"),
                        mock.patch.object(observer, "_bind_v6d_cache_source"),
                    ):
                        wrapped = observer.create_scheduler(lambda **kwargs: scheduler)
                        self.assertIsNone(observer._agent)
                        self.assertEqual([], captured_contexts)
                        # Native EngineCore does the worker handshake between
                        # create_scheduler and bind_engine_core.
                        source._client_context = object()
                        core = SimpleNamespace(
                            model_executor=SimpleNamespace(),
                            scheduler=wrapped, batch_queue=[], step_fn=lambda: None,
                            output_queue=object(), vllm_config=config,
                        )
                        observer.bind_engine_core(core)
                    self.assertEqual([source._client_context], captured_contexts)
                    self.assertIs(remote_config, observer._agent.options["tair_kvcm_config"])
                    self.assertIsNotNone(scheduler.kv_cache_manager.block_pool.listener)

    def test_pp_unified19_startup_preserves_pd_and_v6d(self):
        config = self.make_config()
        config.parallel_config.pipeline_parallel_size = 4
        environment = {
            "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
            "TPU_RAIDEN_JOB_NAME": "prefill",
            "VLLM_NAVI_SCHED_PP_LATENCY_MODE": "unified19",
            "VLLM_NAVI_SCHED_EXECUTION_GEOMETRY_ENABLED": "1",
            "VLLM_NAVI_SCHED_EXECUTION_CACHE_SOURCE_ENABLED": "1",
        }
        with mock.patch.dict(os.environ, environment):
            observer = VLLMEngineObserver(config, self.make_kv_cache_config())
        self.assertIsNone(observer._agent)
        self.assertTrue(observer._agent_options["pd_separated"])
        self.assertTrue(observer._agent_options["pd_unified_capacity_split"])
        self.assertEqual("unified", observer._agent_options["latency_prediction_mode"])
        self.assertFalse(observer._execution_geometry_enabled)
        self.assertFalse(observer._agent_options["execution_cache_source_enabled"])
        fingerprint = json.loads(observer._agent_options["latency_model_fingerprint"])
        self.assertEqual(
            "unified_native_prefill_capacity19_live_requests_v6",
            fingerprint["navi_latency_mode"],
        )
        scheduler = SimpleNamespace()
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver._v6d_peer_cache_enabled",
            return_value=True,
        ) as capability:
            observer._initialize_agent(scheduler)
        capability.assert_called_once_with(scheduler)
        self.assertTrue(observer._agent.options["pd_unified_capacity_split"])
        self.assertTrue(observer._bridge.pd_separated)
        self.assertTrue(observer._bridge.prefill_capacity_split)
        self.assertTrue(observer._agent.options["peer_cache_enabled"])

    def test_pp_unified19_rejects_core_without_completion_feature_support(self):
        config = self.make_config()
        config.parallel_config.pipeline_parallel_size = 4
        environment = {
            "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
            "TPU_RAIDEN_JOB_NAME": "prefill",
            "VLLM_NAVI_SCHED_PP_LATENCY_MODE": "unified19",
        }
        with mock.patch.dict(os.environ, environment):
            observer = VLLMEngineObserver(config, self.make_kv_cache_config())
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.naviSchedModule",
            return_value=SimpleNamespace(_VLLMAgentBridge=FakeBridge),
        ):
            with self.assertRaisesRegex(RuntimeError, "capacity completion features"):
                observer._initialize_agent(SimpleNamespace())
        self.assertIsNone(observer._agent)
        self.assertTrue(observer._shutdown_started)
        self.assertTrue(observer._forward_stop.is_set())

    def test_pp_latency_mode_rejects_invalid_value(self):
        config = self.make_config()
        config.parallel_config.pipeline_parallel_size = 4
        with mock.patch.dict(os.environ, {"VLLM_NAVI_SCHED_PP_LATENCY_MODE": "typo"}):
            with self.assertRaisesRegex(ValueError, "must be split"):
                _native_pp_unified_latency_enabled(config, True)

    def test_pp_unified_startup_preserves_scheduler_v6d_capability(self):
        config = self.make_config()
        config.parallel_config.pipeline_parallel_size = 4
        environment = {
            "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
            "TPU_RAIDEN_JOB_NAME": "prefill",
            "VLLM_NAVI_SCHED_PP_LATENCY_MODE": "unified",
            "VLLM_NAVI_SCHED_EXECUTION_GEOMETRY_ENABLED": "1",
            "VLLM_NAVI_SCHED_EXECUTION_CACHE_SOURCE_ENABLED": "1",
        }
        with mock.patch.dict(os.environ, environment):
            observer = VLLMEngineObserver(config, self.make_kv_cache_config())
        self.assertTrue(observer._native_ingress)
        self.assertFalse(observer._execution_geometry_enabled)
        self.assertIsNone(observer._agent)
        self.assertEqual("unified", observer._agent_options["latency_prediction_mode"])
        self.assertFalse(observer._agent_options["execution_cache_source_enabled"])
        fingerprint = json.loads(observer._agent_options["latency_model_fingerprint"])
        self.assertEqual(
            "unified_native_prefill_admission_v1", fingerprint["navi_latency_mode"]
        )
        self.assertEqual(
            "agent_batch_ready_v1", fingerprint["execution_start_boundary"]
        )
        scheduler = SimpleNamespace()
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver._v6d_peer_cache_enabled",
            return_value=True,
        ) as capability:
            observer._initialize_agent(scheduler)
        capability.assert_called_once_with(scheduler)
        self.assertTrue(observer._agent.options["peer_cache_enabled"])
        self.assertFalse(observer._observed_execution_window_training)

    def test_geometry_startup_passes_resolved_typed_policy_once(self):
        config = self.make_config(pcp_size=3)
        config.parallel_config.cp_kv_cache_interleave_size = 32
        config.compilation_config = SimpleNamespace(compile_sizes=[64, 128, 256])
        scheduler = SimpleNamespace(
            max_num_scheduled_tokens=768,
            scheduler_config=config.scheduler_config,
            hash_block_size=16,
            block_size=16,
            use_eagle=False,
            mamba_has_prefill_checkpoint_blocks=True,
            mamba_partial_cache_hit=False,
        )
        observer = VLLMEngineObserver.__new__(VLLMEngineObserver)
        observer._agent = None
        observer._agent_options = {"execution_cache_source_enabled": True}
        observer._native_ingress = True
        observer._pp_native_work_training = False
        observer._pp_capacity_unified = False
        observer._execution_geometry_enabled = True
        observer._vllm_config = config
        observer._cache_policy = SimpleNamespace(
            block_size=16, linear_step=1, hybrid_tail_required=False
        )
        native = SimpleNamespace(
            TPUPrefillExecutionPolicyConfig=SimpleNamespace,
            _VLLMAgentBridge=FakeBridge,
            VLLM_SHARED_EXECUTION_SAMPLES_VERSION=1,
        )
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.naviSchedModule",
            return_value=native,
        ):
            observer._initialize_agent(scheduler)
            policy = observer._prefill_execution_policy
            self.assertTrue(observer._bridge.observed_execution_window_training)
            self.assertIs(policy, observer._agent.options["prefill_execution_policy"])
            self.assertEqual(3, policy.pcp_size)
            self.assertEqual([64, 128, 256], policy.compile_sizes)
            self.assertEqual(768, policy.max_scheduled_tokens)
            self.assertTrue(policy.mamba_has_prefill_checkpoint_blocks)
            self.assertFalse(policy.async_kv_lookup_enabled)
            self.assertFalse(
                observer._agent.options["projected_execution_geometry_enabled"]
            )
            self.assertTrue(observer._agent.options["execution_cache_source_enabled"])
            self.assertEqual(
                "split_native_start_measured_cache_source_geometry_v35",
                json.loads(observer._agent.options["latency_model_fingerprint"])[
                    "navi_latency_mode"
                ],
            )
            observer._initialize_agent(None)
            self.assertIs(policy, observer._prefill_execution_policy)

    def test_dashllm_native_prefill_reports_observed_batch(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
            },
        ):
            observer = VLLMEngineObserver(
                self.make_config(), self.make_kv_cache_config()
            )
        request = self.make_request(
            prompt_tokens=100, computed_tokens=100, cached_tokens=20
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 80})
        self.run_execution_step(observer, output, {request.request_id: request})
        self.assertEqual(1, len(observer._bridge.started))
        _, _, handle = observer._bridge.started[0]
        self.assertEqual([], observer._bridge.finished)
        observer.on_model_output(output)
        self.assertEqual(
            [(handle, [(stable_request_id(request.request_id), 100, 20)])],
            observer._bridge.finished,
        )
        self.assertEqual([], observer._bridge.backend_batch_executions)

    def test_dashllm_native_ingress_keeps_master_forwarded_batch_ownership(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
            },
        ):
            observer = VLLMEngineObserver(
                self.make_config(), self.make_kv_cache_config()
            )
        scheduler = observer.create_scheduler(LifecycleScheduler, chunk_size=16)
        state = self.add_forward_batch(observer, 204, ["master-forwarded"])
        forwarded = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="master-forwarded",
        )

        scheduler.add_request(forwarded)

        self.assertIs(
            state,
            observer._scheduled_forward_batches[stable_request_id("master-forwarded")],
        )
        for _ in range(2):
            output = scheduler.schedule()
            self.run_execution_step(observer, output, scheduler.requests)
            observer.on_model_output(output)
        self.assertEqual(1, len(observer._bridge.started))
        self.assertEqual(204, observer._bridge.started[0][0])
        self.assertTrue(state.released)

        native = self.make_request(
            prompt_tokens=16,
            computed_tokens=0,
            cached_tokens=0,
            request_id="native-ingress",
        )
        scheduler.add_request(native)
        self.assertIn("native-ingress", scheduler.requests)
        output = scheduler.schedule()
        self.run_execution_step(observer, output, scheduler.requests)
        self.assertEqual(2, len(observer._bridge.started))
        self.assertNotEqual(204, observer._bridge.started[1][0])

    def test_latency_model_fingerprint_separates_runtime_shape(self):
        config = self.make_config()
        config.parallel_config = SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            data_parallel_size=2,
        )
        config.scheduler_config.max_num_seqs = 64
        config.model_config = SimpleNamespace(model="qwen3", dtype="bfloat16")
        with (
            mock.patch.dict(
                os.environ,
                {"VLLM_NAVI_SCHED_ACCELERATOR_TYPE": "rtx_pro_5000"},
            ),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver._runtime_accelerator_type",
                return_value="",
            ),
        ):
            first = _latency_model_fingerprint(config)
            first_values = json.loads(first)
            config.parallel_config.tensor_parallel_size = 2
            second = _latency_model_fingerprint(config)

        self.assertEqual("vllm", first_values["framework"])
        self.assertEqual("rtx_pro_5000", first_values["hardware"])
        self.assertEqual(4, first_values["tensor_parallel_size"])
        self.assertEqual("qwen3", first_values["model"])
        self.assertNotEqual(first, second)

    def test_latency_model_fingerprint_separates_release_versions(self):
        config = self.make_config()
        with mock.patch.dict(os.environ, {"NAVI_SCHED_MODEL_VERSION": "release-a"}):
            first = json.loads(_latency_model_fingerprint(config))
        with mock.patch.dict(os.environ, {"NAVI_SCHED_MODEL_VERSION": "release-b"}):
            second = json.loads(_latency_model_fingerprint(config))
        self.assertEqual(3, first["fingerprint_version"])
        self.assertEqual("release-a", first["navi_sched_model_version"])
        self.assertNotEqual(first, second)

    def test_history_fingerprint_is_stricter_than_formula_fingerprint(self):
        config = self.make_config()
        config.parallel_config.tensor_parallel_size = 4
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.data_parallel_size = 1
        config.scheduler_config.max_num_seqs = 64
        config.model_config = SimpleNamespace(
            model="qwen3",
            revision="revision-a",
            code_revision="code-a",
            dtype="bfloat16",
            quantization="fp8",
            max_model_len=65536,
            hf_config=SimpleNamespace(_commit_hash="commit-a"),
        )
        with mock.patch.dict(
            os.environ,
            {"VLLM_NAVI_SCHED_ACCELERATOR_TYPE": "rtx_pro_5000"},
        ):
            formula_a = _latency_model_fingerprint(config)
            history_a = _latency_history_fingerprint(config, formula_a)
            config.scheduler_config.max_num_seqs = 128
            config.model_config.revision = "revision-b"
            formula_b = _latency_model_fingerprint(config)
            history_b = _latency_history_fingerprint(config, formula_b)

        self.assertEqual(formula_a, formula_b)
        self.assertNotEqual(history_a, history_b)

    def test_formula_fingerprint_includes_quantization_and_speculative_decode(self):
        config = self.make_config()
        config.parallel_config.tensor_parallel_size = 4
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.data_parallel_size = 1
        config.model_config = SimpleNamespace(
            model="qwen3", dtype="bfloat16", quantization="fp8"
        )
        config.speculative_config = SimpleNamespace(
            method="eagle3",
            model="qwen3-draft",
            quantization="int8",
            num_speculative_tokens=3,
            draft_tensor_parallel_size=1,
            parallel_drafting=False,
        )
        with mock.patch.dict(
            os.environ,
            {"VLLM_NAVI_SCHED_ACCELERATOR_TYPE": "rtx_pro_5000"},
        ):
            first = _latency_model_fingerprint(config)
            config.speculative_config.num_speculative_tokens = 5
            second = _latency_model_fingerprint(config)
            config.speculative_config.num_speculative_tokens = 3
            config.model_config.quantization = "int4"
            third = _latency_model_fingerprint(config)

        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_binary_fingerprint_tracks_vllm_package(self):
        vllm = sys.modules["vllm"]
        with mock.patch.object(vllm, "__version__", "0.17.1-a", create=True):
            first = _vllm_binary_fingerprint()
        with mock.patch.object(vllm, "__version__", "0.17.1-b", create=True):
            second = _vllm_binary_fingerprint()
        self.assertNotEqual(first, second)

    def test_latency_model_fingerprint_uses_hardware_across_deployments(self):
        config = self.make_config()
        environment = {
            "GPU_MODEL_DETAIL": "AMD-Instinct-MI308X-OAM",
            "DEPLOYMENT_NAME": "cn_MI308X_2TP_spot",
        }
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver._runtime_accelerator_type",
                return_value="",
            ),
        ):
            first = _latency_model_fingerprint(config)
            os.environ["DEPLOYMENT_NAME"] = "cn_MI308X_2TP_spot_new"
            second = _latency_model_fingerprint(config)
            os.environ["GPU_MODEL_DETAIL"] = "RTX-PRO-5000-72GB"
            third = _latency_model_fingerprint(config)

        self.assertEqual("AMD-Instinct-MI308X-OAM", json.loads(first)["hardware"])
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_latency_model_fingerprint_ignores_temporary_copy_directory(self):
        config = self.make_config()
        config.model_config = SimpleNamespace(
            model="/tmp/model_temp_copy_first/0123456789abcdef0123456789abcdef",
            dtype="bfloat16",
        )
        with mock.patch.dict(
            os.environ,
            {"VLLM_NAVI_SCHED_ACCELERATOR_TYPE": "rtx_pro_5000"},
        ):
            first = _latency_model_fingerprint(config)
            config.model_config.model = (
                "/tmp/model_temp_copy_second/0123456789abcdef0123456789abcdef"
            )
            second = _latency_model_fingerprint(config)
            config.model_config.model = (
                "/tmp/model_temp_copy_second/0123456789abcdef0123456789abcdef/"
            )
            trailing_slash = _latency_model_fingerprint(config)
            config.model_config.model = (
                "/tmp/model_temp_copy_second/fedcba9876543210fedcba9876543210"
            )
            different_model = _latency_model_fingerprint(config)

        self.assertEqual(first, second)
        self.assertEqual(first, trailing_slash)
        self.assertNotEqual(first, different_model)
        self.assertEqual(
            "0123456789abcdef0123456789abcdef",
            json.loads(first)["model"],
        )

    def test_latency_model_fingerprint_preserves_regular_model_identity(self):
        config = self.make_config()
        models = (
            "qwen3",
            "/models/qwen3",
            "/models/0123456789abcdef0123456789abcdef",
            "/tmp/not_model_temp_copy_first/0123456789abcdef0123456789abcdef",
            "",
        )
        with mock.patch.dict(
            os.environ,
            {"VLLM_NAVI_SCHED_ACCELERATOR_TYPE": "rtx_pro_5000"},
        ):
            for model in models:
                with self.subTest(model=model):
                    config.model_config = SimpleNamespace(
                        model=model,
                        dtype="bfloat16",
                    )
                    fingerprint = _latency_model_fingerprint(config)
                    self.assertEqual(model, json.loads(fingerprint)["model"])

    def test_latency_model_fingerprint_prefers_vllm_detected_hardware(self):
        config = self.make_config()
        platforms = types.ModuleType("vllm.platforms")
        platforms.current_platform = SimpleNamespace(
            get_device_name=lambda: "AMD Instinct MI308X OAM"
        )
        environment = {
            "VLLM_NAVI_SCHED_ACCELERATOR_TYPE": "configured-accelerator",
            "GPU_MODEL_DETAIL": "metadata-accelerator",
            "DEPLOYMENT_NAME": "deployment-fallback",
        }
        with (
            mock.patch.dict(sys.modules, {"vllm.platforms": platforms}),
            mock.patch.dict(os.environ, environment, clear=True),
        ):
            fingerprint = _latency_model_fingerprint(config)

        self.assertEqual("AMD Instinct MI308X OAM", json.loads(fingerprint)["hardware"])

    def test_latency_model_fingerprint_falls_back_when_detection_fails(self):
        config = self.make_config()
        platforms = types.ModuleType("vllm.platforms")
        platforms.current_platform = SimpleNamespace(
            get_device_name=mock.Mock(side_effect=RuntimeError("probe failed"))
        )
        with (
            mock.patch.dict(sys.modules, {"vllm.platforms": platforms}),
            mock.patch.dict(
                os.environ, {"GPU_MODEL_DETAIL": "AMD-Instinct-MI308X-OAM"}, clear=True
            ),
        ):
            fingerprint = _latency_model_fingerprint(config)

        self.assertEqual("AMD-Instinct-MI308X-OAM", json.loads(fingerprint)["hardware"])

    def test_learning_rate_defaults_are_hardware_independent(self):
        self.assertEqual(
            {
                "latency_online_learning_rate_scale": 1.0,
                "residual_learning_rate": 0.0,
            },
            _latency_tuning_defaults(),
        )

    def test_hardware_does_not_change_learning_rate_defaults(self):
        environment = {
            "PYTHONHASHSEED": "0",
            "VLLM_NAVI_SCHED_BIZ_NAME": "prefill",
            "VLLM_NAVI_SCHED_SERVER_PORT": "8001",
            "VLLM_NAVI_SCHED_ACCELERATOR_TYPE": "AMD_Instinct_MI308X",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            observer = VLLMEngineObserver(
                self.make_config(), self.make_kv_cache_config()
            )
        self.assertEqual(
            1.0,
            observer._agent.options["latency_online_learning_rate_scale"],
        )
        self.assertEqual(0.0, observer._agent.options["residual_learning_rate"])
        observer.shutdown()

    def test_agent_uses_fixed_parameter_balancing_by_default(self):
        environment = {
            "PYTHONHASHSEED": "0",
            "VLLM_NAVI_SCHED_BIZ_NAME": "prefill",
            "VLLM_NAVI_SCHED_SERVER_PORT": "8001",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            observer = VLLMEngineObserver(
                self.make_config(), self.make_kv_cache_config()
            )
        self.assertEqual(
            [
                4.0,
                4.0,
                4.0,
                0.8786028245332912,
                1.0,
                0.10067484459438397,
                1.0,
                4.0,
                4.0,
                4.0,
                2.361137720592015,
                4.0,
                0.014903028756332782,
                0.00647920238757579,
                0.013073034322480767,
                4.0,
                4.0,
                1.0,
                1.0,
            ],
            observer._agent.options["latency_parameter_learning_rate_scales"],
        )
        self.assertEqual(0.05, observer._agent.options["latency_learning_rate"])
        self.assertEqual(
            1.0, observer._agent.options["latency_online_learning_rate_scale"]
        )
        observer.shutdown()

    def test_agent_adds_dashboard_kmon_tags(self):
        environment = {
            "DEPLOYMENT_NAME": "cn_MI308X_2TP_spot",
            "HIPPO_APP": "whale_prod_Qwen3",
            "HIPPO_ROLE": "Qwen3_cn_MI308X.inference_part0",
            "C2_GROUP": "Qwen3_cn_MI308X_2TP_spot",
            "HIPPO_SLOT_ID": "330485",
            "HIPPO_SLAVE_IP": "215.77.18.161",
            "VLLM_NAVI_SCHED_SERVER_IP": "215.77.99.80",
        }

        self.assertEqual(
            {
                "role": "agent",
                "deployment_name": "cn_MI308X_2TP_spot",
                "hippo_app": "whale_prod_Qwen3",
                "hippo_role": "Qwen3_cn_MI308X.inference_part0",
                "hippo_group": "Qwen3_cn_MI308X_2TP_spot",
                "slot_id": "330485",
                "host": "215.77.99.80",
                "host_ip": "215.77.99.80",
                "container_ip": "215.77.99.80",
                "physical_host_ip": "215.77.18.161",
            },
            make_kmon_tags("agent", environment),
        )

    def test_agent_uses_hippo_role_as_deployment_fallback(self):
        environment = {
            "HIPPO_ROLE": "cn_RTX_PRO_5000_72GB_4TP_spot",
            "VLLM_NAVI_SCHED_SERVER_IP": "215.77.99.81",
        }

        self.assertEqual(
            "cn_RTX_PRO_5000_72GB_4TP_spot",
            make_kmon_tags("agent", environment)["deployment_name"],
        )

    def test_spectrum_kmon_uses_business_app_without_changing_legacy_tags(self):
        environment = {
            "HIPPO_APP": "ds-9a5f276f-1",
            "SPECTRUM_APPLICATION_NAME": "wister.zwz-zhang7-qwen3.7-plus-L20G7-test",
        }
        self.assertEqual(
            "ds-9a5f276f-1", make_kmon_tags("agent", environment)["hippo_app"]
        )
        environment["SPECTRUM_WORKSPACE_ID"] = "bb1035d8"
        self.assertEqual(
            environment["SPECTRUM_APPLICATION_NAME"],
            make_kmon_tags("agent", environment)["hippo_app"],
        )
        tags = make_kmon_tags("agent", environment)
        self.assertEqual(environment["SPECTRUM_APPLICATION_NAME"], tags["hippo_app"])
        self.assertNotIn("hippo_app_id", tags)
        self.assertEqual(
            environment["SPECTRUM_APPLICATION_NAME"], tags["spectrum_application"]
        )
        self.assertEqual("bb1035d8", tags["spectrum_workspace_id"])
        self.assertEqual(
            environment["SPECTRUM_APPLICATION_NAME"],
            make_kmon_tags("client", environment)["hippo_app"],
        )

    def test_latency_error_clip_validation(self):
        for value in ("-1", "nan", "inf"):
            with (
                self.subTest(value=value),
                mock.patch.dict(
                    os.environ, {"VLLM_NAVI_SCHED_LATENCY_ERROR_CLIP_MS": value}
                ),
            ):
                with self.assertRaises(ValueError):
                    _AgentConfig.from_env()
        with mock.patch.dict(
            os.environ, {"VLLM_NAVI_SCHED_LATENCY_ERROR_CLIP_MS": "0"}
        ):
            self.assertEqual(0.0, _AgentConfig.from_env().latency_error_clip_ms)

    def test_agent_latency_configuration_can_be_overridden(self):
        environment = {
            "VLLM_NAVI_SCHED_BATCH_TIMEOUT_MS": "20",
            "VLLM_NAVI_SCHED_LATENCY_FIT_BATCH_SIZE": "1",
            "VLLM_NAVI_SCHED_LATENCY_FIT_INTERVAL": "1",
            "VLLM_NAVI_SCHED_LATENCY_LEARNING_RATE": "0.001",
            "VLLM_NAVI_SCHED_LATENCY_ONLINE_LEARNING_RATE_SCALE": "2.5",
            "VLLM_NAVI_SCHED_LATENCY_CURRENT_SAMPLE_WEIGHT": "3.0",
            "VLLM_NAVI_SCHED_LATENCY_ERROR_CLIP_MS": "1234",
            "VLLM_NAVI_SCHED_LATENCY_PARAMETER_LEARNING_RATE_SCALES": (
                "[1,1,1,1,1,1,1,1,1,1,1,0.4,1,1,16,1,1,0.01,1]"
            ),
            "VLLM_NAVI_SCHED_QUEUE_LEARNING_RATE": "0.003",
            "VLLM_NAVI_SCHED_LATENCY_COLD_START_FIT_EPOCHS": "7",
            "VLLM_NAVI_SCHED_LATENCY_HISTORY_LIMIT": "3000",
            "VLLM_NAVI_SCHED_LATENCY_HISTORY_BUCKET_CAPACITY": "17",
            "VLLM_NAVI_SCHED_LATENCY_PERSIST_INTERVAL": "1",
            "VLLM_NAVI_SCHED_LATENCY_PERSIST_INTERVAL_MS": "20",
            "VLLM_NAVI_SCHED_LATENCY_WARMUP_SAMPLE_COUNT": "9",
            "VLLM_NAVI_SCHED_LATENCY_VALIDATION": "true",
            "VLLM_NAVI_SCHED_LATENCY_MODEL_STATE_FILE": "/tmp/model.state",
            "VLLM_NAVI_SCHED_RESIDUAL_LEARNING_RATE": "0.07",
            "VLLM_NAVI_SCHED_RESIDUAL_UPDATE_ERROR_FLOOR_MS": "25.0",
            "VLLM_NAVI_SCHED_RESIDUAL_UPDATE_ERROR_RATIO": "0.2",
            "VLLM_NAVI_SCHED_RESIDUAL_IDLE_DECAY_START_MS": "4000",
            "VLLM_NAVI_SCHED_RESIDUAL_IDLE_DECAY_END_MS": "9000",
        }
        with mock.patch.dict(os.environ, environment):
            observer = VLLMEngineObserver(
                self.make_config(), self.make_kv_cache_config()
            )

        for key, expected in (
            ("batch_timeout_ms", 20),
            ("latency_fit_batch_size", 1),
            ("latency_fit_interval", 1),
            ("latency_learning_rate", 0.001),
            ("latency_online_learning_rate_scale", 2.5),
            ("latency_current_sample_weight", 3.0),
            ("latency_error_clip_ms", 1234.0),
            (
                "latency_parameter_learning_rate_scales",
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0.4,
                    1,
                    1,
                    16,
                    1,
                    1,
                    0.01,
                    1,
                ],
            ),
            ("queue_learning_rate", 0.003),
            ("latency_cold_start_fit_epochs", 7),
            ("latency_history_limit", 3000),
            ("latency_history_bucket_capacity", 17),
            ("latency_persist_interval", 1),
            ("latency_persist_interval_ms", 20),
            ("latency_warmup_sample_count", 9),
            ("latency_validation_enabled", True),
            ("latency_model_state_file", "/tmp/model.state"),
            ("residual_learning_rate", 0.07),
            ("residual_update_error_floor_ms", 25.0),
            ("residual_update_error_ratio", 0.2),
            ("residual_idle_decay_start_ms", 4000),
            ("residual_idle_decay_end_ms", 9000),
        ):
            self.assertEqual(expected, observer._agent.options[key])

    def test_agent_rejects_invalid_latency_persist_interval_ms(self):
        with mock.patch.dict(
            os.environ,
            {"VLLM_NAVI_SCHED_LATENCY_PERSIST_INTERVAL_MS": "0"},
        ):
            with self.assertRaisesRegex(
                ValueError,
                "VLLM_NAVI_SCHED_LATENCY_PERSIST_INTERVAL_MS must be positive",
            ):
                VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())

    def test_zero_residual_learning_rate_is_accepted(self):
        with mock.patch.dict(
            os.environ,
            {"VLLM_NAVI_SCHED_RESIDUAL_LEARNING_RATE": "0"},
        ):
            observer = VLLMEngineObserver(
                self.make_config(), self.make_kv_cache_config()
            )
        self.assertEqual(0.0, observer._agent.options["residual_learning_rate"])

    def test_agent_latency_validation_can_be_explicitly_disabled(self):
        with mock.patch.dict(
            os.environ,
            {"VLLM_NAVI_SCHED_LATENCY_VALIDATION": "false"},
        ):
            observer = VLLMEngineObserver(
                self.make_config(), self.make_kv_cache_config()
            )

        self.assertFalse(observer._agent.options["latency_validation_enabled"])

    def test_request_lifecycle_audit_links_batch_and_reports_config(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        config = observer._bridge.request_lifecycle_audits[-1]
        self.assertEqual(config["event"], "scheduler_config")
        self.assertTrue(config["lifecycle_supported"])
        observer._forward_request_states["123"] = SimpleNamespace(
            batch_id=77, schedule_batch_id=88
        )
        scheduler.request_observer.listener(
            {
                "event": "slot_release",
                "sequence": 12,
                "request_id": "123",
                "slot_delta": -1,
                "output_tokens": 256,
                "reason": "output_stop",
            }
        )
        event = observer._bridge.request_lifecycle_audits[-1]
        self.assertEqual(event["batch_id"], 77)
        self.assertEqual(event["schedule_batch_id"], 88)
        self.assertEqual(event["slot_delta"], -1)
        self.assertEqual(event["output_tokens"], 256)
        self.assertEqual(event["sequence"], 12)
        self.assertEqual(event["session"], config["session"])
        self.assertEqual(event["audit_failures"], 0)

    def test_scheduler_forwards_v029_prefill_throttle(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        native_schedule = mock.Mock(
            return_value=SimpleNamespace(num_scheduled_tokens={})
        )
        scheduler._scheduler.schedule = native_schedule

        scheduler.schedule(True)

        native_schedule.assert_called_once_with(True)

    def test_multigroup_cache_uses_shared_block_pool_capacity(self):
        observer = VLLMEngineObserver(
            self.make_config(),
            self.make_kv_cache_config(
                num_blocks=1024,
                specs=[FullAttentionSpec(16), MambaSpec(32)],
            ),
        )

        self.assertEqual(1024, observer._agent.options["total_blocks"])

    def test_pure_decode_and_cancelled_steps_are_audited(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        for cancelled in (False, True):
            future = Future()
            if cancelled:
                future.cancel()
            else:
                future.set_result(None)
            step = _ExecutionStep(
                started_ns=time.perf_counter_ns() - 5_000_000,
                decode_request_count=512,
                decode_tokens=512,
                running_request_count=512,
                waiting_request_count=80,
            )
            observer._complete_backend_execution(step, future)
            event = observer._bridge.request_lifecycle_audits[-1]
            self.assertEqual(event["event"], "execution_step")
            self.assertEqual(event["decode_requests"], 512)
            self.assertEqual(event["decode_tokens"], 512)
            self.assertEqual(event["prefill_tokens"], 0)
            self.assertEqual(event["waiting_count"], 80)
            self.assertEqual(event["success"], not cancelled)
            self.assertGreaterEqual(event["callback_elapsed_ms"], 5)
        self.assertEqual(len(observer._bridge.execution_steps), 1)
        elapsed_ms, prefill_tokens, decode_tokens = observer._bridge.execution_steps[0]
        self.assertGreaterEqual(elapsed_ms, 5)
        self.assertEqual(prefill_tokens, 0)
        self.assertEqual(decode_tokens, 512)

    def test_request_audit_failure_is_reported_on_next_event(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        with mock.patch.object(
            observer._bridge,
            "log_request_lifecycle_audit",
            side_effect=RuntimeError("disk unavailable"),
        ):
            observer.on_request_lifecycle_event({"event": "terminal"})
        observer.on_request_lifecycle_event({"event": "slot_release"})
        self.assertEqual(
            observer._bridge.request_lifecycle_audits[-1]["audit_failures"], 1
        )

    def test_pure_decode_audit_samples_queue_lengths_without_scanning_requests(self):
        class LengthOnly:
            def __init__(self, count):
                self.count = count

            def __len__(self):
                return self.count

            def __iter__(self):
                raise AssertionError("pure decode must not scan the request queue")

        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = SimpleNamespace(running=LengthOnly(512), waiting=LengthOnly(80))
        output = SimpleNamespace(
            preempted_req_ids={"preempted"},
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids={"resumed"}),
        )
        step = _ExecutionStep(
            started_ns=time.perf_counter_ns(),
            decode_request_count=511,
            decode_tokens=511,
        )
        observer._start_execution_tracking(step, scheduler, output)
        future = Future()
        future.set_result(None)
        observer._complete_backend_execution(step, future)
        event = observer._bridge.request_lifecycle_audits[-1]
        self.assertTrue(event["workload_counts_collected"])
        self.assertEqual(event["running_count"], 512)
        self.assertEqual(event["waiting_count"], 80)
        self.assertEqual(event["preempted_requests"], 1)
        self.assertEqual(event["resumed_requests"], 1)
        self.assertEqual(observer._active_execution_count, 0)

    def test_master_batch_waits_for_every_request_then_enters_native_scheduler(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        self.add_forward_batch(observer, 77, ["internal-1", "internal-2"])

        class FakeScheduler:
            def __init__(self, **unused):
                self.requests = {}
                self.running = []
                self.max_num_running_reqs = 8
                self.max_num_scheduled_tokens = 1024
                self.scheduler_config = SimpleNamespace(
                    enable_chunked_prefill=False,
                    long_prefill_token_threshold=0,
                )
                self.kv_cache_manager = SimpleNamespace(block_pool=FakeBlockPool())

            def add_request(self, request):
                self.requests[request.request_id] = request

            def has_requests(self):
                return bool(self.requests)

            def schedule(self):
                scheduled = {}
                for request in self.requests.values():
                    scheduled[request.request_id] = request.num_prompt_tokens
                    request.num_computed_tokens = request.num_prompt_tokens
                return SimpleNamespace(num_scheduled_tokens=scheduled)

        scheduler = observer.create_scheduler(FakeScheduler)
        first = self.make_request(
            prompt_tokens=16,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-1",
        )
        second = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-2",
        )

        scheduler.add_request(first)
        self.assertFalse(scheduler._scheduler.has_requests())
        scheduler.add_request(second)
        output = scheduler.schedule()

        self.assertEqual(
            {"internal-1", "internal-2"},
            set(output.num_scheduled_tokens),
        )
        self.run_execution_step(observer, output, scheduler.requests)
        self.assertEqual(77, observer._bridge.started[-1][0])
        self.assertEqual(
            {
                stable_request_id("internal-1"),
                stable_request_id("internal-2"),
            },
            set(observer._bridge.started[-1][1]),
        )
        self.assertEqual(1, len(observer._bridge.finished))

    def test_scheduler_add_request_records_stage_timing(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        self.add_forward_batch(observer, 78, ["internal-1"])
        scheduler = observer.create_scheduler(LifecycleScheduler)
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-1",
        )

        with (
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver._SLOW_ENGINE_STAGE_NS", 0
            ),
            mock.patch.object(observer, "_log_slow_engine_stage") as log_stage,
        ):
            scheduler.add_request(request)

        self.assertIn("internal-1", scheduler._scheduler.requests)
        self.assertEqual(1, log_stage.call_count)
        stage, started_ns, finished_ns = log_stage.call_args.args
        self.assertEqual("add_request", stage)
        self.assertLessEqual(started_ns, finished_ns)
        self.assertEqual("internal-1", log_stage.call_args.kwargs["request_id"])
        self.assertEqual("done", log_stage.call_args.kwargs["phase"])
        for key in (
            "admission_ms",
            "snapshot_ms",
            "native_add_ms",
            "ready_ms",
            "report_ms",
        ):
            self.assertGreaterEqual(log_stage.call_args.kwargs[key], 0.0)
        self.assertEqual(1, log_stage.call_args.kwargs["forwarded_requests"])
        self.assertEqual(0, log_stage.call_args.kwargs["running"])
        self.assertEqual(0, log_stage.call_args.kwargs["waiting"])

        with (
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver._SLOW_ENGINE_STAGE_NS", 0
            ),
            mock.patch.object(
                scheduler, "_add_request", side_effect=RuntimeError("admission failed")
            ),
            mock.patch.object(observer, "_log_slow_engine_stage") as log_stage,
        ):
            with self.assertRaisesRegex(RuntimeError, "admission failed"):
                scheduler.add_request(request)
        self.assertEqual(1, log_stage.call_count)
        self.assertEqual("add_request", log_stage.call_args.args[0])
        self.assertEqual("admission", log_stage.call_args.kwargs["phase"])
        self.assertEqual(-1.0, log_stage.call_args.kwargs["admission_ms"])

    def test_scheduler_add_request_ignores_diagnostic_failure(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=0,
            cached_tokens=0,
            request_id="diagnostic-failure",
        )
        with (
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver._SLOW_ENGINE_STAGE_NS", 0
            ),
            mock.patch.object(
                observer,
                "_log_slow_engine_stage",
                side_effect=RuntimeError("log failed"),
            ),
        ):
            scheduler.add_request(request)
            with mock.patch.object(
                scheduler, "_add_request", side_effect=ValueError("admission failed")
            ):
                with self.assertRaisesRegex(ValueError, "admission failed"):
                    scheduler.add_request(request)

    def test_explicit_local_fallback_skips_forward_metadata(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=0,
            cached_tokens=0,
            request_id="local-fallback",
        )
        request.navi_sched_local_fallback = True

        scheduler.add_request(request)

        self.assertIs(request, scheduler._scheduler.requests["local-fallback"])
        self.assertEqual([], observer._bridge.failed_responses)
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual({}, observer._response_request_ids)
        self.assertGreaterEqual(len(observer._bridge.workloads), 2)

    def test_unmarked_request_without_forward_metadata_is_rejected(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=0,
            cached_tokens=0,
            request_id="unowned-request",
        )

        with self.assertLogs("vllm.navi_sched.agent", level="ERROR"):
            scheduler.add_request(request)

        self.assertNotIn("unowned-request", scheduler._scheduler.requests)
        self.assertEqual(
            [
                (
                    stable_request_id("unowned-request"),
                    "vLLM request has no NaviSched forwarded batch metadata",
                )
            ],
            observer._bridge.failed_responses,
        )
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)

    def test_native_enqueue_records_cohort_order_before_scheduler_add(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        state = self.add_forward_batch(observer, 911, ["first", "second"])
        requests = {
            rid: self.make_request(
                prompt_tokens=64, computed_tokens=0, cached_tokens=-1, request_id=rid
            )
            for rid in ("first", "second")
        }
        recorded = []
        original_add = scheduler._scheduler.add_request

        def record_add(request):
            recorded.append(
                (
                    request.request_id,
                    getattr(request, "_navi_forward_enqueue_order", None),
                )
            )
            original_add(request)

        scheduler._scheduler.add_request = record_add
        scheduler.add_request(requests["second"])
        self.assertEqual([], recorded)
        self.assertEqual({}, scheduler._scheduler.requests)
        scheduler.add_request(requests["first"])
        self.assertEqual([("first", 0), ("second", 1)], recorded)
        self.assertEqual(["first", "second"], list(scheduler._scheduler.requests))
        self.assertTrue(state.admission_ready)
        next_state = self.add_forward_batch(observer, 912, ["third"])
        third = self.make_request(
            prompt_tokens=64, computed_tokens=0, cached_tokens=-1, request_id="third"
        )
        scheduler.add_request(third)
        self.assertEqual([("first", 0), ("second", 1), ("third", 2)], recorded)
        self.assertTrue(next_state.admission_ready)

    def test_partial_native_scheduler_admission_rolls_back_entire_batch(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        state = self.add_forward_batch(observer, 117, ["internal-1", "internal-2"])

        class FailingScheduler(LifecycleScheduler):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.add_count = 0

            def add_request(self, request):
                self.add_count += 1
                super().add_request(request)
                if self.add_count == 2:
                    raise RuntimeError("injected scheduler add failure")

        scheduler = observer.create_scheduler(FailingScheduler)
        first = self.make_request(
            prompt_tokens=16,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-1",
        )
        second = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-2",
        )
        observer._response_request_ids.update(state.external_request_ids)

        scheduler.add_request(first)
        with self.assertRaisesRegex(RuntimeError, "injected scheduler add failure"):
            scheduler.add_request(second)

        self.assertEqual({}, scheduler._scheduler.requests)
        self.assertEqual(
            [
                (
                    ["internal-1", "internal-2"],
                    RequestStatus.FINISHED_ABORTED,
                )
            ],
            scheduler._scheduler.finish_calls,
        )
        self.assertTrue(state.released)
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual({}, observer._forward_request_states)
        self.assertEqual({}, observer._response_request_ids)
        self.assertEqual([117], observer._bridge.released_forward_batches)
        self.assertEqual(
            {
                (
                    stable_request_id("internal-1"),
                    "vLLM scheduler admission failed",
                ),
                (
                    stable_request_id("internal-2"),
                    "vLLM scheduler admission failed",
                ),
            },
            set(observer._bridge.failed_responses),
        )

    def test_queued_batch_abort_releases_all_forward_state(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        self.add_forward_batch(observer, 91, ["internal-1", "internal-2"])
        scheduler = observer.create_scheduler(LifecycleScheduler)
        for request_id in ("internal-1", "internal-2"):
            scheduler.add_request(
                self.make_request(
                    prompt_tokens=32,
                    computed_tokens=0,
                    cached_tokens=0,
                    request_id=request_id,
                )
            )
        state = observer._forward_request_state("internal-1")

        aborted = scheduler.finish_requests(None, "aborted")

        self.assertEqual(
            [("internal-1", 1), ("internal-2", 1)],
            sorted(finished_request_pairs(aborted)),
        )
        self.assertTrue(state.released)
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual([91], observer._bridge.released_forward_batches)

    def test_queued_single_abort_preserves_remaining_request(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        self.add_forward_batch(observer, 92, ["internal-1", "internal-2"])
        scheduler = observer.create_scheduler(LifecycleScheduler)
        for request_id in ("internal-1", "internal-2"):
            scheduler.add_request(
                self.make_request(
                    prompt_tokens=32,
                    computed_tokens=0,
                    cached_tokens=0,
                    request_id=request_id,
                )
            )

        state = observer._forward_request_state("internal-1")
        aborted = scheduler.finish_requests("internal-1", "aborted")
        output = scheduler.schedule()
        self.run_execution_step(observer, output, scheduler.requests)

        self.assertEqual([("internal-1", 1)], finished_request_pairs(aborted))
        self.assertEqual({"internal-2"}, set(output.num_scheduled_tokens))
        self.assertTrue(state.released)
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual([92], observer._bridge.released_forward_batches)
        self.assertEqual(
            [
                (
                    92,
                    [stable_request_id("internal-1")],
                )
            ],
            observer._bridge.aborted_forward_requests,
        )
        self.assertEqual(
            [stable_request_id("internal-2")],
            [value[0] for value in observer._bridge.finished[-1][1]],
        )

    def test_partial_abort_after_last_prefill_step_finishes_remaining_batch(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        self.add_forward_batch(observer, 126, ["internal-1", "internal-2"])
        scheduler = observer.create_scheduler(LifecycleScheduler)
        requests = {
            request_id: self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=0,
                request_id=request_id,
            )
            for request_id in ("internal-1", "internal-2")
        }
        for request in requests.values():
            scheduler.add_request(request)
        state = observer._forward_request_state("internal-1")
        requests["internal-1"].num_computed_tokens = 32
        self.run_execution_step(
            observer,
            SimpleNamespace(num_scheduled_tokens={"internal-1": 32}),
            scheduler.requests,
        )

        self.assertFalse(state.released)
        aborted = scheduler.finish_requests("internal-2", "aborted")

        self.assertEqual([("internal-2", 1)], finished_request_pairs(aborted))
        self.assertTrue(state.released)
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual([126], observer._bridge.released_forward_batches)
        self.assertEqual(
            [stable_request_id("internal-1")],
            [value[0] for value in observer._bridge.finished[-1][1]],
        )

    def test_partial_batch_abort_waits_for_arrival_then_runs_remainder(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        self.add_forward_batch(observer, 93, ["internal-1", "internal-2"])
        scheduler = observer.create_scheduler(LifecycleScheduler)
        scheduler.add_request(
            self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=0,
                request_id="internal-1",
            )
        )

        aborted = scheduler.finish_requests("internal-1", "aborted")
        scheduler.add_request(
            self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=0,
                request_id="internal-2",
            )
        )
        output = scheduler.schedule()
        self.run_execution_step(observer, output, scheduler.requests)

        self.assertEqual([("internal-1", 1)], finished_request_pairs(aborted))
        self.assertEqual({"internal-2"}, set(output.num_scheduled_tokens))
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual([93], observer._bridge.released_forward_batches)
        self.assertEqual(
            [(93, [stable_request_id("internal-1")])],
            observer._bridge.aborted_forward_requests,
        )

    def test_partial_abort_bridge_failure_releases_tracking_without_leak(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        requests = [
            self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=0,
                request_id=request_id,
            )
            for request_id in ("internal-1", "internal-2")
        ]
        state = self.add_forward_batch(
            observer, 125, [request.request_id for request in requests]
        )
        observer._bridge.abort_forward_batch_requests = mock.Mock(return_value=False)
        for request in requests:
            scheduler.add_request(request)

        with self.assertLogs("vllm.navi_sched.agent", level="ERROR"):
            aborted = scheduler.finish_requests("internal-1", "aborted")

        self.assertEqual([("internal-1", 1)], finished_request_pairs(aborted))
        self.assertTrue(state.tracking_aborted)
        self.assertTrue(state.released)
        self.assertEqual([125], observer._bridge.released_forward_batches)
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual({"internal-2"}, set(scheduler._scheduler.requests))
        self.assertIs(state, observer._forward_request_state("internal-2"))

        self.assertEqual(
            [("internal-2", 1)],
            finished_request_pairs(scheduler.finish_requests("internal-2", "finished")),
        )
        scheduler.schedule()
        self.assertEqual({}, observer._forward_request_states)

    def test_abort_before_forward_add_is_applied_during_admission(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        requests = [
            self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=0,
                request_id=request_id,
            )
            for request_id in ("internal-1", "internal-2")
        ]
        native_batch = FakeForwardBatch(
            96, [request.request_id for request in requests]
        )
        request_type = SimpleNamespace(ADD="add", ABORT="abort")
        input_queue = SimpleNamespace(
            items=[], put_nowait=lambda item: input_queue.items.append(item)
        )
        aborts_queue = SimpleNamespace(
            items=[], put_nowait=lambda item: aborts_queue.items.append(item)
        )
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda request: (request, 0),
            input_queue=input_queue,
            aborts_queue=aborts_queue,
        )
        observer._bridge.cancelled_requests = [stable_request_id("internal-1")]
        observer._drain_cancelled_requests(request_type)

        observer._enqueue_forward_batch(
            native_batch,
            [(stable_request_id(request.request_id), request) for request in requests],
            request_type,
        )
        self.assertEqual([], aborts_queue.items)
        for queued_type, value in input_queue.items:
            self.assertEqual(request_type.ADD, queued_type)
            scheduler.add_request(value[0])
        output = scheduler.schedule()
        self.run_execution_step(observer, output, scheduler.requests)

        self.assertEqual({"internal-2"}, set(output.num_scheduled_tokens))
        self.assertEqual(set(), observer._cancelled_before_decode)
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual([96], observer._bridge.released_forward_batches)
        self.assertEqual(
            [(96, [stable_request_id("internal-1")])],
            observer._bridge.aborted_forward_requests,
        )

    def test_queued_add_after_full_cancel_consumes_tombstones(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        requests = [
            self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=0,
                request_id=request_id,
            )
            for request_id in ("internal-1", "internal-2")
        ]
        native_batch = FakeForwardBatch(
            118, [request.request_id for request in requests]
        )
        request_type = SimpleNamespace(ADD="add", ABORT="abort")
        input_queue = SimpleNamespace(
            items=[], put_nowait=lambda item: input_queue.items.append(item)
        )
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda request: (request, 0),
            input_queue=input_queue,
            aborts_queue=SimpleNamespace(put_nowait=lambda unused: None),
        )
        observer._enqueue_forward_batch(
            native_batch,
            [(stable_request_id(request.request_id), request) for request in requests],
            request_type,
        )
        observer._bridge.cancelled_requests = list(native_batch.request_ids)
        observer._drain_cancelled_requests(request_type)

        self.assertEqual([2, 0], [load[2] for load in observer._bridge.workloads[-2:]])
        self.assertEqual(0, observer._bridge.workloads[-1][7])
        self.assertTrue(
            observer._pending_forward_batches[
                native_batch.request_ids[0]
            ].tracking_aborted
        )
        self.assertEqual([118], observer._bridge.released_forward_batches)

        _, cancelled_ids = input_queue.items.pop()
        scheduler.finish_requests(cancelled_ids, "aborted")
        self.assertEqual(
            set(native_batch.request_ids), set(observer._pending_forward_batches)
        )
        for _, (request, _) in input_queue.items:
            scheduler.add_request(request)

        self.assertEqual({}, scheduler._scheduler.requests)
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual({}, observer._response_request_ids)
        self.assertEqual([118], observer._bridge.released_forward_batches)

    def test_queued_add_after_partial_cancel_runs_remaining_request(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        requests = [
            self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=0,
                request_id=request_id,
            )
            for request_id in ("internal-1", "internal-2")
        ]
        native_batch = FakeForwardBatch(
            119, [request.request_id for request in requests]
        )
        request_type = SimpleNamespace(ADD="add", ABORT="abort")
        input_queue = SimpleNamespace(
            items=[], put_nowait=lambda item: input_queue.items.append(item)
        )
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda request: (request, 0),
            input_queue=input_queue,
            aborts_queue=SimpleNamespace(put_nowait=lambda unused: None),
        )
        observer._enqueue_forward_batch(
            native_batch,
            [(stable_request_id(request.request_id), request) for request in requests],
            request_type,
        )
        cancelled_request_id = native_batch.request_ids[0]
        observer._bridge.cancelled_requests = [cancelled_request_id]
        observer._drain_cancelled_requests(request_type)

        self.assertEqual([2, 1], [load[2] for load in observer._bridge.workloads[-2:]])
        self.assertEqual(32, observer._bridge.workloads[-1][7])
        self.assertEqual(
            [(119, [cancelled_request_id])],
            observer._bridge.aborted_forward_requests,
        )

        _, cancelled_ids = input_queue.items.pop()
        scheduler.finish_requests(cancelled_ids, "aborted")
        for _, (request, _) in input_queue.items:
            scheduler.add_request(request)
        output = scheduler.schedule()
        self.run_execution_step(observer, output, scheduler.requests)

        self.assertEqual({"internal-2"}, set(scheduler._scheduler.requests))
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertNotIn(
            stable_request_id("internal-1"), observer._response_request_ids
        )
        self.assertIn(stable_request_id("internal-2"), observer._response_request_ids)
        self.assertEqual([119], observer._bridge.released_forward_batches)
        self.assertEqual(
            [(119, [stable_request_id("internal-1")])],
            observer._bridge.aborted_forward_requests,
        )

    def test_cancel_after_first_queued_add_runs_remaining_request(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        requests = [
            self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=0,
                request_id=request_id,
            )
            for request_id in ("internal-1", "internal-2")
        ]
        native_batch = FakeForwardBatch(
            121, [request.request_id for request in requests]
        )
        request_type = SimpleNamespace(ADD="add", ABORT="abort")
        input_queue = SimpleNamespace(
            items=[], put_nowait=lambda item: input_queue.items.append(item)
        )
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda request: (request, 0),
            input_queue=input_queue,
            aborts_queue=SimpleNamespace(put_nowait=lambda unused: None),
        )
        observer._enqueue_forward_batch(
            native_batch,
            [(stable_request_id(request.request_id), request) for request in requests],
            request_type,
        )

        first_add = input_queue.items.pop(0)
        scheduler.add_request(first_add[1][0])
        observer._bridge.cancelled_requests = [native_batch.request_ids[0]]
        observer._drain_cancelled_requests(request_type)
        _, cancelled_ids = input_queue.items.pop()
        scheduler.finish_requests(cancelled_ids, "aborted")
        second_add = input_queue.items.pop(0)
        scheduler.add_request(second_add[1][0])
        output = scheduler.schedule()
        self.run_execution_step(observer, output, scheduler.requests)

        self.assertEqual({"internal-2"}, set(scheduler._scheduler.requests))
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertNotIn(
            stable_request_id("internal-1"), observer._response_request_ids
        )
        self.assertIn(stable_request_id("internal-2"), observer._response_request_ids)
        self.assertEqual([121], observer._bridge.released_forward_batches)
        self.assertEqual(
            [(121, [stable_request_id("internal-1")])],
            observer._bridge.aborted_forward_requests,
        )

    def test_repeated_cancel_before_queued_add_releases_once(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-1",
        )
        native_batch = FakeForwardBatch(120, [request.request_id])
        request_type = SimpleNamespace(ADD="add", ABORT="abort")
        input_queue = SimpleNamespace(
            items=[], put_nowait=lambda item: input_queue.items.append(item)
        )
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda value: (value, 0),
            input_queue=input_queue,
            aborts_queue=SimpleNamespace(put_nowait=lambda unused: None),
        )
        observer._enqueue_forward_batch(
            native_batch,
            [(stable_request_id(request.request_id), request)],
            request_type,
        )
        for _ in range(2):
            observer._bridge.cancelled_requests = [native_batch.request_ids[0]]
            observer._drain_cancelled_requests(request_type)

        add_item = input_queue.items[0]
        for _, cancelled_ids in input_queue.items[1:]:
            scheduler.finish_requests(cancelled_ids, "aborted")
        scheduler.add_request(add_item[1][0])

        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._response_request_ids)
        self.assertEqual([120], observer._bridge.released_forward_batches)

    def test_cancel_delivery_failure_is_retried_without_losing_request(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request_id = stable_request_id("internal-1")
        observer._response_request_ids[request_id] = "internal-1"
        observer._bridge.cancelled_requests = [request_id]
        aborts_queue = mock.Mock()
        input_queue = mock.Mock()
        input_queue.put_nowait.side_effect = [RuntimeError("queue unavailable"), None]
        observer._engine_core = SimpleNamespace(
            aborts_queue=aborts_queue,
            input_queue=input_queue,
        )
        request_type = SimpleNamespace(ABORT="abort")

        with self.assertLogs("vllm.navi_sched.agent", level="ERROR"):
            self.assertFalse(observer._drain_cancelled_requests(request_type))

        self.assertEqual(["internal-1"], observer._pending_engine_abort_request_ids)
        self.assertIn(request_id, observer._inactive_response_request_ids)
        self.assertEqual([], observer._bridge.cancelled_requests)

        self.assertTrue(observer._drain_cancelled_requests(request_type))
        self.assertEqual([], observer._pending_engine_abort_request_ids)
        self.assertEqual(
            [mock.call(["internal-1"]), mock.call(["internal-1"])],
            aborts_queue.put_nowait.call_args_list,
        )
        self.assertEqual(
            [
                mock.call(("abort", ["internal-1"])),
                mock.call(("abort", ["internal-1"])),
            ],
            input_queue.put_nowait.call_args_list,
        )

    def test_duplicate_vllm_abort_does_not_cancel_reused_request_id(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-1",
        )
        request_type = SimpleNamespace(ADD="add", ABORT="abort")
        input_queue = SimpleNamespace(
            items=[], put_nowait=lambda item: input_queue.items.append(item)
        )
        aborts = []
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda value: (value, 0),
            input_queue=input_queue,
            aborts_queue=SimpleNamespace(put_nowait=aborts.append),
        )
        first_batch = FakeForwardBatch(122, [request.request_id])
        observer._enqueue_forward_batch(
            first_batch,
            [(stable_request_id(request.request_id), request)],
            request_type,
        )
        observer._bridge.cancelled_requests = list(first_batch.request_ids)
        observer._drain_cancelled_requests(request_type)

        self.assertEqual([], scheduler.finish_requests(aborts[0], "aborted"))
        add_item, abort_item = input_queue.items
        scheduler.add_request(add_item[1][0])
        self.assertEqual([], scheduler.finish_requests(abort_item[1], "aborted"))

        replacement = self.make_request(
            prompt_tokens=16,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-1",
        )
        second_batch = FakeForwardBatch(123, [replacement.request_id])
        input_queue.items.clear()
        observer._enqueue_forward_batch(
            second_batch,
            [(stable_request_id(replacement.request_id), replacement)],
            request_type,
        )
        scheduler.add_request(input_queue.items[0][1][0])

        self.assertEqual({"internal-1"}, set(scheduler._scheduler.requests))
        output = scheduler.schedule()
        self.run_execution_step(observer, output, scheduler.requests)
        self.assertEqual([122, 123], observer._bridge.released_forward_batches)

    def test_started_batch_abort_releases_native_handle_and_state(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        self.add_forward_batch(observer, 94, ["internal-1", "internal-2"])
        scheduler = observer.create_scheduler(LifecycleScheduler, chunk_size=16)
        for request_id in ("internal-1", "internal-2"):
            scheduler.add_request(
                self.make_request(
                    prompt_tokens=32,
                    computed_tokens=0,
                    cached_tokens=0,
                    request_id=request_id,
                )
            )
        output = scheduler.schedule()
        state = observer._forward_request_state("internal-1")
        self.run_execution_step(observer, output, scheduler.requests)
        self.assertIsNotNone(state.handle)

        aborted = scheduler.finish_requests(None, "aborted")

        self.assertEqual(
            [("internal-1", 1), ("internal-2", 1)],
            sorted(finished_request_pairs(aborted)),
        )
        self.assertTrue(state.released)
        self.assertIsNone(state.handle)
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual([94], observer._bridge.released_forward_batches)

    def test_released_batch_keeps_preemption_ownership_until_native_completion(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=32,
            cached_tokens=32,
            request_id="old-request",
        )
        self.add_forward_batch(observer, 106, [request.request_id])
        state = observer._admit_request(request)
        self.assertEqual(106, request._navi_forward_batch_id)
        self.assertEqual(106, request._navi_plan_trace_batch_id)
        self.assertIn(request._navi_plan_trace_request_id, state.request_ids)
        observer._release_forward_batch(state)

        self.assertIs(state, observer._forward_request_state(request.request_id))
        observer._cleanup_finished_forward_requests({request.request_id: request})
        self.assertIs(state, observer._forward_request_state(request.request_id))

        observer._cleanup_finished_forward_requests({})
        self.assertIsNone(observer._forward_request_state(request.request_id))

    def test_incomplete_forward_batch_finish_releases_all_state(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=32,
            cached_tokens=0,
            request_id="internal-1",
        )
        self.add_forward_batch(observer, 107, [request.request_id])
        state = observer._admit_request(request)

        with self.assertRaisesRegex(RuntimeError, "incomplete NaviSched"):
            observer._finish_batch(state)

        self.assertTrue(state.released)
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual([107], observer._bridge.released_forward_batches)

    def test_concurrent_forward_batch_release_claims_reporter_once(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-1",
        )
        self.add_forward_batch(observer, 108, [request.request_id])
        state = observer._admit_request(request)
        first_release_entered = threading.Event()
        release_may_finish = threading.Event()
        release_count = 0
        release_count_lock = threading.Lock()

        def block_native_release(unused_native_batch):
            nonlocal release_count
            with release_count_lock:
                release_count += 1
            first_release_entered.set()
            self.assertTrue(release_may_finish.wait(2.0))

        observer._release_native_forward_batch = block_native_release
        first = threading.Thread(target=observer._release_forward_batch, args=(state,))
        second = threading.Thread(target=observer._release_forward_batch, args=(state,))
        first.start()
        self.assertTrue(first_release_entered.wait(2.0))
        second.start()
        second.join(2.0)
        release_may_finish.set()
        first.join(2.0)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(1, release_count)
        self.assertTrue(state.released)

    def test_invalid_forward_metadata_releases_native_batch(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        state = self.add_forward_batch(observer, 108, ["internal-1"])
        state.external_request_ids[stable_request_id("internal-1")] = (
            "different-request"
        )
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-1",
        )

        with self.assertRaisesRegex(ValueError, "does not match NaviSched payload"):
            observer._admit_request(request)

        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual([108], observer._bridge.released_forward_batches)

    def test_completion_error_releases_every_batch_in_execution_step(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        states = []
        progresses = []
        for batch_id, request_name in ((109, "internal-1"), (110, "internal-2")):
            request = self.make_request(
                prompt_tokens=32,
                computed_tokens=32,
                cached_tokens=0,
                request_id=request_name,
            )
            self.add_forward_batch(observer, batch_id, [request_name])
            state = observer._admit_request(request)
            state.pending_prefill_steps = 1
            states.append(state)
            progresses.append(
                _BatchProgress(
                    None,
                    state,
                    completed_request_ids={stable_request_id(request_name)},
                )
            )
        execution_step = _ExecutionStep(started_ns=0, batch_progress=progresses)
        future = Future()
        future.set_result("model-output")

        with self.assertLogs("vllm.navi_sched.agent", level="ERROR"):
            observer._complete_backend_execution(execution_step, future)

        self.assertTrue(all(state.released for state in states))
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual([109, 110], observer._bridge.released_forward_batches)

    def test_image_response_cache_counts_use_frontend_token_namespace(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._response_request_ids[7] = "image"
        # Two native image spans replace one token each in the frontend prompt.
        observer._response_image_positions = {7: ((4, 8), (16, 4))}
        encoded = []
        observer._response_encoder = SimpleNamespace(
            encode=lambda outputs: encoded.append(outputs) or [b"projected"]
        )
        for cached, external, expected_cached, expected_external in (
            (0, 0, 0, 0),
            (4, 0, 4, 0),
            (8, 4, 4, 0),
            (12, 8, 5, 1),
            (17, 5, 9, 4),
            (20, 8, 10, 5),
            (25, 5, 15, 5),
        ):
            value = SimpleNamespace(
                request_id="image",
                finished=False,
                num_cached_tokens=cached,
                num_external_computed_tokens=external,
            )
            outputs = SimpleNamespace(outputs=[value], finished_requests=set())
            with self.subTest(cached=cached, external=external):
                self.assertTrue(observer.route_engine_outputs(7, outputs))
                projected = encoded[-1].outputs[0]
                self.assertEqual(expected_cached, projected.num_cached_tokens)
                self.assertEqual(
                    expected_external, projected.num_external_computed_tokens
                )
                self.assertLessEqual(projected.num_cached_tokens, 16)
                self.assertLessEqual(
                    projected.num_external_computed_tokens, projected.num_cached_tokens
                )
                self.assertEqual(cached, value.num_cached_tokens)
                self.assertEqual(external, value.num_external_computed_tokens)
                self.assertIsNot(outputs, encoded[-1])
        terminal = SimpleNamespace(outputs=[], finished_requests={"image"})
        self.assertTrue(observer.route_engine_outputs(7, terminal))
        self.assertEqual({}, observer._response_image_positions)

    def test_streaming_outputs_use_response_pipe_until_terminal(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._response_encoder = SimpleNamespace(
            encode=lambda outputs: [outputs.frame]
        )
        observer._response_request_ids[7] = "internal-1"
        first = SimpleNamespace(
            frame=b"first",
            outputs=[SimpleNamespace(request_id="internal-1", finished=False)],
            finished_requests=set(),
        )
        last = SimpleNamespace(
            frame=b"last",
            outputs=[SimpleNamespace(request_id="internal-1", finished=True)],
            finished_requests={"internal-1"},
        )

        self.assertTrue(observer.route_engine_outputs(7, first))
        self.assertEqual({7: "internal-1"}, observer._response_request_ids)
        self.assertTrue(observer.route_engine_outputs(7, last))

        self.assertEqual(
            [(7, [b"first"], False), (7, [b"last"], True)],
            observer._bridge.published_responses,
        )
        self.assertEqual({}, observer._response_request_ids)
        self.assertEqual(set(), observer._inactive_response_request_ids)
        self.assertEqual({}, observer._response_image_positions)

    def test_remote_scheduler_stats_stay_on_execution_node(self):
        from vllm.v1.engine import EngineCoreOutputs

        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._response_encoder = SimpleNamespace(
            encode=lambda outputs: [
                b"with-stats" if outputs.scheduler_stats is not None else b"no-stats"
            ]
        )
        observer._response_request_ids[7] = "internal-1"
        scheduler_stats = object()
        outputs = EngineCoreOutputs(
            outputs=[SimpleNamespace(request_id="internal-1", finished=True)],
            finished_requests={"internal-1"},
            scheduler_stats=scheduler_stats,
            timestamp=123.0,
        )
        local_outputs = []
        output_queue = _RoutingOutputQueue(
            SimpleNamespace(put_nowait=local_outputs.append), observer
        )

        output_queue.put_nowait((7, outputs))

        self.assertEqual(
            [(7, [b"no-stats"], True)], observer._bridge.published_responses
        )
        self.assertIsNone(outputs.scheduler_stats)
        self.assertEqual(1, len(local_outputs))
        local_client_index, local_stats = local_outputs[0]
        self.assertEqual(0, local_client_index)
        self.assertIs(scheduler_stats, local_stats.scheduler_stats)
        self.assertEqual(123.0, local_stats.timestamp)

    def test_local_scheduler_stats_are_not_changed(self):
        from vllm.v1.engine import EngineCoreOutputs

        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler_stats = object()
        outputs = EngineCoreOutputs(scheduler_stats=scheduler_stats, timestamp=456.0)
        local_outputs = []
        output_queue = _RoutingOutputQueue(
            SimpleNamespace(put_nowait=local_outputs.append), observer
        )

        output_queue.put_nowait((0, outputs))

        self.assertEqual([(0, outputs)], local_outputs)
        self.assertIs(scheduler_stats, outputs.scheduler_stats)

    def test_concurrent_response_pipes_do_not_mix_requests(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._response_encoder = SimpleNamespace(
            encode=lambda outputs: [outputs.frame]
        )
        observer._response_request_ids.update({7: "first", 8: "second"})

        for request_id, external_id, frame in (
            (8, "second", b"second-output"),
            (7, "first", b"first-output"),
        ):
            outputs = SimpleNamespace(
                frame=frame,
                outputs=[SimpleNamespace(request_id=external_id, finished=True)],
                finished_requests={external_id},
            )
            self.assertTrue(observer.route_engine_outputs(request_id, outputs))

        self.assertEqual(
            [(8, [b"second-output"], True), (7, [b"first-output"], True)],
            observer._bridge.published_responses,
        )
        self.assertEqual({}, observer._response_request_ids)

    def test_mixed_response_is_rejected_and_native_request_is_aborted(self):
        from vllm.v1.engine import EngineCoreRequestType

        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._response_encoder = SimpleNamespace(encode=lambda unused: [b"bad"])
        observer._response_request_ids[7] = "expected"
        aborts = []
        inputs = []
        observer._engine_core = SimpleNamespace(
            aborts_queue=SimpleNamespace(put_nowait=aborts.append),
            input_queue=SimpleNamespace(put_nowait=inputs.append),
        )
        outputs = SimpleNamespace(
            outputs=[SimpleNamespace(request_id="another", finished=False)],
            finished_requests=set(),
        )

        with self.assertLogs("vllm.navi_sched.agent", level="ERROR"):
            self.assertTrue(observer.route_engine_outputs(7, outputs))

        self.assertEqual([], observer._bridge.published_responses)
        self.assertEqual([["expected"]], aborts)
        self.assertEqual([(EngineCoreRequestType.ABORT, ["expected"])], inputs)
        self.assertEqual({7}, observer._inactive_response_request_ids)

    def test_response_pipe_failure_only_aborts_its_request(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._response_encoder = SimpleNamespace(encode=lambda unused: [b"output"])
        observer._response_request_ids.update({7: "failed", 8: "healthy"})
        observer._bridge.publish_forward_response = mock.Mock(side_effect=[False, True])
        aborts = []
        observer._engine_core = SimpleNamespace(
            aborts_queue=SimpleNamespace(put_nowait=aborts.append),
            input_queue=SimpleNamespace(put_nowait=lambda unused: None),
        )
        failed = SimpleNamespace(
            outputs=[SimpleNamespace(request_id="failed", finished=False)],
            finished_requests=set(),
        )
        healthy = SimpleNamespace(
            outputs=[SimpleNamespace(request_id="healthy", finished=True)],
            finished_requests={"healthy"},
        )

        with self.assertLogs("vllm.navi_sched.agent", level="WARNING"):
            self.assertTrue(observer.route_engine_outputs(7, failed))
        self.assertTrue(observer.route_engine_outputs(8, healthy))

        self.assertEqual([["failed"]], aborts)
        self.assertEqual({7}, observer._inactive_response_request_ids)
        self.assertEqual({7: "failed"}, observer._response_request_ids)

    def test_response_pipe_failure_defers_failed_engine_abort(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._response_encoder = SimpleNamespace(encode=lambda unused: [b"output"])
        observer._response_request_ids[7] = "failed"
        observer._bridge.publish_forward_response = mock.Mock(return_value=False)
        aborts_queue = mock.Mock()
        input_queue = mock.Mock()
        input_queue.put_nowait.side_effect = [RuntimeError("queue unavailable"), None]
        observer._engine_core = SimpleNamespace(
            aborts_queue=aborts_queue,
            input_queue=input_queue,
        )
        outputs = SimpleNamespace(
            outputs=[SimpleNamespace(request_id="failed", finished=False)],
            finished_requests=set(),
        )

        with self.assertLogs("vllm.navi_sched.agent", level="ERROR"):
            self.assertTrue(observer.route_engine_outputs(7, outputs))

        self.assertEqual(["failed"], observer._pending_engine_abort_request_ids)
        self.assertTrue(
            observer._drain_cancelled_requests(SimpleNamespace(ABORT="abort"))
        )
        self.assertEqual([], observer._pending_engine_abort_request_ids)
        self.assertEqual(
            [mock.call(["failed"]), mock.call(["failed"])],
            aborts_queue.put_nowait.call_args_list,
        )

    def test_mixed_response_defers_failed_engine_abort_without_escaping(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._response_request_ids[7] = "expected"
        observer._engine_core = SimpleNamespace(
            aborts_queue=SimpleNamespace(put_nowait=lambda unused: None),
            input_queue=SimpleNamespace(
                put_nowait=mock.Mock(side_effect=RuntimeError("queue unavailable"))
            ),
        )
        outputs = SimpleNamespace(
            outputs=[SimpleNamespace(request_id="unexpected", finished=False)],
            finished_requests=set(),
        )

        with self.assertLogs("vllm.navi_sched.agent", level="ERROR"):
            self.assertTrue(observer.route_engine_outputs(7, outputs))

        self.assertEqual(["expected"], observer._pending_engine_abort_request_ids)
        self.assertEqual({7}, observer._inactive_response_request_ids)

    def test_shutdown_fails_pending_response_and_releases_batch(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        state = self.add_forward_batch(observer, 111, ["internal-1"])
        request_id = stable_request_id("internal-1")
        observer._response_request_ids[request_id] = "internal-1"

        observer.shutdown()

        self.assertEqual(
            [(request_id, "NaviSched vLLM Agent stopped")],
            observer._bridge.failed_responses,
        )
        self.assertTrue(state.released)
        self.assertEqual([111], observer._bridge.released_forward_batches)
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual({}, observer._response_request_ids)

    def test_stop_during_native_queue_wait_fails_batch_without_enqueue(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-1",
        )
        native_batch = FakeForwardBatch(112, [request.request_id])
        observer._bridge.queued_forward_batches.append(native_batch)
        observer._bridge.take_next_hook = observer._forward_stop.set
        input_queue = SimpleNamespace(put_nowait=mock.Mock())
        observer._engine_core = SimpleNamespace(engine_index=3, input_queue=input_queue)
        engine_module = sys.modules["vllm.v1.engine"]
        serial_utils = types.ModuleType("vllm.v1.serial_utils")
        serial_utils.MsgpackDecoder = lambda unused: mock.sentinel.decoder
        msgspec = types.ModuleType("msgspec")
        msgspec.msgpack = mock.sentinel.msgpack

        with (
            mock.patch.object(
                engine_module, "EngineCoreRequest", object(), create=True
            ),
            mock.patch.dict(
                sys.modules,
                {"msgspec": msgspec, "vllm.v1.serial_utils": serial_utils},
            ),
            mock.patch.object(
                observer,
                "_decode_forward_batch",
                return_value=[(stable_request_id(request.request_id), request)],
            ),
        ):
            observer._forward_batches()

        input_queue.put_nowait.assert_not_called()
        self.assertEqual(
            [
                (
                    stable_request_id(request.request_id),
                    "NaviSched vLLM Agent is stopping",
                )
            ],
            observer._bridge.failed_responses,
        )
        self.assertEqual([112], observer._bridge.released_forward_batches)
        self.assertEqual([], observer._bridge.queued_forward_batches)

    def test_fully_cancelled_batch_does_not_block_next_native_batch(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._native_ingress = True
        cancelled_batch = FakeForwardBatch(130, ["cancelled-request"])
        next_batch = FakeForwardBatch(131, ["next-request"])
        observer._cancelled_before_decode.update(cancelled_batch.request_ids)
        observer._bridge.queued_forward_batches.extend([cancelled_batch, next_batch])
        engine_module = sys.modules["vllm.v1.engine"]
        serial_utils = types.ModuleType("vllm.v1.serial_utils")
        serial_utils.MsgpackDecoder = lambda unused: mock.sentinel.decoder
        msgspec = types.ModuleType("msgspec")
        msgspec.msgpack = mock.sentinel.msgpack

        def enqueue_then_stop(*unused_args, **unused_kwargs):
            observer._forward_stop.set()

        with (
            mock.patch.object(
                engine_module, "EngineCoreRequest", object(), create=True
            ),
            mock.patch.dict(
                sys.modules,
                {"msgspec": msgspec, "vllm.v1.serial_utils": serial_utils},
            ),
            mock.patch.object(
                observer, "_decode_forward_batch", return_value=[]
            ) as decode,
            mock.patch.object(
                observer, "_enqueue_forward_batch", side_effect=enqueue_then_stop
            ) as enqueue,
            mock.patch.object(observer._forward_stop, "wait", return_value=True),
        ):
            observer._forward_batches()

        self.assertEqual([130], observer._bridge.released_forward_batches)
        self.assertEqual(set(), observer._cancelled_before_decode)
        decode.assert_called_once()
        self.assertIs(next_batch, decode.call_args.args[0])
        enqueue.assert_called_once()
        self.assertIs(next_batch, enqueue.call_args.args[0])
        self.assertEqual([], observer._bridge.queued_forward_batches)

    def test_forward_worker_prioritizes_ready_batch_over_cache_touch(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-1",
        )
        native_batch = FakeForwardBatch(113, [request.request_id])
        observer._bridge.queued_forward_batches.append(native_batch)
        observer._bridge.queued_forward_request_touches.append(
            FakeForwardRequestTouch(native_batch.request_ids[0], b"payload")
        )
        engine_module = sys.modules["vllm.v1.engine"]
        serial_utils = types.ModuleType("vllm.v1.serial_utils")
        serial_utils.MsgpackDecoder = lambda unused: mock.sentinel.decoder
        msgspec = types.ModuleType("msgspec")
        msgspec.msgpack = mock.sentinel.msgpack

        def enqueue_then_stop(*unused_args, **unused_kwargs):
            observer._forward_stop.set()

        with mock.patch.object(
            engine_module, "EngineCoreRequest", object(), create=True
        ):
            with mock.patch.dict(
                sys.modules,
                {"msgspec": msgspec, "vllm.v1.serial_utils": serial_utils},
            ):
                with mock.patch.object(
                    observer,
                    "_decode_forward_batch",
                    return_value=[(stable_request_id(request.request_id), request)],
                ):
                    with mock.patch.object(
                        observer,
                        "_enqueue_forward_batch",
                        side_effect=enqueue_then_stop,
                    ) as enqueue:
                        with mock.patch.object(
                            observer,
                            "_drain_forward_request_touches",
                            wraps=observer._drain_forward_request_touches,
                        ) as drain_touches:
                            observer._forward_batches()

        enqueue.assert_called_once()
        drain_touches.assert_not_called()
        self.assertEqual(
            [native_batch.request_ids[0]],
            [
                touch.request_id
                for touch in observer._bridge.queued_forward_request_touches
            ],
        )

    def test_slow_native_download_does_not_block_later_forward_batch(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        slow_batch = FakeForwardBatch(114, ["slow-image"])
        fast_batch = FakeForwardBatch(115, ["fast-image"])
        slow_id = slow_batch.request_ids[0]
        fast_id = fast_batch.request_ids[0]
        slow_started = threading.Event()
        release_slow = threading.Event()
        fast_enqueued = threading.Event()
        worker_errors = []
        enqueued = []
        lifecycle = []

        class Continuation:
            async def resume_async(self, requests, **kwargs):
                if requests[0].request_id == str(slow_id):
                    slow_started.set()
                    try:
                        while not release_slow.is_set():
                            await asyncio.sleep(0.001)
                    finally:
                        lifecycle.append("download_stopped")
                return requests

            def resume_sync(self, requests, **kwargs):
                return asyncio.run(self.resume_async(requests, **kwargs))

            async def aclose(self):
                lifecycle.append("session_closed")

        observer._render_continuation = Continuation()
        observer._engine_core = SimpleNamespace()
        observer._bridge.queued_forward_batches.extend([slow_batch, fast_batch])
        engine_module = sys.modules["vllm.v1.engine"]
        serial_utils = types.ModuleType("vllm.v1.serial_utils")
        serial_utils.MsgpackDecoder = lambda unused: mock.sentinel.decoder
        msgspec = types.ModuleType("msgspec")
        msgspec.msgpack = mock.sentinel.msgpack

        def enqueue(native_batch, *args, **kwargs):
            enqueued.append(native_batch.batch_id)
            if native_batch.batch_id == fast_batch.batch_id:
                fast_enqueued.set()
                observer._forward_stop.set()

        def worker_main():
            try:
                observer._forward_batches()
            except BaseException as error:
                worker_errors.append(error)

        release_batch = observer._release_native_forward_batch

        def release(native_batch):
            lifecycle.append("batch_released")
            release_batch(native_batch)

        with (
            mock.patch.dict(os.environ, {"VLLM_NAVI_SCHED_DEFER_RENDER": "1"}),
            mock.patch.object(
                engine_module, "EngineCoreRequest", object(), create=True
            ),
            mock.patch.dict(
                sys.modules,
                {
                    "msgspec": msgspec,
                    "vllm.v1.serial_utils": serial_utils,
                },
            ),
            mock.patch.object(
                observer,
                "_decode_forward_request",
                side_effect=(
                    lambda request_id, *args: SimpleNamespace(
                        request_id=str(request_id),
                        navi_sched_render_call=b"original-call",
                    )
                ),
            ),
            mock.patch.object(observer, "_enqueue_forward_batch", side_effect=enqueue),
            mock.patch.object(
                observer, "_release_native_forward_batch", side_effect=release
            ),
        ):
            worker = threading.Thread(target=worker_main, daemon=True)
            worker.start()
            try:
                self.assertTrue(slow_started.wait(2))
                independent = fast_enqueued.wait(1)
            finally:
                release_slow.set()
                observer._forward_stop.set()
                worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual([], worker_errors)
        self.assertTrue(independent, "slow image blocked an independent batch")
        self.assertEqual([fast_batch.batch_id], enqueued)
        self.assertEqual(
            [slow_batch.batch_id], observer._bridge.released_forward_batches
        )
        self.assertEqual(
            ["download_stopped", "batch_released", "session_closed"], lifecycle
        )
        self.assertEqual(
            [(slow_id, "NaviSched vLLM Agent is stopping")],
            observer._bridge.failed_responses,
        )

    def test_async_batch_timing_correlates_success_failure_and_cancel(self):
        for outcome in ("enqueued", "failed", "stopping"):
            with self.subTest(outcome=outcome):
                observer = VLLMEngineObserver(
                    self.make_config(), self.make_kv_cache_config()
                )
                observer._engine_core = SimpleNamespace()
                batch = FakeForwardBatch(119, ["image"], schedule_batch_id=987654321)
                batch.ready_time_ns = 70_000_000
                clock_ns = [100_000_000]
                request = SimpleNamespace(
                    request_id=str(batch.request_ids[0]), navi_sched_render_call=b"call"
                )

                def decode(*args, **kwargs):
                    clock_ns[0] += 10_000_000
                    return [(batch.request_ids[0], request)]

                async def render(*args, **kwargs):
                    clock_ns[0] += 200_000_000
                    if outcome == "failed":
                        raise ValueError("native renderer failed")
                    if outcome == "stopping":
                        raise asyncio.CancelledError()
                    return [request]

                def enqueue(*args, **kwargs):
                    self.assertEqual(200.0, args[5])
                    clock_ns[0] += 5_000_000

                observer._render_continuation = SimpleNamespace(resume_async=render)
                with (
                    mock.patch.object(
                        observer, "_decode_forward_batch", side_effect=decode
                    ),
                    mock.patch.object(
                        observer, "_enqueue_forward_batch", side_effect=enqueue
                    ),
                    mock.patch("time.perf_counter_ns", side_effect=lambda: clock_ns[0]),
                    mock.patch("time.monotonic_ns", side_effect=lambda: clock_ns[0]),
                    self.assertLogs("vllm.navi_sched.agent", level="INFO") as logs,
                ):
                    coroutine = observer._prepare_rendered_batch(
                        batch,
                        mock.sentinel.decoder,
                        mock.sentinel.msgpack,
                        mock.sentinel.request_type,
                    )
                    if outcome == "stopping":
                        with self.assertRaises(asyncio.CancelledError):
                            asyncio.run(coroutine)
                    else:
                        asyncio.run(coroutine)
                timing = [
                    line
                    for line in logs.output
                    if "NAVI_SCHED_VLLM_AGENT_BATCH" in line
                ]
                self.assertEqual(1, len(timing))
                self.assertIn(
                    "schedule_batch_id=987654321 original_size=1 status=" + outcome,
                    timing[0],
                )
                self.assertIn(
                    "native_ready_to_prepare_ms=30.000 decode_ms=10.000", timing[0]
                )
                if outcome == "enqueued":
                    self.assertIn("prepare_ms=210.000 enqueue_ms=5.000", timing[0])
                    self.assertIn("render_ms=200.000 total_ms=215.000", timing[0])
                    self.assertEqual([], observer._bridge.released_forward_batches)
                else:
                    self.assertIn("phase=render phase_ms=200.000", timing[0])
                    self.assertIn("prepare_ms=-1.000 enqueue_ms=-1.000", timing[0])
                    self.assertIn("render_ms=-1.000 total_ms=210.000", timing[0])
                    self.assertEqual([119], observer._bridge.released_forward_batches)

    def test_finish_includes_native_preparation_once_without_extending_total(self):
        for preparation_ms in (0.0, 200.0):
            with self.subTest(preparation_ms=preparation_ms):
                observer = VLLMEngineObserver(
                    self.make_config(), self.make_kv_cache_config()
                )
                request = self.make_request(
                    prompt_tokens=32,
                    computed_tokens=32,
                    cached_tokens=0,
                    request_id="prepared-image",
                )
                self.add_forward_batch(observer, 121, [request.request_id])
                state = observer._admit_request(request)
                state.handle = FakeHandle(observer._bridge.deleted)
                state.initial_cache_hits = {state.request_ids[0]: 0}
                state.native_preparation_ms = preparation_ms
                state.admission_started_ns = 700_000_000
                state.execution_started_ns = 1_000_000_000
                state.execution_step_ms = 40.0
                observer._finish_batch(state, completed_ns=1_050_000_000)

                self.assertEqual(1, len(observer._bridge.finished))
                breakdown = observer._bridge.backend_batch_executions[0]
                self.assertEqual(50.0, breakdown[0])
                self.assertEqual(40.0, breakdown[1])
                self.assertEqual(350.0, breakdown[6])
                self.assertEqual(300.0, breakdown[23])
                self.assertEqual(25, len(breakdown))
                self.assertEqual(preparation_ms, breakdown[24])
                observer._finish_batch(state, completed_ns=2_000_000_000)
                self.assertEqual(1, len(observer._bridge.finished))

    def test_deferred_images_are_prepared_on_api_processes(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._engine_core = SimpleNamespace(
            get_supported_tasks=mock.Mock(
                side_effect=AssertionError("EngineCore must not render images")
            )
        )
        batch = FakeForwardBatch(120, ["image"], schedule_batch_id=123456789)
        request = SimpleNamespace(
            request_id=str(batch.request_ids[0]), navi_sched_render_call=b"call"
        )
        client = SimpleNamespace(resume_async=mock.AsyncMock(return_value=[request]))
        with (
            mock.patch.object(
                observer,
                "_decode_forward_batch",
                return_value=[(batch.request_ids[0], request)],
            ),
            mock.patch.object(observer, "_enqueue_forward_batch") as enqueue,
            mock.patch(
                "navi_sched.adapters.vllm.VLLMNativePreparation.NativePreparationClient",
                return_value=client,
            ) as make_client,
        ):
            asyncio.run(
                observer._prepare_rendered_batch(
                    batch,
                    mock.sentinel.decoder,
                    mock.sentinel.msgpack,
                    mock.sentinel.request_type,
                )
            )
        make_client.assert_called_once_with(observer._native_prepare_port)
        client.resume_async.assert_awaited_once_with(
            [request],
            return_exceptions=True,
            schedule_batch_id=123456789,
            local_batch_id=120,
        )
        enqueue.assert_called_once()
        self.assertEqual([(batch.request_ids[0], request)], enqueue.call_args.args[1])
        self.assertEqual([], observer._bridge.released_forward_batches)

    def test_cache_touch_does_not_download_deferred_media(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=mock.Mock(return_value=(mock.sentinel.request, 0))
        )
        observer._bridge.queued_forward_request_touches.append(
            FakeForwardRequestTouch(123, b"original-call")
        )
        request = SimpleNamespace(navi_sched_render_call=b"original-call")
        with (
            mock.patch.object(
                observer, "_decode_forward_request", return_value=request
            ),
            mock.patch.object(observer, "_resume_rendered_requests") as resume,
        ):
            observer._drain_forward_request_touches(
                mock.sentinel.decoder, mock.sentinel.msgpack
            )
        resume.assert_not_called()
        observer._engine_core.preprocess_add_request.assert_not_called()
        self.assertEqual({}, observer._preprocessed_cache_touches)
        self.assertEqual([], list(observer._cache_touch_queue))

    def test_async_native_render_failure_preserves_other_request(self):
        for fail_all in (False, True):
            with self.subTest(fail_all=fail_all):
                observer = VLLMEngineObserver(
                    self.make_config(), self.make_kv_cache_config()
                )
                names = ["bad-image"] if fail_all else ["bad-image", "text"]
                batch = FakeForwardBatch(116, names)
                bad_id = batch.request_ids[0]
                requests = {
                    request_id: SimpleNamespace(
                        request_id=str(request_id),
                        navi_sched_render_call=b"original-call" if index == 0 else None,
                    )
                    for index, request_id in enumerate(batch.request_ids)
                }
                observer._engine_core = SimpleNamespace()
                observer._render_continuation = SimpleNamespace(
                    resume_async=mock.AsyncMock(
                        return_value=[ValueError("missing image")]
                    )
                )
                with (
                    mock.patch.object(
                        observer,
                        "_decode_forward_request",
                        side_effect=lambda request_id, *args: requests[request_id],
                    ),
                    mock.patch.object(observer, "_enqueue_forward_batch") as enqueue,
                    self.assertLogs("vllm.navi_sched.agent", level="ERROR"),
                ):
                    asyncio.run(
                        observer._prepare_rendered_batch(
                            batch,
                            mock.sentinel.decoder,
                            mock.sentinel.msgpack,
                            mock.sentinel.request_type,
                        )
                    )
                if fail_all:
                    enqueue.assert_not_called()
                    self.assertEqual([116], observer._bridge.released_forward_batches)
                    self.assertEqual(
                        [
                            (
                                bad_id,
                                "navi_sched_media_error:vLLM media preprocessing failed",
                            )
                        ],
                        observer._bridge.failed_responses,
                    )
                else:
                    self.assertEqual([], observer._bridge.released_forward_batches)
                    self.assertEqual(
                        [(116, [bad_id])], observer._bridge.aborted_forward_requests
                    )
                    self.assertEqual(
                        [
                            (
                                bad_id,
                                "navi_sched_media_error:vLLM media preprocessing failed",
                            )
                        ],
                        observer._bridge.failed_responses,
                    )
                    enqueue.assert_called_once()
                    self.assertEqual(
                        [(batch.request_ids[0], requests[batch.request_ids[0]])],
                        enqueue.call_args.args[1],
                    )

    def test_cancelled_render_failure_leaves_no_pending_cancel(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._engine_core = SimpleNamespace()
        batch = FakeForwardBatch(117, ["cancelled-image", "text"])
        cancelled_id, text_id = batch.request_ids
        requests = {
            request_id: SimpleNamespace(
                request_id=str(request_id),
                navi_sched_render_call=(
                    b"original-call" if request_id == cancelled_id else None
                ),
            )
            for request_id in batch.request_ids
        }

        async def cancelled_download(requests, **kwargs):
            # Cancellation can arrive while the native image download is pending.
            await asyncio.sleep(0)
            with observer._response_lock:
                observer._cancelled_before_decode.add(cancelled_id)
            return [ValueError("image download failed")]

        observer._render_continuation = SimpleNamespace(resume_async=cancelled_download)
        with (
            mock.patch.object(
                observer,
                "_decode_forward_request",
                side_effect=lambda request_id, *args: requests[request_id],
            ),
            mock.patch.object(observer, "_enqueue_forward_batch") as enqueue,
            self.assertLogs("vllm.navi_sched.agent", level="ERROR"),
        ):
            asyncio.run(
                observer._prepare_rendered_batch(
                    batch,
                    mock.sentinel.decoder,
                    mock.sentinel.msgpack,
                    mock.sentinel.request_type,
                )
            )
        self.assertEqual(set(), observer._cancelled_before_decode)
        self.assertEqual(
            [(117, [cancelled_id])], observer._bridge.aborted_forward_requests
        )
        self.assertEqual([], observer._bridge.released_forward_batches)
        self.assertEqual(
            [(cancelled_id, "navi_sched_media_error:vLLM media preprocessing failed")],
            observer._bridge.failed_responses,
        )
        enqueue.assert_called_once()
        self.assertEqual([(text_id, requests[text_id])], enqueue.call_args.args[1])

    def test_cancel_during_native_download_excludes_cancelled_requests(self):
        for cancel_all in (False, True):
            with self.subTest(cancel_all=cancel_all):
                observer = VLLMEngineObserver(
                    self.make_config(), self.make_kv_cache_config()
                )
                observer._engine_core = SimpleNamespace()
                batch = FakeForwardBatch(118, ["image-a", "image-b"])
                request_ids = tuple(batch.request_ids)
                cancelled = request_ids if cancel_all else request_ids[:1]
                requests = {
                    request_id: SimpleNamespace(
                        request_id=str(request_id),
                        navi_sched_render_call=b"original-call",
                    )
                    for request_id in request_ids
                }

                async def scenario():
                    started, release = asyncio.Event(), asyncio.Event()

                    async def download(values, **kwargs):
                        started.set()
                        await release.wait()
                        return values

                    observer._render_continuation = SimpleNamespace(
                        resume_async=download
                    )
                    task = asyncio.create_task(
                        observer._prepare_rendered_batch(
                            batch,
                            mock.sentinel.decoder,
                            mock.sentinel.msgpack,
                            mock.sentinel.request_type,
                        )
                    )
                    await asyncio.wait_for(started.wait(), 1)
                    with observer._response_lock:
                        observer._cancelled_before_decode.update(cancelled)
                    release.set()
                    await task

                with (
                    mock.patch.object(
                        observer,
                        "_decode_forward_request",
                        side_effect=lambda request_id, *args: requests[request_id],
                    ),
                    mock.patch.object(observer, "_enqueue_forward_batch") as enqueue,
                ):
                    asyncio.run(scenario())
                self.assertEqual(set(), observer._cancelled_before_decode)
                self.assertEqual([], observer._bridge.failed_responses)
                if cancel_all:
                    enqueue.assert_not_called()
                    self.assertEqual([118], observer._bridge.released_forward_batches)
                    self.assertEqual([], observer._bridge.aborted_forward_requests)
                else:
                    enqueue.assert_called_once()
                    self.assertEqual(
                        [(request_ids[1], requests[request_ids[1]])],
                        enqueue.call_args.args[1],
                    )
                    self.assertEqual([], observer._bridge.released_forward_batches)
                    self.assertEqual(
                        [(118, list(cancelled))],
                        observer._bridge.aborted_forward_requests,
                    )

    def test_forward_worker_discards_cancelled_payloads_before_decode(self):
        for cancelled_count in (1, 2):
            with self.subTest(cancelled_count=cancelled_count):
                observer = VLLMEngineObserver(
                    self.make_config(), self.make_kv_cache_config()
                )
                batch = FakeForwardBatch(125, ["cancelled-1", "cancelled-2"])
                original_ids = tuple(batch.request_ids)
                original_payloads = tuple(batch.payloads)
                observer._bridge.cancelled_requests = list(
                    original_ids[:cancelled_count]
                )
                observer._bridge.queued_forward_batches.append(batch)

                def stop_when_empty():
                    if not observer._bridge.queued_forward_batches:
                        observer._forward_stop.set()

                observer._bridge.take_next_hook = stop_when_empty
                observer._engine_core = SimpleNamespace(
                    input_queue=SimpleNamespace(put_nowait=mock.Mock()),
                    aborts_queue=SimpleNamespace(put_nowait=mock.Mock()),
                )
                engine_module = sys.modules["vllm.v1.engine"]
                serial_utils = types.ModuleType("vllm.v1.serial_utils")
                serial_utils.MsgpackDecoder = lambda unused: mock.sentinel.decoder
                msgspec = types.ModuleType("msgspec")
                msgspec.msgpack = mock.sentinel.msgpack
                with (
                    mock.patch.object(
                        engine_module, "EngineCoreRequest", object(), create=True
                    ),
                    mock.patch.dict(
                        sys.modules,
                        {"msgspec": msgspec, "vllm.v1.serial_utils": serial_utils},
                    ),
                    mock.patch.object(
                        observer,
                        "_decode_forward_request",
                        side_effect=lambda request_id, *unused: SimpleNamespace(
                            request_id=str(request_id)
                        ),
                    ) as decode,
                    mock.patch.object(observer, "_enqueue_forward_batch") as enqueue,
                ):
                    observer._forward_batches()

                expected_ids = list(original_ids[cancelled_count:])
                self.assertEqual(
                    expected_ids, [call.args[0] for call in decode.call_args_list]
                )
                self.assertEqual(set(), observer._cancelled_before_decode)
                self.assertEqual({}, observer._response_request_ids)
                self.assertEqual([], observer._bridge.queued_forward_batches)
                if expected_ids:
                    self.assertEqual(expected_ids, batch.request_ids)
                    self.assertEqual(
                        list(original_payloads[cancelled_count:]), batch.payloads
                    )
                    self.assertEqual([0], batch.predicted_cache_hit_token_counts)
                    self.assertEqual(
                        [(125, list(original_ids[:cancelled_count]))],
                        observer._bridge.aborted_forward_requests,
                    )
                    enqueue.assert_called_once()
                    self.assertEqual([], observer._bridge.released_forward_batches)
                else:
                    enqueue.assert_not_called()
                    self.assertEqual([125], observer._bridge.released_forward_batches)

    def test_before_decode_cancellation_failure_keeps_admission_cleanup(self):
        for failure in (False, RuntimeError("native abort unavailable")):
            with self.subTest(failure=failure):
                observer = VLLMEngineObserver(
                    self.make_config(), self.make_kv_cache_config()
                )
                batch = FakeForwardBatch(126, ["cancelled-1", "live-2"])
                original_ids = tuple(batch.request_ids)
                observer._cancelled_before_decode.add(original_ids[0])
                abort = mock.Mock()
                if isinstance(failure, Exception):
                    abort.side_effect = failure
                else:
                    abort.return_value = failure
                observer._bridge.abort_forward_batch_requests = abort

                self.assertFalse(observer._discard_cancelled_forward_requests(batch))
                self.assertEqual(list(original_ids), batch.request_ids)
                self.assertEqual({original_ids[0]}, observer._cancelled_before_decode)
                self.assertEqual([], observer._bridge.released_forward_batches)
                decoded = [
                    (request_id, SimpleNamespace(request_id=str(request_id)))
                    for request_id in original_ids
                ]
                self.assertEqual(
                    {original_ids[0]}, observer._register_response_requests(decoded)
                )
                self.assertEqual(set(), observer._cancelled_before_decode)

    def test_before_decode_full_cancellation_allows_request_id_reuse(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        batch = FakeForwardBatch(127, ["reused-1"])
        request_id = batch.request_ids[0]
        observer._cancelled_before_decode.add(request_id)
        self.assertTrue(observer._discard_cancelled_forward_requests(batch))
        self.assertEqual([127], observer._bridge.released_forward_batches)
        self.assertEqual(set(), observer._cancelled_before_decode)
        self.assertEqual(
            set(),
            observer._register_response_requests(
                [(request_id, SimpleNamespace(request_id="replacement"))]
            ),
        )
        self.assertEqual({request_id: "replacement"}, observer._response_request_ids)

    def test_unified_forward_worker_ignores_pd_attached_touches(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        self.assertFalse(observer._native_ingress)
        batch = FakeForwardBatch(114, ["unified-request"])
        batch.touches = [FakeForwardRequestTouch(batch.request_ids[0], b"payload")]
        observer._bridge.queued_forward_batches.append(batch)
        engine_module = sys.modules["vllm.v1.engine"]
        serial_utils = types.ModuleType("vllm.v1.serial_utils")
        serial_utils.MsgpackDecoder = lambda unused: mock.sentinel.decoder
        msgspec = types.ModuleType("msgspec")
        msgspec.msgpack = mock.sentinel.msgpack

        def enqueue_then_stop(*unused_args, **unused_kwargs):
            observer._forward_stop.set()

        with (
            mock.patch.object(
                engine_module, "EngineCoreRequest", object(), create=True
            ),
            mock.patch.dict(
                sys.modules,
                {
                    "msgspec": msgspec,
                    "vllm.v1.serial_utils": serial_utils,
                },
            ),
            mock.patch.object(observer, "_decode_forward_batch", return_value=[]),
            mock.patch.object(
                observer, "_enqueue_forward_batch", side_effect=enqueue_then_stop
            ),
            mock.patch.object(
                observer, "_prepare_forward_request_touches"
            ) as prepare_touches,
        ):
            observer._forward_batches()

        prepare_touches.assert_not_called()

    def test_pd_forward_worker_enqueues_next_batch_before_prior_batch_released(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
            },
        ):
            observer = VLLMEngineObserver(
                self.make_config(), self.make_kv_cache_config()
            )
        first = FakeForwardBatch(201, ["first"])
        second = FakeForwardBatch(202, ["second"])
        observer._bridge.queued_forward_batches.extend((first, second))
        first_seen = threading.Event()
        second_seen = threading.Event()
        received = []

        def enqueue(batch, *unused_args, image_predictions=None):
            received.append(batch.batch_id)
            if len(received) == 1:
                first_seen.set()
            else:
                second_seen.set()
                observer._forward_stop.set()

        engine_module = sys.modules["vllm.v1.engine"]
        serial_utils = types.ModuleType("vllm.v1.serial_utils")
        serial_utils.MsgpackDecoder = lambda unused: mock.sentinel.decoder
        msgspec = types.ModuleType("msgspec")
        msgspec.msgpack = mock.sentinel.msgpack
        with (
            mock.patch.object(
                engine_module, "EngineCoreRequest", object(), create=True
            ),
            mock.patch.dict(
                sys.modules,
                {
                    "msgspec": msgspec,
                    "vllm.v1.serial_utils": serial_utils,
                },
            ),
            mock.patch.object(observer, "_decode_forward_batch", return_value=[]),
            mock.patch.object(observer, "_enqueue_forward_batch", side_effect=enqueue),
        ):
            worker = threading.Thread(target=observer._forward_batches)
            worker.start()
            try:
                self.assertTrue(first_seen.wait(1))
                self.assertTrue(second_seen.wait(1))
                self.assertEqual([201, 202], received)
                state = SimpleNamespace(
                    lock=threading.Lock(),
                    released=False,
                    handle=object(),
                    native_batch=first,
                    batch_id=first.batch_id,
                    request_ids=(),
                )
                observer._release_forward_batch(state)
                self.assertTrue(state.released)
                self.assertEqual([201, 202], received)
            finally:
                observer._forward_stop.set()
                worker.join(timeout=1)
            self.assertFalse(worker.is_alive())

    def test_pd_ready_batch_prepares_attached_cache_touch_before_decode(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
            },
        ):
            observer = VLLMEngineObserver(
                self.make_config(), self.make_kv_cache_config()
            )
        batch = FakeForwardBatch(203, ["internal-1"])
        batch.touches = [FakeForwardRequestTouch(batch.request_ids[0], b"payload")]
        observer._bridge.queued_forward_batches.append(batch)
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id=str(batch.request_ids[0]),
        )
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=mock.Mock(return_value=(request,))
        )

        def decode_forward_batch(*unused_args, image_predictions=None):
            self.assertIn(batch.request_ids[0], observer._preprocessed_cache_touches)
            return []

        def enqueue_then_stop(*unused_args, **unused_kwargs):
            observer._forward_stop.set()

        engine_module = sys.modules["vllm.v1.engine"]
        serial_utils = types.ModuleType("vllm.v1.serial_utils")
        serial_utils.MsgpackDecoder = lambda unused: mock.sentinel.decoder
        msgspec = types.ModuleType("msgspec")
        msgspec.msgpack = mock.sentinel.msgpack
        with (
            mock.patch.object(
                engine_module, "EngineCoreRequest", object(), create=True
            ),
            mock.patch.dict(
                sys.modules,
                {
                    "msgspec": msgspec,
                    "vllm.v1.serial_utils": serial_utils,
                },
            ),
            mock.patch.object(
                observer, "_decode_forward_request", return_value=request
            ),
            mock.patch.object(
                observer, "_decode_forward_batch", side_effect=decode_forward_batch
            ),
            mock.patch.object(
                observer, "_enqueue_forward_batch", side_effect=enqueue_then_stop
            ),
        ):
            observer._forward_batches()

        observer._engine_core.preprocess_add_request.assert_called_once()

    def test_failed_decode_discards_cancel_tombstone_for_request_id_reuse(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request_id = stable_request_id("internal-1")
        observer._bridge.cancelled_requests = [request_id]
        observer._engine_core = SimpleNamespace(
            aborts_queue=SimpleNamespace(put_nowait=lambda unused: None),
            input_queue=SimpleNamespace(put_nowait=lambda unused: None),
        )
        observer._drain_cancelled_requests(SimpleNamespace(ADD="add", ABORT="abort"))
        self.assertEqual({request_id}, observer._cancelled_before_decode)

        native_batch = FakeForwardBatch(124, ["internal-1"])
        decoder = SimpleNamespace(
            decode=mock.Mock(side_effect=ValueError("invalid payload"))
        )
        msgpack = SimpleNamespace(decode=lambda unused: [])
        with self.assertRaisesRegex(ValueError, "invalid payload"):
            observer._decode_forward_batch(native_batch, decoder, msgpack)

        self.assertEqual(set(), observer._cancelled_before_decode)
        replacement = self.make_request(
            prompt_tokens=16,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-1",
        )
        self.assertEqual(
            set(),
            observer._register_response_requests([(request_id, replacement)]),
        )

    def test_decode_forward_batch_replaces_payload_identity_with_master_id(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        canonical_request_id = 9_000_000_000_000_007
        native_batch = SimpleNamespace(
            request_ids=[canonical_request_id],
            payloads=[b"serialized-request"],
        )
        decoded_request = SimpleNamespace(
            request_id="chat-old-payload-id", client_index=17
        )
        decoder = SimpleNamespace(decode=mock.Mock(return_value=decoded_request))
        msgpack = SimpleNamespace(decode=mock.Mock(return_value=[b"frame"]))

        decoded = observer._decode_forward_batch(native_batch, decoder, msgpack)

        self.assertEqual([(canonical_request_id, decoded_request)], decoded)
        self.assertEqual(str(canonical_request_id), decoded_request.request_id)
        self.assertEqual(canonical_request_id, decoded_request.client_index)

    def test_decode_resumes_native_renderer_before_admission(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = SimpleNamespace(
            request_id="source", client_index=3, navi_sched_render_call=b"native-call"
        )
        continuation = mock.Mock()
        continuation.resume_sync.return_value = [request]
        observer._render_continuation = continuation
        batch = SimpleNamespace(request_ids=[71], payloads=[b"payload"])
        decoded = observer._decode_forward_batch(
            batch,
            SimpleNamespace(decode=lambda frames: request),
            SimpleNamespace(decode=lambda payload: [b"frame"]),
        )
        continuation.resume_sync.assert_called_once_with(
            [request], return_exceptions=True
        )
        self.assertEqual([(71, request)], decoded)
        self.assertEqual("71", request.request_id)
        self.assertEqual(71, request.client_index)

    def test_image_mapping_reuses_first_decode_and_forwarded_block_hashes(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        feature = lambda offset, length: SimpleNamespace(
            modality="image",
            mm_position=SimpleNamespace(offset=offset, length=length),
        )
        tokens = list(range(48))
        features = [feature(17, 1)]
        request = SimpleNamespace(
            request_id="source",
            client_index=3,
            prompt_token_ids=tokens,
            mm_features=features,
            navi_sched_render_call=b"call",
        )
        batch = SimpleNamespace(
            request_ids=[71],
            payloads=[b"payload"],
            get_request_block_hashes=mock.Mock(return_value=[101, 102, 103]),
        )
        decoder = SimpleNamespace(decode=mock.Mock(return_value=request))
        msgpack = SimpleNamespace(decode=mock.Mock(return_value=[b"frame"]))
        predictions = {}
        observer._decode_forward_batch(
            batch,
            decoder,
            msgpack,
            resume_rendering=False,
            image_predictions=predictions,
        )
        self.assertIs(tokens, predictions[71].prompt_token_ids)
        self.assertIs(features, predictions[71].mm_features)
        request.prompt_token_ids = list(range(79))
        request.mm_features = [feature(17, 32)]
        actual = SimpleNamespace(
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            block_hashes=[value.to_bytes(32, "big") for value in (11, 12, 13, 14)],
        )
        observer._register_response_requests([(71, request)])
        with (
            mock.patch.object(
                observer,
                "_decode_forward_request",
                side_effect=AssertionError(
                    "mapping must not decode the payload again",
                ),
            ),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMImageCacheMap.calculate_vllm_block_hashes",
                side_effect=AssertionError(
                    "mapping must reuse forwarded scheduling hashes"
                ),
            ),
        ):
            observer._register_image_cache_mappings(
                batch, [(71, (actual,))], predictions
            )
        decoder.decode.assert_called_once_with([b"frame"])
        batch.get_request_block_hashes.assert_called_once_with(71)
        record = next(iter(observer._image_cache_map._records.values()))
        self.assertEqual(((102).to_bytes(32, "big"),), record.logical_hashes)
        self.assertEqual((4,), record.required_blocks)
        self.assertEqual({71: ((17, 32),)}, observer._response_image_positions)
        observer._remove_response_requests((71,))
        self.assertEqual({}, observer._response_image_positions)

    def test_text_decode_does_not_capture_image_mapping_metadata(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = SimpleNamespace(request_id="text", client_index=3, mm_features=None)
        batch = SimpleNamespace(request_ids=[71], payloads=[b"payload"])
        predictions = {}
        observer._decode_forward_batch(
            batch,
            SimpleNamespace(decode=lambda frames: request),
            SimpleNamespace(decode=lambda payload: [b"frame"]),
            image_predictions=predictions,
        )
        self.assertEqual({}, predictions)
        observer._register_image_cache_mappings(batch, [], predictions)

    def test_failed_native_render_uses_decode_failure_cleanup(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = SimpleNamespace(
            request_id="source", client_index=3, navi_sched_render_call=b"native-call"
        )
        observer._render_continuation = mock.Mock()
        observer._render_continuation.resume_sync.side_effect = ValueError(
            "native render failed"
        )
        observer._fail_forward_responses = mock.Mock()
        observer._discard_cache_touch_requests = mock.Mock()
        observer._discard_cancelled_before_decode = mock.Mock()
        batch = SimpleNamespace(request_ids=[71], payloads=[b"payload"])
        with self.assertRaisesRegex(ValueError, "native render failed"):
            observer._decode_forward_batch(
                batch,
                SimpleNamespace(decode=lambda frames: request),
                SimpleNamespace(decode=lambda payload: [b"frame"]),
            )
        observer._fail_forward_responses.assert_called_once_with(
            [71], "invalid NaviSched VLLM forward request"
        )
        observer._discard_cache_touch_requests.assert_called_once_with([71])
        observer._discard_cancelled_before_decode.assert_called_once_with([71])

    def test_native_render_failure_isolates_one_request(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        good = SimpleNamespace(navi_sched_render_call=b"good")
        bad = SimpleNamespace(navi_sched_render_call=b"bad")
        text = SimpleNamespace(navi_sched_render_call=None)
        observer._render_continuation = mock.Mock()
        observer._render_continuation.resume_sync.return_value = [
            good,
            ValueError("404"),
        ]
        observer._fail_forward_responses = mock.Mock()
        observer._discard_cache_touch_requests = mock.Mock()
        observer._discard_cancelled_before_decode = mock.Mock()
        batch = SimpleNamespace(
            batch_id=7,
            request_ids=[71, 72, 73],
            payloads=[b"a", b"b", b"c"],
            predicted_cache_hit_token_counts=[16, 32, 48],
        )
        decoded = observer._decode_forward_batch(
            batch,
            SimpleNamespace(decode=mock.Mock(side_effect=[good, bad, text])),
            SimpleNamespace(decode=lambda payload: []),
        )
        self.assertEqual([(71, good), (73, text)], decoded)
        self.assertEqual([71, 73], batch.request_ids)
        self.assertEqual([b"a", b"c"], batch.payloads)
        self.assertEqual([16, 48], batch.predicted_cache_hit_token_counts)
        self.assertEqual([(7, [72])], observer._bridge.aborted_forward_requests)
        observer._fail_forward_responses.assert_called_once_with(
            [72], "navi_sched_media_error:vLLM media preprocessing failed"
        )
        observer._discard_cache_touch_requests.assert_called_once_with([72])
        observer._discard_cancelled_before_decode.assert_called_once_with([72])

    def test_native_render_failure_cleans_all_when_isolation_fails(self):
        for all_failed in (False, True):
            with self.subTest(all_failed=all_failed):
                observer = VLLMEngineObserver(
                    self.make_config(), self.make_kv_cache_config()
                )
                requests = [
                    SimpleNamespace(navi_sched_render_call=b"call") for _ in range(2)
                ]
                observer._render_continuation = mock.Mock()
                observer._render_continuation.resume_sync.return_value = [
                    ValueError("404"),
                    ValueError("404") if all_failed else requests[1],
                ]
                observer._bridge.abort_forward_batch_requests = mock.Mock(
                    return_value=False
                )
                observer._fail_forward_responses = mock.Mock()
                observer._discard_cache_touch_requests = mock.Mock()
                observer._discard_cancelled_before_decode = mock.Mock()
                batch = SimpleNamespace(request_ids=[71, 72], payloads=[b"a", b"b"])
                with self.assertRaisesRegex(
                    (ValueError, RuntimeError),
                    "all forwarded requests|failed to isolate",
                ):
                    observer._decode_forward_batch(
                        batch,
                        SimpleNamespace(decode=mock.Mock(side_effect=requests)),
                        SimpleNamespace(decode=lambda payload: []),
                    )
                observer._fail_forward_responses.assert_called_once_with(
                    [71, 72],
                    (
                        "navi_sched_media_error:vLLM media preprocessing failed"
                        if all_failed
                        else "invalid NaviSched VLLM forward request"
                    ),
                )
                observer._discard_cache_touch_requests.assert_called_once_with([71, 72])
                observer._discard_cancelled_before_decode.assert_called_once_with(
                    [71, 72]
                )
                self.assertEqual(
                    0 if all_failed else 1,
                    observer._bridge.abort_forward_batch_requests.call_count,
                )

    def test_cache_touch_preprocesses_on_arrival_and_reuses_batch_request(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        canonical_request_id = 9_000_000_000_000_011
        payload = b"serialized-request"
        engine_request = SimpleNamespace(request_id="payload-id", client_index=1)
        preprocessed_request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id=str(canonical_request_id),
        )
        preprocess = mock.Mock(return_value=(preprocessed_request, 0))
        input_queue = SimpleNamespace(put_nowait=mock.Mock())
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=preprocess,
            input_queue=input_queue,
        )
        observer._bridge.queued_forward_request_touches.append(
            FakeForwardRequestTouch(canonical_request_id, payload)
        )
        decoder = SimpleNamespace(decode=mock.Mock(return_value=engine_request))
        msgpack = SimpleNamespace(decode=mock.Mock(return_value=[b"frame"]))

        observer._drain_forward_request_touches(decoder, msgpack)

        preprocess.assert_called_once_with(engine_request)
        self.assertIn(canonical_request_id, observer._preprocessed_cache_touches)
        native_batch = SimpleNamespace(
            batch_id=15,
            schedule_batch_id=15,
            schedule_batch_size=1,
            pending_cache_tracked=True,
            request_ids=[canonical_request_id],
            payloads=[payload],
            predicted_cache_hit_token_counts=[0],
        )
        decoded = observer._decode_forward_batch(native_batch, decoder, msgpack)
        observer._enqueue_forward_batch(
            native_batch,
            decoded,
            SimpleNamespace(ADD="add", ABORT="abort"),
            preceding_workload=_ObservedWorkload(max_running_request_count=1),
        )

        preprocess.assert_called_once_with(engine_request)
        input_queue.put_nowait.assert_called_once_with(
            ("add", (preprocessed_request, 0))
        )
        self.assertNotIn(canonical_request_id, observer._preprocessed_cache_touches)

    def test_cache_touch_preparation_is_bounded_to_requested_count(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request_ids = (9_000_000_000_000_021, 9_000_000_000_000_022)
        requests = [
            SimpleNamespace(request_id=f"payload-{index}", client_index=index)
            for index in range(2)
        ]
        preprocessed = [
            self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=0,
                request_id=str(request_id),
            )
            for request_id in request_ids
        ]
        preprocess = mock.Mock(side_effect=[(request, 0) for request in preprocessed])
        observer._engine_core = SimpleNamespace(preprocess_add_request=preprocess)
        observer._bridge.queued_forward_request_touches.extend(
            FakeForwardRequestTouch(request_id, f"payload-{index}".encode())
            for index, request_id in enumerate(request_ids)
        )
        decoder = SimpleNamespace(decode=mock.Mock(side_effect=requests))
        msgpack = SimpleNamespace(decode=mock.Mock(return_value=[b"frame"]))

        observer._drain_forward_request_touches(decoder, msgpack, max_count=1)

        preprocess.assert_called_once_with(requests[0])
        self.assertEqual(
            [request_ids[1]],
            [
                touch.request_id
                for touch in observer._bridge.queued_forward_request_touches
            ],
        )
        self.assertEqual(
            [request_ids[0]],
            list(observer._preprocessed_cache_touches),
        )

    def test_cache_touch_is_promoted_by_owner_scheduler_before_native_schedule(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
        )
        with observer._cache_touch_lock:
            observer._cache_touch_queue.append(
                (stable_request_id(request.request_id), request, time.monotonic_ns())
            )

        scheduler.schedule()

        self.assertEqual(
            [request], scheduler._scheduler.kv_cache_manager.promoted_requests
        )
        self.assertEqual([], list(observer._cache_touch_queue))

    def test_cache_touch_reports_aggregates_through_native_alog_bridge(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
        )
        observer._cache_touch_last_log_ns = 0
        with observer._cache_touch_lock:
            observer._cache_touch_queue.append(
                (stable_request_id(request.request_id), request, time.monotonic_ns())
            )

        scheduler.schedule()

        self.assertEqual(1, len(observer._bridge.cache_touch_audits))
        audit = observer._bridge.cache_touch_audits[0]
        self.assertEqual((1, len(request.block_hashes), 1), audit[:3])
        self.assertGreaterEqual(audit[3], 0.0)
        self.assertGreaterEqual(audit[4], 0.0)
        self.assertEqual(0, observer._cache_touch_count)
        self.assertEqual(0, observer._cache_touch_promoted_blocks)

    def test_cache_touch_large_queue_drains_once_without_loss(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        requests = [
            self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=0,
                request_id=f"request-{index}",
            )
            for index in range(4096)
        ]
        with observer._cache_touch_lock:
            observer._cache_touch_queue.extend(
                (
                    stable_request_id(request.request_id),
                    request,
                    time.monotonic_ns(),
                )
                for request in requests
            )

        scheduler.schedule()
        scheduler.schedule()

        self.assertEqual(
            requests, scheduler._scheduler.kv_cache_manager.promoted_requests
        )
        self.assertEqual([], list(observer._cache_touch_queue))

    def test_cache_touch_failure_does_not_skip_remaining_or_native_schedule(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())

        class FailingTouchScheduler(LifecycleScheduler):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.native_schedule_count = 0
                self.touch_count = 0

                def get_computed_blocks(request):
                    self.touch_count += 1
                    if self.touch_count == 1:
                        raise RuntimeError("injected touch failure")
                    self.kv_cache_manager.promoted_requests.append(request)
                    return SimpleNamespace(blocks=([1],)), 0, 0

                self.kv_cache_manager.get_computed_blocks = get_computed_blocks

            def schedule(self):
                self.native_schedule_count += 1
                return super().schedule()

        scheduler = observer.create_scheduler(FailingTouchScheduler)
        requests = [
            self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=0,
                request_id=f"request-{index}",
            )
            for index in range(2)
        ]
        with observer._cache_touch_lock:
            observer._cache_touch_queue.extend(
                (
                    stable_request_id(request.request_id),
                    request,
                    time.monotonic_ns(),
                )
                for request in requests
            )

        with self.assertLogs("vllm.navi_sched.agent", level="ERROR") as logs:
            scheduler.schedule()

        self.assertIn("injected touch failure", "\n".join(logs.output))
        self.assertEqual(2, scheduler._scheduler.touch_count)
        self.assertEqual(1, scheduler._scheduler.native_schedule_count)
        self.assertEqual(
            [requests[1]],
            scheduler._scheduler.kv_cache_manager.promoted_requests,
        )

    def test_cancel_discards_preprocessed_and_queued_cache_touch(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request_id = stable_request_id("internal-1")
        engine_request = SimpleNamespace(request_id="payload-id", client_index=1)
        preprocessed_request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id=str(request_id),
        )
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda unused: (preprocessed_request, 0),
            aborts_queue=SimpleNamespace(put_nowait=lambda unused: None),
            input_queue=SimpleNamespace(put_nowait=lambda unused: None),
        )
        observer._bridge.queued_forward_request_touches.append(
            FakeForwardRequestTouch(request_id, b"payload")
        )
        observer._drain_forward_request_touches(
            SimpleNamespace(decode=lambda unused: engine_request),
            SimpleNamespace(decode=lambda unused: [b"frame"]),
        )
        observer._bridge.cancelled_requests = [request_id]

        self.assertTrue(
            observer._drain_cancelled_requests(
                SimpleNamespace(ADD="add", ABORT="abort")
            )
        )

        self.assertNotIn(request_id, observer._preprocessed_cache_touches)
        self.assertEqual([], list(observer._cache_touch_queue))

    def test_invalid_cache_touch_payload_does_not_block_later_batch_decode(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request_id = stable_request_id("internal-1")
        observer._engine_core = SimpleNamespace(preprocess_add_request=mock.Mock())
        observer._bridge.queued_forward_request_touches.append(
            FakeForwardRequestTouch(request_id, b"invalid")
        )
        decoder = SimpleNamespace(
            decode=mock.Mock(side_effect=ValueError("invalid payload"))
        )

        observer._drain_forward_request_touches(
            decoder, SimpleNamespace(decode=lambda unused: [b"frame"])
        )

        self.assertNotIn(request_id, observer._preprocessed_cache_touches)
        self.assertEqual([], list(observer._cache_touch_queue))
        observer._engine_core.preprocess_add_request.assert_not_called()

    def test_response_route_registration_is_atomic_on_duplicate(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        first = self.make_request(
            prompt_tokens=16,
            computed_tokens=0,
            cached_tokens=0,
            request_id="first",
        )
        duplicate = self.make_request(
            prompt_tokens=16,
            computed_tokens=0,
            cached_tokens=0,
            request_id="duplicate",
        )
        duplicate_id = stable_request_id(duplicate.request_id)
        observer._response_request_ids[duplicate_id] = duplicate.request_id

        with self.assertRaisesRegex(ValueError, "duplicate NaviSched VLLM response"):
            observer._register_response_requests(
                [
                    (stable_request_id(first.request_id), first),
                    (duplicate_id, duplicate),
                ]
            )

        self.assertEqual(
            {duplicate_id: duplicate.request_id}, observer._response_request_ids
        )

    def test_invalid_predicted_cache_hits_return_one_terminal_error(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-1",
        )
        native_batch = FakeForwardBatch(113, [request.request_id], [])
        observer._engine_core = SimpleNamespace(engine_index=3)
        request_id = stable_request_id(request.request_id)

        with self.assertRaisesRegex(
            ValueError, "invalid NaviSched predicted cache-hit metadata"
        ):
            observer._enqueue_forward_batch(
                native_batch,
                [(request_id, request)],
                SimpleNamespace(ADD="add", ABORT="abort"),
            )

        self.assertEqual(
            [(request_id, "failed to register NaviSched VLLM response route")],
            observer._bridge.failed_responses,
        )
        self.assertEqual([113], observer._bridge.released_forward_batches)
        self.assertEqual({}, observer._response_request_ids)

    def test_partial_forward_enqueue_queues_abort_and_cleans_state(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)

        class FailingQueue:
            def __init__(self):
                self.items = []
                self.add_count = 0

            def put_nowait(self, item):
                if item[0] == "add":
                    self.add_count += 1
                    if self.add_count == 2:
                        raise RuntimeError("queue failed")
                self.items.append(item)

        input_queue = FailingQueue()
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda request: (request, 0),
            input_queue=input_queue,
            aborts_queue=SimpleNamespace(put_nowait=lambda unused: None),
        )
        requests = [
            self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=0,
                request_id=request_id,
            )
            for request_id in ("internal-1", "internal-2")
        ]
        native_batch = FakeForwardBatch(
            95, [request.request_id for request in requests]
        )
        request_type = SimpleNamespace(ADD="add", ABORT="abort")

        with self.assertRaisesRegex(RuntimeError, "queue failed"):
            observer._enqueue_forward_batch(
                native_batch,
                [
                    (stable_request_id(request.request_id), request)
                    for request in requests
                ],
                request_type,
            )

        self.assertEqual([2, 0], [load[2] for load in observer._bridge.workloads[-2:]])
        self.assertEqual(0, observer._bridge.workloads[-1][7])
        add_item, abort_item = input_queue.items
        scheduler.add_request(add_item[1][0])
        scheduler.finish_requests(abort_item[1], "aborted")
        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual({}, observer._response_request_ids)
        self.assertEqual(set(), observer._inactive_response_request_ids)
        self.assertEqual([95], observer._bridge.released_forward_batches)

    def test_forward_enqueue_failure_releases_all_state(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())

        class FailingQueue:
            def put_nowait(self, item):
                request_type, _ = item
                raise RuntimeError(f"{request_type} queue failed")

        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-1",
        )
        native_batch = FakeForwardBatch(97, [request.request_id])
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda value: (value, 0),
            input_queue=FailingQueue(),
            aborts_queue=SimpleNamespace(put_nowait=lambda unused: None),
        )

        with self.assertRaisesRegex(RuntimeError, "add queue failed"):
            observer._enqueue_forward_batch(
                native_batch,
                [(stable_request_id(request.request_id), request)],
                SimpleNamespace(ADD="add", ABORT="abort"),
            )

        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual({}, observer._response_request_ids)
        self.assertEqual(set(), observer._inactive_response_request_ids)
        self.assertEqual([97], observer._bridge.released_forward_batches)

        self.assertEqual([1, 0], [load[2] for load in observer._bridge.workloads[-2:]])
        self.assertEqual(0, observer._bridge.workloads[-1][7])

    def test_preprocess_failure_releases_native_batch_once(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-1",
        )
        native_batch = FakeForwardBatch(97, [request.request_id])
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=mock.Mock(
                side_effect=RuntimeError("invalid request")
            ),
            input_queue=SimpleNamespace(put_nowait=mock.Mock()),
        )

        with self.assertRaisesRegex(RuntimeError, "invalid request"):
            observer._enqueue_forward_batch(
                native_batch,
                [(stable_request_id(request.request_id), request)],
                SimpleNamespace(ADD="add", ABORT="abort"),
            )

        self.assertEqual({}, observer._pending_forward_batches)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual({}, observer._response_request_ids)
        self.assertEqual(set(), observer._inactive_response_request_ids)
        self.assertEqual([97], observer._bridge.released_forward_batches)

    def test_multimodal_request_is_rejected_by_text_only_target(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="multimodal",
        )
        request.mm_features = [object()]
        request_id = stable_request_id(request.request_id)
        native_batch = FakeForwardBatch(98, [request.request_id])
        preprocess = mock.Mock()
        input_queue = SimpleNamespace(put_nowait=mock.Mock())
        observer._engine_core = SimpleNamespace(
            scheduler=SimpleNamespace(supports_mm_inputs=False),
            preprocess_add_request=preprocess,
            input_queue=input_queue,
        )

        observer._enqueue_forward_batch(
            native_batch,
            [(request_id, request)],
            SimpleNamespace(ADD="add", ABORT="abort"),
        )

        preprocess.assert_not_called()
        input_queue.put_nowait.assert_not_called()
        self.assertEqual(
            [(request_id, "multimodal request is unsupported by target model")],
            observer._bridge.failed_responses,
        )
        self.assertEqual([98], observer._bridge.released_forward_batches)
        self.assertEqual({}, observer._response_request_ids)
        self.assertEqual({}, observer._pending_forward_batches)

    def test_text_request_works_without_scheduler_multimodal_flag(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="text-only-r39",
        )
        request_id = stable_request_id(request.request_id)
        native_batch = FakeForwardBatch(101, [request.request_id])
        input_queue = SimpleNamespace(put_nowait=mock.Mock())
        observer._engine_core = SimpleNamespace(
            scheduler=SimpleNamespace(),
            preprocess_add_request=lambda value: (value, 0),
            input_queue=input_queue,
        )

        observer._enqueue_forward_batch(
            native_batch,
            [(request_id, request)],
            SimpleNamespace(ADD="add", ABORT="abort"),
        )

        input_queue.put_nowait.assert_called_once_with(("add", (request, 0)))
        self.assertIn(request_id, observer._pending_forward_batches)
        self.assertEqual([], observer._bridge.failed_responses)

    def test_multimodal_request_rejected_without_scheduler_multimodal_flag(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="image-r39",
        )
        request.mm_features = [object()]
        request_id = stable_request_id(request.request_id)
        native_batch = FakeForwardBatch(102, [request.request_id])
        preprocess = mock.Mock()
        input_queue = SimpleNamespace(put_nowait=mock.Mock())
        observer._engine_core = SimpleNamespace(
            scheduler=SimpleNamespace(),
            preprocess_add_request=preprocess,
            input_queue=input_queue,
        )

        observer._enqueue_forward_batch(
            native_batch,
            [(request_id, request)],
            SimpleNamespace(ADD="add", ABORT="abort"),
        )

        preprocess.assert_not_called()
        input_queue.put_nowait.assert_not_called()
        self.assertEqual(
            [(request_id, "multimodal request is unsupported by target model")],
            observer._bridge.failed_responses,
        )
        self.assertEqual([102], observer._bridge.released_forward_batches)

    def test_multimodal_rejection_retains_text_request_in_same_batch(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        text_request = self.make_request(
            prompt_tokens=32, computed_tokens=0, cached_tokens=0, request_id="text"
        )
        multimodal_request = self.make_request(
            prompt_tokens=32, computed_tokens=0, cached_tokens=0, request_id="image"
        )
        multimodal_request.mm_features = [object()]
        text_id = stable_request_id(text_request.request_id)
        multimodal_id = stable_request_id(multimodal_request.request_id)
        native_batch = FakeForwardBatch(99, ["text", "image"])
        input_queue = SimpleNamespace(put_nowait=mock.Mock())
        observer._engine_core = SimpleNamespace(
            scheduler=SimpleNamespace(supports_mm_inputs=False),
            preprocess_add_request=lambda request: (request, 0),
            input_queue=input_queue,
        )

        observer._enqueue_forward_batch(
            native_batch,
            [(text_id, text_request), (multimodal_id, multimodal_request)],
            SimpleNamespace(ADD="add", ABORT="abort"),
        )

        self.assertEqual([text_id], native_batch.request_ids)
        self.assertEqual(
            [(99, [multimodal_id])], observer._bridge.aborted_forward_requests
        )
        input_queue.put_nowait.assert_called_once_with(("add", (text_request, 0)))
        self.assertEqual(
            [(multimodal_id, "multimodal request is unsupported by target model")],
            observer._bridge.failed_responses,
        )
        self.assertEqual({text_id: "text"}, observer._response_request_ids)
        self.assertIn(text_id, observer._pending_forward_batches)
        self.assertNotIn(multimodal_id, observer._pending_forward_batches)

    def test_multimodal_rejection_fails_closed_if_partial_abort_fails(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        text_request = self.make_request(
            prompt_tokens=32, computed_tokens=0, cached_tokens=0, request_id="text"
        )
        multimodal_request = self.make_request(
            prompt_tokens=32, computed_tokens=0, cached_tokens=0, request_id="image"
        )
        multimodal_request.mm_features = [object()]
        text_id = stable_request_id(text_request.request_id)
        multimodal_id = stable_request_id(multimodal_request.request_id)
        native_batch = FakeForwardBatch(100, ["text", "image"])
        observer._bridge.abort_forward_batch_requests = mock.Mock(return_value=False)
        input_queue = SimpleNamespace(put_nowait=mock.Mock())
        observer._engine_core = SimpleNamespace(
            scheduler=SimpleNamespace(supports_mm_inputs=False),
            preprocess_add_request=mock.Mock(),
            input_queue=input_queue,
        )

        observer._enqueue_forward_batch(
            native_batch,
            [(text_id, text_request), (multimodal_id, multimodal_request)],
            SimpleNamespace(ADD="add", ABORT="abort"),
        )

        input_queue.put_nowait.assert_not_called()
        self.assertEqual(
            {
                (multimodal_id, "multimodal request is unsupported by target model"),
                (text_id, "failed to isolate unsupported multimodal request"),
            },
            set(observer._bridge.failed_responses),
        )
        self.assertEqual({}, observer._response_request_ids)
        self.assertEqual([100], observer._bridge.released_forward_batches)

    def test_multimodal_request_is_allowed_by_multimodal_target(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="multimodal",
        )
        request.mm_features = [object()]
        request_id = stable_request_id(request.request_id)
        native_batch = FakeForwardBatch(99, [request.request_id])
        preprocess = mock.Mock(return_value=(request, 0))
        input_queue = SimpleNamespace(put_nowait=mock.Mock())
        observer._engine_core = SimpleNamespace(
            scheduler=SimpleNamespace(supports_mm_inputs=True),
            preprocess_add_request=preprocess,
            input_queue=input_queue,
        )

        observer._enqueue_forward_batch(
            native_batch,
            [(request_id, request)],
            SimpleNamespace(ADD="add", ABORT="abort"),
        )

        preprocess.assert_called_once_with(request)
        input_queue.put_nowait.assert_called_once_with(("add", (request, 0)))
        self.assertIn(request_id, observer._pending_forward_batches)
        self.assertEqual([], observer._bridge.failed_responses)

    def test_decode_requests_may_share_step_with_active_prefill(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        self.add_forward_batch(observer, 96, ["prefill"])
        prefill = self.make_request(
            prompt_tokens=32,
            computed_tokens=32,
            cached_tokens=0,
            request_id="prefill",
        )
        state = observer._admit_request(prefill)
        decode = self.make_request(
            prompt_tokens=32,
            computed_tokens=33,
            cached_tokens=0,
            request_id="decode",
        )
        decode.num_output_tokens = 1
        decode.max_tokens = 16
        output = SimpleNamespace(num_scheduled_tokens={"prefill": 32, "decode": 1})

        self.run_execution_step(
            observer, output, {"prefill": prefill, "decode": decode}
        )

        self.assertTrue(state.prefill_complete)
        self.assertEqual(1, state.execution_decode_request_count)
        self.assertEqual(1, state.execution_decode_tokens)

    def test_execution_tracking_captures_start_state_and_balances_counts(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        running = [
            SimpleNamespace(
                num_computed_tokens=64,
                num_prompt_tokens=32,
                num_preemptions=0,
            ),
            SimpleNamespace(
                num_computed_tokens=16,
                num_prompt_tokens=32,
                num_preemptions=1,
            ),
        ]
        scheduler = SimpleNamespace(running=running, waiting=[1, 2, 3])
        output = SimpleNamespace(
            preempted_req_ids={"preempted"},
            scheduled_cached_reqs=SimpleNamespace(
                resumed_req_ids={"resumed-1", "resumed-2"}
            ),
        )
        first = _ExecutionStep(
            started_ns=1,
            scheduler_output_key=id(output),
            prefill_request_count=1,
            prefill_tokens=32,
            decode_request_count=1,
            decode_tokens=2,
        )
        output.num_scheduled_tokens = {"prefill": 32, "decode": 2}
        observer._register_scheduled_work(output)

        observer._start_execution_tracking(first, scheduler, output)

        self.assertEqual(2, first.running_request_count)
        self.assertEqual(1, first.running_prefill_request_count)
        self.assertEqual(1, first.running_decode_request_count)
        self.assertEqual(1, first.running_recompute_request_count)
        self.assertEqual(3, first.waiting_request_count)
        self.assertEqual(1, first.preempted_request_count)
        self.assertEqual(2, first.resumed_request_count)
        self.assertEqual(0, first.active_executions_before)

        second_output = SimpleNamespace(num_scheduled_tokens={"decode-2": 1})
        second = _ExecutionStep(
            started_ns=2,
            scheduler_output_key=id(second_output),
            decode_request_count=1,
            decode_tokens=1,
        )
        observer._register_scheduled_work(second_output)
        observer._start_execution_tracking(second, scheduler, second_output)
        self.assertEqual(1, second.active_executions_before)
        self.assertEqual(1, second.active_prefill_executions_before)
        self.assertEqual(1, second.active_decode_executions_before)
        self.assertEqual(34, second.active_scheduled_tokens_before)

        observer._finish_execution_tracking(second)
        observer._finish_execution_tracking(first)
        observer._finish_execution_tracking(first)
        self.assertEqual(0, observer._active_execution_count)
        self.assertEqual(0, observer._active_prefill_execution_count)
        self.assertEqual(0, observer._active_decode_execution_count)
        self.assertEqual(0, observer._active_scheduled_tokens)
        self.assertEqual({}, observer._inflight_scheduled_tokens)
        self.assertEqual({}, observer._scheduled_work_by_output)

    def test_stalled_running_diagnostic_is_rate_limited_and_resets_on_progress(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=100,
            computed_tokens=20,
            cached_tokens=0,
            request_id="stalled",
        )
        request.num_tokens_with_spec = 100
        request.num_output_placeholders = 0
        request.max_tokens = 8
        request.num_preemptions = 2
        request.resumable = True
        request.prompt_token_ids = [1] * 20
        request.num_encoder_inputs = 2
        scheduler = SimpleNamespace(
            max_model_len=65536,
            max_num_scheduled_tokens=4096,
            max_num_encoder_input_tokens=8,
            waiting=["queued"],
            _pause_state="unpaused",
        )
        audit = observer._audit_stalled_running_requests
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.logger.warning"
        ) as warning:
            audit(scheduler, [request], {}, 1)
            audit(scheduler, [request], {}, 25_000_000_001)
            self.assertEqual(0, warning.call_count)

            audit(scheduler, [request], {"stalled": 4}, 31_000_000_001)
            self.assertEqual(1, warning.call_count)
            fields = warning.call_args.args
            self.assertIn("[navi-vllm-stalled-running]", fields[0])
            self.assertEqual("stalled", fields[1])
            self.assertEqual(0, fields[2])
            self.assertEqual(20, fields[4])
            self.assertEqual(80, fields[8])
            self.assertEqual(4, fields[9])
            self.assertEqual(2, fields[12])
            self.assertEqual(1, fields[14])
            self.assertEqual(0, fields[17])
            self.assertEqual(0, fields[19])
            self.assertEqual(65515, fields[20])
            self.assertIn("prompt_id_len=%s", fields[0])
            self.assertEqual((True, 20, 2, 8), fields[-4:])
            audit(scheduler, [request], {}, 60_000_000_001)
            self.assertEqual(1, warning.call_count)
            audit(scheduler, [request], {}, 92_000_000_001)
            self.assertEqual(2, warning.call_count)

            request.num_computed_tokens = 40
            audit(scheduler, [request], {}, 98_000_000_001)
            audit(scheduler, [request], {}, 124_000_000_001)
            self.assertEqual(2, warning.call_count)
            audit(scheduler, [request], {}, 129_000_000_001)
            self.assertEqual(3, warning.call_count)
            self.assertEqual(60, warning.call_args.args[8])

            audit(scheduler, [], {}, 135_000_000_001)
            self.assertEqual({}, observer._running_request_progress)

    def test_stalled_running_diagnostic_bounds_logs_per_scan(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        requests = [
            self.make_request(
                prompt_tokens=100,
                computed_tokens=20,
                cached_tokens=0,
                request_id=f"stalled-{index}",
            )
            for index in range(5)
        ]
        scheduler = SimpleNamespace(waiting=[])
        audit = observer._audit_stalled_running_requests
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.logger.warning"
        ) as warning:
            audit(scheduler, requests, {}, 1)
            audit(scheduler, requests, {}, 31_000_000_001)
            self.assertEqual(4, warning.call_count)
            self.assertEqual(
                {f"stalled-{index}" for index in range(4)},
                {call.args[1] for call in warning.call_args_list},
            )
            audit(scheduler, requests, {}, 36_000_000_001)
            self.assertEqual(5, warning.call_count)
            self.assertEqual("stalled-4", warning.call_args.args[1])

    def test_slow_engine_stage_diagnostic_is_thresholded_and_rate_limited(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.logger.warning"
        ) as warning:
            observer._log_slow_engine_stage("schedule", 1, 99_000_000)
            self.assertEqual(0, warning.call_count)
            observer._log_slow_engine_stage("schedule", 1, 200_000_001, native_ms=125.0)
            self.assertEqual(1, warning.call_count)
            self.assertEqual("schedule", warning.call_args.args[1])
            self.assertIn("native_ms=125.0", warning.call_args.args[3])
            observer._log_slow_engine_stage("schedule", 300_000_000, 500_000_000)
            self.assertEqual(1, warning.call_count)
            observer._log_slow_engine_stage("schedule", 300_000_000, 1_300_000_000)
            self.assertEqual(2, warning.call_count)
            observer._log_slow_engine_stage("route_output", 300_000_000, 500_000_000)
            self.assertEqual(3, warning.call_count)
            observer._log_slow_engine_stage(
                "update_from_output", 300_000_000, 500_000_000
            )
            self.assertEqual(4, warning.call_count)
            observer._log_slow_engine_stage("schedule", 5_200_000_001, 5_400_000_001)
            self.assertEqual(4, warning.call_count)
            observer._log_slow_engine_stage("schedule", 6_400_000_001, 6_600_000_001)
            self.assertEqual(5, warning.call_count)

    def test_core_step_stage_reports_step_and_future_wait_separately(self):
        observer = object.__new__(VLLMEngineObserver)
        observer._bridge = SimpleNamespace(report_core_step=mock.Mock())

        observer._report_core_step_stage("engine_step", 250.0)
        observer._report_core_step_stage("future_result", 175.0)
        observer._report_core_step_stage("batch_queue_future_result", 125.0)

        self.assertEqual(
            [mock.call(250.0, False), mock.call(175.0, True), mock.call(125.0, True)],
            observer._bridge.report_core_step.call_args_list,
        )

    def test_admission_state_captures_only_preceding_vllm_work(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        preceding = self.make_request(
            prompt_tokens=64,
            computed_tokens=40,
            cached_tokens=32,
            request_id="preceding",
        )
        preceding.num_output_tokens = 8
        preceding.max_tokens = 20
        scheduler._scheduler.add_request(preceding)
        scheduler._scheduler.running.append(preceding)
        scheduler._scheduler.waiting = ["queued"]
        self.add_forward_batch(observer, 97, ["first", "second"])

        for request_id in ("first", "second"):
            scheduler.add_request(
                self.make_request(
                    prompt_tokens=32,
                    computed_tokens=0,
                    cached_tokens=0,
                    request_id=request_id,
                )
            )

        state = observer._forward_request_state("first")
        self.assertGreater(state.admission_started_ns, 0)
        workload = state.admission_state.preceding_workload
        self.assertEqual(1, workload.request_count)
        self.assertEqual(1, workload.remaining_prefill_request_count)
        self.assertEqual(1, workload.running_request_count)
        self.assertEqual(1, workload.waiting_request_count)
        self.assertEqual(1, workload.running_decode_request_count)
        self.assertEqual(8, workload.running_decode_generated_tokens)
        self.assertEqual(12, workload.running_decode_remaining_token_budget)
        self.assertEqual(40, workload.remaining_prefill_context_tokens)
        self.assertEqual(24, workload.remaining_prefill_tokens)
        self.assertEqual(576.0, workload.remaining_prefill_token_square_sum)
        self.assertEqual(960.0, workload.context_remaining_prefill_token_product_sum)
        self.assertEqual(40, workload.remaining_total_context_tokens)
        self.assertEqual(24, workload.remaining_total_tokens)
        self.assertEqual(576.0, workload.remaining_total_token_square_sum)
        self.assertEqual(960.0, workload.context_remaining_total_token_product_sum)
        self.assertEqual(64, workload.total_prompt_tokens)
        self.assertEqual(
            (1, 1, 40, 24, 576.0, 960.0),
            observer._prefill_workload_features(workload),
        )

        preceding.num_computed_tokens = 64
        observer._refresh_and_report_workload(
            scheduler._scheduler, [preceding.request_id], "test_progress"
        )
        self.assertEqual(
            24, state.admission_state.preceding_workload.remaining_prefill_tokens
        )

    def test_forward_arrival_captures_ready_time_and_queued_preceding_batch(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        preceding = self.make_request(
            prompt_tokens=64,
            computed_tokens=40,
            cached_tokens=32,
            request_id="preceding",
        )
        preceding.max_tokens = 20
        scheduler._scheduler.add_request(preceding)
        scheduler._scheduler.running.append(preceding)
        observer._refresh_and_report_workload(
            scheduler._scheduler, [preceding.request_id], "preceding"
        )

        input_queue = SimpleNamespace(
            items=[], put_nowait=lambda item: input_queue.items.append(item)
        )
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda request: (request, 0),
            input_queue=input_queue,
        )
        request_type = SimpleNamespace(ADD="add", ABORT="abort")
        first_request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=-1,
            request_id="first",
        )
        first_request.max_tokens = 8
        first_batch = FakeForwardBatch(201, ["first"], [16])
        first_batch.ready_time_ns = time.monotonic_ns() - 5_000_000
        observer._enqueue_forward_batch(
            first_batch,
            [(stable_request_id("first"), first_request)],
            request_type,
            observer._capture_arrival_workload(),
            first_batch.ready_time_ns,
        )

        second_request = self.make_request(
            prompt_tokens=16,
            computed_tokens=0,
            cached_tokens=-1,
            request_id="second",
        )
        second_batch = FakeForwardBatch(202, ["second"], [0])
        second_batch.ready_time_ns = time.monotonic_ns() - 2_000_000
        observer._enqueue_forward_batch(
            second_batch,
            [(stable_request_id("second"), second_request)],
            request_type,
            observer._capture_arrival_workload(),
            second_batch.ready_time_ns,
        )

        first_state = observer._pending_forward_batches[stable_request_id("first")]
        second_state = observer._pending_forward_batches[stable_request_id("second")]
        self.assertEqual(first_batch.ready_time_ns, first_state.admission_started_ns)
        self.assertEqual(second_batch.ready_time_ns, second_state.admission_started_ns)
        self.assertEqual(
            1, first_state.admission_state.preceding_workload.request_count
        )
        workload = second_state.admission_state.preceding_workload
        self.assertEqual(2, workload.request_count)
        self.assertEqual(2, workload.remaining_prefill_request_count)
        self.assertEqual(1, workload.running_request_count)
        self.assertEqual(1, workload.waiting_request_count)
        self.assertEqual(56, workload.remaining_prefill_context_tokens)
        self.assertEqual(40, workload.remaining_prefill_tokens)
        self.assertEqual(832.0, workload.remaining_prefill_token_square_sum)
        self.assertEqual(1216.0, workload.context_remaining_prefill_token_product_sum)
        self.assertEqual(56, workload.remaining_total_context_tokens)
        self.assertEqual(40, workload.remaining_total_tokens)
        self.assertEqual(832.0, workload.remaining_total_token_square_sum)
        self.assertEqual(1216.0, workload.context_remaining_total_token_product_sum)
        self.assertEqual(96, workload.total_prompt_tokens)

    def test_workload_partitions_prefill_by_running_capacity(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        requests = [
            self.make_request(
                prompt_tokens=100,
                computed_tokens=20,
                cached_tokens=20,
                request_id="running-prefill",
            ),
            self.make_request(
                prompt_tokens=50,
                computed_tokens=50,
                cached_tokens=0,
                request_id="running-decode",
            ),
            self.make_request(
                prompt_tokens=64,
                computed_tokens=16,
                cached_tokens=16,
                request_id="capacity-waiting",
            ),
            self.make_request(
                prompt_tokens=32,
                computed_tokens=8,
                cached_tokens=8,
                request_id="overflow-waiting",
            ),
        ]
        scheduler._scheduler.max_num_running_reqs = 3
        scheduler._scheduler.requests = {
            request.request_id: request for request in requests
        }
        scheduler._scheduler.running = requests[:2]
        scheduler._scheduler.waiting = requests[2:]

        observer._refresh_and_report_workload(
            scheduler._scheduler,
            scheduler._scheduler.requests,
            "capacity_partition",
        )

        workload = observer._current_workload
        self.assertTrue(workload.valid())
        self.assertEqual(3, workload.max_running_request_count)
        self.assertEqual(3, workload.running_capacity_slot_count)
        self.assertEqual(
            (2, 36, 128, 8704.0, 2368.0),
            (
                workload.running_capacity_prefill.request_count,
                workload.running_capacity_prefill.context_tokens,
                workload.running_capacity_prefill.compute_tokens,
                workload.running_capacity_prefill.compute_token_square_sum,
                workload.running_capacity_prefill.context_compute_token_product_sum,
            ),
        )
        self.assertEqual(
            (1, 8, 24, 576.0, 192.0),
            (
                workload.waiting_capacity_prefill.request_count,
                workload.waiting_capacity_prefill.context_tokens,
                workload.waiting_capacity_prefill.compute_tokens,
                workload.waiting_capacity_prefill.compute_token_square_sum,
                workload.waiting_capacity_prefill.context_compute_token_product_sum,
            ),
        )
        self.assertEqual(
            (1, 20, 80, 6400.0, 1600.0),
            (
                workload.scheduler_running_prefill.request_count,
                workload.scheduler_running_prefill.context_tokens,
                workload.scheduler_running_prefill.compute_tokens,
                workload.scheduler_running_prefill.compute_token_square_sum,
                workload.scheduler_running_prefill.context_compute_token_product_sum,
            ),
        )
        self.assertEqual(
            (2, 24, 72, 2880.0, 960.0),
            (
                workload.scheduler_waiting_prefill.request_count,
                workload.scheduler_waiting_prefill.context_tokens,
                workload.scheduler_waiting_prefill.compute_tokens,
                workload.scheduler_waiting_prefill.compute_token_square_sum,
                workload.scheduler_waiting_prefill.context_compute_token_product_sum,
            ),
        )

        pending_state = self.add_forward_batch(observer, 205, ["pending"], [4])
        pending_request = self.make_request(
            prompt_tokens=20,
            computed_tokens=0,
            cached_tokens=-1,
            request_id="pending",
        )
        observer._set_admission_state(
            pending_state,
            [pending_request],
            workload,
            time.perf_counter_ns(),
        )
        arrival = observer._capture_arrival_workload()
        self.assertTrue(arrival.valid())
        self.assertEqual(3, arrival.running_capacity_slot_count)
        self.assertEqual(
            workload.running_capacity_prefill, arrival.running_capacity_prefill
        )
        self.assertEqual(
            (2, 12, 40, 832.0, 256.0),
            (
                arrival.waiting_capacity_prefill.request_count,
                arrival.waiting_capacity_prefill.context_tokens,
                arrival.waiting_capacity_prefill.compute_tokens,
                arrival.waiting_capacity_prefill.compute_token_square_sum,
                arrival.waiting_capacity_prefill.context_compute_token_product_sum,
            ),
        )
        self.assertEqual(
            workload.scheduler_running_prefill,
            arrival.scheduler_running_prefill,
        )
        self.assertEqual(
            (3, 28, 88, 3136.0, 1024.0),
            (
                arrival.scheduler_waiting_prefill.request_count,
                arrival.scheduler_waiting_prefill.context_tokens,
                arrival.scheduler_waiting_prefill.compute_tokens,
                arrival.scheduler_waiting_prefill.compute_token_square_sum,
                arrival.scheduler_waiting_prefill.context_compute_token_product_sum,
            ),
        )

        nonfull_with_waiting = replace(
            workload,
            running_capacity_slot_count=2,
        )
        self.assertFalse(nonfull_with_waiting.valid())

        running_prefill_exceeds_slots = replace(
            workload,
            running_capacity_slot_count=1,
            waiting_capacity_prefill=replace(
                workload.waiting_capacity_prefill,
                request_count=0,
                context_tokens=0,
                compute_tokens=0,
                compute_token_square_sum=0.0,
                context_compute_token_product_sum=0.0,
            ),
            remaining_prefill_request_count=2,
            remaining_prefill_context_tokens=36,
            remaining_prefill_tokens=128,
            remaining_prefill_token_square_sum=8704.0,
            context_remaining_prefill_token_product_sum=2368.0,
        )
        self.assertFalse(running_prefill_exceeds_slots.valid())

    def test_running_elapsed_tracks_current_running_requests(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        requests = [
            self.make_request(
                prompt_tokens=32,
                computed_tokens=8,
                cached_tokens=8,
                request_id="running-0",
            ),
            self.make_request(
                prompt_tokens=32,
                computed_tokens=8,
                cached_tokens=8,
                request_id="running-1",
            ),
        ]
        queued = self.make_request(
            prompt_tokens=32, computed_tokens=0, cached_tokens=0, request_id="queued"
        )
        scheduler._scheduler.requests = {
            request.request_id: request for request in requests + [queued]
        }
        scheduler._scheduler.waiting = [queued]
        scheduler._scheduler.running = list(requests)

        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
            return_value=1_000_000_000,
        ):
            observer._refresh_and_report_workload(
                scheduler._scheduler,
                scheduler._scheduler.requests,
                "running_started",
            )
        self.assertEqual(0.0, observer._current_workload.running_elapsed_time_ms)
        self.assertEqual(
            3, observer._current_workload.running_capacity_prefill.request_count
        )
        self.assertEqual(2, observer._bridge.workload_elapsed_request_counts[-1])

        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
            return_value=1_250_000_000,
        ):
            arrival = observer._capture_arrival_workload()
        self.assertEqual(500.0, arrival.running_elapsed_time_ms)

        scheduler._scheduler.running = requests[:1]
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
            return_value=1_600_000_000,
        ):
            observer._refresh_and_report_workload(
                scheduler._scheduler, (), "one_running"
            )
        self.assertEqual(600.0, observer._current_workload.running_elapsed_time_ms)
        self.assertEqual(600.0, observer._bridge.workloads[-1][33])
        self.assertEqual(1, observer._bridge.workload_elapsed_request_counts[-1])

        self.assertFalse(
            replace(
                observer._current_workload,
                running_request_count=0,
                running_elapsed_time_ms=1.0,
            ).valid()
        )

    def test_running_elapsed_excludes_decode_only_requests(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        prefill = self.make_request(
            prompt_tokens=32,
            computed_tokens=8,
            cached_tokens=8,
            request_id="prefill",
        )
        decode = self.make_request(
            prompt_tokens=16,
            computed_tokens=17,
            cached_tokens=0,
            request_id="decode",
        )
        scheduler._scheduler.requests = {
            request.request_id: request for request in (prefill, decode)
        }
        scheduler._scheduler.running = [prefill, decode]

        with (
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
                return_value=1_000_000_000,
            ),
            self.assertLogs("vllm.navi_sched.agent", level="DEBUG") as logs,
        ):
            observer._refresh_and_report_workload(
                scheduler._scheduler,
                scheduler._scheduler.requests,
                "running_started",
            )
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
            return_value=1_250_000_000,
        ):
            arrival = observer._capture_arrival_workload()

        self.assertEqual(2, arrival.running_request_count)
        self.assertEqual(1, arrival.scheduler_running_prefill.request_count)
        self.assertEqual(250.0, arrival.running_elapsed_time_ms)
        self.assertEqual(2, observer._bridge.workloads[-1][4])
        self.assertEqual(1, observer._bridge.workloads[-1][12])
        self.assertIn(
            "running_total=2 running_prefill=1 running_decode=1",
            "\n".join(logs.output),
        )

    def test_workload_reports_running_decode_progress(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        requests = [
            self.make_request(
                prompt_tokens=16,
                computed_tokens=16 + output_tokens,
                cached_tokens=0,
                request_id=f"decode-{output_tokens}",
            )
            for output_tokens in (10, 20)
        ]
        for request, output_tokens in zip(requests, (10, 20), strict=True):
            request.num_output_tokens = output_tokens
            request.max_tokens = 80
        scheduler._scheduler.requests = {
            request.request_id: request for request in requests
        }
        scheduler._scheduler.running = list(requests)

        observer._refresh_and_report_workload(
            scheduler._scheduler,
            scheduler._scheduler.requests,
            "decode_progress",
        )

        workload = observer._current_workload
        self.assertEqual(2, workload.running_decode_request_count)
        self.assertEqual(30, workload.running_decode_generated_tokens)
        self.assertEqual(130, workload.running_decode_remaining_token_budget)
        self.assertEqual(2, observer._bridge.workloads[-1][34])
        self.assertEqual(30, observer._bridge.workloads[-1][35])
        self.assertEqual(130, observer._bridge.workloads[-1][36])

        report_count = len(observer._bridge.workloads)
        requests[0].num_output_tokens += 1
        observer._refresh_and_report_workload(
            scheduler._scheduler,
            scheduler._scheduler.requests,
            "decode_progress_advanced",
        )
        self.assertEqual(report_count + 1, len(observer._bridge.workloads))
        self.assertEqual(31, observer._bridge.workloads[-1][35])
        self.assertEqual(129, observer._bridge.workloads[-1][36])

    def test_workload_reads_decode_progress_once_per_running_request(self):
        class CountedRequest:
            def __init__(self, request, output_tokens):
                self._request = request
                self._output_tokens = output_tokens
                self.output_token_reads = 0
                self.max_token_reads = 0

            @property
            def num_output_tokens(self):
                self.output_token_reads += 1
                return self._output_tokens

            @property
            def max_tokens(self):
                self.max_token_reads += 1
                return 100

            def __getattr__(self, name):
                return getattr(self._request, name)

        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        requests = [
            CountedRequest(
                self.make_request(
                    prompt_tokens=16,
                    computed_tokens=16 + output_tokens,
                    cached_tokens=0,
                    request_id=f"decode-{output_tokens}",
                ),
                output_tokens,
            )
            for output_tokens in (7, 13)
        ]
        waiting = CountedRequest(
            self.make_request(
                prompt_tokens=16,
                computed_tokens=0,
                cached_tokens=0,
                request_id="waiting",
            ),
            99,
        )
        scheduler._scheduler.requests = {
            request.request_id: request for request in (*requests, waiting)
        }
        scheduler._scheduler.running = list(requests)
        scheduler._scheduler.waiting = [waiting]

        observer._refresh_and_report_workload(
            scheduler._scheduler,
            scheduler._scheduler.requests,
            "decode_progress_read_count",
        )

        self.assertEqual([1, 1], [request.output_token_reads for request in requests])
        self.assertEqual([1, 1], [request.max_token_reads for request in requests])
        self.assertEqual(0, waiting.output_token_reads)
        self.assertEqual(0, waiting.max_token_reads)
        self.assertEqual(2, observer._current_workload.running_decode_request_count)
        self.assertEqual(20, observer._current_workload.running_decode_generated_tokens)
        self.assertEqual(
            180,
            observer._current_workload.running_decode_remaining_token_budget,
        )

    def test_running_elapsed_stops_when_prefill_completes(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=8,
            cached_tokens=8,
            request_id="request",
        )
        scheduler._scheduler.requests = {request.request_id: request}
        scheduler._scheduler.running = [request]

        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
            return_value=1_000_000_000,
        ):
            observer._refresh_and_report_workload(
                scheduler._scheduler, (request.request_id,), "prefill_started"
            )
        request.num_computed_tokens = request.num_prompt_tokens
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
            return_value=1_250_000_000,
        ):
            observer._refresh_and_report_workload(
                scheduler._scheduler, (request.request_id,), "prefill_completed"
            )

        workload = observer._current_workload
        self.assertEqual(1, workload.running_request_count)
        self.assertEqual(0, workload.scheduler_running_prefill.request_count)
        self.assertEqual(0.0, workload.running_elapsed_time_ms)
        self.assertEqual({}, observer._running_prefill_started_ns)
        self.assertEqual(1, observer._bridge.workloads[-1][4])
        self.assertEqual(0, observer._bridge.workloads[-1][12])
        self.assertFalse(replace(workload, running_elapsed_time_ms=1.0).valid())

    def test_running_elapsed_resets_after_preemption_and_reentry(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=8,
            cached_tokens=8,
            request_id="request",
        )
        scheduler._scheduler.requests = {request.request_id: request}
        scheduler._scheduler.running = [request]

        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
            return_value=1_000_000_000,
        ):
            observer._refresh_and_report_workload(
                scheduler._scheduler, (request.request_id,), "prefill_started"
            )

        request.num_preemptions = 1
        scheduler._scheduler.running = []
        scheduler._scheduler.waiting = [request]
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
            return_value=1_300_000_000,
        ):
            observer._refresh_and_report_workload(
                scheduler._scheduler, (request.request_id,), "preempted"
            )
        self.assertEqual(0.0, observer._current_workload.running_elapsed_time_ms)
        self.assertEqual({}, observer._running_prefill_started_ns)

        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
            return_value=1_600_000_000,
        ):
            waiting = observer._capture_arrival_workload()
        self.assertEqual(0.0, waiting.running_elapsed_time_ms)

        scheduler._scheduler.waiting = []
        scheduler._scheduler.running = [request]
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
            return_value=1_800_000_000,
        ):
            observer._refresh_and_report_workload(
                scheduler._scheduler, (request.request_id,), "resumed"
            )
        self.assertEqual(0.0, observer._current_workload.running_elapsed_time_ms)

        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
            return_value=2_000_000_000,
        ):
            arrival = observer._capture_arrival_workload()
        self.assertEqual(200.0, arrival.running_elapsed_time_ms)

    def test_pp_known_cache_snapshot_tracks_the_first_forward_result(self):
        for initial_hit, progress, actual in [(0, 0, 70), (20, 0, 70), (0, 60, 20)]:
            with self.subTest(
                initial_hit=initial_hit, progress=progress, actual=actual
            ):
                observer = VLLMEngineObserver(
                    self.make_config(), self.make_kv_cache_config()
                )
                observer._pp_capacity_unified = True
                state = self.add_forward_batch(
                    observer, 1204, ["prior-run", "prior-wait"], [0, 0]
                )
                requests = [
                    self.make_request(
                        prompt_tokens=100,
                        computed_tokens=progress if index == 0 else 0,
                        cached_tokens=initial_hit,
                        request_id=name,
                    )
                    for index, name in enumerate(["prior-run", "prior-wait"])
                ]
                for request in requests:
                    request.prefill_stats = SimpleNamespace(
                        num_prompt_tokens=100,
                        num_local_cached_tokens=initial_hit,
                        num_external_cached_tokens=0,
                    )
                    self.assertTrue(observer._native_cache_lookup_complete(request))
                observer._set_admission_state(state, requests, _ObservedWorkload(), 100)
                for request in requests:
                    observer._admit_request(request)
                native = SimpleNamespace(
                    requests={r.request_id: r for r in requests},
                    running=[requests[0]],
                    waiting=[requests[1]],
                    max_num_running_reqs=1,
                )
                observer._refresh_workload_contributions(native, native.requests, {})
                snapshot = observer._workload_snapshot(native, {}, 0)
                self.assertEqual(2, len(snapshot.prior_cache_forecasts))
                state.cache_lookups[stable_request_id("prior-run")].result = (
                    _ResolvedCacheHit(actual, 150)
                )
                state.cache_lookups[stable_request_id("prior-wait")].result = (
                    _ResolvedCacheHit(50, 150)
                )
                admission = _AdmissionState(preceding_workload=snapshot)
                early_running, early_waiting = observer._capacity_sample_features(
                    admission, [], cache_resolution_deadline_ns=149
                )
                self.assertEqual(
                    max(progress, initial_hit), early_running.context_tokens
                )
                self.assertEqual(initial_hit, early_waiting.context_tokens)
                running, waiting = observer._capacity_sample_features(
                    admission, [], cache_resolution_deadline_ns=200
                )
                self.assertEqual(max(progress, actual), running.context_tokens)
                self.assertEqual(100 - max(progress, actual), running.compute_tokens)
                self.assertEqual(
                    (50, 50), (waiting.context_tokens, waiting.compute_tokens)
                )
                self.assertEqual(
                    max(progress, initial_hit),
                    snapshot.running_capacity_prefill.context_tokens,
                )
                observer._pp_capacity_unified = False
                self.assertIsNone(
                    observer._make_workload_contribution(requests[0], 0, None).cache_lookup
                )

    def test_pp_waiting_cache_is_refreshed_until_first_forward(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._pp_capacity_unified = True
        state = self.add_forward_batch(observer, 1205, ["prior-wait"], [0])
        request = self.make_request(
            prompt_tokens=100,
            computed_tokens=0,
            cached_tokens=0,
            request_id="prior-wait",
        )
        request.prefill_stats = SimpleNamespace(
            num_prompt_tokens=100,
            num_local_cached_tokens=0,
            num_external_cached_tokens=0,
        )
        observer._set_admission_state(state, [request], _ObservedWorkload(), 100)
        observer._admit_request(request)
        native = SimpleNamespace(
            requests={request.request_id: request},
            running=[],
            waiting=[request],
            max_num_running_reqs=1,
        )
        observer._refresh_workload_contributions(native, native.requests, {})
        original = observer._workload_snapshot(native, {}, 0)
        self.assertIn(request.request_id, observer._pending_cache_lookup_request_ids)
        request.prefill_stats.num_external_cached_tokens = 20
        observer._refresh_workload_contributions(native, (), {})
        refreshed = observer._workload_snapshot(native, {}, 0)
        self.assertEqual(20, refreshed.running_capacity_prefill.context_tokens)
        self.assertEqual(80, refreshed.running_capacity_prefill.compute_tokens)
        self.assertEqual(0, original.running_capacity_prefill.context_tokens)
        state.cache_lookups[stable_request_id(request.request_id)].result = (
            _ResolvedCacheHit(20, 150)
        )
        observer._refresh_workload_contributions(native, (), {})
        self.assertNotIn(request.request_id, observer._pending_cache_lookup_request_ids)

    def test_pp_native_snapshot_keeps_results_after_prior_batch_cleanup(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._pp_capacity_unified = True
        state = self.add_forward_batch(
            observer, 1203, ["prior-run", "prior-wait"], [20, 10]
        )
        requests = [
            self.make_request(
                prompt_tokens=100, computed_tokens=0, cached_tokens=-1, request_id=name
            )
            for name in ["prior-run", "prior-wait"]
        ]
        observer._set_admission_state(state, requests, _ObservedWorkload(), 100)
        for request in requests:
            observer._admit_request(request)
        native = SimpleNamespace(
            requests={r.request_id: r for r in requests},
            running=[requests[0]],
            waiting=[requests[1]],
            max_num_running_reqs=1,
        )
        observer._refresh_workload_contributions(native, native.requests, {})
        snapshot = observer._workload_snapshot(native, {}, 0)
        self.assertEqual(
            [True, False], [p.running_capacity for p in snapshot.prior_cache_forecasts]
        )
        state.cache_lookups[stable_request_id("prior-run")].result = _ResolvedCacheHit(
            70, 150
        )
        state.cache_lookups[stable_request_id("prior-wait")].result = _ResolvedCacheHit(
            50, 150
        )
        observer._forward_request_states.clear()
        state.cache_lookups.clear()
        running, waiting = observer._capacity_sample_features(
            _AdmissionState(preceding_workload=snapshot),
            [],
            cache_resolution_deadline_ns=200,
        )
        self.assertEqual((70, 30), (running.context_tokens, running.compute_tokens))
        self.assertEqual((50, 50), (waiting.context_tokens, waiting.compute_tokens))
        self.assertEqual(
            (20, 80),
            (
                snapshot.running_capacity_prefill.context_tokens,
                snapshot.running_capacity_prefill.compute_tokens,
            ),
        )

    def test_non_pp_prepared_work_does_not_capture_prior_cache_results(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        self.assertFalse(observer._pp_capacity_unified)
        state = self.add_forward_batch(observer, 1204, ["unchanged-legacy"], [20])
        request = self.make_request(
            prompt_tokens=100,
            computed_tokens=0,
            cached_tokens=-1,
            request_id="unchanged-legacy",
        )
        base = _ObservedWorkload(max_running_request_count=1)
        observer._set_admission_state(state, [request], base, 100)
        snapshot = observer._workload_with_pending_admissions_locked(base)
        self.assertEqual({}, state.cache_lookups)
        self.assertEqual((), snapshot.prior_cache_forecasts)
        self.assertEqual(
            (20, 80),
            (
                snapshot.running_capacity_prefill.context_tokens,
                snapshot.running_capacity_prefill.compute_tokens,
            ),
        )

    def test_pp_prepared_snapshot_retains_cache_lookup_until_training(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._pp_capacity_unified = True
        state = self.add_forward_batch(observer, 1201, ["prior-a", "prior-b"], [20, 10])
        requests = [
            self.make_request(
                prompt_tokens=100, computed_tokens=0, cached_tokens=-1, request_id=name
            )
            for name in ["prior-a", "prior-b"]
        ]
        base = _ObservedWorkload(max_running_request_count=1)
        observer._set_admission_state(state, requests, base, 100)
        snapshot = observer._workload_with_pending_admissions_locked(base)
        self.assertEqual(
            [True, False], [p.running_capacity for p in snapshot.prior_cache_forecasts]
        )
        lookup = state.cache_lookups[stable_request_id("prior-a")]
        lookup.result = _ResolvedCacheHit(70, 150)
        running, waiting = observer._capacity_sample_features(
            _AdmissionState(preceding_workload=snapshot),
            [],
            cache_resolution_deadline_ns=200,
        )
        self.assertEqual((70, 30), (running.context_tokens, running.compute_tokens))
        self.assertEqual((10, 90), (waiting.context_tokens, waiting.compute_tokens))
        self.assertEqual(20, snapshot.running_capacity_prefill.context_tokens)
        self.assertEqual(80, snapshot.running_capacity_prefill.compute_tokens)

    def test_first_forward_publishes_initial_cache_result_for_prior_snapshots(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=16, computed_tokens=16, cached_tokens=8
        )
        state = self.add_forward_batch(observer, 1202, [request.request_id], [0])
        observer._admit_request(request)
        request_id = stable_request_id(request.request_id)
        state.cache_lookups[request_id] = _RequestCacheLookup(request_id)
        observer._prepare_execution_step(
            SimpleNamespace(num_scheduled_tokens={request.request_id: 8}),
            {request.request_id: request}.get,
            _ExecutionStep(started_ns=100),
        )
        self.assertEqual(
            _ResolvedCacheHit(8, 100), state.cache_lookups[request_id].result
        )

    def test_pp_finished_registry_request_does_not_occupy_capacity(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._pp_capacity_unified = True
        scheduler = observer.create_scheduler(LifecycleScheduler)._scheduler
        finished = self.make_request(
            prompt_tokens=32,
            computed_tokens=32,
            cached_tokens=32,
            request_id="finished-awaiting-kv-cleanup",
        )
        finished.is_finished = lambda: False
        scheduler.max_num_running_reqs = 1
        scheduler.requests = {finished.request_id: finished}
        scheduler.running = []
        scheduler.waiting = []
        observer._refresh_and_report_workload(
            scheduler, scheduler.requests, "live_zero_remaining_registry"
        )
        self.assertEqual(1, observer._current_workload.request_count)
        self.assertEqual(1, observer._current_workload.running_capacity_slot_count)
        finished.is_finished = lambda: True
        observer._refresh_and_report_workload(
            scheduler, scheduler.requests, "terminal_registry"
        )
        base = observer._current_workload
        self.assertEqual(0, base.request_count)
        self.assertEqual(0, base.batch_count)
        self.assertEqual(0, base.running_capacity_slot_count)
        aggregate = observer._workload_aggregate
        observer._refresh_and_report_workload(scheduler, [], "terminal_registry_again")
        self.assertIs(aggregate, observer._workload_aggregate)

        observer._native_handoff_batches[707] = None
        captured = observer._capture_arrival_workload()
        self.assertEqual((707,), captured.native_handoff_batch_ids)
        incoming = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=0,
            request_id="live-incoming",
        )
        state = self.add_forward_batch(observer, 708, [incoming.request_id], [0])
        observer._set_admission_state(
            state, [incoming], captured, time.perf_counter_ns()
        )
        self.assertEqual(
            frozenset({stable_request_id(incoming.request_id)}),
            state.admission_state.incoming_running_capacity_request_ids,
        )
        self.assertIn(finished.request_id, scheduler.requests)
        observer.shutdown()

    def test_non_pp_keeps_legacy_finished_registry_accounting(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)._scheduler
        finished = self.make_request(
            prompt_tokens=32,
            computed_tokens=32,
            cached_tokens=32,
            request_id="legacy-finished",
        )
        finished.is_finished = lambda: True
        scheduler.max_num_running_reqs = 1
        scheduler.requests = {finished.request_id: finished}
        scheduler.running = []
        scheduler.waiting = []
        observer._refresh_and_report_workload(
            scheduler, scheduler.requests, "legacy_terminal_registry"
        )
        self.assertEqual(1, observer._current_workload.request_count)
        self.assertEqual(1, observer._current_workload.running_capacity_slot_count)
        observer.shutdown()

    @staticmethod
    def make_v6d_query_state(**values):
        fields = dict(
            match_future=None,
            match_generation=0,
            match_signature=None,
            resolved_match=None,
            resolved_match_signature=None,
            load_inflight=False,
            load_failed=False,
        )
        fields.update(values)
        return SimpleNamespace(**fields)

    def test_pp_native_query_report_is_frozen_and_independent_of_debug_audits(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._pp_capacity_unified = True
        observer._native_request_state_audit_enabled = False
        events = []
        observer.on_request_lifecycle_event = events.append
        scheduler = observer.create_scheduler(LifecycleScheduler)._scheduler
        request = self.make_request(
            prompt_tokens=32768,
            computed_tokens=16384,
            cached_tokens=-1,
            request_id="external-query",
        )
        request.is_finished = lambda: False
        request.status = SimpleNamespace(name="WAITING")
        future = Future()
        state = self.make_v6d_query_state(
            match_future=future,
            match_generation=2,
            match_signature=(4096, ("native-hash",)),
        )
        observer._v6d_scheduler = SimpleNamespace(
            _request_states={request.request_id: state}
        )
        scheduler.requests = {request.request_id: request}
        scheduler.running = []
        scheduler.waiting = [request]
        scheduler.max_num_running_reqs = 64
        with (
            mock.patch.object(
                future, "result", side_effect=AssertionError("no query result")
            ),
            mock.patch.object(
                observer,
                "_native_v6d_request_audit",
                wraps=observer._native_v6d_request_audit,
            ) as capture,
        ):
            observer._refresh_and_report_workload(
                scheduler, scheduler.requests, "query-input"
            )
        self.assertEqual(1, capture.call_count)
        workload = observer._current_workload
        self.assertEqual((), workload.native_prefill_phases)
        self.assertEqual((), workload.native_prefill_queue_entries)
        self.assertFalse(any(e.get("event") == "native_request_state" for e in events))
        self.assertEqual(1, len(workload.native_cache_queries))
        query = workload.native_cache_queries[0]
        self.assertEqual(observer._navi_request_id(request), query.request_id)
        self.assertEqual(
            (2, 4096, 1, 1, 1, 0),
            (
                query.generation,
                query.local_prefix_tokens,
                query.input_source,
                query.future_state,
                query.load_state,
                query.receive_state,
            ),
        )
        future.set_result((8192, True))
        request.num_computed_tokens = 24576
        self.assertTrue(
            observer._report_workload_locked(workload, "frozen-query-input")
        )
        frozen = observer._bridge.native_cache_query_reports[-1][0]
        self.assertEqual(4096, frozen.local_prefix_tokens)
        self.assertEqual(1, frozen.future_state)
        observer._v6d_scheduler = None
        observer.shutdown()

    def test_pp_without_v6d_does_not_create_native_query_reports(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._pp_capacity_unified = True
        observer._native_request_state_audit_enabled = False
        scheduler = observer.create_scheduler(LifecycleScheduler)._scheduler
        request = self.make_request(
            prompt_tokens=32, computed_tokens=0, cached_tokens=0
        )
        scheduler.requests = {request.request_id: request}
        scheduler.running = []
        scheduler.waiting = [request]
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.make_native_cache_query_report",
            side_effect=AssertionError("V6D disabled"),
        ):
            observer._refresh_and_report_workload(
                scheduler, scheduler.requests, "no-v6d-query"
            )
        self.assertEqual((), observer._current_workload.native_cache_queries)
        observer.shutdown()

    @staticmethod
    def native_pp_test_module():
        from collections import namedtuple

        module = SimpleNamespace(
            _VLLMNativeCacheQuery=lambda **fields: SimpleNamespace(**fields)
        )
        for name, fields in {
            "Span": "request_id computed_tokens scheduled_tokens",
            "Slot": "ordinal dispatch_age_us expected_forward_ms last_stage_reply_age_us features spans",
            "Result": "scheduler_output_id dispatch_age_us slot_ordinal execute_reply_ready result_ready load_request_ids",
            "Request": "request_id queue queue_index phase prompt_tokens computed_tokens cache_hit_tokens shared_prefix_boundary query_age_us load_age_us prefill_slot_release",
            "Decision": "scheduler_output_id features spans load_request_ids",
        }.items():
            setattr(module, "_VLLMNativePP" + name, namedtuple(name, fields))

        class State(
            namedtuple(
                "State",
                "observation_id burst wave_open last_real_ordinal measured_forward_ms slots results requests pending_decision occupied_request_slot_count",
            )
        ):
            def with_prepared_requests(self, prepared):
                ids = [r.request_id for r in self.requests + prepared]
                if len(ids) != len(set(ids)):
                    raise ValueError("duplicate Native/prepared request")
                return self._replace(requests=self.requests + prepared)

        module._VLLMNativePPState = State
        return module

    @staticmethod
    def bind_native_pp_test_reader(observer, scheduler=None):
        from collections import deque

        stepper = SimpleNamespace(
            stages=4,
            wave=SimpleNamespace(open=False),
            in_flight=deque(),
            slots=[],
            first_slot=0,
            last_real=-1,
            measured=0.125,
            core=SimpleNamespace(
                scheduler=SimpleNamespace(running=[], num_waiting_for_streaming_input=0)
            ),
        )
        if scheduler is not None:
            stepper.core.scheduler = scheduler
            scheduler.num_waiting_for_streaming_input = 0
        observer._native_pipeline_state = NativePipelineState(
            stepper,
            SimpleNamespace(
                pcp_size=8, interleave_size=256, compile_sizes=[256, 512, 1024]
            ),
        )
        return observer._native_pipeline_state

    def test_native_pp_metadata_decision_keeps_load_request_ids(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)._scheduler
        reader = self.bind_native_pp_test_reader(observer, scheduler)
        request = self.make_request(
            prompt_tokens=16384, computed_tokens=0, cached_tokens=0, request_id="1007"
        )
        request.client_index = 1007
        scheduler.requests[request.request_id] = request
        observer._v6d_scheduler = object()
        observer._v6d_metadata_path = ()
        metadata = SimpleNamespace(reqs_to_load={request.request_id: object()})
        output = SimpleNamespace(
            num_scheduled_tokens={}, kv_connector_metadata=metadata
        )
        reader.dispatch_active = True
        observer._register_scheduled_work(output, scheduler)
        self.assertEqual((1007,), reader.pending_decision[3])
        future = Future()
        step = SimpleNamespace(
            scheduler_output=output,
            slot=None,
            exec_future=future,
            future=future,
            dispatched=time.monotonic(),
        )
        reader._stepper.in_flight.append(step)
        reader.record_dispatch(None, load_request_ids=reader.pending_decision[3])
        reader.pending_decision = None
        reader.dispatch_active = False
        metadata.reqs_to_load.clear()
        snapshot = reader.capture()
        self.assertEqual((1007,), snapshot.results[0].load_request_ids)
        self.assertEqual((), snapshot.slots)
        value, _ = reader.make_report(self.native_pp_test_module(), [])
        self.assertEqual((1007,), value.results[0].load_request_ids)
        self.assertEqual(-1, value.results[0].slot_ordinal)
        reader._stepper.in_flight.clear()
        reader.capture()
        self.assertEqual({}, reader._load_requests_by_output)
        self.assertEqual((1007,), snapshot.results[0].load_request_ids)
        observer.shutdown()

    def test_native_pp_control_execute_registration_preserves_first_decision(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        reader = self.bind_native_pp_test_reader(observer)
        output = SimpleNamespace(
            num_scheduled_tokens={}, kv_connector_metadata=object()
        )
        reader.dispatch_active = True
        observer._register_scheduled_work(output, reader._stepper.core.scheduler)
        first = reader.pending_decision
        self.assertFalse(reader.failed)
        # _execute_model takes this fallback because a zero-token output has no scheduled work entry.
        observer._register_scheduled_work(output)
        self.assertFalse(reader.failed)
        self.assertIs(first, reader.pending_decision)
        observer.shutdown()

    def test_native_pp_without_v6d_does_not_read_load_metadata(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        reader = self.bind_native_pp_test_reader(observer)

        class UnreadableMetadata:
            @property
            def reqs_to_load(self):
                raise AssertionError("V6D-disabled path read load metadata")

        output = SimpleNamespace(
            num_scheduled_tokens={}, kv_connector_metadata=UnreadableMetadata()
        )
        reader.dispatch_active = True
        observer._register_scheduled_work(output, reader._stepper.core.scheduler)
        self.assertEqual((), reader.pending_decision[3])
        self.assertFalse(reader.failed)
        observer.shutdown()

    def test_native_pp_multiconnector_metadata_uses_bound_path(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)._scheduler
        reader = self.bind_native_pp_test_reader(observer, scheduler)
        request = self.make_request(
            prompt_tokens=16384, computed_tokens=0, cached_tokens=0, request_id="1007"
        )
        request.client_index = 1007
        scheduler.requests[request.request_id] = request
        observer._v6d_scheduler = object()
        observer._v6d_metadata_path = (1, 0)
        metadata = SimpleNamespace(
            metadata=(
                object(),
                SimpleNamespace(
                    metadata=(
                        SimpleNamespace(reqs_to_load={request.request_id: object()}),
                    )
                ),
            )
        )
        output = SimpleNamespace(
            num_scheduled_tokens={}, kv_connector_metadata=metadata
        )
        reader.dispatch_active = True
        observer._register_scheduled_work(output, scheduler)
        self.assertEqual((1007,), reader.pending_decision[3])
        self.assertFalse(reader.failed)
        observer.shutdown()

    def completed_native_query_fixture(self, *, prefix=8192, hit=4096):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._pp_capacity_unified = True
        observer._native_request_state_audit_enabled = False
        scheduler = observer.create_scheduler(LifecycleScheduler)._scheduler
        reader = self.bind_native_pp_test_reader(observer, scheduler)
        request = self.make_request(
            prompt_tokens=32768,
            computed_tokens=0,
            cached_tokens=-1,
            request_id="completed-query",
        )
        request.max_tokens = 1
        request.resumable = False
        request.status = SimpleNamespace(name="WAITING")
        request.prefill_stats = SimpleNamespace(
            num_prompt_tokens=0, num_local_cached_tokens=0, num_external_cached_tokens=0
        )
        signature = (prefix, ("opaque-native-key",))
        future = Future()
        outcome = SimpleNamespace(
            generation=1,
            match_signature=signature,
            hit_length=hit,
            fetch_failed=False,
            lookup_results={"owned-native-object": object()},
        )
        state = self.make_v6d_query_state(
            match_future=future,
            match_generation=1,
            match_signature=signature,
            match_first_poll_at=time.monotonic() - 0.1,
            match_latency_recorded=False,
        )
        observer._v6d_scheduler = SimpleNamespace(
            _request_states={request.request_id: state}
        )
        scheduler.requests = {request.request_id: request}
        scheduler.running = []
        scheduler.waiting = [request]
        scheduler.max_num_running_reqs = 64
        self.addCleanup(lambda: setattr(observer, "_v6d_scheduler", None))
        self.addCleanup(observer.shutdown)
        return observer, scheduler, reader, request, state, future, outcome

    def refresh_completed_native_query(self, observer, scheduler):
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.naviSchedModule",
            return_value=self.native_pp_test_module(),
        ):
            observer._refresh_and_report_workload(
                scheduler, scheduler.requests, "completed-query"
            )
        return observer._current_workload.native_pp_state.requests[0]

    def test_native_pp_completed_query_prefix_is_known_before_native_commit(self):
        observer, scheduler, reader, request, state, future, outcome = (
            self.completed_native_query_fixture()
        )
        future.set_result(outcome)
        before = dict(outcome.lookup_results)
        value = self.refresh_completed_native_query(observer, scheduler)
        self.assertEqual(12288, value.cache_hit_tokens)
        self.assertEqual(6, value.phase)
        self.assertFalse(reader.failed)
        self.assertIs(future, state.match_future)
        self.assertIsNone(state.resolved_match)
        self.assertIs(outcome, future.result())
        self.assertEqual(before, outcome.lookup_results)
        self.assertFalse(state.load_inflight)
        self.assertEqual(
            2, observer._current_workload.native_cache_queries[0].future_state
        )
        self.assertEqual(
            1, observer._current_workload.native_cache_queries[0].load_state
        )

    def test_native_pp_completed_query_miss_preserves_original_local_prefix(self):
        observer, scheduler, reader, request, state, future, outcome = (
            self.completed_native_query_fixture(hit=0)
        )
        future.set_result(outcome)
        value = self.refresh_completed_native_query(observer, scheduler)
        self.assertEqual(8192, value.cache_hit_tokens)
        self.assertEqual(6, value.phase)
        self.assertIsNone(state.resolved_match)

    def test_native_pp_pending_query_does_not_read_future_result(self):
        observer, scheduler, reader, request, state, future, outcome = (
            self.completed_native_query_fixture()
        )
        with mock.patch.object(
            Future, "result", side_effect=AssertionError("pending result read")
        ):
            value = self.refresh_completed_native_query(observer, scheduler)
        self.assertEqual(-1, value.cache_hit_tokens)
        self.assertEqual(5, value.phase)

    def test_native_pp_completed_prequery_is_not_a_resolved_prefix(self):
        observer, scheduler, reader, request, state, future, outcome = (
            self.completed_native_query_fixture(prefix=-1)
        )
        future.set_result(outcome)
        with mock.patch.object(
            Future, "result", side_effect=AssertionError("prequery result read")
        ):
            value = self.refresh_completed_native_query(observer, scheduler)
        self.assertEqual(-1, value.cache_hit_tokens)
        self.assertEqual(6, value.phase)

    def test_native_pp_completed_query_uses_cached_future_base_result(self):
        observer, scheduler, reader, request, state, future, outcome = (
            self.completed_native_query_fixture()
        )
        future.set_result(outcome)
        future.result = mock.Mock(side_effect=AssertionError("worker response drained"))
        value = self.refresh_completed_native_query(observer, scheduler)
        self.assertEqual(12288, value.cache_hit_tokens)
        future.result.assert_not_called()

    def test_native_pp_foreign_future_subclass_is_not_read(self):
        class WorkerFuture(Future):
            def result(self, *args, **kwargs):
                raise AssertionError("foreign worker Future read")

        observer, scheduler, reader, request, state, future, outcome = (
            self.completed_native_query_fixture()
        )
        foreign = WorkerFuture()
        foreign.set_result(outcome)
        state.match_future = foreign
        value = self.refresh_completed_native_query(observer, scheduler)
        self.assertEqual(-1, value.cache_hit_tokens)

    def test_native_pp_failed_cancelled_or_stale_query_stays_unknown(self):
        for kind in (
            "exception",
            "cancelled",
            "fetch_failed",
            "generation",
            "signature",
            "oversized",
        ):
            with self.subTest(kind=kind):
                observer, scheduler, reader, request, state, future, outcome = (
                    self.completed_native_query_fixture()
                )
                if kind == "exception":
                    future.set_exception(RuntimeError("native query failure"))
                elif kind == "cancelled":
                    future.cancel()
                else:
                    if kind == "fetch_failed":
                        outcome.fetch_failed = True
                    if kind == "generation":
                        outcome.generation = 2
                    if kind == "signature":
                        outcome.match_signature = (8192, ("other-key",))
                    if kind == "oversized":
                        outcome.hit_length = 65536
                    future.set_result(outcome)
                value = self.refresh_completed_native_query(observer, scheduler)
                self.assertEqual(-1, value.cache_hit_tokens)
                self.assertFalse(reader.failed)

    def test_non_native_pp_observer_does_not_read_completed_query(self):
        observer, scheduler, reader, request, state, future, outcome = (
            self.completed_native_query_fixture()
        )
        future.set_result(outcome)
        observer._native_pipeline_state = None
        with mock.patch.object(
            Future, "result", side_effect=AssertionError("legacy query read")
        ):
            facts = observer._native_v6d_request_audit(request)
        self.assertIsNone(facts.get("native_v6d_completed_query_prefix_tokens"))

    def test_native_pp_full_completed_query_preserves_raw_cache_prefix(self):
        observer, scheduler, reader, request, state, future, outcome = (
            self.completed_native_query_fixture(hit=24576)
        )
        request.num_tokens = 32768
        future.set_result(outcome)
        value = self.refresh_completed_native_query(observer, scheduler)
        self.assertEqual(32768, value.cache_hit_tokens)
        self.assertEqual(6, value.phase)
        self.assertEqual(0, value.computed_tokens)
        self.assertFalse(state.load_inflight)

    def test_native_pp_full_external_hit_keeps_one_token_in_prediction_queue(self):
        for prompt, cursor, loading in (
            (32768, 0, False),
            (32768, 32768, True),
            (1, 1, True),
        ):
            with self.subTest(prompt=prompt, cursor=cursor, loading=loading):
                observer, scheduler, reader, request, state, future, outcome = (
                    self.completed_native_query_fixture()
                )
                request.num_prompt_tokens = request.num_tokens = prompt
                request.num_computed_tokens = cursor
                request.prefill_stats.num_prompt_tokens = prompt
                request.prefill_stats.num_external_cached_tokens = prompt
                state.match_future = None
                state.match_signature = None
                state.match_latency_recorded = True
                state.load_inflight = loading
                if loading:
                    request.status = SimpleNamespace(name="WAITING_FOR_REMOTE_KVS")
                    scheduler.waiting = []
                    scheduler.skipped_waiting = [request]
                with mock.patch(
                    "navi_sched.adapters.vllm.VLLMEngineObserver.naviSchedModule",
                    return_value=self.native_pp_test_module(),
                ):
                    observer._refresh_and_report_workload(
                        scheduler, scheduler.requests, "full-hit"
                    )
                report = observer._current_workload
                self.assertEqual(1, len(report.native_pp_state.requests))
                value = report.native_pp_state.requests[0]
                self.assertEqual(prompt, value.cache_hit_tokens)
                self.assertEqual(cursor, value.computed_tokens)
                self.assertEqual(3 if loading else 7, value.phase)
                self.assertEqual(1, len(report.native_cache_queries))
                contribution = observer._workload_contributions[request.request_id]
                self.assertEqual(1, contribution.remaining_prefill_tokens)
                self.assertEqual(prompt - 1, contribution.prefill_context_tokens)
                self.assertEqual(cursor, request.num_computed_tokens)
                self.assertEqual(
                    prompt, request.prefill_stats.num_external_cached_tokens
                )
                self.assertFalse(reader.failed)

    def test_full_hit_last_token_rule_preserves_legacy_and_non_prefill_states(self):
        for mode in ("legacy", "running", "streaming", "has_output_tokens"):
            with self.subTest(mode=mode):
                observer, scheduler, reader, request, state, future, outcome = (
                    self.completed_native_query_fixture()
                )
                request.num_tokens = request.num_prompt_tokens
                request.num_computed_tokens = request.num_prompt_tokens
                request.prefill_stats.num_prompt_tokens = request.num_prompt_tokens
                request.prefill_stats.num_external_cached_tokens = (
                    request.num_prompt_tokens
                )
                if mode == "legacy":
                    observer._native_pipeline_state = None
                if mode == "running":
                    request.status = SimpleNamespace(name="RUNNING")
                if mode == "streaming":
                    request.status = SimpleNamespace(name="WAITING_FOR_STREAMING_REQ")
                if mode == "has_output_tokens":
                    request.num_tokens += 1
                value = observer._make_workload_contribution(request, 0, None)
                self.assertEqual(0, value.remaining_prefill_tokens)

    def test_native_pp_current_query_overrides_stats_from_an_earlier_lookup(self):
        observer, scheduler, reader, request, state, future, outcome = (
            self.completed_native_query_fixture()
        )
        future.set_result(outcome)
        request.prefill_stats.num_prompt_tokens = 32768
        request.prefill_stats.num_local_cached_tokens = 4096
        value = self.refresh_completed_native_query(observer, scheduler)
        self.assertEqual(12288, value.cache_hit_tokens)

    def test_native_pp_request_observation_preserves_queue_cursor_and_unknown_cache(
        self,
    ):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._pp_capacity_unified = True
        observer._native_request_state_audit_enabled = False
        scheduler = observer.create_scheduler(LifecycleScheduler)._scheduler
        reader = self.bind_native_pp_test_reader(observer, scheduler)
        running = self.make_request(
            prompt_tokens=32768,
            computed_tokens=16384,
            cached_tokens=4096,
            request_id="pp-running",
        )
        skipped = self.make_request(
            prompt_tokens=16384,
            computed_tokens=8192,
            cached_tokens=-1,
            request_id="pp-skipped",
        )
        waiting = self.make_request(
            prompt_tokens=16384,
            computed_tokens=0,
            cached_tokens=-1,
            request_id="pp-waiting",
        )
        for request in (running, skipped, waiting):
            request.max_tokens = 1
            request.resumable = False
        running.status = SimpleNamespace(name="RUNNING")
        skipped.status = SimpleNamespace(name="WAITING_FOR_REMOTE_KVS")
        waiting.status = SimpleNamespace(name="WAITING")
        scheduler.requests = {r.request_id: r for r in (running, waiting, skipped)}
        scheduler.running = [running]
        scheduler.skipped_waiting = [skipped]
        scheduler.waiting = [waiting]
        scheduler.max_num_running_reqs = 64
        first_poll = time.monotonic() - 0.25
        observer._v6d_scheduler = SimpleNamespace(
            _request_states={
                waiting.request_id: self.make_v6d_query_state(
                    match_future=Future(),
                    match_first_poll_at=first_poll,
                    match_arrived_at=first_poll - 5,
                    match_generation=1,
                    match_signature=(0, ()),
                ),
                skipped.request_id: self.make_v6d_query_state(
                    load_inflight=True, load_started_at=first_poll
                ),
            }
        )
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.naviSchedModule",
            return_value=self.native_pp_test_module(),
        ):
            observer._refresh_and_report_workload(
                scheduler, scheduler.requests, "owner"
            )
        value = observer._current_workload.native_pp_state
        self.assertFalse(reader.failed)
        self.assertEqual(1, value.occupied_request_slot_count)
        self.assertEqual([1, 1, 1], [r.prefill_slot_release for r in value.requests])
        self.assertEqual([0, 1, 2], [r.queue for r in value.requests])
        self.assertEqual([16384, 8192, 0], [r.computed_tokens for r in value.requests])
        self.assertEqual([4096, -1, -1], [r.cache_hit_tokens for r in value.requests])
        self.assertEqual(5, value.requests[-1].phase)
        observed = observer._current_workload.native_pp_observed_time_ns
        self.assertEqual(
            (observed - round(first_poll * 1e9)) // 1000,
            value.requests[-1].query_age_us,
        )
        self.assertEqual((), observer._current_workload.native_prefill_phases)
        self.assertEqual(
            3, len(observer._current_workload.native_prefill_queue_entries)
        )
        observer._v6d_scheduler = None
        observer.shutdown()

    def test_native_pp_frozen_arrival_preserves_original_age_and_prepared_identity(
        self,
    ):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._pp_capacity_unified = True
        scheduler = observer.create_scheduler(LifecycleScheduler)._scheduler
        scheduler.running = []
        scheduler.waiting = []
        scheduler.requests = {}
        scheduler.max_num_running_reqs = 64
        self.bind_native_pp_test_reader(observer, scheduler)
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.naviSchedModule",
            return_value=self.native_pp_test_module(),
        ):
            observer._refresh_and_report_workload(scheduler, (), "empty-owner")
            base = observer._current_workload
            request = self.make_request(
                prompt_tokens=128,
                computed_tokens=0,
                cached_tokens=0,
                request_id="prepared-pp",
            )
            state = self.add_forward_batch(observer, 701, [request.request_id])
            observer._set_admission_state(
                state, [request], base, time.perf_counter_ns()
            )
            frozen = observer._capture_arrival_workload()
            self.assertEqual(
                base.native_pp_observed_time_ns, frozen.native_pp_observed_time_ns
            )
            self.assertEqual(
                base.native_pp_state.observation_id,
                frozen.native_pp_state.observation_id,
            )
            self.assertEqual(0, base.native_pp_state.occupied_request_slot_count)
            self.assertEqual(0, frozen.native_pp_state.occupied_request_slot_count)
            self.assertEqual([], base.native_pp_state.requests)
            row = frozen.native_pp_state.requests[0]
            self.assertEqual(
                (4, 0, 11, 128, 0, -1, -1),
                (
                    row.queue,
                    row.queue_index,
                    row.phase,
                    row.prompt_tokens,
                    row.computed_tokens,
                    row.cache_hit_tokens,
                    row.query_age_us,
                ),
            )
            with mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.time.monotonic_ns",
                return_value=base.native_pp_observed_time_ns + 987654000,
            ):
                self.assertTrue(observer._report_workload_locked(frozen, "arrival"))
            self.assertIs(
                frozen.native_pp_state, observer._bridge.native_pp_reports[-1][0]
            )
            self.assertEqual(987654, observer._bridge.native_pp_reports[-1][1])
            observer._remove_prepared_workload(state, state.request_ids)
            self.assertEqual(
                [], observer._capture_arrival_workload().native_pp_state.requests
            )
        observer.shutdown()

    def test_native_pp_occupied_slots_include_non_prefill_and_paused_streaming(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)._scheduler
        scheduler.running = [object(), object()]
        reader = self.bind_native_pp_test_reader(observer, scheduler)
        scheduler.num_waiting_for_streaming_input = 3
        before = reader.capture()
        self.assertEqual(5, before.occupied_request_slot_count)
        scheduler.num_waiting_for_streaming_input = 2
        after = reader.capture()
        self.assertEqual(4, after.occupied_request_slot_count)
        self.assertNotEqual(reader.signature(before), reader.signature(after))
        self.assertEqual(5, before.occupied_request_slot_count)
        value, _ = reader.make_report(self.native_pp_test_module(), [])
        self.assertEqual(4, value.occupied_request_slot_count)
        observer.shutdown()

    def test_native_pp_slot_release_uses_actual_completion_conditions(self):
        for maximum, resumable, expected in (
            (1, False, 1),
            (1, True, 2),
            (2, False, 2),
        ):
            request = SimpleNamespace(max_tokens=maximum, resumable=resumable)
            self.assertEqual(
                expected, VLLMEngineObserver._native_prefill_slot_release(request)
            )
        with self.assertRaises(ValueError):
            VLLMEngineObserver._native_prefill_slot_release(
                SimpleNamespace(max_tokens=0, resumable=False)
            )
        with self.assertRaises(AttributeError):
            VLLMEngineObserver._native_prefill_slot_release(
                SimpleNamespace(max_tokens=1)
            )

    def test_native_pp_missing_streaming_capacity_is_not_assumed_zero(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        reader = self.bind_native_pp_test_reader(observer)
        del reader._stepper.core.scheduler.num_waiting_for_streaming_input
        with self.assertRaises(AttributeError):
            reader.capture()
        observer.shutdown()

    def test_native_pp_scheduled_span_uses_pre_schedule_cursor_and_pending_fence(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        reader = self.bind_native_pp_test_reader(observer)
        reader.dispatch_active = True
        request = self.make_request(
            prompt_tokens=32768,
            computed_tokens=16384,
            cached_tokens=4096,
            request_id="pending-pp",
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 8192})
        observer._register_scheduled_work(
            output, SimpleNamespace(requests={request.request_id: request})
        )
        key, shape, spans, loads = reader.pending_decision
        self.assertEqual((), loads)
        self.assertEqual(id(output), key)
        self.assertEqual(8192, spans[0].computed_tokens)
        self.assertEqual(8192, spans[0].scheduled_tokens)
        self.assertEqual((1.0, 8.0, 8.0, 64.0, 8.0, 64.0, 1.0, 8.0), shape)
        self.assertEqual(8192, observer._active_scheduled_tokens)
        self.assertEqual(8192, observer._inflight_scheduled_tokens[request.request_id])
        observer.shutdown()

    def test_native_pp_cache_observation_uses_resolved_native_stats(self):
        request = self.make_request(
            prompt_tokens=32768, computed_tokens=16384, cached_tokens=-1
        )
        request.prefill_stats = SimpleNamespace(
            num_prompt_tokens=0, num_local_cached_tokens=0, num_external_cached_tokens=0
        )
        self.assertEqual(
            -1, VLLMEngineObserver._native_resolved_cache_hit_tokens(request)
        )
        request.prefill_stats.num_prompt_tokens = 32768
        request.prefill_stats.num_local_cached_tokens = 4096
        request.prefill_stats.num_external_cached_tokens = 8192
        self.assertEqual(
            12288, VLLMEngineObserver._native_resolved_cache_hit_tokens(request)
        )
        request.prefill_stats = None
        self.assertEqual(
            -1, VLLMEngineObserver._native_resolved_cache_hit_tokens(request)
        )
        request.num_cached_tokens = 12288
        self.assertEqual(
            12288, VLLMEngineObserver._native_resolved_cache_hit_tokens(request)
        )

    def test_unified19_audit_keeps_release_and_capacity_facts(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._pp_capacity_unified = True
        observer._native_request_state_audit_enabled = True
        self.assertIsNone(observer._native_pipeline_state)
        events = []
        observer.on_request_lifecycle_event = events.append
        scheduler = observer.create_scheduler(LifecycleScheduler)._scheduler
        requests = []
        for index, (maximum, resumable) in enumerate(((1, False), (16, False), (1, True))):
            request = self.make_request(prompt_tokens=64, computed_tokens=0,
                                        cached_tokens=0, request_id="audit-%d" % index)
            request.max_tokens = maximum
            request.resumable = resumable
            request.status = SimpleNamespace(name="WAITING")
            request.is_finished = lambda: False
            requests.append(request)
        unknown = self.make_request(prompt_tokens=64, computed_tokens=0,
                                    cached_tokens=0, request_id="audit-unknown")
        unknown.status = SimpleNamespace(name="WAITING")
        unknown.is_finished = lambda: False
        requests.append(unknown)
        scheduler.max_num_running_reqs = 64
        scheduler.requests = {r.request_id: r for r in requests}
        scheduler.running = []
        scheduler.waiting = requests[:]
        queue_before = tuple(scheduler.waiting)
        events.clear()
        observer._refresh_and_report_workload(scheduler, scheduler.requests, "audit_facts")
        workload = observer._current_workload
        audit = next(e for e in events if e["event"] == "native_prefill_workload_phases")
        self.assertEqual([1, 2, 2, 0],
                         [e["native_prefill_slot_release"] for e in audit["queue_entries"]])
        self.assertEqual(64, audit["max_running_request_count"])
        self.assertEqual(workload.running_capacity_slot_count,
                         audit["running_capacity_slot_count"])
        self.assertTrue(audit["phase_totals_match_report"])
        self.assertEqual(4, audit["reported_prefill"]["request_count"])
        self.assertEqual(256, audit["reported_prefill"]["compute_tokens"])
        self.assertEqual(queue_before, tuple(scheduler.waiting))
        requests[0].max_tokens = 128
        observer._report_workload_locked(workload, "frozen_audit_facts")
        later = [e for e in events if e["event"] == "native_prefill_workload_phases"][-1]
        self.assertEqual(audit["queue_entries"], later["queue_entries"])
        self.assertIsNone(observer._native_pipeline_state)
        observer.shutdown()

    def test_pp_workload_phase_audit_sums_stored_contributions(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._pp_capacity_unified = True
        observer._native_request_state_audit_enabled = True
        events = []
        observer.on_request_lifecycle_event = events.append
        scheduler = observer.create_scheduler(LifecycleScheduler)._scheduler
        specs = [
            ("running", 100, 20, 20, "RUNNING"),
            ("pending", 64, 0, -1, "WAITING"),
            ("ready", 32, 0, -1, "WAITING"),
            ("loading", 48, 16, 16, "WAITING_FOR_REMOTE_KVS"),
            ("skipped", 32, 8, 8, "WAITING"),
        ]
        requests = []
        for name, prompt, computed, cached, status in specs:
            request = self.make_request(
                prompt_tokens=prompt,
                computed_tokens=computed,
                cached_tokens=cached,
                request_id=name,
            )
            request.status = SimpleNamespace(name=status)
            request.is_finished = lambda: False
            requests.append(request)
        pending = Future()
        ready = Future()
        ready.set_result((0, False))
        states = {
            "pending": self.make_v6d_query_state(
                match_future=pending, match_generation=1, match_signature=(0, ())
            ),
            "ready": self.make_v6d_query_state(
                match_future=ready, match_generation=1, match_signature=(0, ())
            ),
            "loading": self.make_v6d_query_state(load_inflight=True),
            "skipped": self.make_v6d_query_state(skip_lookup=True),
        }
        observer._v6d_scheduler = SimpleNamespace(_request_states=states)
        scheduler.max_num_running_reqs = 64
        scheduler.requests = {r.request_id: r for r in requests}
        scheduler.running = requests[:1]
        scheduler.waiting = requests[1:]
        events.clear()
        with (
            mock.patch.object(
                pending, "result", side_effect=AssertionError("no query")
            ),
            mock.patch.object(ready, "result", side_effect=AssertionError("no query")),
        ):
            observer._refresh_and_report_workload(
                scheduler, scheduler.requests, "phase_split"
            )
        workload = observer._current_workload
        phases = dict(workload.native_prefill_phases)
        self.assertEqual(
            {
                "native_running",
                "query_pending",
                "query_ready_not_consumed",
                "kv_loading",
                "lookup_skipped_waiting",
            },
            set(phases),
        )
        expected = {
            "request_count": 5,
            "context_tokens": 44,
            "compute_tokens": 232,
            "compute_token_square_sum": 13120.0,
            "context_compute_token_product_sum": 2304.0,
        }
        for field, value in expected.items():
            self.assertEqual(value, sum(getattr(p, field) for p in phases.values()))
        audits = [e for e in events if e["event"] == "native_prefill_workload_phases"]
        self.assertEqual(1, len(audits))
        audit = audits[0]
        self.assertEqual(expected, audit["reported_prefill"])
        self.assertTrue(audit["phase_totals_match_report"])
        self.assertEqual(
            observer._bridge.workloads[-1][0], audit["report_event_time_us"]
        )
        # A later query completion/cursor update must not rewrite the old snapshot.
        pending.set_result((64, True))
        requests[1].num_computed_tokens = 64
        self.assertTrue(
            observer._report_workload_locked(workload, "frozen_phase_split")
        )
        audits = [e for e in events if e["event"] == "native_prefill_workload_phases"]
        self.assertEqual(audit["phases"], audits[-1]["phases"])
        self.assertEqual(
            workload,
            replace(
                workload,
                native_prefill_phases=(),
                native_prefill_phase_observed_time_ns=0,
            ),
        )
        observer._v6d_scheduler = None
        observer.shutdown()

    def test_pp_ordered_queue_audit_is_frozen_and_does_not_reorder_native(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._pp_capacity_unified = True
        observer._native_request_state_audit_enabled = True
        events = []
        observer.on_request_lifecycle_event = events.append
        scheduler = observer.create_scheduler(LifecycleScheduler)._scheduler
        requests = []
        for request_id, status in (
            ("run", "RUNNING"),
            ("load", "WAITING_FOR_REMOTE_KVS"),
            ("wait", "WAITING"),
        ):
            request = self.make_request(
                prompt_tokens=64,
                computed_tokens=16,
                cached_tokens=16,
                request_id=request_id,
            )
            request.status = SimpleNamespace(name=status)
            request.is_finished = lambda: False
            requests.append(request)
        scheduler.requests = {r.request_id: r for r in requests}
        scheduler.running = requests[:1]
        scheduler.skipped_waiting = requests[1:2]
        scheduler.waiting = requests[2:]
        scheduler.finished_recving_kv_req_ids = {"load"}
        scheduler.failed_recving_kv_req_ids = set()
        scheduler.max_num_running_reqs = 64
        observer._v6d_scheduler = SimpleNamespace(
            _request_states={
                "load": self.make_v6d_query_state(
                    load_inflight=True, load_started_at=1.25
                ),
            }
        )
        before = (
            tuple(scheduler.running),
            tuple(scheduler.skipped_waiting),
            tuple(scheduler.waiting),
            frozenset(scheduler.finished_recving_kv_req_ids),
        )
        events.clear()
        observer._refresh_and_report_workload(
            scheduler, scheduler.requests, "queue_order"
        )
        workload = observer._current_workload
        entries = {e.request_id: e for e in workload.native_prefill_queue_entries}
        self.assertEqual({"run", "load", "wait"}, set(entries))
        self.assertEqual(
            ("running", 0), (entries["run"].queue, entries["run"].queue_index)
        )
        self.assertEqual(
            ("skipped_waiting", 0), (entries["load"].queue, entries["load"].queue_index)
        )
        self.assertEqual(
            ("waiting", 0), (entries["wait"].queue, entries["wait"].queue_index)
        )
        self.assertTrue(entries["load"].worker_receive_reported)
        self.assertFalse(entries["load"].worker_receive_failed)
        self.assertTrue(entries["load"].load_inflight)
        self.assertEqual(1250000000, entries["load"].load_started_mono_ns)
        after = (
            tuple(scheduler.running),
            tuple(scheduler.skipped_waiting),
            tuple(scheduler.waiting),
            frozenset(scheduler.finished_recving_kv_req_ids),
        )
        self.assertEqual(before, after)
        audit = next(
            e for e in events if e["event"] == "native_prefill_workload_phases"
        )
        for name, total in audit["reported_prefill"].items():
            self.assertEqual(
                total, sum(e["workload"][name] for e in audit["queue_entries"])
            )
        scheduler.finished_recving_kv_req_ids.clear()
        requests[1].num_computed_tokens = 64
        observer._v6d_scheduler._request_states["load"].load_inflight = False
        self.assertTrue(
            observer._report_workload_locked(workload, "frozen_queue_order")
        )
        later = [e for e in events if e["event"] == "native_prefill_workload_phases"][
            -1
        ]
        self.assertEqual(audit["queue_entries"], later["queue_entries"])
        self.assertEqual(workload, replace(workload, native_prefill_queue_entries=()))
        observer._v6d_scheduler = None
        observer.shutdown()

    def test_non_pp_workload_report_does_not_collect_native_phases(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._native_request_state_audit_enabled = True
        events = []
        observer.on_request_lifecycle_event = events.append
        scheduler = observer.create_scheduler(LifecycleScheduler)._scheduler
        request = self.make_request(
            prompt_tokens=32, computed_tokens=0, cached_tokens=0
        )
        scheduler.requests = {request.request_id: request}
        scheduler.running = [request]
        scheduler.waiting = []
        with mock.patch.object(
            observer,
            "_native_prefill_wait_phase",
            side_effect=AssertionError("non-PP phase collection"),
        ):
            observer._refresh_and_report_workload(
                scheduler, scheduler.requests, "legacy"
            )
        self.assertEqual((), observer._current_workload.native_prefill_phases)
        self.assertEqual((), observer._current_workload.native_prefill_queue_entries)
        self.assertEqual(
            0, observer._current_workload.native_prefill_phase_observed_time_ns
        )
        self.assertFalse(
            any(e["event"] == "native_prefill_workload_phases" for e in events)
        )
        observer.shutdown()

    def test_native_phase_audit_retains_unclassified_reported_work(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        events = []
        observer.on_request_lifecycle_event = events.append
        native = _PrefillWorkloadSlice(1, 8, 24, 576.0, 192.0)
        report = _ObservedWorkload(
            remaining_prefill_request_count=2,
            remaining_prefill_context_tokens=24,
            remaining_prefill_tokens=40,
            remaining_prefill_token_square_sum=832.0,
            context_remaining_prefill_token_product_sum=448.0,
            native_prefill_phases=(("native_running", native),),
            native_prefill_phase_observed_time_ns=100,
        )
        observer._audit_native_prefill_workload(report, 12345, "prepared_requests")
        self.assertFalse(events[-1]["phase_totals_match_report"])
        self.assertEqual(
            {
                "request_count": 1,
                "context_tokens": 16,
                "compute_tokens": 16,
                "compute_token_square_sum": 256.0,
                "context_compute_token_product_sum": 256.0,
            },
            events[-1]["unclassified_prefill"],
        )
        self.assertEqual(12345, events[-1]["report_event_time_us"])
        observer.shutdown()

    def test_native_phase_audit_error_does_not_reject_successful_workload_report(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer.on_request_lifecycle_event = mock.Mock(
            side_effect=RuntimeError("audit sink")
        )
        workload = _ObservedWorkload(native_prefill_phase_observed_time_ns=100)
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.logger.exception"
        ) as log_error:
            reported = observer._report_workload_locked(workload, "audit_failure")
        log_error.assert_called_once()
        self.assertTrue(reported)
        self.assertEqual(1, len(observer._bridge.workloads))
        observer.shutdown()

    def test_pp_audit_reads_v6d_state_without_fixed_chunk_preparation(self):
        from concurrent.futures import Future

        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._pp_capacity_unified = True
        observer._native_request_state_audit_enabled = True
        events = []
        observer.on_request_lifecycle_event = events.append
        request = self.make_request(
            prompt_tokens=8192, computed_tokens=0, cached_tokens=-1
        )
        scheduler = SimpleNamespace(
            running=[],
            waiting=[request],
            requests={request.request_id: request},
            max_num_running_reqs=64,
        )
        future = Future()
        state = self.make_v6d_query_state(
            match_future=future,
            resolved_match=None,
            match_signature=(0, ()),
            match_arrived_at=11.25,
            match_first_poll_at=11.5,
            match_latency_recorded=False,
            match_generation=1,
            pending_load_meta={},
            load_inflight=False,
            load_started_at=None,
            load_failed=False,
            load_job_started=False,
            skip_lookup=False,
        )
        states = {request.request_id: state}
        observer._v6d_scheduler = SimpleNamespace(_request_states=states)
        observer._partition_prefill_workload(scheduler)
        future.set_result((4096, True))
        observer._partition_prefill_workload(scheduler)
        state.resolved_match = (4096, True)
        state.resolved_match_signature = state.match_signature
        state.match_signature = None
        state.match_future = None
        state.match_latency_recorded = True
        state.load_inflight = True
        state.load_started_at = 12.5
        state.pending_load_meta = {0: (["key"], [1])}
        observer._partition_prefill_workload(scheduler)
        observer._partition_prefill_workload(scheduler)
        self.assertEqual(3, len(events))
        self.assertEqual(
            [False, True, None],
            [event["native_v6d_match_future_done"] for event in events],
        )
        self.assertEqual(
            [None, None, 4096], [event["native_v6d_match_tokens"] for event in events]
        )
        self.assertEqual(
            [False, False, True],
            [event["native_v6d_load_inflight"] for event in events],
        )
        self.assertNotIn("native_prefix_query_resolved", events[0])
        self.assertTrue(events[2]["native_v6d_match_async"])
        self.assertTrue(events[2]["native_v6d_load_meta_pending"])
        self.assertEqual(11250000000, events[0]["native_v6d_match_arrived_mono_ns"])
        self.assertEqual(12500000000, events[2]["native_v6d_load_started_mono_ns"])
        self.assertTrue(all(not event["cache_lookup_complete"] for event in events))
        self.assertIs(states[request.request_id], state)
        self.assertEqual((4096, True), future.result())
        observer._v6d_scheduler = None
        observer.shutdown()

    def test_pp_audit_distinguishes_v6d_state_not_created_from_no_v6d(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._pp_capacity_unified = True
        request = self.make_request(
            prompt_tokens=32, computed_tokens=0, cached_tokens=-1
        )
        observer._v6d_scheduler = SimpleNamespace(_request_states={})
        self.assertFalse(
            observer._native_v6d_request_audit(request)["native_v6d_state_present"]
        )
        observer._v6d_scheduler = None
        self.assertEqual({}, observer._native_v6d_request_audit(request))
        observer.shutdown()

    def test_non_pp_audit_never_reads_v6d_request_state(self):
        class ForbiddenV6DState:
            @property
            def _request_states(self):
                raise AssertionError("non-PP must not read V6D request state")

        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=32, computed_tokens=0, cached_tokens=-1
        )
        observer._v6d_scheduler = ForbiddenV6DState()
        self.assertEqual({}, observer._native_v6d_request_audit(request))
        observer._v6d_scheduler = None
        observer.shutdown()

    def test_capacity_sample_uses_request_order_and_actual_cache_hits(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        running_prefill = self.make_request(
            prompt_tokens=32,
            computed_tokens=8,
            cached_tokens=8,
            request_id="running-prefill",
        )
        running_decode = self.make_request(
            prompt_tokens=16,
            computed_tokens=16,
            cached_tokens=0,
            request_id="running-decode",
        )
        scheduler._scheduler.max_num_running_reqs = 3
        scheduler._scheduler.requests = {
            request.request_id: request for request in (running_prefill, running_decode)
        }
        scheduler._scheduler.running = [running_prefill, running_decode]
        observer._refresh_and_report_workload(
            scheduler._scheduler,
            scheduler._scheduler.requests,
            "capacity_sample_base",
        )

        state = self.add_forward_batch(
            observer, 206, ["first-incoming", "second-incoming"], [4, 4]
        )
        incoming = [
            self.make_request(
                prompt_tokens=20,
                computed_tokens=0,
                cached_tokens=-1,
                request_id="first-incoming",
            ),
            self.make_request(
                prompt_tokens=30,
                computed_tokens=0,
                cached_tokens=-1,
                request_id="second-incoming",
            ),
        ]
        observer._set_admission_state(
            state,
            incoming,
            observer._current_workload,
            time.perf_counter_ns(),
        )

        admission = state.admission_state
        self.assertEqual(
            frozenset({stable_request_id("first-incoming")}),
            admission.incoming_running_capacity_request_ids,
        )
        running_capacity, waiting_capacity = observer._capacity_sample_features(
            admission,
            [
                (stable_request_id("first-incoming"), 20, 10),
                (stable_request_id("second-incoming"), 30, 12),
            ],
        )
        self.assertEqual(
            (2, 18, 34, 676.0, 292.0),
            (
                running_capacity.request_count,
                running_capacity.context_tokens,
                running_capacity.compute_tokens,
                running_capacity.compute_token_square_sum,
                running_capacity.context_compute_token_product_sum,
            ),
        )
        self.assertEqual(
            (1, 12, 18, 324.0, 216.0),
            (
                waiting_capacity.request_count,
                waiting_capacity.context_tokens,
                waiting_capacity.compute_tokens,
                waiting_capacity.compute_token_square_sum,
                waiting_capacity.context_compute_token_product_sum,
            ),
        )
        scheduler_running, scheduler_waiting = observer._scheduler_sample_features(
            admission,
            [
                (stable_request_id("first-incoming"), 20, 10),
                (stable_request_id("second-incoming"), 30, 12),
            ],
        )
        self.assertEqual(
            (1, 8, 24, 576.0, 192.0),
            (
                scheduler_running.request_count,
                scheduler_running.context_tokens,
                scheduler_running.compute_tokens,
                scheduler_running.compute_token_square_sum,
                scheduler_running.context_compute_token_product_sum,
            ),
        )
        self.assertEqual(
            (2, 22, 28, 424.0, 316.0),
            (
                scheduler_waiting.request_count,
                scheduler_waiting.context_tokens,
                scheduler_waiting.compute_tokens,
                scheduler_waiting.compute_token_square_sum,
                scheduler_waiting.context_compute_token_product_sum,
            ),
        )

    def test_cancelled_pending_request_is_excluded_from_later_arrival(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer.create_scheduler(LifecycleScheduler)
        input_queue = SimpleNamespace(
            items=[], put_nowait=lambda item: input_queue.items.append(item)
        )
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda request: (request, 0),
            input_queue=input_queue,
        )
        request_type = SimpleNamespace(ADD="add", ABORT="abort")
        requests = [
            self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=-1,
                request_id=request_id,
            )
            for request_id in ("cancelled", "retained")
        ]
        first_batch = FakeForwardBatch(
            203, [request.request_id for request in requests], [0, 0]
        )
        observer._enqueue_forward_batch(
            first_batch,
            [(stable_request_id(request.request_id), request) for request in requests],
            request_type,
        )
        first_state = observer._pending_forward_batches[stable_request_id("cancelled")]
        observer._abort_forward_requests(first_state, {stable_request_id("cancelled")})

        next_request = self.make_request(
            prompt_tokens=16,
            computed_tokens=0,
            cached_tokens=-1,
            request_id="next",
        )
        next_batch = FakeForwardBatch(204, ["next"], [0])
        observer._enqueue_forward_batch(
            next_batch,
            [(stable_request_id("next"), next_request)],
            request_type,
        )

        next_state = observer._pending_forward_batches[stable_request_id("next")]
        workload = next_state.admission_state.preceding_workload
        self.assertEqual(1, workload.request_count)
        self.assertEqual(32, workload.remaining_prefill_tokens)

    def test_workload_tracks_prefill_only_inflight_and_completion(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        self.add_forward_batch(observer, 98, ["request"], [16])
        request = self.make_request(
            prompt_tokens=64,
            computed_tokens=0,
            cached_tokens=-1,
            request_id="request",
        )
        request.max_tokens = 1
        one_output_token = observer._make_workload_contribution(request, 0, None)
        request.max_tokens = 8192
        self.assertEqual(
            one_output_token,
            observer._make_workload_contribution(request, 0, one_output_token),
        )
        scheduler.add_request(request)

        admitted = observer._bridge.workloads[-1]
        self.assertEqual(1, admitted[1])
        self.assertEqual(1, admitted[2])
        self.assertEqual(16, admitted[6])
        self.assertEqual(48, admitted[7])
        self.assertEqual(16, admitted[22])
        self.assertEqual(48, admitted[23])
        self.assertEqual(0, admitted[27])

        scheduler_output = scheduler.schedule()
        scheduled = observer._bridge.workloads[-1]
        self.assertEqual(0, scheduled[6])
        self.assertEqual(64, scheduled[7])
        self.assertEqual(64, scheduled[27])

        future = Future()
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: future,
            scheduler,
        )
        wrapped(scheduler_output, non_block=True)
        future.set_result("model-output")
        self.assertEqual({}, observer._inflight_scheduled_tokens)
        self.assertEqual(0, observer._active_scheduled_tokens)

        self.assertEqual(
            "model-output",
            scheduler.update_from_output(scheduler_output, "model-output"),
        )
        completed = observer._bridge.workloads[-1]
        self.assertEqual(0, completed[3])
        self.assertEqual(0, completed[7])
        self.assertEqual(0, completed[22])
        self.assertEqual(0, completed[23])
        self.assertEqual(0, completed[27])

    def test_admission_keeps_pending_workload_until_scheduler_snapshot(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observed_during_admission = []

        class CapturingScheduler(LifecycleScheduler):
            def add_request(self, request):
                observed_during_admission.append(observer._capture_arrival_workload())
                super().add_request(request)

        scheduler = observer.create_scheduler(CapturingScheduler)
        self.add_forward_batch(observer, 99, ["request"], [16])
        request = self.make_request(
            prompt_tokens=64,
            computed_tokens=0,
            cached_tokens=-1,
            request_id="request",
        )
        scheduler.add_request(request)

        self.assertEqual(1, len(observed_during_admission))
        self.assertEqual(1, observed_during_admission[0].batch_count)
        self.assertEqual(1, observed_during_admission[0].request_count)
        self.assertEqual(48, observed_during_admission[0].remaining_prefill_tokens)
        self.assertEqual({}, observer._pending_forward_batches)
        current = observer._capture_arrival_workload()
        self.assertEqual(1, current.batch_count)
        self.assertEqual(1, current.request_count)

    def test_workload_report_includes_batches_waiting_for_vllm_admission(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        requests = [
            self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=-1,
                request_id=request_id,
            )
            for request_id in ("first", "second")
        ]
        native_batch = FakeForwardBatch(
            207, [request.request_id for request in requests], [8, 16]
        )
        input_queue = SimpleNamespace(
            items=[], put_nowait=lambda item: input_queue.items.append(item)
        )
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda request: (request, 0),
            input_queue=input_queue,
        )
        observer._enqueue_forward_batch(
            native_batch,
            [(stable_request_id(request.request_id), request) for request in requests],
            SimpleNamespace(ADD="add", ABORT="abort"),
        )

        observer._refresh_and_report_workload(
            scheduler._scheduler, (), "pending_admission"
        )

        workload = observer._bridge.workloads[-1]
        self.assertEqual(1, workload[1])
        self.assertEqual(2, workload[2])
        self.assertEqual(2, workload[3])
        self.assertEqual(0, workload[4])
        self.assertEqual(2, workload[5])
        self.assertEqual(24, workload[6])
        self.assertEqual(40, workload[7])
        self.assertEqual(24, workload[22])
        self.assertEqual(40, workload[23])

    def test_queued_forward_load_is_published_before_engine_owner_advances(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        initial_reports = len(observer._bridge.workloads)
        self.assertEqual(0, observer._bridge.workloads[-1][2])
        input_queue = SimpleNamespace(items=[])
        load_at_submission = []

        def enqueue(item):
            load_at_submission.append(observer._bridge.workloads[-1][2])
            input_queue.items.append(item)

        input_queue.put_nowait = enqueue
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda request: (request, 0),
            input_queue=input_queue,
        )
        requests = [
            self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=-1,
                request_id=request_id,
            )
            for request_id in ("queued-first", "queued-second")
        ]
        native_batch = FakeForwardBatch(
            208, [request.request_id for request in requests], [8, 16]
        )
        observer._enqueue_forward_batch(
            native_batch,
            [(stable_request_id(request.request_id), request) for request in requests],
            SimpleNamespace(ADD="add", ABORT="abort"),
        )

        self.assertEqual(2, len(input_queue.items))
        queued = observer._capture_arrival_workload()
        self.assertEqual(2, queued.request_count)
        self.assertEqual(24, queued.remaining_prefill_context_tokens)
        self.assertEqual(40, queued.remaining_prefill_tokens)
        self.assertEqual([2, 2], load_at_submission)
        self.assertEqual(initial_reports + 1, len(observer._bridge.workloads))
        self.assertEqual(2, observer._bridge.workloads[-1][2])
        self.assertEqual(40, observer._bridge.workloads[-1][7])
        self.assertEqual(208, observer._bridge.workloads[-1][29])
        self.assertEqual(2, observer._bridge.workloads[-1][30])
        for kind, value in input_queue.items:
            self.assertEqual("add", kind)
            scheduler.add_request(value[0])
        self.assertGreater(len(observer._bridge.workloads), initial_reports)
        published = observer._bridge.workloads[-1]
        self.assertEqual(2, published[2])
        self.assertEqual(40, published[7])
        self.assertEqual({}, observer._pending_forward_batches)

    def test_large_prepared_backlog_reports_exact_load_and_cancelled_tombstones(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer.create_scheduler(LifecycleScheduler)
        base = observer._capture_arrival_workload()
        states = []
        for index in range(2000):
            request = self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=-1,
                request_id=f"prepared-{index}",
            )
            state = self.add_forward_batch(
                observer, 1000 + index, [request.request_id], [8]
            )
            observer._set_admission_state(
                state, [request], base, time.perf_counter_ns()
            )
            states.append(state)

        elapsed_us = []
        for _ in range(10):
            started = time.perf_counter_ns()
            observer._report_prepared_workload("large_prepared_backlog")
            elapsed_us.append((time.perf_counter_ns() - started) / 1000.0)
            workload = observer._bridge.workloads[-1]
            self.assertEqual(2000, workload[1])
            self.assertEqual(2000, workload[2])
            self.assertEqual(16000, workload[6])
            self.assertEqual(48000, workload[7])
        elapsed_us.sort()
        print(
            "prepared_workload_backlog=2000 "
            f"median_us={elapsed_us[5]:.3f} max_us={elapsed_us[-1]:.3f} "
            "scope=Python_aggregation_and_stub_publication"
        )
        # A snapshot must not wait for an unrelated batch's lifecycle lock.
        with mock.patch.object(states[-1], "lock") as busy_lock:
            busy_lock.__enter__.side_effect = AssertionError("scanned batch lock")
            self.assertEqual(2000, observer._capture_arrival_workload().request_count)
        with observer._workload_lock:
            observer._current_workload = replace(base, max_running_request_count=128)
        workload = observer._capture_arrival_workload()
        self.assertEqual(128, workload.running_capacity_slot_count)
        self.assertEqual(128, workload.running_capacity_prefill.request_count)
        self.assertEqual(1024, workload.running_capacity_prefill.context_tokens)
        self.assertEqual(3072, workload.running_capacity_prefill.compute_tokens)
        self.assertEqual(1872, workload.waiting_capacity_prefill.request_count)
        self.assertEqual(14976, workload.waiting_capacity_prefill.context_tokens)
        self.assertEqual(44928, workload.waiting_capacity_prefill.compute_tokens)
        self.assertEqual(2000, workload.scheduler_waiting_prefill.request_count)
        for state in states:
            observer._abort_forward_requests(state, set(state.request_ids))
        observer._report_prepared_workload("large_prepared_cancellation")
        workload = observer._bridge.workloads[-1]
        self.assertEqual(0, workload[1])
        self.assertEqual(0, workload[2])
        self.assertEqual(0, workload[6])
        self.assertEqual(0, workload[7])
        self.assertEqual(2000, len(observer._pending_forward_batches))
        self.assertEqual(2000, len(observer._bridge.released_forward_batches))

    def test_delayed_old_batch_cleanup_preserves_reused_request_workload(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer.create_scheduler(LifecycleScheduler)
        base = observer._capture_arrival_workload()
        old_request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=-1,
            request_id="reused-prepared",
        )
        old_state = self.add_forward_batch(
            observer, 6000, [old_request.request_id], [8]
        )
        observer._set_admission_state(
            old_state, [old_request], base, time.perf_counter_ns()
        )
        # Pause release after marking the old owner, before its cleanup runs.
        with old_state.lock:
            observer._mark_forward_batch_released(old_state)
        new_request = self.make_request(
            prompt_tokens=96,
            computed_tokens=0,
            cached_tokens=-1,
            request_id=old_request.request_id,
        )
        new_state = self.add_forward_batch(
            observer, 6001, [new_request.request_id], [16]
        )
        observer._set_admission_state(
            new_state, [new_request], base, time.perf_counter_ns()
        )
        observer._cleanup_released_forward_batch(old_state)
        workload = observer._capture_arrival_workload()
        self.assertEqual(1, workload.request_count)
        self.assertEqual(1, workload.batch_count)
        self.assertEqual(96, workload.total_prompt_tokens)
        self.assertEqual(16, workload.remaining_prefill_context_tokens)
        self.assertEqual(80, workload.remaining_prefill_tokens)
        self.assertIs(
            new_state, observer._pending_forward_batches[new_state.request_ids[0]]
        )
        observer._release_forward_batch(new_state)
        self.assertEqual(0, observer._capture_arrival_workload().request_count)

    def test_prepared_workload_concurrent_cancellation_keeps_atomic_totals(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer.create_scheduler(LifecycleScheduler)
        base = observer._capture_arrival_workload()
        states = []
        for index in range(128):
            request = self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=-1,
                request_id=f"concurrent-prepared-{index}",
            )
            state = self.add_forward_batch(
                observer, 4000 + index, [request.request_id], [8]
            )
            observer._set_admission_state(
                state, [request], base, time.perf_counter_ns()
            )
            states.append(state)
        started = threading.Event()
        done = threading.Event()
        failures = []
        samples = []

        def read():
            try:
                while not done.is_set():
                    workload = observer._capture_arrival_workload()
                    self.assertTrue(workload.valid())
                    self.assertEqual(workload.request_count, workload.batch_count)
                    self.assertEqual(
                        8 * workload.request_count,
                        workload.remaining_prefill_context_tokens,
                    )
                    self.assertEqual(
                        24 * workload.request_count, workload.remaining_prefill_tokens
                    )
                    samples.append(workload.request_count)
                    started.set()
            except BaseException as error:
                failures.append(error)
                started.set()

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        try:
            self.assertTrue(started.wait(5))
            for state in states:
                observer._abort_forward_requests(state, set(state.request_ids))
                observer._release_forward_batch(state)
                observer._release_forward_batch(state)
        finally:
            done.set()
            reader.join(5)
        self.assertFalse(reader.is_alive())
        self.assertEqual([], failures)
        self.assertTrue(samples)
        self.assertEqual(0, observer._capture_arrival_workload().request_count)
        self.assertEqual({}, observer._prepared_contributions)
        self.assertEqual({}, observer._prepared_workload.batch_request_counts)

    def test_failed_queued_workload_publication_retries_at_engine_admission(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        publish = observer._bridge.report_workload
        attempts = []

        def reject_first(*values, **options):
            attempts.append(values)
            return False if len(attempts) == 1 else publish(*values, **options)

        observer._bridge.report_workload = reject_first
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=-1,
            request_id="queued-retry",
        )
        native_batch = FakeForwardBatch(209, [request.request_id], [8])
        input_queue = SimpleNamespace(items=[])
        input_queue.put_nowait = input_queue.items.append
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda value: (value, 0),
            input_queue=input_queue,
        )
        with self.assertLogs("vllm.navi_sched.agent", level="ERROR"):
            observer._enqueue_forward_batch(
                native_batch,
                [(stable_request_id(request.request_id), request)],
                SimpleNamespace(ADD="add", ABORT="abort"),
            )
        state = observer._pending_forward_batches[native_batch.request_ids[0]]
        self.assertFalse(state.admission_reported)
        self.assertEqual(0, observer._bridge.workloads[-1][2])
        self.assertEqual(1, len(input_queue.items))
        scheduler.add_request(input_queue.items[0][1][0])
        self.assertTrue(state.admission_reported)
        self.assertEqual([209, 209], [values[29] for values in attempts])
        self.assertEqual(1, observer._bridge.workloads[-1][2])
        self.assertEqual(24, observer._bridge.workloads[-1][7])

    def test_pending_admission_preserves_running_decode_progress(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        running = self.make_request(
            prompt_tokens=32,
            computed_tokens=43,
            cached_tokens=0,
            request_id="running-decode",
        )
        running.num_output_tokens = 11
        running.max_tokens = 40
        scheduler._scheduler.requests = {running.request_id: running}
        scheduler._scheduler.running = [running]

        pending = self.make_request(
            prompt_tokens=32,
            computed_tokens=0,
            cached_tokens=-1,
            request_id="pending",
        )
        native_batch = FakeForwardBatch(208, [pending.request_id], [8])
        input_queue = SimpleNamespace(
            items=[], put_nowait=lambda item: input_queue.items.append(item)
        )
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda request: (request, 0),
            input_queue=input_queue,
        )
        observer._enqueue_forward_batch(
            native_batch,
            [(stable_request_id(pending.request_id), pending)],
            SimpleNamespace(ADD="add", ABORT="abort"),
        )

        observer._refresh_and_report_workload(
            scheduler._scheduler, (), "pending_with_decode"
        )

        workload = observer._current_workload
        self.assertEqual(1, workload.request_count)
        self.assertEqual(1, workload.running_request_count)
        self.assertEqual(1, workload.running_decode_request_count)
        self.assertEqual(11, workload.running_decode_generated_tokens)
        self.assertEqual(29, workload.running_decode_remaining_token_budget)
        self.assertEqual(2, observer._bridge.workloads[-1][2])
        self.assertEqual(1, observer._bridge.workloads[-1][34])
        self.assertEqual(11, observer._bridge.workloads[-1][35])
        self.assertEqual(29, observer._bridge.workloads[-1][36])

    def test_pending_admission_workload_retires_cancelled_requests_immediately(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer.create_scheduler(LifecycleScheduler)
        requests = [
            self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=-1,
                request_id=request_id,
            )
            for request_id in ("cancelled", "retained")
        ]
        native_batch = FakeForwardBatch(
            208, [request.request_id for request in requests], [8, 16]
        )
        input_queue = SimpleNamespace(
            items=[], put_nowait=lambda item: input_queue.items.append(item)
        )
        observer._engine_core = SimpleNamespace(
            preprocess_add_request=lambda request: (request, 0),
            input_queue=input_queue,
            aborts_queue=SimpleNamespace(put_nowait=lambda unused: None),
        )
        request_type = SimpleNamespace(ADD="add", ABORT="abort")
        observer._enqueue_forward_batch(
            native_batch,
            [(stable_request_id(request.request_id), request) for request in requests],
            request_type,
        )

        before = observer._capture_arrival_workload()
        self.assertEqual(
            (1, 2, 24, 40),
            (
                before.batch_count,
                before.request_count,
                before.remaining_prefill_context_tokens,
                before.remaining_prefill_tokens,
            ),
        )

        cancelled_request_id = stable_request_id("cancelled")
        observer._bridge.cancelled_requests = [cancelled_request_id]
        observer._drain_cancelled_requests(request_type)

        partial = observer._capture_arrival_workload()
        self.assertEqual(
            (1, 1, 16, 16),
            (
                partial.batch_count,
                partial.request_count,
                partial.remaining_prefill_context_tokens,
                partial.remaining_prefill_tokens,
            ),
        )
        self.assertEqual(
            [(208, [cancelled_request_id])],
            observer._bridge.aborted_forward_requests,
        )
        self.assertEqual([], observer._bridge.released_forward_batches)

        retained_request_id = stable_request_id("retained")
        observer._bridge.cancelled_requests = [retained_request_id]
        observer._drain_cancelled_requests(request_type)

        after = observer._capture_arrival_workload()
        self.assertEqual(
            (0, 0, 0, 0),
            (
                after.batch_count,
                after.request_count,
                after.remaining_prefill_context_tokens,
                after.remaining_prefill_tokens,
            ),
        )
        self.assertEqual([208], observer._bridge.released_forward_batches)

        observer._bridge.cancelled_requests = [retained_request_id]
        observer._drain_cancelled_requests(request_type)
        self.assertEqual([208], observer._bridge.released_forward_batches)

    def test_scheduler_exposes_native_instance_for_framework_profiling(self):
        import inspect

        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        self.assertIs(inspect.unwrap(scheduler), scheduler._scheduler)

    def test_waiting_identities_include_prepared_requests_and_exclude_running(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        running = self.make_request(
            prompt_tokens=32, computed_tokens=0, cached_tokens=0, request_id="running"
        )
        waiting = self.make_request(
            prompt_tokens=32, computed_tokens=0, cached_tokens=0, request_id="waiting"
        )
        scheduler.add_request(running)
        scheduler.add_request(waiting)
        scheduler._scheduler.running = [running]
        scheduler._scheduler.waiting = [waiting]
        observer._refresh_and_report_workload(
            scheduler._scheduler, (), "waiting_identity_test"
        )
        initial = observer._capture_arrival_workload()
        self.assertEqual((stable_request_id("waiting"),), initial.waiting_request_ids)

        state = self.add_forward_batch(observer, 101, ["prepared"])
        request = self.make_request(
            prompt_tokens=32, computed_tokens=0, cached_tokens=0, request_id="prepared"
        )
        observer._set_admission_state(state, [request], initial, 1)
        observer._native_ingress = True
        observer._bridge.report_workload = mock.Mock(return_value=True)
        core = SimpleNamespace(VLLM_WAITING_REQUEST_IDS_VERSION=1)
        expected = sorted([stable_request_id("waiting"), stable_request_id("prepared")])
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.naviSchedModule",
            return_value=core,
        ):
            observer._report_prepared_workload("prepared", state)
            self.assertEqual(
                expected,
                observer._bridge.report_workload.call_args.kwargs[
                    "waiting_request_ids"
                ],
            )
            self.assertEqual({}, observer._native_handoff_batches)
            observer._report_prepared_workload("after_handoff_ack")
            self.assertEqual(
                expected,
                observer._bridge.report_workload.call_args.kwargs[
                    "waiting_request_ids"
                ],
            )
            observer._remove_prepared_workload(state, state.request_ids)
            observer._report_prepared_workload("cancelled")
            self.assertEqual(
                [stable_request_id("waiting")],
                observer._bridge.report_workload.call_args.kwargs[
                    "waiting_request_ids"
                ],
            )

    def test_native_handoff_covers_all_batches_in_the_same_snapshot(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        first = self.add_forward_batch(observer, 101, ["first"])
        second = self.add_forward_batch(observer, 102, ["second"])
        for state, request_id in ((first, "first"), (second, "second")):
            request = self.make_request(
                prompt_tokens=32,
                computed_tokens=0,
                cached_tokens=0,
                request_id=request_id,
            )
            observer._set_admission_state(state, [request], _ObservedWorkload(), 1)
        observer._report_prepared_workload("both_prepared", first)
        report = observer._bridge.workloads[-1]
        self.assertEqual(2, report[2])
        self.assertEqual(101, report[29])
        self.assertEqual((101, 102), report[37])
        self.assertEqual({}, observer._native_handoff_batches)

    def test_failed_native_handoff_is_retried_without_covering_later_arrivals(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        state = self.add_forward_batch(observer, 101, ["first"])
        request = self.make_request(
            prompt_tokens=32, computed_tokens=0, cached_tokens=0, request_id="first"
        )
        observer._set_admission_state(state, [request], _ObservedWorkload(), 1)
        snapshot = observer._capture_arrival_workload()
        observer._bridge.report_workload = mock.Mock(return_value=False)
        with self.assertLogs("vllm.navi_sched.agent", level="ERROR"):
            self.assertFalse(
                observer._report_workload_locked(snapshot, "failed", state)
            )
        self.assertEqual({101: None}, observer._native_handoff_batches)
        second = self.add_forward_batch(observer, 102, ["second"])
        request = self.make_request(
            prompt_tokens=48, computed_tokens=0, cached_tokens=0, request_id="second"
        )
        observer._set_admission_state(second, [request], _ObservedWorkload(), 1)
        observer._bridge.report_workload = mock.Mock(return_value=True)
        self.assertTrue(
            observer._report_workload_locked(snapshot, "old_snapshot", state)
        )
        self.assertEqual((101,), observer._bridge.report_workload.call_args.args[37])
        self.assertEqual({102: None}, observer._native_handoff_batches)
        observer._report_prepared_workload("new_snapshot", second)
        self.assertEqual((102,), observer._bridge.report_workload.call_args.args[37])
        self.assertEqual(2, observer._bridge.report_workload.call_args.args[2])
        self.assertEqual({}, observer._native_handoff_batches)

    def test_admission_marker_is_reported_when_workload_is_unchanged(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        initial_report_count = len(observer._bridge.workloads)
        admitted_state = self.add_forward_batch(observer, 99, ["request"])

        observer._refresh_and_report_workload(
            scheduler._scheduler,
            (),
            "unchanged_admission",
            admitted_state=admitted_state,
        )

        self.assertEqual(initial_report_count + 1, len(observer._bridge.workloads))
        admitted = observer._bridge.workloads[-1]
        self.assertEqual(99, admitted[28])
        self.assertEqual(99, admitted[29])
        self.assertEqual(1, admitted[30])
        self.assertTrue(admitted[31])
        self.assertEqual([], admitted[32])

    def test_failed_admission_marker_is_retried_on_next_workload_report(self):
        for failure_mode in ("false", "exception"):
            with self.subTest(failure_mode=failure_mode):
                observer = VLLMEngineObserver(
                    self.make_config(), self.make_kv_cache_config()
                )
                scheduler = observer.create_scheduler(LifecycleScheduler)
                self.add_forward_batch(observer, 99, ["request"])
                attempts = []

                def report_workload(*values, **options):
                    attempts.append(values)
                    if len(attempts) == 1:
                        if failure_mode == "exception":
                            raise RuntimeError("injected admission report failure")
                        return False
                    return True

                observer._bridge.report_workload = report_workload
                request = self.make_request(
                    prompt_tokens=32,
                    computed_tokens=0,
                    cached_tokens=0,
                    request_id="request",
                )
                with self.assertLogs("vllm.navi_sched.agent", level="ERROR"):
                    scheduler.add_request(request)

                state = observer._forward_request_state("request")
                self.assertIsNotNone(state)
                self.assertTrue(state.admission_ready)
                self.assertFalse(state.admission_reported)
                self.assertEqual({}, observer._pending_forward_batches)
                self.assertIs(
                    state,
                    observer._scheduled_forward_batches[stable_request_id("request")],
                )
                self.assertEqual([99], [values[29] for values in attempts])

                observer._refresh_and_report_workload(
                    scheduler._scheduler, (), "admission_retry"
                )

                self.assertTrue(state.admission_reported)
                self.assertEqual([99, 99], [values[29] for values in attempts])

    def test_workload_publish_is_serialized_without_holding_state_lock(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        first_publish_entered = threading.Event()
        allow_first_publish = threading.Event()
        second_publish_finished = threading.Event()
        workload_read_finished = threading.Event()
        published = []
        workload_reader = None

        def report_workload(*values, **options):
            published.append(values)
            if len(published) == 1:
                first_publish_entered.set()
                self.assertTrue(allow_first_publish.wait(timeout=1.0))
            return True

        observer._bridge.report_workload = report_workload
        first_state = self.add_forward_batch(observer, 101, ["first"])
        second_state = self.add_forward_batch(observer, 102, ["second"])
        first = threading.Thread(
            target=observer._refresh_and_report_workload,
            args=(scheduler._scheduler, (), "first", first_state),
        )

        def publish_second():
            observer._report_prepared_workload("second", second_state)
            second_publish_finished.set()

        second = threading.Thread(target=publish_second)
        try:
            first.start()
            self.assertTrue(first_publish_entered.wait(timeout=1.0))

            def read_workload():
                observer._capture_arrival_workload()
                workload_read_finished.set()

            workload_reader = threading.Thread(target=read_workload)
            workload_reader.start()
            self.assertTrue(workload_read_finished.wait(timeout=0.2))
            second.start()
            self.assertFalse(second_publish_finished.wait(timeout=0.05))
        finally:
            allow_first_publish.set()
            first.join(timeout=1.0)
            second.join(timeout=1.0)
            if workload_reader is not None:
                workload_reader.join(timeout=1.0)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertIsNotNone(workload_reader)
        self.assertFalse(workload_reader.is_alive())
        self.assertEqual([101, 102], [values[29] for values in published])

    def test_invalid_workload_is_reported_and_rebuilt_without_clamping(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        scheduler = observer.create_scheduler(LifecycleScheduler)
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=8,
            cached_tokens=8,
            request_id="request",
        )
        scheduler._scheduler.add_request(request)
        observer._refresh_and_report_workload(
            scheduler._scheduler, [request.request_id], "initial"
        )
        observer._workload_aggregate.remaining_prefill_tokens = -1

        with self.assertLogs("vllm.navi_sched.agent", level="ERROR") as logs:
            observer._refresh_and_report_workload(scheduler._scheduler, (), "repair")

        self.assertIn(
            "invalid NaviSched vLLM workload; rebuilding from scheduler",
            "\n".join(logs.output),
        )
        repaired = observer._current_workload
        self.assertTrue(repaired.valid())
        self.assertEqual(24, repaired.remaining_prefill_tokens)

    def test_concurrent_execution_callbacks_settle_tokens_once(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=18,
            cached_tokens=0,
            request_id="decode",
        )
        first_output = SimpleNamespace(num_scheduled_tokens={request.request_id: 1})
        second_output = SimpleNamespace(num_scheduled_tokens={request.request_id: 2})
        futures = [Future(), Future()]
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: futures.pop(0),
            SimpleNamespace(requests={request.request_id: request}),
        )
        first = wrapped(first_output, non_block=True)
        second = wrapped(second_output, non_block=True)
        self.assertEqual(3, observer._active_scheduled_tokens)

        threads = [
            threading.Thread(target=future.set_result, args=("done",))
            for future in (first, second)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(0, observer._active_scheduled_tokens)
        self.assertEqual({}, observer._inflight_scheduled_tokens)
        self.assertEqual({}, observer._scheduled_work_by_output)
        self.assertEqual(0, observer._active_execution_count)

    def test_inconsistent_scheduled_tokens_rebuild_from_remaining_outputs(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        first = SimpleNamespace(num_scheduled_tokens={"request": 2})
        second = SimpleNamespace(num_scheduled_tokens={"request": 3})
        observer._register_scheduled_work(first)
        observer._register_scheduled_work(second)
        observer._active_scheduled_tokens = 1
        observer._inflight_scheduled_tokens["request"] = 1

        with self.assertLogs("vllm.navi_sched.agent", level="ERROR") as logs:
            observer._settle_scheduled_work(first)

        output = "\n".join(logs.output)
        self.assertIn("inconsistent NaviSched vLLM scheduled-token tracking", output)
        self.assertIn(
            "rebuilt inconsistent NaviSched vLLM scheduled-token aggregate", output
        )
        self.assertEqual({"request": 3}, observer._inflight_scheduled_tokens)
        self.assertEqual(3, observer._active_scheduled_tokens)
        observer._settle_scheduled_work(second)
        self.assertEqual({}, observer._inflight_scheduled_tokens)
        self.assertEqual(0, observer._active_scheduled_tokens)

    def test_inconsistent_execution_counter_is_reported(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        execution_step = _ExecutionStep(
            started_ns=0,
            execution_tracking_registered=True,
            prefill_tokens=1,
        )

        with self.assertLogs("vllm.navi_sched.agent", level="ERROR") as logs:
            observer._finish_execution_tracking(execution_step)

        self.assertIn(
            "inconsistent NaviSched vLLM execution tracking",
            "\n".join(logs.output),
        )
        self.assertEqual(0, observer._active_execution_count)
        self.assertEqual(0, observer._active_prefill_execution_count)

    def test_inconsistent_pending_prefill_step_is_reported(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        self.add_forward_batch(observer, 98, ["request"])
        request = self.make_request(
            prompt_tokens=32,
            computed_tokens=32,
            cached_tokens=0,
            request_id="request",
        )
        state = observer._admit_request(request)
        scheduler_output = SimpleNamespace(
            num_scheduled_tokens={request.request_id: 32}
        )
        future = Future()
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: future,
            SimpleNamespace(requests={request.request_id: request}),
        )
        wrapped(scheduler_output, non_block=True)
        self.assertEqual(1, state.pending_prefill_steps)
        state.pending_prefill_steps = 0

        with self.assertLogs("vllm.navi_sched.agent", level="ERROR") as logs:
            future.set_result("model-output")

        self.assertIn(
            "inconsistent NaviSched vLLM prefill-step tracking",
            "\n".join(logs.output),
        )
        self.assertTrue(state.prefill_complete)
        self.assertTrue(state.released)
        self.assertEqual(1, len(observer._bridge.finished))

    def test_finish_log_aligns_predicted_and_actual_cache_hits(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        self.add_forward_batch(observer, 99, ["request"], [32])
        request = self.make_request(
            prompt_tokens=64,
            computed_tokens=64,
            cached_tokens=16,
            request_id="request",
        )
        observer._admit_request(request)
        scheduler_output = SimpleNamespace(
            num_scheduled_tokens={request.request_id: 48}
        )
        future = Future()
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: future,
            SimpleNamespace(requests={request.request_id: request}),
        )
        wrapped(scheduler_output, non_block=True)
        with self._assert_async_logs("vllm.navi_sched.agent", level="INFO") as logs:
            future.set_result("model-output")
        output = "\n".join(logs.output)
        self.assertIn("predicted_cache_hit_tokens=32", output)
        self.assertIn("actual_cache_hit_tokens=16", output)
        self.assertIn("cache_hit_bias_tokens=-16", output)
        self.assertIn("cache_hit_mae_tokens=16", output)
        self.assertEqual(16, observer._bridge.finished[0][1][0][2])

    def test_multiple_master_batches_may_share_one_vllm_step(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        requests = {}
        states = []
        for batch_id, request_id in ((201, "first"), (202, "second")):
            self.add_forward_batch(observer, batch_id, [request_id])
            request = self.make_request(
                prompt_tokens=32,
                computed_tokens=32,
                cached_tokens=0,
                request_id=request_id,
            )
            requests[request_id] = request
            states.append(observer._admit_request(request))
        output = SimpleNamespace(
            num_scheduled_tokens={request_id: 32 for request_id in requests}
        )
        future = Future()
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: future,
            SimpleNamespace(requests=requests),
        )

        wrapped(output, non_block=True)
        self.assertEqual([201, 202], [value[0] for value in observer._bridge.started])
        self.assertEqual([], observer._bridge.finished)

        future.set_result("model-output")
        self.assertEqual(2, len(observer._bridge.finished))
        self.assertTrue(all(state.prefill_complete for state in states))
        self.assertTrue(all(state.released for state in states))

    def test_concurrent_chunk_futures_finish_batch_after_both_complete(self):
        observer = VLLMEngineObserver(
            self.make_config(), self.make_kv_cache_config(), max_concurrent_batches=2
        )
        self.add_forward_batch(observer, 203, ["request"])
        request = self.make_request(
            prompt_tokens=64,
            computed_tokens=32,
            cached_tokens=0,
            request_id="request",
        )
        state = observer._admit_request(request)
        futures = [Future(), Future()]
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: futures.pop(0),
            SimpleNamespace(requests={request.request_id: request}),
        )

        first_future = wrapped(
            SimpleNamespace(num_scheduled_tokens={request.request_id: 32}),
            non_block=True,
        )
        request.num_computed_tokens = 64
        second_future = wrapped(
            SimpleNamespace(num_scheduled_tokens={request.request_id: 32}),
            non_block=True,
        )
        second_future.set_result("second")
        self.assertEqual([], observer._bridge.finished)

        first_future.set_result("first")
        self.assertEqual(1, len(observer._bridge.finished))
        self.assertEqual(0, state.pending_prefill_steps)
        self.assertTrue(state.released)

    def test_batch_wall_time_includes_tail_waiting_while_decode_steps_complete(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        ids = [f"incident-{i}" for i in range(25)]
        state = self.add_forward_batch(observer, 8305280128304385264, ids)
        requests = {}
        for request_id in ids:
            request = self.make_request(
                prompt_tokens=32,
                computed_tokens=32,
                cached_tokens=0,
                request_id=request_id,
            )
            requests[request_id] = request
            observer._admit_request(request)
        requests["preceding-decode"] = self.make_request(
            prompt_tokens=32,
            computed_tokens=128,
            cached_tokens=0,
            request_id="preceding-decode",
        )
        origin = 1_000_000_000

        def complete_step(offset_ms, duration_ms, prefill_ids):
            scheduled = {request_id: 32 for request_id in prefill_ids}
            scheduled["preceding-decode"] = 1
            step = _ExecutionStep(started_ns=origin + int(offset_ms * 1_000_000))
            observer._prepare_execution_step(
                SimpleNamespace(num_scheduled_tokens=scheduled), requests.get, step
            )
            future = Future()
            future.set_result("completed")
            observer._complete_execution_step(
                step,
                future,
                completed_ns=step.started_ns + int(duration_ms * 1_000_000),
            )
            return step

        # Replay the observed prefill partition at 490/512 occupied slots. This
        # exercises observer timing; it is not a simulation of vLLM scheduling.
        with self._assert_async_logs("vllm.navi_sched.agent", level="INFO") as logs:
            complete_step(0, 100, ids[:22])
            self.assertEqual(22, len(state.completed_prefill_request_ids))
            self.assertFalse(state.prefill_complete)
            self.assertEqual([], observer._bridge.finished)
            for index in range(1, 291):
                step = complete_step(index * 250, 100, [])
                self.assertEqual(1, step.decode_request_count)
                self.assertEqual(0, step.prefill_request_count)
                self.assertEqual([], step.batch_progress)
                self.assertEqual(0, state.pending_prefill_steps)
                self.assertFalse(state.prefill_complete)
            complete_step(73280, 100, ids[22:23])
            complete_step(78930, 100, ids[23:24])
            self.assertEqual([], observer._bridge.finished)
            complete_step(83516, 386, ids[24:])
        self.assertTrue(state.prefill_complete)
        self.assertTrue(state.released)
        self.assertEqual(25, len(state.completed_prefill_request_ids))
        self.assertEqual(1, len(observer._bridge.finished))
        breakdown = observer._bridge.backend_batch_executions[0]
        self.assertEqual(83902.0, breakdown[0])
        self.assertEqual(686.0, breakdown[1])
        self.assertEqual(4, breakdown[2])
        self.assertEqual(800, breakdown[3])
        self.assertIn("wall_ms=83902.000", "\n".join(logs.output))
        self.assertIn("execute_model_sum_ms=686.000", "\n".join(logs.output))

    def test_recompute_does_not_delay_other_requests_first_prefill(self):
        observer = VLLMEngineObserver(
            self.make_config(), self.make_kv_cache_config(), max_concurrent_batches=2
        )
        self.add_forward_batch(observer, 204, ["first", "second"])
        first = self.make_request(
            prompt_tokens=32,
            computed_tokens=32,
            cached_tokens=0,
            request_id="first",
        )
        second = self.make_request(
            prompt_tokens=64,
            computed_tokens=32,
            cached_tokens=0,
            request_id="second",
        )
        state = None
        for request in (first, second):
            admitted = observer._admit_request(request)
            if admitted is not None:
                state = admitted
        self.assertIsNotNone(state)
        futures = [Future(), Future(), Future()]
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: futures.pop(0),
            SimpleNamespace(requests={"first": first, "second": second}),
        )

        first_prefill = wrapped(
            SimpleNamespace(num_scheduled_tokens={"first": 32}),
            non_block=True,
        )
        first_prefill.set_result("first-prefill")
        self.assertEqual(
            {stable_request_id("first")}, state.completed_prefill_request_ids
        )
        self.assertEqual(0, state.pending_prefill_steps)
        self.assertIs(
            state,
            observer._scheduled_forward_batches.get(stable_request_id("first")),
        )

        first.num_computed_tokens = 32
        recompute = wrapped(
            SimpleNamespace(num_scheduled_tokens={"first": 32}),
            non_block=True,
        )
        self.assertEqual(0, state.pending_prefill_steps)
        second.num_computed_tokens = 64
        second_prefill = wrapped(
            SimpleNamespace(num_scheduled_tokens={"second": 32}),
            non_block=True,
        )
        self.assertEqual(1, state.pending_prefill_steps)
        second_prefill.set_result("second-prefill")

        self.assertEqual(1, len(observer._bridge.finished))
        self.assertTrue(state.released)
        recompute.set_result("recompute")
        self.assertEqual(1, len(observer._bridge.finished))

    def test_execution_source_records_first_forwarded_read_and_keeps_unknown(self):
        for stats, expected in (
            (
                SimpleNamespace(
                    num_local_cached_tokens=8, num_external_cached_tokens=8
                ),
                8,
            ),
            (
                SimpleNamespace(
                    num_local_cached_tokens=16, num_external_cached_tokens=0
                ),
                0,
            ),
            (
                SimpleNamespace(
                    num_local_cached_tokens=8, num_external_cached_tokens=0
                ),
                -1,
            ),
            (None, -1),
        ):
            with self.subTest(expected=expected, stats=stats):
                observer = VLLMEngineObserver(
                    self.make_config(), self.make_kv_cache_config()
                )
                state = self.add_forward_batch(observer, 177, ["source-request"])
                scheduler = observer.create_scheduler(LifecycleScheduler, chunk_size=16)
                request = self.make_request(
                    prompt_tokens=64,
                    computed_tokens=16,
                    cached_tokens=16,
                    request_id="source-request",
                )
                request.shared_prefix_boundary = 0
                request.prefill_stats = stats
                scheduler.add_request(request)
                request.num_computed_tokens = 32
                self.run_execution_step(
                    observer,
                    SimpleNamespace(num_scheduled_tokens={request.request_id: 16}),
                    scheduler.requests,
                )
                request_id = stable_request_id(request.request_id)
                self.assertEqual(
                    expected, state.initial_external_cache_hits[request_id]
                )
                self.assertEqual(
                    [(request_id, 64, 16, 0, expected)],
                    observer._bridge.resolved_execution_cache[0][1],
                )
                # A later chunk has resident context, but must not erase its first read.
                request.prefill_stats = SimpleNamespace(
                    num_local_cached_tokens=32, num_external_cached_tokens=0
                )
                request.num_computed_tokens = 64
                self.run_execution_step(
                    observer,
                    SimpleNamespace(num_scheduled_tokens={request.request_id: 32}),
                    scheduler.requests,
                )
                self.assertEqual(
                    expected, state.initial_external_cache_hits[request_id]
                )
                self.assertEqual(1, len(observer._bridge.resolved_execution_cache))
                self.assertEqual(1, len(observer._bridge.finished))
                self.assertEqual(
                    [(request_id, 64, 16)], observer._bridge.finished[0][1]
                )

    def test_native_training_uses_measured_geometry_not_dispatch_projection(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
            },
        ):
            observer = VLLMEngineObserver(
                self.make_config(), self.make_kv_cache_config()
            )
        observer._execution_geometry_enabled = True
        observer._prefill_execution_policy = object()
        state = self.add_forward_batch(observer, 77, ["internal-measured"])
        request_id = state.request_ids[0]
        request = self.make_request(
            prompt_tokens=100,
            computed_tokens=100,
            cached_tokens=0,
            request_id="internal-measured",
        )
        state.requests = {request_id: request}
        state.initial_cache_hits = {request_id: 0}
        state.initial_execution_policy_inputs = {request_id: (100, 0, 0)}
        state.handle = object()
        state.execution_started_ns = time.perf_counter_ns() - 10_000_000
        state.execution_step_count = 1
        state.execution_step_ms = 10.0
        state.execution_prefill_request_count = 1
        state.execution_prefill_tokens = 100
        state.geometry_execution_step_count = 1
        state.global_padded_tokens = 128
        state.prefill_complete = True
        observer._bridge.project_prefill_execution_geometry = mock.Mock(
            return_value=(9, 4096)
        )
        measured = observer._bridge.finish_batch_with_execution_geometry = mock.Mock(
            return_value=True
        )
        projected = observer._bridge.finish_batch_with_projected_execution_geometry = (
            mock.Mock(return_value=True)
        )
        finished_us = time.time_ns() // 1000
        observer._finish_batch(state, time.perf_counter_ns(), finished_us)
        measured.assert_called_once()
        self.assertEqual((1, 128, finished_us), measured.call_args.args[-3:])
        projected.assert_not_called()
        observer._bridge.project_prefill_execution_geometry.assert_not_called()

    def test_cache_source_fingerprint_requires_execution_geometry(self):
        base = _latency_model_fingerprint(self.make_config())
        policy = SimpleNamespace(
            block_size=768, linear_step=8, hybrid_tail_required=True
        )
        with self.assertRaises(ValueError):
            _native_pd_latency_model_fingerprint(
                base, policy, execution_cache_source_enabled=True
            )
        old = _native_pd_latency_model_fingerprint(
            base,
            policy,
            execution_geometry_enabled=True,
            projected_execution_geometry_enabled=True,
        )
        source = _native_pd_latency_model_fingerprint(
            base,
            policy,
            execution_geometry_enabled=True,
            projected_execution_geometry_enabled=True,
            execution_cache_source_enabled=True,
        )
        self.assertNotEqual(old, source)
        self.assertEqual(
            "split_native_start_cache_source_geometry_v34",
            json.loads(source)["navi_latency_mode"],
        )
        self.assertEqual(
            "native_start_call_v1", json.loads(source)["execution_start_boundary"]
        )
        self.assertEqual(
            1, json.loads(source)["execution_cache_source_feature_protocol"]
        )
        measured = _native_pd_latency_model_fingerprint(
            base,
            policy,
            execution_geometry_enabled=True,
            projected_execution_geometry_enabled=False,
            execution_cache_source_enabled=True,
        )
        decoded = json.loads(measured)
        self.assertNotEqual(source, measured)
        self.assertEqual(
            "split_native_start_measured_cache_source_geometry_v35",
            decoded["navi_latency_mode"],
        )
        self.assertFalse(decoded["projected_execution_geometry_enabled"])
        self.assertEqual("observed_runner", decoded["training_geometry_origin"])
        self.assertEqual(
            "all_cache_resolved_forecast", decoded["dispatch_geometry_origin"]
        )

    def test_master_batch_spans_aligned_prefill_chunks_and_reports_once(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        self.add_forward_batch(observer, 77, ["internal-1", "internal-2"])

        class FakeScheduler:
            def __init__(self, **unused):
                self.requests = {}
                self.running = []
                self.max_num_running_reqs = 8
                self.max_num_scheduled_tokens = 1024
                self.scheduler_config = SimpleNamespace(
                    enable_chunked_prefill=True,
                    long_prefill_token_threshold=0,
                )
                self.kv_cache_manager = SimpleNamespace(block_pool=FakeBlockPool())

            def add_request(self, request):
                self.requests[request.request_id] = request

            def has_requests(self):
                return any(
                    request.num_computed_tokens < request.num_prompt_tokens
                    for request in self.requests.values()
                )

            def schedule(self):
                scheduled = {}
                for request in self.requests.values():
                    remaining = request.num_prompt_tokens - request.num_computed_tokens
                    if remaining <= 0:
                        continue
                    chunk = min(32, remaining)
                    scheduled[request.request_id] = chunk
                    request.num_computed_tokens += chunk
                return SimpleNamespace(num_scheduled_tokens=scheduled)

        scheduler = observer.create_scheduler(FakeScheduler)
        first = self.make_request(
            prompt_tokens=64,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-1",
        )
        second = self.make_request(
            prompt_tokens=96,
            computed_tokens=0,
            cached_tokens=0,
            request_id="internal-2",
        )
        scheduler.add_request(first)
        scheduler.add_request(second)

        batch_state = observer._forward_request_state("internal-1")
        for expected_finished in (0, 0, 1):
            output = scheduler.schedule()
            self.run_execution_step(observer, output, scheduler.requests)
            self.assertEqual(expected_finished, len(observer._bridge.finished))

        self.assertEqual(1, len(observer._bridge.started))
        self.assertEqual(3, batch_state.chunk_count)
        self.assertEqual(
            [
                (stable_request_id("internal-1"), 64, 0),
                (stable_request_id("internal-2"), 96, 0),
            ],
            observer._bridge.finished[0][1],
        )

    def test_master_batch_spans_request_subset_iterations_and_reports_once(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        external_ids = ["internal-1", "internal-2", "internal-3"]
        self.add_forward_batch(observer, 78, external_ids)

        class FakeScheduler:
            def __init__(self, **unused):
                self.requests = {}
                self.running = []
                self.max_num_running_reqs = 8
                self.max_num_scheduled_tokens = 128
                self.scheduler_config = SimpleNamespace(
                    enable_chunked_prefill=True,
                    long_prefill_token_threshold=0,
                )
                self.kv_cache_manager = SimpleNamespace(block_pool=FakeBlockPool())

            def add_request(self, request):
                self.requests[request.request_id] = request

            def has_requests(self):
                return any(
                    request.num_computed_tokens < request.num_prompt_tokens
                    for request in self.requests.values()
                )

            def schedule(self):
                for request in self.requests.values():
                    remaining = request.num_prompt_tokens - request.num_computed_tokens
                    if remaining <= 0:
                        continue
                    request.num_computed_tokens += remaining
                    return SimpleNamespace(
                        num_scheduled_tokens={request.request_id: remaining}
                    )
                return SimpleNamespace(num_scheduled_tokens={})

        scheduler = observer.create_scheduler(FakeScheduler)
        requests = [
            self.make_request(
                prompt_tokens=128,
                computed_tokens=16,
                cached_tokens=16,
                request_id="internal-1",
            ),
            self.make_request(
                prompt_tokens=160,
                computed_tokens=32,
                cached_tokens=32,
                request_id="internal-2",
            ),
            self.make_request(
                prompt_tokens=96,
                computed_tokens=0,
                cached_tokens=0,
                request_id="internal-3",
            ),
        ]
        for request in requests:
            scheduler.add_request(request)

        batch_state = observer._forward_request_state("internal-1")
        for expected_finished in (0, 0, 1):
            output = scheduler.schedule()
            self.assertEqual(1, len(output.num_scheduled_tokens))
            self.run_execution_step(observer, output, scheduler.requests)
            self.assertEqual(expected_finished, len(observer._bridge.finished))

        self.assertEqual(1, len(observer._bridge.started))
        self.assertEqual(3, batch_state.chunk_count)
        self.assertEqual(
            [
                (stable_request_id("internal-1"), 128, 16),
                (stable_request_id("internal-2"), 160, 32),
                (stable_request_id("internal-3"), 96, 0),
            ],
            observer._bridge.finished[0][1],
        )
        self.assertTrue(batch_state.released)

    def test_request_without_forwarded_master_batch_fails_navi_response(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._engine_core = SimpleNamespace(
            engine_index=7,
            output_queue=SimpleNamespace(
                put_nowait=mock.Mock(
                    side_effect=AssertionError("must not use native output socket")
                )
            ),
        )
        scheduler = observer.create_scheduler(LifecycleScheduler)
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=0,
            cached_tokens=0,
        )

        scheduler.add_request(request)

        self.assertEqual({}, scheduler._scheduler.requests)
        self.assertEqual(
            [
                (
                    stable_request_id(request.request_id),
                    "vLLM request has no NaviSched forwarded batch metadata",
                )
            ],
            observer._bridge.failed_responses,
        )
        observer._engine_core.output_queue.put_nowait.assert_not_called()

    def test_chunked_prefill_uses_prior_computed_tokens_as_cache(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=100,
            computed_tokens=100,
            cached_tokens=20,
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 40})

        self.run_execution_step(observer, output, {request.request_id: request})

        request_id = stable_request_id("internal-request")
        self.assertEqual(
            [(request_id, 100, 60)],
            observer._bridge.finished[0][1],
        )

    def test_request_audit_joins_exact_backend_step_without_changing_batches(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        with self._assert_async_logs(
            "vllm.navi_sched.agent", level="INFO"
        ) as delivered:
            for step in range(2):
                requests = [
                    self.make_request(
                        prompt_tokens=64,
                        computed_tokens=64,
                        cached_tokens=0,
                        request_id="step-%d-request-%d" % (step, i),
                    )
                    for i in range(2)
                ]
                output = SimpleNamespace(
                    num_scheduled_tokens={r.request_id: 64 for r in requests}
                )
                self.run_execution_step(
                    observer, output, {r.request_id: r for r in requests}
                )
        events = [
            e
            for e in observer._bridge.request_lifecycle_audits
            if e["event"] == "prefill_cache_observation"
        ]
        logs = [
            record.getMessage()
            for record in delivered.records
            if record.getMessage().startswith("[navi-vllm-execution-step]")
        ]
        self.assertEqual(4, len(events))
        self.assertEqual(2, len(logs))
        ids = [e["backend_step_started_ns"] for e in events]
        self.assertTrue(all(type(v) is int and v > 0 for v in ids))
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[2], ids[3])
        self.assertNotEqual(ids[0], ids[2])
        self.assertEqual(
            [ids[0], ids[2]],
            [int(message.rsplit("backend_step_started_ns=", 1)[1]) for message in logs],
        )
        self.assertTrue(all("backend_step_started_ns=" in message for message in logs))
        self.assertEqual(2, len(observer._bridge.finished))
        self.assertTrue(all(e["scheduled_prefill_tokens"] == 64 for e in events))

    def test_prefill_audit_separates_prefix_hit_from_chunk_context(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=100,
            computed_tokens=100,
            cached_tokens=20,
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 40})
        self.run_execution_step(observer, output, {request.request_id: request})
        events = [
            event
            for event in observer._bridge.request_lifecycle_audits
            if event["event"] == "prefill_cache_observation"
        ]
        self.assertEqual(1, len(events))
        self.assertEqual(20, events[0]["prefix_cache_hit_tokens"])
        self.assertEqual(60, events[0]["computed_context_tokens"])
        self.assertEqual(40, events[0]["scheduled_prefill_tokens"])
        self.assertEqual("local", events[0]["sample_scope"])
        # Execution features retain prior chunk work as context.
        self.assertEqual(
            [(stable_request_id(request.request_id), 100, 60)],
            observer._bridge.finished[0][1],
        )

    def test_prefill_audit_records_explicit_checkpoint_without_changing_execution(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=100, computed_tokens=40, cached_tokens=0
        )
        params = {
            "cache_control_params": {"token_positions": [12, 35, 99]},
            "private_header": "must-not-be-audited",
        }
        request.sampling_params = SimpleNamespace(extra_args=params)
        request.shared_prefix_boundary = 96
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 40})
        self.run_execution_step(observer, output, {request.request_id: request})
        event = next(
            v
            for v in observer._bridge.request_lifecycle_audits
            if v["event"] == "prefill_cache_observation"
        )
        self.assertTrue(event["explicit_checkpoint_positions_present"])
        self.assertEqual(3, event["explicit_checkpoint_valid_position_count"])
        self.assertEqual(99, event["explicit_checkpoint_max_position"])
        self.assertEqual(
            99 // observer._block_size * observer._block_size,
            event["explicit_checkpoint_requested_boundary"],
        )
        self.assertEqual(96, event["shared_prefix_boundary"])
        self.assertEqual(
            [(stable_request_id(request.request_id), 40, 0)],
            observer._bridge.finished[0][1],
        )
        self.assertEqual(
            [12, 35, 99], params["cache_control_params"]["token_positions"]
        )
        self.assertNotIn("private_header", event)
        self.assertNotIn("token_positions", event)
        self.assertNotIn("sampling_params", event)

    def test_explicit_checkpoint_audit_matches_numeric_directive_boundaries(self):
        cases = (
            (None, False, 0, 0, 0),
            ({}, False, 0, 0, 0),
            ({"cache_control_params": {"token_positions": "6144"}}, False, 0, 0, 0),
            ({"cache_control_params": {"token_positions": []}}, True, 0, 0, 0),
            (
                {
                    "cache_control_params": {
                        "token_positions": [True, -1, 0, 6143.0, "6144", 24913]
                    }
                },
                True,
                0,
                0,
                0,
            ),
            (
                {
                    "cache_control_params": {
                        "token_positions": [1, 6143, 6144, 18433, 24912]
                    }
                },
                True,
                5,
                24912,
                24576,
            ),
            (
                {"cache_control_params": {"token_positions": [18433, 1]}},
                True,
                2,
                18433,
                18432,
            ),
            ({"cache_control_params": {"token_positions": [6143]}}, True, 1, 6143, 0),
        )
        for extra, present, count, maximum, boundary in cases:
            with self.subTest(extra=extra):
                request = SimpleNamespace(
                    num_prompt_tokens=24912,
                    sampling_params=SimpleNamespace(extra_args=extra),
                )
                self.assertEqual(
                    {
                        "explicit_checkpoint_positions_present": present,
                        "explicit_checkpoint_valid_position_count": count,
                        "explicit_checkpoint_max_position": maximum,
                        "explicit_checkpoint_requested_boundary": boundary,
                    },
                    VLLMEngineObserver._explicit_checkpoint_audit_fields(request, 6144),
                )

    def test_prefill_audit_cache_stats_compatibility(self):
        cases = (
            (
                "v029",
                SimpleNamespace(
                    num_local_cached_tokens=12, num_external_cached_tokens=8
                ),
                99,
                (12, 8, 20, "prefill_stats"),
            ),
            (
                "v029_zero",
                SimpleNamespace(
                    num_local_cached_tokens=0, num_external_cached_tokens=0
                ),
                99,
                (0, 0, 0, "prefill_stats"),
            ),
            ("legacy", "absent", 20, (-1, -1, 20, "request.num_cached_tokens")),
            ("unknown", "absent", None, (-1, -1, -1, "unknown")),
            ("consumed", None, None, (-1, -1, -1, "unknown")),
            ("consumed_legacy", None, 20, (-1, -1, 20, "request.num_cached_tokens")),
        )
        for name, stats, legacy, expected in cases:
            with self.subTest(name=name):
                observer = VLLMEngineObserver(
                    self.make_config(), self.make_kv_cache_config()
                )
                request = self.make_request(
                    prompt_tokens=100,
                    computed_tokens=100,
                    cached_tokens=20,
                )
                if legacy is None:
                    del request.num_cached_tokens
                else:
                    request.num_cached_tokens = legacy
                if stats != "absent":
                    request.prefill_stats = stats
                output = SimpleNamespace(num_scheduled_tokens={request.request_id: 40})
                self.run_execution_step(observer, output, {request.request_id: request})
                event = next(
                    event
                    for event in observer._bridge.request_lifecycle_audits
                    if event["event"] == "prefill_cache_observation"
                )
                if stats is not None and stats != "absent":
                    stats.num_local_cached_tokens = 999
                    request.prefill_stats = None
                self.assertEqual(
                    expected,
                    (
                        event["prefix_cache_hit_local_tokens"],
                        event["prefix_cache_hit_external_tokens"],
                        event["prefix_cache_hit_tokens"],
                        event["prefix_cache_hit_source"],
                    ),
                )
                self.assertEqual(
                    [(stable_request_id(request.request_id), 100, 60)],
                    observer._bridge.finished[0][1],
                )

    def test_native_chunk_reports_full_publication_prompt(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
            },
        ):
            observer = VLLMEngineObserver(
                self.make_config(), self.make_kv_cache_config()
            )
        request = self.make_request(
            prompt_tokens=137565, computed_tokens=18432, cached_tokens=0
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 18432})
        self.run_execution_step(observer, output, {request.request_id: request})
        self.assertEqual(
            {stable_request_id(request.request_id): 137565},
            observer._bridge.prefix_cache_prompt_tokens,
        )
        self.assertEqual(18432, observer._bridge.started[0][1][0][1])

    def test_matching_forward_batches_deduplicates_and_filters_live_requests(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        state = SimpleNamespace(
            lock=threading.RLock(),
            batch_id=17,
            external_request_ids={1: "a", 2: "b"},
            completed_prefill_request_ids={1},
            pending_prefill_steps=3,
            released=False,
        )
        released = SimpleNamespace(lock=threading.RLock(), released=True)
        observer._pending_forward_batches.update({1: state, 2: state})
        observer._scheduled_forward_batches.update({1: state, 3: released})
        self.assertEqual([state], observer._matching_forward_batches(None))
        self.assertEqual([state], observer._matching_forward_batches({"b"}))
        self.assertEqual([], observer._matching_forward_batches({"unknown"}))
        self.assertEqual({1}, state.completed_prefill_request_ids)
        self.assertEqual(3, state.pending_prefill_steps)
        self.assertFalse(state.released)

    def test_matching_forward_batches_rechecks_release_under_state_lock(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        state = SimpleNamespace(released=False, external_request_ids={1: "a"})

        @contextlib.contextmanager
        def finish_before_lock_acquired():
            state.released = True
            yield

        state.lock = finish_before_lock_acquired()
        observer._scheduled_forward_batches[1] = state
        self.assertEqual([], observer._matching_forward_batches({"a"}))
        self.assertIs(state, observer._scheduled_forward_batches[1])

    def test_native_pipeline_slots_come_from_bound_stepper(self):
        class LazyPushStepper:
            def step(self):
                pass

        stepper = LazyPushStepper()
        stepper.stages = 4
        stepper.in_flight = []
        core = SimpleNamespace(
            model_executor=SimpleNamespace(),
            scheduler=SimpleNamespace(connector=None),
            step_fn=stepper.step,
            batch_queue=stepper.in_flight,
            batch_queue_size=8,
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(pipeline_parallel_size=4)
            ),
        )
        stepper.core = core
        observer = VLLMEngineObserver.__new__(VLLMEngineObserver)
        observer._bridge = mock.Mock()
        module = SimpleNamespace(LazyPushStepper=LazyPushStepper)
        with mock.patch.dict(sys.modules, {"vllm_torchtpu.core.pp_lazy_step": module}):
            observer._bind_backend_pipeline_stages(core)
            observer._bridge.set_backend_execution_pipeline_stages.assert_called_once_with(
                4
            )
            for changes in (
                {"stages": 0},
                {"stages": True},
                {"stages": 3},
                {"core": object()},
                {"in_flight": []},
            ):
                with mock.patch.multiple(stepper, **changes):
                    with self.assertRaisesRegex(RuntimeError, "inconsistent native"):
                        observer._bind_backend_pipeline_stages(core)
            observer._bridge.set_backend_execution_pipeline_stages.return_value = False
            with self.assertRaisesRegex(RuntimeError, "failed to bind"):
                observer._bind_backend_pipeline_stages(core)

    def test_ordinary_engine_retains_single_stage_utilization(self):
        observer = VLLMEngineObserver.__new__(VLLMEngineObserver)
        observer._bridge = mock.Mock()
        observer._bind_backend_pipeline_stages(SimpleNamespace(step_fn=lambda: None))
        observer._bridge.set_backend_execution_pipeline_stages.assert_not_called()

    def test_bind_preserves_vllm_batch_queue_before_event_loop(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
            },
        ):
            observer = VLLMEngineObserver(
                self.make_config(),
                self.make_kv_cache_config(),
                max_concurrent_batches=2,
            )
        queue = []
        core = SimpleNamespace(
            model_executor=SimpleNamespace(),
            scheduler=SimpleNamespace(connector=None),
            batch_queue=queue,
            batch_queue_size=2,
            step=mock.Mock(),
            step_fn=mock.Mock(),
            output_queue=mock.Mock(),
        )
        core.step_with_batch_queue = core.step_fn
        serial_utils = types.ModuleType("vllm.v1.serial_utils")
        serial_utils.MsgpackEncoder = mock.Mock()
        with (
            mock.patch.dict(sys.modules, {"vllm.v1.serial_utils": serial_utils}),
            mock.patch.object(threading.Thread, "start"),
        ):
            observer.bind_engine_core(core)
        self.assertIs(queue, core.batch_queue)
        self.assertEqual(2, core.batch_queue_size)
        self.assertIs(core.step_with_batch_queue, core.step_fn)

    def test_bind_preserves_platform_step_loop_for_native_prefill(self):
        with mock.patch.dict(
            os.environ,
            {"VLLM_NAVI_SCHED_NATIVE_INGRESS": "1", "TPU_RAIDEN_JOB_NAME": "prefill"},
        ):
            observer = VLLMEngineObserver(
                self.make_config(),
                self.make_kv_cache_config(),
                max_concurrent_batches=2,
            )
        queue = []
        platform_step = mock.Mock()
        core = SimpleNamespace(
            model_executor=SimpleNamespace(),
            scheduler=SimpleNamespace(connector=None),
            batch_queue=queue,
            batch_queue_size=4,
            step=mock.Mock(),
            step_fn=platform_step,
            step_with_batch_queue=mock.Mock(),
            output_queue=mock.Mock(),
        )
        serial_utils = types.ModuleType("vllm.v1.serial_utils")
        serial_utils.MsgpackEncoder = mock.Mock()
        with (
            mock.patch.dict(sys.modules, {"vllm.v1.serial_utils": serial_utils}),
            mock.patch.object(threading.Thread, "start"),
        ):
            observer.bind_engine_core(core)
        self.assertIs(queue, core.batch_queue)
        self.assertIs(platform_step, core.step_fn)
        self.assertEqual(4, core.batch_queue_size)

    def test_native_policy_audit_survives_disabled_python_logging(self):
        policy = {
            "pcp_size": 8,
            "compile_sizes": [256, 512, 1024],
            "scheduler_block_size": 4096,
            "hash_block_size": 4096,
            "mamba_partial_cache_hit": None,
        }
        serial_utils = types.ModuleType("vllm.v1.serial_utils")
        serial_utils.MsgpackEncoder = mock.Mock()
        for enabled in (False, True):
            with self.subTest(native_ingress=enabled):
                observer = VLLMEngineObserver.__new__(VLLMEngineObserver)
                observer._engine_core = None
                observer._native_ingress = enabled
                observer._defer_agent_until_engine_bound = False
                observer._request_audit_session = "policy-test"
                observer._request_audit_failures = 0
                observer._bridge = mock.Mock()
                observer._bind_backend_pipeline_stages = mock.Mock()
                observer._enable_pp_kv_block_hashing = mock.Mock()
                observer._bind_v6d_cache_source = mock.Mock()
                observer._forward_batches = mock.Mock()
                queue = []
                step_fn = mock.Mock()
                core = SimpleNamespace(
                    model_executor=SimpleNamespace(),
                    batch_queue=queue, step_fn=step_fn,
                    output_queue=mock.Mock(), scheduler=object(),
                    vllm_config=object(),
                )
                with (
                    mock.patch.dict(sys.modules, {"vllm.v1.serial_utils": serial_utils}),
                    mock.patch.object(threading.Thread, "start"),
                    mock.patch("navi_sched.adapters.vllm.VLLMEngineObserver._execution_timing_probe.bind_engine_loop_audit"),
                    mock.patch("navi_sched.adapters.vllm.VLLMEngineObserver.logger.disabled", True),
                    mock.patch("navi_sched.adapters.vllm.VLLMExecutionGeometry.capture_execution_policy", return_value=policy) as capture,
                ):
                    observer.bind_engine_core(core)
                    with self.assertRaisesRegex(RuntimeError, "already bound"):
                        observer.bind_engine_core(core)
                events = [json.loads(c.args[0]) for c in observer._bridge.log_request_lifecycle_audit.call_args_list]
                self.assertEqual(1 if enabled else 0, len(events))
                if enabled:
                    capture.assert_called_once_with(core.scheduler, core.vllm_config)
                    self.assertEqual("native_execution_policy", events[0]["event"])
                    self.assertEqual(policy, events[0]["policy"])
                    self.assertEqual("policy-test", events[0]["session"])
                    self.assertEqual(0, events[0]["audit_failures"])
                else:
                    capture.assert_not_called()
                self.assertIs(queue, core.batch_queue)
                self.assertIs(step_fn, core.step_fn)

    def test_bind_rejects_nonempty_vllm_batch_queue(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
            },
        ):
            observer = VLLMEngineObserver(
                self.make_config(),
                self.make_kv_cache_config(),
                max_concurrent_batches=2,
            )
        original_step = mock.Mock()
        core = SimpleNamespace(
            model_executor=SimpleNamespace(),
            scheduler=SimpleNamespace(connector=None),
            batch_queue=[object()],
            step_fn=original_step,
            step_with_batch_queue=original_step,
        )
        serial_utils = types.ModuleType("vllm.v1.serial_utils")
        serial_utils.MsgpackEncoder = mock.Mock()
        with (
            mock.patch.dict(sys.modules, {"vllm.v1.serial_utils": serial_utils}),
            self.assertRaisesRegex(RuntimeError, "nonempty vLLM batch queue"),
        ):
            observer.bind_engine_core(core)

    def test_bind_preserves_vllm_batch_queue_for_unified_serving(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "0",
                "TPU_RAIDEN_JOB_NAME": "prefill",
                "DS_ENABLE_DSERV": "0",
            },
        ):
            observer = VLLMEngineObserver(
                self.make_config(),
                self.make_kv_cache_config(),
                max_concurrent_batches=2,
            )
        original_queue = []
        original_step_fn = mock.Mock()
        core = SimpleNamespace(
            model_executor=SimpleNamespace(),
            scheduler=SimpleNamespace(connector=None),
            batch_queue=original_queue,
            batch_queue_size=2,
            step=mock.Mock(),
            step_fn=original_step_fn,
            output_queue=mock.Mock(),
        )
        serial_utils = types.ModuleType("vllm.v1.serial_utils")
        serial_utils.MsgpackEncoder = mock.Mock()
        with (
            mock.patch.dict(sys.modules, {"vllm.v1.serial_utils": serial_utils}),
            mock.patch.object(threading.Thread, "start"),
        ):
            observer.bind_engine_core(core)
        self.assertIs(original_queue, core.batch_queue)
        self.assertEqual(2, core.batch_queue_size)
        self.assertIs(original_step_fn, core.step_fn)

    def test_recompute_prefill_is_not_reported_as_a_local_batch(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=100,
            computed_tokens=100,
            cached_tokens=20,
        )
        self.add_forward_batch(observer, 108, [request.request_id])
        state = observer._admit_request(request)
        observer._release_forward_batch(state)

        request.num_computed_tokens = 40
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 40})

        self.run_execution_step(observer, output, {request.request_id: request})
        self.assertEqual([], observer._bridge.started)
        self.assertEqual([], observer._bridge.finished)

    def test_chunked_prefill_only_reports_hashes_for_computed_blocks(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        hashes = [bytes([value]) * 32 for value in range(1, 7)]
        request = self.make_request(
            prompt_tokens=100,
            computed_tokens=40,
            cached_tokens=0,
            block_hashes=hashes,
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 40})

        self.run_execution_step(observer, output, {request.request_id: request})

        _, requests, _ = observer._bridge.started[0]
        self.assertEqual(
            [
                (
                    stable_request_id("internal-request"),
                    40,
                    16,
                    [
                        int.from_bytes(hashes[0][-8:], "big"),
                        int.from_bytes(hashes[1][-8:], "big"),
                    ],
                )
            ],
            requests,
        )

    def test_decode_only_step_is_not_reported(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=100,
            computed_tokens=101,
            cached_tokens=20,
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 1})

        self.run_execution_step(observer, output, {request.request_id: request})
        self.assertEqual([], observer._bridge.started)

    def test_request_id_hash_collision_is_rejected_before_batch_start(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        first = self.make_request(
            prompt_tokens=16,
            computed_tokens=16,
            cached_tokens=0,
            request_id="internal-1",
        )
        second = self.make_request(
            prompt_tokens=16,
            computed_tokens=16,
            cached_tokens=0,
            request_id="internal-2",
        )
        output = SimpleNamespace(
            num_scheduled_tokens={first.request_id: 16, second.request_id: 16}
        )

        with (
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.stable_request_id",
                return_value=1,
            ),
            self.assertRaisesRegex(ValueError, "unique request ids"),
        ):
            observer._prepare_execution_step(
                output,
                {
                    first.request_id: first,
                    second.request_id: second,
                }.get,
                _ExecutionStep(started_ns=0),
            )

        self.assertEqual([], observer._bridge.started)

    def test_runtime_block_size_is_derived_from_vllm(self):
        observer = VLLMEngineObserver(
            self.make_config(block_size=32),
            self.make_kv_cache_config(specs=[FullAttentionSpec(32)]),
        )

        self.assertEqual(32, observer._block_size)

    def test_prefix_caching_is_required(self):
        config = self.make_config()
        config.cache_config.enable_prefix_caching = False

        with self.assertRaisesRegex(ValueError, "requires vLLM prefix caching"):
            VLLMEngineObserver(config, self.make_kv_cache_config())

    def test_context_parallelism_uses_scheduler_block_size(self):
        observer = VLLMEngineObserver(
            self.make_config(block_size=16, dcp_size=2),
            self.make_kv_cache_config(specs=[FullAttentionSpec(16)]),
        )

        self.assertEqual(32, observer._block_size)
        observer.shutdown()

    def test_accepts_concurrent_vllm_execution_steps(self):
        observer = VLLMEngineObserver(
            self.make_config(),
            self.make_kv_cache_config(),
            max_concurrent_batches=2,
        )
        observer.shutdown()

    def test_wrapped_execute_model_owns_the_batch_lifecycle(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=16,
            cached_tokens=0,
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 16})
        future = Future()

        def execute_model(scheduler_output, non_block):
            self.assertIs(output, scheduler_output)
            self.assertTrue(non_block)
            return future

        wrapped = observer.wrap_execute_model(
            execute_model,
            SimpleNamespace(requests={request.request_id: request}),
        )
        self.assertIs(future, wrapped(output, non_block=True))
        self.assertEqual([1], observer._bridge.backend_execution_request_counts)
        self.assertEqual(1, len(observer._bridge.started))
        self.assertEqual([], observer._bridge.finished)

        future.set_result("model-output")

        self.assertEqual(1, len(observer._bridge.finished))
        self.assertEqual(["begin", "end"], observer._bridge.backend_execution_events)
        self.assertEqual(0, observer._bridge.backend_execution_count)
        self.assertEqual(1, len(observer._bridge.execution_steps))
        self.assertGreater(observer._bridge.execution_steps[0][0], 0)
        step_event = next(
            event
            for event in observer._bridge.request_lifecycle_audits
            if event["event"] == "execution_step"
        )
        self.assertGreaterEqual(step_event["pre_submit_ms"], 0)
        self.assertGreaterEqual(step_event["future_callback_ms"], 0)
        self.assertAlmostEqual(
            step_event["callback_elapsed_ms"],
            step_event["pre_submit_ms"] + step_event["future_callback_ms"],
            places=6,
        )

    def test_execution_capacity_uses_all_running_requests_not_step_subset(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        requests = [
            self.make_request(prompt_tokens=16, computed_tokens=16, cached_tokens=0)
            for _ in range(10)
        ]
        for index, request in enumerate(requests):
            request.request_id = str(index + 1)
        future = Future()
        scheduler = SimpleNamespace(
            requests={request.request_id: request for request in requests},
            running=requests,
            waiting=[SimpleNamespace(request_id="waiting")],
        )
        wrapped = observer.wrap_execute_model(
            lambda output, non_block: future,
            scheduler,
        )
        wrapped(
            SimpleNamespace(num_scheduled_tokens={requests[0].request_id: 16}),
            non_block=True,
        )
        self.assertEqual([10], observer._bridge.backend_execution_request_counts)
        future.set_result("model-output")
        self.assertEqual(["begin", "end"], observer._bridge.backend_execution_events)
        observer.shutdown()

    def test_backend_execution_timing_includes_decode_only_steps(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=17,
            cached_tokens=0,
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 1})
        future = Future()
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: future,
            SimpleNamespace(requests={request.request_id: request}),
        )

        self.assertIs(future, wrapped(output, non_block=True))
        self.assertEqual(1, observer._bridge.backend_execution_count)
        future.set_result("decode-output")

        self.assertEqual(["begin", "end"], observer._bridge.backend_execution_events)
        self.assertEqual(0, observer._bridge.backend_execution_count)
        self.assertEqual(1, len(observer._bridge.execution_steps))
        self.assertEqual((0, 1), observer._bridge.execution_steps[0][1:])

    def test_backend_execution_timing_waits_for_sampled_output(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        execute_future = Future()
        sample_future = Future()
        wrapped = observer.wrap_execute_model(
            lambda output, non_block: execute_future, SimpleNamespace(requests={})
        )
        wrapped(SimpleNamespace(num_scheduled_tokens={}), non_block=True)
        execute_future.set_result(None)
        self.assertEqual(1, observer._bridge.backend_execution_count)
        observer.observe_model_output(execute_future, sample_future)
        self.assertEqual(1, observer._bridge.backend_execution_count)
        sample_future.set_result("model-output")
        self.assertEqual(["begin", "end"], observer._bridge.backend_execution_events)
        self.assertEqual(0, observer._bridge.backend_execution_count)
        self.assertFalse(hasattr(execute_future, "_navi_backend_execution_step"))

    def test_async_output_keeps_forward_batch_running_until_output_completion(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        self.add_forward_batch(observer, 203, ["request"])
        request = self.make_request(
            prompt_tokens=32, computed_tokens=32, cached_tokens=0, request_id="request"
        )
        state = observer._admit_request(request)
        execution, output = Future(), Future()
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: execution,
            SimpleNamespace(requests={request.request_id: request}),
        )
        clock = [1_000_000_000]
        with mock.patch.object(time, "perf_counter_ns", side_effect=lambda: clock[0]):
            wrapped(
                SimpleNamespace(num_scheduled_tokens={request.request_id: 32}),
                non_block=True,
            )
            observer.observe_model_output(execution, output)
            clock[0] += 10_000_000
            execution.set_result(None)
            self.assertEqual([], observer._bridge.finished)
            self.assertFalse(state.prefill_complete)
            self.assertEqual(1, state.pending_prefill_steps)
            clock[0] += 90_000_000
            output.set_result("model-output")
        self.assertEqual(1, len(observer._bridge.finished))
        self.assertTrue(state.prefill_complete)
        self.assertEqual(0, state.pending_prefill_steps)
        self.assertEqual(100.0, state.execution_step_ms)
        self.assertEqual([(100.0, 32, 0)], observer._bridge.execution_steps)

    def test_backend_execution_output_may_be_registered_before_execute_completes(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        execute_future, output_future = Future(), Future()
        wrapped = observer.wrap_execute_model(
            lambda output, non_block: execute_future, SimpleNamespace(requests={})
        )
        wrapped(SimpleNamespace(num_scheduled_tokens={}), non_block=True)
        observer.observe_model_output(execute_future, output_future)
        execute_future.set_result(None)
        self.assertEqual(1, observer._bridge.backend_execution_count)
        output_future.set_result("output")
        self.assertEqual(["begin", "end"], observer._bridge.backend_execution_events)

    def test_async_batch_completion_does_not_depend_on_timing_collection_success(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=32, computed_tokens=32, cached_tokens=0
        )
        execution, output = Future(), Future()
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: execution,
            SimpleNamespace(requests={request.request_id: request}),
        )
        with mock.patch.object(
            observer._bridge, "begin_backend_execution", return_value=False
        ):
            wrapped(
                SimpleNamespace(num_scheduled_tokens={request.request_id: 32}),
                non_block=True,
            )
        execution.set_result(None)
        self.assertEqual([], observer._bridge.finished)
        observer.observe_model_output(execution, output)
        output.set_result("output")
        self.assertEqual(1, len(observer._bridge.finished))
        self.assertEqual(0, observer._active_execution_count)
        self.assertEqual(0, observer._bridge.backend_execution_count)

    def test_async_output_failure_releases_forward_batch_without_fitting(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                observer = VLLMEngineObserver(
                    self.make_config(), self.make_kv_cache_config()
                )
                self.add_forward_batch(observer, 203, ["request"])
                request = self.make_request(
                    prompt_tokens=32,
                    computed_tokens=32,
                    cached_tokens=0,
                    request_id="request",
                )
                state = observer._admit_request(request)
                execution, output = Future(), Future()
                wrapped = observer.wrap_execute_model(
                    lambda unused_output, non_block: execution,
                    SimpleNamespace(requests={request.request_id: request}),
                )
                wrapped(
                    SimpleNamespace(num_scheduled_tokens={request.request_id: 32}),
                    non_block=True,
                )
                execution.set_result(None)
                observer.observe_model_output(execution, output)
                self.assertFalse(state.released)
                if cancelled:
                    output.cancel()
                else:
                    output.set_exception(RuntimeError("sampling failed"))
                self.assertTrue(state.released)
                self.assertFalse(state.prefill_complete)
                self.assertEqual([], observer._bridge.finished)
                self.assertEqual([], observer._bridge.execution_steps)
                self.assertEqual(0, observer._bridge.backend_execution_count)

    def test_backend_execution_timing_handles_output_completion_variants(self):
        for completion in ("result", "cancel", "error", "synchronous"):
            with self.subTest(completion=completion):
                observer = VLLMEngineObserver(
                    self.make_config(), self.make_kv_cache_config()
                )
                execute_future, output_future = Future(), Future()
                wrapped = observer.wrap_execute_model(
                    lambda output, non_block: execute_future,
                    SimpleNamespace(requests={}),
                )
                wrapped(SimpleNamespace(num_scheduled_tokens={}), non_block=True)
                execute_future.set_result(None)
                if completion == "synchronous":
                    observer.observe_model_output(execute_future, "output")
                else:
                    observer.observe_model_output(execute_future, output_future)
                    if completion == "cancel":
                        output_future.cancel()
                    elif completion == "error":
                        output_future.set_exception(RuntimeError("sampling failed"))
                    else:
                        output_future.set_result("output")
                observer.observe_model_output(execute_future, output_future)
                self.assertEqual(
                    ["begin", "end"], observer._bridge.backend_execution_events
                )
                self.assertEqual(0, observer._bridge.backend_execution_count)
                self.assertFalse(
                    hasattr(execute_future, "_navi_backend_execution_step")
                )

    def test_backend_execution_no_sampling_uses_execute_future(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        future = Future()
        wrapped = observer.wrap_execute_model(
            lambda output, non_block: future, SimpleNamespace(requests={})
        )
        wrapped(SimpleNamespace(num_scheduled_tokens={}), non_block=True)
        observer.observe_model_output(future, future)
        future.set_result(None)
        self.assertEqual(["begin", "end"], observer._bridge.backend_execution_events)
        self.assertEqual(0, observer._bridge.backend_execution_count)

    def test_backend_execution_output_can_complete_before_execute_callback(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        future, output_future = Future(), Future()
        wrapped = observer.wrap_execute_model(
            lambda output, non_block: future, SimpleNamespace(requests={})
        )
        wrapped(SimpleNamespace(num_scheduled_tokens={}), non_block=True)
        observer.observe_model_output(future, output_future)
        output_future.set_result("output")
        future.set_result(None)
        self.assertEqual(["begin", "end"], observer._bridge.backend_execution_events)
        self.assertEqual(0, observer._bridge.backend_execution_count)

    def test_backend_execution_output_futures_may_finish_out_of_order(self):
        observer = VLLMEngineObserver(
            self.make_config(), self.make_kv_cache_config(), max_concurrent_batches=2
        )
        executions = [Future(), Future()]
        outputs = [Future(), Future()]
        iterator = iter(executions)
        wrapped = observer.wrap_execute_model(
            lambda output, non_block: next(iterator), SimpleNamespace(requests={})
        )
        for execute, output in zip(executions, outputs):
            wrapped(SimpleNamespace(num_scheduled_tokens={}), non_block=True)
            observer.observe_model_output(execute, output)
            execute.set_result(None)
        self.assertEqual(2, observer._bridge.backend_execution_count)
        outputs[1].set_result("second")
        self.assertEqual(1, observer._bridge.backend_execution_count)
        outputs[0].set_result("first")
        self.assertEqual(0, observer._bridge.backend_execution_count)
        self.assertEqual(
            ["begin", "begin", "end", "end"], observer._bridge.backend_execution_events
        )

    def test_backend_execution_timing_tracks_overlapping_futures(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        output = SimpleNamespace(num_scheduled_tokens={})
        futures = [Future(), Future()]
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: futures.pop(0),
            SimpleNamespace(requests={}),
        )

        first = wrapped(output, non_block=True)
        second = wrapped(output, non_block=True)
        self.assertEqual(2, observer._bridge.backend_execution_count)
        self.assertEqual(2, observer._active_execution_count)
        second.set_result("second")
        self.assertEqual(1, observer._bridge.backend_execution_count)
        self.assertEqual(1, observer._active_execution_count)
        first.set_result("first")

        self.assertEqual(
            ["begin", "begin", "end", "end"],
            observer._bridge.backend_execution_events,
        )
        self.assertEqual(0, observer._bridge.backend_execution_count)
        self.assertEqual(0, observer._active_execution_count)

    def test_backend_execution_timing_ends_after_synchronous_failure(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        output = SimpleNamespace(num_scheduled_tokens={})

        def fail_execute_model(unused_output, non_block):
            raise RuntimeError("model submission failed")

        wrapped = observer.wrap_execute_model(
            fail_execute_model,
            SimpleNamespace(requests={}),
        )
        with self.assertRaisesRegex(RuntimeError, "model submission failed"):
            wrapped(output, non_block=True)

        self.assertEqual(["begin", "end"], observer._bridge.backend_execution_events)
        self.assertEqual(0, observer._bridge.backend_execution_count)
        self.assertEqual(0, observer._active_execution_count)

    def test_execution_diagnostics_separate_prefill_decode_and_gap(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        prefill = self.make_request(
            prompt_tokens=64,
            computed_tokens=32,
            cached_tokens=0,
            request_id="prefill",
        )
        decode = self.make_request(
            prompt_tokens=16,
            computed_tokens=17,
            cached_tokens=0,
            request_id="decode",
        )
        self.add_forward_batch(observer, 91, [prefill.request_id])
        state = observer._admit_request(prefill)
        state.admission_state = _AdmissionState(
            preceding_workload=_ObservedWorkload(request_count=2)
        )
        state.admission_started_ns = 900_000_000
        requests = {prefill.request_id: prefill, decode.request_id: decode}
        futures = [Future(), Future()]
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: futures.pop(0),
            SimpleNamespace(requests=requests),
        )

        first_output = SimpleNamespace(num_scheduled_tokens={prefill.request_id: 32})
        with self._assert_async_logs("vllm.navi_sched.agent", level="INFO") as logs:
            with mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
                side_effect=[
                    1_000_000_000,
                    1_000_000_000,
                    1_000_000_000,  # shadow begin bracket
                    1_010_000_000,
                    1_100_000_000,
                    1_100_000_000,
                    1_100_000_000,  # shadow end bracket
                    1_150_000_000,
                    1_150_000_000,
                    1_150_000_000,  # shadow begin bracket
                    1_160_000_000,
                    1_350_000_000,
                    1_350_000_000,
                    1_350_000_000,  # shadow end bracket
                ],
            ):
                first_future = wrapped(first_output, non_block=True)
                first_future.set_result("first-chunk")

                prefill.num_computed_tokens = 64
                second_output = SimpleNamespace(
                    num_scheduled_tokens={
                        prefill.request_id: 32,
                        decode.request_id: 1,
                    }
                )
                second_future = wrapped(second_output, non_block=True)
                second_future.set_result("second-chunk")

        output = "\n".join(logs.output)
        self.assertIn(
            "batch_ids=91 tracked_batches=1 "
            "prefill_requests=1 prefill_tokens=32 decode_requests=1 "
            "decode_tokens=1 execute_model_ms=200.000 success=1",
            output,
        )
        self.assertIn(
            "batch_id=91 chunks=2 steps=2 prefill_request_steps=2 "
            "prefill_tokens=64 decode_request_steps=1 decode_tokens=1 "
            "wall_ms=350.000 execute_model_sum_ms=300.000 "
            "inter_step_gap_ms=50.000",
            output,
        )
        self.assertIn(
            "start_active_executions=0 start_active_prefill_executions=0 "
            "start_active_decode_executions=0 start_active_scheduled_tokens=0 "
            "start_running_requests=0 start_running_prefill_requests=0 "
            "start_running_decode_requests=0 "
            "start_running_recompute_requests=0 start_waiting_requests=0 "
            "start_preempted_requests=0 start_resumed_requests=0 "
            "start_step_prefill_requests=1 start_step_prefill_tokens=32 "
            "start_step_decode_requests=0 start_step_decode_tokens=0",
            output,
        )
        self.assertIn(
            "sample_batch_count=1 sample_total_prompt_tokens=64 "
            "sample_running_capacity_prefill_requests=0",
            output,
        )
        self.assertEqual(1, len(observer._bridge.finished))
        self.assertEqual(
            (
                350.0,
                300.0,
                2,
                64,
                1,
                1,
                450.0,
                2,
                1,
                0,
                0,
                0,
                0.0,
                0.0,
                1,
                0,
                64,
                4096.0,
                0.0,
                0.0,
                0,
                0,
                0,
                100.0,
                0.0,
            ),
            observer._bridge.backend_batch_executions[0],
        )

    def test_synchronous_execute_failure_releases_forwarded_batch(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=16,
            cached_tokens=0,
        )
        self.add_forward_batch(observer, 77, [request.request_id])
        state = observer._admit_request(request)
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 16})

        def fail_execute_model(unused_output, non_block):
            raise RuntimeError("model submission failed")

        wrapped = observer.wrap_execute_model(
            fail_execute_model,
            SimpleNamespace(requests={request.request_id: request}),
        )
        with self.assertRaisesRegex(RuntimeError, "model submission failed"):
            wrapped(output, non_block=True)

        self.assertTrue(state.released)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual([], observer._bridge.finished)
        observer._bridge.started.clear()
        gc.collect()
        self.assertEqual([True], observer._bridge.deleted)

    def test_asynchronous_forward_failure_releases_without_training(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=16,
            cached_tokens=0,
            request_id="failed-forward",
        )
        self.add_forward_batch(observer, 771, [request.request_id])
        state = observer._admit_request(request)
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 16})
        future = Future()
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: future,
            SimpleNamespace(requests={request.request_id: request}),
        )

        wrapped(output, non_block=True)
        future.set_exception(RuntimeError("model execution failed"))

        self.assertTrue(state.released)
        self.assertEqual([], observer._bridge.finished)
        self.assertEqual({}, observer._scheduled_forward_batches)

    def test_asynchronous_forward_cancel_releases_without_training(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=16,
            cached_tokens=0,
            request_id="cancelled-forward",
        )
        self.add_forward_batch(observer, 772, [request.request_id])
        state = observer._admit_request(request)
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 16})
        future = Future()
        wrapped = observer.wrap_execute_model(
            lambda unused_output, non_block: future,
            SimpleNamespace(requests={request.request_id: request}),
        )

        wrapped(output, non_block=True)
        future.cancel()

        self.assertTrue(state.released)
        self.assertEqual([], observer._bridge.finished)
        self.assertEqual({}, observer._scheduled_forward_batches)

    def test_failed_forward_batch_start_releases_reporter_immediately(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=16,
            cached_tokens=0,
        )
        self.add_forward_batch(observer, 78, [request.request_id])
        state = observer._admit_request(request)
        observer._bridge.start_forward_batch = mock.Mock(return_value=None)
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 16})

        observer._prepare_execution_step(
            output,
            {request.request_id: request}.get,
            _ExecutionStep(started_ns=0),
        )
        self.assertTrue(state.released)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual([78], observer._bridge.released_forward_batches)

    def test_forward_batch_start_exception_releases_reporter_immediately(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=16,
            cached_tokens=0,
        )
        self.add_forward_batch(observer, 79, [request.request_id])
        state = observer._admit_request(request)
        observer._bridge.start_forward_batch = mock.Mock(
            side_effect=RuntimeError("start failed")
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 16})

        with self.assertRaisesRegex(RuntimeError, "start failed"):
            observer._prepare_execution_step(
                output,
                {request.request_id: request}.get,
                _ExecutionStep(started_ns=0),
            )
        self.assertTrue(state.released)
        self.assertEqual({}, observer._scheduled_forward_batches)
        self.assertEqual([79], observer._bridge.released_forward_batches)

    def test_image_cache_aliases_reach_master_only_after_all_native_image_blocks(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        feature = lambda offset, length: SimpleNamespace(
            modality="image",
            mm_position=SimpleNamespace(offset=offset, length=length),
        )
        prediction = SimpleNamespace(
            prompt_token_ids=list(range(48)),
            mm_features=[feature(17, 1)],
        )
        hashes = tuple(value.to_bytes(32, "big") for value in (11, 12, 13, 14))
        actual = SimpleNamespace(
            prompt_token_ids=list(range(79)),
            mm_features=[feature(17, 32)],
            block_hashes=hashes,
        )
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMImageCacheMap.calculate_vllm_block_hashes",
            return_value=[101, 102, 103],
        ):
            observer._image_cache_map.register(71, prediction, actual, 16)
        observer.on_block_lifecycle_event(
            "stored", 0, (hashes[0], hashes[1], hashes[3])
        )
        self.assertEqual(
            [([11, 12, 14], [], [], [], [], [])], observer._bridge.cache_updates
        )
        observer.on_block_lifecycle_event("stored", 0, (hashes[2],))
        self.assertEqual(
            ([13, 102], [], [], [], [], []), observer._bridge.cache_updates[-1]
        )
        observer._remove_response_requests([71])
        self.assertEqual({}, observer._image_cache_map._owners)
        observer.on_block_lifecycle_event("removed", 0, (hashes[1],))
        self.assertEqual(
            ([], [12, 102], [], [], [], []), observer._bridge.cache_updates[-1]
        )
        self.assertEqual({}, observer._image_cache_map._records)

    def test_reports_group_aware_vllm_cache_lifecycle(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        block_hash = bytes(range(32))
        observer.on_block_lifecycle_event(
            "stored", 0, (block_hash,), "producer", "request_store"
        )
        observer.on_block_lifecycle_event(
            "removed", 0, (block_hash,), "consumer", "request_allocation"
        )

        wire_hash = 1736447835066146335
        self.assertEqual(
            [
                ([wire_hash], [], [], [], [], []),
                ([], [wire_hash], [], [], [], []),
            ],
            observer._bridge.cache_updates,
        )

    def test_block_lifecycle_listener_accepts_missing_cause_metadata(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        block_hash = bytes(range(32))

        observer.on_block_lifecycle_event("stored", 0, (block_hash,))

        self.assertEqual(1, len(observer._bridge.cache_lifecycle_audits))
        payload = json.loads(observer._bridge.cache_lifecycle_audits[0])
        self.assertIsNone(payload["cause_request_id"])
        self.assertEqual("unspecified", payload["cause"])
        self.assertEqual(1, len(observer._bridge.cache_updates))

    def test_scheduler_installs_block_lifecycle_listener(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())

        class Scheduler:
            def __init__(self, **unused):
                self.kv_cache_manager = SimpleNamespace(block_pool=FakeBlockPool())

        scheduler = observer.create_scheduler(Scheduler)
        listener = scheduler.kv_cache_manager.block_pool.listener
        self.assertEqual(observer.on_block_lifecycle_event, listener)

    def test_scheduler_uses_legacy_cache_observer_when_setter_absent(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())

        class Scheduler:
            def __init__(self, **unused):
                self.kv_cache_manager = SimpleNamespace(block_pool=SimpleNamespace())

        scheduler = observer.create_scheduler(Scheduler)
        cache_observer = scheduler.kv_cache_manager.block_pool.cache_observer
        self.assertEqual(observer.on_block_lifecycle_event, cache_observer.listener)
        cache_observer.stored(
            [SimpleNamespace(is_null=False)], [bytes(range(32))], 0, "producer"
        )
        self.assertEqual(
            [([1736447835066146335], [], [], [], [], [])],
            observer._bridge.cache_updates,
        )

    def test_scheduler_registers_native_block_events_without_shadow_attribute(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())

        class Scheduler:
            def __init__(self, **unused):
                self.kv_cache_manager = SimpleNamespace(block_pool=FakeBlockPool())

        scheduler = observer.create_scheduler(Scheduler)
        pool = scheduler.kv_cache_manager.block_pool
        self.assertEqual(observer.on_block_lifecycle_event, pool.listener)
        self.assertFalse(hasattr(pool, "cache_observer"))
        block_hash = bytes(range(32))
        pool.listener("stored", 0, (block_hash,))
        pool.listener("removed", 0, (block_hash,))
        pool.listener("stored", 0, (block_hash,))
        pool.listener("cleared", None, ())
        wire_hash = 1736447835066146335
        self.assertEqual(
            [
                ([wire_hash], [], [], [], [], []),
                ([], [wire_hash], [], [], [], []),
                ([wire_hash], [], [], [], [], []),
                ([], [wire_hash], [], [], [], []),
            ],
            observer._bridge.cache_updates,
        )

    def test_cache_lifecycle_tracks_duplicate_blocks_and_clear(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        block_hash = bytes(range(32))

        observer.on_block_lifecycle_event(
            "stored",
            0,
            (block_hash, block_hash),
            "producer",
            "request_store",
        )
        observer.on_block_lifecycle_event(
            "removed", 0, (block_hash,), "first", "request_allocation"
        )
        self.assertEqual(1, len(observer._bridge.cache_updates))

        observer.on_block_lifecycle_event(
            "removed", 0, (block_hash,), "second", "request_allocation"
        )
        observer.on_block_lifecycle_event(
            "stored", 0, (block_hash,), "third", "request_store"
        )
        observer.on_block_lifecycle_event(
            "cleared", None, (), None, "prefix_cache_reset"
        )

        wire_hash = 1736447835066146335
        self.assertEqual(
            [
                ([wire_hash], [], [], [], [], []),
                ([], [wire_hash], [], [], [], []),
                ([wire_hash], [], [], [], [], []),
                ([], [wire_hash], [], [], [], []),
            ],
            observer._bridge.cache_updates,
        )
        payloads = [
            json.loads(message) for message in observer._bridge.cache_lifecycle_audits
        ]
        self.assertEqual([1, 2, 3, 4, 5], [row["sequence"] for row in payloads])
        self.assertEqual(
            ["stored", "removed", "removed", "stored", "cleared"],
            [row["event_type"] for row in payloads],
        )
        self.assertEqual([wire_hash, wire_hash], payloads[0]["event_hashes"])
        self.assertEqual("producer", payloads[0]["cause_request_id"])
        self.assertEqual("request_store", payloads[0]["cause"])
        self.assertEqual(0, payloads[0]["group_id"])
        self.assertGreater(payloads[0]["event_time_us"], 0)
        self.assertEqual([], payloads[1]["removed"])
        self.assertEqual([wire_hash], payloads[2]["removed"])
        self.assertEqual([wire_hash], payloads[3]["added"])
        self.assertEqual([wire_hash], payloads[4]["removed"])
        self.assertEqual([], payloads[4]["event_hashes"])

    def test_cache_lifecycle_reports_cause_batch_when_available(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        observer._forward_request_states["request-a"] = SimpleNamespace(
            batch_id=17, schedule_batch_id=29
        )

        observer.on_block_lifecycle_event(
            "stored",
            0,
            (bytes(range(32)),),
            "request-a",
            "request_store",
        )

        self.assertEqual(1, len(observer._bridge.cache_lifecycle_audits))
        payload = json.loads(observer._bridge.cache_lifecycle_audits[0])
        self.assertEqual("request-a", payload["cause_request_id"])
        self.assertEqual("request_store", payload["cause"])
        self.assertEqual(17, payload["cause_batch_id"])
        self.assertEqual(29, payload["cause_schedule_batch_id"])

    def test_shutdown_stops_agent(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        agent = observer._agent

        observer.shutdown()

        self.assertTrue(agent.stopped)
        self.assertIsNone(observer._agent)

    def test_shutdown_waits_for_forward_thread_without_timeout(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        forward_thread = mock.Mock()
        observer._forward_thread = forward_thread

        observer.shutdown()

        forward_thread.join.assert_called_once_with()

    def test_concurrent_shutdown_stops_forwarding_and_agent_once(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        agent = observer._agent
        stop_entered = threading.Event()
        stop_may_finish = threading.Event()
        original_stop = agent.stop
        observer._bridge.disable_forwarding = mock.Mock(
            wraps=observer._bridge.disable_forwarding
        )

        def blocking_stop():
            stop_entered.set()
            self.assertTrue(stop_may_finish.wait(2.0))
            original_stop()

        agent.stop = mock.Mock(side_effect=blocking_stop)
        first = threading.Thread(target=observer.shutdown)
        second = threading.Thread(target=observer.shutdown)

        first.start()
        self.assertTrue(stop_entered.wait(2.0))
        second.start()
        self.assertTrue(second.is_alive())
        stop_may_finish.set()
        first.join(2.0)
        second.join(2.0)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        observer._bridge.disable_forwarding.assert_called_once_with()
        agent.stop.assert_called_once_with()
        self.assertIsNone(observer._agent)

    def test_failed_batch_releases_native_handle(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=16,
            cached_tokens=0,
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 16})
        future = Future()
        wrapped = observer.wrap_execute_model(
            lambda scheduler_output, non_block: future,
            SimpleNamespace(requests={request.request_id: request}),
        )
        wrapped(output, non_block=True)

        future.set_exception(RuntimeError("model execution failed"))
        observer._bridge.started.clear()
        gc.collect()

        self.assertEqual([True], observer._bridge.deleted)

    def test_cancelled_batch_releases_native_handle(self):
        observer = VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=16,
            cached_tokens=0,
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 16})
        future = Future()
        wrapped = observer.wrap_execute_model(
            lambda scheduler_output, non_block: future,
            SimpleNamespace(requests={request.request_id: request}),
        )
        wrapped(output, non_block=True)

        future.cancel()
        observer._bridge.started.clear()
        gc.collect()

        self.assertEqual([True], observer._bridge.deleted)

    def test_peer_cache_capability_follows_native_connector_not_live_read_status(self):
        from navi_sched.adapters.vllm.VLLMEngineObserver import _v6d_peer_cache_enabled

        class Connector:
            pass

        connector = Connector()
        client = SimpleNamespace(peer_info={}, tair_kvcm_runtime=None)
        connector.scheduler = SimpleNamespace(
            _client_context=SimpleNamespace(client=client)
        )
        scheduler = SimpleNamespace(connector=SimpleNamespace(_connectors=[connector]))
        with mock.patch.dict(
            sys.modules,
            {
                "kvs_connector.v6d_object_connector": SimpleNamespace(
                    V6dObjectConnector=Connector
                ),
            },
        ):
            # Configuration enables V6D awareness before asynchronous metadata is ready.
            # Each resident snapshot carries its separate live peer-read availability.
            self.assertTrue(_v6d_peer_cache_enabled(scheduler))
            for ready in (False, True):
                client.peer_info = dict(kvcm_status=dict(cache_transfer_ready=ready))
                self.assertTrue(_v6d_peer_cache_enabled(scheduler))
            self.assertFalse(_v6d_peer_cache_enabled(None))
            self.assertFalse(
                _v6d_peer_cache_enabled(SimpleNamespace(connector=object()))
            )
            self.assertFalse(_v6d_peer_cache_enabled(SimpleNamespace(connector=None)))
        with mock.patch.dict(sys.modules, {"kvs_connector.v6d_object_connector": None}):
            self.assertFalse(_v6d_peer_cache_enabled(scheduler))

    def test_peer_cache_capability_never_waits_for_native_startup(self):
        from navi_sched.adapters.vllm.VLLMEngineObserver import _v6d_peer_cache_enabled

        class Connector:
            @property
            def scheduler(self):
                raise AssertionError(
                    "Configuration detection must not wait for the client"
                )

        with mock.patch.dict(
            sys.modules,
            {
                "kvs_connector.v6d_object_connector": SimpleNamespace(
                    V6dObjectConnector=Connector
                ),
            },
        ):
            self.assertTrue(
                _v6d_peer_cache_enabled(SimpleNamespace(connector=Connector()))
            )

    def make_continuous_timing_observer(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
            },
        ):
            return VLLMEngineObserver(self.make_config(), self.make_kv_cache_config())

    def run_continuous_timing_step(
        self,
        observer,
        batch_id,
        start_ns,
        end_ns,
        ready_ns=None,
        output_before_callback=False,
        execute_result="output",
    ):
        offset_ns = 1_000_000_000_000_000_000
        ready_ns = start_ns - 1_000_000 if ready_ns is None else ready_ns
        request = self.make_request(
            prompt_tokens=16,
            computed_tokens=16,
            cached_tokens=0,
            request_id=str(batch_id),
        )
        state = self.add_forward_batch(observer, batch_id, [request.request_id])
        state.native_batch.ready_time_us = (offset_ns + ready_ns) // 1000
        observer._admit_request(request)
        state.admission_started_ns = ready_ns
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 16})
        future = Future()
        wrapped = observer.wrap_execute_model(
            lambda unused, non_block: future,
            SimpleNamespace(requests={request.request_id: request}),
        )
        with (
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
                return_value=start_ns,
            ) as perf,
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.time.time_ns",
                return_value=offset_ns + start_ns,
            ) as wall,
        ):
            wrapped(output, non_block=True)
            perf.return_value = end_ns
            wall.return_value = offset_ns + end_ns
            if output_before_callback:
                observer.on_model_output(output)
                perf.return_value = end_ns + 50_000_000
                wall.return_value = offset_ns + end_ns + 50_000_000
                future.set_result(execute_result)
            else:
                future.set_result(execute_result)
                observer.on_model_output(output)
        self.assertTrue(state.released)
        return state

    def test_native_pipeline_steps_finish_independently_at_scheduler_output(self):
        observer = self.make_continuous_timing_observer()
        requests = {}
        states = []
        outputs = []
        futures = [Future() for _ in range(4)]
        for index in range(4):
            batch_id = 12001 + index
            request = self.make_request(
                prompt_tokens=16,
                computed_tokens=16,
                cached_tokens=0,
                request_id=str(batch_id),
            )
            requests[request.request_id] = request
            states.append(
                self.add_forward_batch(observer, batch_id, [request.request_id])
            )
            observer._admit_request(request)
            outputs.append(
                SimpleNamespace(num_scheduled_tokens={request.request_id: 16})
            )
        wrapped = observer.wrap_execute_model(
            mock.Mock(side_effect=futures), SimpleNamespace(requests=requests)
        )
        for output in outputs:
            wrapped(output, non_block=True)
        self.assertEqual(4, observer._active_execution_count)
        self.assertEqual(4, observer._bridge.backend_execution_count)
        self.assertEqual(64, observer._active_scheduled_tokens)
        for future in reversed(futures):
            future.set_result(None)
        self.assertEqual(4, len(observer._pd_ready_finishes))
        self.assertFalse(any(state.released for state in states))
        for index, output in enumerate(outputs):
            observer.on_model_output(output)
            self.assertTrue(states[index].released)
            self.assertEqual(3 - index, observer._active_execution_count)
            self.assertEqual(3 - index, observer._bridge.backend_execution_count)
            self.assertEqual(index + 1, len(observer._bridge.finished))
            self.assertFalse(any(state.released for state in states[index + 1 :]))
        self.assertEqual(0, observer._active_scheduled_tokens)
        self.assertEqual({}, observer._pd_ready_finishes)
        self.assertEqual({}, observer._pd_output_completed_ns)
        self.assertEqual({}, observer._pd_output_geometry)

    def test_native_pipeline_same_batch_uses_wall_label_and_finishes_once(self):
        observer = self.make_continuous_timing_observer()
        request = self.make_request(
            prompt_tokens=64, computed_tokens=0, cached_tokens=0, request_id="13001"
        )
        state = self.add_forward_batch(observer, 13001, [request.request_id])
        observer._admit_request(request)
        futures = [Future() for _ in range(4)]
        outputs = [
            SimpleNamespace(num_scheduled_tokens={request.request_id: 16})
            for _ in futures
        ]
        wrapped = observer.wrap_execute_model(
            mock.Mock(side_effect=futures),
            SimpleNamespace(requests={request.request_id: request}),
        )
        with (
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
                return_value=1_000_000_000,
            ) as perf,
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.time.time_ns",
                return_value=1_000_000_001_000_000_000,
            ) as wall,
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.logger.error"
            ) as error_log,
        ):
            for index, output in enumerate(outputs):
                perf.return_value = 1_000_000_000 + index * 100_000_000
                wall.return_value = 1_000_000_000_000_000_000 + perf.return_value
                request.num_computed_tokens = 16 * (index + 1)
                wrapped(output, non_block=True)
            self.assertEqual(4, state.pending_prefill_steps)
            for future in reversed(futures):
                future.set_result(None)
            for index, output in enumerate(outputs):
                perf.return_value = 1_500_000_000 + index * 100_000_000
                wall.return_value = 1_000_000_000_000_000_000 + perf.return_value
                observer.on_model_output(output)
                self.assertEqual(3 - index, state.pending_prefill_steps)
                self.assertEqual(index == 3, state.released)
        self.assertEqual(1, len(observer._bridge.finished))
        breakdown = observer._bridge.backend_batch_executions[0]
        self.assertEqual(800.0, breakdown[0])
        self.assertEqual(2000.0, breakdown[1])
        error_log.assert_not_called()
        self.assertEqual(0, observer._active_execution_count)
        self.assertEqual(0, observer._bridge.backend_execution_count)
        self.assertEqual(0, observer._active_scheduled_tokens)
        self.assertEqual({}, observer._pd_ready_finishes)
        self.assertEqual({}, observer._pd_output_completed_ns)

    def test_native_none_execute_result_finishes_at_scheduler_output(self):
        for output_before_callback in (False, True):
            with self.subTest(output_before_callback=output_before_callback):
                observer = self.make_continuous_timing_observer()
                for index in range(3):
                    start_ns = 1_000_000_000 + index * 200_000_000
                    state = self.run_continuous_timing_step(
                        observer,
                        11001 + index,
                        start_ns,
                        start_ns + 100_000_000,
                        output_before_callback=output_before_callback,
                        execute_result=None,
                    )
                    self.assertTrue(state.prefill_complete)
                    self.assertEqual(0, observer._active_execution_count)
                    self.assertEqual(0, observer._bridge.backend_execution_count)
                    self.assertEqual({}, observer._pd_ready_finishes)
                    self.assertEqual({}, observer._pd_output_completed_ns)
                    self.assertEqual({}, observer._pd_output_geometry)
                    self.assertEqual(index + 1, len(observer._bridge.finished))
                    self.assertEqual(
                        [(100.0, 16, 0)] * (index + 1),
                        observer._bridge.execution_steps,
                    )

    def test_native_training_start_excludes_pending_gap(self):
        for gap_ns in (1_000_000, 50_000_000, 400_000_000):
            with self.subTest(gap_ns=gap_ns):
                observer = self.make_continuous_timing_observer()
                self.run_continuous_timing_step(
                    observer, 10001, 1_000_000_000, 1_100_000_000
                )
                start_ns = 1_100_000_000 + gap_ns
                state = self.run_continuous_timing_step(
                    observer,
                    10002,
                    start_ns,
                    start_ns + 100_000_000,
                    ready_ns=1_050_000_000,
                )
                self.assertEqual(start_ns, state.execution_started_ns)
                self.assertEqual(100.0, state.execution_step_ms)
                self.assertEqual(
                    ("batch_start", 0), observer._bridge.execution_boundaries[-3]
                )
                self.assertEqual(
                    ("begin", 0), observer._bridge.execution_boundaries[-2]
                )
                self.assertEqual(
                    ("end", (start_ns + 100_000_000) // 1000),
                    observer._bridge.execution_boundaries[-1],
                )

    def test_native_capacity_features_keep_cpp_clock_and_finish_timestamp(self):
        for capacity in (False, True):
            with self.subTest(capacity=capacity):
                observer = self.make_continuous_timing_observer()
                observer._pp_capacity_unified = capacity
                self.run_continuous_timing_step(
                    observer, 10001, 1_000_000_000, 1_100_000_000
                )
                arguments = observer._bridge.backend_batch_executions[-1]
                self.assertEqual(-1.0, arguments[6])
                if capacity:
                    self.assertIs(arguments[-1], True)
                    self.assertEqual(1_000_000_001_100_000, arguments[-2])
                else:
                    self.assertEqual(1_000_000_001_100_000, arguments[-1])

    def test_native_training_actual_output_end_survives_start_fix(self):
        observer = self.make_continuous_timing_observer()
        self.run_continuous_timing_step(observer, 10001, 1_000_000_000, 1_100_000_000)
        state = self.run_continuous_timing_step(
            observer,
            10002,
            1_150_000_000,
            1_250_000_000,
            ready_ns=1_050_000_000,
            output_before_callback=True,
        )
        self.assertEqual(100.0, state.execution_step_ms)
        self.assertEqual(
            1_000_000_001_250_000, observer._bridge.backend_batch_executions[-1][-1]
        )
        self.assertEqual(("end", 1_250_000), observer._bridge.execution_boundaries[-1])

    def test_native_timing_does_not_reuse_previous_finish(self):
        observer = self.make_continuous_timing_observer()
        self.run_continuous_timing_step(observer, 10001, 1_000_000_000, 1_100_000_000)
        state = self.run_continuous_timing_step(
            observer, 10002, 1_150_000_000, 1_250_000_000, ready_ns=1_050_000_000
        )
        self.assertEqual(1_150_000_000, state.execution_started_ns)
        self.assertEqual(100.0, state.execution_step_ms)
        self.assertEqual(
            [(100.0, 16, 0), (100.0, 16, 0)], observer._bridge.execution_steps
        )
        self.assertEqual(0, observer._bridge.execution_boundaries[3][1])
        self.assertEqual(("begin", 0), observer._bridge.execution_boundaries[4])
        self.assertEqual(("end", 1_250_000), observer._bridge.execution_boundaries[5])
        self.assertEqual(
            1_000_000_001_250_000, observer._bridge.backend_batch_executions[-1][-1]
        )

    def test_legacy_probe_matches_native_training_start_scope(self):
        observer = self.make_continuous_timing_observer()
        self.run_continuous_timing_step(observer, 10001, 1_000_000_000, 1_100_000_000)
        state = self.run_continuous_timing_step(
            observer, 10002, 1_150_000_000, 1_250_000_000, ready_ns=1_050_000_000
        )
        event = [
            e
            for e in observer._bridge.request_lifecycle_audits
            if e["event"] == "execution_step"
        ][-1]
        self.assertEqual(100.0, state.execution_step_ms)
        self.assertEqual(1_150_000_000, event["started_perf_ns"])
        self.assertEqual(1_150_000_000, event["execute_entered_perf_ns"])
        self.assertTrue(event["legacy_backend_native_clock_bracket_valid"])
        self.assertEqual(100.0, event["legacy_backend_elapsed_lower_ms"])
        self.assertEqual(100.0, event["legacy_backend_elapsed_upper_ms"])
        self.assertEqual(0.0, event["legacy_backend_clock_uncertainty_ms"])
        self.assertEqual((100.0, 16, 0), observer._bridge.execution_steps[-1])
        self.assertEqual(("begin", 0), observer._bridge.execution_boundaries[-2])
        self.assertEqual(("end", 1_250_000), observer._bridge.execution_boundaries[-1])

    def test_legacy_probe_late_callback_records_late_old_end_without_changing_output_label(
        self,
    ):
        observer = self.make_continuous_timing_observer()
        self.run_continuous_timing_step(
            observer, 10001, 1_000_000_000, 1_100_000_000, output_before_callback=True
        )
        event = next(
            e
            for e in observer._bridge.request_lifecycle_audits
            if e["event"] == "execution_step"
        )
        self.assertEqual(100.0, event["callback_elapsed_ms"])
        self.assertEqual(150.0, event["legacy_backend_elapsed_lower_ms"])
        self.assertEqual(150.0, event["legacy_backend_elapsed_upper_ms"])
        self.assertEqual(1_100_000_000, event["completed_perf_ns"])
        self.assertEqual(1_150_000_000, event["legacy_backend_end_before_perf_ns"])

    def test_legacy_probe_agent_batch_clock_uses_native_start_and_output_end(self):
        observer = self.make_continuous_timing_observer()
        self.run_continuous_timing_step(observer, 10001, 1_000_000_000, 1_100_000_000)
        state = self.run_continuous_timing_step(
            observer, 10002, 1_150_000_000, 1_250_000_000, ready_ns=1_050_000_000
        )
        event = [
            e
            for e in observer._bridge.request_lifecycle_audits
            if e["event"] == "legacy_native_batch_clock"
        ][-1]
        self.assertEqual(2, event["legacy_batch_clock_probe_version"])
        self.assertTrue(event["legacy_batch_call_bracket_valid"])
        self.assertEqual(state.batch_id, event["native_batch_id"])
        self.assertEqual(state.schedule_batch_id, event["native_schedule_batch_id"])
        self.assertEqual(99.998, event["legacy_batch_elapsed_lower_ms"])
        self.assertEqual(100.002, event["legacy_batch_elapsed_upper_ms"])
        self.assertEqual(0.004, event["legacy_batch_clock_uncertainty_ms"])
        self.assertEqual(100.0, state.execution_step_ms)
        self.assertEqual(0, observer._bridge.execution_boundaries[-3][1])
        self.assertEqual(
            1_000_000_001_250_000, observer._bridge.backend_batch_executions[-1][-1]
        )

    def test_legacy_probe_agent_batch_clock_records_actual_late_finish_without_relabeling(
        self,
    ):
        observer = self.make_continuous_timing_observer()
        self.run_continuous_timing_step(
            observer, 10001, 1_000_000_000, 1_100_000_000, output_before_callback=True
        )
        event = next(
            e
            for e in observer._bridge.request_lifecycle_audits
            if e["event"] == "legacy_native_batch_clock"
        )
        self.assertTrue(event["legacy_batch_call_bracket_valid"])
        self.assertEqual(149.998, event["legacy_batch_elapsed_lower_ms"])
        self.assertEqual(150.002, event["legacy_batch_elapsed_upper_ms"])
        self.assertEqual(
            1_000_000_001_150_000_000, event["legacy_batch_end_before_wall_ns"]
        )
        self.assertEqual(
            1_000_000_001_100_000, observer._bridge.backend_batch_executions[-1][-1]
        )

    def test_native_timing_starts_after_ready_without_backdating(self):
        observer = self.make_continuous_timing_observer()
        self.run_continuous_timing_step(observer, 10001, 1_000_000_000, 1_100_000_000)
        state = self.run_continuous_timing_step(
            observer, 10002, 1_150_000_000, 1_250_000_000, ready_ns=1_120_000_000
        )
        self.assertEqual(1_150_000_000, state.execution_started_ns)
        self.assertEqual(100.0, state.execution_step_ms)
        self.assertEqual(100.0, observer._bridge.execution_steps[-1][0])
        self.assertEqual(100.0, observer._bridge.backend_batch_executions[-1][0])

    def test_continuous_timing_genuine_idle_resets_anchor(self):
        observer = self.make_continuous_timing_observer()
        self.run_continuous_timing_step(observer, 10001, 1_000_000_000, 1_100_000_000)
        scheduler = _NaviSchedScheduler(
            SimpleNamespace(has_requests=lambda: False), observer
        )
        self.assertFalse(scheduler.has_requests())
        state = self.run_continuous_timing_step(
            observer, 10002, 5_150_000_000, 5_250_000_000
        )
        self.assertEqual(5_150_000_000, state.execution_started_ns)
        self.assertEqual(100.0, observer._bridge.execution_steps[-1][0])

    def test_native_timing_zero_token_control_does_not_open_execution(self):
        observer = self.make_continuous_timing_observer()
        self.run_continuous_timing_step(observer, 10001, 1_000_000_000, 1_100_000_000)
        output = SimpleNamespace(num_scheduled_tokens={})
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
            return_value=1_140_000_000,
        ):
            self.run_execution_step(observer, output, {})
            observer.on_model_output(output)
        self.assertEqual(3, len(observer._bridge.execution_boundaries))
        scheduler = _NaviSchedScheduler(
            SimpleNamespace(has_requests=lambda: True), observer
        )
        self.assertTrue(scheduler.has_requests())
        self.run_continuous_timing_step(
            observer, 10002, 1_150_000_000, 1_250_000_000, ready_ns=1_050_000_000
        )
        self.assertEqual((100.0, 16, 0), observer._bridge.execution_steps[-1])
        self.assertEqual(
            [("begin", 0), ("end", 1_100_000), ("begin", 0), ("end", 1_250_000)],
            [v for v in observer._bridge.execution_boundaries if v[0] != "batch_start"],
        )

    def test_native_timing_late_callback_uses_captured_output_boundary(self):
        observer = self.make_continuous_timing_observer()
        self.run_continuous_timing_step(
            observer, 10001, 1_000_000_000, 1_100_000_000, output_before_callback=True
        )
        self.assertFalse(hasattr(observer, "_continuous_execution_boundary"))
        self.assertEqual(100.0, observer._bridge.execution_steps[-1][0])
        self.assertEqual(
            1_000_000_001_100_000, observer._bridge.backend_batch_executions[-1][-1]
        )

    def test_native_timing_native_decode_waits_for_output_and_closes_failure(self):
        observer = self.make_continuous_timing_observer()
        request = self.make_request(
            prompt_tokens=16, computed_tokens=17, cached_tokens=0
        )
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 1})
        future = Future()
        wrapped = observer.wrap_execute_model(
            lambda unused, non_block: future,
            SimpleNamespace(requests={request.request_id: request}),
        )
        with (
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
                return_value=1_150_000_000,
            ),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.time.time_ns",
                return_value=1_000_000_001_150_000_000,
            ),
        ):
            wrapped(output, non_block=True)
            future.set_result("decode")
        self.assertEqual(1, observer._bridge.backend_execution_count)
        self.assertEqual([], observer._bridge.execution_steps)
        with (
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
                return_value=1_250_000_000,
            ),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.time.time_ns",
                return_value=1_000_000_001_250_000_000,
            ),
        ):
            observer.on_model_output(output)
        self.assertEqual((100.0, 0, 1), observer._bridge.execution_steps[-1])
        self.assertFalse(hasattr(observer, "_continuous_execution_boundary"))
        future = Future()
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 1})
        wrapped(output, non_block=True)
        future.set_exception(RuntimeError("decode failed"))
        self.assertEqual(0, observer._bridge.backend_execution_count)
        self.assertFalse(hasattr(observer, "_continuous_execution_boundary"))

    def test_native_timing_two_logical_batches_share_one_physical_interval(self):
        observer = self.make_continuous_timing_observer()
        requests, states = {}, []
        for batch_id in (10001, 10002):
            request = self.make_request(
                prompt_tokens=16,
                computed_tokens=16,
                cached_tokens=0,
                request_id=str(batch_id),
            )
            state = self.add_forward_batch(observer, batch_id, [request.request_id])
            state.native_batch.ready_time_us = 1_000_000_001_050_000
            observer._admit_request(request)
            state.admission_started_ns = 1_050_000_000
            requests[request.request_id] = request
            states.append(state)
        output = SimpleNamespace(num_scheduled_tokens={key: 16 for key in requests})
        with (
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
                return_value=1_150_000_000,
            ),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.time.time_ns",
                return_value=1_000_000_001_150_000_000,
            ),
        ):
            self.run_execution_step(observer, output, requests)
        with (
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
                return_value=1_250_000_000,
            ),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.time.time_ns",
                return_value=1_000_000_001_250_000_000,
            ),
        ):
            observer.on_model_output(output)
        self.assertTrue(all(state.released for state in states))
        self.assertEqual([100.0, 100.0], [state.execution_step_ms for state in states])
        self.assertEqual([(100.0, 32, 0)], observer._bridge.execution_steps)
        self.assertEqual(["begin", "end"], observer._bridge.backend_execution_events)
        self.assertEqual(
            [100.0, 100.0],
            [sample[0] for sample in observer._bridge.backend_batch_executions],
        )

    def test_native_timing_abort_leaves_no_retrospective_anchor(self):
        observer = self.make_continuous_timing_observer()
        self.run_continuous_timing_step(observer, 10001, 1_000_000_000, 1_100_000_000)
        observer._abort_execution_step(_ExecutionStep(started_ns=1_150_000_000))
        self.assertFalse(hasattr(observer, "_continuous_execution_boundary"))
        self.assertEqual(0, observer._bridge.backend_execution_count)

    def test_native_timing_shutdown_leaves_no_retrospective_anchor(self):
        observer = self.make_continuous_timing_observer()
        self.run_continuous_timing_step(observer, 10001, 1_000_000_000, 1_100_000_000)
        observer.shutdown()
        self.assertFalse(hasattr(observer, "_continuous_execution_boundary"))
        self.assertEqual(0, observer._bridge.backend_execution_count)


class NaviSchedVLLMPluginTest(unittest.TestCase):
    def setUp(self):
        from vllm import plugins

        for name in (
            "register_engine_observer_factory",
            "register_engine_transport_factory",
        ):
            patcher = mock.patch.object(plugins, name, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_dashllm_native_ingress_preserves_engine_request(self):
        request = SimpleNamespace(request_id="dashllm-request")
        native_scheduler = SimpleNamespace(add_request=mock.Mock())
        observer = SimpleNamespace(
            _native_ingress=True,
            _navi_request_id=lambda value: stable_request_id(value.request_id),
            _pending_forward_batches={},
            _refresh_and_report_workload=mock.Mock(),
        )
        wrapped = _NaviSchedScheduler(native_scheduler, observer)

        wrapped.add_request(request)

        native_scheduler.add_request.assert_called_once_with(request)
        observer._refresh_and_report_workload.assert_called_once_with(
            native_scheduler,
            ("dashllm-request",),
            "native_ingress_admission",
        )

    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_transport_factory"
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_observer_factory"
    )
    def test_dashllm_decode_does_not_route_to_prefill_agent(
        self, register_factory, register_transport_factory
    ):
        with mock.patch.dict(
            os.environ,
            {
                "DS_ENABLE_DSERV": "1",
                "TPU_RAIDEN_JOB_NAME": "decode",
                "VLLM_NAVI_SCHED_ENABLE": "1",
            },
            clear=True,
        ):
            register_vllm_plugin()
            self.assertNotIn("PYTHONHASHSEED", os.environ)

        register_factory.assert_not_called()
        register_transport_factory.assert_not_called()

    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_transport_factory"
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_observer_factory"
    )
    def test_dashllm_prefill_keeps_native_transport(
        self, register_factory, register_transport_factory
    ):
        with mock.patch.dict(
            os.environ,
            {
                "DS_ENABLE_DSERV": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
                "VLLM_NAVI_SCHED_ENABLE": "1",
            },
            clear=True,
        ):
            register_vllm_plugin()

        register_factory.assert_called_once()
        register_transport_factory.assert_not_called()

    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_transport_factory"
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_observer_factory"
    )
    def test_dashllm_prefill_marker_keeps_native_transport(
        self, register_factory, register_transport_factory
    ):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
                "VLLM_NAVI_SCHED_ENABLE": "1",
            },
            clear=True,
        ):
            register_vllm_plugin()

        register_factory.assert_called_once()
        register_transport_factory.assert_not_called()

    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_transport_factory"
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_observer_factory"
    )
    def test_dashllm_prefill_core_redirect_registers_transport(
        self, register_factory, register_transport_factory
    ):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
                "VLLM_NAVI_SCHED_PREFILL_CORE_REDIRECT": "1",
                "TPU_RAIDEN_JOB_NAME": "prefill",
                "VLLM_NAVI_SCHED_ENABLE": "1",
            },
            clear=True,
        ):
            register_vllm_plugin()

        register_factory.assert_called_once()
        register_transport_factory.assert_called_once()

    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_transport_factory"
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_observer_factory"
    )
    def test_v029_registers_without_legacy_frontend_hooks(
        self, register_factory, register_transport_factory
    ):
        from vllm import plugins

        with (
            mock.patch.object(
                plugins, "register_frontend_request_interceptor", None, create=True
            ),
            mock.patch.object(
                plugins, "register_scheduler_iteration_metrics", None, create=True
            ),
            mock.patch.dict(
                os.environ,
                {
                    "VLLM_NAVI_SCHED_ENABLE": "1",
                    "DS_ENABLE_DSERV": "1",
                    "TPU_RAIDEN_JOB_NAME": "prefill",
                    "VLLM_NAVI_SCHED_PREFILL_CORE_REDIRECT": "1",
                },
            ),
        ):
            register_vllm_plugin()

        register_factory.assert_called_once()
        register_transport_factory.assert_called_once()

    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_scheduler_iteration_metrics",
        create=True,
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_frontend_request_interceptor",
        create=True,
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_transport_factory"
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_observer_factory"
    )
    def test_explicitly_disabled_does_not_register(
        self,
        register_factory,
        register_transport_factory,
        register_frontend_interceptor,
        register_scheduler_metrics,
    ):
        with mock.patch.dict(
            os.environ,
            {"VLLM_NAVI_SCHED_ENABLE": "0"},
            clear=True,
        ):
            register_vllm_plugin()
            self.assertNotIn("PYTHONHASHSEED", os.environ)

        register_factory.assert_not_called()
        register_transport_factory.assert_not_called()
        register_frontend_interceptor.assert_not_called()
        register_scheduler_metrics.assert_not_called()

    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_renderer_transport_factory",
        create=True,
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_frontend_app_configurator",
        create=True,
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_scheduler_iteration_metrics",
        create=True,
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_frontend_request_interceptor",
        create=True,
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_transport_factory"
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_observer_factory"
    )
    def test_enabled_by_default_or_explicitly_registers_observer(
        self,
        register_factory,
        register_transport_factory,
        register_frontend_interceptor,
        register_scheduler_metrics,
        register_frontend_configurator,
        register_renderer_factory,
    ):
        for environment in ({}, {"VLLM_NAVI_SCHED_ENABLE": "1"}):
            with (
                self.subTest(environment=environment),
                mock.patch.dict(os.environ, environment, clear=True),
            ):
                register_vllm_plugin()
                self.assertEqual("0", os.environ["PYTHONHASHSEED"])

        self.assertEqual(2, register_factory.call_count)
        self.assertEqual(2, register_transport_factory.call_count)
        self.assertEqual(2, register_frontend_interceptor.call_count)
        self.assertEqual(2, register_scheduler_metrics.call_count)
        self.assertEqual(2, register_frontend_configurator.call_count)
        self.assertEqual(2, register_renderer_factory.call_count)

    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_scheduler_iteration_metrics",
        create=True,
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_frontend_request_interceptor",
        create=True,
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_transport_factory"
    )
    @mock.patch(
        "navi_sched.adapters.vllm.VLLMPluginRegistry.register_engine_observer_factory"
    )
    def test_rejects_invalid_enable_value(
        self,
        register_factory,
        register_transport_factory,
        register_frontend_interceptor,
        register_scheduler_metrics,
    ):
        with (
            mock.patch.dict(
                os.environ,
                {"VLLM_NAVI_SCHED_ENABLE": "invalid"},
                clear=True,
            ),
            self.assertRaisesRegex(
                ValueError,
                "VLLM_NAVI_SCHED_ENABLE must be 0 or 1",
            ),
        ):
            register_vllm_plugin()

        register_factory.assert_not_called()
        register_transport_factory.assert_not_called()
        register_frontend_interceptor.assert_not_called()
        register_scheduler_metrics.assert_not_called()


class NativeKvTransferAuditTest(unittest.TestCase):
    def make_scheduler(self):
        wrapper = _NaviSchedScheduler.__new__(_NaviSchedScheduler)
        wrapper._scheduler = SimpleNamespace(update_from_output=mock.Mock())
        wrapper._observer = SimpleNamespace(
            _settle_scheduled_work=mock.Mock(),
            _refresh_and_report_workload=mock.Mock(),
            _log_slow_engine_stage=mock.Mock(),
            on_request_lifecycle_event=mock.Mock(),
        )
        return wrapper

    def test_received_kv_is_audited_before_native_output_update(self):
        wrapper = self.make_scheduler()
        scheduler_output = SimpleNamespace(num_scheduled_tokens={"running": 128})
        received = {"waiting-a", "waiting-b"}
        transfer_output = SimpleNamespace(finished_recving=received)
        model_output = SimpleNamespace(kv_connector_output=transfer_output)
        result = object()
        order = []
        wrapper._observer.on_request_lifecycle_event.side_effect = (
            lambda event: order.append("audit")
        )
        wrapper._scheduler.update_from_output.side_effect = (
            lambda *args: order.append("native") or result
        )

        self.assertIs(
            result, wrapper.update_from_output(scheduler_output, model_output)
        )
        wrapper._scheduler.update_from_output.assert_called_once_with(
            scheduler_output, model_output
        )
        events = [
            call.args[0]
            for call in wrapper._observer.on_request_lifecycle_event.call_args_list
        ]
        self.assertEqual(received, {event["request_id"] for event in events})
        self.assertTrue(
            all(event["event"] == "kv_transfer_received" for event in events)
        )
        self.assertTrue(all(event["event_time_us"] > 0 for event in events))
        self.assertTrue(all(event["received_perf_ns"] > 0 for event in events))
        self.assertEqual(["audit", "audit", "native"], order)
        wrapper._observer._refresh_and_report_workload.assert_called_once_with(
            wrapper._scheduler,
            {"running", "waiting-a", "waiting-b"},
            "update_from_output",
        )
        self.assertEqual({"waiting-a", "waiting-b"}, received)
        self.assertEqual({"running": 128}, scheduler_output.num_scheduled_tokens)

    def test_missing_and_empty_transfer_outputs_do_not_emit_events(self):
        for model_output in (
            SimpleNamespace(),
            SimpleNamespace(kv_connector_output=None),
            SimpleNamespace(kv_connector_output=SimpleNamespace(finished_recving=None)),
            SimpleNamespace(
                kv_connector_output=SimpleNamespace(finished_recving=set())
            ),
        ):
            with self.subTest(model_output=model_output):
                wrapper = self.make_scheduler()
                output = SimpleNamespace(num_scheduled_tokens={})
                wrapper.update_from_output(output, model_output)
                wrapper._observer.on_request_lifecycle_event.assert_not_called()
                wrapper._scheduler.update_from_output.assert_called_once_with(
                    output, model_output
                )

    def test_native_output_exception_is_preserved_after_audit(self):
        wrapper = self.make_scheduler()
        output = SimpleNamespace(num_scheduled_tokens={})
        model_output = SimpleNamespace(
            kv_connector_output=SimpleNamespace(finished_recving={"waiting"})
        )
        error = RuntimeError("native update failure")
        wrapper._scheduler.update_from_output.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            wrapper.update_from_output(output, model_output)
        self.assertIs(error, caught.exception)
        wrapper._observer.on_request_lifecycle_event.assert_called_once()
        wrapper._observer._settle_scheduled_work.assert_called_once_with(output)

    def test_audit_time_is_separate_from_native_output_time(self):
        wrapper = self.make_scheduler()
        output = SimpleNamespace(num_scheduled_tokens={})
        model_output = SimpleNamespace(
            kv_connector_output=SimpleNamespace(finished_recving={"waiting"})
        )
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.time.perf_counter_ns",
            side_effect=[100_000_000, 200_000_000, 500_000_000, 700_000_000],
        ):
            wrapper.update_from_output(output, model_output)
        details = wrapper._observer._log_slow_engine_stage.call_args.kwargs
        self.assertEqual(100, details["audit_ms"])
        self.assertEqual(300, details["native_ms"])
        self.assertEqual(200, details["report_ms"])


class NativeRequestCompletionAdapterTest(unittest.TestCase):
    def scheduler(self):
        wrapper = _NaviSchedScheduler.__new__(_NaviSchedScheduler)
        wrapper._observer = SimpleNamespace(
            _log_slow_engine_stage=mock.Mock(),
            _matching_forward_batches=lambda requested: [],
            _refresh_and_report_workload=mock.Mock(),
        )
        wrapper._scheduler = SimpleNamespace(
            requests={}, finish_requests=mock.Mock(return_value=[])
        )
        wrapper._add_request = mock.Mock()
        return wrapper

    def test_admission_result_and_exception_are_preserved(self):
        wrapper = self.scheduler()
        request = SimpleNamespace(request_id="a")
        result = object()
        wrapper._add_request.return_value = result
        self.assertIs(result, wrapper.add_request(request))
        self.assertIs(request, wrapper._add_request.call_args.args[0])
        error = RuntimeError("native admission error")
        wrapper._add_request.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            wrapper.add_request(request)
        self.assertIs(error, caught.exception)

    def test_native_finish_result_and_exception_are_preserved(self):
        wrapper = self.scheduler()
        status = SimpleNamespace(name="FINISHED_ABORTED")
        requests = ["a"]
        native_request = SimpleNamespace(request_id="a")
        wrapper._scheduler.finish_requests.return_value = [native_request]
        self.assertEqual([native_request], wrapper.finish_requests(requests, status))
        wrapper._scheduler.finish_requests.assert_called_once_with(requests, status)
        wrapper._observer._refresh_and_report_workload.assert_called_once_with(
            wrapper._scheduler, {"a"}, "finish_requests"
        )
        wrapper._observer._refresh_and_report_workload.reset_mock()
        error = RuntimeError("native finish error")
        wrapper._scheduler.finish_requests.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            wrapper.finish_requests(requests, status)
        self.assertIs(error, caught.exception)
        wrapper._observer._refresh_and_report_workload.assert_not_called()

    def test_output_routing_preserves_native_queue_item_and_error(self):
        observer = SimpleNamespace(
            route_engine_outputs=mock.Mock(return_value=False),
            _log_slow_engine_stage=mock.Mock(),
        )
        queue = mock.Mock()
        wrapper = _RoutingOutputQueue(queue, observer)
        stats = object()
        outputs = SimpleNamespace(
            outputs=[], finished_requests=[], scheduler_stats=stats
        )
        item = (0, outputs)
        wrapper.put_nowait(item)
        queue.put_nowait.assert_called_once_with(item)
        self.assertIs(item, queue.put_nowait.call_args.args[0])
        self.assertIs(stats, outputs.scheduler_stats)
        queue.reset_mock()
        error = RuntimeError("native output routing error")
        observer.route_engine_outputs.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            wrapper.put_nowait(item)
        self.assertIs(error, caught.exception)
        self.assertIs(stats, outputs.scheduler_stats)
        queue.put_nowait.assert_not_called()

    def execution_observer(self):
        observer = object.__new__(VLLMEngineObserver)
        observer._native_ingress = True
        observer._pp_capacity_unified = False
        observer._pd_output_lock = threading.Lock()
        observer._pd_output_geometry = {}
        observer._finish_execution_tracking = mock.Mock()
        observer._end_backend_execution = mock.Mock()
        observer._defer_pd_batch_finish = mock.Mock()
        observer.on_request_lifecycle_event = mock.Mock()
        observer._request_audit_failures = 0
        observer._bridge = mock.Mock()
        return observer

    def test_async_failure_and_cancel_preserve_native_future_and_cleanup_once(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                observer = self.execution_observer()
                future = Future()
                error = RuntimeError("native asynchronous execution failure")
                if cancelled:
                    future.cancel()
                else:
                    future.set_exception(error)
                step = _ExecutionStep(started_ns=time.perf_counter_ns())
                observer._complete_backend_execution(step, future, output_complete=True)
                observer._complete_backend_execution(step, future, output_complete=True)
                observer._finish_execution_tracking.assert_called_once_with(step)
                observer._end_backend_execution.assert_called_once()
                observer._bridge.report_execution_step.assert_not_called()
                self.assertFalse(
                    observer.on_request_lifecycle_event.call_args.args[0]["success"]
                )
                self.assertEqual(cancelled, future.cancelled())
                if not cancelled:
                    self.assertIs(error, future.exception())

    def test_late_model_output_failure_does_not_repeat_execution_timing(self):
        observer = self.execution_observer()
        step = _ExecutionStep(
            started_ns=time.perf_counter_ns(), backend_completion_recorded=True
        )
        future = Future()
        error = RuntimeError("native model output failure")
        future.set_exception(error)
        observer._complete_backend_execution(step, future, output_complete=True)
        observer._finish_execution_tracking.assert_not_called()
        observer._end_backend_execution.assert_not_called()
        observer.on_request_lifecycle_event.assert_not_called()
        self.assertIs(error, future.exception())

    def test_successful_step_reports_once_without_consuming_native_result(self):
        observer = self.execution_observer()
        future = Future()
        result = object()
        future.set_result(result)
        step = _ExecutionStep(started_ns=time.perf_counter_ns())
        observer._complete_backend_execution(step, future, output_complete=True)
        observer._complete_backend_execution(step, future, output_complete=True)
        observer._finish_execution_tracking.assert_called_once_with(step)
        observer._bridge.report_execution_step.assert_called_once()
        self.assertTrue(
            observer.on_request_lifecycle_event.call_args.args[0]["success"]
        )
        self.assertIs(result, future.result())
        self.assertFalse(future.cancelled())


class SerialPrefillTestBridgeTest(unittest.TestCase):
    def make_bridge(self):
        native = mock.Mock()
        batches = [SimpleNamespace(batch_id=1), SimpleNamespace(batch_id=2)]
        native.take_next_forward_batch.side_effect = batches
        return _SerialPrefillTestBridge(native), native, batches

    def test_next_batch_waits_for_prefill_release(self):
        bridge, native, batches = self.make_bridge()
        self.assertIs(bridge.take_next_forward_batch(), batches[0])
        self.assertIsNone(bridge.take_next_forward_batch())
        native.take_next_forward_batch.assert_called_once_with(0)
        self.assertIs(
            bridge.take_cancelled_requests(),
            native.take_cancelled_requests.return_value,
        )
        bridge.release_forward_batch(batches[0])
        self.assertIs(bridge.take_next_forward_batch(), batches[1])
        native.release_forward_batch.assert_called_once_with(batches[0])

    def test_unrelated_or_duplicate_release_does_not_open_another_batch(self):
        bridge, native, batches = self.make_bridge()
        bridge.take_next_forward_batch()
        bridge.release_forward_batch(None)
        bridge.release_forward_batch(SimpleNamespace(batch_id=99))
        self.assertIsNone(bridge.take_next_forward_batch())
        bridge.release_forward_batch(batches[0])
        bridge.take_next_forward_batch()
        bridge.release_forward_batch(batches[0])
        self.assertIsNone(bridge.take_next_forward_batch())

    def test_release_failure_propagates_and_does_not_leave_gate_waiting(self):
        bridge, native, batches = self.make_bridge()
        bridge.take_next_forward_batch()
        error = RuntimeError("release failed")
        native.release_forward_batch.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            bridge.release_forward_batch(batches[0])
        self.assertIs(raised.exception, error)
        self.assertIs(bridge.take_next_forward_batch(), batches[1])

    def test_empty_queue_does_not_reserve_batch(self):
        bridge, native, batches = self.make_bridge()
        native.take_next_forward_batch.side_effect = [None, batches[0]]
        self.assertIsNone(bridge.take_next_forward_batch())
        self.assertIs(bridge.take_next_forward_batch(), batches[0])

    def test_bounded_wait_keeps_cancellation_and_touches_accessible(self):
        bridge, native, batches = self.make_bridge()
        bridge.take_next_forward_batch()
        self.assertIsNone(bridge.take_next_forward_batch(1))
        native.take_next_forward_batch.assert_called_once_with(0)
        bridge.take_cancelled_requests()
        bridge.take_forward_request_touches(1)
        native.take_cancelled_requests.assert_called_once_with()
        native.take_forward_request_touches.assert_called_once_with(1)

    def wait_in_thread(self, bridge):
        entered = threading.Event()

        class ObservedCondition(threading.Condition):
            def wait_for(self, predicate, timeout=None):
                entered.set()
                return super().wait_for(predicate, timeout)

        bridge._condition = ObservedCondition()
        results = []
        thread = threading.Thread(
            target=lambda: results.append(bridge.take_next_forward_batch(2000))
        )
        thread.start()
        self.assertTrue(entered.wait(2))
        return thread, results

    def test_release_wakes_waiting_consumer(self):
        bridge, native, batches = self.make_bridge()
        bridge.take_next_forward_batch()
        thread, results = self.wait_in_thread(bridge)
        bridge.release_forward_batch(batches[0])
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [batches[1]])
        self.assertLessEqual(native.take_next_forward_batch.call_args.args[0], 2000)

    def test_shutdown_wakes_wait_without_taking_next_batch(self):
        bridge, native, batches = self.make_bridge()
        bridge.take_next_forward_batch()
        thread, results = self.wait_in_thread(bridge)
        self.assertIs(
            bridge.disable_forwarding(), native.disable_forwarding.return_value
        )
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [None])
        native.take_next_forward_batch.assert_called_once_with(0)
        self.assertIsNone(bridge.take_next_forward_batch())

    def test_non_pp_and_non_native_experiment_rejected_before_initialization(self):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=1)
        )
        with mock.patch.dict(
            os.environ, {"VLLM_NAVI_SCHED_PP_SERIAL_PREFILL_TEST": "1"}
        ):
            for native, pp in [(True, 1), (False, 4)]:
                config.parallel_config.pipeline_parallel_size = pp
                with mock.patch(
                    "navi_sched.adapters.vllm.VLLMEngineObserver.is_vllm_native_pd_prefill",
                    return_value=native,
                ):
                    with self.assertRaisesRegex(
                        ValueError, "requires Native PD with PP"
                    ):
                        VLLMEngineObserver(config, None)


class PPSampleResultAuditTest(unittest.TestCase):
    def make_stepper(self, dispatch):
        from navi_sched.adapters.vllm import ExecutionTimingProbe

        class LazyPushStepper:
            def __init__(self):
                self.in_flight = []

            def step(self):
                pass

            def _dispatch(self, *args, **kwargs):
                return dispatch(self, *args, **kwargs)

        LazyPushStepper.__module__ = "vllm_torchtpu.core.pp_lazy_step"
        stepper = LazyPushStepper()
        observer = SimpleNamespace(
            _native_request_state_audit_enabled=True,
            on_request_lifecycle_event=mock.Mock(),
        )
        core = SimpleNamespace(step_fn=stepper.step)
        return ExecutionTimingProbe, stepper, observer, core

    def test_sample_completion_has_output_identity_and_preserves_result(self):
        output = SimpleNamespace()
        execute = Future()
        sample = Future()

        def dispatch(stepper, value):
            stepper.in_flight.append(
                SimpleNamespace(
                    scheduler_output=output, exec_future=execute, future=sample
                )
            )
            return value

        probe, stepper, observer, core = self.make_stepper(dispatch)
        self.assertTrue(probe.bind_pp_sample_result_audit(observer, core))
        wrapped = stepper._dispatch
        self.assertTrue(probe.bind_pp_sample_result_audit(observer, core))
        self.assertIs(stepper._dispatch, wrapped)
        self.assertIs(stepper._dispatch(True), True)
        execute.set_result(None)
        observer.on_request_lifecycle_event.assert_not_called()
        sample.set_result("native-output")
        self.assertEqual(sample.result(), "native-output")
        event = observer.on_request_lifecycle_event.call_args.args[0]
        self.assertEqual(event["event"], "pp_sample_result_ready")
        self.assertEqual(event["scheduler_output_key"], id(output))
        self.assertTrue(event["has_sample_call"])
        self.assertFalse(event["cancelled"])
        self.assertGreater(event["ready_perf_ns"], 0)
        observer.on_request_lifecycle_event.assert_called_once()

    def test_already_ready_control_result_is_recorded(self):
        future = Future()
        future.set_result("control")
        output = SimpleNamespace()

        def dispatch(stepper):
            stepper.in_flight.append(
                SimpleNamespace(
                    scheduler_output=output, exec_future=future, future=future
                )
            )
            return True

        probe, stepper, observer, core = self.make_stepper(dispatch)
        self.assertTrue(probe.bind_pp_sample_result_audit(observer, core))
        self.assertTrue(stepper._dispatch())
        event = observer.on_request_lifecycle_event.call_args.args[0]
        self.assertFalse(event["has_sample_call"])
        self.assertEqual(future.result(), "control")

    def test_dropped_dispatch_and_native_exception_are_preserved(self):
        probe, stepper, observer, core = self.make_stepper(lambda unused: False)
        self.assertTrue(probe.bind_pp_sample_result_audit(observer, core))
        self.assertIs(stepper._dispatch(), False)
        observer.on_request_lifecycle_event.assert_not_called()
        error = RuntimeError("native dispatch failed")

        def fail(unused):
            raise error

        probe, stepper, observer, core = self.make_stepper(fail)
        self.assertTrue(probe.bind_pp_sample_result_audit(observer, core))
        with self.assertRaises(RuntimeError) as raised:
            stepper._dispatch()
        self.assertIs(raised.exception, error)
        observer.on_request_lifecycle_event.assert_not_called()

    def test_sink_failure_does_not_change_cancelled_future(self):
        sample = Future()

        def dispatch(stepper):
            stepper.in_flight.append(
                SimpleNamespace(
                    scheduler_output=SimpleNamespace(),
                    exec_future=Future(),
                    future=sample,
                )
            )
            return True

        probe, stepper, observer, core = self.make_stepper(dispatch)
        observer.on_request_lifecycle_event.side_effect = RuntimeError(
            "sink unavailable"
        )
        self.assertTrue(probe.bind_pp_sample_result_audit(observer, core))
        self.assertTrue(stepper._dispatch())
        self.assertTrue(sample.cancel())
        self.assertTrue(sample.cancelled())
        self.assertEqual(observer._pp_sample_audit_failures, 1)

    def test_other_engine_and_disabled_audit_are_untouched(self):
        probe, stepper, observer, core = self.make_stepper(lambda unused: False)
        original = stepper._dispatch
        observer._native_request_state_audit_enabled = False
        self.assertFalse(probe.bind_pp_sample_result_audit(observer, core))
        self.assertEqual(stepper._dispatch, original)
        observer._native_request_state_audit_enabled = True
        core.step_fn = lambda: None
        self.assertFalse(probe.bind_pp_sample_result_audit(observer, core))
        self.assertEqual(stepper._dispatch, original)

    def test_future_callback_does_not_retain_scheduler_output(self):
        import weakref

        class Output:
            pass

        sample = Future()

        def dispatch(stepper):
            stepper.in_flight.append(
                SimpleNamespace(
                    scheduler_output=Output(), exec_future=Future(), future=sample
                )
            )
            return True

        probe, stepper, observer, core = self.make_stepper(dispatch)
        self.assertTrue(probe.bind_pp_sample_result_audit(observer, core))
        stepper._dispatch()
        output_ref = weakref.ref(stepper.in_flight[-1].scheduler_output)
        stepper.in_flight.clear()
        gc.collect()
        self.assertIsNone(output_ref())
        sample.set_result(None)
        observer.on_request_lifecycle_event.assert_called_once()


class PPPipelineStateAuditTest(unittest.TestCase):
    make_stepper = PPSampleResultAuditTest.make_stepper

    def make_pipeline(self, dispatch=None):
        probe, stepper, observer, core = self.make_stepper(
            dispatch or (lambda unused: False)
        )
        stepper.wave = SimpleNamespace(open=True)
        stepper.stages = 4
        stepper.slots = [
            SimpleNamespace(
                dispatched=1.0, expected=0.2, modelled_ms=180.0, started_last=None
            )
        ]
        stepper.first_slot = 0
        stepper.last_real = 0
        stepper.ratio = 1.1
        stepper.measured = 0.0
        stepper._push = mock.Mock(return_value=object())
        stepper._settle = mock.Mock(return_value=object())
        return probe, stepper, observer, core

    def test_records_native_cost_and_future_state_without_retaining_objects(self):
        import weakref

        class Output:
            pass

        def dispatch(stepper):
            stepper.in_flight.append(
                SimpleNamespace(
                    scheduler_output=Output(), exec_future=Future(), future=Future()
                )
            )
            return True

        probe, stepper, observer, core = self.make_pipeline(dispatch)
        self.assertTrue(probe.bind_pp_sample_result_audit(observer, core))
        wrapped = stepper._push
        self.assertTrue(probe.bind_pp_sample_result_audit(observer, core))
        self.assertIs(stepper._push, wrapped)
        self.assertIs(stepper._dispatch(), True)
        event = observer.on_request_lifecycle_event.call_args.args[0]
        self.assertEqual(event["event"], "pp_pipeline_transition")
        self.assertEqual(event["operation"], "dispatch")
        self.assertIsNone(event["before"])
        self.assertEqual(event["after"]["cost_model_ratio"], 1.1)
        self.assertEqual(event["after"]["slots"][0]["modelled_forward_ms"], 180.0)
        self.assertFalse(
            event["after"]["unfinished_host_results"][0]["execute_reply_ready"]
        )
        output = stepper.in_flight[-1].scheduler_output
        self.assertEqual(
            event["after"]["unfinished_host_results"][0]["scheduler_output_key"],
            id(output),
        )
        ref = weakref.ref(output)
        del output
        stepper.in_flight.clear()
        gc.collect()
        self.assertIsNone(ref())
        json.dumps(event)

    def test_push_and_settle_preserve_arguments_identity_and_exception(self):
        probe, stepper, observer, core = self.make_pipeline()
        push = stepper._push
        settle = stepper._settle
        error = RuntimeError("native settle")
        settle.side_effect = error
        self.assertTrue(probe.bind_pp_sample_result_audit(observer, core))
        token = object()
        self.assertIs(stepper._push(token, flag=True), push.return_value)
        push.assert_called_once_with(token, flag=True)
        event = observer.on_request_lifecycle_event.call_args.args[0]
        self.assertEqual(event["operation"], "push")
        self.assertTrue(event["succeeded"])
        with self.assertRaises(RuntimeError) as raised:
            stepper._settle(token)
        self.assertIs(raised.exception, error)
        settle.assert_called_once_with(token)
        event = observer.on_request_lifecycle_event.call_args.args[0]
        self.assertEqual(event["operation"], "settle")
        self.assertFalse(event["succeeded"])

    def test_empty_polls_do_not_snapshot_native_state_or_emit(self):
        probe, stepper, observer, core = self.make_pipeline()
        self.assertTrue(probe.bind_pp_sample_result_audit(observer, core))
        stepper.slots = None
        self.assertIs(stepper._dispatch(), False)
        observer.on_request_lifecycle_event.assert_not_called()
        self.assertFalse(hasattr(observer, "_pp_pipeline_audit_failures"))

    def test_snapshot_and_sink_failures_preserve_native_result(self):
        probe, stepper, observer, core = self.make_pipeline()
        push = stepper._push
        self.assertTrue(probe.bind_pp_sample_result_audit(observer, core))
        stepper.ratio = float("nan")
        observer.on_request_lifecycle_event.side_effect = RuntimeError("sink")
        self.assertIs(stepper._push(), push.return_value)
        self.assertEqual(observer._pp_pipeline_audit_failures, 3)

    def test_snapshot_bounds_slots_and_unfinished_results(self):
        probe, stepper, observer, core = self.make_pipeline()
        stepper.slots *= 20
        stepper.in_flight = [
            SimpleNamespace(
                scheduler_output=object(), exec_future=Future(), future=Future()
            )
            for _ in range(40)
        ]
        self.assertTrue(probe.bind_pp_sample_result_audit(observer, core))
        stepper._push()
        state = observer.on_request_lifecycle_event.call_args.args[0]["after"]
        self.assertEqual(len(state["slots"]), 4)
        self.assertEqual(len(state["unfinished_host_results"]), 8)


class PriorCacheTrainingSnapshotTest(unittest.TestCase):
    def make_snapshot(
        self, forecast_context=20, progress=10, actual=70, resolved_at=50, running=True
    ):
        lookup = _RequestCacheLookup(11)
        if actual is not None:
            lookup.result = _ResolvedCacheHit(actual, resolved_at)
        compute = 100 - forecast_context
        context = forecast_context if compute else 0
        forecast = _AdmissionPriorCacheForecast(
            running, 100, context, compute, progress, lookup
        )
        moment = _PrefillWorkloadSlice(
            int(compute > 0), context, compute, compute**2, context * compute
        )
        return _ObservedWorkload(
            running_capacity_prefill=moment if running else _PrefillWorkloadSlice(),
            waiting_capacity_prefill=_PrefillWorkloadSlice() if running else moment,
            prior_cache_forecasts=(forecast,),
        )

    def sample(self, snapshot, deadline=100):
        return VLLMEngineObserver._capacity_sample_features(
            _AdmissionState(preceding_workload=snapshot),
            [],
            cache_resolution_deadline_ns=deadline,
        )

    def moments(self, aggregate):
        return (
            aggregate.request_count,
            aggregate.context_tokens,
            aggregate.compute_tokens,
            aggregate.compute_token_square_sum,
            aggregate.context_compute_token_product_sum,
        )

    def test_running_prior_uses_resolved_cache_for_all_five_moments(self):
        snapshot = self.make_snapshot()
        running, waiting = self.sample(snapshot)
        self.assertEqual((1, 70, 30, 900, 2100), self.moments(running))
        self.assertEqual((0, 0, 0, 0, 0), self.moments(waiting))
        self.assertEqual(
            (20, 80),
            (
                snapshot.running_capacity_prefill.context_tokens,
                snapshot.running_capacity_prefill.compute_tokens,
            ),
        )

    def test_waiting_prior_is_corrected_in_its_original_capacity_partition(self):
        running, waiting = self.sample(self.make_snapshot(running=False))
        self.assertEqual((0, 0, 0, 0, 0), self.moments(running))
        self.assertEqual((1, 70, 30, 900, 2100), self.moments(waiting))

    def test_cache_overprediction_can_increase_remaining_work(self):
        running, _ = self.sample(self.make_snapshot(forecast_context=80, actual=20))
        self.assertEqual((1, 20, 80, 6400, 1600), self.moments(running))

    def test_already_computed_progress_is_preserved(self):
        running, _ = self.sample(
            self.make_snapshot(forecast_context=60, progress=50, actual=20)
        )
        self.assertEqual((1, 50, 50, 2500, 2500), self.moments(running))

    def test_cache_resolved_after_current_finish_is_not_used(self):
        running, _ = self.sample(self.make_snapshot(resolved_at=101))
        self.assertEqual((1, 20, 80, 6400, 1600), self.moments(running))

    def test_resolution_at_finish_boundary_is_used(self):
        running, _ = self.sample(self.make_snapshot(resolved_at=100))
        self.assertEqual((1, 70, 30, 900, 2100), self.moments(running))

    def test_unresolved_lookup_keeps_original_forecast(self):
        running, _ = self.sample(self.make_snapshot(actual=None))
        self.assertEqual((1, 20, 80, 6400, 1600), self.moments(running))

    def test_non_pp_default_keeps_existing_sample_features(self):
        running, _ = self.sample(self.make_snapshot(), deadline=0)
        self.assertEqual((1, 20, 80, 6400, 1600), self.moments(running))

    def test_fully_cached_prior_removes_all_prefill_moments(self):
        running, _ = self.sample(self.make_snapshot(actual=100))
        self.assertEqual((0, 0, 0, 0, 0), self.moments(running))

    def test_false_full_cache_forecast_restores_prefill_request(self):
        running, _ = self.sample(self.make_snapshot(forecast_context=100, actual=20))
        self.assertEqual((1, 20, 80, 6400, 1600), self.moments(running))

    def test_shared_result_does_not_move_progress_between_snapshots(self):
        old = self.make_snapshot(actual=None)
        old_forecast = old.prior_cache_forecasts[0]
        newer_forecast = replace(
            old_forecast,
            context_tokens=90,
            compute_tokens=10,
            context_before_cache_forecast=90,
        )
        newer = replace(
            old,
            running_capacity_prefill=_PrefillWorkloadSlice(1, 90, 10, 100, 900),
            prior_cache_forecasts=(newer_forecast,),
        )
        old_forecast.cache_lookup.result = _ResolvedCacheHit(70, 50)
        older_running, _ = self.sample(old)
        newer_running, _ = self.sample(newer)
        self.assertEqual(
            (70, 30), (older_running.context_tokens, older_running.compute_tokens)
        )
        self.assertEqual(
            (90, 10), (newer_running.context_tokens, newer_running.compute_tokens)
        )


class NativePPTimingSampleTest(unittest.TestCase):
    def setUp(self):
        self.samples, self.errors = [], []
        scheduler = SimpleNamespace(running=[], num_waiting_for_streaming_input=0)
        self.stepper = SimpleNamespace(
            stages=4,
            wave=SimpleNamespace(open=False),
            slots=[],
            in_flight=[],
            first_slot=0,
            last_real=-1,
            measured=0.1,
            drains=1,
            _dispatch=lambda: False,
            _push=lambda: None,
            core=SimpleNamespace(scheduler=scheduler, step_fn=lambda: None),
        )
        self.reader = NativePipelineState(self.stepper)
        self.reader.bind(
            lambda: None,
            lambda: self.errors.append(sys.exc_info()[1]),
            on_timing_sample=self.samples.append,
        )

    def start(self, future=None, pushes=2):
        now = time.monotonic()
        slot = SimpleNamespace(
            dispatched=now, expected=0.1, started_last=None, started_drain=-1
        )
        self.stepper.wave.open = True
        self.stepper.slots.append(slot)
        self.stepper.last_real = 0
        step = SimpleNamespace(
            slot=slot,
            exec_future=Future(),
            future=future if future is not None else Future(),
        )
        self.stepper.in_flight.append(step)
        self.reader.record_dispatch((1.0, 8.0, 8.0, 0.0, 0.0, 64.0, 1.0, 0.0))
        for _ in range(pushes):
            push = SimpleNamespace(
                dispatched=time.monotonic(),
                expected=0.0,
                started_last=None,
                started_drain=-1,
            )
            self.stepper.slots.append(push)
            self.reader.record_push()
        return step

    def execute_ready(self, step):
        step.slot.started_last = time.monotonic()
        step.slot.started_drain = self.stepper.drains
        step.exec_future.set_result(None)

    def test_sample_clock_is_separate_from_execute_and_preserves_geometry(self):
        step = self.start()
        self.execute_ready(step)
        self.assertEqual([], self.samples)
        self.stepper.drains = 2
        before = time.monotonic_ns()
        step.future.set_result("tokens")
        after = time.monotonic_ns()
        self.assertEqual(1, len(self.samples))
        sample = self.samples[0]
        completed = sample.observed_monotonic_ns - sample.sample_reply_age_ns
        self.assertLessEqual(before, completed)
        self.assertLessEqual(completed, after)
        self.assertEqual((1, 0), (sample.burst, sample.slot_ordinal))
        self.assertEqual(
            (1, 2), (sample.last_stage_reply_drain, sample.sample_reply_drain)
        )
        self.assertEqual((0, 1, 2), tuple(v.ordinal for v in sample.slots))
        self.assertEqual(
            (1.0, 8.0, 8.0, 0.0, 0.0, 64.0, 1.0, 0.0), sample.slots[0].geometry
        )
        self.assertEqual((0.0,) * 8, sample.slots[1].geometry)
        self.assertEqual("tokens", step.future.result())
        self.assertEqual([], self.errors)

    def test_already_ready_sample_has_no_invented_callback_clock(self):
        future = Future()
        future.set_result("tokens")
        self.start(future=future)
        self.assertEqual([], self.samples)
        self.assertEqual(1, self.reader.late_timing_reply_count)

    def test_failed_sample_is_excluded_without_disabling_native_state(self):
        step = self.start()
        step.future.set_exception(RuntimeError("sample failed"))
        self.assertEqual([], self.samples)
        self.assertEqual(1, self.reader.failed_timing_reply_count)
        self.assertFalse(self.reader.failed)

    def test_cancelled_sample_is_excluded(self):
        step = self.start()
        step.future.cancel()
        self.assertEqual([], self.samples)
        self.assertEqual(1, self.reader.failed_timing_reply_count)

    def test_missing_handoff_shape_is_not_a_training_sample(self):
        step = self.start(pushes=1)
        self.execute_ready(step)
        step.future.set_result("tokens")
        self.assertEqual([], self.samples)
        self.assertEqual(1, self.reader.incomplete_timing_sample_count)

    def test_sample_before_execute_reply_is_not_a_training_sample(self):
        step = self.start()
        step.future.set_result("tokens")
        self.assertEqual([], self.samples)
        self.assertEqual(1, self.reader.incomplete_timing_sample_count)

    def test_first_handoff_geometry_survives_history_rotation(self):
        first = self.start()
        first_pushes = tuple(self.stepper.slots[1:])

        def append(real):
            slot = SimpleNamespace(
                dispatched=time.monotonic(),
                expected=0.1 if real else 0.0,
                started_last=None,
                started_drain=-1,
            )
            if len(self.stepper.slots) == self.stepper.stages:
                self.stepper.slots.pop(0)
                self.stepper.first_slot += 1
            self.stepper.slots.append(slot)
            if not real:
                self.reader.record_push()
                return None
            step = SimpleNamespace(slot=slot, exec_future=Future(), future=Future())
            self.stepper.in_flight.append(step)
            self.stepper.last_real = (
                self.stepper.first_slot + len(self.stepper.slots) - 1
            )
            self.reader.record_dispatch((1.0, 8.0, 8.0, 0.0, 0.0, 64.0, 1.0, 0.0))
            return step

        own = append(True)
        self.execute_ready(first)
        first.future.set_result("old tokens")
        # Native reads these execute replies before the later sample reply.
        for push in first_pushes:
            push.started_last = time.monotonic()
            push.started_drain = self.stepper.drains
        self.stepper.in_flight.remove(first)
        for _ in range(3):
            append(True)
        append(False)
        append(False)
        self.assertEqual(list(range(1, 9)), [row[0] for row in self.reader._history])
        self.samples.clear()
        self.execute_ready(own)
        own.future.set_result("tokens")
        self.assertEqual(1, len(self.samples))
        self.assertEqual(
            list(range(6)), [slot.ordinal for slot in self.samples[0].slots]
        )
        self.assertEqual(
            (1.0, 8.0, 8.0, 0.0, 0.0, 64.0, 1.0, 0.0), self.samples[0].slots[0].geometry
        )
        self.assertIsNotNone(self.samples[0].slots[0].last_stage_reply_age_ns)
        self.assertEqual(8, len(self.reader._history))

    def test_disabled_collector_does_not_attach_a_future_callback(self):
        self.reader._on_timing_sample = None
        step = self.start()
        self.assertEqual([], step.future._done_callbacks)

    def test_training_failure_preserves_the_native_sample_result(self):
        error = ValueError("invalid training input")
        self.reader._on_timing_sample = mock.Mock(side_effect=error)
        step = self.start()
        self.execute_ready(step)
        step.future.set_result("tokens")
        self.assertEqual("tokens", step.future.result())
        self.assertTrue(self.reader.failed)
        self.assertEqual([error], self.errors)


class NativePPLoadDispatchClockTest(unittest.TestCase):
    def setUp(self):
        holder = SimpleNamespace()
        self.reader = VLLMEngineObserverTest.bind_native_pp_test_reader(holder)
        self.module = VLLMEngineObserverTest.native_pp_test_module()
        self.stepper = self.reader._stepper
        self.row = SimpleNamespace(
            navi_request_id=1007,
            queue="waiting",
            queue_index=0,
            phase="kv_loading",
            query_generation=1,
            native_prompt_tokens=16384,
            native_computed_tokens=0,
            native_cache_hit_tokens=8192,
            native_shared_prefix_boundary=0,
            lookup_first_poll_mono_ns=1_000_000_000,
            load_started_mono_ns=1_000_000_000,
            native_prefill_slot_release=1,
        )

    def report(self, now_ns, rows=None):
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMNativePipelineState.time.monotonic_ns",
            return_value=now_ns,
        ):
            value, _ = self.reader.make_report(
                self.module, [self.row] if rows is None else rows
            )
        return value

    def dispatch(self, dispatched=10.1, compute=False):
        future = Future()
        slot = None
        if compute:
            slot = SimpleNamespace(
                dispatched=dispatched,
                expected=0.1,
                started_last=None,
                started_drain=-1,
            )
            self.stepper.wave.open = True
            self.stepper.slots.append(slot)
            self.stepper.last_real = 0
        step = SimpleNamespace(
            scheduler_output=object(),
            slot=slot,
            exec_future=future,
            future=future,
            dispatched=dispatched,
        )
        self.stepper.in_flight.append(step)
        geometry = (1.0, 8.0, 8.0, 0.0, 0.0, 64.0, 1.0, 0.0) if compute else None
        self.reader.record_dispatch(geometry, load_request_ids=(1007,))
        return step

    def test_query_prepare_timer_does_not_claim_a_load_dispatch(self):
        self.row.phase = "query_pending"
        value = self.report(10_000_000_000)
        self.assertEqual(9_000_000, value.requests[0].query_age_us)
        self.assertEqual(-1, value.requests[0].load_age_us)

    def test_decided_but_not_dispatched_load_has_unknown_age(self):
        self.reader.pending_decision = (123, (0.0,) * 8, (), (1007,))
        value = self.report(10_000_000_000)
        self.assertEqual((1007,), value.pending_decision.load_request_ids)
        self.assertEqual(-1, value.requests[0].load_age_us)

    def test_metadata_only_carrier_starts_the_load_clock(self):
        self.report(10_000_000_000)
        self.dispatch()
        value = self.report(10_400_000_000)
        self.assertEqual(300_000, value.requests[0].load_age_us)
        self.assertEqual(9_400_000, value.requests[0].query_age_us)
        self.assertEqual(300_000, value.results[0].dispatch_age_us)
        self.assertEqual(-1, value.results[0].slot_ordinal)

    def test_compute_carrier_uses_its_original_dispatch_clock(self):
        self.report(10_000_000_000)
        self.dispatch(compute=True)
        value = self.report(10_400_000_000)
        self.assertEqual(300_000, value.requests[0].load_age_us)
        self.assertEqual(300_000, value.slots[0].dispatch_age_us)

    def test_clock_survives_carrier_retirement_until_load_finishes(self):
        self.report(10_000_000_000)
        self.dispatch()
        self.stepper.in_flight.clear()
        value = self.report(10_900_000_000)
        self.assertEqual([], value.results)
        self.assertEqual(800_000, value.requests[0].load_age_us)
        self.row.phase = "cache_known_waiting"
        value = self.report(11_000_000_000)
        self.assertEqual(-1, value.requests[0].load_age_us)
        self.assertEqual({}, self.reader._load_dispatches)

    def test_new_query_generation_does_not_inherit_the_old_load_age(self):
        self.report(10_000_000_000)
        self.dispatch()
        self.assertEqual(300_000, self.report(10_400_000_000).requests[0].load_age_us)
        self.stepper.in_flight.clear()
        self.row.query_generation = 2
        self.assertEqual(-1, self.report(10_500_000_000).requests[0].load_age_us)
        self.dispatch(dispatched=10.8)
        self.assertEqual(100_000, self.report(10_900_000_000).requests[0].load_age_us)

    def test_absent_requests_release_the_clock_and_generation(self):
        self.report(10_000_000_000)
        self.dispatch()
        self.report(10_400_000_000)
        self.stepper.in_flight.clear()
        self.report(10_900_000_000, rows=[])
        self.assertEqual({}, self.reader._load_dispatches)
        self.assertEqual({}, self.reader._loading_generations)

    def test_unobserved_generation_does_not_invent_dispatch_provenance(self):
        self.dispatch()
        value = self.report(10_400_000_000)
        self.assertEqual(-1, value.requests[0].load_age_us)


class NativePP4TrainingSubmissionTest(unittest.TestCase):
    def make_sample(self):
        slots = tuple(
            SimpleNamespace(
                ordinal=index,
                dispatch_age_ns=(8 - index) * 1_000_000,
                geometry=(1.0, 8.0, 8.0, 0.0, 0.0, 64.0, 1.0, 0.0),
            )
            for index in range(6)
        )
        return SimpleNamespace(
            slots=slots, slot_ordinal=3, sample_reply_age_ns=1_000_000
        )

    def test_submits_full_native_dispatch_to_sample_latency_and_bounded_slots(self):
        observer = VLLMEngineObserver.__new__(VLLMEngineObserver)
        observer._bridge = SimpleNamespace(
            record_native_pp4_timing_sample=mock.Mock(return_value=True)
        )
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineObserver.time.time_ns",
            return_value=1234567890,
        ):
            observer._record_native_pp4_timing_sample(self.make_sample())
        call = observer._bridge.record_native_pp4_timing_sample.call_args.args
        self.assertEqual(3, call[0])
        self.assertEqual(list(range(6)), [slot[0] for slot in call[1]])
        self.assertEqual(
            [float(index) for index in range(6)], [slot[1] for slot in call[1]]
        )
        self.assertEqual(self.make_sample().slots[0].geometry, call[1][0][2])
        self.assertEqual(4.0, call[2])
        self.assertEqual(1234567, call[3])

    def test_rejected_cpp_submission_is_reported_to_reader_error_path(self):
        observer = VLLMEngineObserver.__new__(VLLMEngineObserver)
        observer._bridge = SimpleNamespace(
            record_native_pp4_timing_sample=mock.Mock(return_value=False)
        )
        with self.assertRaisesRegex(RuntimeError, "sample submission failed"):
            observer._record_native_pp4_timing_sample(self.make_sample())


if __name__ == "__main__":
    unittest.main()
