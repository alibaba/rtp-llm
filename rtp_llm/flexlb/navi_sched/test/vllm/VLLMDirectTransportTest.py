import asyncio
import base64
import enum
import hashlib
import importlib
import json
import os
import queue
import socket
import sys
import tempfile
import threading
import time
import types
import unittest
import zlib
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

try:
    import llm_plugin.metrics  # noqa: F401
except ModuleNotFoundError:
    llm_plugin = types.ModuleType("llm_plugin")
    metrics = types.ModuleType("llm_plugin.metrics")
    metrics.AccMetrics = SimpleNamespace(QPS_METRIC="qps")
    metrics.kmonitor = SimpleNamespace(report=lambda *unused, **kwargs: None)
    llm_plugin.metrics = metrics
    sys.modules["llm_plugin"] = llm_plugin
    sys.modules["llm_plugin.metrics"] = metrics


def _install_vllm_test_stubs():
    try:
        import vllm.sampling_params  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    try:
        import msgspec
    except ModuleNotFoundError:
        import copy

        msgspec = types.ModuleType("msgspec")

        class DecodeError(Exception):
            pass

        class Struct:
            def __init_subclass__(cls, **unused):
                super().__init_subclass__()

            def __init__(self, *args, **kwargs):
                annotations = {}
                for base in reversed(type(self).__mro__):
                    annotations.update(getattr(base, "__annotations__", {}))
                names = list(annotations)
                if len(args) > len(names):
                    raise TypeError("too many positional arguments")
                values = dict(zip(names, args))
                values.update(kwargs)
                for name in names:
                    if name in values:
                        value = values.pop(name)
                    elif hasattr(type(self), name):
                        value = copy.deepcopy(getattr(type(self), name))
                    else:
                        raise TypeError(f"missing required argument: {name}")
                    setattr(self, name, value)
                if values:
                    raise TypeError(f"unexpected arguments: {sorted(values)}")

        encoded_values = {}
        encoded_value_id = 0
        encoded_values_lock = threading.Lock()

        def encode(value):
            nonlocal encoded_value_id
            with encoded_values_lock:
                encoded_value_id += 1
                payload = f"msgspec-test:{encoded_value_id}".encode()
                encoded_values[payload] = value
                return payload

        def decode(payload, *, type=None):
            try:
                value = encoded_values[bytes(payload)]
            except KeyError as error:
                raise DecodeError("invalid test msgpack payload") from error
            if type is not None and not isinstance(value, type):
                raise DecodeError(f"expected {type.__name__}")
            return value

        msgspec.DecodeError = DecodeError
        msgspec.Struct = Struct
        msgspec.msgpack = SimpleNamespace(encode=encode, decode=decode)
        sys.modules["msgspec"] = msgspec

    class SamplingParams(msgspec.Struct, array_like=True):
        max_tokens: int = 1

    class FinishReason(enum.IntEnum):
        STOP = 0
        ERROR = 3

    class SchedulerStats(msgspec.Struct, array_like=True):
        num_running_reqs: int = 0
        num_waiting_reqs: int = 0

    class EngineCoreRequest(msgspec.Struct, array_like=True, omit_defaults=True):
        request_id: str
        prompt_token_ids: list[int] | None
        mm_features: object | None
        sampling_params: SamplingParams | None
        pooling_params: object | None
        arrival_time: float
        lora_request: object | None
        cache_salt: str | None
        data_parallel_rank: int | None
        client_index: int = 0
        current_wave: int = 0
        external_req_id: str | None = None
        navi_sched_local_fallback: bool = False

    class EngineCoreOutput(msgspec.Struct, array_like=True, omit_defaults=True):
        request_id: str
        new_token_ids: list[int]
        finish_reason: FinishReason | None = None
        stop_reason: int | str | None = None

        @property
        def finished(self):
            return self.finish_reason is not None

    class EngineCoreOutputs(msgspec.Struct, array_like=True, omit_defaults=True):
        engine_index: int = 0
        outputs: list[EngineCoreOutput] = []
        scheduler_stats: SchedulerStats | None = None
        timestamp: float = 0.0
        finished_requests: set[str] | None = None

    class EngineCoreRequestType(enum.Enum):
        ADD = b"\x00"
        ABORT = b"\x01"

    class MsgpackEncoder:
        @staticmethod
        def encode(value):
            return [msgspec.msgpack.encode(value)]

    class MsgpackDecoder:
        def __init__(self, value_type=None):
            self._value_type = value_type

        def decode(self, buffers):
            payload = buffers[0] if isinstance(buffers, (list, tuple)) else buffers
            return msgspec.msgpack.decode(payload, type=self._value_type)

    class EngineCoreProc:
        @staticmethod
        def _handle_client_request(owner, request_type, request):
            if request_type == EngineCoreRequestType.ADD:
                owner.add_request(*request)
            elif request_type == EngineCoreRequestType.ABORT:
                owner.abort_requests(request)

        @staticmethod
        def _handle_request_preproc_error(owner, request):
            owner.output_queue.put_nowait(
                (
                    request.client_index,
                    EngineCoreOutputs(
                        engine_index=owner.engine_index,
                        finished_requests={request.request_id},
                        outputs=[
                            EngineCoreOutput(
                                request_id=request.request_id,
                                new_token_ids=[],
                                finish_reason=FinishReason.ERROR,
                            )
                        ],
                    ),
                )
            )

    class ConcurrencyException(Exception):
        pass

    modules = {
        name: types.ModuleType(name)
        for name in (
            "vllm",
            "vllm.sampling_params",
            "vllm.v1",
            "vllm.v1.engine",
            "vllm.v1.engine.core",
            "vllm.v1.metrics",
            "vllm.v1.metrics.stats",
            "vllm.v1.serial_utils",
            "vllm.whale_vllm",
            "vllm.whale_vllm.vllm_server",
            "vllm.whale_vllm.vllm_server.utils",
            "vllm.whale_vllm.vllm_server.utils.concurrency_controller",
        )
    }
    modules["vllm.sampling_params"].SamplingParams = SamplingParams
    modules["vllm.v1.engine"].EngineCoreOutput = EngineCoreOutput
    modules["vllm.v1.engine"].EngineCoreOutputs = EngineCoreOutputs
    modules["vllm.v1.engine"].EngineCoreRequest = EngineCoreRequest
    modules["vllm.v1.engine"].EngineCoreRequestType = EngineCoreRequestType
    modules["vllm.v1.engine"].FinishReason = FinishReason
    modules["vllm.v1.engine.core"].EngineCoreProc = EngineCoreProc
    modules["vllm.v1.metrics.stats"].SchedulerStats = SchedulerStats
    modules["vllm.v1.serial_utils"].MsgpackDecoder = MsgpackDecoder
    modules["vllm.v1.serial_utils"].MsgpackEncoder = MsgpackEncoder
    concurrency_module = modules[
        "vllm.whale_vllm.vllm_server.utils.concurrency_controller"
    ]
    concurrency_module.ConcurrencyException = ConcurrencyException
    sys.modules.update(modules)


_install_vllm_test_stubs()

from navi_sched import NaviSchedClient as CoreNaviSchedClient, NaviSchedMaster
from navi_sched.adapters.vllm.NaviSchedAdapterConfig import NaviSchedAdapterConfig
from navi_sched.adapters.vllm.NaviSchedClient import NaviSchedClient
from navi_sched.adapters.vllm.NaviSchedEndpointRouter import NaviSchedEndpointRouter
from navi_sched.adapters.vllm.VLLMBlockHash import stable_request_id
from navi_sched.adapters.vllm.VLLMEngineObserver import VLLMEngineObserver
from navi_sched.adapters.vllm.VLLMEngineTransport import (
    VLLMEngineTransport,
    _make_client_kmon_tags,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
)
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.whale_vllm.vllm_server.utils.concurrency_controller import (
    ConcurrencyException,
)


class _FakeAgent:
    def __init__(self, **unused):
        self.stopped = False

    def init(self):
        return self

    def stop(self):
        self.stopped = True


class FullAttentionSpec:
    def __init__(self, block_size):
        self.block_size = block_size


def _reserve_ports(count):
    sockets = []
    try:
        for _ in range(count):
            value = socket.socket()
            value.bind(("127.0.0.1", 0))
            sockets.append(value)
        return [value.getsockname()[1] for value in sockets]
    finally:
        for value in sockets:
            value.close()


def _running_gig_threads():
    prefixes = (
        "GigClientHB",
        "GigUpdate",
        "GigCallDeg",
        "GigConnMgrCB",
        "GrpcClientWork",
    )
    names = []
    for task_dir in os.scandir("/proc/self/task"):
        try:
            with open(os.path.join(task_dir.path, "comm"), encoding="utf-8") as stream:
                name = stream.read().strip()
        except FileNotFoundError:
            continue
        if name.startswith(prefixes):
            names.append(name)
    return sorted(names)


def _wait_for_gig_threads_to_stop(timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while True:
        names = _running_gig_threads()
        if not names or time.monotonic() >= deadline:
            return names
        time.sleep(0.01)


class _ForwardBatch:
    def __init__(self, request_id, payload):
        self.batch_id = 101
        self.schedule_batch_id = 101
        self.schedule_batch_size = 1
        self.pending_cache_tracked = True
        self.ready_time_ns = time.monotonic_ns()
        self.request_ids = [request_id]
        self.predicted_cache_hit_token_counts = [0]
        self.payloads = [payload]


class _FakeBridge:
    def __init__(self):
        self.batches = queue.Queue()
        self.cancelled_requests = queue.Queue()
        self.responses = queue.Queue()
        self.released_batches = []
        self.published_responses = []
        self.failed_responses = []
        self.response_kind = SimpleNamespace(OUTPUT=1, ERROR=2)

    def enable_forwarding(self):
        return True

    def disable_forwarding(self):
        return True

    def take_next_forward_batch(self, timeout_ms):
        try:
            return self.batches.get(timeout=timeout_ms / 1000.0)
        except queue.Empty:
            return None

    def take_forward_request_touches(self, unused_max_count=0):
        return []

    def take_cancelled_requests(self):
        values = []
        while True:
            try:
                values.append(self.cancelled_requests.get_nowait())
            except queue.Empty:
                return values

    def publish_forward_response(self, request_id, frames, terminal):
        response = SimpleNamespace(
            request_id=request_id,
            kind=self.response_kind.OUTPUT,
            frames=list(frames),
            terminal=terminal,
            error_message="",
        )
        self.published_responses.append(response)
        self.responses.put(response)
        return True

    def fail_forward_response(self, request_id, message):
        response = SimpleNamespace(
            request_id=request_id,
            kind=self.response_kind.ERROR,
            frames=[],
            terminal=True,
            error_message=message,
        )
        self.failed_responses.append(response)
        self.responses.put(response)
        return True

    def release_forward_batch(self, batch):
        self.released_batches.append(batch)

    def report_cache_inserted(self, unused):
        pass

    def report_cache_removed(self, unused):
        pass

    def report_cache_updated(self, *unused):
        pass

    def log_cache_lifecycle_audit(self, unused):
        pass

    def log_cache_touch_audit(self, *unused):
        pass


class _RequestIdAllocatingNaviClient:
    _next_request_id = 1

    def has_master_biz(self):
        return True

    def allocate_request_id(self):
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    def report_local_fallback(self):
        self.local_fallback_count = getattr(self, "local_fallback_count", 0) + 1


class _FakeNaviClient(_RequestIdAllocatingNaviClient):
    def __init__(self, bridge):
        self.bridge = bridge
        self.closed = False
        self.next_canonical_request_id = 900_000_000_000_000_000
        self.local_request_ids = {}

    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        canonical_request_id = self.next_canonical_request_id
        self.next_canonical_request_id += 1
        self.local_request_ids[canonical_request_id] = request_id
        self.bridge.batches.put(_ForwardBatch(canonical_request_id, forward_payload))
        return SimpleNamespace(
            request_id=request_id,
            target=SimpleNamespace(address="127.0.0.1:8001"),
        )

    def take_next_forward_response(self, timeout_ms):
        try:
            response = self.bridge.responses.get(timeout=timeout_ms / 1000.0)
        except queue.Empty:
            return None
        local_request_id = self.local_request_ids.get(response.request_id)
        if local_request_id is None:
            return response
        return SimpleNamespace(
            request_id=local_request_id,
            kind=response.kind,
            frames=response.frames,
            terminal=response.terminal,
            error_message=response.error_message,
        )

    def abort_forward_request(self, request_id):
        self.bridge.cancelled_requests.put(request_id)
        return True

    def close(self):
        self.closed = True


class _FailingNaviClient(_RequestIdAllocatingNaviClient):
    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        raise TimeoutError("master request timed out")


class _RecoveringNaviClient(_RequestIdAllocatingNaviClient):
    def __init__(self):
        self.route_count = 0
        self.local_fallback_count = 0
        self.discovery_count = 0

    def has_master_biz(self):
        self.discovery_count += 1
        return self.discovery_count > 1

    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        self.route_count += 1
        return SimpleNamespace(target=SimpleNamespace(address="127.0.0.1:8001"))


class _RestartingMasterNaviClient(_RequestIdAllocatingNaviClient):
    def __init__(self):
        self.local_fallback_count = 0

    def has_master_biz(self):
        return False

    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        raise AssertionError("missing Master must not enter NaviSched routing")

    def report_local_fallback(self):
        self.local_fallback_count += 1


class _MasterDisappearingNaviClient(_RestartingMasterNaviClient):
    def has_master_biz(self):
        return True

    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        raise RuntimeError("biz [master] not exist")


class _MissingMasterDiscoveryNaviClient(_RequestIdAllocatingNaviClient):
    def __init__(self):
        self.local_fallback_count = 0
        self.route_count = 0
        self.abort_count = 0

    def has_master_biz(self):
        return False

    def allocate_request_id(self):
        raise AssertionError("missing Master must not allocate a NaviSched request")

    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        self.route_count += 1
        raise AssertionError("missing Master must not enter NaviSched routing")

    def abort_forward_request(self, unused_request_id):
        self.abort_count += 1
        return True

    def report_local_fallback(self):
        self.local_fallback_count += 1


class _MissingAgentNaviClient(_RequestIdAllocatingNaviClient):
    def __init__(self):
        self.local_fallback_count = 0

    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        raise RuntimeError("biz [prefill] not exist")

    def report_local_fallback(self):
        self.local_fallback_count += 1


class _AdmissionTimeoutNaviClient(_RequestIdAllocatingNaviClient):
    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        raise RuntimeError("NaviSched request admission timed out")


class _ConnectionFailureNaviClient(_RequestIdAllocatingNaviClient):
    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        raise ConnectionError("Master endpoint refused connection")


class _NativeScheduleFailureNaviClient(_RequestIdAllocatingNaviClient):
    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        raise RuntimeError("session timeout while running NaviSched graph")


class _RateLimitedNaviClient(_RequestIdAllocatingNaviClient):
    def __init__(self):
        self.local_fallback_count = 0

    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        return SimpleNamespace(target=None, reason="rate_limited")

    def report_local_fallback(self):
        self.local_fallback_count += 1


class _AdmissionRejectedNaviClient(_RateLimitedNaviClient):
    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        return SimpleNamespace(target=None, reason="admission_rejected")


class _UnavailableNodeNaviClient(_RequestIdAllocatingNaviClient):
    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        return SimpleNamespace(target=None, reason="no_available_node")


class _ReservationFailedNaviClient(_RequestIdAllocatingNaviClient):
    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        return SimpleNamespace(target=None, reason="reservation_failed")


class _RestartingDecisionNaviClient(_RequestIdAllocatingNaviClient):
    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        return SimpleNamespace(target=None, reason="master_restarting")


class _SchedulerErrorNaviClient(_RequestIdAllocatingNaviClient):
    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        return SimpleNamespace(target=None, reason="scheduler_error")


class _DelayedNaviClient(_RequestIdAllocatingNaviClient):
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.aborted = []

    async def route_async(self, unused_tokens, request_id, forward_payload, block_hash_request=None):
        self.started.set()
        await self.release.wait()
        return SimpleNamespace(target=SimpleNamespace(address="127.0.0.1:8001"))

    def abort_forward_request(self, request_id):
        self.aborted.append(request_id)
        return True


class _FakeEngineClient:
    def __init__(self):
        self.outputs = asyncio.Queue()
        self.added_requests = []
        self.aborted_requests = []
        self.shutdown_called = False

    async def add_request_async(self, request):
        self.added_requests.append(request)

    async def abort_requests_async(self, request_ids):
        self.aborted_requests.append(list(request_ids))

    async def get_output_async(self):
        return await self.outputs.get()

    def shutdown(self):
        self.shutdown_called = True


class _FakeEngineCore:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.aborts_queue = queue.Queue()
        self.engine_index = 0

    @staticmethod
    def preprocess_add_request(request):
        return request, 0

    _handle_request_preproc_error = EngineCoreProc._handle_request_preproc_error


class _FailingPreprocessEngineCore(_FakeEngineCore):
    @staticmethod
    def preprocess_add_request(request):
        raise ValueError("invalid forwarded request")


class _FakeCoreNaviSchedClient:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.initialized = False
        self.stopped = False
        self.master_biz_exists = True
        self.master_biz_queries = []
        self.local_fallback_count = 0

    def init(self):
        self.initialized = True
        return self

    def stop(self):
        self.stopped = True

    def has_master_biz(self, master_biz_name):
        self.master_biz_queries.append(master_biz_name)
        return self.master_biz_exists

    def report_local_fallback(self):
        self.local_fallback_count += 1


class NaviSchedClientDiscoveryTest(unittest.TestCase):
    def test_replay_sampling_is_disabled_by_default_and_passed_to_router(self):
        environment = {
            "VLLM_NAVI_SCHED_MASTER_BIZ_NAME": "master",
            "VLLM_NAVI_SCHED_TARGET_BIZ_NAME": "prefill",
            "VLLM_NAVI_SCHED_MASTER_VIP_DOMAIN": "vllm-master.vip",
            "VLLM_NAVI_SCHED_REPLAY_SAMPLE_INTERVAL": "1000",
        }
        config = NaviSchedAdapterConfig.from_env(environment)
        client = NaviSchedClient(config, native_client=_FakeCoreNaviSchedClient())
        try:
            self.assertEqual(1000, client._router._replay_sample_interval)
            self.assertEqual(0, NaviSchedAdapterConfig.from_env(
                {key: value for key, value in environment.items()
                 if key != "VLLM_NAVI_SCHED_REPLAY_SAMPLE_INTERVAL"}
            ).replay_sample_interval)
            with self.assertRaisesRegex(ValueError, "must be nonnegative"):
                NaviSchedAdapterConfig.from_env({
                    **environment, "VLLM_NAVI_SCHED_REPLAY_SAMPLE_INTERVAL": "-1"
                })
        finally:
            client.close()

    def test_master_vip_uses_published_http_to_grpc_port_diff(self):
        config = NaviSchedAdapterConfig(
            master_biz_name="master",
            target_biz_name="prefill",
            master_vip_domain="vllm-master.vip",
            master_vip_grpc_port_diff=2,
        )

        client_module = importlib.import_module(
            "navi_sched.adapters.vllm.NaviSchedClient"
        )
        environment = {
            "DEPLOYMENT_NAME": "cn_RTX_PRO_5000_72GB_4TP_spot",
            "VLLM_NAVI_SCHED_SERVER_IP": "10.59.7.197",
        }
        with (
            mock.patch.object(
                client_module, "CoreNaviSchedClient", _FakeCoreNaviSchedClient
            ),
            mock.patch.dict("os.environ", environment, clear=False),
        ):
            client = NaviSchedClient(config)

        self.assertTrue(client._client.initialized)
        self.assertEqual(
            {
                "jmenv_domain": "jmenv.tbsite.net",
                "services": ["vllm-master.vip"],
                "grpc_port_diff": 2,
            },
            client._client.kwargs["runtime_config"]["vip_subscription"],
        )
        self.assertEqual("sched_client", client._client.kwargs["name"])
        self.assertTrue(client._client.kwargs["enable_kmon"])
        self.assertEqual(
            {
                "role": "client",
                "deployment_name": "cn_RTX_PRO_5000_72GB_4TP_spot",
                "host": "10.59.7.197",
                "host_ip": "10.59.7.197",
                "container_ip": "10.59.7.197",
            },
            client._client.kwargs["kmon_tags"],
        )
        client.close()
        self.assertIsNone(client._client)

    def test_master_biz_presence_comes_from_native_discovery(self):
        config = NaviSchedAdapterConfig(
            master_biz_name="master",
            target_biz_name="prefill",
            master_vip_domain="vllm-master.vip",
        )
        native_client = _FakeCoreNaviSchedClient()
        client = NaviSchedClient(config, native_client=native_client)

        self.assertTrue(client.has_master_biz())
        native_client.master_biz_exists = False
        self.assertFalse(client.has_master_biz())
        self.assertEqual(["master", "master"], native_client.master_biz_queries)

        client.report_local_fallback()
        self.assertEqual(1, native_client.local_fallback_count)

    def test_client_kmon_tags_distinguish_api_processes(self):
        environment = {
            "DEPLOYMENT_NAME": "cn_MI308X_2TP_spot_new",
            "VLLM_NAVI_SCHED_SERVER_IP": "10.59.7.197",
        }
        with mock.patch.dict("os.environ", environment, clear=False):
            tags = [_make_client_kmon_tags(index) for index in range(4)]

        self.assertEqual(
            ["0", "1", "2", "3"],
            [item["client_index"] for item in tags],
        )
        self.assertTrue(
            all(item["host_ip"] == "10.59.7.197" for item in tags)
        )


class NaviSchedEndpointRouterTest(unittest.IsolatedAsyncioTestCase):
    async def test_failed_sampled_schedule_releases_log_capacity(self):
        client = mock.Mock()
        client.schedule.side_effect = [
            RuntimeError("no master"), SimpleNamespace(replay=None),
            SimpleNamespace(replay=None),
        ]
        router = NaviSchedEndpointRouter(client, "prefill", replay_sample_interval=1)
        try:
            router_module = importlib.import_module(
                "navi_sched.adapters.vllm.NaviSchedEndpointRouter"
            )
            with mock.patch.object(router_module, "logger"):
                with self.assertRaisesRegex(RuntimeError, "no master"):
                    router.route([1], request_id=1)
                router.route([1], request_id=2)
                router.route([1], request_id=3)
            self.assertEqual(
                [True, True, True],
                [call.kwargs.get("collect_replay", False)
                 for call in client.schedule.call_args_list],
            )
        finally:
            router.close()

    async def test_full_replay_log_queue_does_not_request_unrecordable_replay(self):
        entered = threading.Event()
        release = threading.Event()

        def serialize():
            entered.set()
            release.wait(2.0)
            return "{}"

        client = mock.Mock()
        client.schedule.return_value = SimpleNamespace(
            replay=SimpleNamespace(to_json=serialize),
            request_id=7, schedule_id=8,
        )
        router = NaviSchedEndpointRouter(client, "prefill", replay_sample_interval=1)
        try:
            router_module = importlib.import_module(
                "navi_sched.adapters.vllm.NaviSchedEndpointRouter"
            )
            with mock.patch.object(router_module, "logger"):
                router.route([1], request_id=1)
                self.assertTrue(entered.wait(0.5))
                router.route([1], request_id=2)
                router.route([1], request_id=3)
            self.assertEqual(
                [True, True, False],
                [call.kwargs.get("collect_replay", False)
                 for call in client.schedule.call_args_list],
            )
        finally:
            release.set()
            router.close()

    async def test_replay_serialization_does_not_delay_route(self):
        entered = threading.Event()
        release = threading.Event()

        def serialize():
            entered.set()
            release.wait(2.0)
            return "{}"

        client = mock.Mock()
        client.schedule.return_value = SimpleNamespace(
            replay=SimpleNamespace(to_json=serialize),
            request_id=7, schedule_id=8,
        )
        router = NaviSchedEndpointRouter(client, "prefill", replay_sample_interval=1)
        try:
            router_module = importlib.import_module(
                "navi_sched.adapters.vllm.NaviSchedEndpointRouter"
            )
            with mock.patch.object(router_module, "logger"):
                decision = await asyncio.wait_for(
                    router.route_async([1], request_id=7), timeout=0.5
                )
                self.assertIs(client.schedule.return_value, decision)
                self.assertTrue(entered.wait(0.5))
                release.set()
                router.close()
        finally:
            release.set()
            router.close()

    async def test_replay_sample_count_is_bounded_per_router(self):
        client = mock.Mock()
        client.schedule.return_value = SimpleNamespace(replay=None)
        router = NaviSchedEndpointRouter(client, "prefill", replay_sample_interval=1)
        try:
            router_module = importlib.import_module(
                "navi_sched.adapters.vllm.NaviSchedEndpointRouter"
            )
            with (mock.patch.object(router_module, "_MAX_REPLAY_SAMPLES", 2),
                  mock.patch.object(router_module, "logger")):
                for request_id in range(3):
                    router.route([1], request_id=request_id)
            self.assertEqual(
                [True, True, False],
                [call.kwargs.get("collect_replay", False)
                 for call in client.schedule.call_args_list],
            )
        finally:
            router.close()

    async def test_replay_serialization_failure_does_not_break_routing(self):
        replay = SimpleNamespace(to_json=mock.Mock(side_effect=ValueError("bad replay")))
        decision = SimpleNamespace(replay=replay, request_id=7, schedule_id=8)
        client = mock.Mock()
        client.schedule.return_value = decision
        router = NaviSchedEndpointRouter(client, "prefill", replay_sample_interval=1)
        try:
            router_module = importlib.import_module(
                "navi_sched.adapters.vllm.NaviSchedEndpointRouter"
            )
            with mock.patch.object(router_module, "logger") as logger:
                self.assertIs(decision, router.route([1], request_id=7))
                router.close()
            logger.exception.assert_called_once()
        finally:
            router.close()

    async def test_sampled_route_replay_is_complete_and_opt_in(self):
        payload = {"nodes": [{"index": index,
                              "hash": hashlib.sha256(str(index).encode()).hexdigest()}
                             for index in range(2000)]}
        encoded = json.dumps(payload)
        replay = SimpleNamespace(to_json=lambda: encoded)
        decision = SimpleNamespace(replay=replay, request_id=123,
                                   schedule_id=456)
        client = mock.Mock()
        client.schedule.return_value = decision
        router = NaviSchedEndpointRouter(
            client, "prefill", replay_sample_interval=2
        )
        try:
            router_module = importlib.import_module(
                "navi_sched.adapters.vllm.NaviSchedEndpointRouter"
            )
            with mock.patch.object(router_module, "logger") as logger:
                self.assertIs(decision, router.route([1], request_id=123))
                self.assertIs(decision, router.route([2], request_id=124))
                router.close()
            self.assertTrue(client.schedule.call_args_list[0].kwargs["collect_replay"])
            self.assertNotIn("collect_replay", client.schedule.call_args_list[1].kwargs)
            chunks = []
            for call in logger.info.call_args_list:
                text = call.args[0] % call.args[1:]
                fields = dict(item.split("=", 1) for item in text.split()
                              if "=" in item)
                self.assertEqual("123", fields["request_id"])
                self.assertEqual("456", fields["schedule_id"])
                chunks.append((int(fields["part"]), int(fields["total"]),
                               fields["sha256"], fields["data"]))
            self.assertGreater(len(chunks), 1)
            self.assertTrue(all(total == len(chunks) for _, total, _, _ in chunks))
            compressed = base64.b64decode("".join(
                data for _, _, _, data in sorted(chunks)
            ))
            raw = zlib.decompress(compressed)
            self.assertEqual(payload, json.loads(raw))
            self.assertEqual(hashlib.sha256(raw).hexdigest(), chunks[0][2])
        finally:
            router.close()

    async def test_oversized_replay_does_not_break_routing(self):
        replay = SimpleNamespace(to_json=lambda: "x" * (2 * 1024 * 1024 + 1))
        decision = SimpleNamespace(replay=replay, request_id=7, schedule_id=8)
        client = mock.Mock()
        client.schedule.return_value = decision
        router = NaviSchedEndpointRouter(
            client, "prefill", replay_sample_interval=1
        )
        try:
            router_module = importlib.import_module(
                "navi_sched.adapters.vllm.NaviSchedEndpointRouter"
            )
            with mock.patch.object(router_module, "logger") as logger:
                self.assertIs(decision, router.route([1], request_id=7))
                router.close()
            logger.info.assert_not_called()
            logger.warning.assert_called_once()
        finally:
            router.close()

    async def test_route_does_not_share_saturated_default_executor(self):
        default_worker_entered = threading.Event()
        release_default_worker = threading.Event()
        schedule_started = threading.Event()

        def block_default_worker():
            default_worker_entered.set()
            release_default_worker.wait(2.0)

        client = mock.Mock()

        def schedule(*unused_args, **unused_kwargs):
            schedule_started.set()
            return SimpleNamespace(target=SimpleNamespace(address="127.0.0.1:8001"))

        client.schedule.side_effect = schedule
        router = NaviSchedEndpointRouter(client, "prefill")
        executor = ThreadPoolExecutor(max_workers=1)
        asyncio.get_running_loop().set_default_executor(executor)
        blocker = asyncio.create_task(asyncio.to_thread(block_default_worker))
        route = None
        try:
            for _ in range(100):
                if default_worker_entered.is_set():
                    break
                await asyncio.sleep(0.001)
            self.assertTrue(default_worker_entered.is_set())
            route = asyncio.create_task(router.route_async([1], request_id=1))
            await asyncio.sleep(0.05)
            isolated_from_default_executor = schedule_started.is_set()
        finally:
            release_default_worker.set()
            await blocker
            if route is not None:
                await route
            close = getattr(router, "close", None)
            if close is not None:
                close()
            executor.shutdown(wait=True)
        self.assertTrue(isolated_from_default_executor)

    async def test_route_capacity_rejects_without_executor_queueing(self):
        schedule_started = threading.Event()
        release_schedule = threading.Event()

        client = mock.Mock()

        def schedule(*unused_args, **unused_kwargs):
            schedule_started.set()
            release_schedule.wait(2.0)
            return SimpleNamespace(target=SimpleNamespace(address="127.0.0.1:8001"))

        client.schedule.side_effect = schedule
        router = NaviSchedEndpointRouter(
            client, "prefill", route_worker_count=1
        )
        first = asyncio.create_task(router.route_async([1], request_id=1))
        try:
            for _ in range(100):
                if schedule_started.is_set():
                    break
                await asyncio.sleep(0.001)
            self.assertTrue(schedule_started.is_set())
            with self.assertRaisesRegex(
                RuntimeError, "route worker capacity exhausted"
            ):
                await router.route_async([2], request_id=2)
        finally:
            release_schedule.set()
            await first
            router.close()


class VLLMDirectTransportTest(unittest.IsolatedAsyncioTestCase):
    async def test_missing_fallback_metric_never_blocks_native_text_or_image_submission(self):
        for multimodal in (False, True):
            with self.subTest(multimodal=multimodal):
                legacy = SimpleNamespace(has_master_biz=lambda: False)
                transport = self.make_transport(legacy)
                request = SimpleNamespace(
                    request_id="native-fallback", prompt_token_ids=[1],
                    navi_sched_render_call=b"image" if multimodal else None,
                )
                if multimodal:
                    async def resume(requests):
                        self.assertEqual([request], requests)
                        request.prompt_token_ids = [1, 2, 3]
                        request.navi_sched_render_call = None
                    transport._render_continuation = SimpleNamespace(resume_async=resume)
                transport._engine_core = SimpleNamespace(add_request_async=mock.AsyncMock())
                with mock.patch(
                    "navi_sched.adapters.vllm.VLLMEngineTransport.logger",
                ) as logger:
                    await transport.add_request_async(request)
                transport._engine_core.add_request_async.assert_awaited_once_with(request)
                self.assertTrue(request.navi_sched_local_fallback)
                self.assertIsNone(request.navi_sched_render_call)
                self.assertEqual([1, 2, 3] if multimodal else [1], request.prompt_token_ids)
                self.assertEqual({}, transport._request_ids)
                transport._ensure_remote_output_task.assert_not_called()
                self.assertTrue(any(
                    "fallback metric" in str(call.args[0])
                    for call in logger.warning.call_args_list
                ))
                transport._response_executor.shutdown(wait=True)

    async def test_fallback_metric_failure_is_logged_once_and_native_errors_propagate(self):
        client = SimpleNamespace(
            has_master_biz=lambda: False,
            report_local_fallback=mock.Mock(side_effect=RuntimeError("metric unavailable")),
        )
        transport = self.make_transport(client)
        transport._engine_core = SimpleNamespace(add_request_async=mock.AsyncMock())
        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineTransport.logger",
        ) as logger:
            for index in range(2):
                await transport.add_request_async(SimpleNamespace(
                    request_id=str(index), prompt_token_ids=[1], navi_sched_render_call=None,
                ))
        self.assertEqual(2, transport._engine_core.add_request_async.await_count)
        self.assertEqual(1, sum(
            "fallback metric" in str(call.args[0]) for call in logger.warning.call_args_list
        ))
        transport._engine_core.add_request_async.side_effect = ValueError("native enqueue failed")
        with self.assertRaisesRegex(ValueError, "native enqueue failed"):
            await transport.add_request_async(SimpleNamespace(
                request_id="native-error", prompt_token_ids=[1], navi_sched_render_call=None,
            ))
        transport._response_executor.shutdown(wait=True)

    async def test_concurrent_deferred_fallback_initializes_one_native_processor(self):
        transport = self.make_transport(mock.Mock())
        transport._render_config = object()
        transport._render_continuation = None
        transport._render_init_lock = asyncio.Lock()
        started = asyncio.Event()
        release = asyncio.Event()

        async def supported_tasks():
            started.set()
            await release.wait()
            return ("generate",)

        transport._engine_core = SimpleNamespace(
            get_supported_tasks_async=supported_tasks,
            add_request_async=mock.AsyncMock(),
        )
        rendering = 0
        all_rendering = asyncio.Event()
        release_render = asyncio.Event()

        async def resume(requests):
            nonlocal rendering
            rendering += 1
            if rendering == 32:
                all_rendering.set()
            await release_render.wait()

        continuation = SimpleNamespace(resume_async=mock.AsyncMock(side_effect=resume))
        module = types.ModuleType("navi_sched.adapters.vllm.VLLMDeferredRenderer")
        module.NativeRenderContinuation = mock.Mock(return_value=continuation)
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            tasks = [asyncio.create_task(transport._submit_local_fallback(
                SimpleNamespace(request_id=f"image-{i}", navi_sched_render_call=b"call"),
                "master absent",
            )) for i in range(32)]
            try:
                await started.wait()
                await asyncio.sleep(0)
                release.set()
                await asyncio.wait_for(all_rendering.wait(), 1)
                self.assertEqual(1, module.NativeRenderContinuation.call_count)
                self.assertEqual(0, transport._engine_core.add_request_async.await_count)
                release_render.set()
                await asyncio.gather(*tasks)
                self.assertEqual(32, continuation.resume_async.await_count)
                self.assertEqual(32, transport._engine_core.add_request_async.await_count)
            finally:
                release.set()
                release_render.set()
                await asyncio.gather(*tasks, return_exceptions=True)
                transport._response_executor.shutdown(wait=True)

    async def test_cancelled_fallback_initialization_releases_waiting_requests(self):
        transport = self.make_transport(mock.Mock())
        transport._render_config = object()
        transport._render_continuation = None
        transport._render_init_lock = asyncio.Lock()
        started = asyncio.Event()

        async def interrupted_tasks():
            started.set()
            await asyncio.Event().wait()

        transport._engine_core = SimpleNamespace(
            get_supported_tasks_async=mock.AsyncMock(side_effect=interrupted_tasks),
            add_request_async=mock.AsyncMock(),
        )
        continuation = SimpleNamespace(resume_async=mock.AsyncMock())
        module = types.ModuleType("navi_sched.adapters.vllm.VLLMDeferredRenderer")
        module.NativeRenderContinuation = mock.Mock(return_value=continuation)
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            first = asyncio.create_task(transport._submit_local_fallback(
                SimpleNamespace(request_id="cancelled-image", navi_sched_render_call=b"call"),
                "master absent",
            ))
            try:
                await started.wait()
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                self.assertIsNone(transport._render_continuation)
                module.NativeRenderContinuation.assert_not_called()
                transport._engine_core.get_supported_tasks_async.side_effect = None
                transport._engine_core.get_supported_tasks_async.return_value = ("generate",)
                await asyncio.wait_for(transport._submit_local_fallback(
                    SimpleNamespace(request_id="retry-image", navi_sched_render_call=b"call"),
                    "master absent",
                ), 1)
                module.NativeRenderContinuation.assert_called_once_with(
                    transport._render_config, ("generate",),
                )
                transport._engine_core.add_request_async.assert_awaited_once()
            finally:
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
                transport._response_executor.shutdown(wait=True)

    async def test_transport_closed_during_fallback_initialization_never_creates_processor(self):
        transport = self.make_transport(mock.Mock())
        transport._render_config = object()
        transport._render_continuation = None
        transport._render_init_lock = asyncio.Lock()

        async def close_during_tasks():
            transport._closed = True
            return ("generate",)

        transport._engine_core = SimpleNamespace(
            get_supported_tasks_async=close_during_tasks,
            add_request_async=mock.AsyncMock(),
        )
        module = types.ModuleType("navi_sched.adapters.vllm.VLLMDeferredRenderer")
        module.NativeRenderContinuation = mock.Mock()
        try:
            with mock.patch.dict(sys.modules, {module.__name__: module}):
                with self.assertRaisesRegex(RuntimeError, "transport is closed"):
                    await transport._submit_local_fallback(
                        SimpleNamespace(request_id="closed-image", navi_sched_render_call=b"call"),
                        "master absent",
                    )
                module.NativeRenderContinuation.assert_not_called()
                transport._engine_core.add_request_async.assert_not_awaited()
        finally:
            transport._response_executor.shutdown(wait=True)

    async def test_deferred_local_fallback_materializes_before_engine_submission(self):
        transport = self.make_transport(mock.Mock())
        request = SimpleNamespace(request_id="local-image", navi_sched_render_call=b"call")
        order = []

        async def resume(requests):
            self.assertEqual([request], requests)
            order.append("render")
            request.navi_sched_render_call = None

        async def submit(actual):
            self.assertIs(request, actual)
            self.assertIsNone(actual.navi_sched_render_call)
            self.assertTrue(actual.navi_sched_local_fallback)
            order.append("submit")

        transport._render_continuation = SimpleNamespace(resume_async=resume)
        transport._engine_core = SimpleNamespace(add_request_async=submit)
        await transport._submit_local_fallback(request, "master absent")
        self.assertEqual(["render", "submit"], order)
        transport._client.report_local_fallback.assert_called_once_with()

    async def test_failed_deferred_fallback_never_submits_prediction(self):
        transport = self.make_transport(mock.Mock())
        request = SimpleNamespace(request_id="local-image", navi_sched_render_call=b"call")
        transport._render_continuation = SimpleNamespace(
            resume_async=mock.AsyncMock(side_effect=ValueError("download failed"))
        )
        transport._engine_core = SimpleNamespace(add_request_async=mock.AsyncMock())
        with self.assertRaisesRegex(ValueError, "download failed"):
            await transport._submit_local_fallback(request, "master absent")
        transport._engine_core.add_request_async.assert_not_awaited()
        transport._client.report_local_fallback.assert_not_called()

    async def test_transport_preserves_multimodal_cache_identity(self):
        try:
            import vllm.v1.core.kv_cache_utils  # noqa: F401
        except ImportError:
            self.skipTest("vLLM is not installed")
        client = mock.Mock()
        core = mock.Mock()
        core.schedule.return_value = SimpleNamespace(
            target=SimpleNamespace(address="127.0.0.1:8001")
        )
        router = NaviSchedEndpointRouter(core, "prefill")
        client.route_async = router.route_async
        transport = self.make_transport(client)
        request = SimpleNamespace(
            request_id="image-cache-identity", prompt_token_ids=list(range(64)),
            mm_features=[SimpleNamespace(
                identifier="image-test-content-hash",
                mm_position=SimpleNamespace(offset=48, length=32),
            )],
            client_index=0,
        )
        try:
            await transport.add_request_async(request)
        finally:
            router.close()
        self.assertEqual(
            [12687895063098490012, 1814341696565857452,
             2713165534665270607, 11063484669882182000],
            core.schedule.call_args.kwargs.get("block_hashes"),
        )

    @staticmethod
    def make_transport(client):
        transport = object.__new__(VLLMEngineTransport)
        transport._client = client
        transport._adapter_config = NaviSchedAdapterConfig(
            master_biz_name="master",
            target_biz_name="prefill",
            master_addresses=["127.0.0.1:9000"],
        )
        transport._encoder = mock.Mock()
        transport._encoder.encode.return_value = [b"request"]
        transport._msgspec = SimpleNamespace(
            msgpack=SimpleNamespace(encode=lambda value: b"payload")
        )
        transport._request_ids = {}
        transport._external_request_ids = {}
        transport._local_output_task = None
        if (
            isinstance(client, mock.Mock)
            and client.allocate_request_id.side_effect is None
        ):
            next_request_id = iter(range(1, 1_000_000))
            client.allocate_request_id.side_effect = lambda: next(next_request_id)
        transport._response_poll_lock = threading.Lock()
        transport._response_executor = ThreadPoolExecutor(max_workers=1)
        transport._closed = False
        transport._ensure_local_output_task = mock.Mock()
        transport._ensure_remote_output_task = mock.Mock()
        return transport

    async def test_graph_request_ids_are_local_monotonic_tokens(self):
        client = mock.Mock()
        client.allocate_request_id.side_effect = [41, 42]
        client.route_async = mock.AsyncMock(
            return_value=SimpleNamespace(
                target=SimpleNamespace(address="127.0.0.1:8001")
            )
        )
        transport = self.make_transport(client)
        first = SimpleNamespace(
            prompt_token_ids=[1], request_id="chat-first", client_index=7
        )
        second = SimpleNamespace(
            prompt_token_ids=[2], request_id="chat-second", client_index=8
        )

        await transport.add_request_async(first)
        await transport.add_request_async(second)

        self.assertEqual(
            [41, 42],
            [call.kwargs["request_id"] for call in client.route_async.call_args_list],
        )
        self.assertEqual(7, first.client_index)
        self.assertEqual(8, second.client_index)
        self.assertEqual(
            {"chat-first": 41, "chat-second": 42}, transport._request_ids
        )

    async def test_graph_request_id_rejects_active_token(self):
        client = mock.Mock()
        transport = self.make_transport(client)
        transport._external_request_ids[1] = "active"
        client.allocate_request_id.side_effect = None
        client.allocate_request_id.return_value = 1

        with self.assertRaisesRegex(RuntimeError, "active request id"):
            transport._allocate_graph_request_id()

    async def test_shutdown_waits_for_active_response_poll(self):
        poll_entered = threading.Event()
        poll_may_finish = threading.Event()
        client = mock.Mock()

        def take_next_forward_response(unused_timeout_ms):
            poll_entered.set()
            self.assertTrue(poll_may_finish.wait(2.0))
            return None

        client.take_next_forward_response.side_effect = take_next_forward_response
        transport = self.make_transport(client)
        transport._remote_output_task = None
        transport._engine_core = mock.Mock()

        poll = asyncio.create_task(
            asyncio.to_thread(transport._take_next_forward_response, 100)
        )
        self.assertTrue(await asyncio.to_thread(poll_entered.wait, 2.0))
        shutdown = asyncio.create_task(asyncio.to_thread(transport.shutdown))
        await asyncio.sleep(0.02)
        client.close.assert_not_called()

        poll_may_finish.set()
        self.assertIsNone(await poll)
        await shutdown
        client.close.assert_called_once_with()
        transport._engine_core.shutdown.assert_called_once_with()

    async def test_response_poll_does_not_share_saturated_default_executor(self):
        default_worker_entered = threading.Event()
        release_default_worker = threading.Event()
        response_poll_started = threading.Event()

        def block_default_worker():
            default_worker_entered.set()
            release_default_worker.wait(2.0)

        client = mock.Mock()

        def take_next_forward_response(unused_timeout_ms):
            response_poll_started.set()
            return None

        client.take_next_forward_response.side_effect = take_next_forward_response
        transport = self.make_transport(client)
        transport._outputs = asyncio.Queue()
        executor = ThreadPoolExecutor(max_workers=1)
        asyncio.get_running_loop().set_default_executor(executor)
        blocker = asyncio.create_task(asyncio.to_thread(block_default_worker))
        pump = None
        try:
            for _ in range(100):
                if default_worker_entered.is_set():
                    break
                await asyncio.sleep(0.001)
            self.assertTrue(default_worker_entered.is_set())
            pump = asyncio.create_task(transport._pump_remote_outputs())
            for _ in range(100):
                if response_poll_started.is_set():
                    break
                await asyncio.sleep(0.001)
            self.assertTrue(response_poll_started.is_set())
        finally:
            release_default_worker.set()
            await blocker
            if pump is not None:
                pump.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pump
            transport._response_executor.shutdown(
                wait=True, cancel_futures=True
            )
            executor.shutdown(wait=True)

    async def test_decode_failure_returns_terminal_pipe_error(self):
        import msgspec

        bridge = _FakeBridge()
        environment = {
            "PYTHONHASHSEED": "0",
            "VLLM_NAVI_SCHED_BIZ_NAME": "prefill",
            "VLLM_NAVI_SCHED_SERVER_IP": "127.0.0.1",
            "VLLM_NAVI_SCHED_SERVER_PORT": "8001",
            "VLLM_NAVI_SCHED_AGENT_GIG_PORT": "0",
        }
        config = SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=16,
                enable_prefix_caching=True,
                prefix_caching_hash_algo="sha256",
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            scheduler_config=SimpleNamespace(
                enable_chunked_prefill=False,
                max_num_partial_prefills=1,
                max_long_partial_prefills=1,
                long_prefill_token_threshold=0,
                max_num_scheduled_tokens=1024,
            ),
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=1024,
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=FullAttentionSpec(16))],
        )
        with (
            mock.patch.dict("os.environ", environment, clear=False),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.NaviSchedAgent",
                _FakeAgent,
            ),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.naviSchedModule",
                return_value=SimpleNamespace(_VLLMAgentBridge=lambda unused: bridge),
            ),
        ):
            observer = VLLMEngineObserver(config, kv_cache_config)
        observer._engine_core = SimpleNamespace(engine_index=3)
        request_id = "decode-failure"
        payload = msgspec.msgpack.encode([b"invalid-vllm-request"])
        native_batch = _ForwardBatch(stable_request_id(request_id), payload)

        with self.assertRaises(msgspec.DecodeError):
            observer._decode_forward_batch(
                native_batch,
                MsgpackDecoder(EngineCoreRequest),
                msgspec.msgpack,
            )

        self.assertEqual(1, len(bridge.failed_responses))
        response = bridge.failed_responses[0]
        self.assertEqual(stable_request_id(request_id), response.request_id)
        self.assertEqual(
            "invalid NaviSched VLLM forward request", response.error_message
        )
        self.assertTrue(response.terminal)
        observer.shutdown()

    async def test_route_timeout_returns_429_with_request_context(self):
        transport = self.make_transport(_FailingNaviClient())
        transport._engine_core = _FakeEngineClient()
        request = SimpleNamespace(
            prompt_token_ids=[1, 2], request_id="failed-request", client_index=0
        )

        with (
            self.assertLogs("vllm.navi_sched.client", level="ERROR") as captured,
            self.assertRaisesRegex(ConcurrencyException, "master request timed out"),
        ):
            await transport.add_request_async(request)

        self.assertIn("request_id=failed-request", captured.output[0])
        self.assertIn("master_addresses=['127.0.0.1:9000']", captured.output[0])
        self.assertIn("target_biz=prefill", captured.output[0])
        self.assertEqual([], transport._engine_core.added_requests)
        self.assertFalse(request.navi_sched_local_fallback)
        self.assertEqual({}, transport._request_ids)
        self.assertEqual({}, transport._external_request_ids)

    async def test_fallback_does_not_prevent_next_request_from_routing(self):
        client = _RecoveringNaviClient()
        transport = self.make_transport(client)
        transport._engine_core = _FakeEngineClient()
        fallback = SimpleNamespace(
            prompt_token_ids=[1], request_id="fallback", client_index=0
        )
        recovered = SimpleNamespace(
            prompt_token_ids=[2], request_id="recovered", client_index=0
        )

        await transport.add_request_async(fallback)
        await transport.add_request_async(recovered)

        self.assertEqual(1, client.route_count)
        self.assertEqual(2, client.discovery_count)
        self.assertEqual(1, client.local_fallback_count)
        self.assertEqual([fallback], transport._engine_core.added_requests)
        self.assertEqual({"recovered": 1}, transport._request_ids)
        self.assertFalse(recovered.navi_sched_local_fallback)

    async def test_fallback_streams_native_output_to_terminal(self):
        client = _RestartingMasterNaviClient()
        transport = self.make_transport(client)
        transport._engine_core = _FakeEngineClient()
        transport._outputs = asyncio.Queue()
        request = SimpleNamespace(
            prompt_token_ids=[1, 2], request_id="local-output", client_index=5
        )

        await transport.add_request_async(request)
        expected = object()
        pump = asyncio.create_task(transport._pump_local_outputs())
        try:
            await transport._engine_core.outputs.put(expected)
            self.assertIs(
                expected,
                await asyncio.wait_for(transport.get_output_async(), 1.0),
            )
        finally:
            pump.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pump

        self.assertEqual(5, request.client_index)
        self.assertTrue(request.navi_sched_local_fallback)
        self.assertEqual(1, client.local_fallback_count)
        self.assertEqual({}, transport._request_ids)
        self.assertEqual({}, transport._external_request_ids)

    async def test_missing_master_discovery_falls_back_before_navi_side_effects(self):
        client = _MissingMasterDiscoveryNaviClient()
        transport = self.make_transport(client)
        transport._adapter_config = NaviSchedAdapterConfig(
            master_biz_name="master",
            target_biz_name="prefill",
            master_addresses=(),
            master_vip_domain="master.vip",
        )
        transport._engine_core = _FakeEngineClient()
        request = SimpleNamespace(
            prompt_token_ids=[1, 2], request_id="vip-master-missing", client_index=2
        )

        await transport.add_request_async(request)

        self.assertEqual([request], transport._engine_core.added_requests)
        self.assertTrue(request.navi_sched_local_fallback)
        self.assertEqual(1, client.local_fallback_count)
        self.assertEqual(0, client.route_count)
        self.assertEqual(0, client.abort_count)
        transport._ensure_local_output_task.assert_called_once_with()
        transport._ensure_remote_output_task.assert_not_called()
        self.assertEqual({}, transport._request_ids)
        self.assertEqual({}, transport._external_request_ids)

    async def test_missing_master_static_address_falls_back_before_routing(self):
        client = _MissingMasterDiscoveryNaviClient()
        transport = self.make_transport(client)
        transport._engine_core = _FakeEngineClient()
        request = SimpleNamespace(
            prompt_token_ids=[1], request_id="static-master-missing", client_index=0
        )

        await transport.add_request_async(request)

        self.assertEqual([request], transport._engine_core.added_requests)
        self.assertTrue(request.navi_sched_local_fallback)
        self.assertEqual(1, client.local_fallback_count)
        self.assertEqual(0, client.route_count)
        transport._ensure_remote_output_task.assert_not_called()
        self.assertEqual({}, transport._request_ids)

    async def test_missing_master_biz_falls_back_to_native_engine(self):
        client = _RestartingMasterNaviClient()
        transport = self.make_transport(client)
        transport._engine_core = _FakeEngineClient()
        request = SimpleNamespace(
            prompt_token_ids=[1, 2], request_id="master-restarting", client_index=3
        )

        await transport.add_request_async(request)

        self.assertEqual([request], transport._engine_core.added_requests)
        self.assertEqual(3, request.client_index)
        self.assertTrue(request.navi_sched_local_fallback)
        self.assertEqual(1, client.local_fallback_count)
        self.assertEqual({}, transport._request_ids)
        self.assertEqual({}, transport._external_request_ids)

    async def test_master_disappears_after_precheck_does_not_fallback(self):
        client = _MasterDisappearingNaviClient()
        transport = self.make_transport(client)
        transport._engine_core = _FakeEngineClient()
        request = SimpleNamespace(
            prompt_token_ids=[1], request_id="master-disappeared", client_index=0
        )

        with self.assertRaisesRegex(ConcurrencyException, "biz \\[master\\] not exist"):
            await transport.add_request_async(request)

        self.assertEqual([], transport._engine_core.added_requests)
        self.assertFalse(request.navi_sched_local_fallback)
        self.assertEqual(0, client.local_fallback_count)
        self.assertEqual({}, transport._request_ids)

    async def test_missing_agent_biz_returns_429(self):
        client = _MissingAgentNaviClient()
        transport = self.make_transport(client)
        transport._engine_core = _FakeEngineClient()
        request = SimpleNamespace(
            prompt_token_ids=[1, 2], request_id="missing-agent", client_index=3
        )

        with self.assertRaisesRegex(ConcurrencyException, "biz \\[prefill\\] not exist"):
            await transport.add_request_async(request)

        self.assertEqual([], transport._engine_core.added_requests)
        self.assertFalse(request.navi_sched_local_fallback)
        self.assertEqual(0, client.local_fallback_count)
        self.assertEqual({}, transport._request_ids)
        self.assertEqual({}, transport._external_request_ids)

    async def test_local_fallback_abort_uses_native_engine_without_tracking(self):
        transport = self.make_transport(mock.Mock())
        transport._engine_core = _FakeEngineClient()

        await transport.abort_requests_async(["local-fallback"])

        self.assertEqual(
            [["local-fallback"]], transport._engine_core.aborted_requests
        )

    async def test_admission_timeout_returns_429(self):
        transport = self.make_transport(_AdmissionTimeoutNaviClient())
        transport._engine_core = _FakeEngineClient()
        request = SimpleNamespace(
            prompt_token_ids=[1, 2], request_id="admission-timeout", client_index=0
        )

        with self.assertRaisesRegex(ConcurrencyException, "admission timed out"):
            await transport.add_request_async(request)

        self.assertEqual([], transport._engine_core.added_requests)
        self.assertFalse(request.navi_sched_local_fallback)
        self.assertEqual({}, transport._request_ids)
        self.assertEqual({}, transport._external_request_ids)

    async def test_native_schedule_failure_returns_429(self):
        transport = self.make_transport(_NativeScheduleFailureNaviClient())
        transport._engine_core = _FakeEngineClient()
        request = SimpleNamespace(
            prompt_token_ids=[1, 2], request_id="native-failure", client_index=0
        )

        with self.assertRaisesRegex(ConcurrencyException, "session timeout"):
            await transport.add_request_async(request)

        self.assertEqual([], transport._engine_core.added_requests)
        self.assertFalse(request.navi_sched_local_fallback)
        self.assertEqual({}, transport._request_ids)
        self.assertEqual({}, transport._external_request_ids)

    async def test_connection_failure_returns_429(self):
        client = _ConnectionFailureNaviClient()
        transport = self.make_transport(client)
        transport._engine_core = _FakeEngineClient()
        request = SimpleNamespace(
            prompt_token_ids=[1, 2], request_id="connection-failed", client_index=0
        )

        with self.assertRaisesRegex(ConcurrencyException, "refused connection"):
            await transport.add_request_async(request)

        self.assertEqual([], transport._engine_core.added_requests)
        self.assertFalse(request.navi_sched_local_fallback)
        self.assertEqual(0, getattr(client, "local_fallback_count", 0))

    async def test_live_native_declared_master_without_endpoint_returns_429(self):
        original_dir = os.getcwd()
        native_client = None
        client = None
        with tempfile.TemporaryDirectory(prefix="navi-fallback-absent-") as runtime_dir:
            os.chdir(runtime_dir)
            try:
                master_port = _reserve_ports(1)[0]
                config = NaviSchedAdapterConfig(
                    master_biz_name="declared_master",
                    target_biz_name="prefill",
                    master_addresses=(f"127.0.0.1:{master_port}",),
                    timeout_ms=100,
                )
                native_client = CoreNaviSchedClient(
                    "declared_master_client",
                    "declared_master",
                    runtime_config={
                        "local_nodes": [
                            {
                                "biz_name": "declared_master",
                                "ip": "127.0.0.1",
                                "grpc_port": master_port,
                            }
                        ]
                    },
                    name="declared_master_client",
                    enable_kmon=False,
                    enable_snapshot=False,
                ).init()
                client = NaviSchedClient(config, native_client=native_client)
                self.assertTrue(client.has_master_biz())
                transport = self.make_transport(client)
                transport._adapter_config = config
                transport._engine_core = _FakeEngineClient()
                request = SimpleNamespace(
                    prompt_token_ids=[1], request_id="declared-master", client_index=0
                )

                with self.assertRaises(ConcurrencyException):
                    await transport.add_request_async(request)

                self.assertEqual([], transport._engine_core.added_requests)
                self.assertFalse(request.navi_sched_local_fallback)
                self.assertEqual(1, native_client.request_received_count)
                self.assertEqual({}, transport._request_ids)
                transport._ensure_remote_output_task.assert_called_once_with()
            finally:
                if client is not None:
                    client.close()
                elif native_client is not None:
                    native_client.stop()
                os.chdir(original_dir)

    async def test_live_native_discovery_distinguishes_missing_and_rejecting_master(self):
        source_core = _FakeEngineClient()
        original_dir = os.getcwd()
        master = None
        native_client = None
        transport = None
        config = NaviSchedAdapterConfig(
            master_biz_name="missing_master",
            target_biz_name="prefill",
            master_vip_domain="unused.vip",
            timeout_ms=100,
        )
        with tempfile.TemporaryDirectory(prefix="navi-fallback-native-") as runtime_dir:
            os.chdir(runtime_dir)
            try:
                master = NaviSchedMaster(
                    "present_master",
                    runtime_config={"thread_num": 16},
                    max_wait_schedule_count=1,
                    max_wait_schedule_time_ms=10,
                    scheduler_name="random",
                    name="fallback_present_master",
                    enable_kmon=False,
                    enable_snapshot=False,
                    wait_no_query=False,
                ).init()
                native_client = CoreNaviSchedClient(
                    "fallback_native_client",
                    "present_master",
                    runtime_config={
                        "local_nodes": [
                            {
                                "biz_name": "present_master",
                                "ip": "127.0.0.1",
                                "grpc_port": master.grpc_port,
                            }
                        ]
                    },
                    name="fallback_native_client",
                    enable_kmon=False,
                    enable_snapshot=False,
                ).init()
                client = NaviSchedClient(config, native_client=native_client)
                for _ in range(100):
                    if native_client.has_master_biz("present_master"):
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(native_client.has_master_biz("present_master"))
                self.assertFalse(client.has_master_biz())

                environment = {
                    "VLLM_NAVI_SCHED_MASTER_ADDRESSES": "",
                    "VLLM_NAVI_SCHED_MASTER_VIP_DOMAIN": "unused.vip",
                    "VLLM_NAVI_SCHED_MASTER_BIZ_NAME": "missing_master",
                    "VLLM_NAVI_SCHED_TARGET_BIZ_NAME": "prefill",
                }
                vllm_config = SimpleNamespace(
                    cache_config=SimpleNamespace(block_size=16),
                    parallel_config=SimpleNamespace(
                        decode_context_parallel_size=1,
                        prefill_context_parallel_size=1,
                    ),
                )
                with (
                    mock.patch.dict("os.environ", environment, clear=False),
                    mock.patch(
                        "navi_sched.adapters.vllm.VLLMEngineTransport.NaviSchedClient",
                        return_value=client,
                    ),
                ):
                    transport = VLLMEngineTransport(vllm_config, source_core)
                request = EngineCoreRequest(
                    request_id="native-missing-master",
                    prompt_token_ids=[1, 2],
                    mm_features=None,
                    sampling_params=SamplingParams(max_tokens=1),
                    pooling_params=None,
                    arrival_time=time.time(),
                    lora_request=None,
                    cache_salt=None,
                    data_parallel_rank=None,
                )
                await transport.add_request_async(request)

                self.assertEqual([request], source_core.added_requests)
                self.assertTrue(request.navi_sched_local_fallback)
                self.assertEqual(0, native_client.request_received_count)
                self.assertEqual({}, transport._request_ids)
                self.assertIsNone(transport._remote_output_task)
                terminal = EngineCoreOutputs(
                    outputs=[
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[42],
                            finish_reason=FinishReason.STOP,
                        )
                    ],
                    finished_requests={request.request_id},
                )
                await source_core.outputs.put(terminal)
                self.assertIs(
                    terminal,
                    await asyncio.wait_for(transport.get_output_async(), 1.0),
                )

                transport._adapter_config = NaviSchedAdapterConfig(
                    master_biz_name="present_master",
                    target_biz_name="prefill",
                    master_vip_domain="unused.vip",
                    timeout_ms=100,
                )
                client._config = transport._adapter_config
                rejected = EngineCoreRequest(
                    request_id="native-present-master",
                    prompt_token_ids=[3],
                    mm_features=None,
                    sampling_params=SamplingParams(max_tokens=1),
                    pooling_params=None,
                    arrival_time=time.time(),
                    lora_request=None,
                    cache_salt=None,
                    data_parallel_rank=None,
                )
                with self.assertRaisesRegex(
                    ConcurrencyException, "NaviSched has no available node"
                ) as rejected_error:
                    await transport.add_request_async(rejected)
                self.assertNotIn("biz [present_master] not exist", str(rejected_error.exception))
                self.assertNotIn("biz [missing_master] not exist", str(rejected_error.exception))
                self.assertFalse(rejected.navi_sched_local_fallback)
                self.assertEqual([request], source_core.added_requests)
                self.assertEqual(1, native_client.request_received_count)
            finally:
                if transport is not None:
                    transport.shutdown()
                elif native_client is not None:
                    native_client.stop()
                if master is not None:
                    master.stop()
                os.chdir(original_dir)

    async def test_rate_limited_route_uses_vllm_429_exception(self):
        client = _RateLimitedNaviClient()
        transport = self.make_transport(client)
        transport._engine_core = _FakeEngineClient()
        request = SimpleNamespace(
            prompt_token_ids=[1, 2], request_id="rate-limited", client_index=0
        )

        with self.assertRaisesRegex(
            ConcurrencyException, "NaviSched rate limited request"
        ):
            await transport.add_request_async(request)

        self.assertEqual([], transport._engine_core.added_requests)
        self.assertEqual(0, client.local_fallback_count)

    async def test_explicit_admission_reject_does_not_fall_back(self):
        client = _AdmissionRejectedNaviClient()
        transport = self.make_transport(client)
        transport._engine_core = _FakeEngineClient()
        request = SimpleNamespace(
            prompt_token_ids=[1, 2], request_id="admission-rejected", client_index=0
        )

        with self.assertRaisesRegex(
            ConcurrencyException, "NaviSched scheduling rejected request"
        ):
            await transport.add_request_async(request)

        self.assertEqual([], transport._engine_core.added_requests)
        self.assertEqual(0, client.local_fallback_count)

    async def test_no_available_node_route_uses_vllm_429_exception(self):
        transport = self.make_transport(_UnavailableNodeNaviClient())
        request = SimpleNamespace(
            prompt_token_ids=[1, 2], request_id="no-available-node", client_index=0
        )

        with self.assertRaisesRegex(
            ConcurrencyException, "NaviSched has no available node"
        ):
            await transport.add_request_async(request)

    async def test_reservation_failure_uses_vllm_429_exception(self):
        transport = self.make_transport(_ReservationFailedNaviClient())
        request = SimpleNamespace(
            prompt_token_ids=[1, 2], request_id="reservation-failed", client_index=0
        )

        with self.assertRaisesRegex(
            ConcurrencyException, "NaviSched reservation failed"
        ):
            await transport.add_request_async(request)

    async def test_master_restarting_decision_uses_vllm_429_exception(self):
        transport = self.make_transport(_RestartingDecisionNaviClient())
        request = SimpleNamespace(
            prompt_token_ids=[1, 2], request_id="master-restart-gap", client_index=0
        )

        with self.assertRaisesRegex(
            ConcurrencyException, "NaviSched master is restarting"
        ):
            await transport.add_request_async(request)

    async def test_scheduler_error_uses_vllm_429_exception(self):
        transport = self.make_transport(_SchedulerErrorNaviClient())
        request = SimpleNamespace(
            prompt_token_ids=[1, 2], request_id="scheduler-error", client_index=0
        )

        with self.assertRaisesRegex(
            ConcurrencyException, "NaviSched scheduling rejected request"
        ):
            await transport.add_request_async(request)

    async def test_abort_uses_request_graph_and_forgets_mapping(self):
        client = mock.Mock()
        client.abort_forward_request.return_value = True
        transport = self.make_transport(client)
        internal_id = 17
        transport._request_ids["request-1"] = internal_id
        transport._external_request_ids[internal_id] = "request-1"

        with mock.patch(
            "navi_sched.adapters.vllm.VLLMEngineTransport.logger.info"
        ) as log_info:
            await transport.abort_requests_async(["request-1"])

        client.abort_forward_request.assert_called_once_with(internal_id)
        log_info.assert_called_once_with(
            "[navi-vllm-client-abort] source=api request_id=%s "
            "navi_request_id=%s",
            "request-1",
            internal_id,
        )
        self.assertEqual({}, transport._request_ids)
        self.assertEqual({}, transport._external_request_ids)

    async def test_source_cancellation_aborts_route_that_finishes_in_background(self):
        client = _DelayedNaviClient()
        transport = self.make_transport(client)
        request = SimpleNamespace(
            prompt_token_ids=[1, 2], request_id="cancelled-request", client_index=0
        )

        task = asyncio.create_task(transport.add_request_async(request))
        await transport._client.started.wait()
        task.cancel()
        await asyncio.sleep(0)
        transport._client.release.set()

        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual([1], client.aborted)
        self.assertEqual({}, transport._request_ids)
        self.assertEqual({}, transport._external_request_ids)

    async def test_remote_terminal_output_clears_request_mapping(self):
        response = SimpleNamespace(
            request_id=7,
            kind=1,
            frames=[b"output"],
            terminal=True,
            error_message="",
        )
        client = mock.Mock()
        client.take_next_forward_response.side_effect = [response, None]
        transport = self.make_transport(client)
        transport._outputs = asyncio.Queue()
        transport._request_ids = {"finished-request": 7}
        transport._external_request_ids = {7: "finished-request"}
        transport._response_kind = SimpleNamespace(OUTPUT=1, ERROR=2)
        outputs = SimpleNamespace(
            finished_requests={"finished-request"},
            outputs=[
                SimpleNamespace(request_id="finished-request", finished=True)
            ],
        )
        transport._decoder = SimpleNamespace(decode=lambda unused_frames: outputs)
        task = asyncio.create_task(transport._pump_remote_outputs())
        try:
            self.assertIs(
                outputs,
                await asyncio.wait_for(transport._outputs.get(), 1.0),
            )
            self.assertEqual({}, transport._request_ids)
            self.assertEqual({}, transport._external_request_ids)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_remote_terminal_error_preserves_retryable_reason(self):
        client = mock.Mock()
        transport = self.make_transport(client)
        transport._response_kind = SimpleNamespace(OUTPUT=1, ERROR=2)
        transport._engine_core_output_type = EngineCoreOutput
        transport._engine_core_outputs_type = EngineCoreOutputs
        transport._finish_reason = FinishReason
        response = SimpleNamespace(
            request_id=7,
            kind=transport._response_kind.ERROR,
            frames=[],
            terminal=True,
            error_message="agent response pipe reached eof",
        )

        outputs = transport._decode_remote_response(response, "failed-request")

        self.assertEqual({"failed-request"}, outputs.finished_requests)
        self.assertEqual(1, len(outputs.outputs))
        output = outputs.outputs[0]
        self.assertEqual(FinishReason.ERROR, output.finish_reason)
        self.assertEqual(
            "navi_sched_retryable:agent response pipe reached eof",
            output.stop_reason,
        )

    async def test_remote_media_error_is_not_rate_limited(self):
        transport = self.make_transport(mock.Mock())
        transport._response_kind = SimpleNamespace(OUTPUT=1, ERROR=2)
        transport._engine_core_output_type = EngineCoreOutput
        transport._engine_core_outputs_type = EngineCoreOutputs
        transport._finish_reason = FinishReason
        response = SimpleNamespace(
            request_id=7, kind=2, frames=[], terminal=True,
            error_message="navi_sched_media_error:vLLM media preprocessing failed",
        )
        with self.assertLogs("vllm.navi_sched.client", level="WARNING") as logs:
            outputs = transport._decode_remote_response(response, "failed-image")
        self.assertEqual(
            ["WARNING:vllm.navi_sched.client:NAVI_SCHED_VLLM_REMOTE_REQUEST_FAILED "
             "navi_request_id=7 external_request_id=failed-image "
             "category=media_preprocessing"],
            logs.output,
        )
        self.assertEqual({"failed-image"}, outputs.finished_requests)
        self.assertEqual(1, len(outputs.outputs))
        output = outputs.outputs[0]
        self.assertEqual("failed-image", output.request_id)
        self.assertEqual(FinishReason.ERROR, output.finish_reason)
        self.assertEqual([], output.new_token_ids)
        self.assertEqual("vLLM media preprocessing failed", output.stop_reason)

    async def test_request_and_output_round_trip_uses_vllm_types(self):
        bridge = _FakeBridge()
        fake_client = _FakeNaviClient(bridge)
        engine_client = _FakeEngineClient()
        target_core = _FakeEngineCore()
        environment = {
            "PYTHONHASHSEED": "0",
            "VLLM_NAVI_SCHED_BIZ_NAME": "prefill",
            "VLLM_NAVI_SCHED_SERVER_IP": "127.0.0.1",
            "VLLM_NAVI_SCHED_SERVER_PORT": "8001",
            "VLLM_NAVI_SCHED_AGENT_GIG_PORT": "0",
            "VLLM_NAVI_SCHED_BLOCK_SIZE": "999",
            "VLLM_NAVI_SCHED_MASTER_BIZ_NAME": "master",
            "VLLM_NAVI_SCHED_TARGET_BIZ_NAME": "prefill",
            "VLLM_NAVI_SCHED_MASTER_ADDRESSES": "127.0.0.1:9000",
        }
        config = SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=16,
                enable_prefix_caching=True,
                prefix_caching_hash_algo="sha256",
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            scheduler_config=SimpleNamespace(
                enable_chunked_prefill=False,
                max_num_partial_prefills=1,
                max_long_partial_prefills=1,
                long_prefill_token_threshold=0,
                max_num_scheduled_tokens=1024,
            ),
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=1024,
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=FullAttentionSpec(16))],
        )

        with (
            mock.patch.dict("os.environ", environment, clear=False),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.NaviSchedAgent",
                _FakeAgent,
            ),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.naviSchedModule",
                return_value=SimpleNamespace(_VLLMAgentBridge=lambda unused: bridge),
            ),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineTransport.NaviSchedClient",
                return_value=fake_client,
            ),
        ):
            observer = VLLMEngineObserver(config, kv_cache_config)
            observer.bind_engine_core(target_core)
            transport = VLLMEngineTransport(config, engine_client)
            bridge.response_kind = transport._response_kind
            request = EngineCoreRequest(
                request_id="internal-request",
                prompt_token_ids=list(range(32)),
                mm_features=None,
                sampling_params=SamplingParams(max_tokens=1),
                pooling_params=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
                external_req_id="external-request",
            )

            await transport.add_request_async(request)
            request_type, (forwarded, request_wave) = await asyncio.to_thread(
                target_core.input_queue.get, True, 2.0
            )
            self.assertEqual(EngineCoreRequestType.ADD, request_type)
            self.assertEqual(0, request_wave)
            self.assertEqual(request.prompt_token_ids, forwarded.prompt_token_ids)
            self.assertEqual("900000000000000000", forwarded.request_id)
            self.assertEqual(900_000_000_000_000_000, forwarded.client_index)
            owner_core = SimpleNamespace(add_request=mock.Mock())
            EngineCoreProc._handle_client_request(
                owner_core,
                request_type,
                (forwarded, request_wave),
            )
            owner_core.add_request.assert_called_once_with(forwarded, request_wave)

            engine_outputs = EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id=forwarded.request_id,
                        new_token_ids=[42],
                        finish_reason=FinishReason.STOP,
                    )
                ]
            )
            local_output = (0, object())
            target_core.output_queue.put_nowait(local_output)
            self.assertIs(local_output, target_core.output_queue.get_nowait())
            target_core.output_queue.put_nowait(
                (forwarded.client_index, engine_outputs)
            )
            returned = await asyncio.wait_for(transport.get_output_async(), 2.0)
            self.assertEqual("internal-request", returned.outputs[0].request_id)
            self.assertEqual([42], returned.outputs[0].new_token_ids)
            self.assertEqual(FinishReason.STOP, returned.outputs[0].finish_reason)

            transport.shutdown()
            observer.shutdown()
            self.assertTrue(fake_client.closed)
            self.assertTrue(engine_client.shutdown_called)

    async def test_live_navi_graph_streams_vllm_outputs_end_to_end(self):
        model_port, agent_port = _reserve_ports(2)
        master = None
        observer = None
        transport = None
        original_dir = os.getcwd()
        environment = {
            "PYTHONHASHSEED": "0",
            "VLLM_NAVI_SCHED_BIZ_NAME": "live_vllm_prefill",
            "VLLM_NAVI_SCHED_SERVER_IP": "127.0.0.1",
            "VLLM_NAVI_SCHED_SERVER_PORT": str(model_port),
            "VLLM_NAVI_SCHED_AGENT_GIG_PORT": str(agent_port),
            "VLLM_NAVI_SCHED_MASTER_BIZ_NAME": "live_vllm_master",
            "VLLM_NAVI_SCHED_TARGET_BIZ_NAME": "live_vllm_prefill",
            "VLLM_NAVI_SCHED_CLIENT_BIZ_NAME": "live_vllm_client",
            "VLLM_NAVI_SCHED_TIMEOUT_MS": "1000",
        }
        config = SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=16,
                enable_prefix_caching=True,
                prefix_caching_hash_algo="sha256",
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            scheduler_config=SimpleNamespace(
                enable_chunked_prefill=False,
                max_num_partial_prefills=1,
                max_long_partial_prefills=1,
                long_prefill_token_threshold=0,
                max_num_scheduled_tokens=1024,
            ),
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=1024,
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=FullAttentionSpec(16))],
        )
        target_core = _FakeEngineCore()
        source_core = _FakeEngineClient()
        shutdown_elapsed_s = None

        with (
            tempfile.TemporaryDirectory(prefix="navi-vllm-live-") as runtime_dir,
            mock.patch.dict("os.environ", environment, clear=False),
        ):
            os.chdir(runtime_dir)
            try:
                observer = VLLMEngineObserver(config, kv_cache_config)
                observer.bind_engine_core(target_core)
                master = NaviSchedMaster(
                    "live_vllm_master",
                    runtime_config={
                        "thread_num": 16,
                        "local_nodes": [
                            {
                                "biz_name": "live_vllm_prefill",
                                "grpc_port": observer._agent.grpc_port,
                                "support_heartbeat": True,
                            }
                        ],
                    },
                    max_wait_schedule_count=1,
                    scheduler_name="random",
                    name="live_vllm_master",
                    enable_kmon=False,
                    enable_snapshot=False,
                    wait_no_query=False,
                ).init()
                os.environ["VLLM_NAVI_SCHED_MASTER_ADDRESSES"] = (
                    f"127.0.0.1:{master.grpc_port}"
                )
                transport = VLLMEngineTransport(config, source_core)

                forwarded = None
                external_request_id = None
                for attempt in range(100):
                    request = EngineCoreRequest(
                        request_id=f"live-request-{attempt}",
                        prompt_token_ids=list(range(32)),
                        mm_features=None,
                        sampling_params=SamplingParams(max_tokens=2),
                        pooling_params=None,
                        arrival_time=time.time(),
                        lora_request=None,
                        cache_salt=None,
                        data_parallel_rank=None,
                        external_req_id="live-external-request",
                    )
                    candidate_external_request_id = request.request_id
                    try:
                        await transport.add_request_async(request)
                    except ConcurrencyException:
                        await asyncio.sleep(0.02)
                        continue
                    external_request_id = candidate_external_request_id
                    request_type, (forwarded, request_wave) = (
                        await asyncio.to_thread(target_core.input_queue.get, True, 2.0)
                    )
                    self.assertEqual(EngineCoreRequestType.ADD, request_type)
                    self.assertEqual(0, request_wave)
                    break
                self.assertIsNotNone(forwarded)
                self.assertIsNotNone(external_request_id)
                self.assertTrue(forwarded.request_id.isdigit())
                self.assertNotEqual(external_request_id, forwarded.request_id)
                self.assertEqual(int(forwarded.request_id), forwarded.client_index)
                self.assertEqual(request.prompt_token_ids, forwarded.prompt_token_ids)
                scheduled_request = SimpleNamespace(
                    request_id=forwarded.request_id,
                    num_prompt_tokens=len(forwarded.prompt_token_ids),
                    num_computed_tokens=len(forwarded.prompt_token_ids),
                    num_cached_tokens=0,
                    max_tokens=forwarded.sampling_params.max_tokens,
                    num_preemptions=0,
                    block_hashes=[bytes(range(32)), bytes(reversed(range(32)))],
                    client_index=forwarded.client_index,
                )
                state = observer._admit_request(scheduled_request)
                scheduler_output = SimpleNamespace(
                    num_scheduled_tokens={
                        scheduled_request.request_id: len(forwarded.prompt_token_ids)
                    },
                    preempted_req_ids=set(),
                    scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
                )
                scheduler = SimpleNamespace(
                    requests={scheduled_request.request_id: scheduled_request},
                    running=[scheduled_request],
                    waiting=[],
                )
                execution = Future()
                wrapped_execute = observer.wrap_execute_model(
                    lambda unused_output, non_block: execution,
                    scheduler,
                )
                self.assertIs(
                    execution,
                    wrapped_execute(scheduler_output, non_block=True),
                )
                execution.set_result("prefill-complete")
                self.assertTrue(state.prefill_complete)
                self.assertTrue(state.released)

                first = EngineCoreOutputs(
                    outputs=[
                        EngineCoreOutput(
                            request_id=forwarded.request_id,
                            new_token_ids=[41],
                        )
                    ],
                    scheduler_stats=SchedulerStats(
                        num_running_reqs=3,
                        num_waiting_reqs=4,
                    ),
                )
                last = EngineCoreOutputs(
                    outputs=[
                        EngineCoreOutput(
                            request_id=forwarded.request_id,
                            new_token_ids=[42],
                            finish_reason=FinishReason.STOP,
                        )
                    ]
                )
                target_core.output_queue.put_nowait((forwarded.client_index, first))
                target_core.output_queue.put_nowait((forwarded.client_index, last))

                returned_first = await asyncio.wait_for(
                    transport.get_output_async(), 2.0
                )
                returned_last = await asyncio.wait_for(
                    transport.get_output_async(), 2.0
                )
                self.assertEqual([41], returned_first.outputs[0].new_token_ids)
                self.assertEqual(
                    external_request_id, returned_first.outputs[0].request_id
                )
                self.assertIsNone(returned_first.outputs[0].finish_reason)
                self.assertIsNone(returned_first.scheduler_stats)
                self.assertEqual([42], returned_last.outputs[0].new_token_ids)
                self.assertEqual(
                    external_request_id, returned_last.outputs[0].request_id
                )
                self.assertEqual(
                    FinishReason.STOP, returned_last.outputs[0].finish_reason
                )
                local_client_index, local_stats = (
                    target_core.output_queue._output_queue.get_nowait()
                )
                self.assertEqual(0, local_client_index)
                self.assertEqual(3, local_stats.scheduler_stats.num_running_reqs)
                self.assertEqual(4, local_stats.scheduler_stats.num_waiting_reqs)
                self.assertEqual({}, transport._request_ids)
                self.assertEqual({}, transport._external_request_ids)
            finally:
                shutdown_started = time.monotonic()
                if transport is not None:
                    transport.shutdown()
                if observer is not None:
                    observer.shutdown()
                if master is not None:
                    master.stop()
                shutdown_elapsed_s = time.monotonic() - shutdown_started
                os.chdir(original_dir)

        self.assertLess(shutdown_elapsed_s, 5.0)
        self.assertEqual([], _wait_for_gig_threads_to_stop())

    async def test_live_navi_graph_partial_cancel_keeps_remaining_request(self):
        model_port, agent_port = _reserve_ports(2)
        master = None
        observer = None
        transport = None
        original_dir = os.getcwd()
        environment = {
            "PYTHONHASHSEED": "0",
            "VLLM_NAVI_SCHED_BIZ_NAME": "live_partial_prefill",
            "VLLM_NAVI_SCHED_SERVER_IP": "127.0.0.1",
            "VLLM_NAVI_SCHED_SERVER_PORT": str(model_port),
            "VLLM_NAVI_SCHED_AGENT_GIG_PORT": str(agent_port),
            "VLLM_NAVI_SCHED_MASTER_BIZ_NAME": "live_partial_master",
            "VLLM_NAVI_SCHED_TARGET_BIZ_NAME": "live_partial_prefill",
            "VLLM_NAVI_SCHED_CLIENT_BIZ_NAME": "live_partial_client",
            "VLLM_NAVI_SCHED_TIMEOUT_MS": "5000",
            "VLLM_NAVI_SCHED_BATCH_TIMEOUT_MS": "1000",
        }
        config = SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=16,
                enable_prefix_caching=True,
                prefix_caching_hash_algo="sha256",
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            scheduler_config=SimpleNamespace(
                enable_chunked_prefill=False,
                max_num_partial_prefills=1,
                max_long_partial_prefills=1,
                long_prefill_token_threshold=0,
                max_num_scheduled_tokens=1024,
            ),
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=1024,
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=FullAttentionSpec(16))],
        )
        target_core = _FakeEngineCore()
        source_core = _FakeEngineClient()

        with (
            tempfile.TemporaryDirectory(
                prefix="navi-vllm-live-partial-cancel-"
            ) as runtime_dir,
            mock.patch.dict("os.environ", environment, clear=False),
        ):
            os.chdir(runtime_dir)
            try:
                observer = VLLMEngineObserver(config, kv_cache_config)
                observer.bind_engine_core(target_core)
                master = NaviSchedMaster(
                    "live_partial_master",
                    runtime_config={
                        "thread_num": 16,
                        "local_nodes": [
                            {
                                "biz_name": "live_partial_prefill",
                                "grpc_port": observer._agent.grpc_port,
                                "support_heartbeat": True,
                            }
                        ],
                    },
                    max_wait_schedule_count=2,
                    max_wait_schedule_time_ms=1000,
                    scheduler_name="random",
                    name="live_partial_master",
                    enable_kmon=False,
                    enable_snapshot=False,
                    wait_no_query=False,
                ).init()
                os.environ["VLLM_NAVI_SCHED_MASTER_ADDRESSES"] = (
                    f"127.0.0.1:{master.grpc_port}"
                )
                transport = VLLMEngineTransport(config, source_core)
                requests = [
                    EngineCoreRequest(
                        request_id=f"live-partial-request-{index}",
                        prompt_token_ids=list(range(32)),
                        mm_features=None,
                        sampling_params=SamplingParams(max_tokens=2),
                        pooling_params=None,
                        arrival_time=time.time(),
                        lora_request=None,
                        cache_salt=None,
                        data_parallel_rank=None,
                        external_req_id=f"live-partial-external-{index}",
                    )
                    for index in range(2)
                ]
                external_request_ids = [request.request_id for request in requests]

                await asyncio.gather(
                    *(transport.add_request_async(request) for request in requests)
                )
                forwarded = []
                for _ in requests:
                    request_type, (request, request_wave) = await asyncio.to_thread(
                        target_core.input_queue.get, True, 2.0
                    )
                    self.assertEqual(EngineCoreRequestType.ADD, request_type)
                    self.assertEqual(0, request_wave)
                    forwarded.append(request)

                forwarded_by_external_request_id = {
                    request.external_req_id: request for request in forwarded
                }
                aborted_forwarded = forwarded_by_external_request_id[
                    requests[0].external_req_id
                ]
                remaining_forwarded = forwarded_by_external_request_id[
                    requests[1].external_req_id
                ]

                scheduled_requests = []
                for request in forwarded:
                    scheduled = SimpleNamespace(
                        request_id=request.request_id,
                        num_prompt_tokens=len(request.prompt_token_ids),
                        num_computed_tokens=len(request.prompt_token_ids),
                        num_cached_tokens=0,
                        max_tokens=request.sampling_params.max_tokens,
                        num_preemptions=0,
                        block_hashes=[
                            bytes(range(32)),
                            bytes(reversed(range(32))),
                        ],
                        client_index=request.client_index,
                    )
                    observer._admit_request(scheduled)
                    scheduled_requests.append(scheduled)
                scheduled_by_request_id = {
                    request.request_id: request for request in scheduled_requests
                }

                state = observer._forward_request_state(aborted_forwarded.request_id)
                self.assertIsNotNone(state)
                self.assertIs(
                    state,
                    observer._forward_request_state(remaining_forwarded.request_id),
                )
                self.assertEqual(2, len(state.native_batch.request_ids))

                await transport.abort_requests_async([external_request_ids[0]])
                abort_type, aborted_external_ids = await asyncio.to_thread(
                    target_core.input_queue.get, True, 2.0
                )
                self.assertEqual(EngineCoreRequestType.ABORT, abort_type)
                self.assertEqual([aborted_forwarded.request_id], aborted_external_ids)
                self.assertEqual(
                    [aborted_forwarded.request_id],
                    target_core.aborts_queue.get_nowait(),
                )

                aborted_request_id = int(aborted_forwarded.request_id)
                remaining_request_id = int(remaining_forwarded.request_id)
                observer._abort_forward_requests(state, {aborted_request_id})
                self.assertFalse(state.tracking_aborted)
                self.assertFalse(state.released)
                self.assertEqual({aborted_request_id}, state.cancelled_request_ids)
                self.assertEqual(
                    [remaining_request_id], list(state.native_batch.request_ids)
                )
                self.assertEqual(1, len(state.native_batch.payloads))
                self.assertEqual(
                    1, len(state.native_batch.predicted_cache_hit_token_counts)
                )
                self.assertNotIn(
                    aborted_request_id, observer._scheduled_forward_batches
                )
                self.assertIs(
                    state,
                    observer._scheduled_forward_batches[remaining_request_id],
                )

                remaining = scheduled_by_request_id[str(remaining_request_id)]
                scheduler_output = SimpleNamespace(
                    num_scheduled_tokens={
                        remaining.request_id: remaining.num_prompt_tokens
                    },
                    preempted_req_ids=set(),
                    scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
                )
                scheduler = SimpleNamespace(
                    requests={remaining.request_id: remaining},
                    running=[remaining],
                    waiting=[],
                )
                execution = Future()
                wrapped_execute = observer.wrap_execute_model(
                    lambda unused_output, non_block: execution,
                    scheduler,
                )
                self.assertIs(
                    execution,
                    wrapped_execute(scheduler_output, non_block=True),
                )
                execution.set_result("remaining-prefill-complete")
                self.assertTrue(state.prefill_complete)
                self.assertTrue(state.released)

                result = EngineCoreOutputs(
                    outputs=[
                        EngineCoreOutput(
                            request_id=remaining.request_id,
                            new_token_ids=[42],
                            finish_reason=FinishReason.STOP,
                        )
                    ],
                    finished_requests={remaining.request_id},
                )
                target_core.output_queue.put_nowait((remaining.client_index, result))
                returned = await asyncio.wait_for(
                    transport.get_output_async(), 2.0
                )
                self.assertEqual(external_request_ids[1],
                                 returned.outputs[0].request_id)
                self.assertEqual([42], returned.outputs[0].new_token_ids)
                self.assertEqual(
                    FinishReason.STOP, returned.outputs[0].finish_reason
                )
                self.assertEqual({}, transport._request_ids)
                self.assertEqual({}, transport._external_request_ids)
            finally:
                if transport is not None:
                    transport.shutdown()
                if observer is not None:
                    observer.shutdown()
                if master is not None:
                    master.stop()
                os.chdir(original_dir)

        self.assertEqual([], _wait_for_gig_threads_to_stop())

    async def test_live_navi_graph_stops_with_unfinished_request_in_any_role_order(self):
        stop_orders = (
            ("client", "agent", "master"),
            ("agent", "master", "client"),
            ("master", "client", "agent"),
        )
        original_dir = os.getcwd()

        for iteration, stop_order in enumerate(stop_orders):
            with self.subTest(stop_order=stop_order):
                model_port, agent_port = _reserve_ports(2)
                prefix = f"live_vllm_shutdown_{iteration}"
                environment = {
                    "PYTHONHASHSEED": "0",
                    "VLLM_NAVI_SCHED_BIZ_NAME": f"{prefix}_prefill",
                    "VLLM_NAVI_SCHED_SERVER_IP": "127.0.0.1",
                    "VLLM_NAVI_SCHED_SERVER_PORT": str(model_port),
                    "VLLM_NAVI_SCHED_AGENT_GIG_PORT": str(agent_port),
                    "VLLM_NAVI_SCHED_MASTER_BIZ_NAME": f"{prefix}_master",
                    "VLLM_NAVI_SCHED_TARGET_BIZ_NAME": f"{prefix}_prefill",
                    "VLLM_NAVI_SCHED_CLIENT_BIZ_NAME": f"{prefix}_client",
                    "VLLM_NAVI_SCHED_TIMEOUT_MS": "1000",
                }
                config = SimpleNamespace(
                    cache_config=SimpleNamespace(
                        block_size=16,
                        enable_prefix_caching=True,
                        prefix_caching_hash_algo="sha256",
                    ),
                    parallel_config=SimpleNamespace(
                        decode_context_parallel_size=1,
                        prefill_context_parallel_size=1,
                    ),
                    scheduler_config=SimpleNamespace(
                        enable_chunked_prefill=False,
                        max_num_partial_prefills=1,
                        max_long_partial_prefills=1,
                        long_prefill_token_threshold=0,
                        max_num_scheduled_tokens=1024,
                    ),
                )
                kv_cache_config = SimpleNamespace(
                    num_blocks=1024,
                    kv_cache_groups=[
                        SimpleNamespace(kv_cache_spec=FullAttentionSpec(16))
                    ],
                )
                target_core = _FakeEngineCore()
                source_core = _FakeEngineClient()
                master = None
                observer = None
                transport = None

                with (
                    tempfile.TemporaryDirectory(
                        prefix="navi-vllm-live-shutdown-"
                    ) as runtime_dir,
                    mock.patch.dict("os.environ", environment, clear=False),
                ):
                    os.chdir(runtime_dir)
                    try:
                        observer = VLLMEngineObserver(config, kv_cache_config)
                        observer.bind_engine_core(target_core)
                        master = NaviSchedMaster(
                            f"{prefix}_master",
                            runtime_config={
                                "thread_num": 16,
                                "local_nodes": [
                                    {
                                        "biz_name": f"{prefix}_prefill",
                                        "grpc_port": observer._agent.grpc_port,
                                        "support_heartbeat": True,
                                    }
                                ],
                            },
                            max_wait_schedule_count=1,
                            scheduler_name="random",
                            name=f"{prefix}_master",
                            enable_kmon=False,
                            enable_snapshot=False,
                            wait_no_query=False,
                        ).init()
                        os.environ["VLLM_NAVI_SCHED_MASTER_ADDRESSES"] = (
                            f"127.0.0.1:{master.grpc_port}"
                        )
                        transport = VLLMEngineTransport(config, source_core)

                        forwarded = None
                        for attempt in range(100):
                            request = EngineCoreRequest(
                                request_id=f"{prefix}_request_{attempt}",
                                prompt_token_ids=list(range(32)),
                                mm_features=None,
                                sampling_params=SamplingParams(max_tokens=2),
                                pooling_params=None,
                                arrival_time=time.time(),
                                lora_request=None,
                                cache_salt=None,
                                data_parallel_rank=None,
                                external_req_id=f"{prefix}_external",
                            )
                            try:
                                await transport.add_request_async(request)
                            except ConcurrencyException:
                                await asyncio.sleep(0.02)
                                continue
                            request_type, (forwarded, request_wave) = (
                                await asyncio.to_thread(
                                    target_core.input_queue.get, True, 2.0
                                )
                            )
                            self.assertEqual(EngineCoreRequestType.ADD, request_type)
                            self.assertEqual(0, request_wave)
                            break
                        self.assertIsNotNone(forwarded)

                        stoppers = {
                            "client": transport.shutdown,
                            "agent": observer.shutdown,
                            "master": master.stop,
                        }
                        shutdown_started = time.monotonic()
                        for role in stop_order:
                            stoppers[role]()
                        shutdown_elapsed_s = time.monotonic() - shutdown_started
                    finally:
                        if transport is not None:
                            transport.shutdown()
                        if observer is not None:
                            observer.shutdown()
                        if master is not None:
                            master.stop()
                        os.chdir(original_dir)

                self.assertLess(shutdown_elapsed_s, 5.0)
                self.assertTrue(source_core.shutdown_called)
                self.assertEqual([], _wait_for_gig_threads_to_stop())

    async def test_preprocess_failure_returns_terminal_error_to_source(self):
        bridge = _FakeBridge()
        fake_client = _FakeNaviClient(bridge)
        engine_client = _FakeEngineClient()
        target_core = _FailingPreprocessEngineCore()
        environment = {
            "PYTHONHASHSEED": "0",
            "VLLM_NAVI_SCHED_BIZ_NAME": "prefill",
            "VLLM_NAVI_SCHED_SERVER_IP": "127.0.0.1",
            "VLLM_NAVI_SCHED_SERVER_PORT": "8001",
            "VLLM_NAVI_SCHED_AGENT_GIG_PORT": "0",
            "VLLM_NAVI_SCHED_MASTER_BIZ_NAME": "master",
            "VLLM_NAVI_SCHED_TARGET_BIZ_NAME": "prefill",
            "VLLM_NAVI_SCHED_MASTER_ADDRESSES": "127.0.0.1:9000",
        }
        config = SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=16,
                enable_prefix_caching=True,
                prefix_caching_hash_algo="sha256",
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            scheduler_config=SimpleNamespace(
                enable_chunked_prefill=False,
                max_num_partial_prefills=1,
                max_long_partial_prefills=1,
                long_prefill_token_threshold=0,
                max_num_scheduled_tokens=1024,
            ),
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=1024,
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=FullAttentionSpec(16))],
        )

        with (
            mock.patch.dict("os.environ", environment, clear=False),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.NaviSchedAgent",
                _FakeAgent,
            ),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineObserver.naviSchedModule",
                return_value=SimpleNamespace(_VLLMAgentBridge=lambda unused: bridge),
            ),
            mock.patch(
                "navi_sched.adapters.vllm.VLLMEngineTransport.NaviSchedClient",
                return_value=fake_client,
            ),
        ):
            observer = VLLMEngineObserver(config, kv_cache_config)
            observer.bind_engine_core(target_core)
            transport = VLLMEngineTransport(config, engine_client)
            bridge.response_kind = transport._response_kind
            request = EngineCoreRequest(
                request_id="invalid-request",
                prompt_token_ids=list(range(16)),
                mm_features=None,
                sampling_params=SamplingParams(max_tokens=1),
                pooling_params=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
                external_req_id="external-invalid-request",
            )
            external_request_id = request.request_id

            await transport.add_request_async(request)
            returned = await asyncio.wait_for(transport.get_output_async(), 2.0)

            self.assertEqual(FinishReason.ERROR, returned.outputs[0].finish_reason)
            self.assertEqual(external_request_id, returned.outputs[0].request_id)
            self.assertEqual({external_request_id}, returned.finished_requests)
            self.assertTrue(target_core.input_queue.empty())
            self.assertEqual(1, len(bridge.released_batches))
            self.assertFalse(observer._pending_forward_batches)
            self.assertFalse(observer._scheduled_forward_batches)
            self.assertFalse(observer._response_request_ids)
            self.assertFalse(observer._inactive_response_request_ids)

            transport.shutdown()
            observer.shutdown()


if __name__ == "__main__":
    unittest.main()
