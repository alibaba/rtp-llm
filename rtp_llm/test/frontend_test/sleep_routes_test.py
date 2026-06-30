"""Unit tests for the /sleep|wake_up|is_sleeping|sleep_status frontend proxy routes
and the GrpcClientWrapper sleep methods.

The FastAPI app registers only the lightweight sleep routes, and the gRPC client
is replaced by an AsyncMock, so no backend process is required.
"""

import unittest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rtp_llm.frontend.sleep_routes import register_sleep_routes
from rtp_llm.frontend.worker_address_utils import (
    SLEEP_CONTROL_ADDRESSES_ENV,
    SLEEP_INFER_CONTROL_ADDRESSES_ENV,
    get_control_addrs_from_env,
    get_control_addrs_from_world_info,
    get_dp_addrs_from_world_info,
    infer_control_addrs_from_gang_metadata,
)

SLEEP_STATUS_OK: Dict[str, Any] = {
    "sleep_mode_enabled": True,
    "effective": True,
    "supported_levels": [1],
    "supported_modes": ["wait", "abort"],
    "disabled_reason": "",
    "state": "SLEEPING",
    "sleep_epoch": "1",
    "kv_memory_state": "PAUSED",
    "device_kv_cache_valid": False,
    "active_request_count": "0",
    "active_cache_transfer_count": "0",
    "gpu_resource_state": "RELEASED",
    "last_error": "",
}


def lifecycle_operation_for_state(state: str) -> str:
    if state in ("DRAINING", "SUSPENDING"):
        return "sleep"
    if state == "WAKING_UP":
        return "wake_up"
    if state == "ERROR":
        return "error"
    return "none"


class _FakeProtoMessage:
    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class _FakeSleepRequestPB(_FakeProtoMessage):
    def __init__(
        self,
        level: int = 0,
        mode: str = "",
        timeout_ms: int = 0,
        reason: str = "",
        tags=None,
        prepare_only: bool = False,
        commit_only: bool = False,
        **_: Any,
    ):
        self.level = level
        self.mode = mode
        self.timeout_ms = timeout_ms
        self.reason = reason
        self.tags = list(tags or [])
        self.prepare_only = prepare_only
        self.commit_only = commit_only

    def CopyFrom(self, other: "_FakeSleepRequestPB"):
        self.level = other.level
        self.mode = other.mode
        self.timeout_ms = other.timeout_ms
        self.reason = other.reason
        self.tags = list(other.tags)
        self.prepare_only = other.prepare_only
        self.commit_only = other.commit_only


class _FakeWakeUpRequestPB(_FakeProtoMessage):
    def __init__(
        self,
        prepare_only: bool = False,
        commit_only: bool = False,
        **_: Any,
    ):
        self.prepare_only = prepare_only
        self.commit_only = commit_only


class _FakeSleepStatusResponsePB(_FakeProtoMessage):
    def __init__(self, **kwargs: Any):
        defaults = {
            "state": "",
            "sleep_epoch": 0,
            "kv_memory_state": "",
            "device_kv_cache_valid": False,
            "active_request_count": 0,
            "active_cache_transfer_count": 0,
            "gpu_resource_state": "",
            "last_error": "",
            "sleep_mode_enabled": False,
            "effective": False,
            "supported_levels": [],
            "supported_modes": [],
            "disabled_reason": "",
        }
        defaults.update(kwargs)
        for key, value in defaults.items():
            if isinstance(value, list):
                value = list(value)
            setattr(self, key, value)


def _install_sleep_proto_test_fallback(pb2, grpc_client_wrapper_module):
    """Let direct unittest runs work when the checked-in pb2 is stale."""
    if not hasattr(pb2, "SleepRequestPB"):
        pb2.SleepRequestPB = _FakeSleepRequestPB
    if not hasattr(pb2, "WakeUpRequestPB"):
        pb2.WakeUpRequestPB = _FakeWakeUpRequestPB
    if not hasattr(pb2, "SleepStatusResponsePB"):
        pb2.SleepStatusResponsePB = _FakeSleepStatusResponsePB

    message_to_dict = grpc_client_wrapper_module.MessageToDict
    if getattr(message_to_dict, "_supports_sleep_test_fakes", False):
        return

    def message_to_dict_with_sleep_fakes(message, *args, **kwargs):
        if isinstance(message, _FakeProtoMessage):
            return message.to_dict()
        return message_to_dict(message, *args, **kwargs)

    message_to_dict_with_sleep_fakes._supports_sleep_test_fakes = True
    grpc_client_wrapper_module.MessageToDict = message_to_dict_with_sleep_fakes


def build_test_client(grpc_post_request: AsyncMock) -> TestClient:
    grpc_client = MagicMock()
    grpc_client.post_request = grpc_post_request
    app = FastAPI()
    register_sleep_routes(app, grpc_client)
    return TestClient(app)


class FakeFfnDisaggregateConfig:
    def __init__(self):
        self.enable_ffn_disaggregate = False
        self.attention_tp_size = 1
        self.attention_dp_size = 1

    def to_string(self) -> str:
        return "FakeFfnDisaggregateConfig"


class FakeParallelismConfig:
    def __init__(self):
        self.tp_size = 1
        self.world_rank = 0
        self.world_size = 1
        self.local_world_size = 1
        self.ffn_disaggregate_config = FakeFfnDisaggregateConfig()


class FakeServerConfig:
    def __init__(self):
        self.start_port = 20000
        self.worker_info_port_num = 8


class FakeDistributeConfig:
    def __init__(self, gang_config_string="", distribute_config_file=""):
        self.gang_config_string = gang_config_string
        self.distribute_config_file = distribute_config_file


class FakeWorkerInfo:
    def __init__(
        self,
        ip: str,
        local_rank: int,
        world_rank: int,
        name: str,
        server_port: int,
        worker_info_port_num: int,
    ):
        self.ip = ip
        self.local_rank = local_rank
        self.world_rank = world_rank
        self.name = name
        self.server_port = server_port
        self.worker_info_port_num = worker_info_port_num

    @property
    def rpc_server_port(self) -> int:
        return self.server_port + self.local_rank * self.worker_info_port_num + 1


class FakeWorldInfo:
    def __init__(self, members, master, self_worker, num_nodes, initialized):
        self.members = members
        self.master = master
        self.self = self_worker
        self.num_nodes = num_nodes
        self.initialized = initialized


class SleepRoutesTest(unittest.TestCase):

    def test_sleep_success(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request)
        with client:
            response = client.post(
                "/sleep",
                json={"level": 1, "mode": "wait", "timeout_ms": 1000, "reason": "test"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        post_request.assert_awaited_once_with(
            "sleep", {"level": 1, "mode": "wait", "timeout_ms": 1000, "reason": "test"}
        )

    def test_sleep_empty_body(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep")
        self.assertEqual(response.status_code, 200)
        post_request.assert_awaited_once_with("sleep", {})

    def test_sleep_invalid_mode_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"mode": "whatever"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        post_request.assert_not_awaited()

    def test_sleep_invalid_level_type_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"level": "bad"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        post_request.assert_not_awaited()

    def test_sleep_level_zero_passes_to_backend_and_maps_unimplemented(self):
        post_request = AsyncMock(
            return_value={
                "error": "sleep level=0 state-preserving sleep is defined but not implemented",
                "grpc_status": "UNIMPLEMENTED",
            }
        )
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"level": 0})
        self.assertEqual(response.status_code, 501)
        self.assertIn("level=0", response.json()["error"])
        post_request.assert_awaited_once_with("sleep", {"level": 0})

    def test_sleep_invalid_tags_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"tags": "kv_cache"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        post_request.assert_not_awaited()

    def test_sleep_invalid_tag_element_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"tags": ["kv_cache", ""]})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        post_request.assert_not_awaited()

    def test_sleep_null_tags_are_treated_as_empty_list(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"tags": None})
        self.assertEqual(response.status_code, 200)
        post_request.assert_awaited_once_with("sleep", {"tags": None})

    def test_sleep_phase_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"phase": "prepare"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        post_request.assert_not_awaited()

    def test_sleep_prepare_only_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"prepare_only": True})
        self.assertEqual(response.status_code, 400)
        self.assertIn("prepare_only", response.json()["error"])
        post_request.assert_not_awaited()

    def test_sleep_conflict_maps_to_409(self):
        post_request = AsyncMock(
            return_value={
                "error": "sleep rejected in state WAKING_UP",
                "grpc_status": "FAILED_PRECONDITION",
            }
        )
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={})
        self.assertEqual(response.status_code, 409)
        self.assertIn("error", response.json())

    def test_sleep_disabled_maps_to_501(self):
        post_request = AsyncMock(
            return_value={
                "error": "sleep mode is disabled",
                "grpc_status": "UNIMPLEMENTED",
                "sleep_mode_enabled": False,
                "effective": False,
            }
        )
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={})
        self.assertEqual(response.status_code, 501)
        self.assertFalse(response.json()["effective"])

    def test_wake_up_success(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request)
        with client:
            response = client.post("/wake_up")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        post_request.assert_awaited_once_with("wake_up", {})

    def test_wake_up_phase_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/wake_up", json={"phase": "prepare"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        post_request.assert_not_awaited()

    def test_wake_up_commit_only_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/wake_up", json={"commit_only": True})
        self.assertEqual(response.status_code, 400)
        self.assertIn("commit_only", response.json()["error"])
        post_request.assert_not_awaited()

    def test_wake_up_backend_error_maps_to_500(self):
        post_request = AsyncMock(return_value={"error": "backend unreachable"})
        client = build_test_client(post_request)
        with client:
            response = client.post("/wake_up")
        self.assertEqual(response.status_code, 500)

    def test_sleep_status_schema_passthrough(self):
        post_request = AsyncMock(return_value=dict(SLEEP_STATUS_OK))
        client = build_test_client(post_request)
        with client:
            response = client.get("/sleep_status")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        for key in SLEEP_STATUS_OK:
            self.assertIn(key, body)
        post_request.assert_awaited_once_with("sleep_status", {})

    def test_is_sleeping_schema_passthrough(self):
        post_request = AsyncMock(
            return_value={
                "is_sleeping": True,
                "sleep_mode_enabled": True,
                "effective": True,
                "supported_levels": [1],
                "supported_modes": ["wait", "abort"],
                "state": "SLEEPING",
                "disabled_reason": "",
            }
        )
        client = build_test_client(post_request)
        with client:
            response = client.get("/is_sleeping")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["is_sleeping"])
        post_request.assert_awaited_once_with("is_sleeping", {})

    def test_sleep_status_backend_error_maps_to_500(self):
        post_request = AsyncMock(return_value={"error": "no backend"})
        client = build_test_client(post_request)
        with client:
            response = client.get("/sleep_status")
        self.assertEqual(response.status_code, 500)


class GrpcClientWrapperSleepTest(unittest.IsolatedAsyncioTestCase):

    def _build_wrapper(
        self, control_addresses=None, expected_control_address_count=None
    ):
        import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
        from rtp_llm.utils import grpc_client_wrapper

        _install_sleep_proto_test_fallback(pb2, grpc_client_wrapper)
        GrpcClientWrapper = grpc_client_wrapper.GrpcClientWrapper

        wrapper = GrpcClientWrapper(
            server_port=12345,
            control_addresses=control_addresses or ["127.0.0.1:10001"],
            expected_control_address_count=expected_control_address_count,
        )
        wrapper.channel = MagicMock()
        wrapper.stub = MagicMock()
        for address in wrapper.control_addresses:
            wrapper._dp_channels[address] = MagicMock()
            wrapper._dp_stubs[address] = MagicMock()
        return wrapper, pb2

    def _aio_error(self, code, details):
        return grpc.aio.AioRpcError(
            code=code,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details=details,
        )

    def _status_pb(self, pb2, **kwargs):
        defaults = {
            "state": "RUNNING",
            "sleep_mode_enabled": True,
            "effective": True,
            "supported_levels": [1],
            "supported_modes": ["wait", "abort"],
            "kv_memory_state": "ACTIVE",
            "device_kv_cache_valid": True,
            "gpu_resource_state": "ACTIVE",
        }
        defaults.update(kwargs)
        return pb2.SleepStatusResponsePB(**defaults)

    async def test_control_plane_sleep_wake_up_smoke_flow(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(
            control_addresses=addresses,
            expected_control_address_count=len(addresses),
        )
        rank_statuses: Dict[str, Dict[str, Any]] = {
            address: {
                "state": "RUNNING",
                "sleep_epoch": 0,
                "kv_memory_state": "ACTIVE",
                "device_kv_cache_valid": True,
                "active_request_count": 0,
                "active_cache_transfer_count": 0,
                "gpu_resource_state": "ACTIVE",
            }
            for address in addresses
        }
        observed_states = []

        for address in addresses:

            async def get_status(*args, address=address, **kwargs):
                return self._status_pb(pb2, **rank_statuses[address])

            async def sleep_rpc(request, *args, address=address, **kwargs):
                if request.prepare_only:
                    rank_statuses[address].update(
                        state="DRAINING",
                        sleep_epoch=1,
                        kv_memory_state="ACTIVE",
                        device_kv_cache_valid=True,
                        gpu_resource_state="ACTIVE",
                    )
                elif request.commit_only:
                    self.assertEqual(rank_statuses[address]["state"], "DRAINING")
                    rank_statuses[address].update(
                        state="SLEEPING",
                        sleep_epoch=1,
                        kv_memory_state="PAUSED",
                        device_kv_cache_valid=False,
                        gpu_resource_state="RELEASED",
                    )
                else:
                    rank_statuses[address].update(state="SLEEPING")
                observed_states.append(rank_statuses[address]["state"])
                return pb2.EmptyPB()

            async def wake_up_rpc(request, *args, address=address, **kwargs):
                if request.prepare_only:
                    self.assertEqual(rank_statuses[address]["state"], "SLEEPING")
                    rank_statuses[address].update(
                        state="WAKING_UP",
                        kv_memory_state="WAKING_UP",
                        device_kv_cache_valid=False,
                        gpu_resource_state="RESTORING",
                    )
                elif request.commit_only:
                    self.assertEqual(rank_statuses[address]["state"], "WAKING_UP")
                    rank_statuses[address].update(
                        state="RUNNING",
                        kv_memory_state="ACTIVE",
                        device_kv_cache_valid=True,
                        gpu_resource_state="ACTIVE",
                    )
                else:
                    rank_statuses[address].update(state="RUNNING")
                observed_states.append(rank_statuses[address]["state"])
                return pb2.EmptyPB()

            wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
                side_effect=get_status
            )
            wrapper._dp_stubs[address].SleepServing = AsyncMock(side_effect=sleep_rpc)
            wrapper._dp_stubs[address].WakeUpServing = AsyncMock(
                side_effect=wake_up_rpc
            )

        initial_status = await wrapper.get_sleep_status()
        self.assertEqual(initial_status["state"], "RUNNING")
        self.assertEqual(lifecycle_operation_for_state(initial_status["state"]), "none")

        sleep_result = await wrapper.sleep_serving(
            {"level": 1, "mode": "wait", "timeout_ms": 1000, "reason": "smoke"}
        )
        self.assertEqual(sleep_result, {"status": "ok"})
        self.assertIn("DRAINING", observed_states)
        self.assertEqual(lifecycle_operation_for_state("DRAINING"), "sleep")

        sleeping_status = await wrapper.get_sleep_status()
        self.assertEqual(sleeping_status["state"], "SLEEPING")
        self.assertEqual(sleeping_status["gpu_resource_state"], "RELEASED")
        self.assertFalse(bool(sleeping_status["device_kv_cache_valid"]))
        self.assertEqual(
            lifecycle_operation_for_state(sleeping_status["state"]), "none"
        )

        wake_up_result = await wrapper.wake_up_serving()
        self.assertEqual(wake_up_result, {"status": "ok"})
        self.assertIn("WAKING_UP", observed_states)
        self.assertEqual(lifecycle_operation_for_state("WAKING_UP"), "wake_up")

        running_status = await wrapper.get_sleep_status()
        self.assertEqual(running_status["state"], "RUNNING")
        self.assertEqual(running_status["gpu_resource_state"], "ACTIVE")
        self.assertTrue(bool(running_status["device_kv_cache_valid"]))
        self.assertEqual(lifecycle_operation_for_state(running_status["state"]), "none")

        for address in addresses:
            self.assertEqual(wrapper._dp_stubs[address].SleepServing.await_count, 2)
            self.assertEqual(wrapper._dp_stubs[address].WakeUpServing.await_count, 2)

    async def test_get_sleep_status_exposes_in_progress_states_for_control_plane(self):
        cases = [
            ("DRAINING", "ACTIVE", "sleep"),
            ("SUSPENDING", "RELEASING", "sleep"),
            ("WAKING_UP", "RESTORING", "wake_up"),
        ]
        for state, gpu_resource_state, operation in cases:
            with self.subTest(state=state):
                wrapper, pb2 = self._build_wrapper()
                address = wrapper.control_addresses[0]
                wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
                    return_value=self._status_pb(
                        pb2,
                        state=state,
                        gpu_resource_state=gpu_resource_state,
                    )
                )

                result = await wrapper.get_sleep_status()

                self.assertEqual(result["state"], state)
                self.assertEqual(result["gpu_resource_state"], gpu_resource_state)
                self.assertEqual(
                    lifecycle_operation_for_state(result["state"]), operation
                )

    async def test_sleep_serving_broadcasts_all_control_ranks(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        for address in addresses:
            wrapper._dp_stubs[address].SleepServing = AsyncMock(
                return_value=pb2.EmptyPB()
            )
            wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
                return_value=self._status_pb(
                    pb2,
                    state="SLEEPING",
                    sleep_epoch=1,
                    kv_memory_state="PAUSED",
                    device_kv_cache_valid=False,
                    gpu_resource_state="RELEASED",
                )
            )

        result = await wrapper.sleep_serving(
            {"mode": "abort", "timeout_ms": 1000, "reason": "test"}
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(set(result.keys()), {"status"})
        for address in addresses:
            stub = wrapper._dp_stubs[address]
            self.assertEqual(stub.SleepServing.await_count, 2)
            prepare_request = stub.SleepServing.await_args_list[0].args[0]
            commit_request = stub.SleepServing.await_args_list[1].args[0]
            self.assertEqual(prepare_request.level, 1)
            self.assertEqual(prepare_request.mode, "abort")
            self.assertEqual(prepare_request.timeout_ms, 1000)
            self.assertEqual(prepare_request.reason, "test")
            self.assertTrue(prepare_request.prepare_only)
            self.assertFalse(prepare_request.commit_only)
            self.assertFalse(commit_request.prepare_only)
            self.assertTrue(commit_request.commit_only)
            self.assertEqual(commit_request.timeout_ms, 0)

    async def test_sleep_serving_phase_rejected_before_status_probe(self):
        wrapper, pb2 = self._build_wrapper()
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock()

        result = await wrapper.sleep_serving({"phase": "prepare"})

        self.assertEqual(result["grpc_status"], "INVALID_ARGUMENT")
        wrapper._dp_stubs[address].GetSleepStatus.assert_not_awaited()

    async def test_sleep_serving_prepare_only_rejected_before_status_probe(self):
        wrapper, pb2 = self._build_wrapper()
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock()

        result = await wrapper.sleep_serving({"prepare_only": True})

        self.assertEqual(result["grpc_status"], "INVALID_ARGUMENT")
        self.assertIn("prepare_only", result["error"])
        wrapper._dp_stubs[address].GetSleepStatus.assert_not_awaited()

    async def test_wake_up_serving_success(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        for address in addresses:
            wrapper._dp_stubs[address].WakeUpServing = AsyncMock(
                return_value=pb2.EmptyPB()
            )
            wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
                return_value=self._status_pb(
                    pb2,
                    state="RUNNING",
                    sleep_epoch=1,
                    kv_memory_state="ACTIVE",
                    device_kv_cache_valid=False,
                    gpu_resource_state="ACTIVE",
                )
            )

        result = await wrapper.wake_up_serving()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(set(result.keys()), {"status"})
        for address in addresses:
            stub = wrapper._dp_stubs[address]
            self.assertEqual(stub.WakeUpServing.await_count, 2)
            prepare_request = stub.WakeUpServing.await_args_list[0].args[0]
            commit_request = stub.WakeUpServing.await_args_list[1].args[0]
            self.assertTrue(prepare_request.prepare_only)
            self.assertFalse(prepare_request.commit_only)
            self.assertFalse(commit_request.prepare_only)
            self.assertTrue(commit_request.commit_only)

    async def test_wake_up_serving_phase_rejected_before_status_probe(self):
        wrapper, pb2 = self._build_wrapper()
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock()

        result = await wrapper.wake_up_serving({"phase": "prepare"})

        self.assertEqual(result["grpc_status"], "INVALID_ARGUMENT")
        wrapper._dp_stubs[address].GetSleepStatus.assert_not_awaited()

    async def test_wake_up_serving_commit_only_rejected_before_status_probe(self):
        wrapper, pb2 = self._build_wrapper()
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock()

        result = await wrapper.wake_up_serving({"commit_only": True})

        self.assertEqual(result["grpc_status"], "INVALID_ARGUMENT")
        self.assertIn("commit_only", result["error"])
        wrapper._dp_stubs[address].GetSleepStatus.assert_not_awaited()

    async def test_get_sleep_status_returns_full_schema(self):
        wrapper, pb2 = self._build_wrapper()
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
            return_value=self._status_pb(
                pb2,
                state="RUNNING",
                kv_memory_state="ACTIVE",
                device_kv_cache_valid=True,
                gpu_resource_state="ACTIVE",
            )
        )

        result = await wrapper.get_sleep_status()

        expected_keys = {
            "sleep_mode_enabled",
            "effective",
            "supported_levels",
            "supported_modes",
            "disabled_reason",
            "state",
            "sleep_epoch",
            "kv_memory_state",
            "device_kv_cache_valid",
            "active_request_count",
            "active_cache_transfer_count",
            "gpu_resource_state",
            "last_error",
        }
        self.assertEqual(expected_keys, set(result.keys()))
        self.assertEqual(result["state"], "RUNNING")

    async def test_get_sleep_status_disables_sleep_when_control_coverage_incomplete(
        self,
    ):
        wrapper, pb2 = self._build_wrapper(
            control_addresses=["127.0.0.1:10001"],
            expected_control_address_count=2,
        )
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
            return_value=self._status_pb(pb2)
        )

        result = await wrapper.get_sleep_status()

        self.assertFalse(result["effective"])
        self.assertEqual(result["supported_levels"], [])
        self.assertEqual(result["supported_modes"], [])
        self.assertIn("control address coverage incomplete", result["disabled_reason"])

    async def test_get_sleep_status_refreshes_control_addresses_from_resolver(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(
            control_addresses=[addresses[0]],
            expected_control_address_count=2,
        )
        wrapper._control_address_resolver = MagicMock(return_value=addresses)
        for address in addresses:
            wrapper._dp_channels[address] = MagicMock()
            wrapper._dp_stubs[address] = MagicMock()
            wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
                return_value=self._status_pb(pb2)
            )

        result = await wrapper.get_sleep_status()

        self.assertTrue(result["effective"])
        self.assertEqual(wrapper.control_addresses, addresses)
        wrapper._control_address_resolver.assert_called_once()
        for address in addresses:
            wrapper._dp_stubs[address].GetSleepStatus.assert_awaited_once()

    async def test_sleep_serving_rejects_when_control_coverage_incomplete(self):
        wrapper, pb2 = self._build_wrapper(
            control_addresses=["127.0.0.1:10001"],
            expected_control_address_count=2,
        )
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
            return_value=self._status_pb(pb2)
        )
        wrapper._dp_stubs[address].SleepServing = AsyncMock(return_value=pb2.EmptyPB())

        result = await wrapper.sleep_serving({})

        self.assertEqual(result["grpc_status"], "UNIMPLEMENTED")
        self.assertFalse(result["effective"])
        self.assertIn("control address coverage incomplete", result["error"])
        wrapper._dp_stubs[address].SleepServing.assert_not_awaited()

    async def test_sleep_serving_returns_unimplemented_when_disabled(self):
        wrapper, pb2 = self._build_wrapper()
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
            return_value=self._status_pb(
                pb2,
                sleep_mode_enabled=False,
                effective=False,
                supported_levels=[],
                supported_modes=[],
                disabled_reason="sleep mode is disabled",
            )
        )
        wrapper._dp_stubs[address].SleepServing = AsyncMock(return_value=pb2.EmptyPB())

        result = await wrapper.sleep_serving({})

        self.assertEqual(result["grpc_status"], "UNIMPLEMENTED")
        self.assertFalse(result["effective"])
        wrapper._dp_stubs[address].SleepServing.assert_not_awaited()

    async def test_sleep_serving_invalid_request_rejected_before_status_probe(self):
        wrapper, pb2 = self._build_wrapper()
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock()

        result = await wrapper.sleep_serving({"level": "bad"})

        self.assertEqual(result["grpc_status"], "INVALID_ARGUMENT")
        wrapper._dp_stubs[address].GetSleepStatus.assert_not_awaited()

    async def test_sleep_serving_level_zero_returns_unimplemented_before_status_probe(
        self,
    ):
        wrapper, pb2 = self._build_wrapper()
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock()

        result = await wrapper.sleep_serving({"level": 0})

        self.assertEqual(result["grpc_status"], "UNIMPLEMENTED")
        self.assertIn("level=0", result["error"])
        self.assertEqual(result["supported_levels"], [1])
        wrapper._dp_stubs[address].GetSleepStatus.assert_not_awaited()

    async def test_sleep_serving_invalid_tag_element_rejected_before_status_probe(self):
        wrapper, pb2 = self._build_wrapper()
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock()

        result = await wrapper.sleep_serving({"tags": ["kv_cache", 1]})

        self.assertEqual(result["grpc_status"], "INVALID_ARGUMENT")
        wrapper._dp_stubs[address].GetSleepStatus.assert_not_awaited()

    async def test_sleep_serving_null_tags_are_treated_as_empty_list(self):
        wrapper, pb2 = self._build_wrapper()
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].SleepServing = AsyncMock(return_value=pb2.EmptyPB())
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
            side_effect=[
                self._status_pb(pb2),
                self._status_pb(
                    pb2,
                    state="SLEEPING",
                    sleep_epoch=1,
                    kv_memory_state="PAUSED",
                    device_kv_cache_valid=False,
                    gpu_resource_state="RELEASED",
                ),
            ]
        )

        result = await wrapper.sleep_serving({"tags": None})

        self.assertEqual(result["status"], "ok")
        prepare_request = (
            wrapper._dp_stubs[address].SleepServing.await_args_list[0].args[0]
        )
        self.assertEqual(list(prepare_request.tags), [])

    async def test_get_sleep_status_reports_non_converged_as_error(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        wrapper._dp_stubs[addresses[0]].GetSleepStatus = AsyncMock(
            return_value=self._status_pb(pb2, state="SLEEPING", sleep_epoch=2)
        )
        wrapper._dp_stubs[addresses[1]].GetSleepStatus = AsyncMock(
            return_value=self._status_pb(pb2, state="RUNNING", sleep_epoch=1)
        )

        result = await wrapper.get_sleep_status()

        self.assertIn("error", result)
        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertNotIn("state", result)

    async def test_sleep_serving_error_carries_per_rank_status(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        wrapper._dp_stubs[addresses[0]].SleepServing = AsyncMock(
            return_value=pb2.EmptyPB()
        )
        wrapper._dp_stubs[addresses[0]].GetSleepStatus = AsyncMock(
            return_value=self._status_pb(pb2)
        )
        wrapper._dp_stubs[addresses[0]].WakeUpServing = AsyncMock(
            return_value=pb2.EmptyPB()
        )
        wrapper._dp_stubs[addresses[1]].SleepServing = AsyncMock(
            side_effect=self._aio_error(
                grpc.StatusCode.FAILED_PRECONDITION,
                "sleep rejected in state WAKING_UP",
            )
        )
        wrapper._dp_stubs[addresses[1]].GetSleepStatus = AsyncMock(
            return_value=self._status_pb(pb2)
        )
        wrapper._dp_stubs[addresses[1]].WakeUpServing = AsyncMock(
            return_value=pb2.EmptyPB()
        )

        result = await wrapper.sleep_serving({})

        self.assertIn("error", result)
        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertEqual(result["details"][0]["address"], addresses[1])
        for address in addresses:
            wrapper._dp_stubs[address].WakeUpServing.assert_awaited_once()
            abort_request = wrapper._dp_stubs[address].WakeUpServing.await_args.args[0]
            self.assertFalse(abort_request.prepare_only)
            self.assertFalse(abort_request.commit_only)

    async def test_sleep_serving_commit_failure_returns_error_without_abort(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        wrapper._dp_stubs[addresses[0]].SleepServing = AsyncMock(
            return_value=pb2.EmptyPB()
        )
        wrapper._dp_stubs[addresses[0]].GetSleepStatus = AsyncMock(
            return_value=self._status_pb(pb2)
        )
        wrapper._dp_stubs[addresses[1]].SleepServing = AsyncMock(
            side_effect=[
                pb2.EmptyPB(),
                self._aio_error(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "releaseRestorableGpuMemory failed",
                ),
            ]
        )
        wrapper._dp_stubs[addresses[1]].GetSleepStatus = AsyncMock(
            return_value=self._status_pb(pb2)
        )
        for address in addresses:
            wrapper._dp_stubs[address].WakeUpServing = AsyncMock(
                return_value=pb2.EmptyPB()
            )

        result = await wrapper.sleep_serving({})

        self.assertIn("error", result)
        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertEqual(result["details"][0]["address"], addresses[1])
        for address in addresses:
            self.assertEqual(wrapper._dp_stubs[address].SleepServing.await_count, 2)
            wrapper._dp_stubs[address].WakeUpServing.assert_not_awaited()

    async def test_sleep_serving_rejects_non_converged_final_state(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        for address in addresses:
            wrapper._dp_stubs[address].SleepServing = AsyncMock(
                return_value=pb2.EmptyPB()
            )
        wrapper._dp_stubs[addresses[0]].GetSleepStatus = AsyncMock(
            side_effect=[
                self._status_pb(pb2, state="RUNNING", sleep_epoch=0),
                self._status_pb(pb2, state="SLEEPING", sleep_epoch=1),
            ]
        )
        wrapper._dp_stubs[addresses[1]].GetSleepStatus = AsyncMock(
            side_effect=[
                self._status_pb(pb2, state="RUNNING", sleep_epoch=0),
                self._status_pb(pb2, state="RUNNING", sleep_epoch=1),
            ]
        )

        result = await wrapper.sleep_serving({})

        self.assertIn("error", result)
        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertNotIn("state", result)

    async def test_wake_up_serving_rejects_non_converged_final_state(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        for address in addresses:
            wrapper._dp_stubs[address].WakeUpServing = AsyncMock(
                return_value=pb2.EmptyPB()
            )
        wrapper._dp_stubs[addresses[0]].GetSleepStatus = AsyncMock(
            side_effect=[
                self._status_pb(pb2, state="SLEEPING", sleep_epoch=1),
                self._status_pb(pb2, state="RUNNING", sleep_epoch=1),
            ]
        )
        wrapper._dp_stubs[addresses[1]].GetSleepStatus = AsyncMock(
            side_effect=[
                self._status_pb(pb2, state="SLEEPING", sleep_epoch=1),
                self._status_pb(pb2, state="SLEEPING", sleep_epoch=1),
            ]
        )

        result = await wrapper.wake_up_serving()

        self.assertIn("error", result)
        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertNotIn("state", result)
        for address in addresses:
            self.assertEqual(wrapper._dp_stubs[address].WakeUpServing.await_count, 2)

    async def test_wake_up_serving_prepare_failure_does_not_commit(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        wrapper._dp_stubs[addresses[0]].WakeUpServing = AsyncMock(
            return_value=pb2.EmptyPB()
        )
        wrapper._dp_stubs[addresses[0]].GetSleepStatus = AsyncMock(
            return_value=self._status_pb(pb2, state="SLEEPING", sleep_epoch=1)
        )
        wrapper._dp_stubs[addresses[1]].WakeUpServing = AsyncMock(
            side_effect=self._aio_error(
                grpc.StatusCode.FAILED_PRECONDITION,
                "restoreRestorableGpuMemory failed",
            )
        )
        wrapper._dp_stubs[addresses[1]].GetSleepStatus = AsyncMock(
            return_value=self._status_pb(pb2, state="SLEEPING", sleep_epoch=1)
        )

        result = await wrapper.wake_up_serving()

        self.assertIn("error", result)
        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertIn("prepare wake_up", result["error"])
        for address in addresses:
            self.assertEqual(wrapper._dp_stubs[address].WakeUpServing.await_count, 1)
            prepare_request = wrapper._dp_stubs[address].WakeUpServing.await_args.args[
                0
            ]
            self.assertTrue(prepare_request.prepare_only)
            self.assertFalse(prepare_request.commit_only)

    async def test_wake_up_serving_commit_failure_returns_error(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        for address in addresses:
            wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
                return_value=self._status_pb(pb2, state="SLEEPING", sleep_epoch=1)
            )
        wrapper._dp_stubs[addresses[0]].WakeUpServing = AsyncMock(
            return_value=pb2.EmptyPB()
        )
        wrapper._dp_stubs[addresses[1]].WakeUpServing = AsyncMock(
            side_effect=[
                pb2.EmptyPB(),
                self._aio_error(
                    grpc.StatusCode.FAILED_PRECONDITION, "restartEngine failed"
                ),
            ]
        )

        result = await wrapper.wake_up_serving()

        self.assertIn("error", result)
        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertIn("commit wake_up", result["error"])
        for address in addresses:
            self.assertEqual(wrapper._dp_stubs[address].WakeUpServing.await_count, 2)


class SleepControlAddressTest(unittest.TestCase):

    def test_control_addresses_env_override_accepts_csv_and_dedupes(self):
        with patch.dict(
            "os.environ",
            {
                SLEEP_CONTROL_ADDRESSES_ENV: "10.0.0.1:20001,10.0.0.2:20009;10.0.0.1:20001"
            },
            clear=False,
        ):
            self.assertEqual(
                get_control_addrs_from_env(),
                ["10.0.0.1:20001", "10.0.0.2:20009"],
            )

    def test_control_addresses_env_override_accepts_json_list(self):
        with patch.dict(
            "os.environ",
            {SLEEP_CONTROL_ADDRESSES_ENV: '["10.0.0.1:20001", "10.0.0.2:20009"]'},
            clear=False,
        ):
            self.assertEqual(
                get_control_addrs_from_env(),
                ["10.0.0.1:20001", "10.0.0.2:20009"],
            )

    def test_control_addresses_include_all_ranks_but_dp_addresses_do_not(self):
        members = [
            FakeWorkerInfo(
                ip="127.0.0.1",
                local_rank=rank,
                world_rank=rank,
                name=f"rank_{rank}",
                server_port=20000,
                worker_info_port_num=8,
            )
            for rank in range(4)
        ]
        world_info = FakeWorldInfo(
            members=members,
            master=members[0],
            self_worker=members[0],
            num_nodes=1,
            initialized=True,
        )
        pc = FakeParallelismConfig()
        pc.tp_size = 2

        dp_addresses = get_dp_addrs_from_world_info(world_info, pc)
        control_addresses = get_control_addrs_from_world_info(world_info)

        self.assertEqual(dp_addresses, ["127.0.0.1:20001", "127.0.0.1:20017"])
        self.assertEqual(
            control_addresses,
            [
                "127.0.0.1:20001",
                "127.0.0.1:20009",
                "127.0.0.1:20017",
                "127.0.0.1:20025",
            ],
        )

    def test_ffn_disaggregate_control_addresses_still_include_all_ranks(self):
        members = [
            FakeWorkerInfo(
                ip="127.0.0.1",
                local_rank=rank,
                world_rank=rank,
                name=f"rank_{rank}",
                server_port=20000,
                worker_info_port_num=8,
            )
            for rank in range(4)
        ]
        world_info = FakeWorldInfo(
            members=members,
            master=members[0],
            self_worker=members[0],
            num_nodes=1,
            initialized=True,
        )
        pc = FakeParallelismConfig()
        pc.tp_size = 1
        pc.ffn_disaggregate_config.enable_ffn_disaggregate = True
        pc.ffn_disaggregate_config.attention_tp_size = 1
        pc.ffn_disaggregate_config.attention_dp_size = 2

        dp_addresses = get_dp_addrs_from_world_info(world_info, pc)
        control_addresses = get_control_addrs_from_world_info(world_info)

        self.assertEqual(dp_addresses, ["127.0.0.1:20001", "127.0.0.1:20009"])
        self.assertEqual(
            control_addresses,
            [
                "127.0.0.1:20001",
                "127.0.0.1:20009",
                "127.0.0.1:20017",
                "127.0.0.1:20025",
            ],
        )

    def test_infer_control_addresses_from_gang_metadata_is_opt_in(self):
        pc = FakeParallelismConfig()
        pc.world_size = 4
        pc.local_world_size = 2
        gang_config = (
            "name:foo_part0,ip:10.0.0.1,port:20000;"
            "name:foo_part1,ip:10.0.0.2,port:20000"
        )
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                infer_control_addrs_from_gang_metadata(
                    FakeServerConfig(), FakeDistributeConfig(gang_config), pc
                ),
                [],
            )

    def test_infer_control_addresses_from_gang_metadata(self):
        pc = FakeParallelismConfig()
        pc.world_size = 4
        pc.local_world_size = 2
        gang_config = (
            "name:foo_part1,ip:10.0.0.2,port:20000;"
            "name:foo_part0,ip:10.0.0.1,port:20000"
        )
        with patch.dict(
            "os.environ", {SLEEP_INFER_CONTROL_ADDRESSES_ENV: "1"}, clear=True
        ):
            self.assertEqual(
                infer_control_addrs_from_gang_metadata(
                    FakeServerConfig(), FakeDistributeConfig(gang_config), pc
                ),
                [
                    "10.0.0.1:20001",
                    "10.0.0.1:20009",
                    "10.0.0.2:20001",
                    "10.0.0.2:20009",
                ],
            )


if __name__ == "__main__":
    unittest.main()
