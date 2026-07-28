"""Unit tests for the /sleep|wake_up|is_sleeping|sleep_status frontend proxy routes
and the GrpcClientWrapper sleep methods.

The FastAPI app registers only the lightweight sleep routes, and the gRPC client
is replaced by an AsyncMock, so no backend process is required.
"""

import asyncio
import json
import os
import threading
import unittest
from types import SimpleNamespace
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


class _FakeStore:
    def __init__(self):
        self.values: Dict[str, str] = {}

    def compare_set(self, key: str, expected: str, desired: str) -> bytes:
        current = self.values.get(key, "")
        if current == expected:
            self.values[key] = desired
            current = desired
        return current.encode("utf-8")


class _CheckpointFailure(RuntimeError):
    def __init__(self, message: str, *, all_running: bool = False):
        super().__init__(message)
        self.all_running = all_running


class _FakeCheckpointController:
    def __init__(self):
        self.manifest = None
        self.events = []
        self.checkpoint_calls = []
        self.restore_calls = []
        self.checkpoint_side_effect = None
        self.restore_side_effect = None

    def preflight(self, control_addresses, namespace=None):
        self.events.append(("preflight", tuple(control_addresses)))

    def read_manifest(self, control_addresses, namespace=None):
        self.events.append(("read_manifest", tuple(control_addresses)))
        return None if self.manifest is None else dict(self.manifest)

    def clear_manifest(self, control_addresses, namespace=None):
        self.events.append(("clear_manifest", tuple(control_addresses)))
        self.manifest = None

    def checkpoint_all(
        self,
        control_addresses,
        terminal_statuses,
        namespace=None,
        holder_instance=None,
        team=None,
    ):
        statuses = tuple(dict(status) for status in terminal_statuses)
        self.events.append(("checkpoint_all", tuple(control_addresses)))
        self.checkpoint_calls.append((tuple(control_addresses), statuses))
        if self.checkpoint_side_effect is not None:
            self.checkpoint_side_effect()
        self.manifest = {
            "state": "CHECKPOINTED",
            "pids": [int(status["process_id"]) for status in statuses],
        }

    def restore_all(
        self, control_addresses, namespace=None, holder_instance=None, team=None
    ):
        self.events.append(("restore_all", tuple(control_addresses)))
        self.restore_calls.append(tuple(control_addresses))
        if self.restore_side_effect is not None:
            self.restore_side_effect()
        existed = self.manifest is not None
        self.manifest = None
        return existed


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
            "process_id": 0,
            "process_starttime": 0,
            "process_pid_namespace": 0,
            "process_boot_id": "",
            "world_rank": 0,
            "role": "",
            "instance_generation_uuid": "",
            "holder_instance": "",
        }
        defaults.update(kwargs)
        for key, value in defaults.items():
            if isinstance(value, list):
                value = list(value)
            setattr(self, key, value)

class _FakeCudaCheckpointRequestPB(_FakeProtoMessage):
    def __init__(
        self,
        action: str = "",
        transaction_id: str = "",
        sleep_epoch: int = 0,
        lock_timeout_ms: int = 0,
        **_: Any,
    ):
        self.action = action
        self.transaction_id = transaction_id
        self.sleep_epoch = sleep_epoch
        self.lock_timeout_ms = lock_timeout_ms


class _FakeCudaCheckpointResponsePB(_FakeProtoMessage):
    def __init__(self, **kwargs: Any):
        defaults = {
            "success": False,
            "cuda_result": 0,
            "state": "",
            "error": "",
            "transaction_id": "",
            "sleep_epoch": 0,
            "process_id": 0,
            "process_starttime": 0,
            "process_pid_namespace": 0,
            "process_boot_id": "",
            "world_rank": 0,
            "holder_instance": "",
        }
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)


def _install_sleep_proto_test_fallback(pb2, grpc_client_wrapper_module):
    """Let direct unittest runs work when the checked-in pb2 is stale."""
    if not hasattr(pb2, "SleepRequestPB"):
        pb2.SleepRequestPB = _FakeSleepRequestPB
    if not hasattr(pb2, "WakeUpRequestPB"):
        pb2.WakeUpRequestPB = _FakeWakeUpRequestPB
    sleep_status_pb = getattr(pb2, "SleepStatusResponsePB", None)
    sleep_status_fields = getattr(
        getattr(sleep_status_pb, "DESCRIPTOR", None), "fields_by_name", {}
    )
    if sleep_status_pb is None or "process_starttime" not in sleep_status_fields:
        pb2.SleepStatusResponsePB = _FakeSleepStatusResponsePB
    if not hasattr(pb2, "CudaCheckpointRequestPB"):
        pb2.CudaCheckpointRequestPB = _FakeCudaCheckpointRequestPB
    if not hasattr(pb2, "CudaCheckpointResponsePB"):
        pb2.CudaCheckpointResponsePB = _FakeCudaCheckpointResponsePB

    message_to_dict = grpc_client_wrapper_module.MessageToDict
    if getattr(message_to_dict, "_supports_sleep_test_fakes", False):
        return

    def message_to_dict_with_sleep_fakes(message, *args, **kwargs):
        if isinstance(message, _FakeProtoMessage):
            return message.to_dict()
        return message_to_dict(message, *args, **kwargs)

    message_to_dict_with_sleep_fakes._supports_sleep_test_fakes = True
    grpc_client_wrapper_module.MessageToDict = message_to_dict_with_sleep_fakes


def build_test_client(
    grpc_post_request: AsyncMock, configured_sleep_level: int = 1
) -> TestClient:
    grpc_client = MagicMock()
    grpc_client.configured_sleep_level = configured_sleep_level
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
            "sleep",
            {
                "level": 1,
                "mode": "wait",
                "timeout_ms": 1000,
                "reason": "test",
            },
        )

    def test_sleep_level3_is_forwarded(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request, configured_sleep_level=3)
        with client:
            response = client.post("/sleep", json={"level": 3})
        self.assertEqual(response.status_code, 200)
        post_request.assert_awaited_once_with("sleep", {"level": 3})

    def test_sleep_empty_body(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep")
        self.assertEqual(response.status_code, 200)
        post_request.assert_awaited_once_with("sleep", {"level": 1})

    def test_sleep_empty_body_uses_configured_level(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request, configured_sleep_level=2)
        with client:
            response = client.post("/sleep")
        self.assertEqual(response.status_code, 200)
        post_request.assert_awaited_once_with("sleep", {"level": 2})

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
        post_request.assert_awaited_once_with("sleep", {"tags": [], "level": 1})

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
        self,
        control_addresses=None,
        expected_control_address_count=None,
        lifecycle_store=None,
        checkpoint_controller=None,
        sleep_enabled=True,
        configured_level=None,
        level3_enabled=None,
        single_node=None,
        rdma_enabled=False,
    ):
        import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
        from rtp_llm.utils import grpc_client_wrapper

        _install_sleep_proto_test_fallback(pb2, grpc_client_wrapper)
        GrpcClientWrapper = grpc_client_wrapper.GrpcClientWrapper

        if configured_level is None:
            configured_level = 3 if checkpoint_controller is not None else 1
        if level3_enabled is None:
            level3_enabled = sleep_enabled and configured_level == 3
        wrapper = GrpcClientWrapper(
            server_port=12345,
            control_addresses=control_addresses or ["127.0.0.1:10001"],
            expected_control_address_count=expected_control_address_count,
            lifecycle_store=lifecycle_store,
            checkpoint_controller=checkpoint_controller,
            sleep_enabled=sleep_enabled,
            configured_level=configured_level,
            level3_enabled=level3_enabled,
            single_node=single_node,
            rdma_enabled=rdma_enabled,
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

    async def test_disabled_l1_l2_do_not_construct_or_poll_checkpoint_controller(self):
        for sleep_enabled, configured_level in ((False, 1), (True, 1), (True, 2)):
            with self.subTest(
                sleep_enabled=sleep_enabled, configured_level=configured_level
            ):
                store = MagicMock()
                with patch(
                    "rtp_llm.utils.grpc_client_wrapper._CheckpointControllerAdapter"
                ) as adapter_type:
                    wrapper, pb2 = self._build_wrapper(
                        lifecycle_store=store,
                        sleep_enabled=sleep_enabled,
                        configured_level=configured_level,
                    )
                self.assertIsNone(wrapper._checkpoint_controller)
                adapter_type.assert_not_called()
                wrapper.health_check = AsyncMock(return_value={"status": "ok"})
                address = wrapper.control_addresses[0]
                wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
                    return_value=self._status_pb(pb2)
                )
                checkpoint_status = AsyncMock()
                wrapper._checkpoint_status_if_any = checkpoint_status

                result = await wrapper.post_request("health_check", {})
                status = await wrapper.get_sleep_status()

                self.assertEqual(result, {"status": "ok"})
                self.assertEqual(status["state"], "RUNNING")
                checkpoint_status.assert_not_awaited()
                store.compare_set.assert_not_called()

    async def test_config_disabled_rejects_before_store_or_backend(self):
        store = MagicMock()
        wrapper, _ = self._build_wrapper(
            lifecycle_store=store,
            sleep_enabled=False,
            configured_level=2,
        )
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock()
        wrapper._dp_stubs[address].SleepServing = AsyncMock()

        result = await wrapper.sleep_serving({})

        self.assertEqual(result["grpc_status"], "UNIMPLEMENTED")
        store.compare_set.assert_not_called()
        wrapper._dp_stubs[address].GetSleepStatus.assert_not_awaited()
        wrapper._dp_stubs[address].SleepServing.assert_not_awaited()

    async def test_invalid_request_precedes_identity_lease_manifest_and_backend(self):
        store = MagicMock()
        controller = _FakeCheckpointController()
        wrapper, _ = self._build_wrapper(
            lifecycle_store=store,
            checkpoint_controller=controller,
            configured_level=3,
            single_node=True,
        )
        address = wrapper.control_addresses[0]
        wrapper._resolve_instance_identity = AsyncMock()
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock()
        wrapper._dp_stubs[address].SleepServing = AsyncMock()

        result = await wrapper.sleep_serving({"level": "invalid"})

        self.assertEqual(result["grpc_status"], "INVALID_ARGUMENT")
        wrapper._resolve_instance_identity.assert_not_awaited()
        store.compare_set.assert_not_called()
        self.assertEqual(controller.events, [])
        wrapper._dp_stubs[address].GetSleepStatus.assert_not_awaited()
        wrapper._dp_stubs[address].SleepServing.assert_not_awaited()

    async def test_configured_level_two_is_used_when_level_is_omitted(self):
        wrapper, pb2 = self._build_wrapper(configured_level=2)
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].SleepServing = AsyncMock(return_value=pb2.EmptyPB())
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
            side_effect=[
                self._status_pb(pb2, supported_levels=[2]),
                self._status_pb(
                    pb2,
                    state="SLEEPING",
                    sleep_epoch=1,
                    supported_levels=[2],
                    kv_memory_state="PAUSED",
                    device_kv_cache_valid=False,
                    gpu_resource_state="RELEASED",
                ),
            ]
        )

        result = await wrapper.sleep_serving({})

        self.assertEqual(result, {"status": "ok"})
        prepare_request = (
            wrapper._dp_stubs[address].SleepServing.await_args_list[0].args[0]
        )
        self.assertEqual(prepare_request.level, 2)
        self.assertIsNone(wrapper._checkpoint_controller)

    async def test_l1_failed_prepare_rollback_fences_and_retains_lease(self):
        store = _FakeStore()
        wrapper, pb2 = self._build_wrapper(lifecycle_store=store)
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].SleepServing = AsyncMock(
            side_effect=self._aio_error(
                grpc.StatusCode.FAILED_PRECONDITION, "prepare failed"
            )
        )
        wrapper._dp_stubs[address].WakeUpServing = AsyncMock(return_value=pb2.EmptyPB())
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
            side_effect=[self._status_pb(pb2)]
            + [self._status_pb(pb2, state="DRAINING")] * 3
        )

        result = await wrapper.sleep_serving({})

        self.assertTrue(result["recovery_required"])
        self.assertTrue(store.values[wrapper.LIFECYCLE_RECOVERY_KEY])
        self.assertTrue(store.values[wrapper.LIFECYCLE_LEASE_KEY])

    async def test_l1_uncertain_commit_fences_and_retains_lease(self):
        store = _FakeStore()
        wrapper, pb2 = self._build_wrapper(lifecycle_store=store)
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].SleepServing = AsyncMock(return_value=pb2.EmptyPB())
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
            side_effect=[self._status_pb(pb2), self._status_pb(pb2, state="ERROR")]
        )

        result = await wrapper.sleep_serving({})

        self.assertTrue(result["recovery_required"])
        self.assertTrue(store.values[wrapper.LIFECYCLE_RECOVERY_KEY])
        self.assertTrue(store.values[wrapper.LIFECYCLE_LEASE_KEY])

    async def test_l1_failed_wake_prepare_fences_and_retains_lease(self):
        store = _FakeStore()
        wrapper, pb2 = self._build_wrapper(lifecycle_store=store)
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
            return_value=self._status_pb(pb2, state="SLEEPING", sleep_epoch=1)
        )
        wrapper._dp_stubs[address].WakeUpServing = AsyncMock(
            side_effect=self._aio_error(
                grpc.StatusCode.FAILED_PRECONDITION, "wake prepare failed"
            )
        )

        result = await wrapper.wake_up_serving()

        self.assertTrue(result["recovery_required"])
        self.assertTrue(store.values[wrapper.LIFECYCLE_RECOVERY_KEY])
        self.assertTrue(store.values[wrapper.LIFECYCLE_LEASE_KEY])

    async def test_rank_identity_and_epoch_must_match_exactly(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, _ = self._build_wrapper(control_addresses=addresses)
        common = {
            "state": "RUNNING",
            "sleep_epoch": 4,
            "sleep_mode_enabled": True,
            "effective": True,
            "gpu_resource_state": "ACTIVE",
            "kv_memory_state": "ACTIVE",
            "supported_levels": [1],
            "supported_modes": ["wait", "abort"],
            "role": "PREFILL",
        }
        duplicate_rank = [
            {
                **common,
                "address": address,
                "world_rank": 0,
                "process_id": 2001 + rank,
                "instance_generation_uuid": f"generation-{rank}",
            }
            for rank, address in enumerate(addresses)
        ]
        epoch_mismatch = [
            {
                **common,
                "address": address,
                "world_rank": rank,
                "process_id": 2001 + rank,
                "instance_generation_uuid": f"generation-{rank}",
                "sleep_epoch": 4 + rank,
            }
            for rank, address in enumerate(addresses)
        ]

        rank_result = wrapper._aggregate_sleep_status(duplicate_rank)
        epoch_result = wrapper._aggregate_sleep_status(epoch_mismatch)

        self.assertEqual(rank_result["grpc_status"], "FAILED_PRECONDITION")
        self.assertIn("world ranks", rank_result["error"])
        self.assertEqual(epoch_result["grpc_status"], "FAILED_PRECONDITION")
        self.assertIn("did not converge", epoch_result["error"])

    def _status_pb(self, pb2, **kwargs):
        from rtp_llm.utils.grpc_client_wrapper import _local_process_identity

        identity = _local_process_identity()
        defaults = {
            "state": "RUNNING",
            "sleep_mode_enabled": True,
            "effective": True,
            "supported_levels": [1],
            "supported_modes": ["wait", "abort"],
            "kv_memory_state": "ACTIVE",
            "device_kv_cache_valid": True,
            "gpu_resource_state": "ACTIVE",
            "process_id": 1001,
            "process_starttime": identity["starttime"],
            "process_pid_namespace": identity["pid_namespace"],
            "process_boot_id": identity["boot_id"],
        }
        defaults.update(kwargs)
        return pb2.SleepStatusResponsePB(**defaults)

    def _configure_level3_backend(self, wrapper, pb2, events=None):
        from rtp_llm.utils.grpc_client_wrapper import _local_process_identity

        identity = _local_process_identity()
        events = events if events is not None else []
        rank_statuses = {}
        for rank, address in enumerate(wrapper.control_addresses):
            rank_statuses[address] = {
                "state": "RUNNING",
                "sleep_epoch": 0,
                "kv_memory_state": "ACTIVE",
                "device_kv_cache_valid": True,
                "gpu_resource_state": "ACTIVE",
                "supported_levels": [3],
                "process_id": 2001 + rank,
                "process_starttime": identity["starttime"] + rank,
                "process_pid_namespace": identity["pid_namespace"],
                "process_boot_id": identity["boot_id"],
            }

            async def get_status(*args, address=address, **kwargs):
                events.append(("backend_status", address))
                return self._status_pb(pb2, **rank_statuses[address])

            async def sleep_rpc(request, *args, address=address, **kwargs):
                if request.prepare_only:
                    events.append(("sleep_prepare", address))
                    rank_statuses[address].update(state="DRAINING", sleep_epoch=1)
                elif request.commit_only:
                    events.append(("sleep_commit", address))
                    rank_statuses[address].update(
                        state="SLEEPING",
                        kv_memory_state="PAUSED",
                        device_kv_cache_valid=False,
                        gpu_resource_state="RELEASED",
                    )
                return pb2.EmptyPB()

            async def wake_rpc(request, *args, address=address, **kwargs):
                if request.prepare_only:
                    events.append(("wake_prepare", address))
                    rank_statuses[address].update(
                        state="WAKING_UP", gpu_resource_state="RESTORING"
                    )
                elif request.commit_only:
                    events.append(("wake_commit", address))
                    rank_statuses[address].update(
                        state="RUNNING",
                        kv_memory_state="ACTIVE",
                        device_kv_cache_valid=True,
                        gpu_resource_state="ACTIVE",
                    )
                return pb2.EmptyPB()

            wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
                side_effect=get_status
            )
            wrapper._dp_stubs[address].SleepServing = AsyncMock(side_effect=sleep_rpc)
            wrapper._dp_stubs[address].WakeUpServing = AsyncMock(side_effect=wake_rpc)
        return rank_statuses

    def _configure_distributed_level3_backend(self, wrapper, pb2, events=None):
        events = events if events is not None else []
        rank_statuses = {}
        driver_states = {}
        transactions = {}
        for rank, address in enumerate(wrapper.control_addresses):
            rank_statuses[address] = {
                "state": "RUNNING",
                "sleep_epoch": 0,
                "kv_memory_state": "ACTIVE",
                "device_kv_cache_valid": True,
                "gpu_resource_state": "ACTIVE",
                "supported_levels": [3],
                # PIDs may repeat across hosts; the node identity disambiguates.
                "process_id": 2001,
                "process_starttime": 9001 + rank,
                "process_pid_namespace": 101 + rank,
                "process_boot_id": f"boot-node-{rank}",
                "world_rank": rank,
                "role": "RoleType.PREFILL",
                "instance_generation_uuid": f"generation-{rank}",
                "holder_instance": f"keeper-node-{rank}",
            }
            driver_states[address] = "RUNNING"
            transactions[address] = ("", -1)

            async def get_status(*args, address=address, **kwargs):
                events.append(("backend_status", address))
                return self._status_pb(pb2, **rank_statuses[address])

            async def sleep_rpc(request, *args, address=address, **kwargs):
                if request.prepare_only:
                    events.append(("sleep_prepare", address))
                    rank_statuses[address].update(state="DRAINING", sleep_epoch=1)
                elif request.commit_only:
                    events.append(("sleep_commit", address))
                    rank_statuses[address].update(
                        state="SLEEPING",
                        kv_memory_state="PAUSED",
                        device_kv_cache_valid=False,
                        gpu_resource_state="RELEASED",
                    )
                return pb2.EmptyPB()

            async def wake_rpc(request, *args, address=address, **kwargs):
                if request.prepare_only:
                    events.append(("wake_prepare", address))
                    rank_statuses[address].update(
                        state="WAKING_UP", gpu_resource_state="RESTORING"
                    )
                elif request.commit_only:
                    events.append(("wake_commit", address))
                    rank_statuses[address].update(
                        state="RUNNING",
                        kv_memory_state="ACTIVE",
                        device_kv_cache_valid=True,
                        gpu_resource_state="ACTIVE",
                    )
                return pb2.EmptyPB()

            async def checkpoint_rpc(request, *args, address=address, **kwargs):
                action = request.action
                events.append(("cuda_checkpoint", action, address))
                state = driver_states[address]
                owner = transactions[address]
                success = True
                error = ""
                if action == "LOCK":
                    if state == "RUNNING":
                        transactions[address] = (
                            request.transaction_id,
                            request.sleep_epoch,
                        )
                        driver_states[address] = "LOCKED"
                    elif state != "LOCKED" or owner != (
                        request.transaction_id,
                        request.sleep_epoch,
                    ):
                        success = False
                        error = "bad LOCK state"
                elif action == "CHECKPOINT":
                    if state == "LOCKED" and owner == (
                        request.transaction_id,
                        request.sleep_epoch,
                    ):
                        driver_states[address] = "CHECKPOINTED"
                    elif state != "CHECKPOINTED":
                        success = False
                        error = "bad CHECKPOINT state"
                elif action == "RESTORE":
                    if state == "CHECKPOINTED" and owner == (
                        request.transaction_id,
                        request.sleep_epoch,
                    ):
                        driver_states[address] = "LOCKED"
                    elif state != "LOCKED":
                        success = False
                        error = "bad RESTORE state"
                elif action == "UNLOCK":
                    if state == "LOCKED" and owner == (
                        request.transaction_id,
                        request.sleep_epoch,
                    ):
                        driver_states[address] = "RUNNING"
                    elif state != "RUNNING":
                        success = False
                        error = "bad UNLOCK state"
                elif action != "GET_STATE":
                    success = False
                    error = "bad action"
                identity = rank_statuses[address]
                return pb2.CudaCheckpointResponsePB(
                    success=success,
                    cuda_result=0 if success else 1,
                    state=driver_states[address],
                    error=error,
                    transaction_id=transactions[address][0],
                    sleep_epoch=transactions[address][1],
                    process_id=identity["process_id"],
                    process_starttime=identity["process_starttime"],
                    process_pid_namespace=identity["process_pid_namespace"],
                    process_boot_id=identity["process_boot_id"],
                    world_rank=identity["world_rank"],
                    holder_instance=identity["holder_instance"],
                )

            wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
                side_effect=get_status
            )
            wrapper._dp_stubs[address].SleepServing = AsyncMock(side_effect=sleep_rpc)
            wrapper._dp_stubs[address].WakeUpServing = AsyncMock(side_effect=wake_rpc)
            wrapper._dp_stubs[address].CudaCheckpointProcess = AsyncMock(
                side_effect=checkpoint_rpc
            )
        return rank_statuses, driver_states

    async def test_health_check_failure_preserves_lifecycle_channels(self):
        # Regression: a routine health probe timing out during a sleep/wake
        # drain must NOT tear down the shared lifecycle _dp_channels. Closing a
        # channel under a genuinely in-flight SleepServing/WakeUpServing call
        # raises asyncio.CancelledError into that RPC (a BaseException that
        # bypasses every ``except Exception``), cancelling the operation and
        # returning HTTP 500 while the backend keeps transitioning -- a
        # control-plane split brain. health_check may only reset its own
        # channel.
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, _ = self._build_wrapper(control_addresses=addresses)
        wrapper.channel = MagicMock()
        wrapper.channel.close = AsyncMock()
        wrapper.stub = MagicMock()
        wrapper.stub.CheckHealth = AsyncMock(
            side_effect=self._aio_error(
                grpc.StatusCode.DEADLINE_EXCEEDED, "backend draining"
            )
        )
        dp_channels_before = dict(wrapper._dp_channels)
        dp_stubs_before = dict(wrapper._dp_stubs)

        result = await wrapper.health_check()

        self.assertEqual(result["status"], "error")
        # Only the health channel is reset; lifecycle channels stay intact.
        self.assertIsNone(wrapper.channel)
        self.assertIsNone(wrapper.stub)
        self.assertEqual(wrapper._dp_channels, dp_channels_before)
        self.assertEqual(wrapper._dp_stubs, dp_stubs_before)
        for address in addresses:
            self.assertFalse(wrapper._dp_channels[address].close.called)

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

    async def test_sleep_commit_cancellation_is_absorbed_and_reaches_sleeping(self):
        # Regression: once commit starts the device-memory release is
        # irreversible. If the driving request is cancelled mid-commit (a stray
        # channel teardown, a client disconnect, a worker recycle) we must NOT
        # abandon the transition half-committed -- that leaves the instance with
        # part of its GPU memory freed and no owner. The commit must be driven
        # uninterruptibly to the terminal SLEEPING state and still report ok.
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
        commit_entered = asyncio.Event()
        release_commit = asyncio.Event()

        for address in addresses:

            async def get_status(*args, address=address, **kwargs):
                return self._status_pb(pb2, **rank_statuses[address])

            async def sleep_rpc(request, *args, address=address, **kwargs):
                if request.prepare_only:
                    rank_statuses[address].update(state="DRAINING", sleep_epoch=1)
                elif request.commit_only:
                    # Block inside the irreversible commit so the test can cancel
                    # the driving task while the transition is in flight.
                    commit_entered.set()
                    await release_commit.wait()
                    rank_statuses[address].update(
                        state="SLEEPING",
                        kv_memory_state="PAUSED",
                        device_kv_cache_valid=False,
                        gpu_resource_state="RELEASED",
                    )
                return pb2.EmptyPB()

            wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
                side_effect=get_status
            )
            wrapper._dp_stubs[address].SleepServing = AsyncMock(side_effect=sleep_rpc)

        task = asyncio.ensure_future(
            wrapper.sleep_serving(
                {"level": 1, "mode": "wait", "timeout_ms": 1000, "reason": "cancel"}
            )
        )
        await asyncio.wait_for(commit_entered.wait(), timeout=5)
        # Cancel while the irreversible commit is in flight, then let it finish.
        task.cancel()
        await asyncio.sleep(0)
        release_commit.set()

        result = await asyncio.wait_for(task, timeout=5)
        self.assertEqual(result, {"status": "ok"})
        self.assertFalse(task.cancelled())
        sleeping_status = await wrapper.get_sleep_status()
        self.assertEqual(sleeping_status["state"], "SLEEPING")
        self.assertEqual(sleeping_status["gpu_resource_state"], "RELEASED")

    async def test_sleep_prepare_cancellation_rolls_back_to_running(self):
        # Regression: prepare only closes admission and drains -- no device
        # memory is freed, so it is reversible. A cancellation here must roll the
        # drain back to RUNNING (via a WakeUpServing abort) so the instance keeps
        # serving, then honor the cancellation.
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
        prepare_entered = asyncio.Event()
        release_prepare = asyncio.Event()
        wake_calls = {"n": 0}

        for address in addresses:

            async def get_status(*args, address=address, **kwargs):
                return self._status_pb(pb2, **rank_statuses[address])

            async def sleep_rpc(request, *args, address=address, **kwargs):
                if request.prepare_only:
                    rank_statuses[address].update(state="DRAINING", sleep_epoch=1)
                    # Block mid-drain so the test can cancel before commit.
                    prepare_entered.set()
                    await release_prepare.wait()
                return pb2.EmptyPB()

            async def wake_rpc(request, *args, address=address, **kwargs):
                wake_calls["n"] += 1
                rank_statuses[address].update(
                    state="RUNNING", gpu_resource_state="ACTIVE"
                )
                return pb2.EmptyPB()

            wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
                side_effect=get_status
            )
            wrapper._dp_stubs[address].SleepServing = AsyncMock(side_effect=sleep_rpc)
            wrapper._dp_stubs[address].WakeUpServing = AsyncMock(side_effect=wake_rpc)

        task = asyncio.ensure_future(
            wrapper.sleep_serving(
                {"level": 1, "mode": "wait", "timeout_ms": 1000, "reason": "cancel"}
            )
        )
        await asyncio.wait_for(prepare_entered.wait(), timeout=5)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        self.assertTrue(task.cancelled())
        # The reversible drain was rolled back on every control rank.
        self.assertEqual(wake_calls["n"], len(addresses))

    async def _run_failed_level3_prepare_cancellation(
        self, failure_mode: str
    ) -> tuple[Any, _FakeStore]:
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        store = _FakeStore()
        controller = _FakeCheckpointController()
        wrapper, pb2 = self._build_wrapper(
            control_addresses=addresses,
            expected_control_address_count=len(addresses),
            lifecycle_store=store,
            checkpoint_controller=controller,
            single_node=True,
        )
        rank_statuses = self._configure_level3_backend(wrapper, pb2)
        prepare_entered = asyncio.Event()
        release_prepare = asyncio.Event()
        entered = {"count": 0}

        for address in addresses:

            async def sleep_rpc(request, *args, address=address, **kwargs):
                if request.prepare_only:
                    rank_statuses[address].update(state="DRAINING", sleep_epoch=1)
                    entered["count"] += 1
                    if entered["count"] == len(addresses):
                        prepare_entered.set()
                    await release_prepare.wait()
                return pb2.EmptyPB()

            async def wake_rpc(request, *args, address=address, **kwargs):
                if address == addresses[1]:
                    if failure_mode == "rpc_failure":
                        raise self._aio_error(
                            grpc.StatusCode.FAILED_PRECONDITION,
                            "restartEngine failed",
                        )
                    if failure_mode == "timeout":
                        # Even if the follow-up status happens to report RUNNING,
                        # the timed-out rollback RPC is an uncertain transaction.
                        rank_statuses[address].update(state="RUNNING")
                        raise self._aio_error(
                            grpc.StatusCode.DEADLINE_EXCEEDED,
                            "rollback timed out",
                        )
                    if failure_mode == "cancelled_rpc":
                        raise asyncio.CancelledError()
                    if failure_mode == "mixed_state":
                        return pb2.EmptyPB()
                rank_statuses[address].update(state="RUNNING")
                return pb2.EmptyPB()

            wrapper._dp_stubs[address].SleepServing = AsyncMock(side_effect=sleep_rpc)
            wrapper._dp_stubs[address].WakeUpServing = AsyncMock(side_effect=wake_rpc)

        task = asyncio.create_task(
            wrapper.sleep_serving({"level": 3, "mode": "wait", "timeout_ms": 1000})
        )
        await asyncio.wait_for(prepare_entered.wait(), timeout=5)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

        self.assertNotEqual(store.values[wrapper.LIFECYCLE_LEASE_KEY], "")
        self.assertNotEqual(store.values[wrapper.LIFECYCLE_RECOVERY_KEY], "")
        status = await wrapper.get_sleep_status()
        self.assertEqual(status["state"], "RECOVERY_REQUIRED")
        self.assertTrue(status["recovery_required"])
        return wrapper, store

    async def test_level3_prepare_cancellation_rollback_failure_is_persistent(self):
        wrapper, store = await self._run_failed_level3_prepare_cancellation(
            "rpc_failure"
        )

        competitor, _ = self._build_wrapper(
            control_addresses=wrapper.control_addresses,
            lifecycle_store=store,
            checkpoint_controller=_FakeCheckpointController(),
            single_node=True,
        )
        result = await competitor.sleep_serving({"level": 3})

        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertTrue(result["recovery_required"])
        self.assertIn("RECOVERY_REQUIRED", result["error"])
        for address in competitor.control_addresses:
            competitor._dp_stubs[address].SleepServing.assert_not_called()

    async def test_level3_prepare_cancellation_rollback_timeout_is_persistent(self):
        wrapper, _ = await self._run_failed_level3_prepare_cancellation("timeout")

        self.assertIn("rollback RPC failed", wrapper._frontend_lifecycle_error)
        for address in wrapper.control_addresses:
            self.assertEqual(
                wrapper._dp_stubs[address].WakeUpServing.await_count,
                1,
            )

    async def test_level3_prepare_cancellation_mixed_state_is_persistent(self):
        wrapper, _ = await self._run_failed_level3_prepare_cancellation("mixed_state")

        self.assertIn("did not converge to RUNNING", wrapper._frontend_lifecycle_error)
        self.assertEqual(
            wrapper._dp_stubs[wrapper.control_addresses[1]].WakeUpServing.await_count,
            1,
        )

    async def test_level3_prepare_cancellation_rollback_rpc_cancel_is_persistent(self):
        wrapper, _ = await self._run_failed_level3_prepare_cancellation("cancelled_rpc")

        self.assertIn("rollback RPC failed", wrapper._frontend_lifecycle_error)

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
            # Commit re-drains transfers that acquired a lease at the gate-close
            # boundary, so it retains the caller's drain timeout.
            self.assertEqual(commit_request.timeout_ms, 1000)

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

    async def test_wake_up_from_uniform_sleeping_state_proceeds(self):
        # #1 contract (positive): when every control rank reports the SAME
        # SLEEPING state, wake is a well-defined atomic transition and must
        # proceed through prepare + commit to RUNNING.
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        for address in addresses:
            wrapper._dp_stubs[address].WakeUpServing = AsyncMock(
                return_value=pb2.EmptyPB()
            )
            wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
                side_effect=[
                    self._status_pb(
                        pb2,
                        state="SLEEPING",
                        sleep_epoch=1,
                        kv_memory_state="PAUSED",
                        device_kv_cache_valid=False,
                        gpu_resource_state="RELEASED",
                    ),
                    self._status_pb(
                        pb2,
                        state="RUNNING",
                        sleep_epoch=1,
                        kv_memory_state="ACTIVE",
                        device_kv_cache_valid=True,
                        gpu_resource_state="ACTIVE",
                    ),
                ]
            )

        result = await wrapper.wake_up_serving()

        self.assertEqual(result, {"status": "ok"})
        for address in addresses:
            self.assertEqual(wrapper._dp_stubs[address].WakeUpServing.await_count, 2)

    async def test_wake_up_rejects_mixed_initial_rank_state_as_recovery_required(self):
        # #1 contract (negative): a mixed PRE-condition -- one rank already
        # SLEEPING while another is still DRAINING -- is a fault, not a
        # recoverable divergence. sleep/wake are atomic and level-2 discarded GPU
        # memory with no backup, so the ranks cannot be reconciled into a known-
        # good state. wake_up must return RECOVERY_REQUIRED *before* issuing any
        # prepare RPC (never hang, never silently half-wake); the operator
        # restarts the instance.
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        wrapper._dp_stubs[addresses[0]].GetSleepStatus = AsyncMock(
            return_value=self._status_pb(pb2, state="SLEEPING", sleep_epoch=1)
        )
        wrapper._dp_stubs[addresses[1]].GetSleepStatus = AsyncMock(
            return_value=self._status_pb(pb2, state="DRAINING", sleep_epoch=1)
        )
        for address in addresses:
            wrapper._dp_stubs[address].WakeUpServing = AsyncMock(
                return_value=pb2.EmptyPB()
            )

        result = await wrapper.wake_up_serving()

        self.assertIn("error", result)
        self.assertIn("RECOVERY_REQUIRED", result["error"])
        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertTrue(result["recovery_required"])
        self.assertEqual(
            {detail["address"] for detail in result["details"]}, set(addresses)
        )
        for address in addresses:
            wrapper._dp_stubs[address].WakeUpServing.assert_not_awaited()

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
            "process_id",
            "process_starttime",
            "process_pid_namespace",
            "process_boot_id",
            "world_rank",
            "role",
            "instance_generation_uuid",
            "holder_instance",
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
        self.assertTrue(result["recovery_required"])
        self.assertEqual(
            {detail["address"] for detail in result["details"]}, set(addresses)
        )
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

    async def test_independent_wrappers_compete_for_instance_lease(self):
        store = _FakeStore()
        holder, _ = self._build_wrapper(lifecycle_store=store)
        loser, _ = self._build_wrapper(lifecycle_store=store)
        address = loser.control_addresses[0]
        loser._dp_stubs[address].GetSleepStatus = AsyncMock()
        loser._dp_stubs[address].SleepServing = AsyncMock()

        record, error = holder._acquire_lifecycle_lease("sleep")
        self.assertFalse(error)
        result = await loser.sleep_serving({})

        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertIn("holds the instance lease", result["error"])
        loser._dp_stubs[address].GetSleepStatus.assert_not_awaited()
        loser._dp_stubs[address].SleepServing.assert_not_awaited()
        holder._release_lifecycle_lease(record)

    async def test_required_instance_store_unavailable_fails_closed(self):
        wrapper, _ = self._build_wrapper()
        wrapper._require_instance_lease = True
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock()
        wrapper._dp_stubs[address].SleepServing = AsyncMock()

        result = await wrapper.sleep_serving({})

        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertIn("coordination is unavailable", result["error"])
        wrapper._dp_stubs[address].GetSleepStatus.assert_not_awaited()
        wrapper._dp_stubs[address].SleepServing.assert_not_awaited()

    async def test_instance_lease_release_requires_exact_owner_record(self):
        store = _FakeStore()
        holder, _ = self._build_wrapper(lifecycle_store=store)
        other, _ = self._build_wrapper(lifecycle_store=store)

        record, error = holder._acquire_lifecycle_lease("sleep")
        self.assertFalse(error)
        other._release_lifecycle_lease(other._lease_record("sleep"))
        self.assertEqual(
            store.values[holder.LIFECYCLE_LEASE_KEY],
            record,
        )

        holder._release_lifecycle_lease(record)
        self.assertEqual(store.values[holder.LIFECYCLE_LEASE_KEY], "")

    async def test_dead_frontend_lease_is_reclaimed_for_manifest_recovery(self):
        from rtp_llm.utils.grpc_client_wrapper import _local_process_identity

        store = _FakeStore()
        wrapper, _ = self._build_wrapper(lifecycle_store=store)
        identity = _local_process_identity()
        stale_record = json.dumps(
            {
                "holder": "dead-holder",
                "operation": "sleep",
                "pid": 2**30,
                "starttime": 1,
                "pid_namespace": identity["pid_namespace"],
                "boot_id": identity["boot_id"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        store.values[wrapper.LIFECYCLE_LEASE_KEY] = stale_record

        record, error = wrapper._acquire_lifecycle_lease("wake_up")

        self.assertFalse(error)
        self.assertEqual(store.values[wrapper.LIFECYCLE_LEASE_KEY], record)
        self.assertNotEqual(record, stale_record)
        wrapper._release_lifecycle_lease(record)

    async def test_partial_sleep_commit_retries_only_draining_rank(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        for address in addresses:
            wrapper._dp_stubs[address].SleepServing = AsyncMock(
                return_value=pb2.EmptyPB()
            )
        wrapper._dp_stubs[addresses[0]].GetSleepStatus = AsyncMock(
            side_effect=[
                self._status_pb(pb2, state="RUNNING"),
                self._status_pb(pb2, state="SLEEPING"),
                self._status_pb(pb2, state="SLEEPING"),
            ]
        )
        wrapper._dp_stubs[addresses[1]].GetSleepStatus = AsyncMock(
            side_effect=[
                self._status_pb(pb2, state="RUNNING"),
                self._status_pb(pb2, state="DRAINING"),
                self._status_pb(pb2, state="SLEEPING"),
            ]
        )

        result = await wrapper.sleep_serving({})

        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(wrapper._dp_stubs[addresses[0]].SleepServing.await_count, 2)
        self.assertEqual(wrapper._dp_stubs[addresses[1]].SleepServing.await_count, 3)

    async def test_partial_wake_commit_retries_only_waking_rank(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        for address in addresses:
            wrapper._dp_stubs[address].WakeUpServing = AsyncMock(
                return_value=pb2.EmptyPB()
            )
        wrapper._dp_stubs[addresses[0]].GetSleepStatus = AsyncMock(
            side_effect=[
                self._status_pb(pb2, state="SLEEPING"),
                self._status_pb(pb2, state="RUNNING"),
                self._status_pb(pb2, state="RUNNING"),
            ]
        )
        wrapper._dp_stubs[addresses[1]].GetSleepStatus = AsyncMock(
            side_effect=[
                self._status_pb(pb2, state="SLEEPING"),
                self._status_pb(pb2, state="WAKING_UP"),
                self._status_pb(pb2, state="RUNNING"),
            ]
        )

        result = await wrapper.wake_up_serving()

        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(wrapper._dp_stubs[addresses[0]].WakeUpServing.await_count, 2)
        self.assertEqual(wrapper._dp_stubs[addresses[1]].WakeUpServing.await_count, 3)

    async def test_commit_error_or_unreachable_requires_recovery(self):
        for terminal in ("ERROR", "UNREACHABLE"):
            with self.subTest(terminal=terminal):
                wrapper, pb2 = self._build_wrapper()
                address = wrapper.control_addresses[0]
                wrapper._dp_stubs[address].SleepServing = AsyncMock(
                    return_value=pb2.EmptyPB()
                )
                terminal_status = (
                    self._status_pb(pb2, state="ERROR")
                    if terminal == "ERROR"
                    else self._aio_error(
                        grpc.StatusCode.UNAVAILABLE, "rank unavailable"
                    )
                )
                wrapper._dp_stubs[address].GetSleepStatus = AsyncMock(
                    side_effect=[
                        self._status_pb(pb2, state="RUNNING"),
                        terminal_status,
                    ]
                )

                result = await wrapper.sleep_serving({})

                self.assertNotEqual(result.get("status"), "ok")
                self.assertTrue(result["recovery_required"])
                self.assertIn("RECOVERY_REQUIRED", result["error"])

    def test_checkpoint_adapter_maps_real_controller_api_and_status(self):
        try:
            from rtp_llm.utils import checkpoint_controller as controller_api
        except ImportError:
            self.skipTest(
                "checkpoint controller is implemented in integration workspace"
            )

        from rtp_llm.utils.grpc_client_wrapper import _CheckpointControllerAdapter

        processes = (
            controller_api.ProcessRecoveryStatus(
                pid=2001,
                starttime=11,
                rank=7,
                address="127.0.0.1:10001",
                state=controller_api.ProcessState.CHECKPOINTED,
                identity_valid=True,
                driver_state="CHECKPOINTED",
                error=None,
            ),
            controller_api.ProcessRecoveryStatus(
                pid=2002,
                starttime=12,
                rank=1,
                address="127.0.0.1:10009",
                state=controller_api.ProcessState.CHECKPOINTED,
                identity_valid=True,
                driver_state="CHECKPOINTED",
                error=None,
            ),
        )
        checkpoint_status = controller_api.RecoveryStatus(
            epoch="9",
            phase="CHECKPOINTED",
            manifest_exists=True,
            recovery_required=False,
            checkpoint_complete=True,
            restore_complete=False,
            processes=processes,
            last_error=None,
        )
        manifest_path = MagicMock(return_value="/tmp/manifest")
        checkpoint_all = MagicMock(return_value=checkpoint_status)
        restore_all = MagicMock()
        recovery_status = MagicMock(return_value=checkpoint_status)
        adapter = _CheckpointControllerAdapter()
        adapter._module = MagicMock(return_value=controller_api)
        addresses = ("127.0.0.1:10001", "127.0.0.1:10009")
        statuses = (
            {
                "address": addresses[0],
                "state": "SLEEPING",
                "process_id": 2001,
                "sleep_epoch": "9",
                "rank": 7,
                "process_starttime": 11,
                "process_pid_namespace": 88,
                "process_boot_id": "boot-test",
            },
            {
                "address": addresses[1],
                "state": "SLEEPING",
                "process_id": 2002,
                "sleep_epoch": 9,
                "process_starttime": 12,
                "process_pid_namespace": 88,
                "process_boot_id": "boot-test",
            },
        )

        with (
            patch.object(controller_api, "checkpoint_manifest_path", manifest_path),
            patch.object(controller_api, "checkpoint_all", checkpoint_all),
            patch.object(controller_api, "restore_all", restore_all),
            patch.object(controller_api, "recovery_status", recovery_status),
            patch.object(
                controller_api,
                "read_process_starttime",
                side_effect=lambda pid: {2001: 11, 2002: 12}[pid],
            ),
            patch(
                "rtp_llm.utils.grpc_client_wrapper._local_process_identity",
                return_value={
                    "pid": os.getpid(),
                    "starttime": 99,
                    "pid_namespace": 88,
                    "boot_id": "boot-test",
                },
            ),
            patch(
                "rtp_llm.utils.grpc_client_wrapper.os.path.exists",
                return_value=True,
            ),
        ):
            adapter.preflight(addresses)
            result = adapter.checkpoint_all(addresses, statuses)
            manifest = adapter.read_manifest(addresses)

        self.assertEqual(result["state"], "CHECKPOINTED")
        self.assertEqual(manifest["pids"], [2001, 2002])
        manifest_path.assert_called_with(addresses, namespace=None)
        path, targets, epoch = checkpoint_all.call_args.args
        self.assertEqual(path, "/tmp/manifest")
        self.assertEqual(epoch, 9)
        self.assertTrue(
            all(
                isinstance(target, controller_api.CheckpointTarget)
                for target in targets
            )
        )
        self.assertEqual(
            [
                (
                    target.pid,
                    target.rank,
                    target.address,
                    target.expected_starttime,
                )
                for target in targets
            ],
            [
                (2001, 7, addresses[0], 11),
                (2002, 1, addresses[1], 12),
            ],
        )
        self.assertEqual(recovery_status.call_count, 2)
        recovery_status.assert_called_with("/tmp/manifest")

    def test_checkpoint_adapter_skips_driver_when_manifest_is_absent(self):
        from rtp_llm.utils.grpc_client_wrapper import _CheckpointControllerAdapter

        module = SimpleNamespace(
            CheckpointTarget=MagicMock(),
            checkpoint_manifest_path=MagicMock(return_value="/tmp/missing-manifest"),
            checkpoint_all=MagicMock(),
            read_process_starttime=MagicMock(),
            restore_all=MagicMock(),
            recovery_status=MagicMock(side_effect=RuntimeError("driver unavailable")),
        )
        adapter = _CheckpointControllerAdapter()
        adapter._module = MagicMock(return_value=module)

        with patch(
            "rtp_llm.utils.grpc_client_wrapper.os.path.exists", return_value=False
        ):
            manifest = adapter.read_manifest(("127.0.0.1:10001",))

        self.assertIsNone(manifest)
        module.recovery_status.assert_not_called()

    def test_manifest_is_stale_detects_dead_backend_processes(self):
        from rtp_llm.utils.checkpoint_controller import read_process_starttime
        from rtp_llm.utils.grpc_client_wrapper import GrpcClientWrapper

        # No recorded process is live -> stale (prior generation reused addrs).
        self.assertTrue(
            GrpcClientWrapper._manifest_is_stale(
                {
                    "state": "CHECKPOINTED",
                    "processes": [{"pid": 2147480000, "starttime": 1}],
                }
            )
        )
        # This very test process is alive (matching starttime) -> not stale.
        live = {"pid": os.getpid(), "starttime": read_process_starttime(os.getpid())}
        self.assertFalse(
            GrpcClientWrapper._manifest_is_stale(
                {"state": "CHECKPOINTED", "processes": [live]}
            )
        )
        # A manifest without recorded processes is never inferred stale.
        self.assertFalse(
            GrpcClientWrapper._manifest_is_stale({"state": "CHECKPOINTED"})
        )
        self.assertFalse(GrpcClientWrapper._manifest_is_stale(None))

    async def test_stale_checkpoint_manifest_is_discarded_at_startup(self):
        controller = _FakeCheckpointController()
        controller.manifest = {
            "state": "CHECKPOINTED",
            "processes": [{"pid": 2147480000, "starttime": 1}],
            "pids": [2147480000],
        }
        wrapper, _pb2 = self._build_wrapper(checkpoint_controller=controller)

        status = await wrapper._checkpoint_status_if_any()

        # Stale manifest is ignored (falls through to real health check) and
        # discarded, so a fresh frontend is not wedged by a prior generation.
        self.assertIsNone(status)
        self.assertIn(
            ("clear_manifest", tuple(wrapper.control_addresses)), controller.events
        )
        self.assertIsNone(controller.manifest)

    def test_checkpoint_adapter_rejects_backend_process_identity_mismatch(self):
        from rtp_llm.utils.grpc_client_wrapper import _CheckpointControllerAdapter

        base_status = {
            "address": "127.0.0.1:10001",
            "process_id": 2001,
            "sleep_epoch": 3,
            "process_starttime": 11,
            "process_pid_namespace": 88,
            "process_boot_id": "boot-test",
        }
        local_identity = {
            "pid": os.getpid(),
            "starttime": 99,
            "pid_namespace": 88,
            "boot_id": "boot-test",
        }
        cases = (
            ({**base_status, "process_boot_id": "other-boot"}, 11, "same host"),
            (
                {**base_status, "process_pid_namespace": 89},
                11,
                "PID namespace",
            ),
            ({**base_status, "process_starttime": 12}, 11, "PID identity"),
        )

        for status, observed_starttime, message in cases:
            with self.subTest(message=message):
                module = SimpleNamespace(
                    CheckpointTarget=MagicMock(),
                    checkpoint_manifest_path=MagicMock(return_value="/tmp/manifest"),
                    checkpoint_all=MagicMock(),
                    read_process_starttime=MagicMock(return_value=observed_starttime),
                    restore_all=MagicMock(),
                    recovery_status=MagicMock(),
                )
                adapter = _CheckpointControllerAdapter()
                adapter._module = MagicMock(return_value=module)

                with (
                    patch(
                        "rtp_llm.utils.grpc_client_wrapper._local_process_identity",
                        return_value=local_identity,
                    ),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    adapter.checkpoint_all((base_status["address"],), (status,))

                module.checkpoint_all.assert_not_called()

    def test_checkpoint_adapter_preflight_loads_real_driver(self):
        from rtp_llm.utils.grpc_client_wrapper import _CheckpointControllerAdapter

        module = SimpleNamespace(
            CheckpointTarget=MagicMock(),
            checkpoint_manifest_path=MagicMock(return_value="/tmp/manifest"),
            checkpoint_all=MagicMock(),
            read_process_starttime=MagicMock(),
            restore_all=MagicMock(),
            recovery_status=MagicMock(side_effect=RuntimeError("missing CUDA symbol")),
        )
        adapter = _CheckpointControllerAdapter()
        adapter._module = MagicMock(return_value=module)

        with self.assertRaisesRegex(RuntimeError, "missing CUDA symbol"):
            adapter.preflight(("127.0.0.1:10001",))

        module.recovery_status.assert_called_once_with("/tmp/manifest")

    def test_checkpoint_adapter_normalizes_restore_and_rejects_bad_targets(self):
        try:
            from rtp_llm.utils import checkpoint_controller as controller_api
        except ImportError:
            self.skipTest(
                "checkpoint controller is implemented in integration workspace"
            )

        from rtp_llm.utils.grpc_client_wrapper import _CheckpointControllerAdapter

        process = controller_api.ProcessRecoveryStatus(
            pid=2001,
            starttime=11,
            rank=0,
            address="127.0.0.1:10001",
            state=controller_api.ProcessState.UNLOCKED,
            identity_valid=True,
            driver_state="RUNNING",
            error=None,
        )
        restored = controller_api.RecoveryStatus(
            epoch="3",
            phase="UNLOCKED",
            manifest_exists=False,
            recovery_required=False,
            checkpoint_complete=False,
            restore_complete=True,
            processes=(process,),
            last_error=None,
        )
        manifest_path = MagicMock(return_value="/tmp/manifest")
        checkpoint_all = MagicMock()
        restore_all = MagicMock(return_value=restored)
        recovery_status = MagicMock()
        adapter = _CheckpointControllerAdapter()
        adapter._module = MagicMock(return_value=controller_api)
        addresses = ("127.0.0.1:10001", "127.0.0.1:10009")
        base = [
            {
                "address": addresses[0],
                "process_id": 2001,
                "sleep_epoch": 3,
            },
            {
                "address": addresses[1],
                "process_id": 2002,
                "sleep_epoch": 3,
            },
        ]

        with (
            patch.object(controller_api, "checkpoint_manifest_path", manifest_path),
            patch.object(controller_api, "checkpoint_all", checkpoint_all),
            patch.object(controller_api, "restore_all", restore_all),
            patch.object(controller_api, "recovery_status", recovery_status),
        ):
            result = adapter.restore_all(addresses)
            self.assertTrue(result["restore_complete"])
            self.assertTrue(result["all_running"])
            self.assertEqual(result["state"], "RUNNING")

            for bad_statuses in (
                [base[0], {**base[1], "sleep_epoch": 4}],
                [base[0], {**base[1], "process_id": 2001}],
            ):
                with self.assertRaises(RuntimeError):
                    adapter.checkpoint_all(addresses, bad_statuses)
        checkpoint_all.assert_not_called()

    async def test_level3_rdma_sleep_checkpoints_only_after_all_ranks_sleeping(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        controller = _FakeCheckpointController()
        wrapper, pb2 = self._build_wrapper(
            control_addresses=addresses,
            checkpoint_controller=controller,
            single_node=True,
            rdma_enabled=True,
        )
        self._configure_level3_backend(wrapper, pb2, controller.events)

        result = await wrapper.sleep_serving({"level": 3, "timeout_ms": 1000})

        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(len(controller.checkpoint_calls), 1)
        control_addresses, terminal_statuses = controller.checkpoint_calls[0]
        self.assertEqual(control_addresses, tuple(addresses))
        self.assertEqual(
            [status["process_id"] for status in terminal_statuses], [2001, 2002]
        )
        self.assertTrue(
            all(status["state"] == "SLEEPING" for status in terminal_statuses)
        )
        checkpoint_index = next(
            i
            for i, event in enumerate(controller.events)
            if event[0] == "checkpoint_all"
        )
        self.assertTrue(
            all(
                i < checkpoint_index
                for i, event in enumerate(controller.events)
                if event[0] == "sleep_commit"
            )
        )

    async def test_level3_prepare_status_barrier_blocks_commit_and_rolls_back(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        store = _FakeStore()
        controller = _FakeCheckpointController()
        wrapper, pb2 = self._build_wrapper(
            control_addresses=addresses,
            lifecycle_store=store,
            checkpoint_controller=controller,
            single_node=True,
        )
        rank_statuses = self._configure_level3_backend(wrapper, pb2)
        commit_calls = {address: 0 for address in addresses}

        for address in addresses:

            async def sleep_rpc(request, *args, address=address, **kwargs):
                if request.prepare_only and address == addresses[0]:
                    rank_statuses[address].update(state="DRAINING", sleep_epoch=1)
                elif request.commit_only:
                    commit_calls[address] += 1
                return pb2.EmptyPB()

            async def wake_rpc(request, *args, address=address, **kwargs):
                rank_statuses[address].update(state="RUNNING")
                return pb2.EmptyPB()

            wrapper._dp_stubs[address].SleepServing = AsyncMock(side_effect=sleep_rpc)
            wrapper._dp_stubs[address].WakeUpServing = AsyncMock(side_effect=wake_rpc)

        result = await wrapper.sleep_serving({"level": 3, "timeout_ms": 1000})

        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertIn("did not converge", result["error"])
        self.assertEqual(commit_calls, {address: 0 for address in addresses})
        self.assertEqual(controller.checkpoint_calls, [])
        self.assertTrue(
            all(status["state"] == "RUNNING" for status in rank_statuses.values())
        )
        self.assertTrue(result["recovery_required"])
        self.assertTrue(store.values[wrapper.LIFECYCLE_LEASE_KEY])
        self.assertTrue(store.values[wrapper.LIFECYCLE_RECOVERY_KEY])

    async def test_level3_checkpoint_is_cancellation_shielded_and_holds_lease(self):
        store = _FakeStore()
        controller = _FakeCheckpointController()
        checkpoint_entered = threading.Event()
        release_checkpoint = threading.Event()

        def block_checkpoint():
            controller.manifest = {
                "state": "CHECKPOINTING",
                "pids": [2001, 2002],
            }
            checkpoint_entered.set()
            self.assertTrue(release_checkpoint.wait(timeout=5))

        controller.checkpoint_side_effect = block_checkpoint
        wrapper, pb2 = self._build_wrapper(
            control_addresses=["127.0.0.1:10001", "127.0.0.1:10009"],
            lifecycle_store=store,
            checkpoint_controller=controller,
            single_node=True,
        )
        self._configure_level3_backend(wrapper, pb2)

        task = asyncio.create_task(wrapper.sleep_serving({"level": 3}))
        while not checkpoint_entered.is_set():
            await asyncio.sleep(0.01)
        self.assertNotEqual(store.values[wrapper.LIFECYCLE_LEASE_KEY], "")
        task.cancel()
        await asyncio.sleep(0)

        status = await wrapper.get_sleep_status()
        self.assertEqual(status["state"], "CHECKPOINTING")
        release_checkpoint.set()
        result = await asyncio.wait_for(task, timeout=5)

        self.assertEqual(result, {"status": "ok"})
        self.assertFalse(task.cancelled())
        self.assertEqual(store.values[wrapper.LIFECYCLE_LEASE_KEY], "")

    async def _run_level3_commit_failure(self, failure_mode):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        store = _FakeStore()
        controller = _FakeCheckpointController()
        wrapper, pb2 = self._build_wrapper(
            control_addresses=addresses,
            lifecycle_store=store,
            checkpoint_controller=controller,
            single_node=True,
        )
        wrapper.COMMIT_MAX_ATTEMPTS = 2
        rank_statuses = self._configure_level3_backend(wrapper, pb2)
        commit_entered = asyncio.Event()
        release_commit = asyncio.Event()

        for address in addresses:

            async def sleep_rpc(request, *args, address=address, **kwargs):
                if request.prepare_only:
                    rank_statuses[address].update(state="DRAINING", sleep_epoch=1)
                    return pb2.EmptyPB()
                if not request.commit_only:
                    return pb2.EmptyPB()
                if failure_mode == "cancelled" and address == addresses[0]:
                    commit_entered.set()
                    await release_commit.wait()
                if address == addresses[0]:
                    rank_statuses[address].update(state="SLEEPING")
                    return pb2.EmptyPB()
                if failure_mode == "timeout":
                    raise self._aio_error(
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                        "level-3 commit timed out",
                    )
                rank_statuses[address].update(state="ERROR")
                return pb2.EmptyPB()

            wrapper._dp_stubs[address].SleepServing = AsyncMock(side_effect=sleep_rpc)

        task = asyncio.create_task(wrapper.sleep_serving({"level": 3}))
        if failure_mode == "cancelled":
            await asyncio.wait_for(commit_entered.wait(), timeout=5)
            task.cancel()
            await asyncio.sleep(0)
            release_commit.set()
        result = await asyncio.wait_for(task, timeout=5)

        self.assertEqual(result["state"], "RECOVERY_REQUIRED")
        self.assertTrue(result["recovery_required"])
        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertFalse(task.cancelled())
        self.assertNotEqual(store.values[wrapper.LIFECYCLE_LEASE_KEY], "")
        recovery = json.loads(store.values[wrapper.LIFECYCLE_RECOVERY_KEY])
        self.assertEqual(recovery["state"], "RECOVERY_REQUIRED")
        self.assertEqual(wrapper._frontend_lifecycle_state, "RECOVERY_REQUIRED")
        return wrapper, result

    async def test_level3_commit_cancellation_failure_persists_recovery_and_lease(self):
        _, result = await self._run_level3_commit_failure("cancelled")

        self.assertIn("unrecoverable rank state", result["error"])

    async def test_level3_commit_timeout_persists_recovery_and_lease(self):
        wrapper, result = await self._run_level3_commit_failure("timeout")

        self.assertIn("did not converge", result["error"])
        self.assertEqual(
            wrapper._dp_stubs[wrapper.control_addresses[1]].SleepServing.await_count,
            1 + wrapper.COMMIT_MAX_ATTEMPTS,
        )

    async def test_level3_commit_mixed_rank_state_persists_recovery_and_lease(self):
        _, result = await self._run_level3_commit_failure("mixed_rank")

        self.assertIn("unrecoverable rank state", result["error"])
        self.assertEqual(
            {detail.get("state") for detail in result["details"]},
            {"SLEEPING", "ERROR"},
        )

    async def test_level3_checkpointed_status_and_sleep_retry_skip_backend(self):
        controller = _FakeCheckpointController()
        controller.manifest = {
            "state": "CHECKPOINTED",
            "epoch": "17",
            "pids": [2001],
        }
        wrapper, _ = self._build_wrapper(
            checkpoint_controller=controller, single_node=True
        )
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock()
        wrapper._dp_stubs[address].SleepServing = AsyncMock()

        status = await wrapper.get_sleep_status()
        retry = await wrapper.sleep_serving({"level": 3})

        self.assertEqual(status["state"], "CHECKPOINTED")
        self.assertEqual(status["sleep_epoch"], 17)
        self.assertEqual(status["process_ids"], [2001])
        self.assertEqual(retry, {"status": "ok"})
        wrapper._dp_stubs[address].GetSleepStatus.assert_not_awaited()
        wrapper._dp_stubs[address].SleepServing.assert_not_awaited()
        self.assertEqual(controller.checkpoint_calls, [])

    async def test_non_lifecycle_backend_rpcs_do_not_read_checkpoint_manifest(self):
        dispatches = {
            "health_check": "health_check",
            "cache_status": "get_cache_status",
            "worker_status": "get_worker_status",
            "set_log_level": "set_log_level",
            "start_profile": "start_profile",
            "update_eplb_config": "update_eplb_config",
            "update_scheduler_info": "update_scheduler_info",
        }
        for uri, method_name in dispatches.items():
            with self.subTest(uri=uri):
                controller = _FakeCheckpointController()
                controller.manifest = {
                    "state": "CHECKPOINTED",
                    "epoch": "17",
                    "pids": [2001],
                }
                wrapper, _ = self._build_wrapper(
                    checkpoint_controller=controller, single_node=True
                )
                backend_method = AsyncMock(return_value={"status": "backend"})
                setattr(wrapper, method_name, backend_method)

                result = await wrapper.post_request(uri, {})

                self.assertEqual(result, {"status": "backend"})
                backend_method.assert_awaited_once()
                self.assertFalse(
                    any(event[0] == "read_manifest" for event in controller.events)
                )

    async def test_checkpointed_rejects_mismatched_sleep_level_without_backend(self):
        for level in (1, 2):
            with self.subTest(level=level):
                controller = _FakeCheckpointController()
                controller.manifest = {
                    "state": "CHECKPOINTED",
                    "epoch": "17",
                    "pids": [2001],
                }
                wrapper, _ = self._build_wrapper(
                    checkpoint_controller=controller, single_node=True
                )
                address = wrapper.control_addresses[0]
                wrapper._dp_stubs[address].GetSleepStatus = AsyncMock()
                wrapper._dp_stubs[address].SleepServing = AsyncMock()

                result = await wrapper.sleep_serving({"level": level})

                self.assertEqual(result["grpc_status"], "INVALID_ARGUMENT")
                self.assertIn("does not match configured", result["error"])
                wrapper._dp_stubs[address].GetSleepStatus.assert_not_awaited()
                wrapper._dp_stubs[address].SleepServing.assert_not_awaited()
                self.assertEqual(controller.checkpoint_calls, [])

    async def test_level3_wake_restores_before_first_backend_status_rpc(self):
        controller = _FakeCheckpointController()
        wrapper, pb2 = self._build_wrapper(
            control_addresses=["127.0.0.1:10001", "127.0.0.1:10009"],
            checkpoint_controller=controller,
            single_node=True,
        )
        rank_statuses = self._configure_level3_backend(wrapper, pb2, controller.events)
        for status in rank_statuses.values():
            status.update(
                state="SLEEPING",
                sleep_epoch=1,
                kv_memory_state="PAUSED",
                device_kv_cache_valid=False,
                gpu_resource_state="RELEASED",
            )
        controller.manifest = {"state": "CHECKPOINTED", "pids": [2001, 2002]}

        result = await wrapper.wake_up_serving()

        self.assertEqual(result, {"status": "ok"})
        restore_index = next(
            i for i, event in enumerate(controller.events) if event[0] == "restore_all"
        )
        first_status_index = next(
            i
            for i, event in enumerate(controller.events)
            if event[0] == "backend_status"
        )
        self.assertLess(restore_index, first_status_index)
        self.assertEqual(len(controller.restore_calls), 1)

    async def test_level3_restore_failure_never_calls_backend(self):
        controller = _FakeCheckpointController()
        controller.manifest = {"state": "CHECKPOINTED", "pids": [2001]}

        def fail_restore():
            raise RuntimeError("restore failed")

        controller.restore_side_effect = fail_restore
        wrapper, _ = self._build_wrapper(
            checkpoint_controller=controller, single_node=True
        )
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetSleepStatus = AsyncMock()
        wrapper._dp_stubs[address].WakeUpServing = AsyncMock()

        result = await wrapper.wake_up_serving()

        self.assertEqual(result["state"], "RECOVERY_REQUIRED")
        self.assertTrue(result["recovery_required"])
        wrapper._dp_stubs[address].GetSleepStatus.assert_not_awaited()
        wrapper._dp_stubs[address].WakeUpServing.assert_not_awaited()
        self.assertIsNotNone(controller.manifest)

    async def test_level3_restore_is_cancellation_shielded_and_holds_lease(self):
        store = _FakeStore()
        controller = _FakeCheckpointController()
        controller.manifest = {"state": "CHECKPOINTED", "pids": [2001]}
        restore_entered = threading.Event()
        release_restore = threading.Event()

        def block_restore():
            restore_entered.set()
            self.assertTrue(release_restore.wait(timeout=5))

        controller.restore_side_effect = block_restore
        wrapper, pb2 = self._build_wrapper(
            lifecycle_store=store,
            checkpoint_controller=controller,
            single_node=True,
        )
        rank_statuses = self._configure_level3_backend(wrapper, pb2)
        rank_statuses[wrapper.control_addresses[0]].update(state="SLEEPING")

        task = asyncio.create_task(wrapper.wake_up_serving())
        while not restore_entered.is_set():
            await asyncio.sleep(0.01)
        self.assertNotEqual(store.values[wrapper.LIFECYCLE_LEASE_KEY], "")
        task.cancel()
        await asyncio.sleep(0)
        status = await wrapper.get_sleep_status()
        self.assertEqual(status["state"], "RESTORING")
        release_restore.set()
        result = await asyncio.wait_for(task, timeout=5)

        self.assertEqual(result, {"status": "ok"})
        self.assertFalse(task.cancelled())
        self.assertEqual(store.values[wrapper.LIFECYCLE_LEASE_KEY], "")

    async def test_level3_checkpoint_failure_with_running_rollback_wakes_backend(self):
        controller = _FakeCheckpointController()

        def fail_checkpoint():
            controller.manifest = {"state": "RUNNING", "all_running": True}
            raise _CheckpointFailure("checkpoint failed", all_running=True)

        controller.checkpoint_side_effect = fail_checkpoint
        wrapper, pb2 = self._build_wrapper(
            control_addresses=["127.0.0.1:10001", "127.0.0.1:10009"],
            checkpoint_controller=controller,
            single_node=True,
        )
        rank_statuses = self._configure_level3_backend(wrapper, pb2)

        result = await wrapper.sleep_serving({"level": 3})

        self.assertTrue(result["recovered"])
        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertTrue(
            all(status["state"] == "RUNNING" for status in rank_statuses.values())
        )
        for address in wrapper.control_addresses:
            self.assertEqual(wrapper._dp_stubs[address].WakeUpServing.await_count, 2)

    async def test_level3_checkpoint_failure_without_manifest_wakes_backend(self):
        controller = _FakeCheckpointController()

        def fail_before_manifest():
            raise RuntimeError("failed before first driver mutation")

        controller.checkpoint_side_effect = fail_before_manifest
        wrapper, pb2 = self._build_wrapper(
            checkpoint_controller=controller, single_node=True
        )
        rank_statuses = self._configure_level3_backend(wrapper, pb2)

        result = await wrapper.sleep_serving({"level": 3})

        self.assertTrue(result["recovered"])
        self.assertEqual(controller.restore_calls, [])
        self.assertEqual(
            rank_statuses[wrapper.control_addresses[0]]["state"], "RUNNING"
        )

    async def test_level3_wake_clears_running_rollback_manifest_before_backend(self):
        controller = _FakeCheckpointController()
        controller.manifest = {
            "state": "RUNNING",
            "manifest_exists": True,
            "all_running": True,
            "pids": [2001],
        }
        wrapper, pb2 = self._build_wrapper(
            checkpoint_controller=controller, single_node=True
        )
        rank_statuses = self._configure_level3_backend(wrapper, pb2, controller.events)
        rank_statuses[wrapper.control_addresses[0]].update(state="SLEEPING")

        result = await wrapper.wake_up_serving()

        self.assertEqual(result, {"status": "ok"})
        restore_index = next(
            i for i, event in enumerate(controller.events) if event[0] == "restore_all"
        )
        status_index = next(
            i
            for i, event in enumerate(controller.events)
            if event[0] == "backend_status"
        )
        self.assertLess(restore_index, status_index)
        self.assertIsNone(controller.manifest)

    async def test_level3_checkpoint_uncertain_failure_requires_recovery(self):
        controller = _FakeCheckpointController()

        def fail_checkpoint():
            controller.manifest = {
                "state": "RECOVERY_REQUIRED",
                "error": "rank 1 remains LOCKED",
                "pids": [2001, 2002],
            }
            raise RuntimeError("partial checkpoint")

        controller.checkpoint_side_effect = fail_checkpoint
        wrapper, pb2 = self._build_wrapper(
            control_addresses=["127.0.0.1:10001", "127.0.0.1:10009"],
            checkpoint_controller=controller,
            single_node=True,
        )
        self._configure_level3_backend(wrapper, pb2)

        result = await wrapper.sleep_serving({"level": 3})

        self.assertEqual(result["state"], "RECOVERY_REQUIRED")
        self.assertTrue(result["recovery_required"])
        for address in wrapper.control_addresses:
            wrapper._dp_stubs[address].WakeUpServing.assert_not_awaited()
            wrapper._dp_stubs[address].GetSleepStatus.reset_mock()
        status = await wrapper.get_sleep_status()
        self.assertEqual(status["state"], "RECOVERY_REQUIRED")
        for address in wrapper.control_addresses:
            wrapper._dp_stubs[address].GetSleepStatus.assert_not_awaited()

    async def test_level3_wake_retry_is_idempotent(self):
        controller = _FakeCheckpointController()
        controller.manifest = {"state": "CHECKPOINTED", "pids": [2001]}
        wrapper, pb2 = self._build_wrapper(
            checkpoint_controller=controller, single_node=True
        )
        rank_statuses = self._configure_level3_backend(wrapper, pb2)
        rank_statuses[wrapper.control_addresses[0]].update(state="SLEEPING")

        first = await wrapper.wake_up_serving()
        wake_count = wrapper._dp_stubs[
            wrapper.control_addresses[0]
        ].WakeUpServing.await_count
        second = await wrapper.wake_up_serving()

        self.assertEqual(first, {"status": "ok"})
        self.assertEqual(second, {"status": "ok"})
        self.assertEqual(len(controller.restore_calls), 1)
        self.assertEqual(
            wrapper._dp_stubs[wrapper.control_addresses[0]].WakeUpServing.await_count,
            wake_count,
        )

    async def test_level3_multi_node_checkpoints_and_restores_in_rank_order(self):
        addresses = ["10.0.0.1:10001", "10.0.0.2:10001"]
        store = _FakeStore()
        controller = _FakeCheckpointController()
        events = []
        wrapper, pb2 = self._build_wrapper(
            control_addresses=addresses,
            lifecycle_store=store,
            checkpoint_controller=controller,
            single_node=False,
            rdma_enabled=True,
        )
        _, driver_states = self._configure_distributed_level3_backend(
            wrapper, pb2, events
        )

        slept = await wrapper.sleep_serving({"level": 3})
        manifest = json.loads(
            store.values[wrapper._distributed_checkpoint_key()]
        )
        woke = await wrapper.wake_up_serving()

        self.assertEqual(slept, {"status": "ok"})
        self.assertEqual(manifest["state"], "CHECKPOINTED")
        self.assertEqual(
            [target["rank"] for target in manifest["targets"]], [0, 1]
        )
        self.assertEqual(
            manifest["node_holders"],
            {
                "boot-node-0:101": "keeper-node-0",
                "boot-node-1:102": "keeper-node-1",
            },
        )
        checkpoint_actions = [
            (event[1], event[2])
            for event in events
            if event[0] == "cuda_checkpoint" and event[1] != "GET_STATE"
        ]
        self.assertEqual(
            checkpoint_actions,
            [
                ("LOCK", addresses[0]),
                ("LOCK", addresses[1]),
                ("CHECKPOINT", addresses[0]),
                ("CHECKPOINT", addresses[1]),
                ("RESTORE", addresses[0]),
                ("RESTORE", addresses[1]),
                ("UNLOCK", addresses[0]),
                ("UNLOCK", addresses[1]),
            ],
        )
        self.assertEqual(woke, {"status": "ok"})
        self.assertEqual(set(driver_states.values()), {"RUNNING"})
        self.assertEqual(store.values[wrapper._distributed_checkpoint_key()], "")
        self.assertFalse(
            any(
                event[0] in {"preflight", "checkpoint_all", "restore_all"}
                for event in controller.events
            )
        )

    async def test_level3_multi_node_partial_checkpoint_rolls_back_all_ranks(self):
        addresses = ["10.0.0.1:10001", "10.0.0.2:10001"]
        store = _FakeStore()
        controller = _FakeCheckpointController()
        wrapper, pb2 = self._build_wrapper(
            control_addresses=addresses,
            lifecycle_store=store,
            checkpoint_controller=controller,
            single_node=False,
        )
        rank_statuses, driver_states = self._configure_distributed_level3_backend(
            wrapper, pb2
        )
        original_rpc = wrapper._dp_stubs[addresses[1]].CudaCheckpointProcess.side_effect
        failed = False

        async def fail_second_checkpoint(request, *args, **kwargs):
            nonlocal failed
            if request.action == "CHECKPOINT" and not failed:
                failed = True
                identity = rank_statuses[addresses[1]]
                return pb2.CudaCheckpointResponsePB(
                    success=False,
                    cuda_result=801,
                    state=driver_states[addresses[1]],
                    error="injected checkpoint failure",
                    process_id=identity["process_id"],
                    process_starttime=identity["process_starttime"],
                    process_pid_namespace=identity["process_pid_namespace"],
                    process_boot_id=identity["process_boot_id"],
                    world_rank=identity["world_rank"],
                    holder_instance=identity["holder_instance"],
                )
            return await original_rpc(request, *args, **kwargs)

        wrapper._dp_stubs[addresses[1]].CudaCheckpointProcess.side_effect = (
            fail_second_checkpoint
        )

        with self.assertLogs(level="INFO") as captured_logs:
            result = await wrapper.sleep_serving({"level": 3})
        diagnostic_log = "\n".join(captured_logs.output)

        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertTrue(result["recovered"])
        self.assertIn("backend wake compensation completed", result["error"])
        self.assertEqual(set(driver_states.values()), {"RUNNING"})
        self.assertTrue(
            all(status["state"] == "RUNNING" for status in rank_statuses.values())
        )
        self.assertEqual(store.values[wrapper._distributed_checkpoint_key()], "")
        self.assertEqual(
            store.values.get(wrapper._lifecycle_recovery_key(), ""), ""
        )
        self.assertIn(
            "level3 checkpoint rpc end: action=CHECKPOINT "
            f"address={addresses[1]} rank=1 success=False cuda_result=801 "
            "driver_state=LOCKED",
            diagnostic_log,
        )
        self.assertIn(
            "level3 distributed rollback begin:", diagnostic_log
        )
        self.assertIn(
            "level3 distributed rollback end:", diagnostic_log
        )

    async def test_level3_multi_node_new_frontend_recovers_shared_manifest(self):
        addresses = ["10.0.0.1:10001", "10.0.0.2:10001"]
        store = _FakeStore()
        controller = _FakeCheckpointController()
        sleeper, pb2 = self._build_wrapper(
            control_addresses=addresses,
            lifecycle_store=store,
            checkpoint_controller=controller,
            single_node=False,
        )
        _, driver_states = self._configure_distributed_level3_backend(
            sleeper, pb2
        )
        slept = await sleeper.sleep_serving({"level": 3})

        waker, _ = self._build_wrapper(
            control_addresses=addresses,
            lifecycle_store=store,
            checkpoint_controller=controller,
            single_node=False,
        )
        for address in addresses:
            waker._dp_stubs[address] = sleeper._dp_stubs[address]

        woke = await waker.wake_up_serving()

        self.assertEqual(slept, {"status": "ok"})
        self.assertEqual(woke, {"status": "ok"})
        self.assertEqual(set(driver_states.values()), {"RUNNING"})
        self.assertEqual(store.values[waker._distributed_checkpoint_key()], "")

    def test_frontend_level3_options_use_runtime_topology_and_rdma_config(self):
        from rtp_llm.utils.grpc_client_wrapper import sleep_level3_options_from_config

        engine_config = SimpleNamespace(
            parallelism_config=SimpleNamespace(world_size=8, local_world_size=4),
            cache_store_config=SimpleNamespace(cache_store_rdma_mode=True),
            runtime_config=SimpleNamespace(enable_sleep_mode=True, sleep_mode_level=3),
        )
        world_info = SimpleNamespace(num_nodes=2)

        options = sleep_level3_options_from_config(engine_config, world_info)

        self.assertEqual(
            options,
            {
                "sleep_enabled": True,
                "configured_level": 3,
                "level3_enabled": True,
                "single_node": False,
                "rdma_enabled": True,
            },
        )


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


class SleepRoutesAdminAuthTest(unittest.TestCase):
    """Opt-in admin token gate for the lifecycle control routes.

    Backward compatibility contract: with RTP_LLM_SLEEP_ADMIN_TOKEN unset the
    routes stay unauthenticated (existing L1/L2 behavior). When set, a matching
    Authorization: Bearer / X-Sleep-Admin-Token header is required.
    """

    ADMIN_TOKEN = "s3cr3t-admin-token"

    def test_gate_off_allows_sleep_without_credentials(self):
        # Default (env unset): unchanged, unauthenticated behavior preserved.
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RTP_LLM_SLEEP_ADMIN_TOKEN", None)
            with client:
                response = client.post("/sleep", json={"level": 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        post_request.assert_awaited_once_with("sleep", {"level": 1})

    def test_gate_off_allows_wake_up_and_status_without_credentials(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RTP_LLM_SLEEP_ADMIN_TOKEN", None)
            with client:
                wake = client.post("/wake_up")
                status_resp = client.get("/sleep_status")
        self.assertEqual(wake.status_code, 200)
        self.assertEqual(status_resp.status_code, 200)

    def test_gate_on_with_correct_bearer_token_allowed(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request)
        with patch.dict(
            os.environ, {"RTP_LLM_SLEEP_ADMIN_TOKEN": self.ADMIN_TOKEN}, clear=False
        ):
            with client:
                response = client.post(
                    "/sleep",
                    json={"level": 1},
                    headers={"Authorization": f"Bearer {self.ADMIN_TOKEN}"},
                )
        self.assertEqual(response.status_code, 200)
        post_request.assert_awaited_once_with("sleep", {"level": 1})

    def test_gate_on_with_correct_custom_header_token_allowed(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request)
        with patch.dict(
            os.environ, {"RTP_LLM_SLEEP_ADMIN_TOKEN": self.ADMIN_TOKEN}, clear=False
        ):
            with client:
                response = client.post(
                    "/wake_up",
                    headers={"X-Sleep-Admin-Token": self.ADMIN_TOKEN},
                )
        self.assertEqual(response.status_code, 200)
        post_request.assert_awaited_once_with("wake_up", {})

    def test_gate_on_missing_token_returns_401(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request)
        with patch.dict(
            os.environ, {"RTP_LLM_SLEEP_ADMIN_TOKEN": self.ADMIN_TOKEN}, clear=False
        ):
            with client:
                response = client.post("/sleep", json={"level": 1})
        self.assertEqual(response.status_code, 401)
        self.assertIn("error", response.json())
        post_request.assert_not_awaited()

    def test_gate_on_wrong_token_returns_403(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request)
        with patch.dict(
            os.environ, {"RTP_LLM_SLEEP_ADMIN_TOKEN": self.ADMIN_TOKEN}, clear=False
        ):
            with client:
                response = client.post(
                    "/sleep",
                    json={"level": 1},
                    headers={"Authorization": "Bearer not-the-token"},
                )
        self.assertEqual(response.status_code, 403)
        self.assertIn("error", response.json())
        post_request.assert_not_awaited()

    def test_gate_on_wrong_token_blocks_status_route(self):
        post_request = AsyncMock(return_value=dict(SLEEP_STATUS_OK))
        client = build_test_client(post_request)
        with patch.dict(
            os.environ, {"RTP_LLM_SLEEP_ADMIN_TOKEN": self.ADMIN_TOKEN}, clear=False
        ):
            with client:
                response = client.get(
                    "/sleep_status",
                    headers={"X-Sleep-Admin-Token": "wrong"},
                )
        self.assertEqual(response.status_code, 403)
        post_request.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
