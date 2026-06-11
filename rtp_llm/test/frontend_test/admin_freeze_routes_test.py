"""Unit tests for the /admin/freeze|resume|freeze_status frontend proxy routes
(design doc M2) and the GrpcClientWrapper freeze methods.

The FastAPI app is built from a FrontendApp constructed without __init__ (all
heavy members mocked), and the gRPC client is replaced by an AsyncMock, so no
backend process is required.
"""

import unittest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import grpc
from fastapi.testclient import TestClient

from rtp_llm.distribute.distributed_server import WorldInfo
from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.frontend.frontend_app import FrontendApp
from rtp_llm.frontend.frontend_worker import (
    get_control_addrs_from_world_info,
    get_dp_addrs_from_world_info,
)
from rtp_llm.ops import ParallelismConfig

FREEZE_STATUS_OK: Dict[str, Any] = {
    "state": "FROZEN",
    "freeze_epoch": "1",
    "kv_memory_state": "PAUSED",
    "device_kv_cache_valid": False,
    "active_request_count": "0",
    "active_cache_transfer_count": "0",
    "gpu_resource_state": "RELEASED",
    "last_error": "",
}


def build_test_client(grpc_post_request: AsyncMock) -> TestClient:
    app_holder = FrontendApp.__new__(FrontendApp)
    app_holder.server_config = MagicMock()
    app_holder.separated_frontend = True  # skip backend wait on startup
    frontend_server = MagicMock()
    frontend_server.is_embedding = False
    frontend_server._global_controller.max_concurrency = 8
    frontend_server.check_health.return_value = True
    app_holder.frontend_server = frontend_server
    grpc_client = MagicMock()
    grpc_client.post_request = grpc_post_request
    app_holder.grpc_client = grpc_client
    return TestClient(app_holder.create_app())


class AdminFreezeRoutesTest(unittest.TestCase):

    def test_freeze_success(self):
        post_request = AsyncMock(
            return_value={"status": "ok", "state": "FROZEN", "freeze_epoch": 1}
        )
        client = build_test_client(post_request)
        with client:
            response = client.post(
                "/admin/freeze",
                json={"mode": "graceful", "drain_timeout_ms": 1000, "reason": "test"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"status": "ok", "state": "FROZEN", "freeze_epoch": 1}
        )
        post_request.assert_awaited_once_with(
            "freeze", {"mode": "graceful", "drain_timeout_ms": 1000, "reason": "test"}
        )

    def test_freeze_empty_body(self):
        post_request = AsyncMock(
            return_value={"status": "ok", "state": "FROZEN", "freeze_epoch": 1}
        )
        client = build_test_client(post_request)
        with client:
            response = client.post("/admin/freeze")
        self.assertEqual(response.status_code, 200)
        post_request.assert_awaited_once_with("freeze", {})

    def test_freeze_invalid_mode_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/admin/freeze", json={"mode": "whatever"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        post_request.assert_not_awaited()

    def test_freeze_conflict_maps_to_409(self):
        post_request = AsyncMock(
            return_value={
                "error": "freeze rejected in state RESUMING",
                "grpc_status": "FAILED_PRECONDITION",
            }
        )
        client = build_test_client(post_request)
        with client:
            response = client.post("/admin/freeze", json={})
        self.assertEqual(response.status_code, 409)
        self.assertIn("error", response.json())

    def test_resume_success(self):
        post_request = AsyncMock(
            return_value={"status": "ok", "state": "RUNNING", "freeze_epoch": 1}
        )
        client = build_test_client(post_request)
        with client:
            response = client.post("/admin/resume")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "RUNNING")
        post_request.assert_awaited_once_with("resume", {})

    def test_resume_backend_error_maps_to_500(self):
        post_request = AsyncMock(return_value={"error": "backend unreachable"})
        client = build_test_client(post_request)
        with client:
            response = client.post("/admin/resume")
        self.assertEqual(response.status_code, 500)

    def test_freeze_status_schema_passthrough(self):
        post_request = AsyncMock(return_value=dict(FREEZE_STATUS_OK))
        client = build_test_client(post_request)
        with client:
            response = client.get("/admin/freeze_status")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        for key in FREEZE_STATUS_OK:
            self.assertIn(key, body)
        post_request.assert_awaited_once_with("freeze_status", {})

    def test_freeze_status_backend_error_maps_to_500(self):
        post_request = AsyncMock(return_value={"error": "no backend"})
        client = build_test_client(post_request)
        with client:
            response = client.get("/admin/freeze_status")
        self.assertEqual(response.status_code, 500)


class GrpcClientWrapperFreezeTest(unittest.IsolatedAsyncioTestCase):

    def _build_wrapper(self, control_addresses=None):
        import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
        from rtp_llm.utils.grpc_client_wrapper import GrpcClientWrapper

        wrapper = GrpcClientWrapper(
            server_port=12345,
            control_addresses=control_addresses or ["127.0.0.1:10001"],
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

    async def test_freeze_serving_broadcasts_all_control_ranks(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        for address in addresses:
            wrapper._dp_stubs[address].FreezeServing = AsyncMock(
                return_value=pb2.EmptyPB()
            )
            wrapper._dp_stubs[address].GetFreezeStatus = AsyncMock(
                return_value=pb2.FreezeStatusResponsePB(
                    state="FROZEN",
                    freeze_epoch=1,
                    kv_memory_state="PAUSED",
                    device_kv_cache_valid=False,
                    gpu_resource_state="RELEASED",
                )
            )

        result = await wrapper.freeze_serving(
            {"mode": "force", "drain_timeout_ms": 1000, "reason": "test"}
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["state"], "FROZEN")
        self.assertEqual(result["freeze_epoch"], 1)
        self.assertEqual(result["rank_count"], 2)
        for address in addresses:
            stub = wrapper._dp_stubs[address]
            self.assertEqual(stub.FreezeServing.await_count, 2)
            prepare_request = stub.FreezeServing.await_args_list[0].args[0]
            commit_request = stub.FreezeServing.await_args_list[1].args[0]
            self.assertEqual(prepare_request.mode, "force")
            self.assertEqual(prepare_request.drain_timeout_ms, 1000)
            self.assertTrue(prepare_request.force)
            self.assertEqual(prepare_request.reason, "test")
            self.assertTrue(prepare_request.prepare_only)
            self.assertFalse(prepare_request.commit_only)
            self.assertFalse(commit_request.prepare_only)
            self.assertTrue(commit_request.commit_only)
            self.assertEqual(commit_request.drain_timeout_ms, 0)

    async def test_resume_serving_success(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        for address in addresses:
            wrapper._dp_stubs[address].ResumeServing = AsyncMock(
                return_value=pb2.EmptyPB()
            )
            wrapper._dp_stubs[address].GetFreezeStatus = AsyncMock(
                return_value=pb2.FreezeStatusResponsePB(
                    state="RUNNING",
                    freeze_epoch=1,
                    kv_memory_state="ACTIVE",
                    device_kv_cache_valid=False,
                    gpu_resource_state="ACTIVE",
                )
            )

        result = await wrapper.resume_serving()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["state"], "RUNNING")
        self.assertEqual(result["rank_count"], 2)
        for address in addresses:
            wrapper._dp_stubs[address].ResumeServing.assert_awaited_once()

    async def test_get_freeze_status_returns_full_schema(self):
        wrapper, pb2 = self._build_wrapper()
        address = wrapper.control_addresses[0]
        wrapper._dp_stubs[address].GetFreezeStatus = AsyncMock(
            return_value=pb2.FreezeStatusResponsePB(
                state="RUNNING",
                kv_memory_state="ACTIVE",
                device_kv_cache_valid=True,
                gpu_resource_state="ACTIVE",
            )
        )

        result = await wrapper.get_freeze_status()

        expected_keys = {
            "state",
            "freeze_epoch",
            "kv_memory_state",
            "device_kv_cache_valid",
            "active_request_count",
            "active_cache_transfer_count",
            "gpu_resource_state",
            "last_error",
            "rank_count",
            "rank_success_count",
            "results",
        }
        self.assertTrue(expected_keys.issubset(set(result.keys())))
        self.assertEqual(result["state"], "RUNNING")

    async def test_get_freeze_status_reports_mixed_state(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        wrapper._dp_stubs[addresses[0]].GetFreezeStatus = AsyncMock(
            return_value=pb2.FreezeStatusResponsePB(state="FROZEN", freeze_epoch=2)
        )
        wrapper._dp_stubs[addresses[1]].GetFreezeStatus = AsyncMock(
            return_value=pb2.FreezeStatusResponsePB(state="RUNNING", freeze_epoch=1)
        )

        result = await wrapper.get_freeze_status()

        self.assertEqual(result["state"], "MIXED")
        self.assertEqual(result["freeze_epoch"], 2)
        self.assertEqual(result["rank_count"], 2)
        self.assertEqual(len(result["results"]), 2)

    async def test_freeze_serving_error_carries_per_rank_status(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        wrapper._dp_stubs[addresses[0]].FreezeServing = AsyncMock(
            return_value=pb2.EmptyPB()
        )
        wrapper._dp_stubs[addresses[0]].ResumeServing = AsyncMock(
            return_value=pb2.EmptyPB()
        )
        wrapper._dp_stubs[addresses[1]].FreezeServing = AsyncMock(
            side_effect=self._aio_error(
                grpc.StatusCode.FAILED_PRECONDITION,
                "freeze rejected in state RESUMING",
            )
        )
        wrapper._dp_stubs[addresses[1]].ResumeServing = AsyncMock(
            return_value=pb2.EmptyPB()
        )

        result = await wrapper.freeze_serving({})

        self.assertIn("error", result)
        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertEqual(result["rank_count"], 2)
        self.assertEqual(result["rank_success_count"], 1)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(len(result["abort_results"]), 2)
        for address in addresses:
            wrapper._dp_stubs[address].ResumeServing.assert_awaited_once()

    async def test_freeze_serving_rejects_non_converged_final_state(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        for address in addresses:
            wrapper._dp_stubs[address].FreezeServing = AsyncMock(
                return_value=pb2.EmptyPB()
            )
        wrapper._dp_stubs[addresses[0]].GetFreezeStatus = AsyncMock(
            return_value=pb2.FreezeStatusResponsePB(state="FROZEN", freeze_epoch=1)
        )
        wrapper._dp_stubs[addresses[1]].GetFreezeStatus = AsyncMock(
            return_value=pb2.FreezeStatusResponsePB(state="RUNNING", freeze_epoch=1)
        )

        result = await wrapper.freeze_serving({})

        self.assertIn("error", result)
        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertEqual(result["freeze_status"]["state"], "MIXED")

    async def test_resume_serving_rejects_non_converged_final_state(self):
        addresses = ["127.0.0.1:10001", "127.0.0.1:10009"]
        wrapper, pb2 = self._build_wrapper(control_addresses=addresses)
        for address in addresses:
            wrapper._dp_stubs[address].ResumeServing = AsyncMock(
                return_value=pb2.EmptyPB()
            )
        wrapper._dp_stubs[addresses[0]].GetFreezeStatus = AsyncMock(
            return_value=pb2.FreezeStatusResponsePB(state="RUNNING", freeze_epoch=1)
        )
        wrapper._dp_stubs[addresses[1]].GetFreezeStatus = AsyncMock(
            return_value=pb2.FreezeStatusResponsePB(state="FROZEN", freeze_epoch=1)
        )

        result = await wrapper.resume_serving()

        self.assertIn("error", result)
        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")
        self.assertEqual(result["freeze_status"]["state"], "MIXED")


class FreezeControlAddressTest(unittest.TestCase):

    def test_control_addresses_include_all_ranks_but_dp_addresses_do_not(self):
        members = [
            WorkerInfo(
                ip="127.0.0.1",
                local_rank=rank,
                world_rank=rank,
                name=f"rank_{rank}",
                server_port=20000,
                worker_info_port_num=8,
            )
            for rank in range(4)
        ]
        world_info = WorldInfo(
            members=members,
            master=members[0],
            self=members[0],
            num_nodes=1,
            initialized=True,
        )
        pc = ParallelismConfig()
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
            WorkerInfo(
                ip="127.0.0.1",
                local_rank=rank,
                world_rank=rank,
                name=f"rank_{rank}",
                server_port=20000,
                worker_info_port_num=8,
            )
            for rank in range(4)
        ]
        world_info = WorldInfo(
            members=members,
            master=members[0],
            self=members[0],
            num_nodes=1,
            initialized=True,
        )
        pc = ParallelismConfig()
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


if __name__ == "__main__":
    unittest.main()
