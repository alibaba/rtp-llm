"""Unit tests for the /admin/freeze|resume|freeze_status frontend proxy routes
(design doc M2) and the GrpcClientWrapper freeze methods.

The FastAPI app is built from a FrontendApp constructed without __init__ (all
heavy members mocked), and the gRPC client is replaced by an AsyncMock, so no
backend process is required.
"""

import unittest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from rtp_llm.frontend.frontend_app import FrontendApp

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

    def _build_wrapper(self):
        import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
        from rtp_llm.utils.grpc_client_wrapper import GrpcClientWrapper

        wrapper = GrpcClientWrapper(server_port=12345)
        wrapper.channel = MagicMock()
        wrapper.stub = MagicMock()
        return wrapper, pb2

    async def test_freeze_serving_builds_request_and_reports_status(self):
        wrapper, pb2 = self._build_wrapper()
        wrapper.stub.FreezeServing = AsyncMock(return_value=pb2.EmptyPB())
        status_response = pb2.FreezeStatusResponsePB(
            state="FROZEN",
            freeze_epoch=1,
            kv_memory_state="PAUSED",
            device_kv_cache_valid=False,
            gpu_resource_state="RELEASED",
        )
        wrapper.stub.GetFreezeStatus = AsyncMock(return_value=status_response)

        result = await wrapper.freeze_serving(
            {"mode": "force", "drain_timeout_ms": 1000, "reason": "test"}
        )

        self.assertEqual(result, {"status": "ok", "state": "FROZEN", "freeze_epoch": 1})
        request = wrapper.stub.FreezeServing.await_args.args[0]
        self.assertEqual(request.mode, "force")
        self.assertEqual(request.drain_timeout_ms, 1000)
        self.assertTrue(request.force)
        self.assertEqual(request.reason, "test")

    async def test_resume_serving_success(self):
        wrapper, pb2 = self._build_wrapper()
        wrapper.stub.ResumeServing = AsyncMock(return_value=pb2.EmptyPB())
        wrapper.stub.GetFreezeStatus = AsyncMock(
            return_value=pb2.FreezeStatusResponsePB(state="RUNNING", freeze_epoch=1)
        )

        result = await wrapper.resume_serving()

        self.assertEqual(
            result, {"status": "ok", "state": "RUNNING", "freeze_epoch": 1}
        )

    async def test_get_freeze_status_returns_full_schema(self):
        wrapper, pb2 = self._build_wrapper()
        wrapper.stub.GetFreezeStatus = AsyncMock(
            return_value=pb2.FreezeStatusResponsePB(state="RUNNING")
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
        }
        self.assertEqual(expected_keys, set(result.keys()))
        self.assertEqual(result["state"], "RUNNING")

    async def test_freeze_serving_error_carries_grpc_status(self):
        import grpc

        wrapper, _ = self._build_wrapper()

        error = grpc.aio.AioRpcError(
            code=grpc.StatusCode.FAILED_PRECONDITION,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="freeze rejected in state RESUMING",
        )
        wrapper.stub.FreezeServing = AsyncMock(side_effect=error)

        result = await wrapper.freeze_serving({})

        self.assertIn("error", result)
        self.assertEqual(result["grpc_status"], "FAILED_PRECONDITION")


if __name__ == "__main__":
    unittest.main()
