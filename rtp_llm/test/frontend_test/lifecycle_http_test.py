"""HTTP -> wrapper -> controller -> real gRPC sockets, with model-free ranks.

Tests wiring and phase barriers, not CUDA execution or model correctness.
"""

import unittest
from unittest.mock import patch

import grpc
import httpx
from fastapi import FastAPI

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc as pb2_grpc
from rtp_llm.frontend.sleep_routes import register_sleep_routes
from rtp_llm.utils.grpc_client_wrapper import GrpcClientWrapper
from rtp_llm.utils.lifecycle.validation import (
    normalize_lifecycle_request,
    validate_sleep_request,
)


class _Rank(pb2_grpc.RpcServiceServicer):
    def __init__(self, rank, peers, events):
        self.rank = rank
        self.peers = peers
        self.events = events
        self.state = "RUNNING"
        self.epoch = 0
        self.frozen = False
        self.quiesced = False
        self.prepared = False

    async def GetSleepStatus(self, request, context):
        sleeping = self.state == "SLEEPING"
        return pb2.SleepStatusResponsePB(
            state=self.state,
            sleep_epoch=self.epoch,
            sleep_mode_enabled=True,
            effective=True,
            supported_levels=[2],
            supported_modes=["wait", "abort"],
            worker_incarnation=f"rank-{self.rank}",
            quiesce_protocol=2,
            wake_prepare_protocol=1,
            wake_prepared=self.prepared,
            kv_memory_state="PAUSED" if sleeping else "ACTIVE",
            gpu_resource_state="RELEASED" if sleeping else "ACTIVE",
            device_kv_cache_valid=not sleeping,
        )

    async def SleepServing(self, request, context):
        if request.drain_only:
            self.state = "DRAINING"
            self.epoch += 1
            phase = "drain"
        elif request.commit_only:
            if not all(peer.quiesced for peer in self.peers):
                await context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION, "not all quiesced"
                )
            self.state = "SLEEPING"
            phase = "release"
        else:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "unexpected phase")
        self.events.append((self.epoch, self.rank, phase))
        return pb2.EmptyPB()

    async def QuiesceSleep(self, request, context):
        if request.protocol != 2:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "wrong protocol")
        if request.freeze_only:
            if not all(peer.state == "DRAINING" for peer in self.peers):
                await context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION, "not all drained"
                )
            self.frozen = True
            phase = "freeze"
        else:
            if not all(peer.frozen for peer in self.peers):
                await context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION, "not all frozen"
                )
            self.quiesced = True
            phase = "quiesce"
        self.events.append((self.epoch, self.rank, phase))
        return pb2.SleepQuiesceResponsePB()

    async def WakeUpServing(self, request, context):
        if (
            request.expected_incarnation != f"rank-{self.rank}"
            or request.expected_sleep_epoch != self.epoch
        ):
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "identity mismatch"
            )
        if request.resume_metrics_only:
            phase = "metrics"
        elif request.prepare_only:
            self.state = "WAKING_UP"
            self.prepared = True
            phase = "restore"
        elif request.commit_only:
            if not all(peer.prepared for peer in self.peers):
                await context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION, "not all prepared"
                )
            self.state = "RUNNING"
            self.frozen = False
            self.quiesced = False
            phase = "resume"
        else:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "unexpected phase")
        self.events.append((self.epoch, self.rank, phase))
        return pb2.EmptyPB()


class LifecycleHttpTest(unittest.IsolatedAsyncioTestCase):
    async def test_http_and_direct_client_reject_the_same_invalid_requests(self):
        client = GrpcClientWrapper(server_port=1)
        app = FastAPI()
        register_sleep_routes(app, client)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as http:
                for request in (
                    {"level": None},
                    {"level": 3},
                    {"timeout_ms": "bad"},
                    {"mode": "unknown"},
                    {"tags": {}},
                    {"tags": [""]},
                    {"tags": ["weights"]},
                    {"target_round": 100},
                ):
                    with self.subTest(request=request):
                        direct = await client.sleep_serving(request)
                        response = await http.post("/sleep", json=request)
                        self.assertEqual(direct["grpc_status"], "INVALID_ARGUMENT")
                        self.assertEqual(response.status_code, 400, response.text)
                        self.assertEqual(response.json()["error"], direct["error"])
                self.assertEqual(client._control_rpc.channels, {})
        finally:
            await client.close()

    def test_request_defaults_and_normalization_are_preserved(self):
        options = validate_sleep_request({})
        self.assertEqual(
            (options.level, options.mode, options.timeout_ms), (1, "wait", 3600000)
        )
        request = {"level": "2", "timeout_ms": "600000", "tags": None}
        options = validate_sleep_request(request)
        self.assertEqual((options.level, options.timeout_ms), (2, 600000))
        self.assertEqual(request, {"level": "2", "timeout_ms": "600000", "tags": None})
        self.assertEqual(normalize_lifecycle_request(None), {})
        self.assertEqual(normalize_lifecycle_request('{"level": 2}'), {"level": 2})
        with self.assertRaisesRegex(ValueError, "JSON object"):
            normalize_lifecycle_request([])

    async def test_three_cycles_reach_all_ranks_and_keep_phase_barriers(self):
        peers, events, servers, addresses = [], [], [], []
        client = None
        try:
            for rank in range(2):
                peer = _Rank(rank, peers, events)
                peers.append(peer)
                server = grpc.aio.server()
                pb2_grpc.add_RpcServiceServicer_to_server(peer, server)
                port = server.add_insecure_port("127.0.0.1:0")
                self.assertGreater(port, 0)
                await server.start()
                servers.append(server)
                addresses.append(f"127.0.0.1:{port}")
            # The serving DP route sees one root; lifecycle must reach both.
            client = GrpcClientWrapper(
                server_port=1,
                dp_addresses=addresses[:1],
                control_addresses=addresses,
                expected_control_address_count=2,
            )
            app = FastAPI()
            register_sleep_routes(app, client)
            with patch("rtp_llm.utils.lifecycle.controller.set_instance_reporting"):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as http:
                    for epoch in range(1, 4):
                        for peer in peers:
                            peer.prepared = False
                        sleep = await http.post(
                            "/sleep", json={"level": 2, "timeout_ms": 1000}
                        )
                        self.assertEqual(sleep.status_code, 200, sleep.text)
                        status = await http.get("/sleep_status")
                        self.assertEqual(status.status_code, 200, status.text)
                        self.assertEqual(status.json()["state"], "SLEEPING")
                        self.assertEqual(int(status.json()["sleep_epoch"]), epoch)
                        is_sleeping = await http.get("/is_sleeping")
                        self.assertTrue(is_sleeping.json()["is_sleeping"])
                        wake = await http.post("/wake_up", json={})
                        self.assertEqual(wake.status_code, 200, wake.text)
                        status = await http.get("/sleep_status")
                        self.assertEqual(status.json()["state"], "RUNNING")
                        self.assertEqual(int(status.json()["active_request_count"]), 0)
                        for phase in (
                            "drain",
                            "freeze",
                            "quiesce",
                            "release",
                            "restore",
                            "resume",
                            "metrics",
                        ):
                            self.assertEqual(
                                sorted(
                                    rank
                                    for e, rank, p in events
                                    if e == epoch and p == phase
                                ),
                                [0, 1],
                            )
                        phases = [p for e, _, p in events if e == epoch]
                        for first, second in (
                            ("drain", "freeze"),
                            ("freeze", "quiesce"),
                            ("quiesce", "release"),
                            ("restore", "resume"),
                            ("resume", "metrics"),
                        ):
                            self.assertLess(
                                max(
                                    i
                                    for i, phase in enumerate(phases)
                                    if phase == first
                                ),
                                min(
                                    i
                                    for i, phase in enumerate(phases)
                                    if phase == second
                                ),
                            )
        finally:
            if client is not None:
                await client.close()
            for server in servers:
                await server.stop(None)


if __name__ == "__main__":
    unittest.main()
