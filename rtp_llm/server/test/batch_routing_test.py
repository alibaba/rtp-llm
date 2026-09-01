import asyncio
import sys
from enum import IntEnum
from types import SimpleNamespace
from unittest.mock import MagicMock

# Mock the ops module to avoid CUDA dependency in this unit test.
# This MUST be at the very top, before any other rtp_llm import.
class _FakeRoleType(IntEnum):
    UNKNOWN = 0
    PREFILL = 1
    DECODE = 2
    PDFUSION = 3
    VIT = 4
    FRONTEND = 5


mock_ops = MagicMock()
mock_ops.RoleType = _FakeRoleType
mock_comm = MagicMock()
mock_nccl_op = MagicMock()
mock_compute_ops = MagicMock()
mock_comm.nccl_op = mock_nccl_op
mock_ops.comm = mock_comm
mock_ops.compute_ops = mock_compute_ops
sys.modules["rtp_llm.ops"] = mock_ops
sys.modules["rtp_llm.ops.comm"] = mock_comm
sys.modules["rtp_llm.ops.compute_ops"] = mock_compute_ops
sys.modules["rtp_llm.ops.comm.nccl_op"] = mock_nccl_op

from unittest import TestCase, main

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, RoleType
from rtp_llm.config.log_config import setup_logging
from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.base_model_datatypes import GenerateInput, RequestInfo

"""Batch routing/scheduling behaviour of the rtp_llm/server layer.

These exercise `BackendRPCServerVisitor` and `MasterClient` — server-layer components — so they
live with the server tests rather than in the model_rpc test module they grew up in. They still
drive the real `ModelRpcClient._select_batch_address` where a test asserts on the seam between the
two layers, which is the direction the dependency already runs (server -> model_rpc).
"""


class BatchEnqueueRoutingTest(TestCase):
    """A batch RPC is one scheduling unit — one BatchGenerateCall to one backend. If the visitor
    routed each input separately, a multi-worker master could legally stamp a different backend
    on every input, and _select_batch_address would (rightly) reject the batch as having no
    single valid target. So BackendRPCServerVisitor.batch_enqueue must route once and propagate
    the same assignment to every unrouted input.
    """

    @staticmethod
    def _visitor(master_rotation):
        """Visitor stub whose route_ips simulates a round-robin master: each call stamps the
        next address from master_rotation, exactly what a real multi-worker master may do."""
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 1024
        visitor.sp_config = None
        visitor._prefill_cp_active = False
        visitor.host_service = SimpleNamespace(service_available=True)
        # batch_enqueue runs fill_request_info over every input before routing, so the stub needs
        # the same two identity fields the real visitor carries.
        visitor.source_role = "frontend"
        visitor.source_ip = "10.0.0.254"
        sent = {}

        async def fake_batch_enqueue(inputs):
            sent["inputs"] = inputs
            return []

        model_rpc_client = ModelRpcClient.__new__(ModelRpcClient)
        model_rpc_client._decode_entrance = False
        model_rpc_client.batch_enqueue = fake_batch_enqueue
        visitor.model_rpc_client = model_rpc_client
        route_calls = []

        async def fake_route_ips(inp, seq_len_hint=None):
            inp.generate_config.role_addrs = [
                master_rotation[len(route_calls) % len(master_rotation)]
            ]
            route_calls.append((inp, seq_len_hint))

        visitor.route_ips = fake_route_ips
        return visitor, route_calls, sent

    @staticmethod
    def _addr(ip):
        return SimpleNamespace(
            role=RoleType.PDFUSION, ip=ip, http_port=8088, grpc_port=8089
        )

    @staticmethod
    def _input(request_id, role_addrs=()):
        return SimpleNamespace(
            request_id=request_id,
            prompt_length=8,
            # Real RequestInfo rather than another namespace: fill_request_info writes back into
            # it, so the stub must have the same defaults and mutability as production.
            request_info=RequestInfo(),
            headers={},
            generate_config=SimpleNamespace(
                role_addrs=list(role_addrs), max_new_tokens=16, trace_id=None
            ),
        )

    def test_round_robin_master_cannot_scatter_a_batch_across_backends(self):
        visitor, route_calls, sent = self._visitor(
            [self._addr("10.0.0.1"), self._addr("10.0.0.2")]
        )
        inputs = [self._input(i) for i in range(4)]

        asyncio.run(visitor.batch_enqueue(inputs))

        self.assertEqual(
            1, len(route_calls), "a batch is one scheduling unit: one master round-trip"
        )
        # The single routing call must carry the batch's aggregate weight — otherwise
        # the master accounts one request's load while N inputs land on the worker.
        self.assertEqual(
            sum(inp.prompt_length for inp in inputs), route_calls[0][1]
        )
        # Every input carries the first routing decision, and the whole batch resolves to a
        # single target through the real _select_batch_address — the exact seam that a
        # per-input routing loop breaks under a round-robin master.
        client = ModelRpcClient.__new__(ModelRpcClient)
        client._addresses = ["10.9.9.9:1"]
        client._decode_entrance = False
        self.assertEqual("10.0.0.1:8089", client._select_batch_address(inputs))
        self.assertIs(inputs, sent["inputs"])
        # Propagated as copies: one input's later mutation must not silently retarget siblings.
        self.assertIsNot(
            inputs[0].generate_config.role_addrs,
            inputs[1].generate_config.role_addrs,
        )

    def test_dispatcher_pre_assigned_chunk_never_touches_the_master(self):
        visitor, route_calls, sent = self._visitor([self._addr("10.0.0.9")])
        stamped = self._addr("10.0.0.7")
        inputs = [self._input(i, [stamped]) for i in range(3)]

        asyncio.run(visitor.batch_enqueue(inputs))

        self.assertEqual(
            0,
            len(route_calls),
            "pre-assigned chunks carry the dispatcher's decision; re-routing would discard it",
        )
        client = ModelRpcClient.__new__(ModelRpcClient)
        client._addresses = []
        client._decode_entrance = False
        self.assertEqual("10.0.0.7:8089", client._select_batch_address(inputs))

    def test_mixed_batch_is_rejected_before_master_even_if_it_would_agree(self):
        # A batch RPC must have one routing source. Asking the master to route only the
        # unrouted suffix has a remote accounting side effect and cannot prove in advance
        # that it will agree with the caller's existing assignment, so reject first.
        shared = self._addr("10.0.0.7")
        visitor, route_calls, sent = self._visitor([shared])
        inputs = [self._input(0, [shared]), self._input(1), self._input(2)]

        with self.assertRaises(FtRuntimeException) as raised:
            asyncio.run(visitor.batch_enqueue(inputs))

        self.assertEqual(ExceptionType.INVALID_PARAMS, raised.exception.exception_type)
        self.assertEqual(0, len(route_calls))
        self.assertNotIn("inputs", sent)

    def test_mixed_batch_is_rejected_before_master_when_it_would_disagree(self):
        # The same fail-fast contract is independent of the master's next pick: neither
        # routing nor dispatch may happen for a malformed mixed-source batch.
        visitor, route_calls, sent = self._visitor([self._addr("10.0.0.9")])
        inputs = [self._input(0, [self._addr("10.0.0.7")]), self._input(1)]

        with self.assertRaises(FtRuntimeException) as raised:
            asyncio.run(visitor.batch_enqueue(inputs))

        self.assertEqual(ExceptionType.INVALID_PARAMS, raised.exception.exception_type)
        self.assertEqual(0, len(route_calls))
        self.assertNotIn("inputs", sent)

    def test_service_unavailable_skips_routing_but_still_dispatches(self):
        # When the local host service is not ready, batch_enqueue must not contact the master;
        # it still hands the (unrouted) batch to the model rpc client, which will resolve a target
        # from its static address list.
        visitor, route_calls, sent = self._visitor([self._addr("10.0.0.1")])
        visitor.host_service = SimpleNamespace(service_available=False)
        inputs = [self._input(i) for i in range(3)]

        asyncio.run(visitor.batch_enqueue(inputs))

        self.assertEqual(
            0, len(route_calls), "no master round-trip when the local service is not ready"
        )
        self.assertIs(
            inputs, sent["inputs"], "the batch is still dispatched, just unrouted"
        )

    def test_invalid_input_is_rejected_before_any_routing_or_dispatch(self):
        # Validation runs over every input up front, so one bad member fails the whole batch
        # before the master is contacted or anything reaches the model rpc client.
        visitor, route_calls, sent = self._visitor([self._addr("10.0.0.1")])
        bad = self._input(0)
        bad.prompt_length = 0  # empty prompt -> _validate_input raises LONG_PROMPT_ERROR

        with self.assertRaises(FtRuntimeException):
            asyncio.run(visitor.batch_enqueue([bad, self._input(1)]))

        self.assertEqual(
            0, len(route_calls), "validation must fail before the master is contacted"
        )
        self.assertNotIn(
            "inputs", sent, "a rejected batch must never reach the model rpc client"
        )


class SchedulePayloadSeqLenTest(TestCase):
    """The last hop of the aggregate-weight fix: the seq_len that actually reaches the master
    in the /schedule payload. BatchEnqueueRoutingTest pins the first hop (batch_enqueue
    aggregating the batch), RouteIpsSeqLenHintTest pins the two in between, and this class
    pins where the hint finally beats input.prompt_length.
    """

    @staticmethod
    def _client():
        from rtp_llm.server.master_client import MasterClient

        client = MasterClient.__new__(MasterClient)
        client.host_service = SimpleNamespace(
            get_master_addr=lambda: "10.0.0.1:8090",
            get_slave_addr=lambda: None,
        )
        client.master_config = SimpleNamespace(master_default_timeout_ms=3000)
        sent = {}

        async def fake_send(addr, payload, generate_timeout_ms, request_id):
            sent["payload"] = payload
            return SimpleNamespace(
                code=200,
                queue_length=0,
                server_status=[],
                enqueued_by_master=False,
            )

        client._send_schedule_request = fake_send
        return client, sent

    @staticmethod
    def _input(prompt_length):
        return GenerateInput(
            request_id=1,
            token_ids=torch.arange(prompt_length),
            mm_inputs=[],
            generate_config=GenerateConfig(
                max_new_tokens=16,
                timeout_ms=3000,
                ttft_timeout_ms=-1,
                traffic_reject_priority=100,
            ),
        )

    def test_seq_len_hint_is_what_reaches_the_master(self):
        client, sent = self._client()

        asyncio.run(
            client.get_backend_role_addrs(
                block_cache_keys=[],
                cache_key_block_size=8,
                input=self._input(7),
                request_id=1,
                seq_len_hint=210,
            )
        )

        self.assertEqual(210, sent["payload"].seq_len)

    def test_without_a_hint_the_single_input_length_is_reported(self):
        client, sent = self._client()

        asyncio.run(
            client.get_backend_role_addrs(
                block_cache_keys=[],
                cache_key_block_size=8,
                input=self._input(7),
                request_id=1,
            )
        )

        self.assertEqual(7, sent["payload"].seq_len)

    def test_a_zero_hint_is_honoured_rather_than_treated_as_absent(self):
        # `if seq_len_hint else` would silently fall back here; the guard must be `is not None`.
        client, sent = self._client()

        asyncio.run(
            client.get_backend_role_addrs(
                block_cache_keys=[],
                cache_key_block_size=8,
                input=self._input(7),
                request_id=1,
                seq_len_hint=0,
            )
        )

        self.assertEqual(0, sent["payload"].seq_len)


class RouteIpsSeqLenHintTest(TestCase):
    """The stretch between the two ends of the chain. BatchEnqueueRoutingTest stubs route_ips
    out and SchedulePayloadSeqLenTest starts below it, so route_ips ->
    get_master_route_addrs -> get_backend_role_addrs is exactly where dropping the argument
    leaves both of them green while the master silently goes back to being told one request's
    weight for a whole batch.
    """

    @staticmethod
    def _visitor():
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.master_config = None
        visitor.seq_size_per_block = 8
        visitor._page_rr_route_cache_keys = False
        visitor._page_rr_cp_size = 1
        visitor._report_recent_cache_key_metrics = lambda keys: None
        visitor.backend_role_list = [RoleType.PDFUSION]
        visitor.host_service = SimpleNamespace(get_master_addr=lambda: "10.0.0.1:8090")
        seen = {}

        async def fake_get_backend_role_addrs(
            block_cache_keys,
            cache_key_block_size,
            input,
            request_id,
            input_pb=None,
            seq_len_hint=None,
        ):
            seen["seq_len_hint"] = seq_len_hint
            return SimpleNamespace(
                is_ok=True,
                role_addrs=[
                    SimpleNamespace(
                        role=RoleType.PDFUSION,
                        ip="10.0.0.9",
                        http_port=8088,
                        grpc_port=8089,
                    )
                ],
                enqueued_by_master=False,
                connection_failed=False,
                error_code=None,
                error_message=None,
            )

        visitor.master_client = SimpleNamespace(
            get_backend_role_addrs=fake_get_backend_role_addrs
        )
        return visitor, seen

    @staticmethod
    def _input():
        return GenerateInput(
            request_id=1,
            token_ids=torch.tensor([1, 2, 3, 4]),
            mm_inputs=[],
            generate_config=GenerateConfig(max_new_tokens=16),
        )

    def test_the_hint_survives_route_ips_down_to_the_master_client(self):
        visitor, seen = self._visitor()

        asyncio.run(visitor.route_ips(self._input(), seq_len_hint=210))

        self.assertEqual(210, seen["seq_len_hint"])

    def test_no_hint_is_passed_through_as_absent_rather_than_invented(self):
        # The fallback to input.prompt_length belongs to master_client alone; a visitor that
        # substituted a value here would make the zero-hint guard below it unreachable.
        visitor, seen = self._visitor()

        asyncio.run(visitor.route_ips(self._input()))

        self.assertIsNone(seen["seq_len_hint"])


if __name__ == "__main__":
    setup_logging()
    main()
