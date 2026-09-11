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
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    BatchRpcNotStartedError,
    ModelRpcClient,
)
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
        next address from master_rotation, exactly what a real multi-worker master may do.
        """
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
        visitor.placement_events = []

        class RecordingLease:
            def __init__(self, route_index):
                self.route_index = route_index

            def transfer_to_engine(self):
                visitor.placement_events.append(("transfer", self.route_index))

            async def release(self):
                visitor.placement_events.append(("release", self.route_index))

        async def fake_route_ips(
            inp,
            seq_len_hint=None,
            max_new_tokens_hint=None,
            generate_timeout_hint=None,
            placement_only=False,
            batch_seq_lens_hint=None,
            batch_request_ids_hint=None,
        ):
            route_index = len(route_calls)
            inp.generate_config.role_addrs = [
                master_rotation[route_index % len(master_rotation)]
            ]
            route_calls.append(
                (
                    inp,
                    seq_len_hint,
                    max_new_tokens_hint,
                    generate_timeout_hint,
                    placement_only,
                    batch_seq_lens_hint,
                    batch_request_ids_hint,
                )
            )
            return RecordingLease(route_index)

        visitor.route_ips = fake_route_ips
        return visitor, route_calls, sent

    @staticmethod
    def _addr(ip):
        return SimpleNamespace(
            role=RoleType.PDFUSION, ip=ip, http_port=8088, grpc_port=8089
        )

    @staticmethod
    def _input(request_id, role_addrs=()):
        generate_config = GenerateConfig(max_new_tokens=16)
        # Tests intentionally use lightweight address doubles, so assign after Pydantic
        # construction while keeping the production GenerateConfig validation path real.
        generate_config.role_addrs = list(role_addrs)
        return SimpleNamespace(
            request_id=request_id,
            prompt_length=8,
            # Real RequestInfo rather than another namespace: fill_request_info writes back into
            # it, so the stub must have the same defaults and mutability as production.
            request_info=RequestInfo(),
            headers={},
            generate_config=generate_config,
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
        self.assertEqual(sum(inp.prompt_length for inp in inputs), route_calls[0][1])
        self.assertEqual(64, route_calls[0][2])
        self.assertEqual((8, 8, 8, 8), route_calls[0][5])
        self.assertEqual((0, 1, 2, 3), route_calls[0][6])
        self.assertTrue(
            route_calls[0][4],
            "batch routing must request placement only, never enqueue the first item",
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

    def test_placement_demand_is_order_independent(self):
        def route_hints(order):
            visitor, route_calls, _ = self._visitor([self._addr("10.0.0.1")])
            inputs = [self._input(i) for i in order]
            budgets = {0: (3, 7), 1: (11, 19), 2: (5, 13)}
            for inp in inputs:
                inp.prompt_length, inp.generate_config.max_new_tokens = budgets[
                    inp.request_id
                ]
            asyncio.run(visitor.batch_enqueue(inputs))
            return route_calls[0][1:3]

        self.assertEqual((19, 39), route_hints([0, 1, 2]))
        self.assertEqual((19, 39), route_hints([2, 0, 1]))

    def test_multi_output_width_is_included_in_output_reservation(self):
        visitor, route_calls, _ = self._visitor([self._addr("10.0.0.1")])
        first = self._input(0)
        first.generate_config.max_new_tokens = 10
        first.generate_config.num_return_sequences = 3
        second = self._input(1)
        second.generate_config.max_new_tokens = 5
        second.generate_config.num_beams = 2

        asyncio.run(visitor.batch_enqueue([first, second]))

        self.assertEqual(40, route_calls[0][2], "10*3 + 5*2 output KV tokens")

    def test_heterogeneous_routing_identity_is_rejected_before_remote_work(self):
        visitor, route_calls, sent = self._visitor([self._addr("10.0.0.1")])
        first = self._input(0)
        first.headers = {"x-api-key": "tenant-a"}
        second = self._input(1)
        second.headers = {"x-api-key": "tenant-b"}

        with self.assertRaises(FtRuntimeException) as raised:
            asyncio.run(visitor.batch_enqueue([first, second]))

        self.assertEqual(ExceptionType.INVALID_PARAMS, raised.exception.exception_type)
        self.assertEqual([], route_calls)
        self.assertNotIn("inputs", sent)

    def test_duplicate_request_ids_are_rejected_before_remote_work(self):
        visitor, route_calls, sent = self._visitor([self._addr("10.0.0.1")])

        with self.assertRaises(FtRuntimeException) as raised:
            asyncio.run(visitor.batch_enqueue([self._input(7), self._input(7)]))

        self.assertEqual(ExceptionType.INVALID_PARAMS, raised.exception.exception_type)
        self.assertEqual([], route_calls)
        self.assertNotIn("inputs", sent)

    def test_confirmed_connection_failure_replaces_preassigned_target_once(self):
        replacement = self._addr("10.0.0.9")
        visitor, route_calls, _ = self._visitor([replacement])
        failed = self._addr("10.0.0.7")
        inputs = [self._input(0, [failed]), self._input(1, [failed])]
        calls = []
        replacement_deadline = []

        async def fail_then_succeed(batch, *, rpc_deadline=None):
            calls.append(visitor.model_rpc_client._select_batch_address(batch))
            if len(calls) == 1:
                deadline = asyncio.get_running_loop().time() + 1.0
                replacement_deadline.append(deadline)
                raise BatchRpcNotStartedError(
                    ExceptionType.CONNECT_FAILED,
                    "connection refused before dispatch",
                    rpc_deadline=deadline,
                )
            replacement_deadline.append(rpc_deadline)
            return []

        visitor.model_rpc_client.batch_enqueue = fail_then_succeed

        asyncio.run(visitor.batch_enqueue(inputs))

        self.assertEqual(["10.0.0.7:8089", "10.0.0.9:8089"], calls)
        self.assertEqual(
            replacement_deadline[0],
            replacement_deadline[1],
            "the replacement attempt must reuse the first attempt's absolute deadline",
        )
        self.assertEqual(1, len(route_calls), "reroute is bounded to one replacement")
        self.assertTrue(all(not inp.enqueued_by_master for inp in inputs))

    def test_exhausted_batch_deadline_skips_replacement_routing(self):
        visitor, route_calls, _ = self._visitor([self._addr("10.0.0.9")])
        failed = self._addr("10.0.0.7")
        inputs = [self._input(0, [failed]), self._input(1, [failed])]
        calls = []

        async def fail_after_deadline(batch, *, rpc_deadline=None):
            calls.append(batch)
            raise BatchRpcNotStartedError(
                ExceptionType.CONNECT_TIMEOUT,
                "connection deadline expired",
                rpc_deadline=asyncio.get_running_loop().time() - 1.0,
            )

        visitor.model_rpc_client.batch_enqueue = fail_after_deadline

        with self.assertRaises(BatchRpcNotStartedError):
            asyncio.run(visitor.batch_enqueue(inputs))

        self.assertEqual(1, len(calls))
        self.assertEqual([], route_calls)
        self.assertTrue(
            all(inp.generate_config.role_addrs == [failed] for inp in inputs),
            "an expired attempt must not mutate assignments it cannot replace",
        )

    def test_replacement_routing_is_bounded_by_remaining_batch_deadline(self):
        visitor, _, _ = self._visitor([self._addr("10.0.0.9")])
        failed = self._addr("10.0.0.7")
        inputs = [self._input(0, [failed]), self._input(1, [failed])]
        calls = []

        async def fail_before_dispatch(batch, *, rpc_deadline=None):
            calls.append(batch)
            raise BatchRpcNotStartedError(
                ExceptionType.CONNECT_FAILED,
                "connection refused before dispatch",
                rpc_deadline=asyncio.get_running_loop().time() + 0.05,
            )

        async def slow_replacement_route(*args, **kwargs):
            await asyncio.sleep(60)

        visitor.model_rpc_client.batch_enqueue = fail_before_dispatch
        visitor.route_ips = slow_replacement_route

        with self.assertRaises(BatchRpcNotStartedError) as raised:
            asyncio.run(visitor.batch_enqueue(inputs))

        self.assertEqual(ExceptionType.CONNECT_TIMEOUT, raised.exception.exception_type)
        self.assertIn("replacement routing", str(raised.exception))
        self.assertEqual(1, len(calls), "an expired route must never dispatch again")

    def test_translated_connection_errors_without_start_proof_are_never_retried(self):
        # A keepalive timeout or generic UNAVAILABLE after invocation can be translated into the
        # same CONNECT_* taxonomy as an initial dial failure. Only BatchRpcNotStartedError proves
        # that BatchGenerateCall was never invoked.
        for exception_type in (
            ExceptionType.GET_HOST_FAILED,
            ExceptionType.GET_CONNECTION_FAILED,
            ExceptionType.CONNECT_FAILED,
            ExceptionType.CONNECT_TIMEOUT,
        ):
            with self.subTest(exception_type=exception_type):
                visitor, route_calls, _ = self._visitor([self._addr("10.0.0.9")])
                inputs = [
                    self._input(0, [self._addr("10.0.0.7")]),
                    self._input(1, [self._addr("10.0.0.7")]),
                ]

                async def ambiguous(_batch):
                    raise FtRuntimeException(
                        exception_type, "transport failed after invocation"
                    )

                visitor.model_rpc_client.batch_enqueue = ambiguous

                with self.assertRaises(FtRuntimeException):
                    asyncio.run(visitor.batch_enqueue(inputs))

                self.assertEqual(
                    [], route_calls, "ambiguous execution must preserve at-most-once"
                )

    def test_uncertain_connection_reset_is_never_retried(self):
        visitor, route_calls, _ = self._visitor([self._addr("10.0.0.9")])
        failed = self._addr("10.0.0.7")
        inputs = [self._input(0, [failed]), self._input(1, [failed])]

        async def uncertain(_batch):
            raise FtRuntimeException(
                ExceptionType.CONNECTION_RESET_BY_PEER,
                "peer reset after request may have executed",
            )

        visitor.model_rpc_client.batch_enqueue = uncertain

        with self.assertRaises(FtRuntimeException):
            asyncio.run(visitor.batch_enqueue(inputs))

        self.assertEqual(
            [], route_calls, "uncertain execution must preserve at-most-once"
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

    def test_preassigned_batch_does_not_require_homogeneous_routing_metadata(self):
        visitor, route_calls, sent = self._visitor([self._addr("10.0.0.9")])
        stamped = self._addr("10.0.0.7")
        first = self._input(0, [stamped])
        first.headers = {"x-api-key": "tenant-a"}
        second = self._input(1, [stamped])
        second.headers = {"x-api-key": "tenant-b"}

        asyncio.run(visitor.batch_enqueue([first, second]))

        self.assertEqual([], route_calls)
        self.assertIn("inputs", sent)

    def test_static_fallback_does_not_validate_unused_placement_metadata(self):
        visitor, route_calls, sent = self._visitor([self._addr("10.0.0.9")])
        visitor.host_service = SimpleNamespace(service_available=False)
        first = self._input(0)
        first.headers = {"x-api-key": "tenant-a"}
        second = self._input(0)
        second.headers = {"x-api-key": "tenant-b"}

        asyncio.run(visitor.batch_enqueue([first, second]))

        self.assertEqual([], route_calls)
        self.assertIn("inputs", sent)

    def test_both_abandoned_placements_are_released_when_reroute_also_cannot_start(self):
        visitor, route_calls, _ = self._visitor(
            [self._addr("10.0.0.7"), self._addr("10.0.0.9")]
        )
        inputs = [self._input(0), self._input(1)]
        calls = []

        async def never_started(batch, *, rpc_deadline=None):
            calls.append(visitor.model_rpc_client._select_batch_address(batch))
            raise BatchRpcNotStartedError(
                ExceptionType.CONNECT_FAILED,
                "connection refused before dispatch",
                rpc_deadline=asyncio.get_running_loop().time() + 1.0,
            )

        visitor.model_rpc_client.batch_enqueue = never_started

        with self.assertRaises(BatchRpcNotStartedError):
            asyncio.run(visitor.batch_enqueue(inputs))

        self.assertEqual(["10.0.0.7:8089", "10.0.0.9:8089"], calls)
        self.assertEqual(2, len(route_calls))
        self.assertEqual([("release", 0), ("release", 1)], visitor.placement_events)

    def test_same_target_replacement_releases_both_placements_without_retry(self):
        visitor, route_calls, _ = self._visitor([self._addr("10.0.0.7")])
        inputs = [self._input(0), self._input(1)]
        calls = []

        async def never_started(batch, *, rpc_deadline=None):
            calls.append(visitor.model_rpc_client._select_batch_address(batch))
            raise BatchRpcNotStartedError(
                ExceptionType.CONNECT_FAILED,
                "connection refused before dispatch",
                rpc_deadline=asyncio.get_running_loop().time() + 1.0,
            )

        visitor.model_rpc_client.batch_enqueue = never_started

        with self.assertRaises(BatchRpcNotStartedError):
            asyncio.run(visitor.batch_enqueue(inputs))

        self.assertEqual(["10.0.0.7:8089"], calls)
        self.assertEqual(2, len(route_calls))
        self.assertEqual([("release", 0), ("release", 1)], visitor.placement_events)

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
            0,
            len(route_calls),
            "no master round-trip when the local service is not ready",
        )
        self.assertIs(
            inputs, sent["inputs"], "the batch is still dispatched, just unrouted"
        )

    def test_invalid_input_is_rejected_before_any_routing_or_dispatch(self):
        # Validation runs over every input up front, so one bad member fails the whole batch
        # before the master is contacted or anything reaches the model rpc client.
        visitor, route_calls, sent = self._visitor([self._addr("10.0.0.1")])
        bad = self._input(0)
        bad.prompt_length = (
            0  # empty prompt -> _validate_input raises LONG_PROMPT_ERROR
        )

        with self.assertRaises(FtRuntimeException):
            asyncio.run(visitor.batch_enqueue([bad, self._input(1)]))

        self.assertEqual(
            0, len(route_calls), "validation must fail before the master is contacted"
        )
        self.assertNotIn(
            "inputs", sent, "a rejected batch must never reach the model rpc client"
        )

    def test_invalid_generate_config_is_rejected_before_master_reservation(self):
        visitor, route_calls, sent = self._visitor([self._addr("10.0.0.1")])
        bad = self._input(0)
        # Pydantic accepts the union member; GenerateConfig.validate is the RPC-level
        # semantic guard that rejects negative sampling limits.
        bad.generate_config.top_k = -1

        with self.assertRaises(FtRuntimeException):
            asyncio.run(visitor.batch_enqueue([bad, self._input(1)]))

        self.assertEqual(
            [], route_calls, "invalid configuration must not reserve master-side load"
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

    def test_aggregate_output_hint_is_what_reaches_the_master(self):
        client, sent = self._client()

        asyncio.run(
            client.get_backend_role_addrs(
                block_cache_keys=[],
                cache_key_block_size=8,
                input=self._input(7),
                request_id=1,
                max_new_tokens_hint=1234,
            )
        )

        self.assertEqual(1234, sent["payload"].max_new_tokens)

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
            *,
            max_new_tokens_hint=None,
            generate_timeout_hint=None,
            aggregate_demand=False,
            batch_seq_lens=None,
            batch_request_ids=None,
        ):
            seen["seq_len_hint"] = seq_len_hint
            seen["input_pb"] = input_pb
            seen["aggregate_demand"] = aggregate_demand
            seen["batch_seq_lens"] = batch_seq_lens
            seen["batch_request_ids"] = batch_request_ids
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

    def test_normal_route_preserves_historical_one_argument_override(self):
        visitor, _ = self._visitor()
        request = self._input()
        calls = []

        async def historical_override(inp):
            calls.append(inp)
            inp.generate_config.role_addrs = [
                SimpleNamespace(
                    role=RoleType.PDFUSION,
                    ip="10.0.0.9",
                    http_port=8088,
                    grpc_port=8089,
                )
            ]
            return None

        visitor.get_master_route_addrs = historical_override

        asyncio.run(visitor.route_ips(request))

        self.assertEqual([request], calls)

    def test_batch_placement_omits_generate_input_and_cannot_enqueue_first_item(self):
        visitor, seen = self._visitor()
        request = self._input()

        # Keep the pre-existing third positional argument contract: new aggregate-hint
        # parameters are keyword-only and cannot silently reinterpret this boolean.
        asyncio.run(
            visitor.route_ips(
                request,
                210,
                True,
                batch_seq_lens_hint=(100, 110),
                batch_request_ids_hint=(1, 2),
            )
        )

        self.assertIsNone(seen["input_pb"])
        self.assertTrue(seen["aggregate_demand"])
        self.assertEqual((100, 110), seen["batch_seq_lens"])
        self.assertEqual((1, 2), seen["batch_request_ids"])
        self.assertFalse(request.enqueued_by_master)


if __name__ == "__main__":
    setup_logging()
    main()
