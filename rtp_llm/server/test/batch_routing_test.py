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
from unittest.mock import AsyncMock

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, RoleType
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.base_model_datatypes import RequestInfo


class BatchEnqueueRoutingTest(TestCase):
    def setUp(self):
        self.visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        self.visitor.max_seq_len = 1024
        self.visitor.sp_config = None
        self.visitor._prefill_cp_active = False
        self.visitor.source_role = "frontend"
        self.visitor.source_ip = "10.0.0.254"
        self.visitor.host_service = SimpleNamespace(service_available=True)
        self.visitor.route_ips = AsyncMock()
        self.visitor.model_rpc_client = SimpleNamespace(
            batch_enqueue=AsyncMock(return_value=[])
        )

    @staticmethod
    def input(request_id, assigned=True):
        addr = RoleAddr(
            role=RoleType.PDFUSION, ip="10.0.0.1", http_port=8080, grpc_port=8081
        )
        return SimpleNamespace(
            request_id=request_id,
            prompt_length=8,
            request_info=RequestInfo(),
            headers={},
            generate_config=GenerateConfig(
                max_new_tokens=16, role_addrs=[addr] if assigned else []
            ),
        )

    def test_preassigned_batch_never_calls_master(self):
        inputs = [self.input(1), self.input(2)]
        asyncio.run(self.visitor.batch_enqueue(inputs))
        self.visitor.route_ips.assert_not_awaited()
        self.visitor.model_rpc_client.batch_enqueue.assert_awaited_once_with(inputs)

    def test_unassigned_or_mixed_batch_cannot_create_aggregate_master_reservation(self):
        for inputs in (
            [self.input(1, False), self.input(2, False)],
            [self.input(1), self.input(2, False)],
        ):
            with self.assertRaises(FtRuntimeException) as error:
                asyncio.run(self.visitor.batch_enqueue(inputs))
            self.assertEqual(
                ExceptionType.INVALID_PARAMS, error.exception.exception_type
            )
        self.visitor.route_ips.assert_not_awaited()
        self.visitor.model_rpc_client.batch_enqueue.assert_not_awaited()

    def test_static_deployment_keeps_native_batch_rpc(self):
        self.visitor.host_service.service_available = False
        inputs = [self.input(1, False), self.input(2, False)]
        asyncio.run(self.visitor.batch_enqueue(inputs))
        self.visitor.model_rpc_client.batch_enqueue.assert_awaited_once_with(inputs)

    def test_input_validation_precedes_rpc(self):
        invalid = self.input(1)
        invalid.prompt_length = 0
        with self.assertRaises(FtRuntimeException):
            asyncio.run(self.visitor.batch_enqueue([invalid]))
        self.visitor.model_rpc_client.batch_enqueue.assert_not_awaited()

    def test_rpc_failure_is_not_replayed(self):
        self.visitor.model_rpc_client.batch_enqueue.side_effect = RuntimeError(
            "connection lost"
        )
        with self.assertRaises(RuntimeError):
            asyncio.run(self.visitor.batch_enqueue([self.input(1)]))
        self.visitor.model_rpc_client.batch_enqueue.assert_awaited_once()
        self.visitor.route_ips.assert_not_awaited()

    def test_empty_batch_has_no_side_effects(self):
        self.assertEqual([], asyncio.run(self.visitor.batch_enqueue([])))
        self.visitor.model_rpc_client.batch_enqueue.assert_not_awaited()


if __name__ == "__main__":
    main()
