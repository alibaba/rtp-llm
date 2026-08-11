import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.server.master_client import FlexlbResponse, MasterClient
from rtp_llm.utils.base_model_datatypes import GenerateInput


class _HostService:
    master_vip = SimpleNamespace(domain="")

    def get_master_addr(self):
        return "master:1234"

    def get_slave_addr(self):
        return "slave:1234"


class MasterClientDeadlineTest(unittest.IsolatedAsyncioTestCase):
    def _input(self, timeout_ms=100, ttft_timeout_ms=100):
        request = GenerateInput(
            request_id=7,
            token_ids=torch.tensor([1, 2], dtype=torch.int32),
            mm_inputs=[],
            generate_config=GenerateConfig(
                timeout_ms=timeout_ms,
                ttft_timeout_ms=ttft_timeout_ms,
            ),
        )
        request.request_deadline_monotonic_s = 100.1
        request.request_deadline_unix_ms = 200_100
        request.ttft_deadline_monotonic_s = 100.1
        return request

    async def test_master_and_slave_share_one_remaining_ttft_budget(self):
        client = MasterClient(host_service=_HostService())
        client._send_schedule_request = AsyncMock(
            side_effect=[
                FlexlbResponse.connection_failed_response(),
                FlexlbResponse.connection_failed_response(),
            ]
        )
        request = self._input()

        with patch(
            "rtp_llm.server.master_client.current_monotonic_time_s",
            side_effect=[100.02, 100.08, 100.09],
        ), patch(
            "rtp_llm.server.master_client.current_unix_time_ms",
            return_value=200_000,
        ):
            response = await client.get_backend_role_addrs([], request, request.request_id)

        self.assertTrue(response.connection_failed)
        first_call, second_call = client._send_schedule_request.await_args_list
        self.assertEqual(first_call.args[0], "master:1234")
        self.assertEqual(second_call.args[0], "slave:1234")
        self.assertEqual(first_call.args[2], 80)
        self.assertEqual(second_call.args[2], 20)
        self.assertEqual(first_call.args[1]["generate_timeout"], 80)
        self.assertEqual(second_call.args[1]["generate_timeout"], 20)
        self.assertEqual(first_call.args[1]["request_time_ms"], 200_000)
        self.assertEqual(second_call.args[1]["request_time_ms"], 200_000)

    async def test_expired_budget_does_not_call_slave(self):
        client = MasterClient(host_service=_HostService())
        client._send_schedule_request = AsyncMock(
            return_value=FlexlbResponse.connection_failed_response()
        )
        request = self._input()

        with patch(
            "rtp_llm.server.master_client.current_monotonic_time_s",
            side_effect=[100.05, 100.101],
        ), patch(
            "rtp_llm.server.master_client.current_unix_time_ms",
            return_value=200_000,
        ):
            with self.assertRaises(FtRuntimeException) as context:
                await client.get_backend_role_addrs([], request, request.request_id)

        self.assertEqual(
            context.exception.exception_type, ExceptionType.GENERATE_TIMEOUT
        )
        client._send_schedule_request.assert_awaited_once()

    async def test_exhausted_failed_route_does_not_fall_back_to_domain(self):
        host_service = _HostService()
        host_service.get_slave_addr = lambda: None
        client = MasterClient(host_service=host_service)
        client._send_schedule_request = AsyncMock(
            return_value=FlexlbResponse.connection_failed_response()
        )
        request = self._input()

        with patch(
            "rtp_llm.server.master_client.current_monotonic_time_s",
            side_effect=[100.05, 100.101],
        ), patch(
            "rtp_llm.server.master_client.current_unix_time_ms",
            return_value=200_000,
        ):
            with self.assertRaises(FtRuntimeException) as context:
                await client.get_backend_role_addrs([], request, request.request_id)

        self.assertEqual(
            context.exception.exception_type, ExceptionType.GENERATE_TIMEOUT
        )
        client._send_schedule_request.assert_awaited_once()

    async def test_business_error_does_not_retry_slave(self):
        client = MasterClient(host_service=_HostService())
        client._send_schedule_request = AsyncMock(
            return_value=FlexlbResponse.error_response(
                int(ExceptionType.MASTER_NO_AVAILABLE_WORKER), "busy"
            )
        )
        request = self._input()

        with patch(
            "rtp_llm.server.master_client.current_monotonic_time_s",
            return_value=100.01,
        ), patch(
            "rtp_llm.server.master_client.current_unix_time_ms",
            return_value=200_000,
        ):
            response = await client.get_backend_role_addrs([], request, request.request_id)

        self.assertEqual(
            response.error_code, int(ExceptionType.MASTER_NO_AVAILABLE_WORKER)
        )
        client._send_schedule_request.assert_awaited_once()

    async def test_real_elapsed_master_time_is_not_granted_again_to_slave(self):
        client = MasterClient(host_service=_HostService())
        calls = []

        async def send(addr, payload, timeout_ms, request_id):
            calls.append((addr, payload, timeout_ms, request_id))
            if len(calls) == 1:
                await asyncio.sleep(0.06)
                return FlexlbResponse.connection_failed_response()
            return FlexlbResponse.error_response(
                int(ExceptionType.MASTER_NO_AVAILABLE_WORKER), "busy"
            )

        client._send_schedule_request = send
        request = GenerateInput(
            request_id=8,
            token_ids=torch.tensor([1, 2], dtype=torch.int32),
            mm_inputs=[],
            generate_config=GenerateConfig(timeout_ms=1000, ttft_timeout_ms=1000),
        )

        response = await client.get_backend_role_addrs([], request, request.request_id)

        self.assertEqual(
            response.error_code, int(ExceptionType.MASTER_NO_AVAILABLE_WORKER)
        )
        self.assertEqual([call[0] for call in calls], ["master:1234", "slave:1234"])
        self.assertGreaterEqual(calls[0][2] - calls[1][2], 30)
        self.assertEqual(calls[0][1]["request_time_ms"], calls[1][1]["request_time_ms"])


if __name__ == "__main__":
    unittest.main()
