import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.ops import SpeculativeExecutionConfig, SpeculativeType
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor


class BackendRPCServerVisitorTest(unittest.TestCase):
    def _visitor(self, sp_type=SpeculativeType.MTP, propose_step=3):
        visitor = object.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 10
        visitor.sp_config = SpeculativeExecutionConfig()
        visitor.sp_config.type = sp_type
        visitor.sp_config.gen_num_per_cycle = propose_step
        visitor.sp_config.model_type = ""
        return visitor

    def _input(self, prompt_length, *, force_disable_sp_run=False, batch_size=1):
        token_ids = torch.zeros((batch_size, prompt_length), dtype=torch.int32)
        return SimpleNamespace(
            prompt_length=prompt_length,
            token_ids=token_ids,
            generate_config=SimpleNamespace(
                max_new_tokens=1,
                force_disable_sp_run=force_disable_sp_run,
                num_return_sequences=1,
                num_beams=1,
                return_all_probs=False,
            ),
        )

    def _assert_long_prompt(self, visitor, request):
        with self.assertRaises(FtRuntimeException) as context:
            visitor._validate_input(request)
        self.assertEqual(
            context.exception.exception_type,
            ExceptionType.LONG_PROMPT_ERROR,
        )
        if (
            visitor.sp_config is not None
            and visitor.sp_config.type != SpeculativeType.NONE
        ):
            self.assertIn("model max tokens is 10", context.exception.message)
            self.assertIn("speculative reserved tokens is 3", context.exception.message)
            self.assertIn("effective max tokens is 7", context.exception.message)

    def test_mtp_reserves_proposal_tokens_from_sequence_limit(self):
        visitor = self._visitor(propose_step=3)

        visitor._validate_input(self._input(prompt_length=6))
        self._assert_long_prompt(visitor, self._input(prompt_length=7))

    def test_force_disable_does_not_bypass_mtp_sequence_limit(self):
        visitor = self._visitor(propose_step=3)

        self._assert_long_prompt(
            visitor,
            self._input(prompt_length=7, force_disable_sp_run=True),
        )

    def test_eagle_reserves_proposal_tokens_from_sequence_limit(self):
        visitor = self._visitor(sp_type=SpeculativeType.EAGLE, propose_step=3)

        visitor._validate_input(self._input(prompt_length=6))
        self._assert_long_prompt(visitor, self._input(prompt_length=7))

    def test_non_mtp_request_uses_full_sequence_limit(self):
        visitor = self._visitor(sp_type=SpeculativeType.NONE, propose_step=3)
        visitor.sp_config.model_type = "misleading-mtp-model-name"

        visitor._validate_input(self._input(prompt_length=9))
        self._assert_long_prompt(visitor, self._input(prompt_length=10))

    def test_missing_sp_config_uses_full_sequence_limit(self):
        visitor = self._visitor(sp_type=SpeculativeType.NONE, propose_step=3)
        visitor.sp_config = None

        visitor._validate_input(self._input(prompt_length=9))
        self._assert_long_prompt(visitor, self._input(prompt_length=10))

    def test_force_disable_does_not_bypass_active_sp_constraints(self):
        visitor = self._visitor(propose_step=3)
        request = self._input(
            prompt_length=2,
            force_disable_sp_run=True,
            batch_size=2,
        )

        with self.assertRaises(FtRuntimeException) as context:
            visitor.check_sp_supported(request)
        self.assertEqual(
            context.exception.exception_type,
            ExceptionType.UNSUPPORTED_OPERATION,
        )


    def test_enqueue_initializes_total_and_ttft_deadlines_once_before_route(self):
        visitor = self._visitor(sp_type=SpeculativeType.NONE)
        visitor.host_service = SimpleNamespace(service_available=True)
        visitor.route_ips = AsyncMock()
        visitor.model_rpc_client = MagicMock()
        visitor.model_rpc_client.enqueue.return_value = object()
        request = self._input(prompt_length=2)
        request.request_id = 17
        request.generate_config.timeout_ms = 1000
        request.generate_config.ttft_timeout_ms = 250

        with patch(
            "rtp_llm.utils.base_model_datatypes.current_monotonic_time_s",
            return_value=10.0,
        ), patch(
            "rtp_llm.utils.base_model_datatypes.current_unix_time_ms",
            return_value=20_000,
        ):
            result = __import__("asyncio").run(visitor.enqueue(request))

        self.assertIs(result, visitor.model_rpc_client.enqueue.return_value)
        self.assertEqual(request.request_deadline_monotonic_s, 11.0)
        self.assertEqual(request.request_deadline_unix_ms, 21_000)
        self.assertEqual(request.ttft_deadline_monotonic_s, 10.25)
        visitor.route_ips.assert_awaited_once_with(request)

        # A retry/re-entry must retain the original budget rather than start again.
        with patch(
            "rtp_llm.utils.base_model_datatypes.current_monotonic_time_s",
            return_value=10.5,
        ), patch(
            "rtp_llm.utils.base_model_datatypes.current_unix_time_ms",
            return_value=20_500,
        ):
            __import__("asyncio").run(visitor.enqueue(request))
        self.assertEqual(request.request_deadline_monotonic_s, 11.0)
        self.assertEqual(request.request_deadline_unix_ms, 21_000)
        self.assertEqual(request.ttft_deadline_monotonic_s, 10.25)

    def test_batch_deadlines_share_the_same_ingress_time(self):
        visitor = self._visitor(sp_type=SpeculativeType.NONE)
        visitor.host_service = SimpleNamespace(service_available=False)
        visitor.model_rpc_client = MagicMock()
        visitor.model_rpc_client.batch_enqueue = AsyncMock(return_value=[])
        requests = [self._input(prompt_length=2), self._input(prompt_length=2)]
        for index, request in enumerate(requests):
            request.request_id = index
            request.generate_config.timeout_ms = 1000
            request.generate_config.ttft_timeout_ms = -1

        with patch(
            "rtp_llm.server.backend_rpc_server_visitor.current_monotonic_time_s",
            return_value=30.0,
        ), patch(
            "rtp_llm.server.backend_rpc_server_visitor.current_unix_time_ms",
            return_value=40_000,
        ):
            __import__("asyncio").run(visitor.batch_enqueue(requests))

        self.assertEqual(
            [request.request_deadline_monotonic_s for request in requests],
            [31.0, 31.0],
        )
        self.assertEqual(
            [request.request_deadline_unix_ms for request in requests],
            [41_000, 41_000],
        )



if __name__ == "__main__":
    unittest.main()
