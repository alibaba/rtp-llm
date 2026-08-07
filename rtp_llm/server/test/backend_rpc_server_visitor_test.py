import unittest
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
