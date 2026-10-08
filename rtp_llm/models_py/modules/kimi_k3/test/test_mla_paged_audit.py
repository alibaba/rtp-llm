"""Exercise audit contracts against the actual compiled attention-input ABI."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.kimi_k3 import mla_verify
from rtp_llm.ops.compute_ops import PyAttentionInputs


class PagedMlaAuditTest(unittest.TestCase):
    def test_real_attention_inputs_for_all_decode_phases(self):
        cases = (
            ("target_verify", True, False, True, 4, torch.float8_e4m3fn),
            ("proposal_or_decode", False, False, False, 1, torch.bfloat16),
            ("mtp_update", False, True, False, 4, torch.bfloat16),
        )
        for phase, verify, update, fp8, q, dtype in cases:
            for graph in (False, True):
                with self.subTest(phase=phase, graph=graph):
                    inputs = PyAttentionInputs()
                    inputs.is_target_verify = verify
                    inputs.is_mtp_draft_update = update
                    inputs.is_cuda_graph = graph
                    inputs.logical_request_count = 32
                    inputs.physical_request_count = 32
                    inputs.physical_token_count = 32 * q
                    inputs.input_lengths = torch.full((32,), q, dtype=torch.int32)
                    impl = object.__new__(mla_verify.KimiK3MlaVerifyImpl)
                    impl.graph_mode = graph
                    impl.batch = 32
                    impl.tokens = 32 * q
                    impl.page_rr = True
                    impl.native = SimpleNamespace(
                        fp8_compute=fp8, workspace=torch.empty(1, dtype=torch.uint8)
                    )
                    with patch.object(
                        mla_verify, "_AUDIT_CONTRACTS", True
                    ), patch.object(
                        mla_verify, "_audit_contracts", set()
                    ), patch.object(
                        mla_verify.logging, "info"
                    ) as log, patch.object(
                        torch.cuda,
                        "synchronize",
                        side_effect=AssertionError("audit must not synchronize CUDA"),
                    ), patch.object(
                        torch.cuda,
                        "_lazy_init",
                        side_effect=AssertionError("audit must not initialize CUDA"),
                    ):
                        impl._record_input_contract(inputs)
                        impl._record_input_contract(inputs)
                    log.assert_called_once()
                    values = log.call_args.args[1:]
                    self.assertEqual(
                        values[:7], (phase, graph, 32, 32, q, 32 * q, dtype)
                    )
                    self.assertEqual(values[7], "tokenspeed_page_rr")
                    self.assertEqual(values[8], impl.native.workspace.data_ptr())


if __name__ == "__main__":
    unittest.main()
