import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpModel
from rtp_llm.ops import RoleType


class KimiK3MtpLocalProjectionTest(unittest.TestCase):
    def test_fp8_collective_flags_do_not_initialize_bf16_draft(self):
        model = object.__new__(KimiK3MtpModel)
        nn.Module.__init__(model)
        model.tp_size = 8
        model.parallelism_config = SimpleNamespace(role_type=RoleType.PREFILL)
        model._fp8_collective = None
        with (
            patch.dict(
                "os.environ",
                {"RTP_LLM_FP8_AG_GEMM": "1", "RTP_LLM_FP8_GEMM_RS": "1"},
            ),
            patch(
                "rtp_llm.models_py.model_desc.module_base.GptModelBase.initialize",
                return_value=True,
            ),
        ):
            self.assertTrue(model.initialize(SimpleNamespace(kv_cache=None)))
        self.assertIsNone(model._fp8_collective)

    def test_local_projection_matches_full_projection_slice(self):
        torch.manual_seed(20260928)
        model = object.__new__(KimiK3MtpModel)
        nn.Module.__init__(model)
        model.tp_size = 4
        model.tp_rank = 2
        model.embed_tokens = nn.Embedding(128, 16)
        model.enorm = nn.LayerNorm(16)
        model.hnorm = nn.LayerNorm(16)
        model.eh_proj = nn.Linear(32, 16, bias=False)
        inputs = SimpleNamespace(
            input_ids=torch.randint(0, 128, (32,)),
            combo_position_ids=torch.arange(32),
            input_hiddens=torch.randn(32, 16),
        )
        embedding = model.embed_tokens(inputs.input_ids)
        full = model.eh_proj(torch.cat((
            model.enorm(embedding * (inputs.combo_position_ids[:, None] != 0)),
            model.hnorm(inputs.input_hiddens),
        ), dim=-1))
        expected = full.narrow(0, 16, 8)
        projected_rows = []
        hook = model.eh_proj.register_forward_pre_hook(
            lambda _module, args: projected_rows.append(args[0].shape[0])
        )
        try:
            actual = model._project_local_mtp_input(inputs)
        finally:
            hook.remove()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(projected_rows, [8])


if __name__ == '__main__':
    unittest.main()
