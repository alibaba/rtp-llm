import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpModel
from rtp_llm.models_py.modules import Embedding
from rtp_llm.ops import RoleType


class KimiK3MtpLocalProjectionTest(unittest.TestCase):
    def test_moe_workspace_covers_prefill_and_padded_verify(self):
        from rtp_llm.models_py.model_desc import kimi_k3

        for tp in (4, 8):
            for role in (RoleType.DECODE, RoleType.PREFILL, RoleType.PDFUSION):
                for batch, width in ((63, 3), (64, 4), (100, 3)):
                    with self.subTest(tp=tp, role=role, batch=batch, width=width):
                        config = SimpleNamespace(
                            num_layers=1, vocab_size=16,
                            capture_aux_hidden_layer_ids=[],
                            moe_prefill_max_tokens_per_rank=262144,
                            max_seq_len=1048576,
                            gen_num_per_cycle=width - 1,
                            k3_runtime_config=SimpleNamespace(attn_res_block_size=0),
                            layernorm_eps=1e-5,
                        )
                        parallel = SimpleNamespace(
                            role_type=role, tp_size=tp, tp_rank=0,
                            dp_size=8 // tp, ep_size=8, pp_size=1,
                            get_attn_tp_size=lambda: tp, enable_sp=True,
                        )
                        weights = SimpleNamespace(
                            weights=[{}], get_global_weight=lambda _: torch.ones(1),
                        )
                        layer = nn.Identity()
                        layer.layer_type = object()
                        with (
                            patch.dict(os.environ, {
                                "RTP_LLM_FP8_COLLECTIVE_MAX_TOKENS": "65536",
                            }),
                            patch.object(kimi_k3, "Embedding", return_value=nn.Identity()),
                            patch.object(kimi_k3, "RMSNorm", return_value=nn.Identity()),
                            patch.object(kimi_k3, "KimiK3DecoderLayer", return_value=layer) as ctor,
                        ):
                            kimi_k3.KimiK3Model(config, parallel, weights, batch)
                        # Enumerate whole-request SP padding independently.
                        verify = 1
                        for count in range(1, batch + 1):
                            for q in (1, width):
                                padded = count
                                while padded * q % tp:
                                    padded += 1
                                verify = max(verify, padded * q // tp)
                        expected = (
                            verify if role == RoleType.DECODE
                            else max(verify, 262144 // tp)
                        )
                        self.assertEqual(ctor.call_args.args[-1], expected)

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
            self.assertTrue(model.initialize(SimpleNamespace(kv_cache=None, is_decode_role=False)))
        self.assertIsNone(model._fp8_collective)

    def test_local_projection_matches_full_projection_slice(self):
        torch.manual_seed(20260928)
        model = object.__new__(KimiK3MtpModel)
        nn.Module.__init__(model)
        model.tp_size = 4
        model.tp_rank = 2
        full_embedding = nn.Embedding(128, 16)
        local_width = full_embedding.weight.shape[1] // model.tp_size
        local_weight = full_embedding.weight.detach().narrow(
            1, model.tp_rank * local_width, local_width
        ).contiguous()
        model.embed_tokens = Embedding(
            SimpleNamespace(),
            SimpleNamespace(get_attn_tp_size=lambda: model.tp_size),
            local_weight,
        )
        model.enorm = nn.LayerNorm(16)
        model.hnorm = nn.LayerNorm(16)
        model.eh_proj = nn.Linear(32, 16, bias=False)
        inputs = SimpleNamespace(
            input_ids=torch.randint(0, 128, (32,)),
            combo_position_ids=torch.arange(32),
            input_hiddens=torch.randn(32, 16),
        )
        embedding = full_embedding(inputs.input_ids)
        full = model.eh_proj(torch.cat((
            model.enorm(embedding * (inputs.combo_position_ids[:, None] != 0)),
            model.hnorm(inputs.input_hiddens),
        ), dim=-1))
        expected = full.narrow(0, 16, 8)
        projected_rows = []
        hook = model.eh_proj.register_forward_pre_hook(
            lambda _module, args: projected_rows.append(args[0].shape[0])
        )
        def embedding_kernel(output, ids, weight, *_):
            output.copy_(torch.nn.functional.embedding(ids, weight))

        def gather_hidden_shards(local):
            torch.testing.assert_close(
                local, embedding.narrow(1, model.tp_rank * local_width, local_width),
                rtol=0, atol=0,
            )
            return torch.cat(embedding.split(local_width, dim=1), dim=0)

        try:
            with (
                patch(
                    "rtp_llm.models_py.modules.base.common.embedding.rtp_llm_ops.embedding",
                    side_effect=embedding_kernel,
                ),
                patch.object(model, "_gather_modeling_boundary", side_effect=gather_hidden_shards) as gather,
            ):
                actual = model._project_local_mtp_input(inputs)
                gather.assert_called_once()
        finally:
            hook.remove()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(projected_rows, [8])


if __name__ == '__main__':
    unittest.main()
