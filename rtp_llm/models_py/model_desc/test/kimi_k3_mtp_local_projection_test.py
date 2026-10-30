import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpModel
from rtp_llm.models_py.modules import Embedding
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.ops import RoleType


class KimiK3MtpLocalProjectionTest(unittest.TestCase):
    def test_moe_workspace_uses_resolved_k3_capacity_for_every_role(self):
        from rtp_llm.models_py.model_desc import kimi_k3

        for tp in (4, 8):
            for role in (RoleType.DECODE, RoleType.PREFILL, RoleType.PDFUSION):
                for capacity in (192, 4096, 262144):
                    with self.subTest(tp=tp, role=role, capacity=capacity):
                        config = SimpleNamespace(
                            num_layers=1,
                            vocab_size=16,
                            capture_aux_hidden_layer_ids=[],
                            moe_prefill_max_tokens_per_rank=262144,
                            max_seq_len=1048576,
                            gen_num_per_cycle=2,
                            k3_runtime_config=SimpleNamespace(attn_res_block_size=0),
                            layernorm_eps=1e-5,
                        )
                        parallel = SimpleNamespace(
                            role_type=role,
                            tp_size=tp,
                            tp_rank=0,
                            dp_size=8 // tp,
                            ep_size=8,
                            pp_size=1,
                            get_attn_tp_size=lambda: tp,
                            enable_sp=True,
                        )
                        weights = SimpleNamespace(
                            weights=[{}],
                            get_global_weight=lambda _: torch.ones(1),
                        )
                        layer = nn.Identity()
                        layer.layer_type = object()
                        with (
                            patch.dict(
                                os.environ,
                                {
                                    "RTP_LLM_FP8_COLLECTIVE_MAX_TOKENS": "65536",
                                },
                            ),
                            patch.object(
                                kimi_k3, "Embedding", return_value=nn.Identity()
                            ),
                            patch.object(
                                kimi_k3, "RMSNorm", return_value=nn.Identity()
                            ),
                            patch.object(
                                kimi_k3, "KimiK3DecoderLayer", return_value=layer
                            ) as ctor,
                        ):
                            kimi_k3.KimiK3Model(
                                config, parallel, weights, 63, max_moe_tokens=capacity
                            )
                        # Neither the generic field nor a second Decode formula
                        # may replace the K3 loader's resolved physical capacity.
                        self.assertEqual(ctor.call_args.args[-1], capacity // tp)

    def _bf16_mtp_model(self, device="cpu"):
        model = object.__new__(KimiK3MtpModel)
        nn.Module.__init__(model)
        model.tp_size = 8
        model.parallelism_config = SimpleNamespace(role_type=RoleType.PREFILL)
        model.config = SimpleNamespace(hidden_size=8)
        model.fp8_collective_max_tokens = 64
        model._fp8_collective = None
        attention = nn.Module()
        attention.input = CudaF16Linear(
            torch.ones(8, 8, dtype=torch.bfloat16, device=device)
        )
        attention.output = CudaF16Linear(
            torch.ones(8, 8, dtype=torch.bfloat16, device=device)
        )
        attention._bf16_collective = None
        layer = nn.Module()
        layer.attention = attention
        model.layers = nn.ModuleList([layer])
        return model

    def test_fp8_collective_flags_do_not_initialize_bf16_draft(self):
        model = self._bf16_mtp_model()
        init_resource = SimpleNamespace(kv_cache=None, is_decode_role=False)
        with (
            patch.dict(
                "os.environ",
                {"RTP_LLM_FP8_AG_GEMM": "1", "RTP_LLM_FP8_GEMM_RS": "1"},
            ),
            patch(
                "rtp_llm.models_py.model_desc.module_base.GptModelBase.initialize",
                return_value=True,
            ),
            patch(
                "rtp_llm.models_py.modules.kimi_k3.kernel_jit_warmup.warmup_kimi_k3_kernel_jit",
            ) as warmup,
            patch(
                "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_comm_fusion.Fp8CollectiveProjection",
            ) as fp8_collective,
        ):
            self.assertTrue(model.initialize(init_resource))
        warmup.assert_called_once_with(model, init_resource)
        fp8_collective.assert_not_called()
        self.assertIsNone(model._fp8_collective)

    def test_bf16_draft_initializes_eligible_prefill_collective(self):
        model = self._bf16_mtp_model(device="cuda")
        init_resource = SimpleNamespace(kv_cache=None, is_decode_role=False)
        group = object()
        with (
            patch(
                "rtp_llm.models_py.model_desc.module_base.GptModelBase.initialize",
                return_value=True,
            ),
            patch(
                "rtp_llm.models_py.modules.kimi_k3.kernel_jit_warmup.warmup_kimi_k3_kernel_jit",
            ),
            patch("torch.cuda.get_device_capability", return_value=(10, 3)),
            patch(
                "rtp_llm.models_py.model_desc.kimi_k3._get_group",
                return_value=group,
            ),
            patch(
                "rtp_llm.models_py.modules.factory.linear.impl.cuda.bf16_comm_fusion.Bf16CollectiveProjection",
            ) as bf16_collective,
        ):
            self.assertTrue(model.initialize(init_resource))
        attention = model.layers[0].attention
        bf16_collective.assert_called_once_with(
            group,
            attention.input.weight.device,
            max_m=64,
            hidden_size=8,
            output_projections=[attention.output],
        )
        self.assertIs(attention._bf16_collective, bf16_collective.return_value)
        self.assertIsNone(model._fp8_collective)

    def test_local_projection_matches_full_projection_slice(self):
        for rank in range(4):
            with self.subTest(tp_rank=rank):
                self._assert_local_projection_matches_full_projection_slice(rank)

    def _assert_local_projection_matches_full_projection_slice(self, rank):
        torch.manual_seed(20260928)
        model = object.__new__(KimiK3MtpModel)
        nn.Module.__init__(model)
        model.tp_size = 4
        model.tp_rank = rank
        model.kv_cache = None
        full_embedding = nn.Embedding(128, 16)
        local_width = full_embedding.weight.shape[1] // model.tp_size
        local_weight = (
            full_embedding.weight.detach()
            .narrow(1, model.tp_rank * local_width, local_width)
            .contiguous()
        )
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
            multimodal_inputs=SimpleNamespace(multimodal_features=[]),
            attention_inputs={"default": SimpleNamespace(valid_token_mask=None)},
        )
        embedding = full_embedding(inputs.input_ids)
        full = model.eh_proj(
            torch.cat(
                (
                    model.enorm(embedding),
                    model.hnorm(inputs.input_hiddens),
                ),
                dim=-1,
            )
        )
        expected = full.narrow(0, rank * 8, 8)
        projected_rows = []
        hook = model.eh_proj.register_forward_pre_hook(
            lambda _module, args: projected_rows.append(args[0].shape[0])
        )

        def embedding_kernel(output, ids, weight, *_):
            output.copy_(torch.nn.functional.embedding(ids, weight))

        def gather_hidden_shards(local):
            torch.testing.assert_close(
                local,
                embedding.narrow(1, model.tp_rank * local_width, local_width),
                rtol=0,
                atol=0,
            )
            return torch.cat(embedding.split(local_width, dim=1), dim=0)

        try:
            with (
                patch(
                    "rtp_llm.models_py.modules.base.common.embedding.rtp_llm_ops.embedding",
                    side_effect=embedding_kernel,
                ),
                patch.object(
                    model, "_gather_modeling_boundary", side_effect=gather_hidden_shards
                ) as gather,
            ):
                actual = model._project_local_mtp_input(inputs)
                gather.assert_called_once()
        finally:
            hook.remove()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(projected_rows, [8])


if __name__ == "__main__":
    unittest.main()
