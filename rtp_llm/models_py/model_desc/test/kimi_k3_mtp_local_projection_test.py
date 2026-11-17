import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpModel
from rtp_llm.models_py.modules import Embedding
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.ops import ParallelismConfig, RoleType
from rtp_llm.utils.model_weight import W


class KimiK3MtpLocalProjectionTest(unittest.TestCase):
    def _target_model(self, dtype, role, tp_size=4):
        model = object.__new__(KimiK3Model)
        nn.Module.__init__(model)
        model.tp_size = tp_size
        model.parallelism_config = SimpleNamespace(role_type=role)
        model.config = SimpleNamespace(
            hidden_size=7168,
            k3_runtime_config=SimpleNamespace(attn_res_block_size=0),
            attention_projection_quant_config=(
                SimpleNamespace(get_method=lambda: "FP8_PER_BLOCK")
                if dtype == torch.float8_e4m3fn
                else None
            ),
        )
        model._max_moe_tokens = 262144
        model._projection_staging_tokens = 262144
        model._fp8_collective = None
        attention = nn.Module()
        for name in ("input", "output"):
            setattr(
                attention,
                name,
                SimpleNamespace(
                    weight=torch.empty(1, dtype=dtype),
                    scale_ue8m0=dtype == torch.float8_e4m3fn,
                ),
            )
        layer = nn.Module()
        layer.attention = attention
        layer.block_size = 0
        layer.mlp = nn.Identity()
        model.layers = nn.ModuleList([layer])
        return model

    def test_fp8_target_automatically_initializes_both_collectives_at_resolved_capacity(
        self,
    ):
        for role in (RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION):
            with self.subTest(role=role):
                model = self._target_model(torch.float8_e4m3fn, role)
                group = object()
                with (
                    patch(
                        "rtp_llm.models_py.model_desc.kimi_k3._get_group",
                        return_value=group,
                    ),
                    patch(
                        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_comm_fusion.Fp8CollectiveProjection"
                    ) as collective,
                ):
                    model._initialize_fp8_collective()
                    # Repeated model initialization must retain captured workspaces.
                    model._initialize_fp8_collective()
                collective.assert_called_once_with(
                    group,
                    model.layers[0].attention.input.weight.device,
                    max_m=262144,
                    hidden_size=7168,
                    enable_ag=True,
                    enable_rs=True,
                    decode_staging=role == RoleType.DECODE,
                )
                self.assertIs(model._fp8_collective, collective.return_value)
                self.assertIs(
                    model.layers[0].attention._fp8_collective,
                    collective.return_value,
                )

    def test_bf16_target_and_single_rank_do_not_initialize_fp8_collectives(self):
        models = (
            self._target_model(torch.bfloat16, RoleType.PREFILL),
            self._target_model(torch.float8_e4m3fn, RoleType.PREFILL, tp_size=1),
        )
        with patch(
            "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_comm_fusion.Fp8CollectiveProjection"
        ) as collective:
            for model in models:
                model._initialize_fp8_collective()
                self.assertIsNone(model._fp8_collective)
        collective.assert_not_called()

    def test_fp8_target_rejects_mixed_projection_precision(self):
        model = self._target_model(torch.float8_e4m3fn, RoleType.PREFILL)
        model.layers[0].attention.output.weight = torch.empty(1, dtype=torch.bfloat16)
        model.layers[0].attention.output.scale_ue8m0 = False
        with self.assertRaisesRegex(ValueError, "grouped E4M3 projections"):
            model._initialize_fp8_collective()

    def _constructor_fixture(self, tp, dp, role):
        config = SimpleNamespace(
            num_layers=1,
            is_mtp=False,
            vocab_size=16,
            capture_aux_hidden_layer_ids=[],
            hidden_state_capture_layer_ids=[],
            moe_prefill_max_tokens_per_rank=262144,
            max_seq_len=1048576,
            gen_num_per_cycle=2,
            k3_runtime_config=SimpleNamespace(
                attn_res_block_size=0, media_placeholder_token_id=None
            ),
            layernorm_eps=1e-5,
        )
        parallel = ParallelismConfig()
        parallel.role_type = role
        parallel.tp_size, parallel.dp_size, parallel.ep_size = tp, dp, tp * dp
        parallel.ffn_sp_size, parallel.ffn_tp_size = tp, 1
        parallel.enable_sp = tp > 1
        weights = SimpleNamespace(
            weights=[
                {
                    W.mla_kv_b_w: torch.ones(1, 1),
                    "kimi_k3.mtp.enorm": torch.ones(1),
                    "kimi_k3.mtp.hnorm": torch.ones(1),
                }
            ],
            get_global_weight=lambda _: torch.ones(1),
        )
        layer = nn.Identity()
        layer.layer_type = object()
        return config, parallel, weights, layer

    def test_moe_workspace_uses_resolved_k3_capacity_for_every_role(self):
        from rtp_llm.models_py.model_desc import kimi_k3

        for tp in (4, 8):
            for role in (RoleType.DECODE, RoleType.PREFILL, RoleType.PDFUSION):
                for capacity in (192, 4096, 262144):
                    with self.subTest(tp=tp, role=role, capacity=capacity):
                        config, parallel, weights, layer = self._constructor_fixture(
                            tp, 8 // tp, role
                        )
                        with (
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

    def test_ffn_sp_contract_for_target_and_native_mtp_after_config_hook(self):
        from rtp_llm.models.kimi_k3.kimi_k31 import KimiK31Mtp
        from rtp_llm.models_py.model_desc import kimi_k3, kimi_k3_mtp

        cases = (
            (1, 1, 1, 1, True),
            (8, 1, 8, 1, True),
            (4, 2, 4, 1, True),
            (8, 1, 4, 2, False),
            (4, 2, 2, 2, False),
            (4, 2, 4, 2, False),
            (1, 1, 1, 2, False),
        )
        for role in (RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION):
            for native_mtp in (False, True):
                for tp, dp, ffn_sp, ffn_tp, allowed in cases:
                    with self.subTest(
                        role=role,
                        native_mtp=native_mtp,
                        tp=tp,
                        dp=dp,
                        ffn_sp=ffn_sp,
                        ffn_tp=ffn_tp,
                    ):
                        config, parallel, weights, layer = self._constructor_fixture(
                            tp, dp, role
                        )
                        parallel.ffn_sp_size, parallel.ffn_tp_size = ffn_sp, ffn_tp
                        model_cls = KimiK3Model
                        if native_mtp:
                            parallel = KimiK31Mtp.speculative_parallelism_config(
                                parallel
                            )
                            model_cls = KimiK3MtpModel
                            self.assertFalse(parallel.prefill_cp_config.is_enabled())
                        self.assertEqual(parallel.ffn_sp_size, ffn_sp)
                        self.assertEqual(parallel.ffn_tp_size, ffn_tp)
                        with (
                            patch.object(
                                kimi_k3, "Embedding", return_value=nn.Identity()
                            ),
                            patch.object(
                                kimi_k3, "RMSNorm", return_value=nn.Identity()
                            ),
                            patch.object(
                                kimi_k3, "KimiK3DecoderLayer", return_value=layer
                            ) as ctor,
                            patch.object(
                                kimi_k3_mtp, "RMSNorm", return_value=nn.Identity()
                            ),
                            patch.object(
                                kimi_k3_mtp.LinearFactory,
                                "create_linear_from_weights",
                                return_value=nn.Identity(),
                            ),
                        ):
                            if allowed:
                                model = model_cls(
                                    config, parallel, weights, 63, max_moe_tokens=192
                                )
                                self.assertEqual(model.tp_size, tp)
                                self.assertEqual(ctor.call_args.args[-1], 192 // tp)
                            else:
                                with self.assertRaisesRegex(
                                    ValueError,
                                    f"tp_size={tp}, ffn_sp_size={ffn_sp}, ffn_tp_size={ffn_tp}",
                                ):
                                    model_cls(
                                        config,
                                        parallel,
                                        weights,
                                        63,
                                        max_moe_tokens=192,
                                    )
                                ctor.assert_not_called()

    def test_ffn_sp_guard_preserves_enable_sp_requirement(self):
        config, parallel, weights, _ = self._constructor_fixture(4, 2, RoleType.DECODE)
        parallel.enable_sp = False
        with self.assertRaisesRegex(ValueError, "requires sequence parallelism"):
            KimiK3Model(config, parallel, weights, 63, max_moe_tokens=192)

    def _bf16_mtp_model(self, device="cpu"):
        model = object.__new__(KimiK3MtpModel)
        nn.Module.__init__(model)
        model.tp_size = 8
        model.parallelism_config = SimpleNamespace(role_type=RoleType.PREFILL)
        model.config = SimpleNamespace(hidden_size=8)
        model._max_moe_tokens = 64
        model._projection_staging_tokens = 64
        model._fp8_collective = None
        model._outer_natural_enabled = False
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
        layer.block_size = 0
        model.layers = nn.ModuleList([layer])
        return model

    def test_fp8_collective_does_not_initialize_bf16_draft(self):
        model = self._bf16_mtp_model()
        init_resource = SimpleNamespace(kv_cache=None, is_decode_role=False)
        with (
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
            decode_staging=False,
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
