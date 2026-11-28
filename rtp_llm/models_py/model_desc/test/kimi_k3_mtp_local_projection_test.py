import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpModel
from rtp_llm.models_py.modules import Embedding, RMSNorm
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.modules.hybrid.query_layout import QueryLayout
from rtp_llm.models_py.modules.kimi_k3.attention import KimiK3MLA
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
                Fp8BlockWiseQuantConfig() if dtype == torch.float8_e4m3fn else None
            ),
        )
        model._max_moe_tokens = 262144
        model._projection_staging_tokens = model._max_moe_tokens
        model._fp8_collective = None
        model._bf16_collective_decided = False
        attention = object.__new__(KimiK3MLA)
        nn.Module.__init__(attention)
        attention._fp8_collective = None
        attention._fp8_input_collective = None
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
                self.assertIs(
                    model.layers[0].attention._fp8_input_collective,
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
                self.assertIsNone(model.layers[0].attention._fp8_collective)
                self.assertIsNone(model.layers[0].attention._fp8_input_collective)
        collective.assert_not_called()

    def test_fp8_target_rejects_mixed_projection_precision(self):
        model = self._target_model(torch.float8_e4m3fn, RoleType.PREFILL)
        model.layers[0].attention.output = CudaF16Linear(
            torch.empty((1, 1), dtype=torch.bfloat16)
        )
        with self.assertRaisesRegex(ValueError, "grouped E4M3 projections"):
            model._initialize_fp8_collective()

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
                            is_mtp=False,
                            k3_runtime_config=SimpleNamespace(
                                attn_res_block_size=0, dsa=None
                            ),
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
                            prefill_cp_config=ParallelismConfig().prefill_cp_config,
                        )
                        weights = SimpleNamespace(
                            weights=[{}],
                            get_global_weight=lambda _: torch.ones(1),
                        )
                        layer = nn.Identity()
                        layer.layer_type = object()
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

    def _bf16_mtp_model(self, device="cpu"):
        model = object.__new__(KimiK3MtpModel)
        nn.Module.__init__(model)
        model.tp_size = 8
        parallel = ParallelismConfig()
        parallel.world_size = parallel.local_world_size = parallel.tp_size = 8
        parallel.role_type = RoleType.PREFILL
        model.parallelism_config = parallel
        model.config = SimpleNamespace(
            hidden_size=8,
            is_mtp=True,
            k3_runtime_config=SimpleNamespace(attn_res_block_size=0, dsa=None),
        )
        model._max_moe_tokens = 64
        model._projection_staging_tokens = model._max_moe_tokens
        model._fp8_collective = None
        model._bf16_collective_decided = False
        model.prefill_query_cp = False
        model.kv_cache = None
        model._setup_dsa()
        model.register_buffer("_mtp_prefill_last_hidden_buffer", None, persistent=False)
        model._mtp_prefill_last_hidden_valid_rows = 0
        attention = nn.Module()
        attention.input = CudaF16Linear(
            torch.ones(8, 8, dtype=torch.bfloat16, device=device)
        )
        attention.output = CudaF16Linear(
            torch.ones(8, 8, dtype=torch.bfloat16, device=device)
        )
        attention._bf16_collective = None
        attention._fp8_collective = None
        attention._fp8_input_collective = None
        attention._mtp_bf16_collectives = None
        layer = nn.Module()
        layer.attention = attention
        layer.block_size = 0
        model.layers = nn.ModuleList([layer])
        return model

    def test_fp8_collective_does_not_initialize_bf16_draft(self):
        model = self._bf16_mtp_model()
        init_resource = SimpleNamespace(
            kv_cache=None, is_decode_role=False, is_speculative=True
        )
        group = object()
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
            patch(
                "rtp_llm.models_py.model_desc.kimi_k3._get_group", return_value=group
            ),
            patch(
                "rtp_llm.models_py.model_desc.kimi_k3.agree_on_optional_collective",
                return_value=False,
            ) as agree,
        ):
            self.assertTrue(model.initialize(init_resource))
            model._initialize_bf16_collective(init_resource)
        warmup.assert_called_once_with(model, init_resource)
        fp8_collective.assert_not_called()
        agree.assert_called_once_with(group, False, "k3_bf16_projection")
        self.assertIsNone(model._fp8_collective)

    def test_bf16_unsupported_fusion_topology_keeps_regular_projections(self):
        model = self._bf16_mtp_model(device="cuda")
        model.tp_size = 6
        model._projection_staging_tokens = 66
        with (
            patch("torch.cuda.get_device_capability", return_value=(10, 3)),
            patch("rtp_llm.models_py.model_desc.kimi_k3._get_group"),
            patch(
                "rtp_llm.models_py.modules.factory.linear.impl.cuda.bf16_comm_fusion.Bf16CollectiveProjection",
            ) as collective,
        ):
            model._initialize_bf16_collective(SimpleNamespace(is_decode_role=False))
        collective.assert_not_called()
        self.assertIsNone(model.layers[0].attention._bf16_collective)

    def test_bf16_draft_initializes_eligible_prefill_collective(self):
        model = self._bf16_mtp_model(device="cuda")
        init_resource = SimpleNamespace(
            kv_cache=None, is_decode_role=False, is_speculative=True
        )
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
                "rtp_llm.models_py.model_desc.kimi_k3.agree_on_optional_collective",
                return_value=True,
            ) as exchange,
            patch(
                "rtp_llm.models_py.modules.factory.linear.impl.cuda.bf16_comm_fusion.Bf16CollectiveProjection",
            ) as bf16_collective,
        ):
            self.assertTrue(model.initialize(init_resource))
        attention = model.layers[0].attention
        exchange.assert_called_once_with(group, True, "k3_bf16_projection")
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
        model._query_layout = None
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

    def test_local_embedding_owns_only_rank_local_storage(self):
        for rank in range(4):
            with self.subTest(tp_rank=rank):
                model = object.__new__(KimiK3MtpModel)
                nn.Module.__init__(model)
                model.tp_size = 4
                model.tp_rank = rank
                model._query_layout = None
                model.enorm = nn.Identity()
                model.hnorm = nn.Identity()
                model.eh_proj = nn.Identity()
                embedded = torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16)
                inputs = SimpleNamespace(
                    input_ids=torch.arange(32), input_hiddens=torch.zeros_like(embedded)
                )
                local_embeddings = []
                hook = model.enorm.register_forward_pre_hook(
                    lambda _module, args: local_embeddings.append(args[0])
                )
                try:
                    with patch.object(model, "_embed_inputs", return_value=embedded):
                        model._project_local_mtp_input(inputs)
                finally:
                    hook.remove()
                local = local_embeddings[0]
                torch.testing.assert_close(local, embedded.narrow(0, rank * 8, 8))
                self.assertNotEqual(
                    local.untyped_storage().data_ptr(),
                    embedded.untyped_storage().data_ptr(),
                )
                self.assertEqual(
                    local.untyped_storage().nbytes(),
                    local.numel() * local.element_size(),
                )

    def _position_zero_projection_model(self, parallel):
        """Construct the real projection stack without unrelated decoder layers."""
        hidden_size = 16
        full_embedding = (
            torch.arange(32 * hidden_size, device="cuda", dtype=torch.float32)
            .reshape(32, hidden_size)
            .add_(1)
            .div_(hidden_size)
            .bfloat16()
        )
        config = SimpleNamespace(
            num_layers=0,
            vocab_size=32,
            hidden_size=hidden_size,
            is_mtp=True,
            layernorm_eps=1e-6,
            capture_aux_hidden_layer_ids=[],
            k3_runtime_config=SimpleNamespace(attn_res_block_size=0, dsa=None),
        )
        shard_width = hidden_size // parallel.tp_size
        weights = {
            W.embedding: full_embedding[
                :, parallel.tp_rank * shard_width : (parallel.tp_rank + 1) * shard_width
            ].contiguous(),
            W.final_ln_gamma: torch.ones(
                hidden_size, dtype=torch.bfloat16, device="cuda"
            ),
        }
        model = object.__new__(KimiK3MtpModel)
        KimiK3Model.__init__(
            model,
            config,
            parallel,
            SimpleNamespace(weights=[], get_global_weight=weights.__getitem__),
            2,
            max_moe_tokens=8,
        )
        model.enorm = RMSNorm(weights[W.final_ln_gamma], config.layernorm_eps)
        model.hnorm = RMSNorm(weights[W.final_ln_gamma], config.layernorm_eps)
        # Select only the embedding contribution so zeroing it cannot be hidden
        # by the target hidden-state half of the actual MTP projection.
        model.eh_proj = CudaF16Linear(
            torch.cat(
                (
                    torch.eye(hidden_size, dtype=torch.bfloat16, device="cuda"),
                    torch.zeros(
                        (hidden_size, hidden_size), dtype=torch.bfloat16, device="cuda"
                    ),
                ),
                dim=0,
            )
        )
        return model, full_embedding

    def _k31_query_cp_target_parallel(self):
        from rtp_llm.models.kimi_k3.kimi_k31 import KimiK31ModelConfig

        parallel = ParallelismConfig()
        parallel.tp_size = parallel.world_size = parallel.local_world_size = 2
        parallel.ep_size = 2
        parallel.role_type = RoleType.PREFILL
        parallel.prefill_cp_config.kv_cache_sharded = True
        KimiK31ModelConfig().configure_parallelism(parallel)
        return parallel

    def test_query_cp_projection_preserves_valid_position_zero(self):
        """Exercise the helper's CP branch, independently of K3.1 draft routing."""
        self.assertTrue(torch.cuda.is_available(), "requires the CUDA CI worker")
        parallel = ParallelismConfig()
        parallel.tp_size = parallel.world_size = parallel.local_world_size = 2
        parallel.ep_size = 2
        parallel.role_type = RoleType.PREFILL
        model, full_embedding = self._position_zero_projection_model(parallel)
        model.embed_tokens = Embedding(
            model.config, parallel, full_embedding, replicated=True
        )
        # A three-token request is padded to four for CP2. The actual Zigzag
        # inverse permutation gives rank 0 natural row 0 and the padding row.
        info = SimpleNamespace(
            prefill_actual_input_lengths_cpu=torch.tensor([3], dtype=torch.int32),
            prefill_prefix_lengths_cpu=torch.tensor([0], dtype=torch.int32),
            prefill_qkv_restore_indice=torch.tensor(
                [0, 2, 3, 1], dtype=torch.int32, device="cuda"
            ),
            prefill_qkv_padding_mask=torch.tensor(
                [True, True, True, False], device="cuda"
            ),
        )
        layout = QueryLayout(info, 2, 0, "cuda")
        self.assertEqual(layout.local_positions.tolist(), [0, -1])
        self.assertEqual(layout.local_valid.tolist(), [True, False])
        model._query_layout = layout
        inputs = SimpleNamespace(
            input_ids=torch.tensor([3, 0], dtype=torch.int32, device="cuda"),
            # The production text gatherer omits explicit positions. Use the
            # real QueryLayout fallback, including its invalid-padding marker.
            combo_position_ids=torch.empty(0, dtype=torch.int32, device="cuda"),
            input_hiddens=torch.zeros((2, 16), dtype=torch.bfloat16, device="cuda"),
        )
        embedding = torch.nn.functional.embedding(inputs.input_ids, full_embedding)
        embedding = embedding * layout.local_valid[:, None]
        expected = model.eh_proj(
            torch.cat(
                (model.enorm(embedding), model.hnorm(inputs.input_hiddens)), dim=-1
            )
        )
        self.assertGreater(torch.count_nonzero(expected[0]).item(), 0)
        actual = model._project_local_mtp_input(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_k31_query_cp_target_routes_draft_position_zero_through_global_sp(self):
        """The supported K3.1 draft configuration must bypass the CP helper branch."""
        from rtp_llm.models.kimi_k3.kimi_k31 import KimiK31Mtp

        self.assertTrue(torch.cuda.is_available(), "requires the CUDA CI worker")
        target = self._k31_query_cp_target_parallel()
        self.assertTrue(target.prefill_cp_config.is_enabled())
        self.assertTrue(target.prefill_cp_config.preserve_tp_layout)
        draft = KimiK31Mtp.speculative_parallelism_config(target)
        self.assertFalse(draft.prefill_cp_config.is_enabled())
        self.assertFalse(draft.prefill_cp_config.preserve_tp_layout)
        self.assertTrue(draft.prefill_cp_config.kv_cache_sharded)
        model, full_embedding = self._position_zero_projection_model(draft)
        self.assertFalse(model.prefill_query_cp)
        self.assertIsNone(model.dsa_workspace)
        self.assertIsNone(model._query_layout)
        self.assertTrue(model.requires_global_mtp_prefill_hidden)
        inputs = SimpleNamespace(
            input_ids=torch.tensor([3, 4, 5, 0], dtype=torch.int32, device="cuda"),
            combo_position_ids=torch.tensor(
                [0, 1, 2, 0], dtype=torch.int32, device="cuda"
            ),
            input_hiddens=torch.zeros((4, 16), dtype=torch.bfloat16, device="cuda"),
            multimodal_inputs=SimpleNamespace(multimodal_features=[]),
        )
        embedding = torch.nn.functional.embedding(inputs.input_ids, full_embedding)
        expected = model.eh_proj(
            torch.cat(
                (model.enorm(embedding), model.hnorm(inputs.input_hiddens)), dim=-1
            )
        )[:2]

        def gather_hidden_shards(local):
            # Only replace the cross-rank transport on this one-GPU target;
            # embedding, SP slicing, normalization and projection run normally.
            torch.testing.assert_close(local, embedding[:, :8], rtol=0, atol=0)
            return torch.cat(embedding.split(8, dim=1), dim=0)

        with patch.object(
            model, "_gather_modeling_boundary", side_effect=gather_hidden_shards
        ) as gather:
            actual = model._project_local_mtp_input(inputs)
        gather.assert_called_once()
        self.assertGreater(torch.count_nonzero(actual[0]).item(), 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0.015625)


if __name__ == "__main__":
    unittest.main()
