import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer, GenericMoeModel
from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel
from rtp_llm.models_py.model_desc.glm5 import (
    DecodeLayerOutput,
    Glm5MoeLayer,
    Glm5MoeModel,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import MlaAttention
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_runtime import (
    MegaMoeWrapper,
    SharedExpertMegaMoeWrapper,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.prepared_mega_moe import (
    PreparedMegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.prepared_mega_moe import (
    PreparedMegaMoeRouter,
)
from rtp_llm.models_py.modules.hybrid.glm5_cmp import Glm5Cmp
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W


class RecordingLayer(nn.Module):
    def __init__(self, has_indexer, step):
        super().__init__()
        self.self_attn = MlaAttention.__new__(MlaAttention)
        nn.Module.__init__(self.self_attn)
        self.self_attn.has_indexer = has_indexer
        self.step = step
        self.calls = []

    def forward(self, hidden, residual, fmha, **kwargs):
        self.calls.append(kwargs)
        return DecodeLayerOutput(
            hidden + self.step,
            residual + self.step,
            torch.full((hidden.shape[0], 2), self.step, dtype=torch.int32),
        )


class Glm5FmhaDispatchTest(unittest.TestCase):
    def test_factory_fallback_respects_shared_expert_ownership(self):
        for gated in (False, True):
            with self.subTest(gated=gated):
                quant = object()
                config = SimpleNamespace(
                    model_type="glm_5",
                    hidden_size=4,
                    inter_size=8,
                    expert_num=8,
                    eplb_config=SimpleNamespace(phy_exp_num=lambda n: n),
                    moe_k=2,
                    quant_config=quant,
                    moe_style=2,
                    activation_type="silu",
                )
                parallel = SimpleNamespace(
                    role_type=RoleType.DECODE,
                    get_ffn_tp_size=lambda: 1,
                    ep_size=2,
                )
                weights = {W.shared_expert_gate: torch.ones(4, 1)} if gated else {}
                adapter = SimpleNamespace(has_shared_expert_gate=False)
                backend = Mock(
                    includes_shared_expert=not gated,
                    expert_num=8,
                    topk_ids_dtype=torch.int64,
                )

                # The main factory consumes shared weights only for fused strategies.
                def create(selected, layer_weights):
                    self.assertEqual(selected.has_shared_expert_gate, gated)
                    return backend

                package = "rtp_llm.models_py.model_desc.generic_moe."
                with (
                    patch(
                        package + "LinearFactory.create_linear_from_weights"
                    ) as linear,
                    patch(package + "SelectTopk"),
                    patch(package + "MoEConfigAdapter", return_value=adapter),
                    patch(package + "FusedMoeFactory") as factory,
                    patch(package + "DenseMLP") as shared,
                ):
                    factory.return_value.create_fused_moe.side_effect = create
                    layer = Glm5MoeLayer(
                        config,
                        parallel,
                        weights,
                        SimpleNamespace(fake_balance_expert=False, moe_strategy="auto"),
                    )
                    self.assertEqual(layer.fused_moe.includes_shared_expert, not gated)
                    self.assertEqual(layer.add_shared_expert, gated)
                    self.assertEqual(layer.shared_expert is None, not gated)
                    self.assertEqual(shared.call_count, int(gated))
                    if gated:
                        self.assertIs(linear.call_args.kwargs["quant_config"], quant)
                        self.assertIsNone(linear.call_args.kwargs["hw_kernel_config"])

    def test_mtp_indexer_sharing_is_selected_by_model_capability(self):
        from rtp_llm.models_py.model_desc.generic_moe_mtp import (
            _mtp_indexer_share_active,
        )

        model = SimpleNamespace(index_share_for_mtp_iteration=True)
        parallel = SimpleNamespace(
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: False)
        )
        self.assertTrue(_mtp_indexer_share_active(model, parallel, 1, 512))
        self.assertFalse(_mtp_indexer_share_active(model, parallel, 2, 512))
        self.assertFalse(_mtp_indexer_share_active(model, parallel, 1, 0))
        parallel.prefill_cp_config.is_enabled = lambda: True
        self.assertFalse(_mtp_indexer_share_active(model, parallel, 1, 512))
        parallel.prefill_cp_config.is_enabled = lambda: False
        model.index_share_for_mtp_iteration = False
        self.assertFalse(_mtp_indexer_share_active(model, parallel, 1, 512))

    def make_moe_layer(self, *, fused_shared=False, cp_prefill=False, ffn_tp_size=1):
        config = SimpleNamespace(
            model_type="glm_5",
            swiglu_limit=0.0,
            hidden_size=4,
            inter_size=8,
            expert_num=256,
            moe_k=8,
            moe_style=2,
            quant_config=None,
            activation_type="silu",
            moe_n_group=1,
            moe_topk_group=1,
            has_moe_norm=True,
            routed_scaling_factor=2.5,
            eplb_config=SimpleNamespace(phy_exp_num=lambda n: n),
        )
        parallel = SimpleNamespace(
            role_type=RoleType.PREFILL if cp_prefill else RoleType.DECODE,
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: cp_prefill),
            get_ffn_tp_size=lambda: ffn_tp_size,
            ep_size=2,
        )
        backend_type = SharedExpertMegaMoeWrapper if fused_shared else MegaMoeWrapper
        backend = backend_type.__new__(backend_type)
        nn.Module.__init__(backend)
        kernel = Mock(return_value=torch.full((3, 4), 2.0))
        kernel._mega_buf = SimpleNamespace(num_max_tokens_per_rank=8)
        kernel.forward = Mock(return_value=torch.full((3, 4), 2.0))
        kernel.forward_prepacked = Mock(return_value=torch.full((3, 4), 2.0))
        backend.mega_moe = kernel
        backend.expert_num = 256
        executor = object.__new__(PreparedMegaMoeExecutor)
        executor.runtime = backend
        executor.includes_shared_expert = fused_shared
        backend = FusedMoe(
            PreparedMegaMoeRouter(SimpleNamespace(tp_size=1), FusedMoEQuantConfig()),
            executor,
            256,
            "mega_moe",
        )
        package = "rtp_llm.models_py.model_desc.generic_moe."
        with (
            patch(
                package + "LinearFactory.create_linear_from_weights",
                return_value=Mock(),
            ),
            patch(package + "SelectTopk"),
            patch(package + "GroupTopK", return_value=Mock()),
            patch(
                package + "DenseMLP", return_value=Mock(return_value=torch.ones(3, 4))
            ),
            patch(
                "rtp_llm.models_py.model_desc.glm5.FusedMoeFactory.create_fused_moe",
                return_value=backend,
            ),
            patch(
                "rtp_llm.models_py.model_desc.glm5.MoEConfigAdapter",
                return_value=SimpleNamespace(),
            ),
        ):
            layer = Glm5MoeLayer(
                config,
                parallel,
                {W.e_score_correction_b: torch.zeros(256)},
                SimpleNamespace(fake_balance_expert=False, moe_strategy="mega_moe"),
                layer_idx=7,
            )
        self.assertIsInstance(layer, GenericMoeLayer)
        return layer

    def test_common_routing_preserves_fixed_gate_rows_and_fp32_scores(self):
        layer = self.make_moe_layer(cp_prefill=True)
        hidden = torch.ones(3, 4)
        logits = torch.ones(3, 256, dtype=torch.bfloat16)
        with patch(
            "rtp_llm.models_py.model_desc.glm5.fixed_m_linear", return_value=logits
        ) as linear:
            ids, weights = layer._route(hidden)
        linear.assert_called_once_with(layer.gate, hidden, 8192)
        self.assertEqual(ids.dtype, torch.int64)
        self.assertEqual(weights.dtype, torch.float32)
        topk_args = layer.group_topk.call_args.kwargs
        self.assertEqual(topk_args["scores"].dtype, torch.float32)
        self.assertEqual(topk_args["routed_scaling_factor"], 2.5)
        with patch(
            "rtp_llm.models_py.model_desc.glm5.fixed_m_linear",
            side_effect=AssertionError("empty gate must not pad"),
        ):
            layer.gate.return_value = torch.empty(0, 256, dtype=torch.bfloat16)
            self.assertEqual(layer._gate_logits(hidden[:0]).shape, (0, 256))

    def test_common_shared_merge_forwards_quant_inputs_and_reduces_only_shared(self):
        layer = self.make_moe_layer(ffn_tp_size=2)
        layer.gate.return_value = torch.zeros(3, 256, dtype=torch.bfloat16)
        hidden = torch.zeros(3, 4)
        quant = torch.ones(3, 4)
        scale = torch.ones(3, 1)
        with patch(
            "rtp_llm.models_py.model_desc.generic_moe.all_reduce",
            side_effect=lambda x, group: x * 3,
        ) as reduce:
            result = layer(hidden, x_fp8=quant, x_scale=scale)
        torch.testing.assert_close(result, torch.full((3, 4), 5.0))
        reduce.assert_called_once()
        torch.testing.assert_close(reduce.call_args.args[0], torch.ones(3, 4))
        self.assertIs(layer.shared_expert.call_args.kwargs["x_fp8"], quant)
        self.assertIs(layer.shared_expert.call_args.kwargs["x_scale"], scale)
        self.assertTrue(layer.shared_expert.call_args.kwargs["skip_allreduce"])

    def test_fused_shared_prepack_has_no_placeholder_or_double_execution(self):
        layer = self.make_moe_layer(fused_shared=True)
        self.assertIsNone(layer.shared_expert)
        cmp = Glm5Cmp.__new__(Glm5Cmp)
        cmp.config, cmp.mlp = layer.config, layer
        cmp._is_moe_layer, cmp._router_weight = True, torch.ones(256, 4)
        self.assertIsNone(cmp._static_moe_prepack_disabled_reason())
        result = layer.forward_prepacked(
            torch.zeros(3, 4), torch.zeros(3, 8), torch.ones(3, 8)
        )
        torch.testing.assert_close(result, torch.full((3, 4), 2.0))
        layer.fused_moe.fused_experts.runtime.mega_moe.forward_prepacked.assert_called_once()

    def test_graph_clone_preserves_common_routing_and_collective_state(self):
        layer = self.make_moe_layer(ffn_tp_size=2)
        backend = self.make_moe_layer().fused_moe
        with patch.object(
            layer.fused_moe, "clone_for_cuda_graph", return_value=backend
        ) as clone_backend:
            clone = layer.clone_for_cuda_graph(share_mega_buf=True)
        clone_backend.assert_called_once_with(share_mega_buf=True)
        self.assertIs(clone.fused_moe, backend)
        self.assertIsNot(clone.fused_moe, layer.fused_moe)
        self.assertIs(clone.group_topk, layer.group_topk)
        self.assertEqual(clone.layer_idx, 7)
        self.assertTrue(clone.use_ep_shared_allreduce)
        self.assertFalse(clone.use_unified_tp_allreduce)

    def test_specialized_models_delegate_main_fmha_with_valid_super_dispatch(self):
        sentinel = object()
        for model_type in (Glm5MoeModel, GenericMoeMTPModel):
            with self.subTest(model=model_type.__name__):
                self.assertTrue(issubclass(model_type, GenericMoeModel))
                model = model_type.__new__(model_type)
                nn.Module.__init__(model)
                model.config = SimpleNamespace(
                    attn_config=SimpleNamespace(use_mla=False, is_sparse=False)
                )
                inputs = SimpleNamespace()
                with patch.object(
                    GptModelBase, "prepare_fmha_impl", return_value=sentinel
                ) as prepare:
                    self.assertIs(
                        model.prepare_fmha_impl(inputs, True, "legacy"), sentinel
                    )
                    prepare.assert_called_once_with(inputs, True, "legacy")

    def test_shared_loop_routes_only_owned_groups_and_carries_topk(self):
        for model_type in (Glm5MoeModel, GenericMoeMTPModel):
            with self.subTest(model=model_type.__name__):
                model = model_type.__new__(model_type)
                nn.Module.__init__(model)
                model.layers = nn.ModuleList(
                    [RecordingLayer(True, 1), RecordingLayer(False, 2)]
                )
                model.layer_num = 2
                model.kv_cache = SimpleNamespace(
                    get_layer_cache_groups=lambda idx: [
                        SimpleNamespace(tag=tag)
                        for tag in (
                            ("default", "indexer_kv") if idx == 0 else ("default",)
                        )
                    ]
                )
                hidden = torch.zeros(3, 4)
                with patch(
                    "rtp_llm.models_py.model_desc.glm5.should_enable_glm5_cmp",
                    return_value=False,
                ):
                    result = model._forward_decoder_layers(
                        hidden, {"default": object(), "indexer_kv": object()}
                    )
                self.assertTrue(torch.equal(result.hidden_states, hidden + 3))
                self.assertTrue(torch.equal(result.residual, hidden + 3))
                self.assertEqual(
                    set(model.layers[0].calls[0]["kv_cache"]), {"default", "indexer_kv"}
                )
                self.assertEqual(set(model.layers[1].calls[0]["kv_cache"]), {"default"})
                self.assertIsNone(model.layers[0].calls[0]["prev_topk_indices"])
                self.assertEqual(
                    model.layers[1].calls[0]["prev_topk_indices"].tolist(), [[1, 1]] * 3
                )
                self.assertEqual(result.topk_indices.tolist(), [[2, 2]] * 3)

    def test_shared_loop_preserves_draft_reuse_and_cacheless_warmup(self):
        model = GenericMoeMTPModel.__new__(GenericMoeMTPModel)
        nn.Module.__init__(model)
        model.layers = nn.ModuleList([RecordingLayer(True, 1)])
        model.layer_num = 1
        model.kv_cache = None
        seed = torch.ones(3, 2, dtype=torch.int32)
        with patch(
            "rtp_llm.models_py.model_desc.glm5.should_enable_glm5_cmp",
            return_value=False,
        ) as select_cmp:
            model._forward_decoder_layers(
                torch.zeros(3, 4), object(), seed, force_reuse_topk_indices=True
            )
        call = model.layers[0].calls[0]
        self.assertIs(call["prev_topk_indices"], seed)
        self.assertTrue(call["force_reuse_topk_indices"])
        self.assertIsNone(call["kv_cache"])
        self.assertTrue(select_cmp.call_args.kwargs["force_reuse_topk_indices"])


if __name__ == "__main__":
    unittest.main()
