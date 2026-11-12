import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.eplb.test.test_support import make_layer
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models_py.model_desc.generic_moe import (
    GenericMoeDecoderLayer,
    GenericMoeLayer,
    GenericMoeModel,
)
from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel
from rtp_llm.models_py.modules.glm5_mega_moe import mega_moe_wrapper
from rtp_llm.models_py.modules.glm5_mega_moe.test_fused_moe_wrapper_layout import (
    MegaMoeWrapperLayoutTest,
    _config,
    _FakeMegaMoE,
    _parallelism,
)
from rtp_llm.models_py.utils.expert_load_monitor import ExpertLoadContext
from rtp_llm.utils.model_weight import W


class RoutedExperts(nn.Module):
    topk_ids_dtype = torch.int64

    def __init__(self, plan):
        super().__init__()
        self.weights = plan.weights[0]
        self.prepacked_ids = None

    def forward(self, hidden_states, topk_weights, topk_ids, activation):
        return (self.weights[topk_ids] * topk_weights.unsqueeze(-1)).sum(1)

    def forward_prepacked(self, hidden_states):
        return self.weights[self.prepacked_ids].sum(1)


class Glm5WrapperEplbTest(unittest.TestCase):
    def make_layer(self):
        layer = GenericMoeLayer.__new__(GenericMoeLayer)
        nn.Module.__init__(layer)
        plan = make_layer()
        layer.eplb = plan
        layer.expert_load_layer_id = 0
        layer._bound_expert_load = None
        layer._hy4_fp32_router = False
        layer._cmp_moe_streams = None
        layer.gate = nn.Identity()
        layer.gate_chunk_rows = 0
        layer.top_k = 1
        layer.correction_bias = None
        layer.fake_balance_expert = None
        layer.shared_expert = None
        layer.shared_expert_gate = None
        layer.ep_size = layer.ffn_tp_size = 1
        layer._use_mega_moe_fused_shared = False
        layer.fused_moe = RoutedExperts(plan)

        def select_topk(logits, ids, weights):
            ids.fill_(3)
            weights.fill_(1)

        layer.select_topk = select_topk
        return layer

    def test_regular_forward_routes_logical_ids_before_expert_execution(self):
        layer = self.make_layer()
        hidden = torch.zeros(4, 6)
        output = layer(hidden)
        expected = layer.eplb.weights[0][3].expand(4, -1)
        self.assertTrue(torch.equal(output, expected))

    def test_native_prefill_skips_legacy_gate_topk_and_remapping(self):
        layer = self.make_layer()
        hidden = torch.zeros(4, 6)
        physical = layer.eplb.map_ids(torch.zeros(4, 1, dtype=torch.int64))
        with patch.object(
            layer, "native_gate_supported", return_value=True
        ), patch.object(
            layer, "native_gate", return_value=(torch.ones(4, 1), physical)
        ), patch.object(
            layer.gate, "forward", side_effect=AssertionError("legacy gate")
        ), patch.object(
            layer, "select_topk", side_effect=AssertionError("separate topk")
        ), patch.object(
            layer.eplb, "map_ids", side_effect=AssertionError("double mapping")
        ):
            output = layer(hidden)
        torch.testing.assert_close(
            output, layer.eplb.weights[0][0].float().expand(4, -1)
        )

    def test_gate_pack_uses_logits_without_separate_topk_or_mapping(self):
        layer = self.make_layer()
        layer.config = SimpleNamespace(routed_scaling_factor=2.5)
        hidden = torch.zeros(4, 6, dtype=torch.bfloat16)
        layer.fused_moe.supports_gate_pack = lambda *args: True
        layer.fused_moe.forward_gate_pack = Mock(return_value=hidden)
        with patch.object(
            layer, "native_gate", side_effect=AssertionError("native gate")
        ), patch.object(
            layer, "select_topk", side_effect=AssertionError("separate TopK")
        ), patch.object(
            layer.eplb, "map_ids", side_effect=AssertionError("separate mapping")
        ):
            output = layer(hidden)
        self.assertIs(output, hidden)
        args = layer.fused_moe.forward_gate_pack.call_args.args
        self.assertIs(args[1], hidden)
        self.assertEqual(args[1].dtype, torch.bfloat16)
        self.assertIs(args[-1], layer.eplb)

    def test_gate_pack_chunks_keep_global_token_phase_and_own_outputs(self):
        wrapper = object.__new__(mega_moe_wrapper.MegaMoeWrapper)
        nn.Module.__init__(wrapper)
        wrapper._activation_clamp = None
        workspace = torch.empty(3, 6)
        calls = []
        plan = make_layer()

        def execute(x, weights, ids, **kwargs):
            self.assertIsNone(weights)
            self.assertIsNone(ids)
            self.assertIs(kwargs["eplb"], plan)
            self.assertFalse(kwargs["inputs_prepacked"])
            calls.append(kwargs["token_offset"])
            torch.testing.assert_close(kwargs["gate_logits"], x)
            workspace[: x.shape[0]].copy_(x)
            return workspace[: x.shape[0]]

        wrapper.mega_moe = SimpleNamespace(
            _mega_buf=SimpleNamespace(num_max_tokens_per_rank=3),
            _forward_impl=execute,
        )
        x = torch.arange(48).reshape(8, 6).float()
        output = wrapper.forward_gate_pack(x, x, None, 2.5, plan)
        self.assertEqual(calls, [0, 3, 6])
        workspace.zero_()
        torch.testing.assert_close(output, x)

    def test_gate_pack_auto_keeps_native_gate_for_small_batches(self):
        for mode, rows, fused in (
            ("auto", 1, False),
            ("auto", 8191, False),
            ("auto", 8192, True),
            ("auto", 8193, True),
            ("1", 1, True),
        ):
            with self.subTest(mode=mode, rows=rows):
                layer = self.make_layer()
                layer.config = SimpleNamespace(routed_scaling_factor=2.5)
                hidden = torch.zeros(rows, 6, dtype=torch.bfloat16)
                layer.fused_moe.supports_gate_pack = lambda *args: True
                layer.fused_moe.forward_gate_pack = Mock(return_value=hidden)
                with patch.dict(
                    "os.environ", {"RTP_LLM_GLM5_GATE_PACK": mode}
                ), patch.object(
                    layer, "native_gate_supported", return_value=True
                ), patch.object(
                    layer,
                    "native_gate",
                    return_value=(
                        torch.ones(rows, 1),
                        torch.zeros(rows, 1, dtype=torch.int64),
                    ),
                ) as native:
                    layer(hidden)
                self.assertEqual(
                    layer.fused_moe.forward_gate_pack.call_count, int(fused)
                )
                self.assertEqual(native.call_count, int(not fused))

    def test_cmp_native_gate_uses_prepacked_outputs_without_projection_or_topk(self):
        from rtp_llm.models_py.modules.hybrid.glm5_cmp import Glm5Cmp

        cmp = Glm5Cmp.__new__(Glm5Cmp)
        cmp.layer_idx = 0
        cmp._is_moe_layer = True
        hidden = torch.zeros(4, 6)
        cmp.ops = Mock()
        cmp.ops.mla_absorbed_output_bmm_quant.return_value = (hidden, hidden)
        cmp.ops.project_attention_output.return_value = hidden
        cmp._output_projection = (hidden, hidden)
        cmp._attention_impl = lambda _: SimpleNamespace(weights=[{W.mla_vc: hidden}])
        cmp.post_attention_layernorm = SimpleNamespace(
            weight=torch.ones(6), variance_epsilon=1e-5
        )
        cmp.mlp = Mock()
        cmp.mlp.native_gate_supported.return_value = True
        ids = torch.zeros(4, 1, dtype=torch.int64)
        weights = torch.ones(4, 1)
        result = cmp.mla_post_moe_pre(
            hidden, hidden, None, hidden, hidden, ids, weights
        )
        self.assertTrue(cmp.eplb_routing_fused)
        self.assertIs(result[2], ids)
        self.assertIs(result[3], weights)
        self.assertIs(cmp.mlp.native_gate.call_args.kwargs["out"][0], ids)
        self.assertIs(cmp.mlp.native_gate.call_args.kwargs["out"][1], weights)
        cmp.ops.router_proj.assert_not_called()
        cmp.ops.router_topk.assert_not_called()

    def test_cmp_prepacked_forward_updates_the_buffer_consumed_by_mega_moe(self):
        layer = self.make_layer()
        logical = torch.zeros(4, 1, dtype=torch.int64)
        layer.fused_moe.prepacked_ids = logical
        layer._use_mega_moe_fused_shared = True
        output = layer.forward_prepacked(torch.zeros(4, 6), logical, torch.ones(4, 1))
        expected = layer.eplb.weights[0][0].expand(4, -1)
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(logical[:, 0].tolist(), [0, 4, 0, 4])

    def test_cmp_binds_native_counter_and_clears_it_for_draft(self):
        layer = self.make_layer()
        layer.fused_moe.mega_moe = nn.Module()
        decoder = GenericMoeDecoderLayer.__new__(GenericMoeDecoderLayer)
        nn.Module.__init__(decoder)
        decoder.mlp = layer
        counts = torch.zeros(6, dtype=torch.int32)
        hidden = torch.zeros(4, 6)
        with patch.object(decoder, "_forward_cmp", return_value=hidden) as cmp:
            decoder(
                hidden,
                hidden,
                None,
                enable_cmp=True,
                expert_load=ExpertLoadContext(counts),
            )
            self.assertIs(layer.fused_moe.mega_moe.expert_token_counts, counts)
            decoder(hidden, hidden, None, enable_cmp=True)
            self.assertIsNone(layer.fused_moe.mega_moe.expert_token_counts)
            self.assertEqual(cmp.call_count, 2)

    def test_cmp_physical_ids_are_not_mapped_twice(self):
        layer = self.make_layer()
        ids = layer.eplb.map_ids(torch.zeros(4, 1, dtype=torch.int64))
        layer.fused_moe.prepacked_ids = ids
        layer._use_mega_moe_fused_shared = True
        with patch.object(
            layer.eplb, "map_ids", side_effect=AssertionError("double map")
        ):
            layer.forward_prepacked(
                torch.zeros(4, 6), ids, torch.ones(4, 1), ids_are_physical=True
            )
        self.assertEqual(ids[:, 0].tolist(), [0, 4, 0, 4])

    def test_fake_balance_overrides_physical_routes_before_mapping(self):
        layer = self.make_layer()
        ids = torch.full((4, 1), 5, dtype=torch.int64)
        layer.fake_balance_expert = lambda ids, weights: ids.zero_()
        layer.fused_moe.prepacked_ids = ids
        layer._use_mega_moe_fused_shared = True
        layer.forward_prepacked(
            torch.zeros(4, 6), ids, torch.ones(4, 1), ids_are_physical=True
        )
        self.assertEqual(ids[:, 0].tolist(), [0, 4, 0, 4])

    def test_pack_forward_preserves_logical_ids_and_passes_plan(self):
        layer = self.make_layer()
        layer.fused_moe.supports_eplb_input_pack = lambda plan, x: True
        with patch.object(
            layer.eplb, "map_ids", side_effect=AssertionError("early map")
        ):
            with patch.object(
                layer.fused_moe, "forward", return_value=torch.zeros(4, 6)
            ) as forward:
                layer(torch.zeros(4, 6))
        self.assertIs(forward.call_args.kwargs["eplb"], layer.eplb)
        self.assertTrue(
            torch.equal(forward.call_args.kwargs["topk_ids"], torch.full((4, 1), 3))
        )

    def test_chunked_pack_preserves_token_offset(self):
        wrapper = mega_moe_wrapper.MegaMoeWrapper.__new__(
            mega_moe_wrapper.MegaMoeWrapper
        )
        nn.Module.__init__(wrapper)
        wrapper._activation_clamp = None
        wrapper.mega_moe = SimpleNamespace(
            _mega_buf=SimpleNamespace(num_max_tokens_per_rank=3)
        )
        plan = make_layer()
        offsets = []

        def forward(x, weights, ids, *, eplb, token_offset):
            self.assertIs(eplb, plan)
            offsets.append(token_offset)
            return x

        x = torch.randn(8, 6)
        result = wrapper._forward_chunked(
            x, torch.ones(8, 1), torch.zeros(8, 1), forward, eplb=plan
        )
        self.assertEqual(offsets, [0, 3, 6])
        torch.testing.assert_close(result, x)

    def test_mega_moe_allocates_physical_slots(self):
        config = _config()
        config.eplb_config = SimpleNamespace(phy_exp_num=lambda logical: logical + 2)
        weights = {
            W.moe_w1: torch.ones(4, 8, 8).to(torch.float8_e4m3fn),
            W.moe_w2: torch.ones(4, 8, 4).to(torch.float8_e4m3fn),
            W.moe_s1: torch.ones(4, 8, 1),
            W.moe_s2: torch.ones(4, 8, 1),
        }
        with patch.object(mega_moe_wrapper, "GLM5MegaMoE", _FakeMegaMoE):
            wrapper = mega_moe_wrapper.MegaMoeWrapper(config, _parallelism(), weights)
        self.assertEqual(wrapper.mega_moe.params["n_routed_experts"], 4)

    def test_registration_returns_live_executor_tensors_for_target_and_mtp(self):
        wrapper = mega_moe_wrapper.MegaMoeWrapper.__new__(
            mega_moe_wrapper.MegaMoeWrapper
        )
        nn.Module.__init__(wrapper)
        wrapper._activation_clamp = None
        tensors = tuple(torch.zeros(6, 2) for _ in range(4))
        wrapper.mega_moe = SimpleNamespace(
            _mega_l1_w=tensors[0],
            _mega_l1_sf=tensors[1],
            _mega_l2_w=tensors[2],
            _mega_l2_sf=tensors[3],
        )
        for actual, expected in zip(wrapper.get_eplb_weights(), tensors):
            self.assertIs(actual, expected)
        layer = self.make_layer()
        for model_type in (GenericMoeModel, GenericMoeMTPModel):
            model = model_type.__new__(model_type)
            nn.Module.__init__(model)
            model.layers = nn.ModuleList([nn.Sequential(layer)])
            self.assertEqual(model.get_eplb_layers(), [layer.eplb])

    def test_draft_does_not_inherit_target_eplb_configuration(self):
        from rtp_llm.config.py_config_modules import ModelArgs
        from rtp_llm.ops import EPLBConfig, ProfilingDebugLoggingConfig, SpeculativeType

        eplb = EPLBConfig()
        eplb.redundant_expert = 4
        eplb.eplb_metrics_report_interval = 3
        eplb.eplb_metrics_snapshot_max_files = 2
        target = SimpleNamespace(max_seq_len=1024, eplb_config=eplb)
        draft = SimpleNamespace(
            tokenizer_path="", ckpt_path="", num_layers=1, eplb_config=EPLBConfig()
        )
        sp = SimpleNamespace(
            type=SpeculativeType.EAGLE,
            checkpoint_path="/model",
            model_type="glm_5_mtp",
            quantization="",
        )
        profiling = ProfilingDebugLoggingConfig()
        profiling.hack_layer_num = 4
        profiling.check_nan = True
        engine = SimpleNamespace(
            sp_config=sp,
            kv_cache_config=None,
            profiling_debug_logging_config=profiling,
        )
        model_cls = SimpleNamespace(_create_config=lambda path: draft)
        with patch.object(ModelFactory, "get_model_cls", return_value=model_cls):
            with patch("rtp_llm.model_factory.build_model_config") as build_config:
                result = ModelFactory.create_propose_model_config(
                    engine, target, ModelArgs()
                )
        self.assertEqual(
            build_config.call_args.kwargs[
                "profiling_debug_logging_config"
            ].hack_layer_num,
            1,
        )
        self.assertEqual(engine.profiling_debug_logging_config.hack_layer_num, 4)
        self.assertTrue(
            build_config.call_args.kwargs["profiling_debug_logging_config"].check_nan
        )
        self.assertEqual(result.eplb_config.redundant_expert, 0)
        self.assertEqual(
            result.eplb_config.eplb_metrics_report_interval,
            EPLBConfig().eplb_metrics_report_interval,
        )
        self.assertEqual(result.eplb_config.eplb_metrics_snapshot_max_files, 0)
        self.assertIsNot(result.eplb_config, eplb)
        result.eplb_config.redundant_expert = 8
        self.assertEqual(eplb.redundant_expert, 4)


if __name__ == "__main__":
    unittest.main()
