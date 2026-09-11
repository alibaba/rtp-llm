"""CPU-only contracts for ordinary BF16-router packing in TP8 CMP.

These run the actual GenericMoeLayer/CMP control flow and exact wrapper classes
with CPU buffers and CUDA operator spies. They do not validate the fused CUDA
kernel, symmetric-memory ABI, distributed MegaMoE, or model accuracy.
"""

import builtins
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.model_desc import generic_moe as generic
from rtp_llm.models_py.modules.glm5_mega_moe.input_packer import (
    FusedMegaMoeInputPacker,
    TorchMegaMoeInputPacker,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe import GLM5MegaMoE
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_wrapper import MegaMoeWrapper
from rtp_llm.models_py.modules.hybrid import glm5_cmp as bridge

_HELPER = "rtp_llm.models_py.triton_kernels.sparse_mla.glm5_moe_router_pack"


def empty_module(cls):
    value = object.__new__(cls)
    torch.nn.Module.__init__(value)
    return value


def make_fixture(rows=4, capacity=16):
    order = []
    parallelism = SimpleNamespace(tp_size=8, ep_size=8, ffn_tp_size=8)
    parallelism.get_attn_tp_size = lambda: parallelism.tp_size
    parallelism.get_ffn_tp_size = lambda: parallelism.ffn_tp_size
    parallelism.prefill_cp_config = SimpleNamespace(
        is_enabled=lambda: False, kv_cache_sharded=False
    )
    config = SimpleNamespace(
        model_type="glm_5",
        moe_layer_index=(0,),
        hidden_size=6144,
        expert_num=256,
        moe_k=8,
        moe_n_group=1,
        moe_topk_group=1,
        has_moe_norm=True,
        routed_scaling_factor=2.5,
        attn_config=SimpleNamespace(
            use_mla=True, head_num=64, kernel_tokens_per_block=64
        ),
    )
    hidden = torch.ones((rows, 6144), dtype=torch.bfloat16)
    logits = torch.arange(rows * 256).reshape(rows, 256).bfloat16() / 256
    routed, shared, reduced = (torch.full_like(hidden, value) for value in (2, 3, 7))
    buf = SimpleNamespace(
        num_max_tokens_per_rank=capacity,
        x=torch.empty((capacity, 6144), dtype=torch.float8_e4m3fn),
        x_sf=torch.empty((capacity, 48), dtype=torch.int32),
        topk_idx=torch.empty((capacity, 8), dtype=torch.int64),
        topk_weights=torch.empty((capacity, 8), dtype=torch.float32),
    )
    mega = empty_module(GLM5MegaMoE)
    mega.cfg = SimpleNamespace(
        dim=6144, n_routed_experts=256, n_activated_experts=8, ep_size=8
    )
    mega._mega_buf = buf
    mega._input_packer = FusedMegaMoeInputPacker()
    wrapper = empty_module(MegaMoeWrapper)
    wrapper.mega_moe, wrapper.expert_num = mega, 256
    wrapper.forward = Mock(
        side_effect=lambda **_: order.append("ordinary_mega") or routed
    )
    wrapper.forward_prepacked = Mock(
        side_effect=lambda _: order.append("prepacked_mega") or routed
    )
    mlp = empty_module(generic.GenericMoeLayer)
    mlp.config, mlp.parallelism_config = config, parallelism
    mlp.hidden_dim, mlp.num_experts, mlp.top_k = 6144, 256, 8
    mlp.ffn_tp_size, mlp.ep_size, mlp.gate_chunk_rows = 8, 8, 0
    mlp.fake_balance_expert, mlp.shared_expert_gate = None, None
    mlp._use_mega_moe_fused_shared = False
    mlp.fused_moe = wrapper
    mlp.correction_bias = torch.zeros(256)
    mlp.gate = Mock(side_effect=lambda _: order.append("bf16_gate") or logits)
    mlp.gate.weight = torch.empty((256, 6144), dtype=torch.bfloat16)
    mlp.shared_expert = Mock(
        side_effect=lambda *_, **__: order.append("shared") or shared
    )
    mlp.shared_expert.accepts_fp8_input = True
    mlp.shared_expert.up_proj = SimpleNamespace(scale_ue8m0=True)
    mlp.select_topk = Mock(side_effect=lambda *_: order.append("select_topk"))
    linear = SimpleNamespace(weight=torch.empty(1), weight_scales=torch.empty(1))
    attention = SimpleNamespace(
        num_heads=8,
        has_indexer=False,
        fused_qkv_a_proj=linear,
        q_b_proj=linear,
        o_proj=linear,
    )
    norm = SimpleNamespace(weight=torch.ones(6144), variance_epsilon=1.0e-5)
    cmp = bridge.Glm5Cmp(
        layer_idx=0,
        config=config,
        parallelism_config=parallelism,
        self_attn=attention,
        input_layernorm=norm,
        mlp=mlp,
        post_attention_layernorm=norm,
    )
    cmp._initialize_for_cmp(SimpleNamespace())
    return SimpleNamespace(
        cmp=cmp,
        mlp=mlp,
        wrapper=wrapper,
        mega=mega,
        buf=buf,
        order=order,
        hidden=hidden,
        logits=logits,
        routed=routed,
        shared=shared,
        reduced=reduced,
        config=config,
        parallelism=parallelism,
        rows=rows,
    )


class OrdinaryRouterPackIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.f = make_fixture()

        def forbidden(*args, **kwargs):
            raise AssertionError("CPU integration reached CUDA or tensor host read")

        for name in (
            "item",
            "cpu",
            "numpy",
            "tolist",
            "__bool__",
            "__int__",
            "__float__",
        ):
            self.stack.enter_context(patch.object(torch.Tensor, name, forbidden))
        for name in ("_lazy_init", "synchronize", "set_device"):
            self.stack.enter_context(patch.object(torch.cuda, name, forbidden))
        self.stack.enter_context(
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
        )

    def mock_pack(self, fixture=None):
        f = fixture or self.f

        def run(hidden, logits, bias, **outputs):
            f.order.append("router_pack")
            self.assertIs(hidden, f.hidden)
            self.assertEqual(logits.dtype, torch.float32)
            self.assertTrue(torch.equal(logits, f.logits.float()))
            self.assertIs(bias, f.mlp.correction_bias)
            expected = f.wrapper.prepacked_input_views(f.rows)
            for value, target in zip(outputs.values(), expected):
                self.assertEqual(value.data_ptr(), target.data_ptr())
            outputs["activation_out"].view(torch.uint8).fill_(42)
            outputs["scales_out"].fill_(0x7F7F7F7F)
            outputs["topk_ids_out"].copy_(torch.arange(8).expand(f.rows, -1))
            outputs["topk_weights_out"].fill_(0.3125)
            return tuple(outputs.values())

        pack = Mock(side_effect=run)
        self.stack.enter_context(
            patch.dict(sys.modules, {_HELPER: SimpleNamespace(fused_router_pack=pack)})
        )
        return pack

    def reduce(self):
        f = self.f
        return self.stack.enter_context(
            patch.object(
                generic,
                "all_reduce",
                side_effect=lambda value, **_: f.order.append("shared_reduce")
                or f.reduced,
            )
        )

    def test_callback_preserves_bf16_gate_and_one_shared_reduce_and_fp8_aliases(self):
        f = self.f
        pack, reduce = self.mock_pack(), self.reduce()
        group = self.stack.enter_context(
            patch.object(
                generic, "GroupTopK", side_effect=AssertionError("duplicate TopK")
            )
        )
        x_fp8 = torch.empty_like(f.hidden, dtype=torch.float8_e4m3fn)
        x_scale = torch.empty((f.rows, 48), dtype=torch.int32)
        callback = f.cmp.moe_router_pack_callback(f.rows)
        self.assertIs(callback.__self__, f.cmp)
        result = f.mlp(f.hidden, x_fp8=x_fp8, x_scale=x_scale, router_pack=callback)
        self.assertEqual(
            f.order,
            ["bf16_gate", "router_pack", "prepacked_mega", "shared", "shared_reduce"],
        )
        pack.assert_called_once()
        group.assert_not_called()
        f.wrapper.forward.assert_not_called()
        f.wrapper.forward_prepacked.assert_called_once_with(f.hidden)
        reduce.assert_called_once_with(f.shared, group=generic.Group.TP)
        kwargs = f.mlp.shared_expert.call_args.kwargs
        self.assertIs(kwargs["x_fp8"], x_fp8)
        self.assertIs(kwargs["x_scale"], x_scale)
        self.assertTrue(kwargs["skip_allreduce"])
        self.assertTrue(torch.equal(result, f.routed + f.reduced))

    def test_fixed_m_linear_stays_before_float_and_callback(self):
        f = self.f
        f.mlp.gate_chunk_rows = 8192
        pack, _ = self.mock_pack(), self.reduce()
        fixed = self.stack.enter_context(
            patch.object(
                generic,
                "fixed_m_linear",
                side_effect=lambda *_: f.order.append("fixed_m_linear") or f.logits,
            )
        )
        f.mlp(f.hidden, router_pack=f.cmp.moe_router_pack_callback(f.rows))
        fixed.assert_called_once_with(f.mlp.gate, f.hidden, 8192)
        f.mlp.gate.assert_not_called()
        self.assertEqual(f.order[:2], ["fixed_m_linear", "router_pack"])
        self.assertEqual(pack.call_args.args[1].dtype, torch.float32)

    def test_original_three_positional_inputs_select_topk_and_fake_balance_survive(
        self,
    ):
        f = self.f
        self.reduce()
        f.mlp.correction_bias = None
        f.mlp.fake_balance_expert = Mock(
            side_effect=lambda *_: f.order.append("fake_balance")
        )
        x_fp8 = torch.empty_like(f.hidden, dtype=torch.float8_e4m3fn)
        x_scale = torch.empty((f.rows, 48), dtype=torch.int32)
        f.mlp(f.hidden, x_fp8, x_scale)
        self.assertEqual(
            f.order,
            [
                "bf16_gate",
                "select_topk",
                "fake_balance",
                "ordinary_mega",
                "shared",
                "shared_reduce",
            ],
        )
        f.wrapper.forward_prepacked.assert_not_called()
        self.assertIs(f.mlp.shared_expert.call_args.kwargs["x_fp8"], x_fp8)
        self.assertIs(f.mlp.shared_expert.call_args.kwargs["x_scale"], x_scale)

    def test_none_callback_keeps_ordinary_topk_packer_and_shared_tail(self):
        f = self.f
        self.reduce()
        original_import = builtins.__import__

        def no_helper(name, *args, **kwargs):
            if name == _HELPER:
                raise AssertionError("default path imported fused helper")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_helper), patch.object(
            generic,
            "GroupTopK",
            return_value=Mock(side_effect=lambda **_: f.order.append("group_topk")),
        ):
            f.mlp(f.hidden)
        self.assertEqual(
            f.order,
            ["bf16_gate", "group_topk", "ordinary_mega", "shared", "shared_reduce"],
        )
        f.wrapper.forward_prepacked.assert_not_called()

    def test_static_gate_rejects_unsupported_without_importing_helper(self):
        cases = (
            ("parallelism", "tp_size", 1),
            ("parallelism", "ep_size", 1),
            ("parallelism", "ffn_tp_size", 1),
            ("mlp", "ep_size", 1),
            ("mlp", "ffn_tp_size", 1),
            ("mlp", "fake_balance_expert", object()),
            ("mlp", "_use_mega_moe_fused_shared", True),
            ("mlp", "shared_expert_gate", object()),
            ("config", "hidden_size", 4096),
            ("config", "expert_num", 128),
            ("config", "moe_k", 4),
            ("config", "moe_n_group", 2),
            ("config", "moe_topk_group", 2),
            ("config", "has_moe_norm", False),
            ("config", "routed_scaling_factor", 1.0),
            ("config", "model_type", "other"),
            ("cmp", "disable_attention_post_moe_pre", True),
        )
        for target, name, value in cases:
            with self.subTest(target=target, name=name, value=value):
                f = make_fixture()
                setattr(getattr(f, target), name, value)
                self.assertIsNotNone(f.cmp._static_router_pack_disabled_reason())
        for dtype, shape in ((torch.bfloat16, (256,)), (torch.float32, (128,))):
            f = make_fixture()
            f.mlp.correction_bias = torch.empty(shape, dtype=dtype)
            self.assertIsNotNone(f.cmp._static_router_pack_disabled_reason())
        for enabled, sharded in ((True, False), (False, True)):
            f = make_fixture()
            f.parallelism.prefill_cp_config = SimpleNamespace(
                is_enabled=lambda: enabled, kv_cache_sharded=sharded
            )
            self.assertIsNotNone(f.cmp._static_router_pack_disabled_reason())

    def test_exact_wrapper_and_underlying_consumer_and_buffer_abi_required(self):
        f = self.f
        self.assertIsNone(f.cmp._static_router_pack_disabled_reason())
        f.wrapper.mega_moe = torch.nn.Identity()
        self.assertIsNotNone(f.cmp._static_router_pack_disabled_reason())
        f.wrapper.mega_moe = f.mega
        f.mega.cfg.ep_size = 1
        self.assertIsNotNone(f.cmp._static_router_pack_disabled_reason())
        f.mega.cfg.ep_size = 8
        f.buf.topk_idx = f.buf.topk_idx.int()
        self.assertIsNotNone(f.cmp._static_router_pack_disabled_reason())
        f.mlp.fused_moe = torch.nn.Identity()
        self.assertIsNotNone(f.cmp._static_router_pack_disabled_reason())

    def test_budget_and_empty_rows_fall_back_before_helper_or_views(self):
        f = self.f
        with patch.object(
            f.wrapper,
            "prepacked_input_views",
            side_effect=AssertionError("capacity should short circuit"),
        ):
            for rows in (0, -1, 17, 257):
                self.assertIsNone(f.cmp.moe_router_pack_callback(rows))
            self.assertIsNotNone(f.cmp.moe_router_pack_callback(16))

    def test_torch_and_unknown_packers_keep_ordinary_routing_and_consumer(self):
        class DifferentFusedPacker(FusedMegaMoeInputPacker):
            pass

        f = self.f
        self.reduce()
        original_import = builtins.__import__

        def no_helper(name, *args, **kwargs):
            if name == _HELPER:
                raise AssertionError("unsupported packer imported fused helper")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_helper), patch.object(
            generic,
            "GroupTopK",
            return_value=Mock(side_effect=lambda **_: f.order.append("group_topk")),
        ):
            for packer in (
                TorchMegaMoeInputPacker(),
                SimpleNamespace(name="fused"),
                DifferentFusedPacker(),
                None,
            ):
                with self.subTest(packer=type(packer).__name__):
                    f.order.clear()
                    f.wrapper.forward.reset_mock()
                    f.mega._input_packer = packer
                    f.cmp._router_pack_disabled_reason = (
                        f.cmp._static_router_pack_disabled_reason()
                    )
                    self.assertIn(
                        "exact fused input packer", f.cmp._router_pack_disabled_reason
                    )
                    callback = f.cmp.moe_router_pack_callback(f.rows)
                    self.assertIsNone(callback)
                    f.mlp(f.hidden, router_pack=callback)
                    self.assertEqual(
                        f.order,
                        [
                            "bf16_gate",
                            "group_topk",
                            "ordinary_mega",
                            "shared",
                            "shared_reduce",
                        ],
                    )
                    f.wrapper.forward.assert_called_once()
                    f.wrapper.forward_prepacked.assert_not_called()

    def test_clone_rechecks_its_actual_input_packer(self):
        base, second = self.f, make_fixture()
        second.mega._input_packer = TorchMegaMoeInputPacker()
        clone = base.cmp.clone_for_cuda_graph(mlp=second.mlp)
        self.assertIsNone(base.cmp._router_pack_disabled_reason)
        self.assertIsNotNone(base.cmp.moe_router_pack_callback(base.rows))
        self.assertIn("exact fused input packer", clone._router_pack_disabled_reason)
        self.assertIsNone(clone.moe_router_pack_callback(second.rows))

    def test_insufficient_capacity_really_uses_existing_wrapper_chunked_forward(self):
        f = self.f
        self.reduce()
        f.buf.num_max_tokens_per_rank = 2
        self.assertIsNone(f.cmp.moe_router_pack_callback(f.rows))
        # Restore the actual wrapper method so its existing chunk loop executes.
        del f.wrapper.forward
        with patch.object(
            f.mega, "forward", side_effect=lambda hidden, *_: torch.full_like(hidden, 2)
        ) as chunks, patch.object(generic, "GroupTopK", return_value=Mock()):
            result = f.mlp(f.hidden, router_pack=f.cmp.moe_router_pack_callback(f.rows))
        self.assertEqual(chunks.call_count, 2)
        self.assertEqual(
            [call.args[0].shape[0] for call in chunks.call_args_list], [2, 2]
        )
        f.wrapper.forward_prepacked.assert_not_called()
        self.assertTrue(torch.equal(result, f.routed + f.reduced))

    def test_callback_reacquires_dynamic_prefix_views_without_cached_row_shapes(self):
        f = self.f
        callback = f.cmp.moe_router_pack_callback(1)
        seen = []

        def pack(hidden, logits, bias, **outputs):
            rows = hidden.size(0)
            self.assertEqual(logits.shape, (rows, 256))
            self.assertEqual(outputs["activation_out"].shape, (rows, 6144))
            self.assertEqual(outputs["scales_out"].shape, (rows, 48))
            self.assertEqual(outputs["topk_ids_out"].shape, (rows, 8))
            self.assertEqual(outputs["activation_out"].data_ptr(), f.buf.x.data_ptr())
            seen.append(rows)

        with patch.dict(
            sys.modules, {_HELPER: SimpleNamespace(fused_router_pack=pack)}
        ):
            for rows in (1, 6, 4, 16, 1):
                ids, weights = callback(
                    torch.empty((rows, 6144), dtype=torch.bfloat16),
                    torch.empty((rows, 256)),
                )
                self.assertEqual(ids.shape, (rows, 8))
                self.assertEqual(weights.shape, (rows, 8))
        self.assertEqual(seen, [1, 6, 4, 16, 1])

    def test_clone_callback_targets_new_actual_buffers_not_base_bound_method(self):
        base, second = self.f, make_fixture()
        clone = base.cmp.clone_for_cuda_graph(mlp=second.mlp)
        self.assertIsNone(clone._router_pack_disabled_reason)
        callback = clone.moe_router_pack_callback(second.rows)
        self.assertIs(callback.__self__, clone)
        base.buf.x.view(torch.uint8).fill_(17)
        saved = base.buf.x.view(torch.uint8).clone()
        second.cmp = clone
        self.mock_pack(second)
        ids, weights = callback(second.hidden, second.logits.float())
        self.assertEqual(ids.data_ptr(), second.buf.topk_idx.data_ptr())
        self.assertEqual(weights.data_ptr(), second.buf.topk_weights.data_ptr())
        self.assertTrue(torch.equal(base.buf.x.view(torch.uint8), saved))
        self.assertNotEqual(base.buf.x.data_ptr(), second.buf.x.data_ptr())

    def test_decoder_only_dual_output_moe_branch_passes_callback(self):
        f = self.f
        layer = empty_module(generic.GenericMoeDecoderLayer)
        layer._fuse_post_norm_quant, layer._fuse_post_norm_quant_moe = False, True
        layer.mlp = f.mlp
        layer.post_attention_layernorm = SimpleNamespace(
            weight=torch.ones(6144), variance_epsilon=1.0e-5
        )
        x_fp8 = torch.empty_like(f.hidden, dtype=torch.float8_e4m3fn)
        x_scale = torch.empty((f.rows, 48), dtype=torch.int32)
        residual = torch.zeros_like(f.hidden)
        callback = f.cmp.moe_router_pack_callback(f.rows)
        self.mock_pack()
        self.reduce()
        norm = self.stack.enter_context(
            patch.object(
                generic,
                "fused_add_rmsnorm_fp8_quant_with_bf16_output",
                side_effect=lambda *_, **__: f.order.append("ordinary_dual_norm")
                or (f.hidden, x_fp8, x_scale),
            )
        )
        output, result_residual = layer._fwd_mlp_or_moe(
            f.hidden, residual, router_pack=callback
        )
        self.assertIs(result_residual, residual)
        self.assertEqual(
            f.order,
            [
                "ordinary_dual_norm",
                "bf16_gate",
                "router_pack",
                "prepacked_mega",
                "shared",
                "shared_reduce",
            ],
        )
        self.assertEqual(norm.call_args.kwargs["group_size"], 128)
        self.assertIs(f.mlp.shared_expert.call_args.kwargs["x_fp8"], x_fp8)
        self.assertIs(f.mlp.shared_expert.call_args.kwargs["x_scale"], x_scale)
        self.assertTrue(torch.equal(output, f.routed + f.reduced))
        for dense in (False, True):
            layer = empty_module(generic.GenericMoeDecoderLayer)
            layer._fuse_post_norm_quant_moe = False
            layer._fuse_post_norm_quant = dense
            layer.mlp = Mock(return_value=f.hidden)
            layer.mlp.up_proj = SimpleNamespace(scale_ue8m0=True)
            layer.post_attention_layernorm = Mock(return_value=(f.hidden, residual))
            layer.post_attention_layernorm.weight = torch.ones(6144)
            layer.post_attention_layernorm.variance_epsilon = 1.0e-5
            with patch.object(
                generic, "fused_add_rmsnorm_fp8_quant", return_value=(x_fp8, x_scale)
            ):
                layer._fwd_mlp_or_moe(
                    f.hidden,
                    residual,
                    router_pack=Mock(side_effect=AssertionError("non-MoE callback")),
                )
            self.assertNotIn("router_pack", layer.mlp.call_args.kwargs)

    def test_forward_cmp_fallback_selects_callback_without_native_norm_router(self):
        f = self.f
        layer = empty_module(generic.GenericMoeDecoderLayer)
        layer.cmp, layer.mlp = f.cmp, f.mlp
        layer._fuse_post_norm_quant_moe = True
        residual = torch.zeros_like(f.hidden)
        topk = torch.empty((f.rows, 2048), dtype=torch.int32)
        self.assertIsNotNone(
            f.cmp._moe_prepack_disabled_reason,
            "native FP32-router path must stay disabled for TP8",
        )
        f.cmp.mla_prologue = Mock(return_value=(residual, object(), topk))
        f.cmp.sparse_mla = Mock(return_value=torch.empty((f.rows, 8, 512)))
        f.cmp.mla_post_moe_pre = Mock(return_value=(f.hidden, residual))
        layer._fwd_mlp_or_moe = Mock(return_value=(f.hidden, residual))
        result = layer._forward_cmp(f.hidden, residual, None, None, None)
        self.assertIs(result.hidden_states, f.hidden)
        callback = layer._fwd_mlp_or_moe.call_args.kwargs["router_pack"]
        self.assertIs(callback.__self__, f.cmp)
        self.assertTrue(
            all(
                value is None
                for value in f.cmp.mla_post_moe_pre.call_args.kwargs.values()
            )
        )

    def test_non_cmp_decoder_does_not_even_select_router_callback(self):
        f = self.f
        layer = empty_module(generic.GenericMoeDecoderLayer)
        layer._fuse_input_norm_quant = False
        layer.cmp = SimpleNamespace(
            moe_router_pack_callback=Mock(
                side_effect=AssertionError("non-CMP callback selection")
            )
        )
        residual = torch.zeros_like(f.hidden)
        layer.input_layernorm = Mock(return_value=(f.hidden, residual))
        layer.self_attn = Mock(return_value=f.hidden)
        layer._fwd_mlp_or_moe = Mock(return_value=(f.hidden, residual))
        output = layer.forward(f.hidden, residual, None, enable_cmp=False)
        layer._fwd_mlp_or_moe.assert_called_once_with(f.hidden, residual)
        layer.cmp.moe_router_pack_callback.assert_not_called()
        self.assertIs(output.hidden_states, f.hidden)


if __name__ == "__main__":
    unittest.main()
