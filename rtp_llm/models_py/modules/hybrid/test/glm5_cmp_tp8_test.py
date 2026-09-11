"""CPU TP8 CMP routing/order tests; these do not validate NCCL or GPU numerics."""

import importlib
import itertools
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.hybrid import glm5_cmp as bridge
from rtp_llm.ops import CPRotateMethod, ParallelismConfig, PrefillCPConfig


def parallelism(tp=8, *, attn_tp=None, sharded=False):
    attn_tp = tp if attn_tp is None else attn_tp
    return SimpleNamespace(
        tp_size=tp,
        get_attn_tp_size=lambda: attn_tp,
        prefill_cp_config=SimpleNamespace(
            is_enabled=lambda: attn_tp != tp,
            is_prefill_enabled=lambda: True,
            kv_cache_sharded=sharded,
            prefill_cp_size=8,
        ),
    )


def make_cmp(tp=8, *, heads=None):
    norm = SimpleNamespace(weight=torch.ones(6144), variance_epsilon=1e-5)
    return bridge.Glm5Cmp(
        layer_idx=0,
        config=SimpleNamespace(
            model_type="glm_5",
            moe_layer_index=(0,),
            gen_num_per_cycle=5,
            attn_config=SimpleNamespace(
                use_mla=True, kernel_tokens_per_block=64, head_num=64
            ),
        ),
        parallelism_config=parallelism(tp),
        self_attn=SimpleNamespace(
            num_heads=64 // tp if heads is None else heads, has_indexer=True
        ),
        input_layernorm=norm,
        mlp=SimpleNamespace(correction_bias=torch.zeros(256)),
        post_attention_layernorm=norm,
    )


class Glm5CmpTp8Test(unittest.TestCase):
    def setUp(self):
        self.guards = ExitStack()
        self.addCleanup(self.guards.close)

        def forbidden(*args, **kwargs):
            raise AssertionError("CPU contract test reached CUDA or a tensor host read")

        for name in ("_lazy_init", "synchronize", "stream", "set_device"):
            self.guards.enter_context(patch.object(torch.cuda, name, forbidden))
        for name in (
            "item",
            "cpu",
            "numpy",
            "tolist",
            "__bool__",
            "__int__",
            "__float__",
        ):
            self.guards.enter_context(patch.object(torch.Tensor, name, forbidden))
        self.guards.enter_context(
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
        )

    def test_parallelism_gate_keeps_tp1_and_only_unsharded_tp8(self):
        for tp, attn_tp, sharded, supported in (
            (1, 1, False, True),
            (1, 1, True, True),
            (8, 8, False, True),
            (8, 8, True, False),
            (8, 1, False, False),
            (8, 1, True, False),
            (2, 2, False, False),
            (4, 4, False, False),
            (1, 8, False, False),
        ):
            with self.subTest(tp=tp, attn_tp=attn_tp, sharded=sharded):
                reason = bridge._unsupported_parallelism_reason(
                    parallelism(tp, attn_tp=attn_tp, sharded=sharded)
                )
                self.assertEqual(reason is None, supported)

    def test_tp8_requires_eight_local_query_heads(self):
        self.assertIsNone(make_cmp(8, heads=8)._disabled_reason)
        for heads in (1, 16, 64):
            self.assertIsNotNone(make_cmp(8, heads=heads)._disabled_reason)
        self.assertIsNone(make_cmp(1, heads=64)._disabled_reason)

    def test_real_pybind_parallelism_distinguishes_prefill_metadata_from_cp(self):
        # These are the actual bound C++ objects, not stubs of get_attn_tp_size.
        for method, sharded, attn_tp, supported in (
            (CPRotateMethod.PREFILL_CP, False, 8, True),
            (CPRotateMethod.PREFILL_CP, True, 8, False),
            (CPRotateMethod.DISABLED, False, 8, True),
            (CPRotateMethod.DISABLED, True, 8, False),
            (CPRotateMethod.ALL_GATHER, False, 1, False),
            (CPRotateMethod.ALL_GATHER_WITH_OVERLAP, False, 1, False),
            (CPRotateMethod.ALLTOALL, False, 1, False),
            (CPRotateMethod.ALL_GATHER, True, 1, False),
        ):
            with self.subTest(method=method, sharded=sharded):
                config = ParallelismConfig()
                config.tp_size = 8
                cp = PrefillCPConfig()
                cp.method = method
                cp.prefill_cp_size = 8
                cp.kv_cache_sharded = sharded
                config.prefill_cp_config = cp
                self.assertEqual(config.get_attn_tp_size(), attn_tp)
                self.assertEqual(config.prefill_cp_config.is_enabled(), attn_tp == 1)
                self.assertEqual(
                    bridge._unsupported_parallelism_reason(config) is None, supported
                )
        config = ParallelismConfig()
        config.tp_size, config.dp_size, config.world_size = 1, 8, 8
        cp = PrefillCPConfig()
        cp.method = CPRotateMethod.PREFILL_CP
        cp.prefill_cp_size = 8
        cp.kv_cache_sharded = True
        config.prefill_cp_config = cp
        self.assertEqual(config.get_attn_tp_size(), 1)
        self.assertFalse(config.prefill_cp_config.is_enabled())
        self.assertTrue(config.prefill_cp_config.is_prefill_enabled())
        self.assertIsNone(bridge._unsupported_parallelism_reason(config))

    def test_invalid_local_heads_fail_before_any_projection_call(self):
        for heads in (0, 1, 16, 32, 128):
            with self.subTest(heads=heads):
                cmp = make_cmp(8, heads=heads)
                cmp._q_b_proj = (object(), object())
                cmp.ops = SimpleNamespace(q_b_proj=Mock(), indexer_q_proj=Mock())
                with self.assertRaisesRegex(ValueError, "64 or 8 local query heads"):
                    cmp._project_query(
                        torch.empty((1, 2048), dtype=torch.float8_e4m3fn),
                        torch.empty((1, 4), dtype=torch.int32),
                        object(),
                    )
                cmp.ops.q_b_proj.assert_not_called()
                cmp.ops.indexer_q_proj.assert_not_called()

    def test_dynamic_backend_gate_and_multi_query_rows(self):
        for tp, queries, backend in itertools.product(
            (1, 8), (1, 4, 6), ("trtllm_gen_656_compat", "flashmla")
        ):
            with self.subTest(tp=tp, queries=queries, backend=backend):
                cmp = make_cmp(tp)
                rows = 2 * queries
                impl = SimpleNamespace(
                    weights={},
                    fmha_impl=SimpleNamespace(backend_name=backend),
                    fmha_params=SimpleNamespace(
                        expanded_seq_lens=torch.zeros(rows, dtype=torch.int32)
                    ),
                    attn_inputs=SimpleNamespace(
                        is_prefill=False,
                        is_target_verify=queries > 1,
                        kv_cache_block_id_device=torch.ones((2, 2), dtype=torch.int32),
                    ),
                )
                hidden = torch.empty((rows, 6144), dtype=torch.bfloat16)
                for method in (
                    cmp._unsupported_call_reason,
                    cmp._unsupported_reuse_call_reason,
                ):
                    self.assertEqual(
                        method(hidden, impl, object()) is None,
                        tp == 1 or backend == "trtllm_gen_656_compat",
                    )
                if tp == 8 and backend != "trtllm_gen_656_compat":
                    with patch.object(bridge, "_load_ops") as load_ops:
                        self.assertFalse(
                            bridge.should_enable_glm5_cmp(
                                [SimpleNamespace(cmp=cmp)],
                                1,
                                hidden,
                                impl,
                                SimpleNamespace(get_layer_cache=lambda _: object()),
                            )
                        )
                    load_ops.assert_not_called()
                impl.attn_inputs.is_prefill = True
                impl.attn_inputs.is_target_verify = False
                self.assertIsNotNone(
                    cmp._unsupported_call_reason(hidden, impl, object())
                )

    def test_tp8_preserves_draft_prefill_and_row_limit(self):
        cmp = make_cmp(8)
        cmp.config.model_type = "glm_5_mtp"
        cmp._draft_prefill_clone = True
        impl = SimpleNamespace(
            weights={},
            fmha_impl=SimpleNamespace(backend_name="trtllm_gen_656_compat"),
            fmha_params=SimpleNamespace(
                expanded_seq_lens=torch.zeros(12, dtype=torch.int32)
            ),
            attn_inputs=SimpleNamespace(
                is_prefill=True,
                is_target_verify=False,
                kv_cache_block_id_device=torch.ones((2, 2), dtype=torch.int32),
            ),
        )
        self.assertIsNone(
            cmp._unsupported_reuse_call_reason(torch.empty((12, 6144)), impl, object())
        )
        impl.fmha_params.expanded_seq_lens = torch.zeros(8, dtype=torch.int32)
        self.assertIsNotNone(
            cmp._unsupported_reuse_call_reason(torch.empty((8, 6144)), impl, object())
        )
        impl.attn_inputs.is_prefill = False
        for rows, supported in ((256, True), (257, False), (384, False)):
            with self.subTest(rows=rows):
                impl.fmha_params.expanded_seq_lens = torch.zeros(
                    rows, dtype=torch.int32
                )
                impl.attn_inputs.kv_cache_block_id_device = torch.ones(
                    (1, 2), dtype=torch.int32
                )
                self.assertEqual(
                    cmp._unsupported_call_reason(
                        torch.empty((rows, 6144)), impl, object()
                    )
                    is None,
                    supported,
                )

    def test_reduce_uses_returned_tensor_and_exact_tp_group(self):
        for tp in (1, 8):
            with self.subTest(tp=tp):
                cmp = make_cmp(tp)
                local, reduced = torch.empty((2, 6144)), torch.empty((2, 6144))
                with patch.object(bridge, "all_reduce", return_value=reduced) as reduce:
                    self.assertIs(
                        cmp._reduce_attention_output(local),
                        reduced if tp == 8 else local,
                    )
                    if tp == 8:
                        reduce.assert_called_once_with(local, group=bridge.Group.TP)
                    else:
                        reduce.assert_not_called()
                    self.assertIs(
                        object.__new__(bridge.Glm5Cmp)._reduce_attention_output(local),
                        local,
                    )

    def test_fused_post_reduces_before_residual_and_moe(self):
        for tp, prepack in itertools.product((1, 8), (False, True)):
            with self.subTest(tp=tp, prepack=prepack):
                cmp = make_cmp(tp)
                calls = []
                latent = torch.empty((2, 64 // tp, 512), dtype=torch.bfloat16)
                residual = torch.empty((2, 6144), dtype=torch.bfloat16)
                local, reduced = torch.empty_like(residual), torch.empty_like(residual)
                impl = SimpleNamespace(
                    weights=[{bridge.W.mla_vc: object()}], fmha_params=None
                )
                cmp._output_projection = (object(), object())
                cmp._router_weight = object()
                activation, activation_scale, indices, weights = (
                    torch.empty(1) for _ in range(4)
                )

                def record(name, result):
                    def run(*args, **kwargs):
                        calls.append(name)
                        return result

                    return run

                def norm(attention_output, incoming_residual, *args, **kwargs):
                    calls.append("residual_norm")
                    self.assertIs(attention_output, reduced if tp == 8 else local)
                    self.assertIs(incoming_residual, residual)
                    self.assertIs(kwargs["out"][2], activation)
                    self.assertIs(kwargs["out"][3], activation_scale)

                cmp.ops = SimpleNamespace(
                    mla_absorbed_output_bmm_quant=Mock(
                        side_effect=record("wvc", (object(), object()))
                    ),
                    project_attention_output=Mock(side_effect=record("o_proj", local)),
                    add_rms_norm_mega_moe_quant=Mock(side_effect=norm),
                    router_proj=Mock(side_effect=record("router", object())),
                    router_topk=Mock(side_effect=record("topk", None)),
                )
                kwargs = (
                    {
                        "moe_activation": activation,
                        "moe_scale": activation_scale,
                        "routed_indices": indices,
                        "routed_weights": weights,
                    }
                    if prepack
                    else {}
                )
                with patch.object(
                    bridge, "all_reduce", side_effect=record("reduce", reduced)
                ) as reduce:
                    result = cmp.mla_post_moe_pre(latent, residual, impl, **kwargs)
                self.assertEqual(
                    calls,
                    ["wvc", "o_proj"]
                    + (["reduce"] if tp == 8 else [])
                    + (["residual_norm", "router", "topk"] if prepack else []),
                )
                if prepack:
                    self.assertIs(result[2], indices)
                    self.assertIs(result[3], weights)
                else:
                    self.assertIs(result[0], reduced if tp == 8 else local)
                    self.assertIs(result[1], residual)
                self.assertEqual(reduce.call_count, 1 if tp == 8 else 0)

    def test_standard_post_fallback_reduces_once(self):
        for tp in (1, 8):
            with self.subTest(tp=tp):
                cmp = make_cmp(tp)
                cmp.disable_attention_post_moe_pre = True
                latent = torch.empty((2, 64 // tp, 512), dtype=torch.bfloat16)
                expanded = torch.empty((2, 64 // tp, 256), dtype=torch.bfloat16)
                local, reduced, residual = (torch.empty((2, 6144)) for _ in range(3))
                calls = []
                impl = SimpleNamespace(
                    weights={},
                    fmha_params=None,
                    _apply_output_bmm=Mock(
                        side_effect=lambda *args: calls.append("wvc") or expanded
                    ),
                )
                cmp.self_attn.o_proj = Mock(
                    side_effect=lambda *args: calls.append("o_proj") or local
                )
                with patch.object(
                    bridge,
                    "all_reduce",
                    side_effect=lambda *a, **kw: calls.append("reduce") or reduced,
                ):
                    result = cmp.mla_post_moe_pre(latent, residual, impl)
                self.assertEqual(
                    calls, ["wvc", "o_proj"] + (["reduce"] if tp == 8 else [])
                )
                self.assertIs(result[0], reduced if tp == 8 else local)
                self.assertIs(result[1], residual)
                self.assertEqual(
                    cmp.self_attn.o_proj.call_args.args[0].shape, (2, (64 // tp) * 256)
                )

    def test_project_query_routes_tp1_unchanged_and_tp8_to_fused_epilogue(self):
        helper = importlib.import_module(
            "rtp_llm.models_py.triton_kernels.sparse_mla.glm5_cmp_q_b"
        )
        for tp, rows, preallocated in itertools.product(
            (1, 8), (1, 4, 6), (False, True)
        ):
            with self.subTest(tp=tp, rows=rows, preallocated=preallocated):
                cmp = make_cmp(tp)
                heads = 64 // tp
                q_fp8 = torch.empty((rows, 2048), dtype=torch.float8_e4m3fn)
                q_scale = torch.empty((rows, 4), dtype=torch.int32)
                projected = torch.empty((rows, heads * 256), dtype=torch.bfloat16)
                outputs = (
                    torch.empty((rows, heads, 192), dtype=torch.bfloat16),
                    torch.empty((rows, heads, 576), dtype=torch.bfloat16),
                )
                cmp._q_b_proj = (object(), object())
                impl = SimpleNamespace(
                    _cos_sin_cache=torch.empty((16, 64), dtype=torch.float32),
                    _is_neox_style=True,
                    fmha_params=SimpleNamespace(
                        positions_d=torch.zeros(rows, dtype=torch.int32)
                    ),
                )
                cmp.ops = SimpleNamespace(
                    q_b_proj=Mock(return_value=outputs),
                    indexer_q_proj=Mock(return_value=projected),
                )
                with patch.object(
                    helper,
                    "q_b_proj_h8",
                    side_effect=lambda *a, **kw: kw["out"],
                ) as fused:
                    result = cmp._project_query(
                        q_fp8,
                        q_scale,
                        impl,
                        out=outputs if preallocated else None,
                    )
                if tp == 1:
                    self.assertIs(result, outputs)
                    cmp.ops.q_b_proj.assert_called_once()
                    args, kwargs = cmp.ops.q_b_proj.call_args
                    self.assertEqual(len(args), 6)
                    for actual, expected in zip(
                        args,
                        (
                            q_fp8,
                            q_scale,
                            *cmp._q_b_proj,
                            impl._cos_sin_cache,
                            impl.fmha_params.positions_d,
                        ),
                    ):
                        self.assertIs(actual, expected)
                    if preallocated:
                        self.assertIs(kwargs["out"], outputs)
                    else:
                        self.assertNotIn("out", kwargs)
                    cmp.ops.indexer_q_proj.assert_not_called()
                    fused.assert_not_called()
                else:
                    cmp.ops.q_b_proj.assert_not_called()
                    cmp.ops.indexer_q_proj.assert_not_called()
                    fused.assert_called_once()
                    args, kwargs = fused.call_args
                    self.assertIs(args[0], q_fp8)
                    self.assertIs(args[1], q_scale)
                    self.assertEqual(args[2:4], cmp._q_b_proj)
                    if preallocated:
                        self.assertIs(kwargs["out"], outputs)
                        self.assertIs(result[0], outputs[0])
                        self.assertIs(result[1], outputs[1])
                    self.assertEqual(result[0].shape, (rows, 8, 192))
                    self.assertEqual(result[1].shape, (rows, 8, 576))
                    self.assertIs(args[4], impl._cos_sin_cache)
                    self.assertIs(args[5], impl.fmha_params.positions_d)
                    self.assertTrue(kwargs["is_neox_style"])

    def test_graph_clone_preserves_local_heads_and_does_not_share_events(self):
        cmp = make_cmp(8)
        cmp._events, cmp.ops = object(), object()
        cmp._q_b_proj = (object(), object())
        cmp._moe_prepack_abi_validated = True
        clone = cmp.clone_for_cuda_graph(mlp=cmp.mlp, draft_prefill=True)
        self.assertIsNone(clone._disabled_reason)
        self.assertIs(clone.parallelism_config, cmp.parallelism_config)
        self.assertEqual(clone.self_attn.num_heads, 8)
        self.assertIs(clone._q_b_proj, cmp._q_b_proj)
        self.assertIs(clone.ops, cmp.ops)
        self.assertIsNone(clone._events)
        self.assertTrue(clone._draft_prefill_clone)
        self.assertFalse(clone._moe_prepack_abi_validated)


if __name__ == "__main__":
    unittest.main()
