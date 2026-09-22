"""CPU-only contract tests: no CUDA contexts, allocations, or kernel launches."""

from __future__ import annotations

import ast
import copy
import os
import sys
import types
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest import mock

import torch
from torch import nn

_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
for name in (
    "rtp_llm",
    "rtp_llm.models_py",
    "rtp_llm.models_py.modules",
    "rtp_llm.models_py.modules.dsv4",
):
    package = types.ModuleType(name)
    package.__path__ = [str(_REPO.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, package)

from rtp_llm.models_py.modules.dsv4 import dsv41_kernel_jit_warmup as warmup


def fake_module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def shaped_module(name, **attrs):
    result = type(name, (nn.Module,), {})()
    for key, value in attrs.items():
        setattr(result, key, value)
    return result


class V41KernelJitWarmupTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
        self.stack.enter_context(mock.patch.object(warmup, "_DENSE_WARMED", set()))
        self.stack.enter_context(mock.patch.object(warmup, "_SHARED_WARMED", set()))
        self.stack.enter_context(mock.patch.object(warmup, "_ENGRAM_WARMED", set()))
        # Fail immediately if this test accidentally touches the CUDA runtime.
        for name in ("_lazy_init", "init", "synchronize", "current_device"):
            self.stack.enter_context(
                mock.patch.object(
                    torch.cuda, name, side_effect=AssertionError("GPU call in CPU test")
                )
            )

    def test_default_gate_and_both_disable_switches(self):
        self.assertTrue(warmup.model_warm_up_enabled())
        for env in ({"WARM_UP": "0"}, {"MODEL_WARM_UP": "0"}):
            with self.subTest(env=env), mock.patch.dict(os.environ, env):
                self.assertFalse(warmup.model_warm_up_enabled())
                warmup.warmup_v41_dense_jit(object(), max_m=8, device="cuda:0")
                warmup.warmup_v41_shared_expert_jit(object(), max_m=8, device="cuda:0")
                warmup.warmup_v41_engram_jit(object(), max_m=8, device="cuda:0")
                warmup.warmup_v41_prefill_jit(object(), device="cuda:0")
        self.assertFalse(warmup._DENSE_WARMED)
        self.assertFalse(warmup._SHARED_WARMED)
        self.assertFalse(warmup._ENGRAM_WARMED)

    def test_cpu_and_decode_skip_before_launch(self):
        model = types.SimpleNamespace(_is_decode_role=False)
        warmup.warmup_v41_dense_jit(model, max_m=8, device="cpu")
        warmup.warmup_v41_shared_expert_jit(model, max_m=8, device="cpu")
        warmup.warmup_v41_engram_jit(model, max_m=8, device="cpu")
        warmup.warmup_v41_prefill_jit(model, device="cpu")
        model._is_decode_role = True
        warmup.warmup_v41_prefill_jit(model, device="cuda:0")

    def test_collector_deduplicates_mxfp8_and_keeps_group128_out(self):
        model = nn.Module()
        model.q = shaped_module("V41MXFP8Linear", N=32, K=128)
        model.k = shaped_module("V41MXFP8Linear", N=32, K=128)
        model.other = shaped_module("FP8Linear", N=64, K=128)
        weight = torch.empty((8, 32, 128))
        model.a = shaped_module("AttentionV41FP8", _wo_a_stk_w=weight)
        model.b = shaped_module("AttentionV41FP8", _wo_a_stk_w=weight.clone())
        model.c = shaped_module("AttentionV41FP8", _wo_a_stk_w=None)
        linears, outputs = warmup.collect_v41_dense_shapes(model)
        self.assertEqual(set(linears), {(32, 128)})
        self.assertEqual(linears[(32, 128)], ("q", model.q))
        self.assertEqual(set(outputs), {(8, 32, 128)})
        self.assertEqual(outputs[(8, 32, 128)], ("a", model.a))

    def test_m_grid_includes_both_quantizer_threshold_sides(self):
        with mock.patch.object(
            warmup.common, "_generate_dense_gemm_warmup_m_grid", return_value=(32, 64)
        ) as grid:
            result = warmup._dense_m_grid(1000, 384, 5120, 152)
        self.assertEqual(result, (1, 2, 3, 4, 16, 17, 32, 64, 819, 820))
        self.assertEqual(grid.call_args.kwargs["kind"], "v41_fp8")

    def test_grouped_grid_and_small_capacity_stay_bounded(self):
        with mock.patch.object(
            warmup.common, "_generate_dense_gemm_warmup_m_grid", return_value=(1, 4)
        ) as grid:
            result = warmup._dense_m_grid(4, 256, 4096, 152, groups=8)
        self.assertEqual(result, (1, 2, 3, 4))
        self.assertEqual(grid.call_args.kwargs["kind"], "v41_fp8_batched")
        self.assertEqual(grid.call_args.kwargs["num_groups"], 8)

    def test_v41_single_accumulator_allows_n256_without_changing_v4(self):
        kwargs = dict(m_value=1024, n_value=32768, k_value=1536, num_sms=152)
        current = warmup.common._sm100_dense_layout_signature(kind="v41_fp8", **kwargs)
        legacy = warmup.common._sm100_dense_layout_signature(kind="fp8", **kwargs)
        self.assertEqual(current[:6], (0, 128, 256, 128, 2, 1))
        self.assertEqual(legacy[:6], (0, 128, 224, 128, 2, 1))
        grouped = warmup.common._sm100_dense_layout_signature(
            kind="v41_fp8_batched", num_groups=8, **kwargs
        )
        self.assertEqual(grouped[-2:], (8, 1))

    def dense_fixture(self, *, fails=False):
        operation = mock.Mock(
            side_effect=RuntimeError("compile failed") if fails else None
        )
        output_name = "rtp_llm.models_py.modules.dsv4.fp8._v41_output_projection"
        output = fake_module(output_name, is_supported=mock.Mock(return_value=False))
        package_name = output_name.rsplit(".", 1)[0]
        package = fake_module(package_name, _v41_output_projection=output)
        package.__path__ = []
        self.stack.enter_context(
            mock.patch.dict(sys.modules, {package_name: package, output_name: output})
        )
        replacements = {
            "_is_cuda_device": lambda device: True,
            "_assert_not_capturing": lambda: None,
            "_get_deep_gemm_num_sms": lambda device: 152,
            "_sync_cuda": lambda device: None,
            "_release_cuda_cache": lambda device: None,
            "_run_deepgemm_warmup_launches_serialized": lambda label, launch: launch(),
            "_run_deepgemm_warmup_launch_with_retry": lambda label, desc, launch, **kw: launch(),
        }
        for name, value in replacements.items():
            self.stack.enter_context(mock.patch.object(warmup.common, name, value))
        self.stack.enter_context(
            mock.patch.object(
                warmup,
                "collect_v41_dense_shapes",
                return_value=({(32, 128): ("q", operation)}, {}),
            )
        )
        self.stack.enter_context(
            mock.patch.object(warmup, "_dense_m_grid", return_value=(1, 3))
        )
        self.stack.enter_context(
            mock.patch.object(
                warmup.torch,
                "zeros",
                side_effect=lambda size, **kw: torch.empty(size, dtype=kw["dtype"]),
            )
        )
        return operation

    def test_dense_launches_actual_wrapper_and_memoizes_only_success(self):
        operation = self.dense_fixture()
        warmup.warmup_v41_dense_jit(object(), max_m=4, device="cuda:0")
        self.assertEqual(
            [tuple(c.args[0].shape) for c in operation.call_args_list],
            [(1, 128), (3, 128)],
        )
        self.assertTrue(
            all(c.args[0].dtype == torch.bfloat16 for c in operation.call_args_list)
        )
        warmup.warmup_v41_dense_jit(object(), max_m=4, device="cuda:0")
        self.assertEqual(operation.call_count, 2)
        # Retired A/B flags must not invalidate an otherwise identical warmup.
        with mock.patch.dict(os.environ, {"DSV41_FUSED_OUTPUT_PROJECTION": "0"}):
            warmup.warmup_v41_dense_jit(object(), max_m=4, device="cuda:0")
        self.assertEqual(operation.call_count, 2)
        with mock.patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "v2"}):
            warmup.warmup_v41_dense_jit(object(), max_m=4, device="cuda:0")
        self.assertEqual(operation.call_count, 4)

    def test_dense_failed_compile_does_not_poison_retry(self):
        operation = self.dense_fixture(fails=True)
        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            warmup.warmup_v41_dense_jit(object(), max_m=4, device="cuda:0")
        self.assertFalse(warmup._DENSE_WARMED)
        operation.side_effect = None
        warmup.warmup_v41_dense_jit(object(), max_m=4, device="cuda:0")
        self.assertEqual(len(warmup._DENSE_WARMED), 1)
        self.assertEqual(operation.call_count, 3)

    def test_dense_failed_sync_does_not_memoize(self):
        self.dense_fixture()
        with mock.patch.object(
            warmup.common, "_sync_cuda", side_effect=RuntimeError("sync failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "sync failed"):
                warmup.warmup_v41_dense_jit(object(), max_m=4, device="cuda:0")
        self.assertFalse(warmup._DENSE_WARMED)

    def prefill_fixture(self):
        calls = {}
        for name in (
            "_assert_not_capturing",
            "_sync_cuda",
            "_release_cuda_cache",
            "warmup_prefill_cp_metadata_jit",
            "warmup_dense_gemm_jit",
        ):
            calls[name] = self.stack.enter_context(
                mock.patch.object(warmup.common, name)
            )
        calls["legacy_shapes"] = [("shared_expert", 512, 1024)]
        self.stack.enter_context(
            mock.patch.object(
                warmup.common,
                "_collect_dsv4_dense_gemm_shapes",
                return_value=calls["legacy_shapes"],
            )
        )
        self.stack.enter_context(
            mock.patch.object(warmup.common, "_is_cuda_device", return_value=True)
        )
        self.stack.enter_context(
            mock.patch.object(warmup.common, "_dist_rank", return_value=6)
        )
        for name in (
            "warmup_v41_dense_jit",
            "warmup_v41_shared_expert_jit",
            "warmup_v41_engram_jit",
        ):
            calls[name] = self.stack.enter_context(mock.patch.object(warmup, name))
        calls["attention"] = mock.Mock()
        calls["hc"] = mock.Mock()
        modules = {
            "rtp_llm.models_py.modules.dsv4.fp8._v41_attention_jit_warmup": fake_module(
                "attention", warmup_v41_attention_jit=calls["attention"]
            ),
            "rtp_llm.models_py.modules.dsv4.hc.v41_jit_warmup": fake_module(
                "hc", warmup_v41_hc_jit=calls["hc"]
            ),
        }
        self.stack.enter_context(mock.patch.dict(sys.modules, modules))
        model = types.SimpleNamespace(
            _is_decode_role=False,
            parallelism_config=types.SimpleNamespace(
                tp_size=4,
                prefill_cp_config=types.SimpleNamespace(
                    is_enabled=lambda: True, kv_cache_sharded=True
                ),
            ),
            _max_context_batch_size=8,
            _max_generate_batch_size=16,
            _resolve_prefill_q_token_capacity=lambda: 131080,
            _v4_args=types.SimpleNamespace(
                max_seq_len=524288, max_tokens_per_rank=4096
            ),
            v4=object(),
            kv_cache=None,
        )
        return model, calls

    def test_prefill_capacity_cp_rank_and_second_cache_binding(self):
        model, calls = self.prefill_fixture()
        warmup.warmup_v41_prefill_jit(model, device="cuda:0")
        cache = object()
        model.kv_cache = cache
        warmup.warmup_v41_prefill_jit(model, device="cuda:0")
        self.assertEqual(calls["attention"].call_count, 2)
        before, after = calls["attention"].call_args_list
        self.assertIsNone(before.kwargs["kv_cache"])
        self.assertIs(after.kwargs["kv_cache"], cache)
        self.assertEqual(after.kwargs["max_m"], 131080)
        self.assertEqual(after.kwargs["max_batch_size"], 16)
        self.assertEqual(after.kwargs["cp_size"], 4)
        self.assertEqual(after.kwargs["cp_rank"], 2)
        self.assertTrue(after.kwargs["kv_cache_sharded"])
        self.assertFalse(
            calls["warmup_prefill_cp_metadata_jit"].call_args.kwargs["fp8_kv_cache"]
        )
        self.assertEqual(calls["hc"].call_args.kwargs["max_m"], 131080)
        calls["warmup_v41_shared_expert_jit"].assert_called_with(
            model, max_m=131080, device=torch.device("cuda:0")
        )
        self.assertEqual(
            calls["warmup_dense_gemm_jit"].call_args.args, (calls["legacy_shapes"],)
        )
        self.assertEqual(
            calls["warmup_dense_gemm_jit"].call_args.kwargs["max_m"], 131080
        )

    def test_prefill_row_limit_tracks_scheduler_tokens_cp_and_concurrency(self):
        model, _ = self.prefill_fixture()
        cases = (
            # token budget, CP, context cap, generation cap, expected local rows
            (4096, 1, 1, 16, 4096),
            (4096, 4, 1, 16, 1054),
            (4096, 4, 1, 1, 1024),
            (4097, 4, 8, 4, 1040),
            (4, 4, 16, 16, 8),
            (1, 8, 16, 16, 2),
            (524288, 4, 1, 16, 131080),
            (0, 4, 1, 16, 131080),
        )
        for tokens, cp, context, generation, expected in cases:
            with self.subTest(tokens=tokens, cp=cp, context=context, gen=generation):
                model._max_prefill_batch_tokens = tokens
                model._max_context_batch_size = context
                model._max_generate_batch_size = generation
                self.assertEqual(
                    warmup.resolve_v41_prefill_warmup_max_m(model, cp), expected
                )

    def test_prefill_passes_one_runtime_bound_to_each_kernel_family(self):
        model, calls = self.prefill_fixture()
        model._max_prefill_batch_tokens = 4096
        warmup.warmup_v41_prefill_jit(model, device="cuda:0")
        for name in (
            "warmup_dense_gemm_jit",
            "warmup_v41_dense_jit",
            "warmup_v41_shared_expert_jit",
            "warmup_v41_engram_jit",
            "hc",
            "attention",
        ):
            self.assertEqual(calls[name].call_args.kwargs["max_m"], 1054, name)
        self.assertEqual(calls["attention"].call_args.kwargs["max_batch_size"], 16)

    def engram_fixture(self):
        layout = types.SimpleNamespace(
            layer_ids=(1,), n_hash_cols=64, head_dim=32, max_ngram_size=3
        )
        state = types.SimpleNamespace(layout=layout)
        state.hash_token_windows = mock.Mock(
            side_effect=RuntimeError("hash compile failed")
        )
        embed = mock.Mock()
        embed.weight = torch.empty((2, 32))
        embed._num_sms = 2
        layer = types.SimpleNamespace(
            embed_tokens=embed,
            q_weight=torch.ones((4, 8)),
            k_weight=torch.ones((4, 8)),
            eps=1e-20,
            layer_hash_index=0,
        )
        model = types.SimpleNamespace(
            _engram_hash_state=state,
            v4=types.SimpleNamespace(layers=[types.SimpleNamespace(engram=layer)]),
        )
        gate = mock.Mock()
        name = "rtp_llm.models_py.modules.dsv4.engram"
        self.stack.enter_context(
            mock.patch.dict(
                sys.modules, {name: fake_module(name, gated_engram_residual=gate)}
            )
        )
        for name, replacement in {
            "_is_cuda_device": lambda device: True,
            "_assert_not_capturing": lambda: None,
            "_get_deep_gemm_num_sms": mock.Mock(
                side_effect=AssertionError("lookup must use HostEngramEmbedding SMs")
            ),
            "_sync_cuda": lambda device: None,
            "_run_triton_warmup_launch_with_retry": lambda label, desc, launch, **kw: launch(),
        }.items():
            self.stack.enter_context(
                mock.patch.object(warmup.common, name, replacement)
            )
        cpu_zeros, cpu_ones = torch.zeros, torch.ones
        self.stack.enter_context(
            mock.patch.object(
                warmup.torch,
                "zeros",
                side_effect=lambda *args, **kw: cpu_zeros(
                    *args, **{k: v for k, v in kw.items() if k != "device"}
                ),
            )
        )
        self.stack.enter_context(
            mock.patch.object(
                warmup.torch,
                "ones",
                side_effect=lambda *args, **kw: cpu_ones(
                    *args, **{k: v for k, v in kw.items() if k != "device"}
                ),
            )
        )
        return model, state, embed, gate

    def test_engram_failure_is_not_memoized_and_retry_finishes(self):
        model, state, embed, gate = self.engram_fixture()
        with self.assertRaisesRegex(RuntimeError, "hash compile failed"):
            warmup.warmup_v41_engram_jit(model, max_m=3, device="cuda:0")
        self.assertFalse(warmup._ENGRAM_WARMED)
        state.hash_token_windows.side_effect = lambda windows, **kw: torch.empty(
            (windows.shape[0], 1, 64), dtype=torch.int64
        )
        warmup.warmup_v41_engram_jit(model, max_m=3, device="cuda:0")
        self.assertEqual(len(warmup._ENGRAM_WARMED), 1)
        self.assertEqual(embed.call_count, 3)
        self.assertEqual(gate.call_count, 2)
        self.assertIsNone(gate.call_args_list[0].args[-1])
        self.assertEqual(gate.call_args_list[1].args[-1].dtype, torch.bool)
        warmup.warmup_v41_engram_jit(model, max_m=3, device="cuda:0")
        self.assertEqual(embed.call_count, 3)

    def test_engram_uses_physical_sm_limit_and_memoizes_it(self):
        model, state, embed, _ = self.engram_fixture()
        embed._num_sms = 148
        state.hash_token_windows.side_effect = lambda windows, **kw: torch.empty(
            (windows.shape[0], 1, 64), dtype=torch.int64
        )
        warmup.warmup_v41_engram_jit(model, max_m=128, device="cuda:0")
        # Physical lookup saturation is ceil(148*16/64)=37 rows; include the
        # following 16 rows for runtime scalar/alignment variants.
        self.assertEqual(embed.call_count, 53)
        self.assertEqual(
            [c.args[0].shape[0] for c in embed.call_args_list], list(range(1, 54))
        )
        warmup.warmup_v41_engram_jit(model, max_m=128, device="cuda:0")
        self.assertEqual(embed.call_count, 53)
        embed._num_sms = 132
        warmup.warmup_v41_engram_jit(model, max_m=128, device="cuda:0")
        self.assertEqual(embed.call_count, 53 + 49)
        self.assertEqual(len(warmup._ENGRAM_WARMED), 2)

    def shared_fixture(self):
        model = nn.Module()
        shared = shaped_module(
            "W13SharedExpert",
            w13=shaped_module("V41MXFP8Linear", N=24, K=8),
            w2=shaped_module("V41MXFP8Linear", N=8, K=12),
            swiglu_limit=7.0,
        )
        model.moe = shaped_module(
            "MoE",
            shared_experts=shared,
            _shared_executor=types.SimpleNamespace(name="mxfp8", _out=object()),
            _routed_includes_shared=False,
            _strategy=types.SimpleNamespace(
                name="mega",
                _mega_y=torch.full((8, 16), 3.0, dtype=torch.bfloat16)[:, ::2],
            ),
        )
        silu, combine = mock.Mock(), mock.Mock()
        base = "rtp_llm.models_py.modules.dsv4"
        modules = {
            f"{base}.moe._silu_mul_bf16_triton": fake_module(
                "silu", silu_mul_split_bf16=silu
            ),
            f"{base}.moe._shared_expert_triton": fake_module(
                "combine", fused_moe_epilogue=combine
            ),
        }
        self.stack.enter_context(mock.patch.dict(sys.modules, modules))
        for name, replacement in {
            "_is_cuda_device": lambda device: True,
            "_assert_not_capturing": lambda: None,
            "_sync_cuda": lambda device: None,
            "_run_triton_warmup_launch_with_retry": lambda label, desc, launch, **kw: launch(),
        }.items():
            self.stack.enter_context(
                mock.patch.object(warmup.common, name, replacement)
            )
        for name in ("zeros", "empty_strided"):
            factory = getattr(torch, name)
            self.stack.enter_context(
                mock.patch.object(
                    torch,
                    name,
                    side_effect=lambda *args, _factory=factory, **kw: _factory(
                        *args, **{k: v for k, v in kw.items() if k != "device"}
                    ),
                )
            )
        return model, silu, combine

    def test_shared_private_shapes_dtype_stride_and_success_memo(self):
        model, silu, combine = self.shared_fixture()
        # Another layer with the same kernel contract must not duplicate JIT.
        model.other = copy.deepcopy(model.moe)
        original_output = model.moe._shared_executor._out
        original_routed = model.moe._strategy._mega_y.clone()
        warmup.warmup_v41_shared_expert_jit(model, max_m=8192, device="cuda:0")
        self.assertEqual(
            [tuple(c.args[0].shape) for c in silu.call_args_list],
            [(1, 24)],
        )
        for call in silu.call_args_list:
            (gate_up,) = call.args
            self.assertEqual(gate_up.dtype, torch.bfloat16)
            self.assertTrue(gate_up.is_contiguous())
            self.assertEqual(call.kwargs, {"clamp_limit": 7.0})
        combine.assert_called_once()
        routed, shared, out_dtype = combine.call_args.args
        self.assertEqual(routed.shape, (2, 8))
        self.assertEqual(routed.stride(), (16, 2))
        self.assertEqual(routed.dtype, torch.bfloat16)
        self.assertEqual(shared.stride(), (8, 1))
        self.assertEqual(shared.dtype, torch.bfloat16)
        self.assertEqual(out_dtype, torch.bfloat16)
        self.assertNotEqual(routed.data_ptr(), model.moe._strategy._mega_y.data_ptr())
        self.assertIs(model.moe._shared_executor._out, original_output)
        torch.testing.assert_close(model.moe._strategy._mega_y, original_routed)
        warmup.warmup_v41_shared_expert_jit(model, max_m=8192, device="cuda:0")
        self.assertEqual(silu.call_count, 1)
        combine.assert_called_once()

    def test_shared_minimum_capacity_and_fp32_routed(self):
        model, silu, combine = self.shared_fixture()
        model.moe._strategy = types.SimpleNamespace(name="grouped_fp4")
        model.moe.shared_experts.swiglu_limit = 0.0
        warmup.warmup_v41_shared_expert_jit(model, max_m=1, device="cuda:0")
        self.assertEqual(silu.call_count, 1)
        self.assertEqual(silu.call_args.args[0].shape, (1, 24))
        self.assertEqual(silu.call_args.kwargs["clamp_limit"], 0.0)
        self.assertEqual(combine.call_args.args[0].dtype, torch.float32)

    def test_shared_failure_and_failed_sync_do_not_memoize(self):
        model, silu, combine = self.shared_fixture()
        silu.side_effect = RuntimeError("silu compile failed")
        with self.assertRaisesRegex(RuntimeError, "silu compile failed"):
            warmup.warmup_v41_shared_expert_jit(model, max_m=16, device="cuda:0")
        self.assertFalse(warmup._SHARED_WARMED)
        combine.assert_not_called()
        silu.side_effect = None
        with mock.patch.object(
            warmup.common, "_sync_cuda", side_effect=RuntimeError("sync failed")
        ), self.assertRaisesRegex(RuntimeError, "sync failed"):
            warmup.warmup_v41_shared_expert_jit(model, max_m=16, device="cuda:0")
        self.assertFalse(warmup._SHARED_WARMED)
        warmup.warmup_v41_shared_expert_jit(model, max_m=16, device="cuda:0")
        self.assertEqual(len(warmup._SHARED_WARMED), 1)

    def test_shared_fused_strategy_skip_and_eager_add_switch(self):
        model, silu, combine = self.shared_fixture()
        model.moe._routed_includes_shared = True
        warmup.warmup_v41_shared_expert_jit(model, max_m=16, device="cuda:0")
        silu.assert_not_called()
        combine.assert_not_called()
        self.assertFalse(warmup._SHARED_WARMED)
        model.moe._routed_includes_shared = False
        with mock.patch.dict(os.environ, {"DSV4_SHARED_EXPERT_BF16_ADD": "1"}):
            warmup.warmup_v41_shared_expert_jit(model, max_m=16, device="cuda:0")
        self.assertEqual(silu.call_count, 1)
        combine.assert_not_called()
        warmup.warmup_v41_shared_expert_jit(model, max_m=16, device="cuda:0")
        combine.assert_called_once()

    def test_commit_only_has_no_shared_expert_launches(self):
        model, silu, combine = self.shared_fixture()
        commit = nn.Module()
        commit.layers = nn.ModuleList([nn.Module()])
        commit.layers[0].attn = nn.Module()
        warmup.warmup_v41_shared_expert_jit(commit, max_m=512, device="cuda:0")
        silu.assert_not_called()
        combine.assert_not_called()
        self.assertFalse(warmup._SHARED_WARMED)

    def test_shared_geometry_change_rewarms_and_invalid_geometry_fails(self):
        model, silu, combine = self.shared_fixture()
        warmup.warmup_v41_shared_expert_jit(model, max_m=4, device="cuda:0")
        self.assertEqual(silu.call_count, 1)
        model.moe.shared_experts.w13.N = 32
        model.moe.shared_experts.w2.K = 16
        model.moe.shared_experts.swiglu_limit = 0.0
        warmup.warmup_v41_shared_expert_jit(model, max_m=32, device="cuda:0")
        self.assertEqual(silu.call_count, 2)
        self.assertEqual(silu.call_args.args[0].shape, (1, 32))
        self.assertEqual(silu.call_args.kwargs["clamp_limit"], 0.0)
        self.assertEqual(len(warmup._SHARED_WARMED), 2)
        model.moe.shared_experts.w13.N = 31
        with self.assertRaisesRegex(ValueError, "geometry"):
            warmup.warmup_v41_shared_expert_jit(model, max_m=32, device="cuda:0")
        self.assertEqual(silu.call_count, 2)

    def test_prefill_failure_propagates_before_final_sync(self):
        model, calls = self.prefill_fixture()
        calls["attention"].side_effect = RuntimeError("attention compile failed")
        with self.assertRaisesRegex(RuntimeError, "attention compile failed"):
            warmup.warmup_v41_prefill_jit(model, device="cuda:0")
        calls["_sync_cuda"].assert_not_called()
        calls["_release_cuda_cache"].assert_not_called()

    def test_non_cp_topology_does_not_use_tensor_parallel_size(self):
        model = types.SimpleNamespace(
            parallelism_config=types.SimpleNamespace(tp_size=4, prefill_cp_config=None)
        )
        self.assertEqual(warmup._prefill_topology(model), (1, False))

    def test_initialize_materialized_branch_warms_after_base_binds_cache(self):
        source = _REPO / "rtp_llm/models_py/model_desc/deepseek_v4_model.py"
        tree = ast.parse(source.read_text())
        method = copy.deepcopy(
            next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "_initialize_impl"
            )
        )
        for arg in method.args.args:
            arg.annotation = None
        method.returns = None
        # Execute the real early-return branch without importing model bindings.
        cls = ast.ClassDef(
            name="Extracted",
            bases=[ast.Name(id="Base", ctx=ast.Load())],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))

        class Base:
            def initialize(self, resource):
                self.kv_cache = resource.kv_cache

        namespace = {"Base": Base}
        exec(compile(module, str(source), "exec"), namespace)
        instance = namespace["Extracted"]()
        instance._materialized = True
        for commit_only in (False, True):
            with self.subTest(commit_only=commit_only):
                instance._v4_args = types.SimpleNamespace(
                    v41_config=object(), commit_only=commit_only
                )
                instance.v4 = types.SimpleNamespace(
                    embed=(
                        None
                        if commit_only
                        else types.SimpleNamespace(
                            weight=types.SimpleNamespace(device="cuda:0")
                        )
                    )
                )
                instance.main_proj = types.SimpleNamespace(
                    weight=types.SimpleNamespace(device="cuda:3")
                )
                cache = object()
                seen = []
                with mock.patch.object(
                    warmup,
                    "warmup_v41_prefill_jit",
                    side_effect=lambda model, **kw: seen.append(
                        (model.kv_cache, kw["device"])
                    ),
                ):
                    self.assertTrue(
                        instance._initialize_impl(types.SimpleNamespace(kv_cache=cache))
                    )
                self.assertEqual(seen, [(cache, "cuda:3" if commit_only else "cuda:0")])

    def test_commit_initial_warmup_occurs_after_fc_load_and_buffer_bind(self):
        source = _REPO / "rtp_llm/models_py/model_desc/deepseek_v4_model.py"
        method = copy.deepcopy(
            next(
                node
                for node in ast.walk(ast.parse(source.read_text()))
                if isinstance(node, ast.FunctionDef)
                and node.name == "_initialize_commit_only"
            )
        )
        for arg in method.args.args:
            arg.annotation = None
        method.returns = None
        tree = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        events = []
        attn = types.SimpleNamespace(init_rope_cache=lambda **kw: events.append("rope"))
        transformer = types.SimpleNamespace(
            layers=[types.SimpleNamespace(attn=attn)], embed=None
        )
        namespace = {
            "logging": mock.Mock(),
            "torch": torch,
            "model_build_scope": lambda token: nullcontext(),
            "feature_weights_region": nullcontext,
            "V4Transformer": lambda *args, **kw: transformer,
        }
        exec(compile(tree, str(source), "exec"), namespace)
        instance = types.SimpleNamespace(
            weight=types.SimpleNamespace(weights=[object()], global_weights={}),
            _build_scope_token=object(),
            _v4_args=types.SimpleNamespace(v41_config=object()),
            _resolve_prefill_q_token_capacity=lambda: 321,
            _materialized=False,
        )

        def load_fc(weight):
            self.assertIs(weight, instance.weight)
            instance._fc = object()
            events.append("fc")

        def bind(device):
            self.assertTrue(hasattr(instance, "_fc"))
            self.assertEqual(device, torch.device("cuda:3"))
            events.append("bind")

        def run_warmup(model, *, device):
            self.assertIs(model, instance)
            self.assertFalse(hasattr(model, "weight"))
            self.assertTrue(hasattr(model, "_fc"))
            self.assertFalse(model._materialized)
            self.assertEqual(device, torch.device("cuda:3"))
            events.append("warmup")

        instance._load_scoped_extra_weights = load_fc
        instance._bind_runtime_buffers = bind
        with mock.patch.object(
            warmup, "warmup_v41_prefill_jit", side_effect=run_warmup
        ):
            self.assertTrue(
                namespace["_initialize_commit_only"](instance, object(), "cuda:3")
            )
        self.assertEqual(events, ["rope", "fc", "bind", "warmup"])
        self.assertTrue(instance._materialized)


if __name__ == "__main__":
    unittest.main()
