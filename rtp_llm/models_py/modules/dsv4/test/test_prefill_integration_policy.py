"""Execute projection and warmup dispatch with CPU tensors and provider doubles."""

from __future__ import annotations

import inspect
import re
import sys
import types
import unittest
from unittest.mock import Mock, call, patch

import torch

from rtp_llm.models_py.modules.dsv4 import dsv4_kernel_jit_warmup as warmup_module
from rtp_llm.models_py.modules.dsv4.fp8 import (
    _indexer_score as idx_score_mod,
    attention as attn_mod,
    attention_backend,
)
from rtp_llm.models_py.modules.dsv4.fp8.sm120_attention_backend import (
    Sm120AttentionBackend,
)


class IntegrationPolicyTest(unittest.TestCase):
    def test_every_dummy_launcher_reference_is_defined(self):
        """Regression: every ``_launch_dummy_*`` name the warmup module
        references must be defined at module level — a merge once dropped
        ``_launch_dummy_mhc_pre_wrapper`` while its call site survived."""
        source = inspect.getsource(warmup_module)
        referenced = set(re.findall(r"\b(_launch_dummy_[A-Za-z0-9_]+)\b", source))
        self.assertTrue(referenced)
        for name in referenced:
            self.assertTrue(
                callable(getattr(warmup_module, name, None)),
                f"{name} referenced in warmup module is not callable at module scope",
            )

    def test_woa_and_warmup_choose_same_arm_at_invocation(self):
        for sm120 in (False, True):
            with self.subTest(sm120=sm120):
                self._check_dispatch(sm120)

    def test_mqa_warmup_uses_runtime_provider_capability(self):
        for supported in (False, True):
            with self.subTest(supported=supported):
                capability = Mock(return_value=supported)
                launch = Mock()
                cpu = torch.device("cpu")
                with (
                    patch.object(
                        warmup_module, "model_warm_up_enabled", return_value=True
                    ),
                    patch.object(warmup_module, "_is_cuda_device", return_value=True),
                    patch.object(warmup_module, "_assert_not_capturing"),
                    patch.object(
                        warmup_module, "_FP8_MQA_LOGITS_JIT_WARMED_KEYS", set()
                    ),
                    patch.object(warmup_module, "_dist_rank", return_value=0),
                    patch.object(
                        warmup_module,
                        "_run_deepgemm_warmup_launch_with_retry",
                        side_effect=lambda name, label, fn, **kw: fn(),
                    ),
                    patch.object(
                        warmup_module,
                        "_run_deepgemm_warmup_launches_serialized",
                        side_effect=lambda name, fn: fn(),
                    ),
                    patch.object(
                        warmup_module, "_launch_dummy_fp8_mqa_logits", launch
                    ),
                    patch.object(warmup_module, "_sync_cuda"),
                    patch.object(warmup_module, "_release_cuda_cache"),
                    patch.object(
                        idx_score_mod, "supports_dsv4_mqa_logits", capability
                    ),
                ):
                    for _ in range(2):
                        warmup_module.warmup_fp8_mqa_logits_jit(
                            {(8, 128): {}}, device=cpu
                        )
                self.assertEqual(launch.call_count, int(supported))
                expected_calls = (
                    [call(cpu), call(cpu, num_heads=8)] * 2
                    if supported
                    else [call(cpu)] * 2
                )
                self.assertEqual(capability.call_args_list, expected_calls)

    def _check_dispatch(self, sm120):
        m, groups, width, rank = 3, 2, 512, 128
        activation = torch.ones(m, groups, width, dtype=torch.float8_e4m3fn)
        scales = torch.empty(m, groups, 1, dtype=torch.int32).fill_(0x7F7F7F7F)
        packed_k = (width + 511) // 512
        weight_scales = torch.empty(
            rank * packed_k * groups, dtype=torch.int32
        ).as_strided((groups, rank, packed_k), (rank * packed_k, 1, rank))
        weight_scales.fill_(0x7F7F7F7F)
        owner = types.SimpleNamespace(
            o_lora_rank=rank,
            _wo_a_stk_w=torch.ones(groups, rank, width, dtype=torch.float8_e4m3fn),
            _wo_a_stk_s=weight_scales,
        )

        def einsum(equation, inputs, weights, out, **kwargs):
            self.assertEqual(equation, "bhr,hdr->bhd")
            self.assertIs(inputs[0], activation)
            self.assertIs(inputs[1], scales)
            self.assertIs(weights[0], owner._wo_a_stk_w)
            self.assertIs(weights[1], owner._wo_a_stk_s)
            self.assertEqual(kwargs["recipe"], (1, 1, 128))
            out.fill_(7)

        deep_gemm = types.SimpleNamespace(fp8_einsum=Mock(side_effect=einsum))
        launch = Mock()
        key, info = (groups, rank, width), {"weights": owner._wo_a_stk_w}
        selected = (
            Sm120AttentionBackend()
            if sm120
            else attention_backend.SparseAttentionBackend()
        )
        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(
                attention_backend,
                "get_sparse_attention_backend",
                return_value=selected,
            ),
            patch.object(warmup_module, "model_warm_up_enabled", return_value=True),
            patch.object(warmup_module, "_is_cuda_device", return_value=True),
            patch.object(warmup_module, "_assert_not_capturing"),
            patch.object(warmup_module, "_get_deep_gemm_num_sms", return_value=128),
            patch.object(
                warmup_module,
                "_generate_dense_gemm_warmup_m_grid",
                return_value=(1, m),
            ),
            patch.object(warmup_module, "_BATCHED_FP8_EINSUM_JIT_WARMED_KEYS", set()),
            patch.object(warmup_module, "_dist_rank", return_value=0),
            patch.object(
                warmup_module,
                "_run_deepgemm_warmup_launch_with_retry",
                side_effect=lambda name, label, fn, **kw: fn(),
            ),
            patch.object(
                warmup_module,
                "_run_deepgemm_warmup_launches_serialized",
                side_effect=lambda name, fn: fn(),
            ),
            patch.object(warmup_module, "_launch_dummy_batched_fp8_einsum", launch),
            patch.object(warmup_module, "_sync_cuda"),
            patch.object(warmup_module, "_release_cuda_cache"),
        ):
            out = attn_mod.AttentionFP8._wo_a_einsum_from_fp8(
                owner, activation, scales, 1, m
            )
            warmup_module.warmup_batched_fp8_einsum_jit(
                {key: info}, max_m=m, device=torch.device("cpu")
            )
            warmup_module.warmup_batched_fp8_einsum_jit(
                {key: info}, max_m=m, device=torch.device("cpu")
            )
        self.assertTrue(
            torch.equal(out, torch.full((1, m, groups, rank), 7, dtype=torch.bfloat16))
        )
        self.assertEqual(deep_gemm.fp8_einsum.call_count, 1)
        self.assertEqual(launch.call_count, 2)
        self.assertEqual(
            [c.kwargs["m_value"] for c in launch.call_args_list], [1, m]
        )
        for c in launch.call_args_list:
            self.assertEqual(c.kwargs["key"], key)
            self.assertIs(c.kwargs["info"], info)


if __name__ == "__main__":
    unittest.main(verbosity=2)
