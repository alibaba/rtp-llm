"""Decode graphs must not select an expert-capacity slab for eager prefill."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
    deepgemm_hybrid_executor as hybrid,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
    deepgemm_masked_executor_v2 as masked,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.deepgemm_sm90_prefill_silu_fp8_test import (
    make_prefill_payload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.deepgemm_sm90_scale_layout_test import (
    clone_payload,
    invoke,
    make_executor,
)


def execute(executor, payload):
    return executor.execute(
        payload, "SiGLU", None, None, False, None
    ).fused_expert_output


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DeepGemmSM90MixedGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (9, 0):
            raise unittest.SkipTest("SM90 required")
        torch.manual_seed(20261117)
        cls.reference = make_executor(graph=False)
        cls.reference._sm90_mn_major_scales = False
        cls.reference._sm90_prefill_silu_fp8 = False
        cls.config = cls.reference.config
        cls.config.enable_cuda_graph = True
        cls.config.moe_strategy = "auto"
        cls.config.model_config._qwen4_full_fp8 = True
        cls.config.model_config._qwen4_full_fp8_nvfp4 = False
        cls.candidate = masked.DeepGemmMaskedExecutorV2(
            cls.config, cls.reference.quant_config, cls.reference.weights
        )
        assert cls.candidate._qwen4_sm90_mixed_graph

    def test_dispatch_boundary_and_graph_warmup(self):
        candidate = self.candidate
        limit = candidate.masked_max_token_num
        for rows in (1, 16, 64, limit, limit + 1, 8192, 32768):
            payload = SimpleNamespace(expert_x=torch.empty((rows, 1), device="meta"))
            for preparing in (False, True):
                with patch.object(
                    masked, "_is_cuda_graph_warmup_or_capture", return_value=preparing
                ), patch.object(candidate, "execute_masked") as old, patch.object(
                    candidate, "execute_contiguous"
                ) as compact:
                    candidate.execute(payload, "SiGLU", None, None, False, None)
                    use_compact = rows > limit and not preparing
                    self.assertEqual(compact.call_count, int(use_compact))
                    self.assertEqual(old.call_count, int(not use_compact))
                    if use_compact:
                        self.assertTrue(compact.call_args.kwargs["use_eager_layout"])
        self.assertTrue(candidate.enable_cuda_graph)

    def test_auto_model_architecture_and_graph_guards(self):
        for attribute, value in (
            ("moe_strategy", "fp8_per_block_no_dp_masked"),
            ("enable_cuda_graph", False),
        ):
            with patch.object(self.config, attribute, value):
                other = masked.DeepGemmMaskedExecutorV2(
                    self.config, self.reference.quant_config, self.reference.weights
                )
                self.assertFalse(other._qwen4_sm90_mixed_graph)
        for attribute, value in (
            ("_qwen4_full_fp8", False),
            ("_qwen4_full_fp8_nvfp4", True),
        ):
            with patch.object(self.config.model_config, attribute, value):
                other = masked.DeepGemmMaskedExecutorV2(
                    self.config, self.reference.quant_config, self.reference.weights
                )
                self.assertFalse(other._qwen4_sm90_mixed_graph)
        for capability in ((8, 0), (10, 0), (12, 0)):
            with patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ):
                other = masked.DeepGemmMaskedExecutorV2(
                    self.config, self.reference.quant_config, self.reference.weights
                )
                self.assertFalse(other._qwen4_sm90_mixed_graph)
        with patch.object(
            hybrid, "_is_cuda_graph_warmup_or_capture", return_value=True
        ):
            with self.assertRaisesRegex(ValueError, "graph capture/warmup"):
                self.candidate.execute_contiguous(
                    None, "SiGLU", None, None, False, None, use_eager_layout=True
                )

    def test_large_prefill_native_result_and_compact_memory(self):
        for rows in (self.candidate.masked_max_token_num + 1, 8192, 32768):
            for routing in ("uniform", "skewed", "all_local", "sparse", "remote"):
                with self.subTest(rows=rows, routing=routing):
                    payload = make_prefill_payload(rows, routing)
                    expected = invoke(self.reference, clone_payload(payload), False)
                    # Exercise the real router contract: GPU counts only.
                    payload.expert_tokens_meta.expert_num_tokens_cpu = None
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    actual = execute(self.candidate, clone_payload(payload))
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_allocated() - before
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertLess(peak, 8 * 1024**3)
                    self.assertTrue(self.candidate.enable_cuda_graph)

    def test_decode_graph_replay_changed_routes(self):
        for rows in (1, 16, 64):
            payload = make_prefill_payload(rows)
            for _ in range(3):
                execute(self.candidate, payload)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = execute(self.candidate, payload)
            for routing in ("skewed", "all_local", "sparse", "remote"):
                fresh = make_prefill_payload(rows, routing)
                expected = invoke(self.reference, clone_payload(fresh), True)
                for attr in (
                    "expert_x",
                    "expert_x_scale",
                    "expert_topk_ids",
                    "expert_topk_weights",
                ):
                    getattr(payload, attr).copy_(getattr(fresh, attr))
                payload.expert_tokens_meta.expert_num_tokens.copy_(
                    fresh.expert_tokens_meta.expert_num_tokens
                )
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            graph.reset()


if __name__ == "__main__":
    unittest.main()
