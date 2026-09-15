import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv4.moe.mega_jit_warmup import mega_moe_config_signature
from rtp_llm.models_py.modules.kimi_k3.kernel_jit_warmup import (
    _kimi_k3_nvcc_rank_tmpdir,
    build_kimi_k3_kernel_warmup_plan,
    generate_kimi_k3_mega_moe_token_counts,
    kimi_k3_kernel_jit_warmup_enabled,
)


class _Parallelism:
    def __init__(self, tp_size):
        self.tp_size = tp_size

    def get_attn_tp_size(self):
        return self.tp_size


class _Mega(nn.Module):
    def __init__(self, capacity=8192):
        super().__init__()
        self._mega_l1_w = torch.empty(0)
        self._mega_l2_w = torch.empty(0)
        self._mega_buf = SimpleNamespace(num_max_tokens_per_rank=capacity)
        self.expert_num = 384
        self.local_expert_count = 48
        self.top_k = 8
        self.latent_size = 3584
        self._mega_intermediate_size = 3072
        self.ep_size = 8


class _Model(nn.Module):
    def __init__(self, *, tp_size=8, max_seq_len=1 << 20, max_batch=16):
        super().__init__()
        self.parallelism_config = _Parallelism(tp_size)
        self.config = SimpleNamespace(max_seq_len=max_seq_len, gen_num_per_cycle=5)
        self._max_generate_batch_size = max_batch
        self.mega = _Mega()


class KimiK3KernelJitWarmupTest(unittest.TestCase):
    def test_default_enabled_and_explicit_disable(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(kimi_k3_kernel_jit_warmup_enabled())
        with mock.patch.dict(
            os.environ, {"KIMI_K3_STARTUP_REAL_WARMUP": "off"}, clear=True
        ):
            self.assertFalse(kimi_k3_kernel_jit_warmup_enabled())

    def test_mega_token_counts_cover_each_signature_and_exact_cap(self):
        kwargs = dict(
            num_ranks=8,
            num_experts=384,
            num_experts_per_rank=48,
            num_topk=8,
            intermediate_hidden=3072,
            num_sms=132,
            max_tokens_per_rank=8192,
        )
        counts = generate_kimi_k3_mega_moe_token_counts(**kwargs)
        self.assertEqual(counts[-1], 8192)
        signature_kwargs = {
            key: value for key, value in kwargs.items() if key != "max_tokens_per_rank"
        }
        actual_signatures = {
            mega_moe_config_signature(num_tokens=count, **signature_kwargs)
            for count in counts
        }
        reachable_signatures = {
            mega_moe_config_signature(num_tokens=count, **signature_kwargs)
            for count in range(1, 8193)
        }
        self.assertEqual(actual_signatures, reachable_signatures)

    def test_prefill_plan_is_chunk_and_tp_aware(self):
        model = _Model()
        init_resource = SimpleNamespace(
            is_decode_role=False,
            max_context_batch_size=1,
            max_decode_graph_batch_size=1,
        )
        with mock.patch.dict(
            os.environ, {"KIMI_K3_PREFILL_CHUNK_TOKENS": "65536"}, clear=True
        ):
            plan = build_kimi_k3_kernel_warmup_plan(model, init_resource, num_sms=132)
        self.assertEqual(plan.dense_max_m, 65536)
        self.assertEqual(plan.mega_max_tokens_per_rank, 8192)
        self.assertEqual(plan.mega_token_counts[-1], 8192)

    def test_decode_plan_uses_graph_batch_and_speculative_width(self):
        model = _Model(max_batch=8)
        init_resource = SimpleNamespace(
            is_decode_role=True,
            max_context_batch_size=1,
            max_decode_graph_batch_size=16,
            decode_capture_batch_sizes=(1, 2, 4, 8, 16),
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            plan = build_kimi_k3_kernel_warmup_plan(model, init_resource, num_sms=132)
        self.assertEqual(plan.dense_max_m, 96)
        self.assertEqual(plan.mega_max_tokens_per_rank, 12)
        self.assertEqual(plan.mega_token_counts[-1], 12)
        self.assertEqual(plan.decode_batch_sizes, (1, 2, 4, 8, 16))

    def test_nvcc_tmpdir_uses_shared_jit_cache_without_k3_override(self):
        with mock.patch.dict(
            os.environ,
            {
                "DG_JIT_CACHE_DIR": "/cache/deepgemm",
                "TRITON_CACHE_DIR": "/cache/triton",
            },
            clear=True,
        ):
            self.assertEqual(
                _kimi_k3_nvcc_rank_tmpdir(3),
                "/cache/deepgemm/rtp_llm_kimi_k3_mega_moe_nvcc/rank_3",
            )


if __name__ == "__main__":
    unittest.main()
