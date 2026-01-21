import itertools
import logging
import time
import unittest
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from rtp_llm.models_py.modules.factory.attention.attn_factory import ConfigManager
from rtp_llm.models_py.modules.factory.attention.cuda_headwise_impl.headwise import (
    HeadWisePrefillAttnOp,
)
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs


class TestHeadwisePrefillOp(unittest.TestCase):
    """
    Test suite for HeadWisePrefillAttnOp with correctness verification
    using flex_attention as a reference.
    """

    # -----------------------------
    # Data structures (从原始代码移动)
    # -----------------------------
    @dataclass(frozen=True)
    class Case:
        batch_size: int
        kv_len: int
        qo_len: int
        window_left: int
        num_kv_heads: int
        num_qo_heads: int
        head_dim: int
        page_size: int

    class KVCache:
        def __init__(self, kv_cache_base: torch.Tensor):
            self.kv_cache_base = kv_cache_base

    class headwise_config:
        def __init__(self, config):
            self.headwise_config = config

    # -----------------------------
    # setUp & tearDown (unittest的生命周期方法)
    # -----------------------------
    def setUp(self):
        """在每个测试方法运行前执行"""
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        self.index_dtype = torch.int32
        self.sink_token_num = 4
        self.swa_token_num = 8192
        self.seqlen_threshold = 16384

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        # cache op/config
        self._cached_op_key: Optional[Tuple[int, int, int, int]] = None
        self._op: Optional[HeadWisePrefillAttnOp] = None

        # 编译 flex_attention 以加速参考计算
        self.compiled_flex_attention = torch.compile(flex_attention, dynamic=True)

    # -------------------------
    # Helper methods (从原始代码移动并调整)
    # -------------------------
    def _build_attention_config(self, case: Case) -> AttentionConfigs:
        cfg = AttentionConfigs()
        cfg.head_num = case.num_qo_heads
        cfg.kv_head_num = case.num_kv_heads
        cfg.size_per_head = case.head_dim
        cfg.tokens_per_block = case.page_size
        config = {
            "sink_token_num": self.sink_token_num,
            "swa_token_num": self.swa_token_num,
            "seqlen_threshold": self.seqlen_threshold,
            "0": [0] * case.num_qo_heads,
        }
        hcg = self.headwise_config(config)
        ConfigManager.set_headwise_config(hcg)
        return cfg

    @staticmethod
    def _build_parallelism_config() -> ParallelismConfig:
        pcfg = ParallelismConfig()
        pcfg.tp_rank = 0
        return pcfg

    def _get_or_create_op(self, case: Case) -> HeadWisePrefillAttnOp:
        key = (case.num_qo_heads, case.num_kv_heads, case.head_dim, case.page_size)
        if self._op is None or self._cached_op_key != key:
            with self.subTest(msg=f"Rebuilding op for case: {case}"):
                attn_cfg = self._build_attention_config(case)
                par_cfg = self._build_parallelism_config()
                self._op = HeadWisePrefillAttnOp(attn_cfg, par_cfg)
                self._cached_op_key = key
        return self._op

    def _make_attn_inputs(self, case: Case) -> PyAttentionInputs:
        attn_inputs = PyAttentionInputs()
        attn_inputs.input_lengths = torch.tensor(
            [case.qo_len] * case.batch_size, device=self.device, dtype=torch.int32
        )
        attn_inputs.prefix_lengths = torch.tensor(
            [case.kv_len - case.qo_len] * case.batch_size,
            device=self.device,
            dtype=torch.int32,
        )
        num_pages_per_seq = (case.kv_len + case.page_size - 1) // case.page_size
        total_num_pages = num_pages_per_seq * case.batch_size
        attn_inputs.kv_cache_block_id_device = torch.arange(
            0, total_num_pages, device=self.device, dtype=self.index_dtype
        ).view(case.batch_size, num_pages_per_seq)
        return attn_inputs

    def _make_paged_kv_cache(
        self, k: torch.Tensor, v: torch.Tensor, case: Case
    ) -> KVCache:
        num_pages_per_seq = (case.kv_len + case.page_size - 1) // case.page_size
        total_num_pages = num_pages_per_seq * case.batch_size
        k_cache = k.view(
            total_num_pages, case.page_size, case.num_kv_heads, case.head_dim
        ).transpose(1, 2)
        v_cache = v.view(
            total_num_pages, case.page_size, case.num_kv_heads, case.head_dim
        ).transpose(1, 2)
        kv_cache_base = torch.stack([k_cache, v_cache], dim=1)
        return self.KVCache(kv_cache_base)

    def _make_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, case: Case
    ) -> torch.Tensor:
        return torch.cat(
            [q, k[: case.batch_size * case.qo_len], v[: case.batch_size * case.qo_len]],
            dim=1,
        )

    def _make_flex_mask(self, s_q: int, s_k: int, window_left: int) -> torch.Tensor:
        ret_len = s_k - s_q

        def _sink_mask(b, h, q_idx, kv_idx):
            causal_window = q_idx + ret_len >= kv_idx
            sliding_window = q_idx + ret_len - kv_idx <= window_left
            sink_window = kv_idx < self.sink_token_num
            return (sliding_window | sink_window) & causal_window

        return create_block_mask(
            _sink_mask, None, None, s_q, s_k, device=self.device, _compile=False
        )

    def forward_flex(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, case: Case
    ) -> torch.Tensor:
        B, S_q, S_k = case.batch_size, case.qo_len, case.kv_len
        H_q, H_k, D = case.num_qo_heads, case.num_kv_heads, case.head_dim
        q_ = q.view(B, S_q, H_q, D).transpose(1, 2).contiguous()
        k_ = k.view(B, S_k, H_k, D).transpose(1, 2).contiguous()
        v_ = v.view(B, S_k, H_k, D).transpose(1, 2).contiguous()
        mask = self._make_flex_mask(S_q, S_k, case.window_left)
        out = self.compiled_flex_attention(q_, k_, v_, block_mask=mask, enable_gqa=True)
        return out.transpose(1, 2).reshape(B * S_q, H_q, D)

    def forward_headwise_op(
        self,
        case: Case,
        qkv: torch.Tensor,
        cache: KVCache,
        attn_inputs: PyAttentionInputs,
    ) -> torch.Tensor:
        op = self._get_or_create_op(case)
        op.support(
            attn_inputs
        )  # It's good practice to call support to check compatibility
        op.prepare(attn_inputs)
        op._get_headwise_config(0)
        return op.forward(qkv, cache, None)

    # -------------------------
    # Core Test Runner Logic
    # -------------------------
    def _run_correctness_check(self, case: Case, rtol=1e-2, atol=1e-2):
        """The core logic, similar to the original run_one."""
        logging.info(f"Running test for case: {case}")

        q = torch.randn(
            case.batch_size * case.qo_len,
            case.num_qo_heads,
            case.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        k = torch.randn(
            case.batch_size * case.kv_len,
            case.num_kv_heads,
            case.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        v = torch.randn(
            case.batch_size * case.kv_len,
            case.num_kv_heads,
            case.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        cache = self._make_paged_kv_cache(k, v, case)
        qkv = self._make_qkv(q, k, v, case)
        attn_inputs = self._make_attn_inputs(case)

        out_op = self.forward_headwise_op(case, qkv, cache, attn_inputs)
        out_ref = self.forward_flex(q, k, v, case)

        out_op_flat = out_op.reshape(case.batch_size * case.qo_len, -1)
        out_ref_flat = out_ref.reshape(case.batch_size * case.qo_len, -1)

        torch.testing.assert_close(out_ref_flat, out_op_flat, rtol=rtol, atol=atol)
        logging.info(f"✓ Test passed for case: {case}")

    # -------------------------
    # Test Cases (replaces the old main loop)
    # -------------------------
    def test_long_context_prefill(self):
        """Tests prefill with long context lengths (32k, 65k)."""
        logging.info("\n=== Testing Long Context Prefill ===")

        kv_lens = [32768, 65536]
        qo_lens = [32768, 65536]

        for i in range(len(kv_lens)):
            case = self.Case(
                batch_size=2,
                kv_len=kv_lens[i],
                qo_len=qo_lens[i],
                window_left=8192,
                num_kv_heads=1,
                num_qo_heads=8,
                head_dim=128,
                page_size=128,
            )

            with self.subTest(case=case):
                self._run_correctness_check(case)

    def test_various_page_sizes(self):
        """Tests different page sizes for memory layout."""
        logging.info("\n=== Testing Various Page Sizes ===")

        page_sizes = [128, 256, 512]
        for ps in page_sizes:
            case = self.Case(
                batch_size=2,
                kv_len=32768,
                qo_len=32768,
                window_left=8192,
                num_kv_heads=1,
                num_qo_heads=8,
                head_dim=128,
                page_size=ps,
            )
            with self.subTest(page_size=ps, case=case):
                self._run_correctness_check(case)

    def test_various_head(self):
        """Tests different page sizes for memory layout."""
        logging.info("\n=== Testing Various Page Sizes ===")

        qo_head = [8, 8, 8]
        kv_head = [1, 4, 8]
        for i in range(len(qo_head)):
            case = self.Case(
                batch_size=2,
                kv_len=32768,
                qo_len=32768,
                window_left=8192,
                num_kv_heads=kv_head[i],
                num_qo_heads=qo_head[i],
                head_dim=128,
                page_size=128,
            )
            with self.subTest(case=case):
                self._run_correctness_check(case)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    unittest.main()
