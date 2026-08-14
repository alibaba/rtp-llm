import math
import unittest
from typing import List
from unittest import mock

import torch
from flashinfer.cascade import merge_state
from flashinfer.prefill import single_prefill_with_kv_cache

from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferHybridPrefillAttnOp,
    PyFlashinferHybridPrefillImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    BaseAttentionTest,
    fill_paged_kv_cache,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType, RopeStyle
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, get_typemeta


class TestPyFlashinferHybridPrefillAttnOp(BaseAttentionTest):
    """Correctness tests for PyFlashinferHybridPrefillAttnOp."""

    def _create_prefix_kv_cache(
        self,
        k_prefix: List[torch.Tensor],
        v_prefix: List[torch.Tensor],
        prefix_lengths: List[int],
        sequence_lengths: List[int],
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        block_table: torch.Tensor,
        cache_dtype: torch.dtype,
    ) -> LayerKVCache:
        # Only the prefix is populated; the op under test writes the new tokens,
        # so the cache must still be sized for the full sequences.
        return fill_paged_kv_cache(
            k_prefix,
            v_prefix,
            prefix_lengths,
            block_table,
            page_size,
            num_kv_heads,
            head_dim,
            cache_dtype,
            self.device,
            total_pages=sum(
                math.ceil(seq_len / page_size) for seq_len in sequence_lengths
            ),
        )

    def _reference_chunked_prefill(
        self,
        q_new: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        prefix_len: int,
        input_len: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
    ) -> torch.Tensor:
        if q_dtype != q_new.dtype or kv_dtype != k_full.dtype:
            # Hybrid FP8 uses the same quantized Q/K/V for both attention states.
            q_quant = q_new.to(q_dtype).to(q_new.dtype)
            k_quant = k_full.to(kv_dtype).to(k_full.dtype)
            v_quant = v_full.to(kv_dtype).to(v_full.dtype)
            prefix_out, prefix_lse = single_prefill_with_kv_cache(
                q_quant,
                k_quant[:prefix_len],
                v_quant[:prefix_len],
                causal=False,
                kv_layout="NHD",
                return_lse=True,
            )
            new_out, new_lse = single_prefill_with_kv_cache(
                q_quant,
                k_quant[prefix_len:],
                v_quant[prefix_len:],
                causal=True,
                kv_layout="NHD",
                return_lse=True,
            )
            return merge_state(new_out, new_lse, prefix_out, prefix_lse)[0]
        q_full = torch.zeros(
            prefix_len + input_len,
            q_new.shape[1],
            q_new.shape[2],
            dtype=q_new.dtype,
            device=q_new.device,
        )
        q_full[prefix_len:] = q_new
        return single_prefill_with_kv_cache(
            q_full, k_full, v_full, causal=True, kv_layout="NHD"
        )[prefix_len:]

    def _test_hybrid_prefill_correctness(
        self,
        prefix_lengths: List[int],
        input_lengths: List[int],
        head_num: int,
        head_num_kv: int,
        size_per_head: int,
        page_size: int,
    ):
        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=page_size,
        )
        cache_dtype = self.cache_dtype(config.attn_configs)

        attn_inputs = self._create_chunked_prefill_attention_inputs(
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            seq_size_per_block=page_size,
        )
        attn_op = PyFlashinferHybridPrefillAttnOp(config.attn_configs, attn_inputs)
        self.assertTrue(attn_op.support(attn_inputs))
        attn_op.prepare(attn_inputs)

        q_chunks = []
        k_new_chunks = []
        v_new_chunks = []
        k_prefix_chunks = []
        v_prefix_chunks = []
        ref_chunks = []
        for prefix_len, input_len in zip(prefix_lengths, input_lengths):
            q_new = torch.randn(
                input_len,
                head_num,
                size_per_head,
                dtype=torch.float16,
                device=self.device,
            )
            k_prefix = torch.randn(
                prefix_len,
                head_num_kv,
                size_per_head,
                dtype=torch.float16,
                device=self.device,
            )
            v_prefix = torch.randn_like(k_prefix)
            # Round-trip through the cache dtype
            k_prefix = k_prefix.to(cache_dtype).to(k_prefix.dtype)
            v_prefix = v_prefix.to(cache_dtype).to(v_prefix.dtype)
            k_new = torch.randn(
                input_len,
                head_num_kv,
                size_per_head,
                dtype=torch.float16,
                device=self.device,
            )
            v_new = torch.randn_like(k_new)

            ref_chunks.append(
                self._reference_chunked_prefill(
                    q_new,
                    torch.cat([k_prefix, k_new], dim=0),
                    torch.cat([v_prefix, v_new], dim=0),
                    prefix_len,
                    input_len,
                    attn_op.q_dtype,
                    attn_op.kv_dtype,
                )
            )
            q_chunks.append(q_new)
            k_prefix_chunks.append(k_prefix)
            v_prefix_chunks.append(v_prefix)
            k_new_chunks.append(k_new)
            v_new_chunks.append(v_new)

        q = torch.cat(q_chunks, dim=0)
        k_new = torch.cat(k_new_chunks, dim=0)
        v_new = torch.cat(v_new_chunks, dim=0)
        ref_output = torch.cat(ref_chunks, dim=0)

        sequence_lengths = [p + i for p, i in zip(prefix_lengths, input_lengths)]
        kv_cache = self._create_prefix_kv_cache(
            k_prefix_chunks,
            v_prefix_chunks,
            prefix_lengths,
            sequence_lengths,
            page_size,
            head_num_kv,
            size_per_head,
            attn_inputs.kv_cache_kernel_block_id,
            cache_dtype,
        )

        output = attn_op.forward(q, k_new, v_new, kv_cache)
        self._assert_output_close(output, ref_output, name="Hybrid prefill output")

    def test_impl_forward(self):
        """Verify the full Hybrid Impl."""
        prefix_lengths = [13, 31]
        input_lengths = [7, 5]
        sequence_lengths = [p + i for p, i in zip(prefix_lengths, input_lengths)]
        head_num = 8
        head_num_kv = 2
        # test fa2 on sm80 and fa3 on sm90
        head_dim = 128
        page_size = 16

        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=head_dim,
            seq_size_per_block=page_size,
            data_type="bf16",
        )
        compute_dtype = config.attn_configs.dtype
        cache_dtype = self.cache_dtype(config.attn_configs)
        config.attn_configs.need_rope_kv_cache = True
        config.attn_configs.rope_config.style = RopeStyle.Base
        config.attn_configs.rope_config.dim = head_dim
        config.attn_configs.rope_config.base = 10000
        config.attn_configs.rope_config.max_pos = 128
        config.attn_configs.max_seq_len = 128

        attn_inputs = self._create_chunked_prefill_attention_inputs(
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            seq_size_per_block=page_size,
            dtype=compute_dtype,
        )

        q_chunks = []
        k_new_chunks = []
        v_new_chunks = []
        k_prefix_chunks = []
        v_prefix_chunks = []
        for prefix_len, input_len in zip(prefix_lengths, input_lengths):
            q_chunks.append(
                torch.randn(
                    input_len,
                    head_num,
                    head_dim,
                    dtype=compute_dtype,
                    device=self.device,
                )
            )
            k_prefix = torch.randn(
                prefix_len,
                head_num_kv,
                head_dim,
                dtype=compute_dtype,
                device=self.device,
            )
            v_prefix = torch.randn_like(k_prefix)
            k_prefix_chunks.append(k_prefix.to(cache_dtype).to(compute_dtype))
            v_prefix_chunks.append(v_prefix.to(cache_dtype).to(compute_dtype))
            k_new_chunks.append(
                torch.randn(
                    input_len,
                    head_num_kv,
                    head_dim,
                    dtype=compute_dtype,
                    device=self.device,
                )
            )
            v_new_chunks.append(torch.randn_like(k_new_chunks[-1]))

        qkv = torch.cat(
            [
                torch.cat(q_chunks).flatten(1),
                torch.cat(k_new_chunks).flatten(1),
                torch.cat(v_new_chunks).flatten(1),
            ],
            dim=-1,
        )
        kv_cache = self._create_prefix_kv_cache(
            k_prefix_chunks,
            v_prefix_chunks,
            prefix_lengths,
            sequence_lengths,
            page_size,
            head_num_kv,
            head_dim,
            attn_inputs.kv_cache_kernel_block_id,
            cache_dtype=cache_dtype,
        )
        impl = PyFlashinferHybridPrefillImpl(
            config.attn_configs, attn_inputs, config.parallelism_config
        )
        self.assertFalse(impl.support_cuda_graph())

        if self.kv_cache_dtype == KvCacheDataType.FP8:
            self.assertEqual(impl.fmha_impl.kv_dtype, torch.float8_e4m3fn)

        self.assertIsNotNone(impl.rope_impl)
        expected_q, expected_k, expected_v = impl.rope_impl.forward(qkv.clone())
        ref_chunks = []
        token_offset = 0
        for batch_idx, (prefix_len, input_len) in enumerate(
            zip(prefix_lengths, input_lengths)
        ):
            token_slice = slice(token_offset, token_offset + input_len)
            ref_chunks.append(
                self._reference_chunked_prefill(
                    expected_q[token_slice],
                    torch.cat(
                        [k_prefix_chunks[batch_idx], expected_k[token_slice]], dim=0
                    ),
                    torch.cat(
                        [v_prefix_chunks[batch_idx], expected_v[token_slice]], dim=0
                    ),
                    prefix_len,
                    input_len,
                    impl.fmha_impl.q_dtype,
                    impl.fmha_impl.kv_dtype,
                )
            )
            token_offset += input_len
        ref_output = torch.cat(ref_chunks)

        # Cache writes are a pure dtype cast + copy (no arithmetic), so the
        # cache comparisons below are intentionally bitwise (rtol=0, atol=0).
        cache_before = kv_cache.kv_cache_base.clone()
        expected_cache = cache_before.clone()
        token_offset = 0
        for batch_idx, (prefix_len, input_len) in enumerate(
            zip(prefix_lengths, input_lengths)
        ):
            for chunk_offset in range(input_len):
                position = prefix_len + chunk_offset
                page_id = int(
                    attn_inputs.kv_cache_kernel_block_id[
                        batch_idx, position // page_size
                    ].item()
                )
                page_offset = position % page_size
                expected_cache[page_id, 0, :, page_offset, :] = expected_k[
                    token_offset + chunk_offset
                ].to(cache_dtype)
                expected_cache[page_id, 1, :, page_offset, :] = expected_v[
                    token_offset + chunk_offset
                ].to(cache_dtype)
            token_offset += input_len

        events = []
        ragged_run = impl.fmha_impl.ragged_wrapper.run
        paged_run = impl.fmha_impl.prefix_paged_wrapper.run
        cache_write_forward = impl.kv_cache_write_op.forward

        def observed_ragged_run(*args, **kwargs):
            events.append("ragged_attention")
            torch.testing.assert_close(
                kv_cache.kv_cache_base.float(),
                cache_before.float(),
                rtol=0,
                atol=0,
            )
            return ragged_run(*args, **kwargs)

        def observed_cache_write_forward(key, value, cache):
            events.append("cache_write")
            return cache_write_forward(key, value, cache)

        def observed_paged_run(*args, **kwargs):
            events.append("paged_attention")
            torch.testing.assert_close(
                kv_cache.kv_cache_base.float(),
                expected_cache.float(),
                rtol=0,
                atol=0,
            )
            return paged_run(*args, **kwargs)

        with mock.patch.object(
            impl.fmha_impl.ragged_wrapper, "run", side_effect=observed_ragged_run
        ), mock.patch.object(
            impl.kv_cache_write_op, "forward", side_effect=observed_cache_write_forward
        ), mock.patch.object(
            impl.fmha_impl.prefix_paged_wrapper, "run", side_effect=observed_paged_run
        ):
            output = impl.forward(qkv.clone(), kv_cache)

        self.assertEqual(events, ["ragged_attention", "cache_write", "paged_attention"])
        torch.testing.assert_close(
            kv_cache.kv_cache_base.float(),
            expected_cache.float(),
            rtol=0,
            atol=0,
        )
        self._assert_output_close(output, ref_output, name="Hybrid Impl forward output")

    def test_chunked_prefill_single_batch(self):
        self._test_hybrid_prefill_correctness(
            prefix_lengths=[4884],
            input_lengths=[5],
            head_num=40,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )

    def test_chunked_prefill_multi_batch_varied(self):
        self._test_hybrid_prefill_correctness(
            prefix_lengths=[32, 96, 160],
            input_lengths=[8, 16, 24],
            head_num=16,
            head_num_kv=4,
            size_per_head=64,
            page_size=16,
        )

    def test_chunked_prefill_multi_batch_uniform(self):
        self._test_hybrid_prefill_correctness(
            prefix_lengths=[64, 64, 64, 64],
            input_lengths=[16, 16, 16, 16],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )

    def test_chunked_prefill_small_page_size(self):
        self._test_hybrid_prefill_correctness(
            prefix_lengths=[128, 256],
            input_lengths=[16, 32],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=32,
        )

    def test_chunked_prefill_large_page_size(self):
        self._test_hybrid_prefill_correctness(
            prefix_lengths=[128, 256],
            input_lengths=[16, 32],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=128,
        )

    def test_chunked_prefill_many_heads(self):
        self._test_hybrid_prefill_correctness(
            prefix_lengths=[64, 128],
            input_lengths=[16, 32],
            head_num=64,
            head_num_kv=16,
            size_per_head=128,
            page_size=64,
        )

    def test_chunked_prefill_gqa(self):
        self._test_hybrid_prefill_correctness(
            prefix_lengths=[64, 128],
            input_lengths=[16, 32],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )

    def test_reuse_page_layout(self):
        page_size = 16
        prefix_lengths = [1, page_size, page_size + 1]
        input_lengths = [3, 5, 7]
        page_nums = torch.tensor([1, 1, 2], dtype=torch.int32)
        page_starts = torch.tensor([0, 1, 2], dtype=torch.int32)
        config = self._create_config(
            head_num=8,
            head_num_kv=2,
            size_per_head=64,
            seq_size_per_block=page_size,
        )
        attn_inputs = self._create_chunked_prefill_attention_inputs(
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            seq_size_per_block=page_size,
        )
        block_table_host = attn_inputs.kv_cache_kernel_block_id.clone()
        attn_inputs.kv_cache_kernel_block_id = torch.empty(0, dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id_device = block_table_host.to(self.device)
        attn_op = PyFlashinferHybridPrefillAttnOp(config.attn_configs, attn_inputs)
        self.assertTrue(attn_op.support(attn_inputs))
        with mock.patch.object(
            attn_op.prefix_paged_wrapper,
            "plan",
            wraps=attn_op.prefix_paged_wrapper.plan,
        ) as prefix_plan:
            attn_op.prepare(attn_inputs)

        prefix_plan.assert_called_once()
        prefix_plan_args = prefix_plan.call_args.args
        torch.testing.assert_close(
            prefix_plan_args[1], torch.tensor([0, 1, 2, 4], dtype=torch.int32)
        )

        reuse_info = attn_op.fmha_params.batch_reuse_info_vec_h
        self.assertEqual(tuple(reuse_info.shape), (len(prefix_lengths), 4))
        torch.testing.assert_close(
            reuse_info[:, 1], torch.tensor(prefix_lengths, dtype=torch.int32)
        )
        torch.testing.assert_close(reuse_info[:, 2], page_starts)
        torch.testing.assert_close(reuse_info[:, 3], page_nums)
        self.assertEqual(
            attn_op.fmha_params.reuse_cache_page_indice_h.numel(),
            page_nums.sum().item(),
        )

        expected_page_indices = torch.cat(
            [
                block_table_host[i, :page_num]
                for i, page_num in enumerate(page_nums.tolist())
            ]
        )
        torch.testing.assert_close(
            attn_op.fmha_params.reuse_cache_page_indice_h, expected_page_indices
        )


class TestHybridPrefillDisableGating(unittest.TestCase):
    def test_disable_gating(self):
        config = FMHAConfig()
        impl_name = PyFlashinferHybridPrefillImpl.__name__

        config.disable_flashinfer_hybrid_prefill = False
        config.disable_flashinfer_native = False
        self.assertFalse(attn_factory._is_fmha_impl_disabled(impl_name, config))

        config.disable_flashinfer_hybrid_prefill = True
        self.assertTrue(attn_factory._is_fmha_impl_disabled(impl_name, config))

        config.disable_flashinfer_hybrid_prefill = False
        config.disable_flashinfer_native = True
        self.assertTrue(attn_factory._is_fmha_impl_disabled(impl_name, config))


class TestHybridPrefillSupport(unittest.TestCase):
    @staticmethod
    def _config() -> AttentionConfigs:
        config = AttentionConfigs()
        config.rope_config.style = RopeStyle.Base
        return config

    @staticmethod
    def _inputs(prefix_lengths: list[int]) -> PyAttentionInputs:
        inputs = PyAttentionInputs()
        inputs.prefix_lengths = torch.tensor(prefix_lengths, dtype=torch.int32)
        inputs.kv_cache_kernel_block_id = torch.zeros(
            (len(prefix_lengths), 1), dtype=torch.int32
        )
        inputs.is_cuda_graph = False
        return inputs

    def test_rejects_mixed_prefixes(self):
        inputs = self._inputs([0, 32])

        self.assertFalse(PyFlashinferHybridPrefillAttnOp.support(inputs))

    def test_rejects_empty_block_table(self):
        inputs = self._inputs([32])
        inputs.kv_cache_kernel_block_id = torch.empty(0, dtype=torch.int32)

        self.assertFalse(PyFlashinferHybridPrefillAttnOp.support(inputs))

    def test_rejects_cuda_graph(self):
        inputs = self._inputs([32])
        inputs.is_cuda_graph = True

        self.assertFalse(PyFlashinferHybridPrefillImpl.support(self._config(), inputs))


class TestPyFlashinferHybridPrefillAttnOpFP8(TestPyFlashinferHybridPrefillAttnOp):
    kv_cache_dtype = KvCacheDataType.FP8
    rtol = 4e-2
    atol = 4e-2
    max_mismatch_rate = 1e-5


if __name__ == "__main__":
    unittest.main()
