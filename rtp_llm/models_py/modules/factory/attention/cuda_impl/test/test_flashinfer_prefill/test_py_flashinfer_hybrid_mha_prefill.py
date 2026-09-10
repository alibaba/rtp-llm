import math
import unittest
from types import SimpleNamespace
from typing import List
from unittest import mock

import torch
from flashinfer.cascade import merge_state
from flashinfer.prefill import single_prefill_with_kv_cache

from rtp_llm.models_py.modules.factory.attention import attn_factory, common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferHybridPrefillAttnOp,
    PyFlashinferHybridPrefillImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    BaseAttentionTest,
    fill_paged_kv_cache,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType, RopeStyle
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    get_typemeta,
    rtp_llm_ops,
)


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
        batch_size: int,
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
            batch_size, prefix_lengths, input_lengths, page_size
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
            len(prefix_lengths),
            prefix_lengths,
            input_lengths,
            page_size,
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
            batch_size=1,
            prefix_lengths=[4884],
            input_lengths=[5],
            head_num=40,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )

    def test_chunked_prefill_multi_batch_varied(self):
        self._test_hybrid_prefill_correctness(
            batch_size=3,
            prefix_lengths=[32, 96, 160],
            input_lengths=[8, 16, 24],
            head_num=16,
            head_num_kv=4,
            size_per_head=64,
            page_size=16,
        )

    def test_chunked_prefill_multi_batch_uniform(self):
        self._test_hybrid_prefill_correctness(
            batch_size=4,
            prefix_lengths=[64, 64, 64, 64],
            input_lengths=[16, 16, 16, 16],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )

    def test_chunked_prefill_small_page_size(self):
        self._test_hybrid_prefill_correctness(
            batch_size=2,
            prefix_lengths=[128, 256],
            input_lengths=[16, 32],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=32,
        )

    def test_chunked_prefill_large_page_size(self):
        self._test_hybrid_prefill_correctness(
            batch_size=2,
            prefix_lengths=[128, 256],
            input_lengths=[16, 32],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=128,
        )

    def test_chunked_prefill_many_heads(self):
        self._test_hybrid_prefill_correctness(
            batch_size=2,
            prefix_lengths=[64, 128],
            input_lengths=[16, 32],
            head_num=64,
            head_num_kv=16,
            size_per_head=128,
            page_size=64,
        )

    def test_chunked_prefill_gqa(self):
        self._test_hybrid_prefill_correctness(
            batch_size=2,
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
            len(prefix_lengths), prefix_lengths, input_lengths, page_size
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


class _NoReleaseHybridAttnOp(PyFlashinferHybridPrefillAttnOp):
    def __del__(self):
        pass


class TestDynamicFp8HybridUnit(unittest.TestCase):
    def test_compact_prefix_plan_and_write_before_gather(self):
        params = SimpleNamespace(
            fill_params=mock.Mock(),
            batch_reuse_info_vec_h=torch.tensor(
                [[0, 1, 0, 1], [0, 9, 1, 2]], dtype=torch.int32
            ),
            reuse_cache_page_indice_h=torch.tensor([12, 4, 9], dtype=torch.int32),
            reuse_cache_page_indice_d=torch.tensor([12, 4, 9], dtype=torch.int32),
        )
        events = []
        ragged = SimpleNamespace(
            plan=mock.Mock(),
            run=mock.Mock(
                side_effect=lambda *args, **kwargs: (
                    events.append("ragged")
                    or (torch.zeros_like(args[0]), torch.zeros(args[0].shape[:2]))
                )
            ),
        )
        prefix = SimpleNamespace(
            plan=mock.Mock(),
            run=mock.Mock(
                side_effect=lambda *args, **kwargs: (
                    events.append("prefix")
                    or (torch.zeros_like(args[0]), torch.zeros(args[0].shape[:2]))
                )
            ),
        )
        op = _NoReleaseHybridAttnOp.__new__(_NoReleaseHybridAttnOp)
        op.dynamic_fp8 = True
        op.fmha_params = params
        op.ragged_wrapper = ragged
        op.prefix_paged_wrapper = prefix
        op.local_head_num = 4
        op.local_kv_head_num = 2
        op.head_dim_qk = 4
        op.head_dim_vo = 4
        op.physical_page_size = 32
        op.page_size = 8
        op.subdivision = 4
        op.dtype = torch.bfloat16
        op.q_dtype = torch.bfloat16
        op.kv_dtype = torch.bfloat16
        op.is_causal = True
        inputs = SimpleNamespace(
            kv_cache_kernel_block_id=torch.tensor([[12, 4], [9, 0]]),
            kv_cache_kernel_block_id_device=None,
            prefix_lengths=torch.tensor([1, 9], dtype=torch.int32),
            sequence_lengths=torch.tensor([2, 10], dtype=torch.int32),
            input_lengths=torch.tensor([1, 1], dtype=torch.int32),
            cu_seqlens_device=torch.tensor([0, 1, 2], dtype=torch.int32),
        )
        op.prepare(inputs)
        torch.testing.assert_close(
            prefix.plan.call_args.args[2],
            torch.tensor([0, 1, 2], dtype=torch.int32),
        )
        torch.testing.assert_close(
            params.reuse_cache_page_indice_d,
            torch.tensor([12, 4, 9], dtype=torch.int32),
        )

        q = torch.randn(2, 4, 4, dtype=torch.bfloat16)
        k = torch.randn(2, 2, 4, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cache = SimpleNamespace(
            kv_cache_base=torch.empty(16, 2, 2, 8, 4, dtype=torch.float8_e4m3fn),
            kv_scale_base=torch.empty(16, 2 * 2 * 8, dtype=torch.float32),
        )
        writer = SimpleNamespace(
            forward=mock.Mock(side_effect=lambda *args: events.append("write"))
        )

        def gather_side_effect(*args):
            events.append("gather")

        with mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "rtp_llm_ops.gather_and_dequantize_fp8_kv_cache",
            side_effect=gather_side_effect,
            create=True,
        ) as gather_mock, mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "quantize_to_fp8_if_needed",
            side_effect=AssertionError("unit-scale cast must not run"),
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "merge_state_in_place"
        ):
            op.forward(q, k, v, cache, writer)

        self.assertEqual(events, ["ragged", "write", "gather", "prefix"])
        self.assertIs(gather_mock.call_args.args[2], params.reuse_cache_page_indice_d)
        self.assertEqual(gather_mock.call_args.args[3].dtype, torch.bfloat16)

    def test_mode2_hybrid_attention_matches_base_reference_with_subdivision(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        device = torch.device("cuda")
        torch.manual_seed(2026)
        harness = TestPyFlashinferHybridPrefillAttnOp()
        harness.device = device
        prefix_lengths = [7, 9]
        input_lengths = [2, 3]
        sequence_lengths = [
            prefix + current for prefix, current in zip(prefix_lengths, input_lengths)
        ]
        kernel_page_size = 4
        physical_page_size = 8
        subdivision = physical_page_size // kernel_page_size
        config = harness._create_config(
            head_num=4,
            head_num_kv=2,
            size_per_head=64,
            seq_size_per_block=kernel_page_size,
        )
        config.attn_configs.tokens_per_block = physical_page_size
        config.attn_configs.kv_cache_dtype = KvCacheDataType.FP8
        config.attn_configs.fp8_kv_cache_mode = 2
        config.attn_configs.is_causal = True
        inputs = harness._create_chunked_prefill_attention_inputs(
            len(prefix_lengths),
            prefix_lengths,
            input_lengths,
            kernel_page_size,
        )
        block_table = torch.zeros_like(inputs.kv_cache_kernel_block_id)
        physical_page_offset = 0
        for batch_idx, sequence_length in enumerate(sequence_lengths):
            kernel_pages = math.ceil(sequence_length / kernel_page_size)
            for logical_page in range(kernel_pages):
                physical_page = physical_page_offset + logical_page // subdivision
                block_table[batch_idx, logical_page] = (
                    physical_page * subdivision + logical_page % subdivision
                )
            physical_page_offset += math.ceil(sequence_length / physical_page_size)
        physical_page_count = physical_page_offset
        torch.testing.assert_close(
            block_table,
            torch.tensor([[0, 1, 2], [4, 5, 6]], dtype=torch.int32),
        )
        self.assertEqual(physical_page_count, 4)
        inputs.kv_cache_kernel_block_id = block_table
        inputs.kv_cache_kernel_block_id_device = block_table.to(device)

        op = PyFlashinferHybridPrefillAttnOp(config.attn_configs, inputs)
        params = op.prepare(inputs)

        q_chunks = []
        k_prefix_chunks = []
        v_prefix_chunks = []
        k_new_chunks = []
        v_new_chunks = []
        prefix_k_tokens = []
        prefix_v_tokens = []
        target_pages = []
        target_offsets = []
        block_table = inputs.kv_cache_kernel_block_id
        for batch_idx, (prefix_length, input_length) in enumerate(
            zip(prefix_lengths, input_lengths)
        ):
            q_new = torch.randn(input_length, 4, 64, device=device, dtype=torch.float16)
            k_prefix = torch.randn(
                prefix_length, 2, 64, device=device, dtype=torch.float16
            )
            v_prefix = torch.randn_like(k_prefix)
            k_new = torch.randn(input_length, 2, 64, device=device, dtype=torch.float16)
            v_new = torch.randn_like(k_new)
            q_chunks.append(q_new)
            k_prefix_chunks.append(k_prefix)
            v_prefix_chunks.append(v_prefix)
            k_new_chunks.append(k_new)
            v_new_chunks.append(v_new)
            for position in range(prefix_length):
                kernel_page = int(
                    block_table[batch_idx, position // kernel_page_size].item()
                )
                prefix_k_tokens.append(k_prefix[position])
                prefix_v_tokens.append(v_prefix[position])
                target_pages.append(kernel_page // subdivision)
                target_offsets.append(
                    kernel_page % subdivision * kernel_page_size
                    + position % kernel_page_size
                )

        kernel_page_count = physical_page_count * subdivision
        cache = LayerKVCache()
        cache.kv_cache_base = torch.zeros(
            kernel_page_count,
            2,
            2,
            kernel_page_size,
            64,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        cache.kv_scale_base = torch.ones(
            kernel_page_count,
            2 * 2 * kernel_page_size,
            device=device,
            dtype=torch.float32,
        )
        rtp_llm_ops.quantize_and_write_fp8_kv_cache(
            torch.stack(prefix_k_tokens),
            torch.stack(prefix_v_tokens),
            cache.kv_cache_base,
            cache.kv_scale_base,
            torch.tensor(target_pages, device=device, dtype=torch.int32),
            torch.tensor(target_offsets, device=device, dtype=torch.int32),
            physical_page_size,
            kernel_page_size,
            subdivision,
        )
        scale_view = cache.kv_scale_base.view(kernel_page_count, 2, 2, kernel_page_size)
        reference_chunks = []
        for batch_idx, (prefix_length, input_length) in enumerate(
            zip(prefix_lengths, input_lengths)
        ):
            restored_prefix_k = []
            restored_prefix_v = []
            for position in range(prefix_length):
                kernel_page = int(
                    block_table[batch_idx, position // kernel_page_size].item()
                )
                storage_token = position % kernel_page_size
                restored_prefix_k.append(
                    cache.kv_cache_base[kernel_page, 0, :, storage_token].float()
                    * scale_view[kernel_page, 0, :, storage_token].unsqueeze(-1)
                )
                restored_prefix_v.append(
                    cache.kv_cache_base[kernel_page, 1, :, storage_token].float()
                    * scale_view[kernel_page, 1, :, storage_token].unsqueeze(-1)
                )
            reference_chunks.append(
                harness._reference_chunked_prefill(
                    q_chunks[batch_idx],
                    torch.cat(
                        [
                            torch.stack(restored_prefix_k).to(
                                k_prefix_chunks[batch_idx].dtype
                            ),
                            k_new_chunks[batch_idx],
                        ]
                    ),
                    torch.cat(
                        [
                            torch.stack(restored_prefix_v).to(
                                v_prefix_chunks[batch_idx].dtype
                            ),
                            v_new_chunks[batch_idx],
                        ]
                    ),
                    prefix_length,
                    input_length,
                    torch.float16,
                    torch.float16,
                )
            )

        writer = KVCacheWriteOp(
            num_kv_heads=2,
            head_size=64,
            physical_page_size=physical_page_size,
            kernel_page_size=kernel_page_size,
            dynamic_mode=True,
        )
        writer.set_params(params)

        output = op.forward(
            torch.cat(q_chunks),
            torch.cat(k_new_chunks),
            torch.cat(v_new_chunks),
            cache,
            writer,
        )
        reference = torch.cat(reference_chunks)
        torch.testing.assert_close(output, reference, rtol=0.06, atol=0.04)

        for batch_idx, (prefix_length, input_length) in enumerate(
            zip(prefix_lengths, input_lengths)
        ):
            for chunk_offset in range(input_length):
                position = prefix_length + chunk_offset
                kernel_page = int(
                    block_table[batch_idx, position // kernel_page_size].item()
                )
                storage_token = position % kernel_page_size
                for kv, source in enumerate(
                    (k_new_chunks[batch_idx], v_new_chunks[batch_idx])
                ):
                    restored = cache.kv_cache_base[
                        kernel_page, kv, :, storage_token
                    ].float() * scale_view[kernel_page, kv, :, storage_token].unsqueeze(
                        -1
                    )
                    torch.testing.assert_close(
                        restored,
                        source[chunk_offset].float(),
                        rtol=0.13,
                        atol=0.02,
                    )


class TestDynamicFp8FactoryGating(unittest.TestCase):
    @staticmethod
    def _config() -> SimpleNamespace:
        return SimpleNamespace(
            fp8_kv_cache_mode=2,
            use_mla=False,
            use_logn_attn=False,
            rope_config=SimpleNamespace(style=RopeStyle.Base),
        )

    @staticmethod
    def _impl(name, constructor=None, supports_cuda_graph=False):
        class Impl:
            accepts_fmha_config = False

            @staticmethod
            def support(attn_configs, attn_inputs):
                return True

            @classmethod
            def support_parallelism_config(cls, parallelism_config):
                return True

            def __new__(cls, *args, **kwargs):
                if constructor is not None:
                    return constructor()
                return super().__new__(cls)

            def support_cuda_graph(self):
                return supports_cuda_graph

        Impl.__name__ = name
        return Impl

    def test_only_native_flashinfer_backend_is_instantiated(self):
        disallowed = self._impl(
            "OtherBackend",
            constructor=lambda: self.fail("disallowed backend was instantiated"),
        )
        allowed = self._impl("PyFlashinferPrefillImpl")
        inputs = SimpleNamespace(is_prefill=True)
        with mock.patch.object(attn_factory, "PREFILL_MHA_IMPS", [disallowed, allowed]):
            result = attn_factory.get_fmha_impl(
                self._config(), None, inputs, parallelism_config=None
            )
        self.assertIsInstance(result, allowed)

    def test_decode_only_instantiates_native_flashinfer_backend(self):
        disallowed = self._impl(
            "OtherDecodeBackend",
            constructor=lambda: self.fail("disallowed decode backend was instantiated"),
        )
        allowed = self._impl("PyFlashinferDecodeImpl")
        inputs = SimpleNamespace(is_prefill=False)
        with mock.patch.object(attn_factory, "DECODE_MHA_IMPS", [disallowed, allowed]):
            result = attn_factory.get_fmha_impl(
                self._config(), None, inputs, parallelism_config=None
            )
        self.assertIsInstance(result, allowed)

    def test_mode2_cuda_graph_allows_native_single_token_decode(self):
        allowed = self._impl("PyFlashinferDecodeImpl", supports_cuda_graph=True)
        inputs = SimpleNamespace(is_prefill=False)
        with mock.patch.object(attn_factory, "DECODE_MHA_IMPS", [allowed]):
            result = attn_factory.get_fmha_impl(
                self._config(),
                None,
                inputs,
                is_cuda_graph=True,
                parallelism_config=None,
            )
        self.assertIsInstance(result, allowed)

    def test_allowed_backend_constructor_failure_does_not_fallback(self):
        fallback_calls = []

        def fail_constructor():
            raise RuntimeError("construction failed")

        broken = self._impl("PyFlashinferPrefillImpl", fail_constructor)
        fallback = self._impl(
            "PyFlashinferPagedPrefillImpl",
            constructor=lambda: fallback_calls.append(True),
        )
        with mock.patch.object(attn_factory, "PREFILL_MHA_IMPS", [broken, fallback]):
            with self.assertRaisesRegex(RuntimeError, "required attention backend"):
                attn_factory.get_fmha_impl(
                    self._config(),
                    None,
                    SimpleNamespace(is_prefill=True),
                    parallelism_config=None,
                )
        self.assertEqual(fallback_calls, [])

    def test_rejects_unsupported_mode2_features_before_selection(self):
        cases = (
            ("cuda_graph", lambda config: None, True, "CUDA graph"),
            ("mla", lambda config: setattr(config, "use_mla", True), False, "MLA"),
            (
                "mrope",
                lambda config: setattr(config.rope_config, "style", RopeStyle.Mrope),
                False,
                "MRoPE",
            ),
            (
                "logn",
                lambda config: setattr(config, "use_logn_attn", True),
                False,
                "use_logn_attn",
            ),
        )
        for name, mutate, is_cuda_graph, message in cases:
            with self.subTest(name=name):
                config = self._config()
                mutate(config)
                with self.assertRaisesRegex(ValueError, message):
                    attn_factory.get_fmha_impl(
                        config,
                        None,
                        SimpleNamespace(is_prefill=True),
                        is_cuda_graph=is_cuda_graph,
                    )


class TestPyFlashinferHybridPrefillAttnOpFP8(TestPyFlashinferHybridPrefillAttnOp):
    kv_cache_dtype = KvCacheDataType.FP8
    rtol = 4e-2
    atol = 4e-2
    max_mismatch_rate = 1e-5

    def test_mode1_interleaved_mrope_preserves_hybrid_order(self):
        prefix_lengths = [5, 7]
        input_lengths = [3, 2]
        sequence_lengths = [
            prefix + current for prefix, current in zip(prefix_lengths, input_lengths)
        ]
        page_size = 4
        config = self._create_config(
            head_num=4,
            head_num_kv=2,
            size_per_head=256,
            seq_size_per_block=page_size,
        )
        self._enable_qwen35_mrope_mode1(config)
        inputs = self._create_chunked_prefill_attention_inputs(
            len(prefix_lengths), prefix_lengths, input_lengths, page_size
        )
        position_ids = self._add_qwen35_mrope_inputs(
            inputs, input_lengths, prefix_lengths
        )
        self.assertTrue(
            PyFlashinferHybridPrefillImpl.support(config.attn_configs, inputs)
        )

        total_tokens = sum(input_lengths)
        qkv = torch.randn(
            total_tokens,
            (config.head_num + 2 * config.head_num_kv) * config.size_per_head,
            dtype=config.attn_configs.dtype,
            device=self.device,
        )
        q, k, v = torch.split(
            qkv,
            [
                config.head_num * config.size_per_head,
                config.head_num_kv * config.size_per_head,
                config.head_num_kv * config.size_per_head,
            ],
            dim=-1,
        )
        q = q.reshape(total_tokens, config.head_num, config.size_per_head)
        k = k.reshape(total_tokens, config.head_num_kv, config.size_per_head)
        v = v.reshape(total_tokens, config.head_num_kv, config.size_per_head)
        expected_q, expected_k = self._apply_qwen35_mrope_reference(q, k, position_ids)

        k_prefix = [
            torch.randn(
                length,
                config.head_num_kv,
                config.size_per_head,
                dtype=config.attn_configs.dtype,
                device=self.device,
            )
            for length in prefix_lengths
        ]
        v_prefix = [torch.randn_like(key) for key in k_prefix]
        kv_cache = self._create_prefix_kv_cache(
            k_prefix,
            v_prefix,
            prefix_lengths,
            sequence_lengths,
            page_size,
            config.head_num_kv,
            config.size_per_head,
            inputs.kv_cache_kernel_block_id,
            torch.float8_e4m3fn,
        )
        cache_before = kv_cache.kv_cache_base.clone()
        expected_cache = cache_before.clone()
        token_offset = 0
        for batch_idx, (prefix_length, input_length) in enumerate(
            zip(prefix_lengths, input_lengths)
        ):
            for chunk_offset in range(input_length):
                position = prefix_length + chunk_offset
                page_id = int(
                    inputs.kv_cache_kernel_block_id[
                        batch_idx, position // page_size
                    ].item()
                )
                page_offset = position % page_size
                expected_cache[page_id, 0, :, page_offset] = expected_k[
                    token_offset + chunk_offset
                ].to(torch.float8_e4m3fn)
                expected_cache[page_id, 1, :, page_offset] = v[
                    token_offset + chunk_offset
                ].to(torch.float8_e4m3fn)
            token_offset += input_length

        impl = PyFlashinferHybridPrefillImpl(
            config.attn_configs, inputs, config.parallelism_config
        )
        events = []
        fused_forward = impl.fused_mrope_impl.forward
        cache_write = impl.kv_cache_write_op.forward

        def observed_fused_forward(qkv_input, cache, params):
            events.append("fused_rope")
            self.assertIsNone(cache)
            return fused_forward(qkv_input, cache, params)

        def observed_ragged(query, key, value, *args, **kwargs):
            events.append("ragged_attention")
            torch.testing.assert_close(
                kv_cache.kv_cache_base.float(), cache_before.float(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                query.float(), expected_q.float(), rtol=1e-2, atol=1e-2
            )
            return (
                torch.zeros_like(expected_q),
                torch.zeros(
                    expected_q.shape[:2], dtype=torch.float32, device=self.device
                ),
            )

        def observed_cache_write(key, value, cache):
            events.append("cache_write")
            return cache_write(key, value, cache)

        def observed_paged(query, cache, *args, **kwargs):
            events.append("paged_attention")
            self.assertIs(cache, kv_cache.kv_cache_base)
            torch.testing.assert_close(
                kv_cache.kv_cache_base.float(), expected_cache.float(), rtol=0, atol=0
            )
            return (
                torch.zeros_like(expected_q),
                torch.zeros(
                    expected_q.shape[:2], dtype=torch.float32, device=self.device
                ),
            )

        def observed_cache_store(*args, **kwargs):
            events.append("cache_store")

        with mock.patch.object(
            impl.fused_mrope_impl,
            "forward",
            side_effect=observed_fused_forward,
        ) as fused_mock, mock.patch.object(
            impl.fmha_impl.ragged_wrapper,
            "run",
            side_effect=observed_ragged,
        ), mock.patch.object(
            impl.kv_cache_write_op,
            "forward",
            side_effect=observed_cache_write,
        ) as write_mock, mock.patch.object(
            impl.fmha_impl.prefix_paged_wrapper,
            "run",
            side_effect=observed_paged,
        ), mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "merge_state_in_place"
        ), mock.patch.object(
            common,
            "apply_write_cache_store",
            side_effect=observed_cache_store,
        ):
            impl.forward(qkv.clone(), kv_cache)

        fused_mock.assert_called_once()
        write_mock.assert_called_once()
        self.assertEqual(
            events,
            [
                "fused_rope",
                "ragged_attention",
                "cache_write",
                "paged_attention",
                "cache_store",
            ],
        )
        torch.testing.assert_close(
            kv_cache.kv_cache_base.float(), expected_cache.float(), rtol=0, atol=0
        )
        self.assertTrue(torch.all(kv_cache.kv_scale_base == 1).item())


if __name__ == "__main__":
    unittest.main()
