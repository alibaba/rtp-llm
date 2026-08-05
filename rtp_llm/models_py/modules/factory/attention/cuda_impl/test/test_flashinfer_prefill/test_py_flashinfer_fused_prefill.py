"""Production-wrapper tests for fused PyFlashinfer prefill preprocessing."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPagedPrefillImpl,
    PyFlashinferPrefillAttnOp,
    PyFlashinferPrefillImpl,
    PyFlashinferPrefillPagedAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    BaseAttentionTest,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, RopeStyle
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyPrefillCudaGaphCopyParams,
    init_exec_ctx,
)
from rtp_llm.ops.fused_rope_kvcache_op import InvalidFusedPrefillInputError

FP8_E4M3_MAX = 448.0
FP8_ROPE_MAX_MISMATCH_RATIO = 0.05
PAGE_SIZE = 8


class TestPyFlashinferFusedPrefill(BaseAttentionTest):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")

        py_env_configs = PyEnvConfigs()
        py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size = 64
        engine_config = EngineConfig.create(py_env_configs)
        model_config = ModelConfig()
        model_config.max_seq_len = 2048
        pc = engine_config.parallelism_config
        # A CUDA device that exists but cannot initialize the runtime is a test
        # failure, not a reason to silently skip the only fused-path coverage.
        init_exec_ctx(
            device_id=pc.world_rank % pc.local_world_size,
            trace_memory=engine_config.profiling_debug_logging_config.trace_memory,
            enable_comm_overlap=engine_config.device_resource_config.enable_comm_overlap,
            mla_ops_type=int(model_config.mla_ops_type),
        )

    def _attn_configs(
        self,
        cache_dtype: KvCacheDataType,
        rope_style: RopeStyle = RopeStyle.Base,
        need_rope_kv_cache: bool = True,
    ) -> AttentionConfigs:
        return self._create_config(
            head_num=8,
            head_num_kv=2,
            size_per_head=128,
            seq_size_per_block=PAGE_SIZE,
            data_type="bf16",
            rope_style=rope_style,
            kv_cache_dtype=cache_dtype,
            need_rope_kv_cache=need_rope_kv_cache,
            is_causal=True,
            max_seq_len=2048,
        ).attn_configs

    def _cache_pair(
        self, config: AttentionConfigs, total_blocks: int
    ) -> tuple[LayerKVCache, LayerKVCache]:
        """Two caches starting from identical bytes: reference and fused target."""
        dtype = (
            torch.float8_e4m3fn
            if config.kv_cache_dtype == KvCacheDataType.FP8
            else torch.bfloat16
        )
        reference, _, _ = self._create_kv_cache(
            total_blocks,
            config.kernel_tokens_per_block,
            config.kv_head_num,
            config.size_per_head,
            dtype=dtype,
            fp8_scale_fill=float("nan"),
        )
        fused, _, _ = self._create_kv_cache(
            total_blocks,
            config.kernel_tokens_per_block,
            config.kv_head_num,
            config.size_per_head,
            dtype=dtype,
            content=reference.kv_cache_base,
            fp8_scale_fill=float("nan"),
        )
        return reference, fused

    def _random_qkv(
        self, config: AttentionConfigs, total_tokens: int, scale: float = 0.5
    ) -> torch.Tensor:
        hidden_size = (config.head_num + 2 * config.kv_head_num) * config.size_per_head
        return (
            torch.randn(
                total_tokens,
                hidden_size,
                dtype=torch.bfloat16,
                device=self.device,
            )
            * scale
        )

    def _split_qkv(
        self, config: AttentionConfigs, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_tokens = qkv.shape[0]
        q_raw, k_raw, v_raw = torch.split(
            qkv.reshape(total_tokens, -1),
            [
                config.head_num * config.size_per_head,
                config.kv_head_num * config.size_per_head,
                config.kv_head_num * config.size_per_head,
            ],
            dim=-1,
        )
        return (
            q_raw.reshape(total_tokens, config.head_num, config.size_per_head),
            k_raw.reshape(total_tokens, config.kv_head_num, config.size_per_head),
            v_raw.reshape(total_tokens, config.kv_head_num, config.size_per_head),
        )

    def _reference_rope(
        self, config: AttentionConfigs, qkv: torch.Tensor, params
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split reference RoPE, or a plain QKV split when RoPE is disabled."""
        if config.rope_config.style == RopeStyle.No:
            return self._split_qkv(config, qkv)
        reference_rope = MhaRotaryEmbeddingOp(config)
        reference_rope.set_params(params)
        return reference_rope.forward(qkv.clone())

    def _write_reference_cache(
        self,
        cache: LayerKVCache,
        key: torch.Tensor,
        value: torch.Tensor,
        input_lengths: list[int],
        prefix_lengths: list[int],
        block_ids: torch.Tensor,
        page_size: int,
    ) -> None:
        page_indices, page_offsets = self._cache_write_locations(
            input_lengths, prefix_lengths, block_ids, page_size
        )

        if cache.kv_cache_base.dtype == torch.float8_e4m3fn:
            # FP8 KV cache uses a direct saturating cast. Its scale buffer is a
            # kernel-written all-ones marker, verified independently below.
            key = key.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            value = value.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        else:
            key = key.to(cache.kv_cache_base.dtype)
            value = value.to(cache.kv_cache_base.dtype)

        cache.kv_cache_base[page_indices, 0, :, page_offsets, :] = key
        cache.kv_cache_base[page_indices, 1, :, page_offsets, :] = value

    def _cache_write_locations(
        self,
        input_lengths: list[int],
        prefix_lengths: list[int],
        block_ids: torch.Tensor,
        page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_indices = torch.repeat_interleave(
            torch.arange(len(input_lengths), dtype=torch.long),
            torch.tensor(input_lengths, dtype=torch.long),
        )
        cache_positions = torch.cat(
            [
                torch.arange(prefix_len, prefix_len + input_len, dtype=torch.long)
                for input_len, prefix_len in zip(input_lengths, prefix_lengths)
            ]
        )
        page_indices = block_ids[
            batch_indices, torch.div(cache_positions, page_size, rounding_mode="floor")
        ].to(self.device)
        return page_indices, cache_positions.remainder(page_size).to(self.device)

    def _assert_fp8_scale_writes(
        self,
        cache: LayerKVCache,
        input_lengths: list[int],
        prefix_lengths: list[int],
        block_ids: torch.Tensor,
        page_size: int,
    ) -> None:
        self.assertIsNotNone(cache.kv_scale_base)
        scale = cache.kv_scale_base.view(
            cache.kv_cache_base.size(0),
            2,
            cache.kv_cache_base.size(2),
            page_size,
        )
        page_indices, page_offsets = self._cache_write_locations(
            input_lengths, prefix_lengths, block_ids, page_size
        )
        written = torch.zeros_like(scale, dtype=torch.bool)
        for page_index, page_offset in zip(
            page_indices.tolist(), page_offsets.tolist(), strict=True
        ):
            written[page_index, :, :, page_offset] = True

        self.assertGreater(int(written.sum().item()), 0)
        torch.testing.assert_close(
            scale[written], torch.ones_like(scale[written]), rtol=0, atol=0
        )
        self.assertTrue(
            torch.isnan(scale[~written]).all().item(),
            "FP8 scale writes must not touch cache slots outside the new tokens",
        )

    def _assert_cache_equal(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        exact: bool,
        written_locations: tuple[torch.Tensor, torch.Tensor],
        check_rounding_bias: bool = True,
    ) -> None:
        """Compare a written cache against the reference write.

        exact=True is used whenever no RoPE runs before the write: the fused op
        then only converts and stores, so a truncating conversion or a wrong
        rounding mode must show up as a bit difference. With RoPE enabled the
        fused CUDA math and the split PyTorch reference can land on adjacent
        FP8 codes, so mismatches are allowed within one E4M3 step.
        """
        page_indices, page_offsets = written_locations
        actual = actual[page_indices, :, :, page_offsets, :]
        expected = expected[page_indices, :, :, page_offsets, :]

        if actual.dtype != torch.float8_e4m3fn:
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
            return

        actual_bits = actual.view(torch.uint8)
        expected_bits = expected.view(torch.uint8)
        if exact:
            self.assertTrue(
                torch.equal(actual_bits, expected_bits),
                "FP8 cache must match the saturating reference bit-exactly "
                "when no RoPE precedes the write",
            )
            return

        mismatch = actual_bits != expected_bits
        if not mismatch.any():
            return
        mismatch_ratio = mismatch.float().mean().item()
        self.assertLessEqual(
            mismatch_ratio,
            FP8_ROPE_MAX_MISMATCH_RATIO,
            "FP8 cache differs from the reference in too many elements: "
            f"{mismatch_ratio:.2%} > {FP8_ROPE_MAX_MISMATCH_RATIO:.2%}",
        )
        actual_float = actual.float()[mismatch]
        expected_float = expected.float()[mismatch]
        delta = actual_float - expected_float
        one_ulp = torch.maximum(
            expected_float.abs() / 8,
            torch.full_like(expected_float, 2**-9),
        )
        self.assertTrue(
            torch.all(delta.abs() <= one_ulp),
            "FP8 cache differs from the reference by more than one E4M3 step",
        )
        # A two-sided sign check is only meaningful with enough mismatch
        # samples; small adjacent-code sets are legitimately one-sided.
        if check_rounding_bias and delta.numel() >= 32:
            self.assertTrue(
                torch.any(delta < 0) and torch.any(delta > 0),
                "FP8 cache mismatches are all biased in the same direction",
            )

    def _run_cache_accuracy(
        self,
        impl_kind: str,
        cache_dtype: KvCacheDataType,
        rope_style: RopeStyle = RopeStyle.Base,
        need_rope_kv_cache: bool = True,
    ) -> None:
        input_lengths = [17, 9]
        prefix_lengths = [0, 0]
        config = self._attn_configs(cache_dtype, rope_style, need_rope_kv_cache)
        inputs = self._create_prefill_attention_inputs(
            len(input_lengths),
            input_lengths,
            PAGE_SIZE,
            dtype=torch.bfloat16,
            prefix_lengths=prefix_lengths,
        )
        block_ids = inputs.kv_cache_block_id
        total_blocks = int(block_ids.max().item()) + 1
        reference_cache, fused_cache = self._cache_pair(config, total_blocks)
        qkv = self._random_qkv(config, sum(input_lengths))

        if impl_kind == "paged":
            reference_attn = PyFlashinferPrefillPagedAttnOp(config, inputs)
            fused_impl = PyFlashinferPagedPrefillImpl(config, inputs)
        else:
            reference_attn = PyFlashinferPrefillAttnOp(config)
            fused_impl = PyFlashinferPrefillImpl(config, inputs)
        reference_params = reference_attn.prepare(inputs)

        query, key, value = self._reference_rope(config, qkv, reference_params)
        self._write_reference_cache(
            reference_cache,
            key,
            value,
            input_lengths,
            prefix_lengths,
            block_ids,
            PAGE_SIZE,
        )
        reference_input = (
            query
            if impl_kind == "paged"
            else torch.cat([query.flatten(1), key.flatten(1), value.flatten(1)], dim=-1)
        )
        reference_output = reference_attn.forward(reference_input, reference_cache)
        fused_output = fused_impl.forward(qkv.clone(), fused_cache)

        self._assert_cache_equal(
            fused_cache.kv_cache_base,
            reference_cache.kv_cache_base,
            exact=rope_style == RopeStyle.No,
            written_locations=self._cache_write_locations(
                input_lengths,
                prefix_lengths,
                inputs.kv_cache_kernel_block_id,
                PAGE_SIZE,
            ),
        )
        if cache_dtype == KvCacheDataType.FP8:
            self._assert_fp8_scale_writes(
                fused_cache,
                input_lengths,
                prefix_lengths,
                block_ids,
                PAGE_SIZE,
            )
        torch.testing.assert_close(
            fused_output.float(), reference_output.float(), rtol=2e-2, atol=2e-2
        )

    def test_paged_and_ragged_base_and_fp8_cache(self) -> None:
        for impl_kind in ("paged", "ragged"):
            for cache_dtype in (KvCacheDataType.BASE, KvCacheDataType.FP8):
                with self.subTest(impl=impl_kind, cache_dtype=cache_dtype):
                    self._run_cache_accuracy(impl_kind, cache_dtype)

    def test_no_rope_with_cache_still_uses_fused_writer(self) -> None:
        self._run_cache_accuracy(
            "paged",
            KvCacheDataType.FP8,
            rope_style=RopeStyle.No,
            need_rope_kv_cache=False,
        )

    def test_active_rope_with_cache_matches_trt_fused_gate(self) -> None:
        # Cache presence selects the fused writer even when the legacy config
        # flag is false, matching FlashInferTRTLLMFMHAv2PrefillImpl.
        self._run_cache_accuracy(
            "paged",
            KvCacheDataType.FP8,
            rope_style=RopeStyle.Base,
            need_rope_kv_cache=False,
        )

    def _run_ragged_without_cache(
        self,
        config: AttentionConfigs,
        inputs: PyAttentionInputs,
        input_lengths: list[int],
    ) -> None:
        qkv = self._random_qkv(config, sum(input_lengths))
        reference_attn = PyFlashinferPrefillAttnOp(config)
        reference_params = reference_attn.prepare(inputs)
        query, key, value = self._reference_rope(config, qkv, reference_params)
        reference_input = torch.cat(
            [query.flatten(1), key.flatten(1), value.flatten(1)], dim=-1
        )
        expected = reference_attn.forward(reference_input, None)
        actual = PyFlashinferPrefillImpl(config, inputs).forward(qkv.clone(), None)
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=2e-2
        )

    def test_ragged_no_cache_accepts_empty_prefix(self) -> None:
        input_lengths = [5, 3]
        config = self._attn_configs(
            KvCacheDataType.BASE,
            rope_style=RopeStyle.No,
            need_rope_kv_cache=False,
        )
        inputs = self._create_prefill_attention_inputs(
            len(input_lengths),
            input_lengths,
            PAGE_SIZE,
            dtype=torch.bfloat16,
            with_kv_cache_block_ids=False,
            empty_prefix=True,
        )
        self._run_ragged_without_cache(config, inputs, input_lengths)

    def test_ragged_rope_without_cache_applies_fused_rope(self) -> None:
        # RoPE required but no cache to write: the fused op must return
        # post-RoPE QKV with store_cache=False.
        input_lengths = [17, 9]
        config = self._attn_configs(KvCacheDataType.BASE, rope_style=RopeStyle.Base)
        inputs = self._create_prefill_attention_inputs(
            len(input_lengths),
            input_lengths,
            PAGE_SIZE,
            dtype=torch.bfloat16,
            with_kv_cache_block_ids=False,
        )
        self._run_ragged_without_cache(config, inputs, input_lengths)

    def test_prefix_reused_blocks_and_fp8_boundaries(self) -> None:
        input_lengths = [5, 3]
        prefix_lengths = [8, 0]
        block_ids = torch.tensor([[5, 4], [2, 3]], dtype=torch.int32)
        total_blocks = 6
        config = self._attn_configs(KvCacheDataType.FP8)
        inputs = self._create_prefill_attention_inputs(
            len(input_lengths),
            input_lengths,
            PAGE_SIZE,
            dtype=torch.bfloat16,
            prefix_lengths=prefix_lengths,
            block_ids=block_ids,
        )
        qkv = self._random_qkv(config, sum(input_lengths), scale=1.0)
        key_start = config.head_num * config.size_per_head
        value_start = (config.head_num + config.kv_head_num) * config.size_per_head
        boundary_values = torch.tensor(
            [440.0, 450.0, 460.0, 1000.0, -440.0, -450.0, -460.0, -1000.0],
            dtype=torch.bfloat16,
            device=self.device,
        ).repeat(config.kv_head_num * config.size_per_head // 8)
        # Token 5 is at position zero, so K exercises the exact boundary values
        # before saturating conversion; the other K tokens still cover RoPE first.
        qkv[5, key_start:value_start] = boundary_values
        qkv[5, value_start:] = boundary_values
        qkv[0, key_start:value_start] = 1000
        qkv[1, key_start:value_start] = -1000

        reference_cache, fused_cache = self._cache_pair(config, total_blocks)
        reference_attn = PyFlashinferPrefillPagedAttnOp(config, inputs)
        params = reference_attn.prepare(inputs)
        query, key, value = self._reference_rope(config, qkv, params)
        self._write_reference_cache(
            reference_cache,
            key,
            value,
            input_lengths,
            prefix_lengths,
            block_ids,
            PAGE_SIZE,
        )
        expected = reference_attn.forward(query, reference_cache)
        actual = PyFlashinferPagedPrefillImpl(config, inputs).forward(qkv, fused_cache)

        self._assert_cache_equal(
            fused_cache.kv_cache_base,
            reference_cache.kv_cache_base,
            exact=False,
            written_locations=self._cache_write_locations(
                input_lengths, prefix_lengths, block_ids, PAGE_SIZE
            ),
            # This fixture deliberately saturates positive and negative FP8
            # boundary values. Its sparse mismatch signs are not a statistical
            # rounding-bias sample; the randomized accuracy cases above are.
            check_rounding_bias=False,
        )
        self._assert_fp8_scale_writes(
            fused_cache,
            input_lengths,
            prefix_lengths,
            block_ids,
            PAGE_SIZE,
        )
        expected_boundary = (
            boundary_values.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
            .to(torch.float8_e4m3fn)
            .reshape(config.kv_head_num, config.size_per_head)
        )
        for kv_index in (0, 1):
            self.assertTrue(
                torch.equal(
                    fused_cache.kv_cache_base[2, kv_index, :, 0, :].view(torch.uint8),
                    expected_boundary.view(torch.uint8),
                ),
                "FP8 saturation boundary must be written bit-exactly",
            )
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=2e-2
        )

    def test_python_cuda_graph_prepare_updates_fused_cache_metadata(self) -> None:
        capture_lengths = [12, 4]
        capture_prefix = [16, 16]
        replay_lengths = [8, 8]
        replay_prefix = [8, 16]
        capture_blocks = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
        replay_blocks = torch.tensor([[4, 5, 6, 7], [0, 1, 2, 3]], dtype=torch.int32)
        config = self._attn_configs(KvCacheDataType.FP8)
        capture_inputs = self._create_prefill_attention_inputs(
            len(capture_lengths),
            capture_lengths,
            PAGE_SIZE,
            dtype=torch.bfloat16,
            prefix_lengths=capture_prefix,
            block_ids=capture_blocks,
            is_cuda_graph=True,
            graph_step=max(capture_lengths),
        )
        replay_inputs = self._create_prefill_attention_inputs(
            len(replay_lengths),
            replay_lengths,
            PAGE_SIZE,
            dtype=torch.bfloat16,
            prefix_lengths=replay_prefix,
            block_ids=replay_blocks,
            is_cuda_graph=True,
            graph_step=max(capture_lengths),
        )
        for inputs in (capture_inputs, replay_inputs):
            copy_params = PyPrefillCudaGaphCopyParams()
            copy_params.cuda_graph_prefill_batch_size = torch.tensor(
                [len(capture_lengths)], dtype=torch.int32
            ).pin_memory()
            copy_params.max_seq_len = max(capture_lengths)
            copy_params.max_batch_size = len(capture_lengths)
            inputs.prefill_cuda_graph_copy_params = copy_params
        graph_impl = PyFlashinferPagedPrefillImpl(config, capture_inputs)
        graph_cache, eager_cache = self._cache_pair(config, 8)
        cache_seed = graph_cache.kv_cache_base.clone()
        static_qkv = self._random_qkv(config, sum(capture_lengths), scale=1.0)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            graph_impl.forward(static_qkv, graph_cache)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph_cache.kv_cache_base.copy_(cache_seed)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = graph_impl.forward(static_qkv, graph_cache)

        replay_qkv = torch.randn_like(static_qkv)
        static_qkv.copy_(replay_qkv)
        graph_cache.kv_cache_base.copy_(cache_seed)
        capture_inputs.input_lengths.copy_(replay_inputs.input_lengths)
        capture_inputs.prefix_lengths.copy_(replay_inputs.prefix_lengths)
        capture_inputs.sequence_lengths.copy_(replay_inputs.sequence_lengths)
        capture_inputs.cu_seqlens_device.copy_(replay_inputs.cu_seqlens_device)
        capture_inputs.cu_kv_seqlens_device.copy_(replay_inputs.cu_kv_seqlens_device)
        capture_inputs.combo_position_ids.copy_(replay_inputs.combo_position_ids)
        # This impl-level test mirrors the runner's in-place tensor refresh.
        # The fixture built replay padding with the selected graph's explicit
        # capture step, so the stride formula has a single source of truth.
        capture_inputs.padding_offset.copy_(replay_inputs.padding_offset)
        capture_inputs.context_total_kv_length = replay_inputs.context_total_kv_length
        capture_inputs.kv_cache_kernel_block_id.copy_(
            replay_inputs.kv_cache_kernel_block_id
        )
        capture_inputs.kv_cache_kernel_block_id_device.copy_(
            replay_inputs.kv_cache_kernel_block_id_device
        )
        graph_impl.prepare_cuda_graph(capture_inputs)
        graph_cache.kv_scale_base.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()

        eager_inputs = self._create_prefill_attention_inputs(
            len(replay_lengths),
            replay_lengths,
            PAGE_SIZE,
            dtype=torch.bfloat16,
            prefix_lengths=replay_prefix,
            block_ids=replay_blocks,
        )
        eager_out = PyFlashinferPagedPrefillImpl(config, eager_inputs).forward(
            replay_qkv, eager_cache
        )
        self.assertTrue(
            torch.equal(
                graph_cache.kv_cache_base.view(torch.uint8),
                eager_cache.kv_cache_base.view(torch.uint8),
            ),
            "CUDA graph replay must update fused cache metadata bit-exactly",
        )
        torch.testing.assert_close(
            static_out.float(), eager_out.float(), rtol=2e-2, atol=2e-2
        )
        self._assert_fp8_scale_writes(
            graph_cache,
            replay_lengths,
            replay_prefix,
            replay_blocks,
            PAGE_SIZE,
        )
        self._assert_fp8_scale_writes(
            eager_cache,
            replay_lengths,
            replay_prefix,
            replay_blocks,
            PAGE_SIZE,
        )

    def test_cuda_graph_refresh_rejects_out_of_bound_replay(self) -> None:
        capture_lengths = [8, 8]
        capture_prefix = [8, 8]
        config = self._attn_configs(KvCacheDataType.FP8)
        capture_inputs = self._create_prefill_attention_inputs(
            len(capture_lengths),
            capture_lengths,
            PAGE_SIZE,
            dtype=torch.bfloat16,
            prefix_lengths=capture_prefix,
            is_cuda_graph=True,
        )
        impl = PyFlashinferPagedPrefillImpl(config, capture_inputs)

        # Same total rows as capture (FlashInfer rejects a row-count increase by
        # itself), but one request exceeds the captured max_seq_len, which only
        # our guard catches.
        over_bound = self._create_prefill_attention_inputs(
            len(capture_lengths),
            [15, 1],
            PAGE_SIZE,
            dtype=torch.bfloat16,
            prefix_lengths=capture_prefix,
            is_cuda_graph=True,
        )
        with self.assertRaisesRegex(RuntimeError, "captured max_seq_len"):
            impl.prepare_cuda_graph(over_bound)

        # Dropping the prefix flips count_length and use_paged_fmha.
        no_prefix = self._create_prefill_attention_inputs(
            len(capture_lengths),
            capture_lengths,
            PAGE_SIZE,
            dtype=torch.bfloat16,
            prefix_lengths=[0, 0],
            is_cuda_graph=True,
        )
        with self.assertRaisesRegex(RuntimeError, "flip whether a prefix is present"):
            impl.prepare_cuda_graph(no_prefix)

        replacement_buffers = self._create_prefill_attention_inputs(
            len(capture_lengths),
            capture_lengths,
            PAGE_SIZE,
            dtype=torch.bfloat16,
            prefix_lengths=capture_prefix,
            is_cuda_graph=True,
        )
        with self.assertRaisesRegex(
            RuntimeError, "stable capture buffer for padding_offset"
        ):
            impl.prepare_cuda_graph(replacement_buffers)

    def test_prefill_shape_validation_rejects_wrong_rows_and_rank(self) -> None:
        input_lengths = [2, 1]
        inputs = self._create_prefill_attention_inputs(
            len(input_lengths),
            input_lengths,
            PAGE_SIZE,
            dtype=torch.bfloat16,
            prefix_lengths=[0, 0],
        )
        config = self._attn_configs(KvCacheDataType.BASE)
        total_tokens = sum(input_lengths)
        qkv_width = (config.head_num + 2 * config.kv_head_num) * config.size_per_head

        ragged = PyFlashinferPrefillAttnOp(config)
        valid_ragged = torch.empty(
            total_tokens,
            qkv_width,
            dtype=torch.bfloat16,
            device=self.device,
        )
        with self.assertRaisesRegex(
            RuntimeError, r"ragged prefill prepare\(\) must be called"
        ):
            ragged.forward(valid_ragged, None)
        ragged.prepare(inputs)
        invalid_ragged = (
            torch.empty(
                total_tokens - 1,
                qkv_width,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.empty(
                total_tokens,
                1,
                1,
                qkv_width,
                dtype=torch.bfloat16,
                device=self.device,
            ),
        )
        for qkv in invalid_ragged:
            with self.subTest(kind="ragged", shape=tuple(qkv.shape)):
                with self.assertRaisesRegex(
                    ValueError, "expected_tokens=3.*expected_numel"
                ):
                    ragged.forward(qkv, None)

        paged = PyFlashinferPrefillPagedAttnOp(config, inputs)
        cache, _, _ = self._create_kv_cache(
            2,
            PAGE_SIZE,
            config.kv_head_num,
            config.size_per_head,
            dtype=torch.bfloat16,
        )
        valid_paged = torch.empty(
            total_tokens,
            config.head_num,
            config.size_per_head,
            dtype=torch.bfloat16,
            device=self.device,
        )
        with self.assertRaisesRegex(
            RuntimeError, r"paged prefill prepare\(\) must be called"
        ):
            paged.forward(valid_paged, cache)
        paged.prepare(inputs)
        invalid_paged = (
            torch.empty(
                total_tokens - 1,
                config.head_num,
                config.size_per_head,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            torch.empty(
                total_tokens,
                1,
                config.head_num,
                config.size_per_head,
                dtype=torch.bfloat16,
                device=self.device,
            ),
        )
        for query in invalid_paged:
            with self.subTest(kind="paged", shape=tuple(query.shape)):
                with self.assertRaisesRegex(
                    ValueError, "expected_tokens=3.*expected_numel"
                ):
                    paged.forward(query, cache)

    def test_empty_prefill_batch_has_actionable_error(self) -> None:
        inputs = self._create_prefill_attention_inputs(
            1,
            [1],
            PAGE_SIZE,
            dtype=torch.bfloat16,
            with_kv_cache_block_ids=False,
        )
        inputs.input_lengths = torch.empty(0, dtype=torch.int32).pin_memory()
        config = self._attn_configs(KvCacheDataType.BASE)
        with self.assertRaisesRegex(RuntimeError, "requires non-empty input_lengths"):
            PyFlashinferPrefillImpl(config, inputs)

    def test_base_fixture_models_zero_length_cuda_graph_padding(self) -> None:
        inputs = self._create_prefill_attention_inputs(
            2,
            [3, 0],
            PAGE_SIZE,
            dtype=torch.bfloat16,
            prefix_lengths=[0, 0],
            is_cuda_graph=True,
        )
        self.assertEqual(inputs.input_lengths.tolist(), [3, 0])
        self.assertEqual(inputs.cu_seqlens.tolist(), [0, 3, 3])
        self.assertEqual(inputs.padding_offset.tolist(), [0, 0, 0])

        with self.assertRaisesRegex(ValueError, "reserved for padded CUDA graph"):
            self._create_prefill_attention_inputs(
                2,
                [3, 0],
                PAGE_SIZE,
                dtype=torch.bfloat16,
                prefix_lengths=[0, 0],
            )

    def test_factory_does_not_swallow_invalid_fused_prefill_input(self) -> None:
        class InvalidInputImpl(FMHAImplBase):
            @staticmethod
            def support(attn_configs, attn_inputs) -> bool:
                return True

            def __init__(self, attn_configs, attn_inputs, parallelism_config=None):
                raise InvalidFusedPrefillInputError("invalid fused prefill fixture")

            def forward(self, qkv, kv_cache, layer_idx=0):
                raise AssertionError("unreachable")

        inputs = self._create_prefill_attention_inputs(
            1,
            [1],
            PAGE_SIZE,
            dtype=torch.bfloat16,
            with_kv_cache_block_ids=False,
        )
        config = self._attn_configs(KvCacheDataType.BASE)
        with patch.object(attn_factory, "PREFILL_MHA_IMPS", [InvalidInputImpl]):
            with self.assertRaisesRegex(
                InvalidFusedPrefillInputError, "invalid fused prefill fixture"
            ):
                attn_factory.get_fmha_impl(config, None, inputs)


if __name__ == "__main__":
    unittest.main()
