import unittest
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_cp_flashinfer import (
    CPFlashInferImpl,
)
from rtp_llm.ops.compute_ops import get_typemeta
from rtp_llm.ops.fused_rope_kvcache_op import FusedRopeKVCachePrefillOpBase


class _FakeRope:
    def __init__(self) -> None:
        self.calls = 0

    def forward(self, qkv, kv_cache, params):
        self.calls += 1
        return qkv + 1


class _FakeFmha:
    def forward(self, fmha_input, kv_cache, params):
        return fmha_input


class TestCPFlashInferImpl(unittest.TestCase):
    @staticmethod
    def _make_impl(need_rope_kv_cache: bool) -> CPFlashInferImpl:
        impl = object.__new__(CPFlashInferImpl)
        impl.need_rope_kv_cache = need_rope_kv_cache
        impl.rope_kvcache_impl = _FakeRope()
        impl.rope_params = object()
        impl.fmha_impl = _FakeFmha()
        impl.fmha_params = object()
        impl.attn_inputs = SimpleNamespace(cache_store_inputs=None)
        impl.write_cache_store_impl = None
        return impl

    def test_support_requires_runtime_cp_metadata(self) -> None:
        no_cp_inputs = SimpleNamespace(context_parallel_info=None)
        cp_inputs = SimpleNamespace(context_parallel_info=object())

        self.assertFalse(CPFlashInferImpl.support(None, no_cp_inputs))
        self.assertTrue(CPFlashInferImpl.support(None, cp_inputs))

    def test_layer_zero_still_runs_configured_rope(self) -> None:
        impl = self._make_impl(need_rope_kv_cache=True)
        qkv = torch.tensor([1.0])

        output = impl.forward(qkv, kv_cache=None, layer_idx=0)

        self.assertEqual(impl.rope_kvcache_impl.calls, 1)
        self.assertTrue(torch.equal(output, qkv + 1))

    def test_nonzero_layer_does_not_override_disabled_rope(self) -> None:
        impl = self._make_impl(need_rope_kv_cache=False)
        qkv = torch.tensor([1.0])

        output = impl.forward(qkv, kv_cache=None, layer_idx=7)

        self.assertEqual(impl.rope_kvcache_impl.calls, 0)
        self.assertTrue(torch.equal(output, qkv))


class TestCPRopePositionIds(unittest.TestCase):
    @staticmethod
    def _make_attn_inputs(
        shuffle_indices,
        combo_position_ids=None,
        prefix_lengths=(0,),
        chunk_lengths=None,
        actual_input_lengths=None,
    ):
        token_count = len(shuffle_indices)
        if chunk_lengths is None:
            chunk_lengths = (token_count,)
        if actual_input_lengths is None:
            actual_input_lengths = (max(shuffle_indices) + 1,)
        cu_seqlens = torch.tensor([0, *accumulate(chunk_lengths)], dtype=torch.int32)
        return SimpleNamespace(
            kv_cache_kernel_block_id=None,
            combo_position_ids=combo_position_ids,
            context_parallel_info=SimpleNamespace(
                prefill_shuffle_indices=torch.tensor(
                    shuffle_indices, dtype=torch.int32
                ),
                prefill_cp_chunk_lengths=torch.tensor(
                    chunk_lengths, dtype=torch.int32
                ),
                prefill_actual_input_lengths_cpu=torch.tensor(
                    actual_input_lengths, dtype=torch.int32
                ),
            ),
            total_tokens=token_count,
            padding_offset=None,
            cu_seqlens_device=cu_seqlens,
            cu_kv_seqlens_device=cu_seqlens,
            input_lengths=torch.tensor(chunk_lengths, dtype=torch.int32),
            prefix_lengths=torch.tensor(prefix_lengths, dtype=torch.int32),
            prefix_lengths_device=torch.tensor(prefix_lengths, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
            context_total_kv_length=token_count + sum(prefix_lengths),
            dtype=get_typemeta(torch.empty(0, dtype=torch.bfloat16)),
        )

    @staticmethod
    def _make_op(index_factor: int) -> FusedRopeKVCachePrefillOpBase:
        return FusedRopeKVCachePrefillOpBase(
            SimpleNamespace(rope_config=SimpleNamespace(index_factor=index_factor))
        )

    def test_plain_rope_uses_cp_shuffle_indices(self) -> None:
        attn_inputs = self._make_attn_inputs(
            [0, 1, 6, 7], torch.empty(0, dtype=torch.int32)
        )

        params = self._make_op(index_factor=1).prepare(attn_inputs)

        self.assertIs(
            params.position_ids,
            attn_inputs.context_parallel_info.prefill_shuffle_indices,
        )

    def test_plain_rope_adds_prefix_per_request_and_zeros_padding(self) -> None:
        attn_inputs = self._make_attn_inputs(
            [0, 1, 6, 7, 0, 3],
            torch.empty(0, dtype=torch.int32),
            prefix_lengths=(4, 8),
            chunk_lengths=(4, 2),
            actual_input_lengths=(7, 4),
        )

        params = self._make_op(index_factor=1).prepare(attn_inputs)

        self.assertEqual(params.position_ids.tolist(), [4, 5, 10, 0, 8, 11])

    def test_explicit_mrope_position_ids_take_precedence(self) -> None:
        combo_position_ids = torch.tensor(
            [0, 10, 20, 1, 11, 21, 6, 16, 26, 7, 17, 27],
            dtype=torch.int32,
        )
        attn_inputs = self._make_attn_inputs([0, 1, 6, 7], combo_position_ids)

        params = self._make_op(index_factor=3).prepare(attn_inputs)

        self.assertIs(params.position_ids, combo_position_ids)


class _CaptureDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_meta = None

    def forward(
        self,
        hidden_states,
        residual,
        fmha_impl,
        kv_cache,
        attention_inputs,
        attn_meta,
    ):
        self.attn_meta = attn_meta
        return hidden_states, residual


class _IdentityNorm(torch.nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states, residual


class TestQwen3NextCPRuntimeSelection(unittest.TestCase):
    def test_target_verify_uses_non_cp_metadata(self):
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextModel
        from rtp_llm.ops import CPRotateMethod, ParallelismConfig

        parallelism_config = ParallelismConfig()
        parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER

        model = object.__new__(Qwen3NextModel)
        torch.nn.Module.__init__(model)
        model.parallelism_config = parallelism_config
        capture_layer = _CaptureDecoderLayer()
        model.layers = torch.nn.ModuleList([capture_layer])
        model.norm = _IdentityNorm()
        model.kv_cache = None

        hidden_states = torch.randn(4, 8)
        attention_inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=True,
            prefix_lengths=torch.tensor([3], dtype=torch.int32),
            cu_seqlens_device=torch.tensor([0, 4], dtype=torch.int32),
            context_parallel_info=None,
            cache_store_inputs=object(),
        )
        inputs = SimpleNamespace(attention_inputs=attention_inputs)
        fmha_impl = SimpleNamespace(fmha_params=None)

        with patch.object(Qwen3NextModel, "word_embedding", return_value=hidden_states):
            with patch(
                "rtp_llm.models_py.model_desc.qwen3_next.select_block_map_for_layer"
            ):
                output = model.forward(inputs, fmha_impl)

        self.assertIsNone(capture_layer.attn_meta.prefill_conv1d_meta)
        self.assertTrue(capture_layer.attn_meta.is_target_verify)
        self.assertFalse(capture_layer.attn_meta.is_cp_linear_attn)
        self.assertIsNone(capture_layer.attn_meta.cp_write_cache_store_impl)
        torch.testing.assert_close(output.hidden_states, hidden_states)

if __name__ == "__main__":
    unittest.main()
