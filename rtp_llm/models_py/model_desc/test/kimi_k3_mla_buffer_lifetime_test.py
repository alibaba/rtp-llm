import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.kimi_k3 import mla_prefill
from rtp_llm.ops import KvCacheDataType


class KimiK3MlaBufferLifetimeTest(unittest.TestCase):
    def test_expanded_bf16_kv_released_before_fp8_attention(self):
        op = mla_prefill.KimiK3MlaPrefillOp.__new__(
            mla_prefill.KimiK3MlaPrefillOp
        )
        op.kv_cache_type = KvCacheDataType.FP8
        op._prefix_plan = SimpleNamespace(chunked=False)
        op.num_heads = 2
        op.v_head_dim = 4
        key_ref, value_ref = [], []

        def project(*_args):
            key = torch.ones((3, 2, 4), dtype=torch.bfloat16)
            value = torch.ones((3, 2, 4), dtype=torch.bfloat16)
            key_ref.append(weakref.ref(key))
            value_ref.append(weakref.ref(value))
            return key, value

        def quantize(q, key, value):
            return tuple(
                torch.empty_like(x, dtype=torch.float8_e4m3fn)
                for x in (q, key, value)
            )

        def attention(*_args):
            gc.collect()
            self.assertIsNone(key_ref[0]())
            self.assertIsNone(value_ref[0]())
            return torch.zeros((3, 2, 4), dtype=torch.bfloat16)

        op.prefill_wrapper = SimpleNamespace(run=attention)
        q = torch.ones((3, 2, 4), dtype=torch.bfloat16)
        compressed = torch.ones((3, 8), dtype=torch.bfloat16)
        rope = torch.ones((3, 4), dtype=torch.bfloat16)
        with (
            patch.object(op, "_reuse_kv_cache_indexed_batched", side_effect=lambda a, b, _c: (a, b)),
            patch.object(op, "_make_kv_b_proj", return_value=None),
            patch.object(op, "_project_kv", side_effect=project),
            patch.object(mla_prefill, "quantize_qkv_fp8", new=quantize),
        ):
            result = op.forward(q, compressed, rope, None, 0)
        self.assertEqual(result.shape, (3, 2, 4))

    def test_sliced_prefix_kv_released_before_each_attention_call(self):
        op = mla_prefill.KimiK3MlaPrefillOp.__new__(
            mla_prefill.KimiK3MlaPrefillOp
        )
        op.kv_cache_type = KvCacheDataType.FP8
        op._prefix_plan = SimpleNamespace(
            chunked=True, slices=(SimpleNamespace(owner=0, start=0, length=3),)
        )
        op.num_heads = 2
        op.v_head_dim = 4
        op.qk_rope_head_dim = 2
        op.kv_lora_rank = 8
        op.token_per_block = 3
        op.qo_indptr = torch.tensor((0, 3), dtype=torch.int32)
        op._prefix_q_offsets = (0, 3)
        op._prefix_lens = (3,)
        op.reuse_cache_page_indice = torch.tensor((0,), dtype=torch.int32)
        op.batch_reuse_info_vec = torch.tensor((0,), dtype=torch.int32)
        refs = []

        def project(*_args):
            key = torch.ones((3, 2, 4), dtype=torch.bfloat16)
            value = torch.ones((3, 2, 4), dtype=torch.bfloat16)
            refs.append((weakref.ref(key), weakref.ref(value)))
            return key, value

        def quantize(*tensors):
            return tuple(torch.empty_like(x, dtype=torch.float8_e4m3fn) for x in tensors)

        def run_partial(*_args, **kwargs):
            gc.collect()
            expected = 0 if kwargs["causal"] else 1
            self.assertIsNone(refs[expected][0]())
            self.assertIsNone(refs[expected][1]())
            return (torch.zeros((3, 2, 4), dtype=torch.bfloat16),
                    torch.zeros((3, 2), dtype=torch.float32))

        op.prefill_wrapper = SimpleNamespace(run_partial=run_partial, max_q=3)
        cache = SimpleNamespace(
            kv_cache_base=torch.empty((3, 10), dtype=torch.float8_e4m3fn)
        )
        q = torch.ones((3, 2, 4), dtype=torch.bfloat16)
        compressed = torch.ones((3, 8), dtype=torch.bfloat16)
        rope = torch.ones((3, 2), dtype=torch.bfloat16)
        with (
            patch.object(op, "_make_kv_b_proj", return_value=None),
            patch.object(op, "_project_kv", new=project),
            patch.object(mla_prefill, "quantize_qkv_fp8", new=quantize),
            patch.object(mla_prefill, "quantize_fp8", new=lambda x: quantize(x)[0]),
            patch.object(mla_prefill, "gather_fp8_prefix_slice", new=lambda *_args, **_kwargs: None),
            patch.object(mla_prefill, "merge_mla_states_in_place", new=lambda *_args: None),
        ):
            result = op.forward(q, compressed, rope, cache, 0)
        self.assertEqual(len(refs), 2)
        self.assertEqual(result.shape, (3, 2, 4))

    def test_fused_cache_insert_releases_bf16_projection_before_attention(self):
        op = mla_prefill.KimiK3MlaPrefillOp.__new__(
            mla_prefill.KimiK3MlaPrefillOp
        )
        op.kv_cache_type = KvCacheDataType.FP8
        op._prefix_plan = SimpleNamespace(chunked=False)
        op._prefix_lens = (0,)
        op.qk_nope_head_dim = 2
        op.qk_rope_head_dim = 2
        op.v_head_dim = 4
        op.num_heads = 2
        op.kv_lora_rank = 8
        op.token_per_block = 3
        projected_refs = []
        cache_writes = []

        class Projection:
            def supports_skip_head_mid(self, *_args):
                return False

            def __call__(self, *_args):
                projected = torch.ones((3, 2, 6), dtype=torch.bfloat16)
                projected_refs.append(weakref.ref(projected))
                return projected

        def fused_epilogue(q, *_args, **_kwargs):
            return tuple(
                torch.empty_like(q, dtype=torch.float8_e4m3fn)
                for _ in range(3)
            )

        def attention(*_args):
            gc.collect()
            self.assertEqual(cache_writes, [True])
            self.assertIsNone(projected_refs[0]())
            return torch.zeros((3, 2, 4), dtype=torch.bfloat16)

        op.prefill_wrapper = SimpleNamespace(run=attention)
        cache = SimpleNamespace(
            kv_cache_base=torch.empty((3, 10), dtype=torch.float8_e4m3fn)
        )
        q = torch.ones((3, 2, 4), dtype=torch.bfloat16)
        compressed = torch.ones((3, 8), dtype=torch.bfloat16)
        rope = torch.ones((3, 2), dtype=torch.bfloat16)
        with (
            patch.object(op, "_make_kv_b_proj", return_value=Projection()),
            patch.object(mla_prefill, "fused_mla_fp8_epilogue", new=fused_epilogue),
            patch.object(mla_prefill.mla_fp8_kernels, "_FP8_DIAGNOSTICS", False),
        ):
            result = op.forward_with_cache_insert(
                q, compressed, rope, cache, 0,
                torch.arange(3, dtype=torch.int64), 1.0, 1.0,
                lambda: cache_writes.append(True),
            )
        self.assertEqual(result.shape, (3, 2, 4))


if __name__ == "__main__":
    unittest.main()
