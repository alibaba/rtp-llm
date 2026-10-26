"""Real CUDA copies across the LayerKVCache physical/kernel page boundary."""

import unittest

import torch

from rtp_llm.models_py.modules.hybrid.full_checkpoint import save_full_checkpoints
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs


class FullCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "this target requires NVIDIA CUDA")

    def _case(self, mla, with_scale):
        kernel, base, pages = 2, 8, 80
        shape = (pages, kernel, 7) if mla else (pages, 2, 3, kernel, 4)
        values = torch.arange(torch.tensor(shape).prod().item(), device="cuda")
        cache = LayerKVCache(values.reshape(shape).to(torch.float32), kernel, 0, "full")
        if with_scale:
            scale_shape = (pages, kernel, 2) if mla else (pages, 2 * 3 * kernel)
            cache.kv_scale_base = torch.arange(
                torch.tensor(scale_shape).prod().item(),
                device="cuda",
                dtype=torch.float32,
            ).reshape(scale_shape)
        inputs = PyAttentionInputs()
        inputs.is_prefill = True
        inputs.prefix_lengths = torch.tensor([0, 9], dtype=torch.int32)
        inputs.input_lengths = torch.tensor([16, 10], dtype=torch.int32)
        inputs.kv_cache_kernel_block_id = torch.tensor(
            [list(range(4, 16)), list(range(20, 32))], dtype=torch.int32
        )
        inputs.cache_save_positions = torch.tensor(
            [[5, 16, -1, 17], [11, -1, 19, 8]], dtype=torch.int64
        )
        inputs.cache_save_block_ids = torch.tensor(
            [[10, 2, -1, 13], [11, -1, 12, 14]], dtype=torch.int32
        )
        before = cache.kv_cache_base.clone()
        scale_before = cache.kv_scale_base.clone() if with_scale else None
        save_full_checkpoints(cache, inputs, base)
        expected = before.clone()
        scale_expected = scale_before.clone() if with_scale else None
        # Independent endpoint oracle: concatenate each physical block's kernel
        # pages into its token axis, then replace only the prefix owned by P.
        for source_base, target_base, count in [(1, 10, 5), (6, 11, 3), (7, 12, 3)]:
            for actual, wanted in [(before, expected)] + (
                [(scale_before, scale_expected)] if with_scale else []
            ):
                if not mla and actual.ndim == 2:
                    actual = actual.view(pages, 2, 3, kernel)
                    wanted = wanted.view(pages, 2, 3, kernel)
                for token in range(count):
                    src, offset = divmod(source_base * base + token, kernel)
                    dst = target_base * (base // kernel) + token // kernel
                    if mla:
                        wanted[dst, offset] = actual[src, offset]
                    else:
                        wanted[dst, :, :, offset] = actual[src, :, :, offset]
        torch.cuda.synchronize()
        torch.testing.assert_close(cache.kv_cache_base, expected, rtol=0, atol=0)
        if with_scale:
            torch.testing.assert_close(
                cache.kv_scale_base, scale_expected, rtol=0, atol=0
            )
        # Exact full-tensor comparison above covers untouched source pages,
        # destination suffix sentinels, padded slots, aliases, and out-of-step P.
        saved = cache.kv_cache_base[40:44].clone()
        restored = torch.full_like(saved, -1)
        restored.copy_(saved)
        axis = 1 if mla else 3
        for token in range(5):
            page, offset = divmod(token, kernel)
            torch.testing.assert_close(
                restored[page].select(axis - 1, offset),
                before[4 + page].select(axis - 1, offset),
                rtol=0,
                atol=0,
            )

    def test_mha_kernel_pages(self):
        self._case(mla=False, with_scale=False)

    def test_mha_flattened_scale_pages(self):
        self._case(mla=False, with_scale=True)

    def test_mla_kernel_pages(self):
        self._case(mla=True, with_scale=False)

    def test_mla_scale_pages(self):
        self._case(mla=True, with_scale=True)


if __name__ == "__main__":
    unittest.main()
