"""Warmup uses live cache strides for every context-parallel width."""

import math
import unittest

import torch

from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV
from rtp_llm.models_py.modules.dsv41.fp8._v41_attention_jit_warmup import (
    PoolLayout,
    _warm_swa_byte_slices,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class AttentionPoolWarmupTest(unittest.TestCase):
    def test_actual_ring_stride_for_unsharded_and_cp_pools(self):
        for cp_size in (1, 2, 3, 4, 5, 8):
            for entries in (128, 136):
                alignment = math.lcm(16896, cp_size)
                stride = ((entries * 528 + alignment - 1) // alignment) * alignment
                with self.subTest(cp_size=cp_size, entries=entries):
                    layout = PoolLayout(SWA_KV, entries, stride, 4096, 4096, 1)
                    _warm_swa_byte_slices(
                        layout, cp_size, cp_size - 1, torch.device("cuda")
                    )
                    torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
