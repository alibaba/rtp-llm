"""``SparseAttnV4DecodeFp8Op`` smoke construction test.

The op no longer carries a Python reference fallback (production / dev /
CI all have flash_mla wheel; the CPU dequant-loop reference was a slow
dead path). Numerical equivalence is covered by the SM100 smoke gates
(production correctness).
"""

import unittest

from rtp_llm.models_py.modules.dsv4.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)


class TestFlashMlaAvailability(unittest.TestCase):

    def test_op_constructs(self):
        op = SparseAttnV4DecodeFp8Op(n_heads=4, head_dim=512, softmax_scale=1.0)
        self.assertEqual(op.n_heads, 4)
        self.assertEqual(op.head_dim, 512)


if __name__ == "__main__":
    unittest.main()
