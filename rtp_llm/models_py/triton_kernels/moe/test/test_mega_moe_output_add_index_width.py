import re
import unittest

import torch

from rtp_llm.models_py.triton_kernels.moe.output_add import _add_moe_output


class MegaMoEOutputAddIndexWidthTest(unittest.TestCase):
    def test_large_bf16_output_uses_wide_indices(self):
        hidden = 7168
        rows = (1 << 31) // hidden + 1
        size = rows * hidden
        block = 1024

        # Passing dtypes to Triton warmup compiles the production specialization
        # without allocating the tensors or launching the large grid.
        compiled = _add_moe_output.warmup(
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            size,
            True,
            block,
            grid=((size + block - 1) // block,),
        )
        ttir = compiled.asm["ttir"]
        offset_types = re.findall(r"\btt\.addptr\b[^\n]*,\s*tensor<\d+xi(\d+)>", ttir)
        self.assertEqual(
            set(offset_types),
            {"64"},
            f"Expected 64-bit pointer offsets for {size} BF16 elements, "
            f"got offset widths {offset_types}.",
        )
        # A cast after computing the tile base cannot recover an overflow.
        self.assertRegex(ttir, r"\barith\.(muli|shli)\b[^\n]*:\s*i64\b")


if __name__ == "__main__":
    unittest.main()
