import itertools
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules import FusedSiluAndMul


class ActivationTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 1024]
    # 855/1710/3420 exercise d % (16 // sizeof) != 0 (Qwen2.5-VL vision MLP
    # uses d = 3420); 512/4096 are the aligned control group.
    HIDDEN_SIZES = [512, 855, 1710, 3420, 4096]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        self.device = "cuda"

    def _run_silu_and_mul_test(self, num_tokens: int, hidden_size: int, dtype: _dtype):
        torch.manual_seed(0)
        gate_up = torch.randn(
            num_tokens, 2 * hidden_size, dtype=dtype, device=self.device
        )
        ref = (
            torch.nn.functional.silu(gate_up[..., :hidden_size].float())
            * gate_up[..., hidden_size:].float()
        )
        # Output closeness cannot tell the two routes apart, so assert the
        # routing of FusedSiluAndMul.forward directly: aligned d must call the
        # flashinfer act_and_mul kernel, misaligned d the torch fallback.
        # (Imports stay lazy to match the module under test.)
        import flashinfer

        calls = []
        real = flashinfer.activation.silu_and_mul

        def spy(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        flashinfer.activation.silu_and_mul = spy
        try:
            out = FusedSiluAndMul()(gate_up)
        finally:
            flashinfer.activation.silu_and_mul = real
        d = gate_up.shape[-1] // 2
        vec_size = 16 // gate_up.element_size()
        if d % vec_size == 0:
            self.assertEqual(len(calls), 1, "aligned d must use flashinfer")
        else:
            self.assertEqual(len(calls), 0, "misaligned d must use torch fallback")
        # Tolerance stays at 1e-2: routing is asserted above and this bound
        # only needs to catch gross errors. Tightening is unsafe for bf16 --
        # the fallback rounds at sigmoid and each mul (~3 * 2**-8 worst case).
        self.assertTrue(torch.allclose(out.float(), ref, atol=1e-2, rtol=1e-2))

    def test_silu_and_mul(self):
        for params in itertools.product(self.NUM_TOKENS, self.HIDDEN_SIZES, self.DTYPES):
            with self.subTest(num_tokens=params[0], hidden_size=params[1], dtype=params[2]):
                self._run_silu_and_mul_test(*params)


if __name__ == "__main__":
    main()
