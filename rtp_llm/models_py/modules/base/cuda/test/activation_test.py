import itertools
from unittest import SkipTest, TestCase, main

import torch
from rtp_llm.models_py.modules import FusedSiluAndMul
from torch import dtype as _dtype


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
        for params in itertools.product(
            self.NUM_TOKENS, self.HIDDEN_SIZES, self.DTYPES
        ):
            with self.subTest(
                num_tokens=params[0], hidden_size=params[1], dtype=params[2]
            ):
                self._run_silu_and_mul_test(*params)

    def test_situ_row_counts_reuse_jit(self):
        from rtp_llm.models_py.triton_kernels.moe.activation import (
            _situ_and_mul_kernel,
            situ_and_mul,
        )
        from triton import knobs

        for width in (6144, 33792):
            for split, inplace, up_beta in ((True, False, 1.5), (False, True, None)):
                _situ_and_mul_kernel.device_caches.clear()
                attempts, previous = [], knobs.runtime.jit_cache_hook

                def listen(**event):
                    if event["fn"].jit_function is _situ_and_mul_kernel:
                        attempts.append(event["key"])
                    if previous is not None:
                        return previous(**event)
                    return None

                knobs.runtime.jit_cache_hook = listen
                try:
                    for rows in (1, 17):
                        if split:
                            gate, up = torch.randn(
                                rows,
                                2 * width,
                                device=self.device,
                                dtype=torch.bfloat16,
                            ).split(width, dim=-1)
                        else:
                            gate = torch.randn(
                                rows, width, device=self.device, dtype=torch.bfloat16
                            )
                            up = torch.randn_like(gate)
                        g, u = gate.float(), up.float()
                        if up_beta is not None:
                            u = up_beta * torch.tanh(u / up_beta)
                        expected = (
                            1.7 * torch.tanh(g / 1.7) * torch.sigmoid(g) * u
                        ).to(gate.dtype)
                        output = situ_and_mul(gate, up, 1.7, up_beta, inplace=inplace)
                        if inplace:
                            self.assertEqual(output.data_ptr(), gate.data_ptr())
                        torch.testing.assert_close(
                            output, expected, rtol=0.016, atol=0.016
                        )
                finally:
                    knobs.runtime.jit_cache_hook = previous
                self.assertEqual(len(attempts), 1, attempts)


if __name__ == "__main__":
    main()
