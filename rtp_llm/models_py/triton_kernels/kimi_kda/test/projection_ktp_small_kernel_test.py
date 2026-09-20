"""Manual CUDA correctness tests for projection-KTP small layout kernels."""

import unittest
from functools import partial
from itertools import product

import torch

import rtp_llm.models_py.modules.kimi_k3.projection_ktp as projection_ktp
import rtp_llm.models_py.triton_kernels.kimi_kda.projection_ktp as small_kernels
from rtp_llm.model_loader.per_block_fp8_quant_weight import per_block_cast_to_fp8
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import _transform_scale_ue8m0
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)

_BATCHES = (1, 2, 4, 8, 16, 32, 3, 7, 17, 31, 33)
_KTP_SIZES = (1, 2, 4, 8, 16)


def projection_options(ranks=8, rank=0):
    return dict(
        total_heads=96,
        head_dim=128,
        forget_latent_size=128,
        ktp_size=ranks,
        ktp_rank=rank,
    )


def bf16(*shape):
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16)


assert_exact = partial(torch.testing.assert_close, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ProjectionKtpSmallKernelTest(unittest.TestCase):
    def test_fp8_strided_forget_is_bitwise_exact(self):
        width, rows = 1536, 8
        projected = bf16(rows, 4 * width + 128 + 96)
        hidden = bf16(rows, 64)
        weight, scales = per_block_cast_to_fp8(
            bf16(width, 128) * 0.02, 128, use_ue8m0=True
        )
        forget = CudaFp8DeepGEMMLinear(weight, _transform_scale_ue8m0(scales, mn=width))
        expected_gate = forget(projected[:, 4 * width : 4 * width + 128].contiguous())
        expected = torch.cat(
            (projected[:, : 4 * width], expected_gate, projected[:, -96:-84]), dim=1
        )
        output = torch.empty_like(expected)

        def run():
            return projection_ktp.pack_ktp_projection_payload(
                hidden,
                lambda _: projected,
                forget,
                output=output,
                optimize=True,
                **projection_options()
            )

        actual = run()
        self.assertIs(actual, output)
        assert_exact(actual, expected)

    def test_production_pack_and_unpack_layouts_are_bitwise_exact(self):
        generator = torch.Generator(device="cuda").manual_seed(20260918)

        def tagged(rows, columns):
            return torch.randint(
                -64,
                65,
                (rows, columns),
                dtype=torch.int16,
                device="cuda",
                generator=generator,
            ).to(torch.bfloat16)

        for ranks, batch in product(_KTP_SIZES, _BATCHES):
            with self.subTest(ktp_size=ranks, batch=batch):
                heads, width = 96 // ranks, 96 // ranks * 128
                rows, payload_width = ranks * batch, 5 * width + heads
                projected, gate = tagged(rows, 4 * width + 224), tagged(rows, width)
                packed = torch.empty(
                    (rows, payload_width), device="cuda", dtype=torch.bfloat16
                )
                actual = small_kernels.pack_ktp_projection_payload_cuda(
                    projected,
                    gate,
                    packed,
                    local_projection_size=width,
                    forget_latent_size=128,
                    local_heads=heads,
                    ktp_rank=ranks - 1,
                )
                beta = 4 * width + 128 + (ranks - 1) * heads
                expected = torch.cat(
                    (
                        projected[:, : 4 * width],
                        gate,
                        projected[:, beta : beta + heads],
                    ),
                    dim=1,
                )
                assert_exact(actual, expected)

                received = tagged(rows, payload_width)
                actual = small_kernels.reassemble_ktp_projection_payload_cuda(
                    received,
                    ktp_size=ranks,
                    physical_batch=batch,
                    local_projection_size=width,
                    local_heads=heads,
                )
                sections = received.reshape(ranks, batch, payload_width).split(
                    (width,) * 5 + (heads,), dim=-1
                )
                expected = [
                    section.permute(1, 0, 2).reshape(batch, -1).clone()
                    for section in sections
                ]
                received.zero_()  # Outputs must own storage, including KTP1.
                for value, want in zip(actual, expected):
                    self.assertNotEqual(
                        value.untyped_storage().data_ptr(),
                        received.untyped_storage().data_ptr(),
                    )
                    assert_exact(value, want)

    def test_pack_unpack_graph_replay_updates_stable_outputs(self):
        ranks, batch, heads, width = 8, 3, 12, 1536
        projected, gate = bf16(ranks * batch, 4 * width + 224), bf16(
            ranks * batch, width
        )
        packed = torch.empty(
            (ranks * batch, 5 * width + heads), device="cuda", dtype=torch.bfloat16
        )

        def run():
            small_kernels.pack_ktp_projection_payload_cuda(
                projected,
                gate,
                packed,
                local_projection_size=width,
                forget_latent_size=128,
                local_heads=heads,
                ktp_rank=0,
            )
            return small_kernels.reassemble_ktp_projection_payload_cuda(
                packed,
                ktp_size=ranks,
                physical_batch=batch,
                local_projection_size=width,
                local_heads=heads,
            )

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            unpacked = run()
        pointers = tuple(value.data_ptr() for value in unpacked)
        projected.fill_(7)
        gate.fill_(11)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(tuple(value.data_ptr() for value in unpacked), pointers)
        for index, value in enumerate(unpacked):
            self.assertTrue(torch.all(value == (11 if index == 4 else 7)))


if __name__ == "__main__":
    unittest.main()
