import os
import unittest

import torch

from rtp_llm.models_py.kernels.cuda import batch_invariant as ops

_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
_CUDA = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] >= 8
)


@unittest.skipUnless(_CUDA or _INTERPRET, "requires NVIDIA SM80+ or Triton interpreter")
class BatchInvariantTest(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if _CUDA else "cpu"
        torch.manual_seed(12)

    def metadata(self, values):
        return torch.tensor(values, device=self.device, dtype=torch.int32)

    def test_linear_with_padding_and_reordering(self):
        dtype = torch.float16
        x = torch.randn((35, 67), device=self.device, dtype=dtype)
        weight = torch.randn((67, 89), device=self.device, dtype=dtype).T
        bias = torch.randn(178, device=self.device, dtype=dtype)[::2]

        def linear(rows):
            if _CUDA:
                return ops.linear(rows, weight, bias, out_dtype=torch.float32)
            output = torch.empty((rows.size(0), weight.size(0)), dtype=torch.float32)
            ops._linear[((rows.size(0) + 15) // 16, 2)](
                rows,
                weight,
                bias,
                output,
                rows.size(0),
                89,
                67,
                weight.stride(0),
                weight.stride(1),
                bias.stride(0),
                True,
                16,
                64,
                32,
                num_warps=4,
            )
            return output

        baseline = linear(x[8:9])
        for batch in (x, x.flip(0), x[8:].contiguous()):
            row = (batch == x[8]).all(dim=1).nonzero().item()
            self.assertTrue(torch.equal(linear(batch)[row : row + 1], baseline))
        expected = x[8:9].float() @ weight.float().T + bias.float()
        torch.testing.assert_close(baseline, expected, atol=3e-5, rtol=3e-5)

    def test_rms_norm_reorders_rows(self):
        x = torch.randn((19, 127), device=self.device, dtype=torch.float16)
        gamma = torch.randn(127, device=self.device, dtype=x.dtype)
        baseline = ops.rms_norm(x[8:9], gamma, 1e-6)
        reordered = ops.rms_norm(x.flip(0), gamma, 1e-6)
        self.assertTrue(torch.equal(reordered[10:11], baseline))
        row = x[8:9].float()
        expected = row * (row.square().mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
        torch.testing.assert_close(baseline, (expected * gamma).half())

    def test_block_fp8_scales_follow_their_token_and_k_group(self):
        rows, cols, groups = 19, 133, 5
        inner = groups * 128
        x = torch.randint(-2, 3, (rows, inner), device=self.device).half()
        weight = torch.randint(-2, 3, (cols, inner), device=self.device).half()
        x_scales = 2.0 ** torch.randint(-5, 0, (rows, groups), device=self.device)
        weight_scales = 2.0 ** torch.randint(-5, 0, (2, groups), device=self.device)
        expanded_scales = weight_scales.repeat_interleave(128, dim=0)[:cols]
        expected = (x.float() * x_scales.repeat_interleave(128, dim=1)) @ (
            weight.float() * expanded_scales.repeat_interleave(128, dim=1)
        ).T

        def pack_scales(scales):
            exponents = scales.view(torch.int32) >> 23
            padded = torch.nn.functional.pad(exponents, (0, 3)).view(-1, 2, 4)
            shifts = torch.arange(0, 32, 8, dtype=torch.int32, device=self.device)
            return (padded << shifts).sum(dim=-1, dtype=torch.int32)

        if _CUDA and torch.cuda.get_device_capability() >= (8, 9):
            x, weight = x.to(torch.float8_e4m3fn), weight.to(torch.float8_e4m3fn)
        for packed in (False, True):
            with self.subTest(packed=packed):
                activations = pack_scales(x_scales) if packed else x_scales
                weights = pack_scales(expanded_scales) if packed else weight_scales
                baseline = ops._launch_linear(
                    x[8:9], weight, None, torch.float32, activations[8:9], weights
                )
                mixed = ops._launch_linear(
                    x, weight, None, torch.float32, activations, weights
                )
                torch.testing.assert_close(mixed, expected, atol=0, rtol=0)
                self.assertTrue(torch.equal(mixed[8:9], baseline))

    @unittest.skipUnless(_CUDA, "requires NVIDIA GPU")
    def test_fp8_scaling_is_local_to_each_token(self):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_per_tensor_linear import (
            CudaFp8PerTensorLinear,
        )

        x = torch.randn((19, 96), device=self.device, dtype=torch.bfloat16)
        x[0] *= 50
        weight = torch.randn((113, 96), device=self.device).to(torch.float8_e4m3fn)
        scale = torch.tensor(0.01, device=self.device)
        for static_scale in (None, scale):
            baseline = ops.fp8_per_tensor_linear(
                x[8:9], weight, scale, static_scale, out_dtype=torch.float32
            )
            mixed = ops.fp8_per_tensor_linear(
                x, weight, scale, static_scale, out_dtype=torch.float32
            )
            self.assertTrue(torch.equal(mixed[8:9], baseline))
            linear = CudaFp8PerTensorLinear(
                weight, scale, static_scale, batch_invariant=True
            )
            self.assertTrue(torch.equal(linear(x[8:9]), baseline.to(x.dtype)))

    @unittest.skipUnless(
        _CUDA and torch.cuda.get_device_capability() >= (8, 9), "requires NVIDIA SM89+"
    )
    def test_block_fp8_linear_quantizes_activations_independently(self):
        x = torch.randn((19, 640), device=self.device, dtype=torch.bfloat16)
        x[0] *= 50
        weight = torch.randn((133, 640), device=self.device).to(torch.float8_e4m3fn)
        for scales in (
            torch.full((2, 5), 0.125, device=self.device),
            torch.full((133, 2), 0x7C7C7C7C, device=self.device, dtype=torch.int32),
        ):
            baseline = ops.fp8_per_block_linear(x[8:9], weight, scales)
            mixed = ops.fp8_per_block_linear(x, weight, scales)
            self.assertTrue(torch.equal(mixed[8:9], baseline))

    def test_attention_across_prefill_chunk_decode_and_cache_pages(self):
        length, page_size, q_heads, kv_heads, head_dim = 39, 16, 4, 2, 32
        width = (q_heads + 2 * kv_heads) * head_dim
        target = torch.randn((length, width), device=self.device, dtype=torch.float16)
        other = torch.randn((5, width), device=self.device, dtype=torch.float16)
        angles = torch.arange(length, device=self.device)[
            :, None
        ].float() * torch.linspace(0.01, 0.07, head_dim // 2, device=self.device)
        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=1)
        scale = head_dim**-0.5

        def new_cache():
            return torch.zeros(
                (8, 2, kv_heads, page_size, head_dim),
                device=self.device,
                dtype=target.dtype,
            )

        def run(qkv, cache, pages, offsets, prefixes_or_lengths):
            offsets = self.metadata(offsets) if offsets is not None else None
            layout = ops.attention_token_layout(
                offsets,
                self.metadata(prefixes_or_lengths),
                pages,
                qkv.size(0),
                page_size,
            )
            q, k, v = (
                qkv.clone()
                .view(qkv.size(0), -1, head_dim)
                .split((q_heads, kv_heads, kv_heads), dim=1)
            )
            ops.apply_rope(q, k, layout, cos_sin, None)
            ops.append_kv(k, v, cache, layout)
            result = ops.attention(
                q, k, v, layout, cache, pages, causal=True, scale=scale
            )
            return result, (q, k, v)

        pages = self.metadata([[1, 2, 3]])
        baseline, (q, k, v) = run(target, new_cache(), pages, [0, length], [0])
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                q.float().transpose(0, 1),
                k.float().repeat_interleave(q_heads // kv_heads, dim=1).transpose(0, 1),
                v.float().repeat_interleave(q_heads // kv_heads, dim=1).transpose(0, 1),
                is_causal=True,
            )
            .transpose(0, 1)
            .reshape(length, -1)
        )
        torch.testing.assert_close(baseline.float(), expected, atol=2e-3, rtol=2e-3)

        mixed_pages = self.metadata([[4, 0, 0], [7, 5, 6]])
        mixed, _ = run(
            torch.cat((other, target)),
            new_cache(),
            mixed_pages,
            [0, 5, 5 + length],
            [0, 0],
        )
        self.assertTrue(torch.equal(mixed[5:], baseline))

        cache = new_cache()
        start, _ = run(target[:19], cache, pages, [0, 19], [0])
        suffix, _ = run(target[19:-1], cache, pages, [0, length - 20], [19])
        decoded, _ = run(target[-1:], cache, pages, None, [length])
        self.assertTrue(torch.equal(torch.cat((start, suffix, decoded)), baseline))

        # A rounded graph batch can have zero-length rows and page zero.
        padded, _ = run(
            torch.cat((target, other)),
            new_cache(),
            self.metadata([[0, 0, 0], [1, 2, 3], [0, 0, 0]]),
            [0, 0, length, length + 5],
            [0, 0, 0],
        )
        self.assertTrue(torch.equal(padded[:length], baseline))
        self.assertEqual(torch.count_nonzero(padded[length:]).item(), 0)

    @unittest.skipUnless(_CUDA, "requires NVIDIA GPU")
    def test_graph_replay_uses_updated_metadata(self):
        dtype = torch.bfloat16
        page_size = 16
        cache = torch.randn((4, 2, 1, page_size, 32), device=self.device, dtype=dtype)
        q = torch.randn((3, 2, 32), device=self.device, dtype=dtype)
        k = torch.empty((3, 1, 32), device=self.device, dtype=dtype)
        pages = self.metadata([[1, 2], [0, 0], [0, 0]])
        lengths = self.metadata([7, 1, 1])

        def run(query, key, blocks, sizes):
            layout = ops.attention_token_layout(
                None, sizes, blocks, query.size(0), page_size
            )
            return ops.attention(
                query, key, key, layout, cache, blocks, causal=True, scale=32**-0.5
            )

        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            run(q, k, pages, lengths)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(q, k, pages, lengths)

        pages.copy_(self.metadata([[3, 2], [0, 0], [0, 0]]))
        lengths.copy_(self.metadata([21, 1, 1]))
        graph.replay()
        eager = run(q[:1], k[:1], pages[:1], lengths[:1])
        self.assertTrue(torch.equal(output[:1], eager))
        self.assertEqual(torch.count_nonzero(output[1:]).item(), 0)


if __name__ == "__main__":
    unittest.main()
