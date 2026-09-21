"""GPU regression for int64 addressing, including actual >INT32_MAX accesses."""

import math
import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.activation import _silu_and_mul_kernel
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
    _layer_norm_fwd_1pass_kernel,
)
from rtp_llm.models_py.triton_kernels.fla.block import (
    store_ssm_state_to_block_map_kernel,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.chunk import chunk_kda


class AddressOverflowGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("This target requires a CUDA GPU")
        print(
            "GPU:", torch.cuda.get_device_name(), "Triton/PyTorch:", torch.__version__
        )
        torch.backends.cuda.matmul.allow_tf32 = False

    def test_gated_norm_large_input_stride(self):
        stride, width = 1 << 30, 128
        # Only three rows are touched. The third starts at element 2^31.
        storage = torch.empty(2 * stride + width, device="cuda", dtype=torch.bfloat16)
        x = storage.as_strided((3, width), (stride, 1))
        values = (
            torch.linspace(-1, 1, 3 * width, device="cuda").reshape(3, width).bfloat16()
        )
        x.copy_(values)
        y, z = torch.empty_like(values), torch.zeros_like(values)
        w = torch.ones(width, device="cuda", dtype=torch.bfloat16)
        rstd = torch.empty(3, device="cuda")
        _layer_norm_fwd_1pass_kernel[(3, 1)](
            x,
            y,
            w,
            None,
            z,
            None,
            rstd,
            stride,
            width,
            width,
            3,
            width,
            1e-5,
            BLOCK_N=128,
            NORM_BEFORE_GATE=True,
            IS_RMS_NORM=True,
            SIGMOID_GATE=True,
        )
        ref = (
            values.float()
            * torch.rsqrt(values.float().square().mean(-1, keepdim=True) + 1e-5)
            * 0.5
        )
        torch.testing.assert_close(y.float(), ref, rtol=0.005, atol=0.004)

    def test_silu_large_input_stride(self):
        stride, width = 1 << 30, 128
        storage = torch.empty(
            2 * stride + 2 * width, device="cuda", dtype=torch.bfloat16
        )
        x = storage.as_strided((3, 2 * width), (stride, 1))
        x[:, :width].fill_(2)
        x[:, width:].fill_(1)
        output = torch.empty((3, width), device="cuda", dtype=torch.bfloat16)
        _silu_and_mul_kernel[(3, 1)](output, x, width, stride, width, BLOCK_SIZE_N=128)
        torch.testing.assert_close(
            output.float(),
            torch.full_like(output.float(), 2 / (1 + math.exp(-1.0))),
            rtol=0.005,
            atol=0.004,
        )

    def test_checkpoint_source_beyond_int32(self):
        heads, dim, source_index = 16, 128, 8192
        state_size = heads * dim * dim
        # 8 GiB of virtual tensor storage; initialize only the source tile read.
        h = torch.empty(source_index * state_size + dim * dim, device="cuda")
        expected = torch.arange(64 * dim, device="cuda", dtype=torch.float32) / (
            64 * dim
        )
        h[
            source_index * state_size : source_index * state_size + expected.numel()
        ].copy_(expected)
        chunk_indices = torch.zeros((source_index, 2), dtype=torch.int32, device="cuda")
        chunk_indices[-1, 1] = source_index - 1
        cu = torch.tensor(
            [0, (source_index + 1) * 64], dtype=torch.int32, device="cuda"
        )
        prefix = torch.zeros(1, dtype=torch.int32, device="cuda")
        table = torch.zeros(
            (1, source_index // 4 + 1), dtype=torch.int32, device="cuda"
        )
        table[0, source_index // 4 - 1] = 1
        cache_stride = state_size + 64
        cache = torch.full((2 * cache_stride,), -1.0, device="cuda")
        final = torch.zeros((heads, dim, dim), device="cuda")
        store_ssm_state_to_block_map_kernel[(source_index, 1, 1)](
            chunk_indices,
            h,
            final,
            prefix,
            cu,
            table,
            cache,
            table.shape[1],
            HEAD_NUM=heads,
            V=dim,
            K=dim,
            BLOCK_V=64,
            SEQ_SIZE_PER_BLOCK=256,
            CHUNK_SIZE=64,
            CONV_STRIDE_TOKEN=cache_stride,
        )
        torch.testing.assert_close(
            cache[cache_stride : cache_stride + expected.numel()],
            expected,
            rtol=0,
            atol=0,
        )
        self.assertTrue(torch.all(cache[:cache_stride] == -1).item())

    def test_kda_ragged_chunk_sizes_match_recurrence(self):
        torch.manual_seed(20260921)
        heads, dim = 2, 128
        lengths = (67, 129)
        tokens = sum(lengths)
        q, k, v = [
            torch.randn((1, tokens, heads, dim), device="cuda", dtype=torch.bfloat16)
            * 0.1
            for _ in range(3)
        ]
        q = (
            q.float() / torch.sqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
        ).bfloat16()
        k = (
            k.float() / torch.sqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
        ).bfloat16()
        # Exercise the fused gate path, including nonzero bos for request 2.
        raw_g = torch.randn_like(q) * 0.1 - 4
        a = torch.zeros(heads, device="cuda")
        bias = torch.zeros(heads * dim, device="cuda")
        beta = torch.rand((1, tokens, heads), device="cuda") * 0.5
        gate = -5 * torch.sigmoid(raw_g.float())
        cu = torch.tensor([0, lengths[0], tokens], dtype=torch.int32, device="cuda")
        expected, final = [], []
        start = 0
        for length in lengths:
            state = torch.zeros((heads, dim, dim), device="cuda")
            for t in range(start, start + length):
                state *= torch.exp(gate[0, t]).unsqueeze(-1)
                key = k[0, t].float().unsqueeze(-1)
                correction = (state * key).sum(-2)
                delta = beta[0, t, :, None] * (v[0, t].float() - correction)
                state += key * delta.unsqueeze(-2)
                expected.append(
                    torch.einsum("hk,hkv->hv", q[0, t].float(), state) * dim**-0.5
                )
            final.append(state.clone())
            start += length
        expected = torch.stack(expected).unsqueeze(0)
        final = torch.stack(final)
        for chunk, fused_state in ((64, False), (64, True), (128, False), (256, False)):
            with self.subTest(chunk=chunk, fused_state=fused_state):
                out, actual_final = chunk_kda(
                    q,
                    k,
                    v,
                    raw_g,
                    beta,
                    output_final_state=True,
                    cu_seqlens=cu,
                    use_gate_in_kernel=True,
                    A_log=a,
                    dt_bias=bias,
                    safe_gate=True,
                    lower_bound=-5.0,
                    chunk_size=chunk,
                    fuse_state_recurrence=fused_state,
                )
                torch.testing.assert_close(out.float(), expected, rtol=0.03, atol=5e-4)
                torch.testing.assert_close(
                    actual_final.float(), final, rtol=0.03, atol=1e-3
                )


if __name__ == "__main__":
    unittest.main()
