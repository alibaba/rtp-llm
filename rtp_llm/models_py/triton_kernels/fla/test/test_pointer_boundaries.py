"""PyTorch references for masked loads/stores after block-pointer migration."""

import itertools
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.triton_kernels.fla import chunk_scaled_dot_kkt
from rtp_llm.models_py.triton_kernels.fla.cumsum import chunk_local_cumsum
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd
from rtp_llm.models_py.triton_kernels.fla.solve_tril import solve_tril


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class PointerBoundaryTest(unittest.TestCase):
    def test_kkt_partial_chunk_both_gate_domains(self):
        torch.manual_seed(20)
        k = torch.randn(1, 97, 2, 35, device="cuda", dtype=torch.bfloat16)
        beta = torch.rand(1, 97, 4, device="cuda")
        g = -torch.rand(1, 97, 4, device="cuda").cumsum(1) * 0.01
        expected = torch.zeros(1, 97, 4, 64, device="cuda")
        for pos in (0, 64):
            n = min(64, 97 - pos)
            key = (
                k[0, pos : pos + n].float().repeat_interleave(2, dim=1).transpose(0, 1)
            )
            gate = g[0, pos : pos + n].transpose(0, 1)
            value = (key @ key.transpose(-1, -2)) * (
                gate[:, :, None] - gate[:, None, :]
            ).exp()
            value *= beta[0, pos : pos + n].transpose(0, 1)[:, :, None]
            expected[0, pos : pos + n, :, :n] = value.tril(-1).transpose(0, 1)
        for log2 in (False, True):
            with self.subTest(log2=log2), mock.patch.object(
                chunk_scaled_dot_kkt, "is_amd", log2
            ):
                gates = g * 1.4426950408889634 if log2 else g
                actual = chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd(k, beta, gates)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_cumsum_partial_chunks_and_strided_heads(self):
        for vector, head_first, reverse, varlen, dtype in itertools.product(
            (False, True),
            (False, True),
            (False, True),
            (False, True),
            (torch.bfloat16, torch.float32),
        ):
            with self.subTest(
                vector=vector,
                head_first=head_first,
                reverse=reverse,
                varlen=varlen,
                dtype=dtype,
            ):
                torch.manual_seed(17)
                shape = (1 if varlen else 2, 97, 2) + ((35,) if vector else ())
                x = torch.randn(shape, device="cuda", dtype=dtype)
                segments = (0, 15, 79, 97) if varlen else (0, 97)
                expected = torch.empty_like(x, dtype=torch.float32)
                for start, end in itertools.pairwise(segments):
                    for pos in range(start, end, 64):
                        stop = min(pos + 64, end)
                        part = x[:, pos:stop].float()
                        if reverse:
                            part = part.flip(1).cumsum(1).flip(1)
                        else:
                            part = part.cumsum(1)
                        expected[:, pos:stop] = part * 0.5
                cu = (
                    torch.tensor(segments, device="cuda", dtype=torch.int32)
                    if varlen
                    else None
                )
                inputs = x.transpose(1, 2).contiguous() if head_first else x
                if head_first and varlen:
                    # This legacy API packs each sequence as [H, T_seq, ...],
                    # rather than placing all sequences in one [H, total_T] view.
                    inputs = torch.cat(
                        [
                            x[:, start:end].transpose(1, 2).contiguous().flatten()
                            for start, end in itertools.pairwise(segments)
                        ]
                    ).view_as(inputs)
                actual = chunk_local_cumsum(
                    inputs,
                    64,
                    reverse=reverse,
                    scale=0.5,
                    cu_seqlens=cu,
                    head_first=head_first,
                )
                if head_first and varlen:
                    parts = []
                    offset = 0
                    for start, end in itertools.pairwise(segments):
                        size = x[:, start:end].numel()
                        packed_shape = (1, 2, end - start) + ((35,) if vector else ())
                        parts.append(
                            actual.flatten()[offset : offset + size]
                            .view(packed_shape)
                            .transpose(1, 2)
                        )
                        offset += size
                    actual = torch.cat(parts, dim=1)
                elif head_first:
                    actual = actual.transpose(1, 2)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_l2norm_partial_tile(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                torch.manual_seed(18)
                x = torch.randn(13, 35, device="cuda", dtype=dtype)
                expected = x.float() * torch.rsqrt(
                    x.float().square().sum(-1, keepdim=True) + 1e-6
                )
                torch.testing.assert_close(l2norm_fwd(x), expected.to(dtype))

    def test_triangular_solve_partial_tile(self):
        for chunk in (16, 32, 64):
            with self.subTest(chunk=chunk):
                torch.manual_seed(19)
                x = torch.zeros(1, 97, 2, chunk, device="cuda")
                expected = torch.zeros_like(x)
                for pos in range(0, 97, chunk):
                    n = min(chunk, 97 - pos)
                    lower = torch.randn(2, n, n, device="cuda").tril(-1) * 0.02
                    identity = torch.eye(n, device="cuda").expand(2, n, n)
                    inverse = torch.linalg.solve_triangular(
                        identity + lower, identity, upper=False
                    )
                    x[0, pos : pos + n, :, :n] = lower.transpose(0, 1)
                    expected[0, pos : pos + n, :, :n] = inverse.transpose(0, 1)
                actual = solve_tril(x)
                # The existing 32x32 merge writes only the lower block triangle;
                # its upper-right tile is not part of the consumer contract.
                rows = torch.arange(97, device="cuda") % chunk
                columns = torch.arange(chunk, device="cuda")
                valid = (columns[None, :] <= rows[:, None])[None, :, None, :].expand_as(
                    x
                )
                torch.testing.assert_close(
                    actual[valid], expected[valid], rtol=1e-4, atol=1e-5
                )


if __name__ == "__main__":
    unittest.main()
