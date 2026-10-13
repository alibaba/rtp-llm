"""Gated causal conv1d kernel vs the decomposed torch path (LFM2 semantics).

Oracle for every case is the production decomposed path the kernel replaces:
``projected.chunk(3) -> B * X (torch bf16 mul) -> causal_conv1d_fn /
causal_conv1d_update -> C * conv_out (torch bf16 mul)`` using the shared
(ungated) causal_conv1d kernels. Outputs must match within bf16 tolerance
(measured: bit-exact) and conv states must match bitwise, since the fused
kernel applies the same bf16 roundings in-register.
"""

import logging
import math
import unittest

import torch

from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.gated_causal_conv1d import (
    gated_causal_conv1d_fn,
    gated_causal_conv1d_update,
)

logging.basicConfig(
    level="INFO",
    format="[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DTYPE = torch.bfloat16
K = 3


def gated_bx(projected):
    b, _, x = projected.chunk(3, dim=-1)
    return b * x


def make_block_map(batch, block_nums, device):
    max_blocks = max(block_nums)
    # match CUDA graph replay: a narrow view over a wider persistent allocation
    storage = torch.full((batch, max_blocks + 1), -1, dtype=torch.int32, device=device)
    block_map = storage[:, :max_blocks]
    offset = 0
    for i in range(batch):
        block_map[i, : block_nums[i]] = torch.arange(
            offset, offset + block_nums[i], dtype=torch.int32, device=device
        )
        offset += block_nums[i]
    return block_map


def assert_state_bitwise(test, state_fused, state_ref, name):
    test.assertEqual(state_fused.dtype, state_ref.dtype)
    if state_fused.dtype == torch.bfloat16:
        test.assertTrue(
            torch.equal(state_fused.view(torch.int16), state_ref.view(torch.int16)),
            f"{name}: conv_state differs from the decomposed path",
        )
    else:
        test.assertTrue(torch.equal(state_fused, state_ref), f"{name}: conv_state differs")


class TestGatedCausalConv1d(unittest.TestCase):
    """Prefill (varlen / prefix continuation / non-divisible sizes)."""

    def _run_prefill(self, batch, seqlens, dim, spb, with_prefix, use_metadata, seed):
        device = "cuda"
        torch.manual_seed(seed)
        origin_len = [s + (spb * 2 if with_prefix else 0) for s in seqlens]
        projected_full = torch.randn(
            sum(origin_len), 3 * dim, device=device, dtype=DTYPE
        )
        if with_prefix:
            prefix_lengths = torch.tensor(
                [len_ - s for len_, s in zip(origin_len, seqlens)],
                device=device,
                dtype=torch.int32,
            )
            # block aligned prefix (kernel read/write contract)
            prefix_lengths = torch.div(prefix_lengths, spb, rounding_mode="floor") * spb
            seqlens = [o - p for o, p in zip(origin_len, prefix_lengths.tolist())]
        else:
            prefix_lengths = torch.zeros(batch, device=device, dtype=torch.int32)
        query_start_loc = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seqlens, dtype=torch.int64), 0).tolist()),
            device=device,
            dtype=torch.int32,
        )
        proj_list, offsets = [], 0
        for i in range(batch):
            p = prefix_lengths[i].item()
            proj_list.append(projected_full[offsets + p : offsets + origin_len[i]])
            offsets += origin_len[i]
        projected = torch.cat(proj_list, dim=0)
        weight = torch.randn(dim, K, device=device, dtype=DTYPE)
        block_num = [math.ceil(length / spb) for length in origin_len]
        conv_seed = torch.randn(
            sum(block_num), K - 1, dim, device=device, dtype=DTYPE
        )
        if with_prefix:
            # seed prefix blocks with the gated B*X of the earlier tokens
            bx_full = gated_bx(projected_full)
            offsets, block_offsets = 0, 0
            for i in range(batch):
                reuse = prefix_lengths[i].item() // spb
                for j in range(reuse):
                    end = offsets + (j + 1) * spb
                    conv_seed[block_offsets + j] = bx_full[end - (K - 1) : end]
                block_offsets += block_num[i]
                offsets += origin_len[i]
        block_map = make_block_map(batch, block_num, device)
        conv_ref = conv_seed.clone().transpose(1, 2)  # [blocks, dim, k - 1]
        conv_fused = conv_seed.clone().transpose(1, 2)
        metadata = (
            prepare_causal_conv1d_metadata(
                query_start_loc=query_start_loc, device=device
            )
            if use_metadata
            else None
        )

        # decomposed reference (the production path this kernel replaces)
        b, c, x = projected.chunk(3, dim=-1)
        bx = b * x
        out_ref = causal_conv1d_fn(
            bx.transpose(0, 1),
            weight,
            None,
            conv_ref,
            query_start_loc,
            block_map,
            prefix_lengths,
            spb,
            activation=None,
            metadata=metadata,
        )
        y_ref = c * out_ref.transpose(0, 1)

        # fused kernel
        y_fused = gated_causal_conv1d_fn(
            projected,
            weight,
            conv_fused,
            query_start_loc,
            block_map,
            prefix_lengths,
            spb,
            activation=None,
            metadata=metadata,
        )

        torch.testing.assert_close(y_fused, y_ref, rtol=1e-2, atol=2e-2)
        assert_state_bitwise(self, conv_fused, conv_ref, "prefill")

        # state semantics: every block holds the last K-1 gated inputs of its
        # token window in sequence coordinates (prefix blocks keep the seeded
        # values, which follow the same rule), checked against the raw input
        bx_full = gated_bx(projected_full)
        state_offset = 0
        for i in range(batch):
            origin = origin_len[i]
            origin_start = sum(origin_len[:i])
            for blk in range(block_num[i]):
                end = min((blk + 1) * spb, origin)
                start = end - (K - 1)
                if start < 0:
                    continue
                expected = bx_full[origin_start + start : origin_start + end]
                torch.testing.assert_close(
                    conv_fused[state_offset + blk].transpose(0, 1),
                    expected,
                    rtol=0,
                    atol=0,
                )
            state_offset += block_num[i]

    def test_prefill_basic(self):
        self._run_prefill(3, [37, 129, 512], 2048, 16, False, False, 1234)

    def test_prefill_cache_and_metadata(self):
        self._run_prefill(4, [100, 205, 17, 4096], 2048, 16, False, True, 42)

    def test_prefill_prefix_continuation(self):
        self._run_prefill(2, [33, 77], 2048, 16, True, True, 7)

    def test_prefill_non_divisible_sizes(self):
        # dim not divisible by BLOCK_N=256, seqlens not divisible by BLOCK_M=8
        self._run_prefill(3, [13, 37, 101], 2000, 5, False, False, 99)

    def _run_decode(self, batch, seq, dim, spb, seed, two_d=False):
        device = "cuda"
        torch.manual_seed(seed)
        tokens = batch * seq
        projected = torch.randn(tokens, 3 * dim, device=device, dtype=DTYPE)
        weight = torch.randn(dim, K, device=device, dtype=DTYPE)
        sequence_lengths = torch.randint(
            10, 1024, (batch,), dtype=torch.int32, device=device
        )
        # the update kernel writes state blocks up to (S-1)//spb + seq - 1
        block_nums = [
            math.ceil(s.item() / spb) + seq - 1 for s in sequence_lengths
        ]
        block_map = make_block_map(batch, block_nums, device)
        state_seed = torch.randn(
            sum(block_nums), K - 1, dim, device=device, dtype=DTYPE
        )
        conv_ref = state_seed.clone().transpose(-1, -2)
        conv_fused = state_seed.clone().transpose(-1, -2)

        b, c, x = projected.chunk(3, dim=-1)
        bx = b * x
        if two_d:
            assert seq == 1
            out_ref = causal_conv1d_update(
                bx,
                conv_ref,
                weight,
                bias=None,
                activation=None,
                cache_seqlens=None,
                block_map=block_map,
                seq_size_per_block=spb,
                sequence_lengths=sequence_lengths,
            )
            y_ref = c * out_ref
            y_fused = gated_causal_conv1d_update(
                projected,
                conv_fused,
                weight,
                bias=None,
                activation=None,
                cache_seqlens=None,
                block_map=block_map,
                seq_size_per_block=spb,
                sequence_lengths=sequence_lengths,
            )
            self.assertEqual(y_fused.shape, (tokens, dim))
        else:
            bx3 = bx.reshape(batch, seq, -1).transpose(1, 2)
            out_ref = causal_conv1d_update(
                bx3,
                conv_ref,
                weight,
                bias=None,
                activation=None,
                cache_seqlens=None,
                block_map=block_map,
                seq_size_per_block=spb,
                sequence_lengths=sequence_lengths,
            )
            y_ref = c * out_ref.transpose(1, 2).reshape(tokens, -1)
            proj3 = projected.reshape(batch, seq, -1).transpose(1, 2)
            out_fused = gated_causal_conv1d_update(
                proj3,
                conv_fused,
                weight,
                bias=None,
                activation=None,
                cache_seqlens=None,
                block_map=block_map,
                seq_size_per_block=spb,
                sequence_lengths=sequence_lengths,
            )
            y_fused = out_fused.transpose(1, 2).reshape(tokens, -1)

        torch.testing.assert_close(y_fused, y_ref, rtol=1e-2, atol=2e-2)
        assert_state_bitwise(self, conv_fused, conv_ref, "decode")

    def test_decode_single_token(self):
        self._run_decode(8, 1, 2048, 128, 5)

    def test_decode_single_token_2d(self):
        self._run_decode(8, 1, 2048, 128, 6, two_d=True)

    def test_decode_multi_token(self):
        # target-verify style: several tokens per sequence in one step
        self._run_decode(4, 4, 2048, 128, 8)

    def test_decode_non_divisible_sizes(self):
        self._run_decode(3, 2, 2000, 5, 11)

    def test_chained_prefill_then_decode(self):
        """End-to-end state continuity: prefill, then decode continues."""
        device = "cuda"
        torch.manual_seed(21)
        dim, spb = 2048, 16
        batch, pre_len, dec_tokens = 2, 61, 3
        full_len = pre_len + dec_tokens
        projected = torch.randn(
            batch * full_len, 3 * dim, device=device, dtype=DTYPE
        )
        weight = torch.randn(dim, K, device=device, dtype=DTYPE)
        block_num = [math.ceil(pre_len / spb) + dec_tokens - 1 for _ in range(batch)]
        block_map = make_block_map(batch, block_num, device)
        state_seed = torch.randn(
            sum(block_num), K - 1, dim, device=device, dtype=DTYPE
        )
        conv_fused = state_seed.clone().transpose(1, 2)
        conv_ref = state_seed.clone().transpose(1, 2)

        # token layout: [L0 | D0 | L1 | D1]
        pre_proj = torch.cat(
            [projected[0:pre_len], projected[full_len : full_len + pre_len]], dim=0
        )
        dec_proj = torch.cat(
            [projected[pre_len:full_len], projected[full_len + pre_len :]], dim=0
        )
        pre_qsl = torch.tensor(
            [0, pre_len, 2 * pre_len], device=device, dtype=torch.int32
        )
        prefix0 = torch.zeros(batch, device=device, dtype=torch.int32)
        seq_lens = torch.tensor(
            [pre_len + 1, pre_len + 1], device=device, dtype=torch.int32
        )

        y_pre = gated_causal_conv1d_fn(
            pre_proj, weight, conv_fused, pre_qsl, block_map, prefix0, spb
        )
        dec_proj3 = dec_proj.reshape(batch, dec_tokens, 3 * dim).transpose(1, 2)
        out_dec = gated_causal_conv1d_update(
            dec_proj3,
            conv_fused,
            weight,
            bias=None,
            activation=None,
            cache_seqlens=None,
            block_map=block_map,
            seq_size_per_block=spb,
            sequence_lengths=seq_lens,
        )
        y_dec = out_dec.transpose(1, 2).reshape(batch * dec_tokens, dim)
        y_seq0 = torch.cat([y_pre[0:pre_len], y_dec[0:dec_tokens]], dim=0)
        y_seq1 = torch.cat([y_pre[pre_len:], y_dec[dec_tokens:]], dim=0)
        y_fused = torch.cat([y_seq0, y_seq1], dim=0)

        b, c, x = pre_proj.chunk(3, dim=-1)
        out_pre_ref = causal_conv1d_fn(
            (b * x).transpose(0, 1),
            weight,
            None,
            conv_ref,
            pre_qsl,
            block_map,
            prefix0,
            spb,
            activation=None,
        )
        y_pre_ref = c * out_pre_ref.transpose(0, 1)
        bd, cd, xd = dec_proj.chunk(3, dim=-1)
        out_dec_ref = causal_conv1d_update(
            (bd * xd).reshape(batch, dec_tokens, -1).transpose(1, 2),
            conv_ref,
            weight,
            bias=None,
            activation=None,
            cache_seqlens=None,
            block_map=block_map,
            seq_size_per_block=spb,
            sequence_lengths=seq_lens,
        )
        y_dec_ref = cd * out_dec_ref.transpose(1, 2).reshape(batch * dec_tokens, dim)
        y_seq0_ref = torch.cat(
            [y_pre_ref[0:pre_len], y_dec_ref[0:dec_tokens]], dim=0
        )
        y_seq1_ref = torch.cat(
            [y_pre_ref[pre_len:], y_dec_ref[dec_tokens:]], dim=0
        )
        y_ref = torch.cat([y_seq0_ref, y_seq1_ref], dim=0)

        torch.testing.assert_close(y_fused, y_ref, rtol=1e-2, atol=2e-2)
        assert_state_bitwise(self, conv_fused, conv_ref, "chained")


if __name__ == "__main__":
    unittest.main()
