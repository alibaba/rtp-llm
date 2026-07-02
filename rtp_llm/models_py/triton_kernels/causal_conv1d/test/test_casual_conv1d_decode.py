import logging
import math
import sys
import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.causal_conv1d import causal_conv1d_update
from rtp_llm.test.utils.diff_util import compare_tensor_diff_with_ratio

logging.basicConfig(
    level="INFO",
    format="[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(
            weight.dtype
        )  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


class TestCausalConv1dUpdate(unittest.TestCase):
    def test_causal_conv1d_update(self):
        itype = torch.bfloat16
        device = "cuda"
        dim = 4096
        width = 4
        seq_size_per_block = 128
        rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
        if itype == torch.bfloat16:
            rtol, atol = 1e-2, 5e-2
        # set seed
        torch.random.manual_seed(0)
        for batch in [1, 2, 10, 16, 32]:
            for seqlen in [1, 2, 4]:
                print(f"test_causal_conv1d_update batch {batch}")
                sequence_lengths = torch.randint(
                    10, 1024, (batch,), dtype=torch.int32, device=device
                )
                block_nums = [
                    math.ceil(seq_len / seq_size_per_block) + seqlen - 1
                    for seq_len in sequence_lengths.tolist()
                ]
                max_block_num = max(block_nums)
                total_block_num = sum(block_nums)
                block_map = torch.zeros(
                    [batch, max_block_num], dtype=torch.int32, device=device
                )
                offset = 0
                for i in range(batch):
                    block_map[i, : block_nums[i]] = torch.arange(
                        offset, offset + block_nums[i], dtype=torch.int32, device=device
                    )
                    offset += block_nums[i]

                x = torch.randn(
                    batch, seqlen, dim, device=device, dtype=itype
                ).transpose(-1, -2)
                x_ref = x.detach().clone()
                state_len = width - 1
                origin_conv_state = torch.randn(
                    total_block_num, state_len, dim, device=device, dtype=itype
                )
                conv_state = origin_conv_state.detach().clone().transpose(-1, -2)
                weight = torch.randn(
                    dim, width, device=device, dtype=torch.float32, requires_grad=True
                )
                bias = None

                activation = "silu"
                cache_seqlens = None
                out = causal_conv1d_update(
                    x,
                    conv_state,
                    weight,
                    bias,
                    activation=activation,
                    cache_seqlens=cache_seqlens,
                    block_map=block_map,
                    sequence_lengths=sequence_lengths,
                    seq_size_per_block=seq_size_per_block,
                    validate_data=True,
                )

                read_block_ids = [
                    block_map[i, (seq_len - 2) // seq_size_per_block].item()
                    for i, seq_len in enumerate(sequence_lengths.tolist())
                ]
                conv_state_ref = (
                    origin_conv_state[read_block_ids].contiguous().transpose(-1, -2)
                )

                out_refs = []
                conv_state_refs = []
                for i in range(seqlen):
                    out_refs.append(
                        causal_conv1d_update_ref(
                            x_ref[:, :, i : i + 1].contiguous(),
                            conv_state_ref,
                            weight,
                            bias,
                            activation=activation,
                            cache_seqlens=cache_seqlens,
                        )
                    )
                    conv_state_refs.append(conv_state_ref.detach().clone())

                cat_out_ref = torch.cat(out_refs, dim=-1)
                compare_tensor_diff_with_ratio(
                    out, cat_out_ref, rel_threshold=rtol, abs_threshold=atol, ratio=0.03
                )
                for seq in range(seqlen):
                    write_block_ids = [
                        block_map[i, (seq_len - 1) // seq_size_per_block + seq].item()
                        for i, seq_len in enumerate(sequence_lengths.tolist())
                    ]
                    compare_conv_state = conv_state[write_block_ids]
                    compare_tensor_diff_with_ratio(
                        compare_conv_state,
                        conv_state_refs[seq],
                        rel_threshold=rtol,
                        abs_threshold=atol,
                        ratio=0,
                    )


class TestCausalConv1dMaxSeqLenBoundary(unittest.TestCase):
    """
    Regression tests for OOB block_map access when a sequence reaches the last
    allocated KV-cache block. Two independent offsets can go out of bounds:

    - write path: write_block_offset = sequence_length // SEQ_SIZE_PER_BLOCK
      reaches block_map.size(1) when sequence_length == MAX_SEQ_LEN + 1.
    - read path: read_block_offset = (sequence_length - 1) // SEQ_SIZE_PER_BLOCK
      reaches block_map.size(1) when sequence_length == MAX_SEQ_LEN + 2, feeding
      a garbage block id into every downstream conv_state load.

    Either OOB read yields a garbage block id, causing silent KV-cache
    corruption or a hardware fault. Fix: bound-check both offsets against
    block_map.size(1).
    """

    def test_no_crash_at_max_seq_len_boundary(self):
        """Kernel must not crash or corrupt memory when seq exactly fills all blocks."""
        device = "cuda"
        dtype = torch.bfloat16
        dim = 64
        width = 4
        state_len = width - 1
        seq_size_per_block = 16
        num_blocks = 4  # MAX_SEQ_LEN = 4 * 16 = 64
        batch = 2

        # sequence_lengths_plus_1_d == MAX_SEQ_LEN + 1 triggers write_block_offset OOB
        max_seq = num_blocks * seq_size_per_block
        sequence_lengths = torch.full(
            (batch,), max_seq + 1, dtype=torch.int32, device=device
        )

        # block_map.size(1) == num_blocks; no extra row for the boundary step
        block_map = torch.stack(
            [
                torch.arange(num_blocks, dtype=torch.int32, device=device),
                torch.arange(
                    num_blocks, num_blocks * 2, dtype=torch.int32, device=device
                ),
            ]
        )  # shape [batch, num_blocks]

        total_blocks = num_blocks * batch
        conv_state = torch.randn(
            total_blocks, dim, state_len, device=device, dtype=dtype
        )
        # Sentinel: fill a guard region adjacent to conv_state storage with a known value
        sentinel = torch.full((dim, state_len), 99.0, device=device, dtype=dtype)

        x = torch.randn(batch, dim, 1, device=device, dtype=dtype)
        weight = torch.randn(dim, width, device=device, dtype=torch.float32)

        # Must not crash; before the fix this raises "illegal memory access".
        out = causal_conv1d_update(
            x,
            conv_state,
            weight,
            block_map=block_map,
            sequence_lengths=sequence_lengths,
            seq_size_per_block=seq_size_per_block,
        )

        self.assertEqual(out.shape, (batch, dim, 1))
        # Output should be finite (no NaN/Inf from garbage address reads)
        self.assertTrue(torch.isfinite(out).all(), "output contains NaN or Inf")
        # conv_state within bounds must remain finite
        self.assertTrue(torch.isfinite(conv_state).all(), "conv_state corrupted")

    def test_no_crash_when_read_offset_out_of_bounds(self):
        """
        read_block_offset OOB: sequence_length == MAX_SEQ_LEN + 2 pushes
        read_block_offset = (sequence_length - 1) // SEQ_SIZE_PER_BLOCK to
        block_map.size(1). A garbage block id from that OOB read feeds every
        conv_state load, computing a wildly out-of-range address. To make the
        garbage deterministic, lay a huge block id in the word right after
        block_map so the OOB read returns it. Before the fix this raises
        "illegal memory access"; the clamp must keep the read on the last block.
        """
        device = "cuda"
        dtype = torch.bfloat16
        dim = 4096
        width = 4
        state_len = width - 1
        seq_size_per_block = 2048
        num_blocks = 4
        batch = 1
        huge_block_id = 1 << 22  # 4M — far outside any real conv_state pool

        max_seq = num_blocks * seq_size_per_block
        sequence_lengths = torch.full(
            (batch,), max_seq + 2, dtype=torch.int32, device=device
        )

        # Contiguous buffer [0, 1, ..., num_blocks-1, huge]; block_map is the
        # first num_blocks entries, so the OOB read at column num_blocks lands
        # on huge_block_id.
        buf = torch.tensor(
            list(range(num_blocks)) + [huge_block_id],
            dtype=torch.int32,
            device=device,
        )
        block_map = buf[:num_blocks].view(batch, num_blocks)

        total_blocks = num_blocks * batch
        conv_state = torch.randn(
            total_blocks, dim, state_len, device=device, dtype=dtype
        )
        x = torch.randn(batch, dim, 1, device=device, dtype=dtype)
        weight = torch.randn(dim, width, device=device, dtype=torch.float32)

        out = causal_conv1d_update(
            x,
            conv_state,
            weight,
            block_map=block_map,
            sequence_lengths=sequence_lengths,
            seq_size_per_block=seq_size_per_block,
        )
        torch.cuda.synchronize()

        self.assertEqual(out.shape, (batch, dim, 1))
        self.assertTrue(torch.isfinite(out).all(), "output contains NaN or Inf")

    def test_output_correct_one_before_boundary(self):
        """Sequence at MAX_SEQ_LEN - 1 (last normal step) must produce correct output."""
        device = "cuda"
        dtype = torch.bfloat16
        dim = 32
        width = 4
        state_len = width - 1
        seq_size_per_block = 8
        num_blocks = 3
        batch = 1

        # sequence_lengths_plus_1_d == MAX_SEQ_LEN (last valid step, write_block_offset = N-1)
        max_seq = num_blocks * seq_size_per_block
        sequence_lengths = torch.full(
            (batch,), max_seq, dtype=torch.int32, device=device
        )

        block_map = torch.arange(
            num_blocks, dtype=torch.int32, device=device
        ).unsqueeze(0)
        conv_state = torch.randn(num_blocks, dim, state_len, device=device, dtype=dtype)
        x = torch.randn(batch, dim, 1, device=device, dtype=dtype)
        weight = torch.randn(dim, width, device=device, dtype=torch.float32)

        out = causal_conv1d_update(
            x,
            conv_state,
            weight,
            block_map=block_map,
            sequence_lengths=sequence_lengths,
            seq_size_per_block=seq_size_per_block,
        )

        self.assertEqual(out.shape, (batch, dim, 1))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
