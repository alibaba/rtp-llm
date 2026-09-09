import itertools
import logging
import math
import os
import random
import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.causal_conv1d import causal_conv1d_fn

logging.basicConfig(
    level="INFO",
    format="[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def causal_conv1d_ref(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """
    x: (dim, seq_length)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (dim, seq_length)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not return_final_states else (out, final_states_out)


class TestCausalConv1dPrefill(unittest.TestCase):
    def test_grouped_output_preserves_fp32_arithmetic_and_prefix_cache(self):
        torch.manual_seed(530909)
        lengths = [1, 3, 127, 129, 513]
        cu = torch.tensor(
            [0] + list(itertools.accumulate(lengths)), device="cuda", dtype=torch.int32
        )
        for channels in (128, 129, 1024):
            for width in (2, 3, 4):
                for prefix in (0, 128):
                    for activation in (None, "silu"):
                        with self.subTest(
                            channels=channels,
                            width=width,
                            prefix=prefix,
                            activation=activation,
                        ):
                            dim = 3 * channels
                            x = torch.randn(
                                sum(lengths), dim, device="cuda", dtype=torch.bfloat16
                            )
                            weight = torch.randn(dim, width, device="cuda")
                            bias = torch.randn(dim, device="cuda")
                            prefixes = torch.full(
                                (len(lengths),),
                                prefix,
                                device="cuda",
                                dtype=torch.int32,
                            )
                            counts = [math.ceil((prefix + n) / 128) for n in lengths]
                            slots = torch.randperm(
                                sum(counts), device="cuda", dtype=torch.int32
                            )
                            block_map = torch.full(
                                (len(lengths), max(counts)),
                                -1,
                                device="cuda",
                                dtype=torch.int32,
                            )
                            offset = 0
                            for row, count in enumerate(counts):
                                block_map[row, :count] = slots[offset : offset + count]
                                offset += count
                            state = torch.randn(
                                sum(counts),
                                width - 1,
                                dim,
                                device="cuda",
                                dtype=torch.bfloat16,
                            )
                            old_state, new_state = state.clone(), state.clone()
                            kwargs = dict(
                                x=x.T,
                                weight=weight,
                                bias=bias,
                                query_start_loc=cu,
                                block_map=block_map,
                                prefix_lengths=prefixes,
                                seq_size_per_block=128,
                                activation=activation,
                            )
                            old = causal_conv1d_fn(
                                conv_states=old_state.transpose(1, 2), **kwargs
                            ).T
                            new = causal_conv1d_fn(
                                conv_states=new_state.transpose(1, 2),
                                output_groups=3,
                                **kwargs,
                            )
                            for actual, expected in zip(
                                new.unbind(0), old.split(channels, -1)
                            ):
                                self.assertTrue(actual.is_contiguous())
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )
                            torch.testing.assert_close(
                                new_state, old_state, rtol=0, atol=0
                            )

    @unittest.skipUnless(
        os.environ.get("GLM53_TEST_LARGE_CONV") == "1", "12 GiB conv test"
    )
    def test_grouped_output_offsets_above_int32(self):
        # The V plane begins at element 2**31 for the real B8/128K TP8 shape.
        tokens, channels = 1048576, 1024
        x = torch.ones(tokens, 3 * channels, device="cuda", dtype=torch.bfloat16)
        x[:, channels : 2 * channels] = 2
        x[:, 2 * channels :] = 3
        weight = torch.zeros(3 * channels, 4, device="cuda")
        weight[:, -1] = 1
        output = causal_conv1d_fn(
            x.T,
            weight,
            None,
            None,
            torch.arange(9, device="cuda", dtype=torch.int32) * 131072,
            None,
            torch.zeros(8, device="cuda", dtype=torch.int32),
            128,
            activation=None,
            output_groups=3,
        )
        for group in range(3):
            self.assertTrue(bool((output[group] == group + 1).all()))

    # without KVCache
    def test_basic(self):
        device = "cuda"
        itype = torch.bfloat16
        if itype == torch.bfloat16:
            rtol, atol = 1e-2, 5e-2
        rtolw, atolw = (1e-3, 1e-3)
        # set seed
        torch.random.manual_seed(0)

        conv_kernel_size = 4
        dim = 4096
        for batch in [1, 2, 10]:
            # batch = 1
            input_length = [random.randint(10, 4096) for _ in range(batch)]
            input_length_tensor = torch.tensor(
                input_length, device=device, dtype=torch.int32
            )
            cu_seq_len = [0]
            for length in input_length:
                cu_seq_len.append(cu_seq_len[-1] + length)
            cu_seq_len = torch.tensor(cu_seq_len, device=device, dtype=torch.int32)
            prefix_lengths = torch.tensor([0] * batch, device=device, dtype=torch.int32)
            query_start_loc = cu_seq_len
            x = torch.randn(sum(input_length), dim, device=device, dtype=itype)
            weight = torch.randn(
                dim, conv_kernel_size, device=device, dtype=torch.float32
            )
            bias = None
            initial_states = None
            # if has_initial_states:
            #     initial_states = torch.randn(batch, width - 1, dim, device=device, dtype=itype).transpose(1, 2).requires_grad_()
            # else:
            #     initial_states = None
            x_ref = x.detach().clone()
            weight_ref = weight.detach()
            bias_ref = bias.detach().clone() if bias is not None else None
            initial_states_ref = (
                initial_states.detach().clone() if initial_states is not None else None
            )
            activation = "silu"
            out = causal_conv1d_fn(
                x.transpose(0, 1),
                weight,
                bias,
                None,
                query_start_loc,
                None,
                prefix_lengths,
                8,
                activation=activation,
            )

            out_ref = []
            offset = 0
            for b in range(batch):
                x_s = x_ref.transpose(0, 1)[:, offset : offset + input_length[b]]
                out_ref.append(
                    causal_conv1d_ref(x_s, weight_ref, bias_ref, activation=activation)
                )
                offset += input_length[b]
            out_ref = torch.cat(out_ref, dim=1)
            torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)

    # without initial state
    def test_prefill(self):
        device = "cuda"
        itype = torch.bfloat16
        if itype == torch.bfloat16:
            rtol, atol = 1e-2, 5e-2
        rtolw, atolw = (1e-3, 1e-3)
        # set seed
        torch.random.manual_seed(0)

        conv_kernel_size = 4
        dim = 4096
        for batch in [1, 4, 16]:
            # batch = 1
            logging.info(f"test_prefill batch {batch}")
            input_length = [random.randint(10, 4096) for _ in range(batch)]
            input_length_tensor = torch.tensor(
                input_length, device=device, dtype=torch.int32
            )
            seq_size_per_block = 16
            block_num = [
                math.ceil(length / seq_size_per_block) for length in input_length
            ]
            conv_states = torch.randn(
                [sum(block_num), conv_kernel_size - 1, dim],
                device=device,
                dtype=torch.float32,
            )

            offset = 0
            block_indices_list = []
            for i in range(batch):
                block_indices_list.append(
                    torch.arange(
                        offset, offset + block_num[i], device=device, dtype=torch.int32
                    )
                )
                offset += block_num[i]

            # 对齐成二维tensor，右边补-1
            max_blocks = max(block_num) if block_num else 0
            block_indices_tensor = torch.full(
                (batch, max_blocks), -1, device=device, dtype=torch.int32
            )
            for i, indices in enumerate(block_indices_list):
                block_indices_tensor[i, : len(indices)] = indices

            cu_seq_len = [0]
            for length in input_length:
                cu_seq_len.append(cu_seq_len[-1] + length)
            cu_seq_len = torch.tensor(cu_seq_len, device=device, dtype=torch.int32)
            prefix_lengths = torch.tensor([0] * batch, device=device, dtype=torch.int32)
            query_start_loc = cu_seq_len
            x = torch.randn(sum(input_length), dim, device=device, dtype=itype)
            weight = torch.randn(
                dim, conv_kernel_size, device=device, dtype=torch.float32
            )
            bias = None
            initial_states = None
            # if has_initial_states:
            #     initial_states = torch.randn(batch, width - 1, dim, device=device, dtype=itype).transpose(1, 2).requires_grad_()
            # else:
            #     initial_states = None
            x_ref = x.detach().clone()
            weight_ref = weight.detach()
            bias_ref = bias.detach().clone() if bias is not None else None
            initial_states_ref = (
                initial_states.detach().clone() if initial_states is not None else None
            )
            activation = "silu"
            out = causal_conv1d_fn(
                x.transpose(0, 1),
                weight,
                bias,
                conv_states.transpose(1, 2),
                query_start_loc,
                block_indices_tensor,
                prefix_lengths,
                seq_size_per_block,
                activation=activation,
            )

            out_ref = []
            offset = 0
            for b in range(batch):
                x_s = x_ref.transpose(0, 1)[:, offset : offset + input_length[b]]
                out_ref.append(
                    causal_conv1d_ref(x_s, weight_ref, bias_ref, activation=activation)
                )
                offset += input_length[b]
            out_ref = torch.cat(out_ref, dim=1)
            torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)
            offset = 0
            for b in range(batch):
                for i in range(1, block_num[b] + 1):
                    end_idx = (
                        offset + seq_size_per_block * i
                        if seq_size_per_block * i <= input_length[b]
                        else offset + input_length[b]
                    )
                    start_idx = end_idx - 3
                    torch.testing.assert_close(
                        conv_states[int(block_indices_tensor[b][i - 1])],
                        x[start_idx:end_idx].float(),
                        rtol=rtol,
                        atol=atol,
                    )
                offset += input_length[b]

    # with initial state
    def test_prefill_with_initial_state(self):
        device = "cuda"
        itype = torch.bfloat16
        if itype == torch.bfloat16:
            rtol, atol = 1e-2, 5e-2
        rtolw, atolw = (1e-3, 1e-3)
        # set seed
        torch.random.manual_seed(0)

        conv_kernel_size = 4
        dim = 4096
        for batch in [1, 2, 10]:
            logging.info(f"test_prefill_with_initial_state batch {batch}")
            # batch = 1
            origin_length = [random.randint(16, 4096) for _ in range(batch)]
            x = torch.randn(sum(origin_length), dim, device=device, dtype=itype)
            seq_size_per_block = 16
            block_num = [
                math.ceil(length / seq_size_per_block) for length in origin_length
            ]
            conv_states = torch.randn(
                [sum(block_num), conv_kernel_size - 1, dim],
                device=device,
                dtype=torch.float32,
            )
            reuse_block_num = [
                max(1, random.randint(0, block_num[i] - 1)) for i in range(batch)
            ]
            prefix_lengths = torch.tensor(
                [reuse_block_num[i] * seq_size_per_block for i in range(batch)],
                device=device,
                dtype=torch.int32,
            )
            input_length = [origin_length[i] - prefix_lengths[i] for i in range(batch)]

            offset = 0
            block_indices_list = []
            for i in range(batch):
                block_indices_list.append(
                    torch.arange(
                        offset, offset + block_num[i], device=device, dtype=torch.int32
                    )
                )
                offset += block_num[i]
            # 对齐成二维tensor，右边补-1
            max_blocks = max(block_num) if block_num else 0
            block_indices_tensor = torch.full(
                (batch, max_blocks), -1, device=device, dtype=torch.int32
            )
            for i, indices in enumerate(block_indices_list):
                block_indices_tensor[i, : len(indices)] = indices

            new_x = []
            offset = 0
            block_offset = 0
            for i in range(batch):
                new_x.append(x[offset + prefix_lengths[i] : offset + origin_length[i]])
                for j in range(reuse_block_num[i]):
                    conv_states[block_offset + j] = x[
                        offset
                        + (j + 1) * seq_size_per_block
                        - conv_kernel_size
                        + 1 : offset
                        + (j + 1) * seq_size_per_block
                    ]
                block_offset += block_num[i]
                offset += origin_length[i]
            new_x = torch.cat(new_x, dim=0)

            cu_seq_len = [0]
            for length in input_length:
                cu_seq_len.append(cu_seq_len[-1] + length)
            cu_seq_len = torch.tensor(cu_seq_len, device=device, dtype=torch.int32)
            query_start_loc = cu_seq_len
            weight = torch.randn(
                dim, conv_kernel_size, device=device, dtype=torch.float32
            )
            bias = None
            initial_states = None
            # if has_initial_states:
            #     initial_states = torch.randn(batch, width - 1, dim, device=device, dtype=itype).transpose(1, 2).requires_grad_()
            # else:
            #     initial_states = None
            x_ref = x.detach().clone()
            weight_ref = weight.detach()
            bias_ref = bias.detach().clone() if bias is not None else None
            activation = "silu"
            out = causal_conv1d_fn(
                new_x.transpose(0, 1),
                weight,
                bias,
                conv_states.transpose(1, 2),
                query_start_loc,
                block_indices_tensor,
                prefix_lengths,
                seq_size_per_block,
                activation=activation,
            )
            out_ref = []
            offset = 0
            for b in range(batch):
                x_s = x_ref.transpose(0, 1)[:, offset : offset + origin_length[b]]
                out_ref.append(
                    causal_conv1d_ref(x_s, weight_ref, bias_ref, activation=activation)[
                        :, prefix_lengths[b] :
                    ]
                )

                offset += origin_length[b]
            out_ref = torch.cat(out_ref, dim=1)
            torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)

            offset = 0
            for b in range(batch):
                for i in range(1, block_num[b] + 1):
                    end_idx = (
                        offset + seq_size_per_block * i
                        if seq_size_per_block * i <= origin_length[b]
                        else offset + origin_length[b]
                    )
                    start_idx = end_idx - 3
                    torch.testing.assert_close(
                        conv_states[int(block_indices_tensor[b][i - 1])],
                        x[start_idx:end_idx].float(),
                        rtol=rtol,
                        atol=atol,
                    )
                offset += origin_length[b]


if __name__ == "__main__":
    unittest.main()
