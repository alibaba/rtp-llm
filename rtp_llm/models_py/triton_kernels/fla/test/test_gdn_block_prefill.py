import logging
import math
import random
import unittest
from typing import List

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.offset import linear_offset_64
from rtp_llm.models_py.triton_kernels.fla import (
    load_initial_state_from_block_map,
    store_ssm_state_to_block_map,
)

logging.basicConfig(
    level="INFO",
    format="[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SSM_STATE_DTYPES = [torch.bfloat16, torch.float32]
INTERMEDIATE_DTYPE = torch.float32


@triton.jit
def _linear_offset_test_kernel(output, index, stride):
    tl.store(output, linear_offset_64(index, stride))


@triton.jit
def _constexpr_linear_offset_test_kernel(output, stride):
    tl.store(output, linear_offset_64(8192, stride))


class BlockTest(unittest.TestCase):
    def test_chunk_boundaries_interleaved_and_decode(self):
        for block in (32, 64, 96, 128):
            with self.subTest(block=block):
                self._check_chunk_boundaries_interleaved_and_decode(block)

    def _check_chunk_boundaries_interleaved_and_decode(self, block):
        from rtp_llm.models_py.triton_kernels.causal_conv1d.causal_conv1d import (
            causal_conv1d_fn,
            causal_conv1d_update,
        )
        from rtp_llm.models_py.triton_kernels.fla import (
            fused_recurrent_gated_delta_rule,
        )
        from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
        from rtp_llm.models_py.triton_kernels.fla.utils import assert_close

        torch.manual_seed(42)
        heads, dim = 2, 64
        channels = 3 * heads * dim
        lengths = [2 * block + 2, 3 * block + 2]
        raw = [
            torch.randn(n + 4, channels, device="cuda", dtype=torch.bfloat16)
            for n in lengths
        ]
        gates = [
            torch.nn.functional.logsigmoid(torch.rand(n + 4, heads, device="cuda"))
            for n in lengths
        ]
        betas = [
            torch.rand(n + 4, heads, device="cuda", dtype=torch.bfloat16).sigmoid()
            for n in lengths
        ]
        weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
        # Disjoint physical slots and a non-contiguous table exercise batch reordering.
        table = torch.tensor(
            [[3, 1, 5, -1, -1], [7, 2, 6, 4, -1]], device="cuda", dtype=torch.int32
        )[:, :4]
        conv = torch.full(
            (9, 3, channels), 17, device="cuda", dtype=torch.bfloat16
        ).transpose(1, 2)
        ssm = torch.full((9, heads, dim, dim), 17, device="cuda", dtype=torch.float32)
        positions = [0, 0]

        def ints(values):
            return torch.tensor(values, device="cuda", dtype=torch.int32)

        def qkv(mixed, batch=1):
            return tuple(
                x.reshape(batch, -1, heads, dim).contiguous()
                for x in mixed.chunk(3, dim=-1)
            )

        def gather_chunks(values, rows, grants):
            return torch.cat(
                [
                    values[r][positions[r] : positions[r] + n]
                    for r, n in zip(rows, grants)
                ]
            )

        def assert_untouched(state, before, touched):
            untouched = [i for i in range(state.shape[0]) if i not in touched]
            torch.testing.assert_close(
                state[untouched], before[untouched], rtol=0, atol=0
            )

        def reference(request, end):
            ref_conv = torch.full_like(conv, 17)
            mixed = causal_conv1d_fn(
                raw[request][:end].T,
                weight,
                None,
                ref_conv,
                ints([0, end]),
                table[request : request + 1],
                ints([0]),
                block,
            ).T
            q, k, v = qkv(mixed)
            out, _, state = chunk_gated_delta_rule(
                q,
                k,
                v,
                gates[request][:end].unsqueeze(0),
                betas[request][:end].unsqueeze(0),
                output_final_state=True,
                cu_seqlens=ints([0, end]),
                use_qk_l2norm_in_kernel=True,
            )
            return out, state[0], ref_conv[int(table[request, (end - 1) // block])]

        def decode(rows, maps, conv_state, ssm_state):
            seq = ints([positions[r] + 1 for r in rows])
            mixed = causal_conv1d_update(
                torch.stack([raw[r][positions[r]] for r in rows]),
                conv_state,
                weight,
                activation="silu",
                block_map=maps,
                seq_size_per_block=block,
                sequence_lengths=seq,
            )
            q, k, v = qkv(mixed, batch=len(rows))
            out, _ = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=torch.stack([gates[r][positions[r]] for r in rows]).unsqueeze(1),
                beta=torch.stack([betas[r][positions[r]] for r in rows]).unsqueeze(1),
                initial_state=ssm_state,
                inplace_final_state=True,
                block_map=maps,
                sequence_lengths=seq,
                seq_size_per_block=block,
                use_qk_l2norm_in_kernel=True,
            )
            return out

        # Dynamic grants, reversed rows, a two-token tail, then four decode steps.
        # Non-64-aligned blocks use final states to continue each request; their
        # missing intermediate SSM states must not be treated as reusable prefixes.
        for rows, grants in (
            ([0, 1], [block, 2 * block]),
            ([1, 0], [block, block]),
            ([0, 1], [2, 2]),
        ):
            before_conv, before_ssm = conv.clone(), ssm.clone()
            prefixes = ints([positions[r] for r in rows])
            maps = table[rows]
            cu = ints([0, grants[0], sum(grants)])
            mixed = gather_chunks(raw, rows, grants)
            mixed = causal_conv1d_fn(
                mixed.T, weight, None, conv, cu, maps, prefixes, block
            ).T
            initial = torch.empty(
                len(rows), heads, dim, dim, device="cuda", dtype=torch.float32
            )
            load_initial_state_from_block_map(prefixes, maps, ssm, initial, block)
            q, k, v = qkv(mixed)
            out, h, final = chunk_gated_delta_rule(
                q,
                k,
                v,
                gather_chunks(gates, rows, grants).unsqueeze(0),
                gather_chunks(betas, rows, grants).unsqueeze(0),
                initial_state=initial,
                output_final_state=True,
                cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True,
            )
            store_ssm_state_to_block_map(h, final, prefixes, cu, maps, ssm, block, 64)
            touched_conv, touched_ssm = set(), set()
            offset = 0
            for r, n in zip(rows, grants):
                start, end = positions[r], positions[r] + n
                ref_out, _, _ = reference(r, end)
                assert_close(
                    "chunk output",
                    ref_out[:, start:end],
                    out[:, offset : offset + n],
                    0.005,
                )
                for boundary in list(range(start + block, end, block)) + [end]:
                    _, ref_ssm, ref_conv = reference(r, boundary)
                    slot = int(table[r, (boundary - 1) // block])
                    touched_conv.add(slot)
                    torch.testing.assert_close(conv[slot], ref_conv, rtol=0, atol=0)
                    if boundary == end or (boundary - start) % 64 == 0:
                        touched_ssm.add(slot)
                        assert_close("boundary SSM", ref_ssm, ssm[slot], 0.005)
                positions[r] = end
                offset += n
            assert_untouched(conv, before_conv, touched_conv)
            assert_untouched(ssm, before_ssm, touched_ssm)

        # Compare full-prefill -> recurrent decode against chunked-prefill ->
        # recurrent decode, using the same backend on both sides. A full-prefill
        # rerun over each extra token changes the native BF16 algorithm instead.
        baseline_conv = torch.full_like(conv, 17)
        baseline_ssm = torch.full_like(ssm, 17)
        for r, length in enumerate(lengths):
            _, state, conv_state = reference(r, length)
            slot = int(table[r, (length - 1) // block])
            baseline_ssm[slot] = state
            baseline_conv[slot] = conv_state

        for step in range(4):
            rows = [1, 0] if step % 2 == 0 else [0, 1]
            before_conv, before_ssm = conv.clone(), ssm.clone()
            out = decode(rows, table[rows], conv, ssm)
            # The reference keeps a stable row order while the chunked request
            # batch reverses it on alternate decode steps.
            ref_output = decode([0, 1], table, baseline_conv, baseline_ssm)
            touched = set()
            for row, r in enumerate(rows):
                positions[r] += 1
                slot = int(table[r, (positions[r] - 1) // block])
                ref_out = ref_output[r : r + 1]
                ref_ssm, ref_conv = baseline_ssm[slot], baseline_conv[slot]
                touched.add(slot)
                assert_close(
                    "decode output", ref_out[:, -1:], out[row : row + 1], 0.005
                )
                assert_close("decode SSM", ref_ssm, ssm[slot], 0.005)
                torch.testing.assert_close(conv[slot], ref_conv, rtol=0, atol=0)
            assert_untouched(conv, before_conv, touched)
            assert_untouched(ssm, before_ssm, touched)

    def test_large_chunk_state_offset_uses_int64(self):
        output = torch.empty(1, dtype=torch.int64, device="cuda")
        cases = [
            (8192, 16 * 128 * 128),
            (4096, 32 * 128 * 128),
        ]
        for chunk_index, ssm_per_chunk in cases:
            with self.subTest(chunk_index=chunk_index, ssm_per_chunk=ssm_per_chunk):
                _linear_offset_test_kernel[(1,)](output, chunk_index, ssm_per_chunk)
                self.assertEqual(output.item(), chunk_index * ssm_per_chunk)
                self.assertEqual(output.item(), 1 << 31)

        _constexpr_linear_offset_test_kernel[(1,)](output, 16 * 128 * 128)
        self.assertEqual(output.item(), 1 << 31)

    def test_load_initial_state_from_block_map(self):
        device = torch.device("cuda")
        head_nums = [1, 4, 8, 16]
        k_sizes = [128, 256]
        v_size = 128
        batch_size = [1, 4, 16]
        seq_size_per_block = 16

        def test_one_case(
            k_size: int,
            head_num: int,
            bs: int,
            ssm_state_dtype: torch.dtype,
            use_narrow_block_map: bool = False,
        ):
            logging.info(
                f"test_load_initial_state_from_block_map: k_size={k_size} head_num={head_num} bs={bs} "
                f"ssm_state_dtype={ssm_state_dtype} initial_dtype={INTERMEDIATE_DTYPE}"
            )
            if bs > 1:
                prefix_length = [random.randint(10, 1024) for _ in range(bs - 1)] + [0]
            else:
                prefix_length = [random.randint(10, 1024)]
            block_num = [
                math.ceil(prefix_length[i] / seq_size_per_block) for i in range(bs)
            ]
            total_block_num = sum(block_num)

            ssm_element_size = torch.tensor([], dtype=ssm_state_dtype).element_size()
            ssm_elements_per_block = head_num * v_size * k_size
            padding_bytes = 1024
            padding_elements = padding_bytes // ssm_element_size
            total_elements_per_block = ssm_elements_per_block + padding_elements

            ssm_states_with_padding = torch.randn(
                total_block_num,
                total_elements_per_block,
                dtype=ssm_state_dtype,
                device=device,
            )
            ssm_states = ssm_states_with_padding[:, :ssm_elements_per_block].view(
                total_block_num, head_num, v_size, k_size
            )
            initial_states = torch.empty(
                bs, head_num, v_size, k_size, device=device, dtype=INTERMEDIATE_DTYPE
            )

            max_block_num = max(block_num)
            storage_width = max_block_num + int(use_narrow_block_map)
            block_map = torch.ones([bs, storage_width], dtype=torch.int32)
            offset = 0
            for i in range(bs):
                block_map[i, : block_num[i]] = torch.arange(
                    offset, offset + block_num[i], dtype=torch.int32
                )
                offset += block_num[i]
            block_map = block_map.to(device)
            if use_narrow_block_map:
                block_map = block_map[:, :max_block_num]
            prefix_length_t = torch.tensor(
                prefix_length, device=device, dtype=torch.int32
            )
            load_initial_state_from_block_map(
                prefix_length_t,
                block_map,
                ssm_states,
                initial_states,
                seq_size_per_block,
            )

            for i in range(bs):
                if prefix_length[i] > 0:
                    expect_value = ssm_states[
                        block_map[i][(prefix_length[i] - 1) // seq_size_per_block]
                    ].to(INTERMEDIATE_DTYPE)
                    torch.testing.assert_close(initial_states[i], expect_value)
                else:
                    torch.testing.assert_close(
                        initial_states[i], torch.zeros_like(initial_states[i])
                    )

        for ssm_state_dtype in SSM_STATE_DTYPES:
            for k_size in k_sizes:
                for head_num in head_nums:
                    for bs in batch_size:
                        test_one_case(k_size, head_num, bs, ssm_state_dtype)
        test_one_case(128, 4, 4, torch.float32, use_narrow_block_map=True)

    def test_store_ssm_state_to_block_map(self):
        device = torch.device("cuda")
        head_nums = [4, 8]
        k_sizes = [128]
        v_size = 128
        batch_size = [1, 4, 16]
        seq_size_per_block = 128
        chunk_size = 64

        def _test_one_case(
            k_size: int,
            head_num: int,
            bs: int,
            prefix_lengths: List[int],
            input_lengths: List[int],
            ssm_state_dtype: torch.dtype,
            use_narrow_block_map: bool = False,
        ):
            logging.info(
                f"test_store_ssm_state_to_block_map: k_size={k_size} head_num={head_num} bs={bs} "
                f"ssm_state_dtype={ssm_state_dtype} intermediate_dtype={INTERMEDIATE_DTYPE}"
            )
            block_num = [
                math.ceil((input_lengths[i] + prefix_lengths[i]) / seq_size_per_block)
                for i in range(bs)
            ]
            chunk_lengths = [
                math.ceil(input_lengths[i] / chunk_size) for i in range(bs)
            ]
            total_block_num = sum(block_num)
            total_chunk_size = sum(chunk_lengths)
            map_width = max(block_num)
            storage_width = map_width + int(use_narrow_block_map)
            block_map = torch.ones([bs, storage_width], dtype=torch.int32)
            offset = 0
            for i in range(bs):
                block_map[i, : block_num[i]] = torch.arange(
                    offset, offset + block_num[i], dtype=torch.int32
                )
                offset += block_num[i]

            ssm_element_size = torch.tensor([], dtype=ssm_state_dtype).element_size()
            ssm_elements_per_block = head_num * v_size * k_size
            padding_bytes = 1024
            padding_elements = padding_bytes // ssm_element_size
            total_elements_per_block = ssm_elements_per_block + padding_elements

            ssm_states_with_padding = torch.ones(
                total_block_num,
                total_elements_per_block,
                dtype=ssm_state_dtype,
                device=device,
            )
            ssm_states = ssm_states_with_padding[:, :ssm_elements_per_block].view(
                total_block_num, head_num, v_size, k_size
            )

            final_states = torch.randn(
                bs, head_num, v_size, k_size, device=device, dtype=INTERMEDIATE_DTYPE
            )
            h = torch.randn(
                total_chunk_size,
                head_num,
                v_size,
                k_size,
                device=device,
                dtype=INTERMEDIATE_DTYPE,
            )

            prefix_lengths_t = torch.tensor(
                prefix_lengths, device=device, dtype=torch.int32
            )
            cu_seq_len = [0]
            for length in input_lengths:
                cu_seq_len.append(cu_seq_len[-1] + length)
            cu_seq_len = torch.tensor(cu_seq_len, device=device, dtype=torch.int32)
            block_map_gpu = block_map.to(device)
            if use_narrow_block_map:
                block_map_gpu = block_map_gpu[:, :map_width]
            store_ssm_state_to_block_map(
                h,
                final_states,
                prefix_lengths_t,
                cu_seq_len,
                block_map_gpu,
                ssm_states,
                seq_size_per_block,
                chunk_size,
            )

            chunk_offset = 0
            for i in range(bs):
                prefix_offset = prefix_lengths[i] // seq_size_per_block
                for block_idx in range(block_num[i] - 1):
                    block_idx -= prefix_offset
                    if block_idx < 0:
                        continue
                    chunk_idx = (
                        chunk_offset
                        + (block_idx + 1) * seq_size_per_block // chunk_size
                    )
                    torch.testing.assert_close(
                        ssm_states[block_map[i][block_idx + prefix_offset]],
                        h[chunk_idx].to(ssm_state_dtype),
                    )
                torch.testing.assert_close(
                    ssm_states[block_map[i][block_num[i] - 1]],
                    final_states[i].to(ssm_state_dtype),
                )
                chunk_offset += chunk_lengths[i]

        for ssm_state_dtype in SSM_STATE_DTYPES:
            for k_size in k_sizes:
                for head_num in head_nums:
                    for bs in batch_size:
                        if bs > 1:
                            prefix_lengths = [
                                random.randint(1, 10) * seq_size_per_block
                                for _ in range(bs - 1)
                            ] + [0]
                        else:
                            prefix_lengths = [
                                random.randint(1, 10) * seq_size_per_block
                            ]
                        input_lengths = [random.randint(10, 1024) for _ in range(bs)]
                        _test_one_case(
                            k_size,
                            head_num,
                            bs,
                            prefix_lengths,
                            input_lengths,
                            ssm_state_dtype,
                        )
                        input_lengths = [
                            random.randint(1, 10) * seq_size_per_block
                            for _ in range(bs)
                        ]
                        _test_one_case(
                            k_size,
                            head_num,
                            bs,
                            prefix_lengths,
                            input_lengths,
                            ssm_state_dtype,
                        )
        _test_one_case(
            128,
            4,
            4,
            [128, 256, 384, 0],
            [64, 128, 192, 256],
            torch.float32,
            use_narrow_block_map=True,
        )


if __name__ == "__main__":
    unittest.main()
