import unittest

import torch

from rtp_llm.models_py.triton_kernels.kimi_kda.fused_recurrent import (
    fused_recurrent_kda,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.short_conv import (
    kimi_kda_short_conv_decode,
    kimi_kda_short_conv_paged_decode,
    kimi_kda_short_conv_paged_target_verify,
)


class KimiKDAShortConvPagedDecodeTest(unittest.TestCase):
    # The fused multi-step kernel preserves BF16 checkpoints bitwise. Its
    # unrolled FP32 reduction may round the final BF16 output by one ULP versus
    # separately compiled one-token launches.
    _TARGET_OUTPUT_RTOL = 2**-7
    _TARGET_OUTPUT_ATOL = 2**-10

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        torch.manual_seed(20260730)

    @staticmethod
    def _block_map(batch: int, pages: int) -> torch.Tensor:
        block_ids = torch.arange(
            1,
            batch * pages + 1,
            dtype=torch.int32,
            device="cuda",
        )
        return block_ids.reshape(batch, pages)

    @staticmethod
    def _reference(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        weight: torch.Tensor,
        initial_state: torch.Tensor,
        block_map: torch.Tensor,
        sequence_lengths_plus_one: torch.Tensor,
        page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, projection_size = q.shape
        outputs = torch.empty(
            (3, batch, projection_size), dtype=q.dtype, device=q.device
        )
        final_state = initial_state.clone()
        packed_inputs = (q, k, v)
        lengths = sequence_lengths_plus_one.cpu().tolist()
        host_block_map = block_map.cpu().tolist()
        for batch_index, length_plus_one in enumerate(lengths):
            read_page = max(0, (length_plus_one - 2) // page_size)
            write_page = max(0, (length_plus_one - 1) // page_size)
            read_block = host_block_map[batch_index][read_page]
            write_block = host_block_map[batch_index][write_page]
            for projection, projected in enumerate(packed_inputs):
                begin = projection * projection_size
                end = begin + projection_size
                history = (
                    initial_state[read_block, :, begin:end].transpose(0, 1).contiguous()
                    if read_block > 0 and length_plus_one > 1
                    else projected.new_zeros(projection_size, weight.shape[1] - 1)
                )
                outputs[projection, batch_index] = kimi_kda_short_conv_decode(
                    projected[batch_index],
                    weight[begin:end],
                    history,
                )
                if write_block > 0 and length_plus_one > 0:
                    updated = torch.cat(
                        (history[:, 1:], projected[batch_index, :, None]), dim=1
                    )
                    final_state[write_block, :, begin:end] = updated.transpose(0, 1)
        return outputs, final_state

    @staticmethod
    def _target_verify_reference(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        weight: torch.Tensor,
        initial_state: torch.Tensor,
        block_map: torch.Tensor,
        sequence_lengths_plus_one: torch.Tensor,
        page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Independent safe replay of the pre-fusion step-loop semantics."""

        batch, steps, projection_size = q.shape
        outputs = torch.empty(
            (3, batch, steps, projection_size), dtype=q.dtype, device=q.device
        )
        final_state = initial_state.clone()
        packed_inputs = (q, k, v)
        host_lengths = sequence_lengths_plus_one.cpu().tolist()
        host_block_map = block_map.cpu().tolist()
        max_pages = block_map.shape[1]
        physical_blocks = initial_state.shape[0]
        width = weight.shape[1]

        for batch_index, length_plus_one in enumerate(host_lengths):
            checkpoint_page_base = (length_plus_one - 1) // page_size
            for projection, projected in enumerate(packed_inputs):
                begin = projection * projection_size
                end = begin + projection_size
                previous_checkpoint = projected.new_zeros(projection_size, width - 1)
                for step in range(steps):
                    checkpoint_page = checkpoint_page_base + step
                    checkpoint_block = (
                        host_block_map[batch_index][checkpoint_page]
                        if 0 <= checkpoint_page < max_pages
                        else 0
                    )
                    checkpoint_valid = 0 < checkpoint_block < physical_blocks
                    if step > 0 and checkpoint_valid:
                        final_state[checkpoint_block, :, begin:end] = (
                            previous_checkpoint.transpose(0, 1)
                        )

                    read_page = (length_plus_one + step - 2) // page_size
                    logical_write_page = (length_plus_one + step - 1) // page_size
                    if read_page == logical_write_page:
                        read_block = checkpoint_block
                    else:
                        read_block = (
                            host_block_map[batch_index][read_page]
                            if 0 <= read_page < max_pages
                            else 0
                        )
                    read_valid = (
                        length_plus_one + step > 1 and 0 < read_block < physical_blocks
                    )
                    history = (
                        final_state[read_block, :, begin:end]
                        .transpose(0, 1)
                        .contiguous()
                        if read_valid
                        else projected.new_zeros(projection_size, width - 1)
                    )
                    token = projected[batch_index, step]
                    outputs[projection, batch_index, step] = kimi_kda_short_conv_decode(
                        token,
                        weight[begin:end],
                        history,
                    )
                    previous_checkpoint = torch.cat(
                        (history[:, 1:], token[:, None]), dim=1
                    )
                    if checkpoint_valid:
                        final_state[checkpoint_block, :, begin:end] = (
                            previous_checkpoint.transpose(0, 1)
                        )
        return outputs, final_state

    @staticmethod
    def _iterative_target_verify_short_conv(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        weight: torch.Tensor,
        initial_state: torch.Tensor,
        block_map: torch.Tensor,
        sequence_lengths_plus_one: torch.Tensor,
        page_size: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Run the pre-fusion Python step loop as the precision baseline."""

        batch, steps, _ = q.shape
        state = initial_state.clone()
        batch_idx = torch.arange(batch, device=block_map.device)
        reserve_base = torch.div(
            sequence_lengths_plus_one - 1,
            page_size,
            rounding_mode="floor",
        ).to(torch.long)
        output_steps: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for step in range(steps):
            reserve_col = reserve_base + step
            logical_col = torch.div(
                sequence_lengths_plus_one + step - 1,
                page_size,
                rounding_mode="floor",
            ).to(torch.long)
            dest_ids = block_map[batch_idx, reserve_col].to(torch.long)
            if step > 0:
                src_ids = block_map[batch_idx, reserve_col - 1].to(torch.long)
                state[dest_ids] = state[src_ids]
            step_block_map = block_map.clone()
            step_block_map[batch_idx, logical_col] = dest_ids.to(step_block_map.dtype)
            output_steps.append(
                kimi_kda_short_conv_paged_decode(
                    q[:, step, :].contiguous(),
                    k[:, step, :].contiguous(),
                    v[:, step, :].contiguous(),
                    weight,
                    state,
                    step_block_map,
                    sequence_lengths_plus_one + step,
                    page_size,
                )
            )

        outputs = tuple(
            torch.stack(
                [step_output[projection] for step_output in output_steps], dim=1
            )
            for projection in range(3)
        )
        return outputs, state

    def _inputs(self):
        batch = 4
        projection_size = 256
        page_size = 8
        pages = 3
        width = 4
        block_map = self._block_map(batch, pages)
        block_count = batch * pages + 1
        packed_qkv = torch.randn(
            batch, 3 * projection_size, dtype=torch.bfloat16, device="cuda"
        )
        q, k, v = torch.split(packed_qkv, projection_size, dim=1)
        weight = torch.randn(
            3 * projection_size, width, dtype=torch.float32, device="cuda"
        )
        conv_state = torch.randn(
            block_count,
            width - 1,
            3 * projection_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        # Covers first-token initialization and the 8 -> 9 page transition.
        lengths_plus_one = torch.tensor([1, 8, 9, 10], dtype=torch.int32, device="cuda")
        return (
            q,
            k,
            v,
            weight,
            conv_state,
            block_map,
            lengths_plus_one,
            page_size,
        )

    def test_matches_per_request_decode_at_page_boundaries(self) -> None:
        inputs = self._inputs()
        q, k, v, weight, conv_state, block_map, lengths, page_size = inputs
        initial_state = conv_state.clone()
        expected_output, expected_state = self._reference(
            q, k, v, weight, initial_state, block_map, lengths, page_size
        )

        actual_output = kimi_kda_short_conv_paged_decode(*inputs[:4], *inputs[4:])

        torch.testing.assert_close(
            torch.stack(actual_output), expected_output, rtol=0, atol=0
        )
        torch.testing.assert_close(conv_state, expected_state, rtol=0, atol=0)

    def test_synthetic_stream_does_not_modify_sentinel_block(self) -> None:
        inputs = list(self._inputs())
        block_map = inputs[5].clone()
        block_map[0].zero_()
        inputs[5] = block_map
        initial_state = inputs[4].clone()
        expected_output, expected_state = self._reference(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            initial_state,
            inputs[5],
            inputs[6],
            inputs[7],
        )

        actual_output = kimi_kda_short_conv_paged_decode(*inputs[:4], *inputs[4:])

        torch.testing.assert_close(
            torch.stack(actual_output), expected_output, rtol=0, atol=0
        )
        torch.testing.assert_close(inputs[4], expected_state, rtol=0, atol=0)
        torch.testing.assert_close(inputs[4][0], initial_state[0], rtol=0, atol=0)

    def test_out_of_range_physical_block_is_zero_initialized_and_not_written(
        self,
    ) -> None:
        inputs = list(self._inputs())
        q, k, v, weight, conv_state, block_map, lengths, page_size = inputs
        block_map = block_map[:1].clone()
        block_map[0].fill_(conv_state.shape[0] + 17)
        lengths = lengths[:1].clone()
        initial_state = conv_state.clone()
        zero_history = torch.zeros(
            q.shape[1], weight.shape[1] - 1, dtype=q.dtype, device=q.device
        )
        expected = []
        for projection, projected in enumerate((q[:1], k[:1], v[:1])):
            begin = projection * q.shape[1]
            end = begin + q.shape[1]
            expected.append(
                kimi_kda_short_conv_decode(
                    projected[0], weight[begin:end], zero_history
                )
            )

        actual = kimi_kda_short_conv_paged_decode(
            q[:1],
            k[:1],
            v[:1],
            weight,
            conv_state,
            block_map,
            lengths,
            page_size,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(
            torch.stack([item[0] for item in actual]),
            torch.stack(expected),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(conv_state, initial_state, rtol=0, atol=0)

    def test_cuda_graph_replay_uses_live_indices_and_inputs(self) -> None:
        inputs = list(self._inputs())
        q, k, v, weight, conv_state, block_map, lengths, page_size = inputs
        initial_state = conv_state.clone()

        # Compile on a side stream before capture, as required by CUDAGraph.
        warmup_state = initial_state.clone()
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            kimi_kda_short_conv_paged_decode(
                q,
                k,
                v,
                weight,
                warmup_state,
                block_map,
                lengths,
                page_size,
            )
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = kimi_kda_short_conv_paged_decode(
                q,
                k,
                v,
                weight,
                conv_state,
                block_map,
                lengths,
                page_size,
            )

        replay_q = torch.randn_like(q)
        replay_k = torch.randn_like(k)
        replay_v = torch.randn_like(v)
        replay_lengths = torch.tensor([3, 9, 10, 17], dtype=torch.int32, device="cuda")
        replay_block_map = block_map.clone()
        replay_block_map[0].zero_()
        expected_output, expected_state = self._reference(
            replay_q,
            replay_k,
            replay_v,
            weight,
            initial_state,
            replay_block_map,
            replay_lengths,
            page_size,
        )
        conv_state.copy_(initial_state)
        q.copy_(replay_q)
        k.copy_(replay_k)
        v.copy_(replay_v)
        lengths.copy_(replay_lengths)
        block_map.copy_(replay_block_map)

        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            torch.stack(captured_output), expected_output, rtol=0, atol=0
        )
        torch.testing.assert_close(conv_state, expected_state, rtol=0, atol=0)
        torch.testing.assert_close(conv_state[0], initial_state[0], rtol=0, atol=0)

    def test_target_verify_matches_iterative_steps_at_page_boundaries(self) -> None:
        batch = 4
        steps = 4
        projection_size = 256
        page_size = 8
        pages = 6
        width = 4
        block_map = self._block_map(batch, pages)
        block_count = batch * pages + 1
        q = torch.randn(
            batch,
            steps,
            projection_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        weight = torch.randn(
            3 * projection_size,
            width,
            dtype=torch.float32,
            device="cuda",
        )
        initial_state = torch.randn(
            block_count,
            width - 1,
            3 * projection_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        # Includes first-token initialization, an in-page update, and both
        # sides of the 8 -> 9 write-page transition that broke the old fused path.
        lengths = torch.tensor([1, 7, 8, 9], dtype=torch.int32, device="cuda")

        expected_planes, expected_state = self._iterative_target_verify_short_conv(
            q,
            k,
            v,
            weight,
            initial_state,
            block_map,
            lengths,
            page_size,
        )
        expected = torch.stack(expected_planes)

        fused_state = initial_state.clone()
        actual = torch.stack(
            kimi_kda_short_conv_paged_target_verify(
                q,
                k,
                v,
                weight,
                fused_state,
                block_map,
                lengths,
                page_size,
            )
        )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=self._TARGET_OUTPUT_RTOL,
            atol=self._TARGET_OUTPUT_ATOL,
        )
        torch.testing.assert_close(fused_state, expected_state, rtol=0, atol=0)

    def test_target_verify_matches_previous_step_loop_precision_matrix(self) -> None:
        page_size = 8
        cases = (
            # batch, steps, projection, width, heterogeneous lengths
            (1, 1, 63, 2, [1]),
            (2, 2, 64, 3, [7, 8]),
            (2, 3, 65, 4, [8, 9]),
            (4, 4, 127, 5, [1, 7, 8, 9]),
            (2, 4, 129, 7, [15, 17]),
            (1, 4, 1536, 4, [9]),
            (1, 4, 1537, 4, [16]),
        )
        for batch, steps, projection_size, width, host_lengths in cases:
            with self.subTest(
                batch=batch,
                steps=steps,
                projection_size=projection_size,
                width=width,
                lengths=host_lengths,
            ):
                lengths = torch.tensor(host_lengths, dtype=torch.int32, device="cuda")
                max_write_page = max(
                    (length - 1) // page_size + steps - 1 for length in host_lengths
                )
                pages = max_write_page + 2
                block_map = self._block_map(batch, pages)
                physical_blocks = batch * pages + 1
                q = torch.randn(
                    batch,
                    steps,
                    projection_size,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                k = torch.randn_like(q)
                v = torch.randn_like(q)
                weight = torch.randn(
                    3 * projection_size,
                    width,
                    dtype=torch.float32,
                    device="cuda",
                )
                initial_state = torch.randn(
                    physical_blocks,
                    width - 1,
                    3 * projection_size,
                    dtype=torch.bfloat16,
                    device="cuda",
                )

                expected, expected_state = self._iterative_target_verify_short_conv(
                    q,
                    k,
                    v,
                    weight,
                    initial_state,
                    block_map,
                    lengths,
                    page_size,
                )
                actual_state = initial_state.clone()
                actual = kimi_kda_short_conv_paged_target_verify(
                    q,
                    k,
                    v,
                    weight,
                    actual_state,
                    block_map,
                    lengths,
                    page_size,
                )

                for actual_projection, expected_projection in zip(actual, expected):
                    torch.testing.assert_close(
                        actual_projection,
                        expected_projection,
                        rtol=self._TARGET_OUTPUT_RTOL,
                        atol=self._TARGET_OUTPUT_ATOL,
                    )
                torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)

    def test_target_verify_masks_invalid_and_missing_checkpoint_blocks(self) -> None:
        batch = 3
        steps = 4
        projection_size = 65
        page_size = 8
        pages = 2
        width = 4
        block_map = self._block_map(batch, pages)
        physical_blocks = batch * pages + 1
        # Request zero exercises the block-0 sentinel, request one uses an
        # invalid physical ID, and request two runs out of logical checkpoint
        # columns after token zero.
        block_map[0].zero_()
        block_map[1].fill_(physical_blocks + 17)
        lengths = torch.tensor([1, 8, 9], dtype=torch.int32, device="cuda")
        q = torch.randn(
            batch,
            steps,
            projection_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        weight = torch.randn(
            3 * projection_size,
            width,
            dtype=torch.float32,
            device="cuda",
        )
        initial_state = torch.randn(
            physical_blocks,
            width - 1,
            3 * projection_size,
            dtype=torch.bfloat16,
            device="cuda",
        )

        expected, expected_state = self._target_verify_reference(
            q,
            k,
            v,
            weight,
            initial_state,
            block_map,
            lengths,
            page_size,
        )
        actual_state = initial_state.clone()
        actual = torch.stack(
            kimi_kda_short_conv_paged_target_verify(
                q,
                k,
                v,
                weight,
                actual_state,
                block_map,
                lengths,
                page_size,
            )
        )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=self._TARGET_OUTPUT_RTOL,
            atol=self._TARGET_OUTPUT_ATOL,
        )
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
        torch.testing.assert_close(actual_state[0], initial_state[0], rtol=0, atol=0)

    def test_target_verify_preserves_physical_block_alias_order(self) -> None:
        batch = 2
        steps = 4
        projection_size = 65
        page_size = 8
        width = 4
        # Exercise both a single block reused by every logical page and a
        # non-adjacent alias. The reduced barrier schedule must preserve the
        # old loop's physical write/read order in both cases.
        block_map = torch.tensor(
            [[1, 1, 1, 1, 1], [2, 3, 2, 4, 3]],
            dtype=torch.int32,
            device="cuda",
        )
        physical_blocks = 5
        lengths = torch.tensor([7, 8], dtype=torch.int32, device="cuda")
        q = torch.randn(
            batch,
            steps,
            projection_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        weight = torch.randn(
            3 * projection_size,
            width,
            dtype=torch.float32,
            device="cuda",
        )
        initial_state = torch.randn(
            physical_blocks,
            width - 1,
            3 * projection_size,
            dtype=torch.bfloat16,
            device="cuda",
        )

        expected, expected_state = self._target_verify_reference(
            q,
            k,
            v,
            weight,
            initial_state,
            block_map,
            lengths,
            page_size,
        )
        actual_state = initial_state.clone()
        actual = torch.stack(
            kimi_kda_short_conv_paged_target_verify(
                q,
                k,
                v,
                weight,
                actual_state,
                block_map,
                lengths,
                page_size,
            )
        )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=self._TARGET_OUTPUT_RTOL,
            atol=self._TARGET_OUTPUT_ATOL,
        )
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)

    def test_target_verify_cuda_graph_replay_uses_live_inputs_and_pages(self) -> None:
        batch = 2
        steps = 4
        projection_size = 1536
        page_size = 8
        pages = 7
        width = 4
        block_map = self._block_map(batch, pages)
        lengths = torch.tensor([8, 9], dtype=torch.int32, device="cuda")
        q = torch.randn(
            batch,
            steps,
            projection_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        weight = torch.randn(
            3 * projection_size,
            width,
            dtype=torch.float32,
            device="cuda",
        )
        initial_state = torch.randn(
            batch * pages + 1,
            width - 1,
            3 * projection_size,
            dtype=torch.bfloat16,
            device="cuda",
        )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            kimi_kda_short_conv_paged_target_verify(
                q,
                k,
                v,
                weight,
                initial_state.clone(),
                block_map,
                lengths,
                page_size,
            )
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph_state = initial_state.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = kimi_kda_short_conv_paged_target_verify(
                q,
                k,
                v,
                weight,
                graph_state,
                block_map,
                lengths,
                page_size,
            )

        replay_lengths = (
            torch.tensor([1, 7], dtype=torch.int32, device="cuda"),
            torch.tensor([9, 17], dtype=torch.int32, device="cuda"),
        )
        for replay_index, next_lengths in enumerate(replay_lengths):
            with self.subTest(replay=replay_index):
                replay_q = torch.randn_like(q)
                replay_k = torch.randn_like(k)
                replay_v = torch.randn_like(v)
                replay_block_map = self._block_map(batch, pages)
                if replay_index == 1:
                    replay_block_map = replay_block_map.flip(1).contiguous()
                expected, expected_state = self._iterative_target_verify_short_conv(
                    replay_q,
                    replay_k,
                    replay_v,
                    weight,
                    initial_state,
                    replay_block_map,
                    next_lengths,
                    page_size,
                )

                graph_state.copy_(initial_state)
                q.copy_(replay_q)
                k.copy_(replay_k)
                v.copy_(replay_v)
                block_map.copy_(replay_block_map)
                lengths.copy_(next_lengths)
                graph.replay()
                torch.cuda.synchronize()

                for actual_projection, expected_projection in zip(
                    graph_output, expected
                ):
                    torch.testing.assert_close(
                        actual_projection,
                        expected_projection,
                        rtol=self._TARGET_OUTPUT_RTOL,
                        atol=self._TARGET_OUTPUT_ATOL,
                    )
                torch.testing.assert_close(graph_state, expected_state, rtol=0, atol=0)

    def test_target_verify_recurrent_matches_iterative_checkpoints(self) -> None:
        batch = 2
        steps = 4
        heads = 2
        state_dim = 32
        page_size = 8
        pages = 6
        block_map = self._block_map(batch, pages)
        lengths = torch.tensor([8, 9], dtype=torch.int32, device="cuda")
        q = torch.randn(
            batch,
            steps,
            heads,
            state_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        raw_gate = torch.randn_like(q)
        raw_beta = torch.randn(batch, steps, heads, dtype=torch.bfloat16, device="cuda")
        a_log = torch.randn(heads, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(heads * state_dim, dtype=torch.float32, device="cuda")
        initial_state = torch.randn(
            batch * pages + 1,
            heads,
            state_dim,
            state_dim,
            dtype=torch.float32,
            device="cuda",
        )
        one_token_cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
        batch_idx = torch.arange(batch, device="cuda")
        reserve_base = torch.div(lengths - 1, page_size, rounding_mode="floor").to(
            torch.long
        )

        iterative_state = initial_state.clone()
        iterative_outputs: list[torch.Tensor] = []
        for step in range(steps):
            reserve_col = reserve_base + step
            logical_col = torch.div(
                lengths + step - 1, page_size, rounding_mode="floor"
            ).to(torch.long)
            dest_ids = block_map[batch_idx, reserve_col].to(torch.long)
            if step > 0:
                src_ids = block_map[batch_idx, reserve_col - 1].to(torch.long)
                iterative_state[dest_ids] = iterative_state[src_ids]
            step_map = block_map.clone()
            step_map[batch_idx, logical_col] = dest_ids.to(step_map.dtype)
            output, _ = fused_recurrent_kda(
                q[:, step : step + 1].contiguous(),
                k[:, step : step + 1].contiguous(),
                v[:, step : step + 1].contiguous(),
                raw_gate[:, step : step + 1].contiguous(),
                raw_beta[:, step : step + 1].contiguous(),
                initial_state=iterative_state,
                A_log=a_log,
                dt_bias=dt_bias,
                inplace_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=-20.0,
                state_v_first=False,
                cu_seqlens=one_token_cu,
                block_map=step_map,
                seq_size_per_block=page_size,
                sequence_lengths=lengths + step,
            )
            iterative_outputs.append(output[:, 0])
        expected = torch.stack(iterative_outputs, dim=1)

        fused_state = initial_state.clone()
        full_cu = torch.arange(
            0,
            (batch + 1) * steps,
            steps,
            dtype=torch.int32,
            device="cuda",
        )
        actual, _ = fused_recurrent_kda(
            q,
            k,
            v,
            raw_gate,
            raw_beta,
            initial_state=fused_state,
            A_log=a_log,
            dt_bias=dt_bias,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=-20.0,
            state_v_first=False,
            cu_seqlens=full_cu,
            block_map=block_map,
            seq_size_per_block=page_size,
            sequence_lengths=lengths,
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(fused_state, iterative_state, rtol=0, atol=0)

    def test_target_verify_full_kda_matches_previous_step_loop_bitwise(self) -> None:
        batch = 2
        steps = 4
        heads = 12
        state_dim = 128
        projection_size = heads * state_dim
        page_size = 8
        pages = 7
        width = 4
        block_map = self._block_map(batch, pages)
        lengths = torch.tensor([8, 9], dtype=torch.int32, device="cuda")
        q = torch.randn(
            batch,
            steps,
            projection_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        conv_weight = torch.randn(
            3 * projection_size,
            width,
            dtype=torch.float32,
            device="cuda",
        )
        initial_conv_state = torch.randn(
            batch * pages + 1,
            width - 1,
            3 * projection_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        raw_gate = torch.randn(
            batch,
            steps,
            heads,
            state_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        raw_beta = torch.randn(
            batch,
            steps,
            heads,
            dtype=torch.bfloat16,
            device="cuda",
        )
        a_log = torch.randn(heads, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(heads * state_dim, dtype=torch.float32, device="cuda")
        initial_ssm_state = torch.randn(
            batch * pages + 1,
            heads,
            state_dim,
            state_dim,
            dtype=torch.float32,
            device="cuda",
        )

        expected_conv, expected_conv_state = self._iterative_target_verify_short_conv(
            q,
            k,
            v,
            conv_weight,
            initial_conv_state,
            block_map,
            lengths,
            page_size,
        )
        one_token_cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
        batch_idx = torch.arange(batch, device="cuda")
        reserve_base = torch.div(lengths - 1, page_size, rounding_mode="floor").to(
            torch.long
        )
        expected_ssm_state = initial_ssm_state.clone()
        expected_output_steps: list[torch.Tensor] = []
        for step in range(steps):
            reserve_col = reserve_base + step
            logical_col = torch.div(
                lengths + step - 1,
                page_size,
                rounding_mode="floor",
            ).to(torch.long)
            dest_ids = block_map[batch_idx, reserve_col].to(torch.long)
            if step > 0:
                src_ids = block_map[batch_idx, reserve_col - 1].to(torch.long)
                expected_ssm_state[dest_ids] = expected_ssm_state[src_ids]
            step_map = block_map.clone()
            step_map[batch_idx, logical_col] = dest_ids.to(step_map.dtype)
            step_output, _ = fused_recurrent_kda(
                expected_conv[0][:, step : step + 1]
                .contiguous()
                .reshape(batch, 1, heads, state_dim),
                expected_conv[1][:, step : step + 1]
                .contiguous()
                .reshape(batch, 1, heads, state_dim),
                expected_conv[2][:, step : step + 1]
                .contiguous()
                .reshape(batch, 1, heads, state_dim),
                raw_gate[:, step : step + 1].contiguous(),
                raw_beta[:, step : step + 1].float().contiguous(),
                initial_state=expected_ssm_state,
                A_log=a_log,
                dt_bias=dt_bias,
                inplace_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=-20.0,
                state_v_first=False,
                cu_seqlens=one_token_cu,
                block_map=step_map,
                seq_size_per_block=page_size,
                sequence_lengths=lengths + step,
            )
            expected_output_steps.append(step_output[:, 0])
        expected_output = torch.stack(expected_output_steps, dim=1)

        actual_conv_state = initial_conv_state.clone()
        actual_conv = kimi_kda_short_conv_paged_target_verify(
            q,
            k,
            v,
            conv_weight,
            actual_conv_state,
            block_map,
            lengths,
            page_size,
        )
        actual_ssm_state = initial_ssm_state.clone()
        full_cu = torch.arange(
            0,
            (batch + 1) * steps,
            steps,
            dtype=torch.int32,
            device="cuda",
        )
        actual_output, _ = fused_recurrent_kda(
            actual_conv[0].reshape(batch, steps, heads, state_dim),
            actual_conv[1].reshape(batch, steps, heads, state_dim),
            actual_conv[2].reshape(batch, steps, heads, state_dim),
            raw_gate,
            raw_beta.float(),
            initial_state=actual_ssm_state,
            A_log=a_log,
            dt_bias=dt_bias,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=-20.0,
            state_v_first=False,
            cu_seqlens=full_cu,
            block_map=block_map,
            seq_size_per_block=page_size,
            sequence_lengths=lengths,
        )

        torch.testing.assert_close(
            actual_output,
            expected_output,
            rtol=self._TARGET_OUTPUT_RTOL,
            atol=self._TARGET_OUTPUT_ATOL,
        )
        torch.testing.assert_close(
            actual_conv_state, expected_conv_state, rtol=0, atol=0
        )
        torch.testing.assert_close(actual_ssm_state, expected_ssm_state, rtol=0, atol=0)
        checkpoint_cols = (
            reserve_base[:, None] + torch.arange(steps, device="cuda")[None, :]
        )
        checkpoint_ids = block_map[batch_idx[:, None], checkpoint_cols].to(torch.long)
        torch.testing.assert_close(
            actual_conv_state[checkpoint_ids],
            expected_conv_state[checkpoint_ids],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual_ssm_state[checkpoint_ids],
            expected_ssm_state[checkpoint_ids],
            rtol=0,
            atol=0,
        )

    def test_paged_recurrent_matches_gathered_batch_in_eager_and_graph(self) -> None:
        batch = 4
        heads = 2
        state_dim = 128
        page_size = 8
        block_map = self._block_map(batch, 3)
        lengths_plus_one = torch.tensor([2, 8, 9, 10], dtype=torch.int32, device="cuda")
        cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
        state_cache = torch.randn(
            batch * 3 + 1,
            heads,
            state_dim,
            state_dim,
            dtype=torch.float32,
            device="cuda",
        )
        initial_cache = state_cache.clone()
        q = torch.randn(1, batch, heads, state_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        raw_gate = torch.randn_like(q)
        raw_beta = torch.randn(1, batch, heads, dtype=torch.float32, device="cuda")
        a_log = torch.randn(heads, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(heads * state_dim, dtype=torch.float32, device="cuda")
        host_lengths = lengths_plus_one.cpu().tolist()
        host_block_map = block_map.cpu().tolist()
        read_blocks = [
            host_block_map[index][(length - 2) // page_size]
            for index, length in enumerate(host_lengths)
        ]
        write_blocks = [
            host_block_map[index][(length - 1) // page_size]
            for index, length in enumerate(host_lengths)
        ]
        gathered_state = initial_cache[read_blocks].contiguous()

        expected_output, expected_final = fused_recurrent_kda(
            q,
            k,
            v,
            raw_gate,
            raw_beta,
            initial_state=gathered_state,
            A_log=a_log,
            dt_bias=dt_bias,
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=-20.0,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
        )

        def run_paged(cache: torch.Tensor):
            return fused_recurrent_kda(
                q,
                k,
                v,
                raw_gate,
                raw_beta,
                initial_state=cache,
                A_log=a_log,
                dt_bias=dt_bias,
                inplace_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=-20.0,
                state_v_first=True,
                cu_seqlens=cu_seqlens,
                block_map=block_map,
                seq_size_per_block=page_size,
                sequence_lengths=lengths_plus_one,
            )

        actual_output, _ = run_paged(state_cache)

        expected_cache = initial_cache.clone()
        expected_cache[write_blocks] = expected_final
        torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(state_cache, expected_cache, rtol=0, atol=0)

        warmup_cache = initial_cache.clone()
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            run_paged(warmup_cache)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph_cache = initial_cache.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output, _ = run_paged(graph_cache)
        graph_cache.copy_(initial_cache)

        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(graph_cache, expected_cache, rtol=0, atol=0)

    def test_tp8_local_shape_two_graph_replays_match_eager_bitwise(self) -> None:
        batch = 2
        heads = 12
        state_dim = 128
        projection_size = heads * state_dim
        page_size = 8
        pages = 3
        block_map = self._block_map(batch, pages)
        block_count = batch * pages + 1
        lengths_plus_one = torch.tensor([2, 9], dtype=torch.int32, device="cuda")
        cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
        packed_qkv = torch.randn(
            batch,
            3 * projection_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        q, k, v = torch.split(packed_qkv, projection_size, dim=1)
        conv_weight = torch.randn(
            3 * projection_size,
            4,
            dtype=torch.float32,
            device="cuda",
        )
        raw_gate = torch.randn(
            1,
            batch,
            heads,
            state_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        raw_beta = torch.randn(1, batch, heads, dtype=torch.float32, device="cuda")
        a_log = torch.randn(heads, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(heads * state_dim, dtype=torch.float32, device="cuda")
        initial_conv_cache = torch.randn(
            block_count,
            3,
            3 * projection_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        initial_state_cache = torch.randn(
            block_count,
            heads,
            state_dim,
            state_dim,
            dtype=torch.float32,
            device="cuda",
        )

        def run_step(
            step_q: torch.Tensor,
            step_k: torch.Tensor,
            step_v: torch.Tensor,
            step_gate: torch.Tensor,
            step_beta: torch.Tensor,
            conv_cache: torch.Tensor,
            state_cache: torch.Tensor,
            step_block_map: torch.Tensor,
            step_lengths: torch.Tensor,
        ) -> torch.Tensor:
            conv_q, conv_k, conv_v = kimi_kda_short_conv_paged_decode(
                step_q,
                step_k,
                step_v,
                conv_weight,
                conv_cache,
                step_block_map,
                step_lengths,
                page_size,
            )
            head_shape = (1, batch, heads, state_dim)
            output, _ = fused_recurrent_kda(
                conv_q.reshape(head_shape),
                conv_k.reshape(head_shape),
                conv_v.reshape(head_shape),
                step_gate,
                step_beta,
                initial_state=state_cache,
                A_log=a_log,
                dt_bias=dt_bias,
                inplace_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=-20.0,
                state_v_first=True,
                cu_seqlens=cu_seqlens,
                block_map=step_block_map,
                seq_size_per_block=page_size,
                sequence_lengths=step_lengths,
            )
            return output

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            run_step(
                q,
                k,
                v,
                raw_gate,
                raw_beta,
                initial_conv_cache.clone(),
                initial_state_cache.clone(),
                block_map,
                lengths_plus_one,
            )
        torch.cuda.current_stream().wait_stream(warmup_stream)

        expected_capture_conv_cache = initial_conv_cache.clone()
        expected_capture_state_cache = initial_state_cache.clone()
        expected_capture_output = run_step(
            q,
            k,
            v,
            raw_gate,
            raw_beta,
            expected_capture_conv_cache,
            expected_capture_state_cache,
            block_map,
            lengths_plus_one,
        )
        graph_conv_cache = initial_conv_cache.clone()
        graph_state_cache = initial_state_cache.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = run_step(
                q,
                k,
                v,
                raw_gate,
                raw_beta,
                graph_conv_cache,
                graph_state_cache,
                block_map,
                lengths_plus_one,
            )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            graph_output, expected_capture_output, rtol=0, atol=0
        )
        torch.testing.assert_close(
            graph_conv_cache, expected_capture_conv_cache, rtol=0, atol=0
        )
        torch.testing.assert_close(
            graph_state_cache, expected_capture_state_cache, rtol=0, atol=0
        )

        replay_lengths = (
            torch.tensor([3, 10], dtype=torch.int32, device="cuda"),
            torch.tensor([9, 17], dtype=torch.int32, device="cuda"),
        )
        for replay_index, next_lengths in enumerate(replay_lengths):
            with self.subTest(replay=replay_index):
                replay_q = torch.randn_like(q)
                replay_k = torch.randn_like(k)
                replay_v = torch.randn_like(v)
                replay_gate = torch.randn_like(raw_gate)
                replay_beta = torch.randn_like(raw_beta)
                replay_block_map = self._block_map(batch, pages)
                if replay_index == 1:
                    replay_block_map[1].zero_()

                expected_conv_cache = initial_conv_cache.clone()
                expected_state_cache = initial_state_cache.clone()
                expected_output = run_step(
                    replay_q,
                    replay_k,
                    replay_v,
                    replay_gate,
                    replay_beta,
                    expected_conv_cache,
                    expected_state_cache,
                    replay_block_map,
                    next_lengths,
                )

                graph_conv_cache.copy_(initial_conv_cache)
                graph_state_cache.copy_(initial_state_cache)
                q.copy_(replay_q)
                k.copy_(replay_k)
                v.copy_(replay_v)
                raw_gate.copy_(replay_gate)
                raw_beta.copy_(replay_beta)
                block_map.copy_(replay_block_map)
                lengths_plus_one.copy_(next_lengths)

                graph.replay()
                torch.cuda.synchronize()

                torch.testing.assert_close(
                    graph_output, expected_output, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    graph_conv_cache, expected_conv_cache, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    graph_state_cache, expected_state_cache, rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
