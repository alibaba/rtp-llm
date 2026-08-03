from unittest import TestCase, main, skipUnless

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


CUDA_AVAILABLE = torch.cuda.is_available()


def _reference(
    sequence_lengths_plus_1: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
):
    safe_page_size = max(page_size, 1)
    sequence_lengths = sequence_lengths_plus_1.clamp_min(1).to(torch.int32)
    max_blocks = block_table.shape[1]
    pages_per_batch = torch.div(
        sequence_lengths + safe_page_size - 1,
        safe_page_size,
        rounding_mode="floor",
    ).clamp_max(max_blocks)
    decode_page_indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            pages_per_batch.cumsum(0).to(torch.int32),
        )
    )
    page_indices = torch.cat(
        [
            block_table[batch, : int(page_count)]
            for batch, page_count in enumerate(pages_per_batch)
        ]
    )
    batch_size = sequence_lengths.numel()
    return {
        "batch_indice_d": torch.arange(batch_size, dtype=torch.int32),
        "page_indice_d": page_indices.to(torch.int32),
        "decode_page_indptr_d": decode_page_indptr,
        "paged_kv_last_page_len_d": ((sequence_lengths - 1) % safe_page_size + 1).to(
            torch.int32
        ),
        "qo_indptr_d": torch.arange(batch_size + 1, dtype=torch.int32),
        "kvlen_d": sequence_lengths,
        "positions_d": sequence_lengths - 1,
    }


@skipUnless(CUDA_AVAILABLE, "requires CUDA")
class FlashInferDecodeParamsTest(TestCase):
    @staticmethod
    def _reserve_decode_buffers(
        params, batch_size: int, max_blocks: int, page_size: int
    ) -> None:
        # The replay-only API deliberately forbids allocations. Mirror the
        # production capture path by reserving its persistent buffers first.
        params.fill_params(
            torch.empty(0, dtype=torch.int32),
            torch.zeros(batch_size, dtype=torch.int32),
            torch.ones(batch_size, dtype=torch.int32),
            torch.zeros((batch_size, max_blocks), dtype=torch.int32),
            max(page_size, 1),
            False,
        )

    def _run_case(
        self,
        batch_size: int,
        page_size: int,
        max_blocks: int,
    ) -> None:
        boundary_lengths = torch.tensor(
            [
                0,
                1,
                max(page_size - 1, 0),
                max(page_size, 0),
                max(page_size + 1, 1),
                max(2 * page_size - 1, 1),
                max(2 * page_size, 1),
                max(2 * page_size + 1, 1),
                65535,
                65536,
                65537,
            ],
            dtype=torch.int32,
        )
        repeats = (
            batch_size + boundary_lengths.numel() - 1
        ) // boundary_lengths.numel()
        sequence_lengths = boundary_lengths.repeat(repeats)[:batch_size]
        block_table = (
            torch.arange(batch_size * max_blocks, dtype=torch.int32).reshape(
                batch_size, max_blocks
            )
            * 17
            + torch.arange(batch_size, dtype=torch.int32).unsqueeze(1) * 13
            + 7
        )
        expected = _reference(sequence_lengths, block_table, page_size)

        params = rtp_llm_ops.FlashInferMlaAttnParams()
        self._reserve_decode_buffers(params, batch_size, max_blocks, page_size)
        params.fill_decode_cuda_graph_params(
            sequence_lengths.cuda(), block_table.cuda(), page_size
        )
        torch.cuda.synchronize()

        valid_pages = int(expected["decode_page_indptr_d"][-1])
        actual = {
            "batch_indice_d": params.batch_indice_d.cpu(),
            "page_indice_d": params.page_indice_d[:valid_pages].cpu(),
            "decode_page_indptr_d": params.decode_page_indptr_d.cpu(),
            "paged_kv_last_page_len_d": params.paged_kv_last_page_len_d.cpu(),
            "qo_indptr_d": params.qo_indptr_d.cpu(),
            "kvlen_d": params.kvlen_d.cpu(),
            "positions_d": params.positions_d.cpu(),
        }
        for name, expected_tensor in expected.items():
            torch.testing.assert_close(actual[name], expected_tensor, rtol=0, atol=0)

    def test_parallel_and_serial_fallback_shapes(self) -> None:
        cases = [
            (1, 128, 513),
            (8, 1, 7),
            (31, 64, 8),
            (32, 128, 8),
            (33, 128, 8),
            (128, 128, 513),
            (1024, 4096, 17),
            (1025, 128, 4),
        ]
        for batch_size, page_size, max_blocks in cases:
            with self.subTest(
                batch_size=batch_size,
                page_size=page_size,
                max_blocks=max_blocks,
            ):
                self._run_case(batch_size, page_size, max_blocks)

    def test_non_positive_page_size_preserves_existing_clamp(self) -> None:
        self._run_case(batch_size=8, page_size=0, max_blocks=7)

    def test_cuda_graph_replay_reads_live_inputs_without_reallocation(self) -> None:
        batch_size = 33
        page_size = 128
        max_blocks = 8
        sequence_lengths = torch.tensor(
            [1 + (batch * 29) % 700 for batch in range(batch_size)],
            dtype=torch.int32,
            device="cuda",
        )
        block_table = torch.arange(
            batch_size * max_blocks, dtype=torch.int32, device="cuda"
        ).reshape(batch_size, max_blocks)
        params = rtp_llm_ops.FlashInferMlaAttnParams()

        # Allocate all persistent buffers before capture.
        self._reserve_decode_buffers(params, batch_size, max_blocks, page_size)
        params.fill_decode_cuda_graph_params(sequence_lengths, block_table, page_size)
        torch.cuda.synchronize()
        output_names = (
            "batch_indice_d",
            "page_indice_d",
            "decode_page_indptr_d",
            "paged_kv_last_page_len_d",
            "qo_indptr_d",
            "kvlen_d",
            "positions_d",
        )
        pointers_before = {
            name: getattr(params, name).data_ptr() for name in output_names
        }

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            params.fill_decode_cuda_graph_params(
                sequence_lengths, block_table, page_size
            )

        for replay in range(10):
            updated_lengths = torch.tensor(
                [1 + (replay * 71 + batch * 43) % 1024 for batch in range(batch_size)],
                dtype=torch.int32,
            )
            updated_blocks = (
                torch.arange(batch_size * max_blocks, dtype=torch.int32).reshape(
                    batch_size, max_blocks
                )
                * (replay + 3)
                + 101
            )
            sequence_lengths.copy_(updated_lengths)
            block_table.copy_(updated_blocks)
            graph.replay()
            torch.cuda.synchronize()

            expected = _reference(updated_lengths, updated_blocks, page_size)
            valid_pages = int(expected["decode_page_indptr_d"][-1])
            actual = {
                "batch_indice_d": params.batch_indice_d.cpu(),
                "page_indice_d": params.page_indice_d[:valid_pages].cpu(),
                "decode_page_indptr_d": params.decode_page_indptr_d.cpu(),
                "paged_kv_last_page_len_d": params.paged_kv_last_page_len_d.cpu(),
                "qo_indptr_d": params.qo_indptr_d.cpu(),
                "kvlen_d": params.kvlen_d.cpu(),
                "positions_d": params.positions_d.cpu(),
            }
            for name, expected_tensor in expected.items():
                torch.testing.assert_close(
                    actual[name], expected_tensor, rtol=0, atol=0
                )

        self.assertEqual(
            pointers_before,
            {name: getattr(params, name).data_ptr() for name in output_names},
        )

    def test_rejects_invalid_inputs(self) -> None:
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        sequence_lengths = torch.ones(2, dtype=torch.int32, device="cuda")
        block_table = torch.ones((2, 4), dtype=torch.int32, device="cuda")

        invalid_calls = [
            (sequence_lengths.cpu(), block_table, 128),
            (sequence_lengths.to(torch.int64), block_table, 128),
            (sequence_lengths, block_table.cpu(), 128),
            (sequence_lengths, block_table.to(torch.int64), 128),
            (sequence_lengths, block_table.reshape(-1), 128),
            (sequence_lengths, block_table[:1], 128),
        ]
        for sequence_arg, block_arg, page_size in invalid_calls:
            with self.subTest(
                sequence_device=sequence_arg.device,
                sequence_dtype=sequence_arg.dtype,
                block_device=block_arg.device,
                block_dtype=block_arg.dtype,
                block_shape=tuple(block_arg.shape),
            ):
                with self.assertRaises(RuntimeError):
                    params.fill_decode_cuda_graph_params(
                        sequence_arg, block_arg, page_size
                    )


if __name__ == "__main__":
    main()
