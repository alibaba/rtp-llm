import logging
import unittest
from typing import List

import torch

from rtp_llm.ops.compute_ops import (
    cuda_graph_copy_large2small,
    cuda_graph_copy_small2large,
)


class CudaGraphCopyKernelTest(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        torch.cuda.set_device(0)
        self.device = torch.device("cuda")

    def _calculate_cu_seq_len(self, input_lengths: List[int]) -> List[int]:
        """Calculate cumulative sequence lengths"""
        cu_seq_len = [0]
        for length in input_lengths:
            cu_seq_len.append(cu_seq_len[-1] + length)
        return cu_seq_len

    def _test_cuda_graph_copy_small2large(self, dtype: torch.dtype):
        """Test copying from compact format to aligned format"""
        logging.info(
            f"================== testCudaGraphCopySmall2Large<{dtype}> =================="
        )

        batch_size = 64
        max_seq_len = 512
        hidden_size = 768
        max_batch_size = 64

        # Input lengths for each batch
        input_lengths = [
            114,
            181,
            148,
            117,
            132,
            127,
            134,
            84,
            121,
            107,
            151,
            191,
            107,
            175,
            172,
            107,
            103,
            82,
            123,
            109,
            128,
            115,
            153,
            128,
            122,
            164,
            165,
            158,
            100,
            142,
            144,
            155,
            97,
            100,
            191,
            183,
            136,
            89,
            136,
            149,
            104,
            130,
            162,
            102,
            191,
            98,
            111,
            115,
            96,
            151,
            100,
            95,
            96,
            108,
            97,
            134,
            159,
            86,
            108,
            99,
            102,
            75,
            99,
            125,
        ]

        input_lengths = [length + 13 for length in input_lengths]

        # Calculate total compact size
        total_token_sum = sum(input_lengths)
        total_compact_size = total_token_sum * hidden_size

        # Allocate host memory and initialize input data
        h_input_compact = torch.randn(
            total_token_sum, hidden_size, dtype=dtype, device="cpu"
        )

        # Calculate expected output manually using vectorized operations
        h_expected = torch.zeros(
            max_batch_size * max_seq_len, hidden_size, dtype=dtype, device="cpu"
        )
        offset = 0
        for b in range(batch_size):
            seq_len = input_lengths[b]
            # Copy entire sequence at once (vectorized)
            h_expected[b * max_seq_len : b * max_seq_len + seq_len, :] = (
                h_input_compact[offset : offset + seq_len, :]
            )
            offset += seq_len
        # Calculate cu_seq_len
        cu_seq_len = self._calculate_cu_seq_len(input_lengths)

        # Create device tensors
        d_input_compact = h_input_compact.to(self.device)
        d_output_aligned = torch.zeros(
            max_batch_size * max_seq_len, hidden_size, dtype=dtype, device=self.device
        )
        d_input_lengths = torch.tensor(
            input_lengths, dtype=torch.int32, device=self.device
        )
        # batch_size and cu_seq_len should be CPU pinned memory for CUDA Graph
        d_batch_size = torch.tensor([batch_size], dtype=torch.int32).pin_memory()
        d_cu_seq_len = torch.tensor(cu_seq_len, dtype=torch.int32).pin_memory()
        # Warm up
        cuda_graph_copy_small2large(
            d_input_compact,
            d_output_aligned,
            d_batch_size,
            max_batch_size,
            max_seq_len,
            d_input_lengths,
            hidden_size,
            d_cu_seq_len,
        )
        # Synchronize before timing
        torch.cuda.synchronize()

        # Measure execution time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        cuda_graph_copy_small2large(
            d_input_compact,
            d_output_aligned,
            d_batch_size,
            max_batch_size,
            max_seq_len,
            d_input_lengths,
            hidden_size,
            d_cu_seq_len,
        )
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event)

        # Copy result back
        h_output_aligned = d_output_aligned.cpu()

        # Verify results - compare valid parts of aligned tensors
        tolerance = 1e-5
        # Extract valid outputs from both actual and expected aligned tensors
        valid_actual = []
        valid_expected = []
        for b in range(batch_size):
            batch_start = b * max_seq_len
            batch_end = batch_start + input_lengths[b]
            valid_actual.append(h_output_aligned[batch_start:batch_end])
            valid_expected.append(h_expected[batch_start:batch_end])

        if valid_actual:
            actual_compact = torch.cat(valid_actual, dim=0)
            expected_compact = torch.cat(valid_expected, dim=0)
            torch.testing.assert_close(
                actual_compact,
                expected_compact,
                rtol=tolerance,
                atol=tolerance,
                msg="Small2Large copy mismatch",
            )

        logging.info(f"Small2Large copy completed in {elapsed_time:.6f} ms")

    def _test_cuda_graph_copy_large2small(self, dtype: torch.dtype):
        """Test copying from aligned format to compact format"""
        logging.info(
            f"================== testCudaGraphCopyLarge2Small<{dtype}> =================="
        )

        batch_size = 64
        max_seq_len = 512
        hidden_size = 768
        max_batch_size = 64

        # Input lengths for each batch
        input_lengths = [
            114,
            181,
            148,
            117,
            132,
            127,
            134,
            84,
            121,
            107,
            151,
            191,
            107,
            175,
            172,
            107,
            103,
            82,
            123,
            109,
            128,
            115,
            153,
            128,
            122,
            164,
            165,
            158,
            100,
            142,
            144,
            155,
            97,
            100,
            191,
            183,
            136,
            89,
            136,
            149,
            104,
            130,
            162,
            102,
            191,
            98,
            111,
            115,
            96,
            151,
            100,
            95,
            96,
            108,
            97,
            134,
            159,
            86,
            108,
            99,
            102,
            75,
            99,
            125,
        ]
        # Add 13 to each length
        input_lengths = [length + 13 for length in input_lengths]

        # Calculate total compact size
        total_token_sum = sum(input_lengths)
        total_compact_size = total_token_sum * hidden_size

        # Allocate host memory and initialize input data using vectorized operations
        h_input_aligned = torch.zeros(
            max_batch_size * max_seq_len, hidden_size, dtype=dtype, device="cpu"
        )

        # Fill only the valid sequence lengths for each batch (vectorized)
        for b in range(batch_size):
            seq_len = input_lengths[b]
            h_input_aligned[b * max_seq_len : b * max_seq_len + seq_len, :] = (
                torch.randn(seq_len, hidden_size, dtype=dtype)
            )

        # Calculate expected output manually using vectorized operations
        h_expected = torch.zeros(
            total_token_sum, hidden_size, dtype=dtype, device="cpu"
        )
        offset = 0
        for b in range(batch_size):
            seq_len = input_lengths[b]
            # Copy from aligned to compact (vectorized)
            h_expected[offset : offset + seq_len, :] = h_input_aligned[
                b * max_seq_len : b * max_seq_len + seq_len, :
            ]
            offset += seq_len

        # Calculate cu_seq_len
        cu_seq_len = self._calculate_cu_seq_len(input_lengths)

        # Create device tensors
        d_input_aligned = h_input_aligned.to(self.device)
        d_output_compact = torch.zeros(
            total_token_sum, hidden_size, dtype=dtype, device=self.device
        )
        d_input_lengths = torch.tensor(
            input_lengths, dtype=torch.int32, device=self.device
        )
        # batch_size and cu_seq_len should be CPU pinned memory for CUDA Graph
        d_batch_size = torch.tensor([batch_size], dtype=torch.int32).pin_memory()
        d_cu_seq_len = torch.tensor(cu_seq_len, dtype=torch.int32).pin_memory()

        # Measure execution time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        cuda_graph_copy_large2small(
            d_input_aligned,
            d_output_compact,
            d_batch_size,
            max_batch_size,
            max_seq_len,
            d_input_lengths,
            hidden_size,
            d_cu_seq_len,
        )
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event)

        # Copy result back
        h_output_compact = d_output_compact.cpu()

        # Verify results
        tolerance = 1e-5
        torch.testing.assert_close(
            h_output_compact,
            h_expected,
            rtol=tolerance,
            atol=tolerance,
            msg="Large2Small copy mismatch",
        )

        logging.info(f"Large2Small copy completed in {elapsed_time:.6f} ms")

    def _test_cuda_graph_copy_round_trip(self, dtype: torch.dtype):
        """Test round trip copy: Small2Large -> Large2Small"""
        logging.info(
            f"================== testCudaGraphCopyRoundTrip<{dtype}> =================="
        )

        batch_size = 64
        max_seq_len = 512
        hidden_size = 768
        max_batch_size = 64

        # Input lengths for each batch
        input_lengths = [
            114,
            181,
            148,
            117,
            132,
            127,
            134,
            84,
            121,
            107,
            151,
            191,
            107,
            175,
            172,
            107,
            103,
            82,
            123,
            109,
            128,
            115,
            153,
            128,
            122,
            164,
            165,
            158,
            100,
            142,
            144,
            155,
            97,
            100,
            191,
            183,
            136,
            89,
            136,
            149,
            104,
            130,
            162,
            102,
            191,
            98,
            111,
            115,
            96,
            151,
            100,
            95,
            96,
            108,
            97,
            134,
            159,
            86,
            108,
            99,
            102,
            75,
            99,
            125,
        ]

        input_lengths = [length + 13 for length in input_lengths]

        # Calculate total compact size
        total_token_sum = sum(input_lengths)
        total_compact_size = total_token_sum * hidden_size

        # Allocate host memory and initialize input data
        h_input_compact = torch.randn(
            total_token_sum, hidden_size, dtype=dtype, device="cpu"
        )
        h_expected = h_input_compact.clone()  # Expected output should be same as input

        # Calculate cu_seq_len
        cu_seq_len = self._calculate_cu_seq_len(input_lengths)

        # Create device tensors
        d_input_compact = h_input_compact.to(self.device)
        d_intermediate_aligned = torch.zeros(
            max_batch_size * max_seq_len, hidden_size, dtype=dtype, device=self.device
        )
        d_output_compact = torch.zeros(
            total_token_sum, hidden_size, dtype=dtype, device=self.device
        )
        d_input_lengths = torch.tensor(
            input_lengths, dtype=torch.int32, device=self.device
        )
        # batch_size and cu_seq_len should be CPU pinned memory for CUDA Graph
        d_batch_size = torch.tensor([batch_size], dtype=torch.int32).pin_memory()
        d_cu_seq_len = torch.tensor(cu_seq_len, dtype=torch.int32).pin_memory()

        # Measure execution time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        # Launch kernels: Small2Large -> Large2Small
        cuda_graph_copy_small2large(
            d_input_compact,
            d_intermediate_aligned,
            d_batch_size,
            max_batch_size,
            max_seq_len,
            d_input_lengths,
            hidden_size,
            d_cu_seq_len,
        )
        cuda_graph_copy_large2small(
            d_intermediate_aligned,
            d_output_compact,
            d_batch_size,
            max_batch_size,
            max_seq_len,
            d_input_lengths,
            hidden_size,
            d_cu_seq_len,
        )
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event)

        # Copy result back
        h_output_compact = d_output_compact.cpu()

        # Verify results
        tolerance = 1e-5
        torch.testing.assert_close(
            h_output_compact,
            h_expected,
            rtol=tolerance,
            atol=tolerance,
            msg="Round trip copy mismatch",
        )

        logging.info(f"Round trip copy completed in {elapsed_time:.3f} ms")

    def test_small2large_bfloat16(self):
        self._test_cuda_graph_copy_small2large(torch.bfloat16)

    def test_large2small_bfloat16(self):
        self._test_cuda_graph_copy_large2small(torch.bfloat16)

    def test_round_trip_bfloat16(self):
        self._test_cuda_graph_copy_round_trip(torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
