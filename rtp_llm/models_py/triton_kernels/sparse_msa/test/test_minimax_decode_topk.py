import unittest

import torch


def _op_available() -> bool:
    try:
        from rtp_llm.ops.compute_ops import rtp_llm_ops
    except Exception:
        return False
    return hasattr(rtp_llm_ops, "minimax_decode_topk")


def _reference_topk(
    score: torch.Tensor, seq_lens: torch.Tensor, block_size: int, topk: int
) -> torch.Tensor:
    num_heads, batch_size, max_seqblock = score.shape
    ref = torch.full(
        (num_heads, batch_size, topk),
        -1,
        device=score.device,
        dtype=torch.int32,
    )
    seq_lens_cpu = seq_lens.cpu().tolist()
    for h in range(num_heads):
        for b in range(batch_size):
            num_blocks = (int(seq_lens_cpu[b]) + block_size - 1) // block_size
            num_blocks = min(max(num_blocks, 0), max_seqblock)
            if num_blocks <= 0:
                continue
            if num_blocks <= topk:
                ref[h, b, :num_blocks] = torch.arange(
                    num_blocks, device=score.device, dtype=torch.int32
                )
                continue
            values = torch.nan_to_num(
                score[h, b, :num_blocks].float(), nan=float("-inf")
            )
            ref[h, b] = torch.topk(values, k=topk, sorted=True).indices.to(torch.int32)
    return ref


def _assert_same_topk_set(
    case: unittest.TestCase, actual: torch.Tensor, expected: torch.Tensor
) -> None:
    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()
    for h in range(actual_cpu.shape[0]):
        for b in range(actual_cpu.shape[1]):
            actual_valid = actual_cpu[h, b][actual_cpu[h, b] >= 0].sort().values
            expected_valid = expected_cpu[h, b][expected_cpu[h, b] >= 0].sort().values
            case.assertTrue(
                torch.equal(actual_valid, expected_valid),
                f"mismatch at head={h}, batch={b}: "
                f"actual={actual_cpu[h, b].tolist()} expected={expected_cpu[h, b].tolist()}",
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(_op_available(), "minimax_decode_topk op is unavailable")
class MinimaxDecodeTopkTest(unittest.TestCase):
    def _run_case(
        self,
        num_heads: int,
        batch_size: int,
        max_seqblock: int,
        block_size: int,
        topk: int,
        seq_dtype: torch.dtype,
    ) -> None:
        torch.manual_seed(20260706 + max_seqblock + topk)
        score = torch.randn(
            num_heads, batch_size, max_seqblock, device="cuda", dtype=torch.float32
        )
        # Make scores unique so exact topk membership is deterministic.
        score += torch.arange(max_seqblock, device="cuda", dtype=torch.float32) * 1e-5
        block_counts = torch.linspace(
            1, max_seqblock, batch_size, device="cuda", dtype=torch.int64
        )
        seq_lens = (block_counts * block_size - block_size // 2).to(seq_dtype)
        expected = _reference_topk(score, seq_lens, block_size, topk)

        from rtp_llm.ops.compute_ops import rtp_llm_ops

        actual = torch.empty_like(expected)
        rtp_llm_ops.minimax_decode_topk(score, seq_lens, actual, block_size, topk)
        torch.cuda.synchronize()
        _assert_same_topk_set(self, actual, expected)

    def test_shapes_and_seq_len_dtypes(self) -> None:
        cases = (
            (1, 1, 1, 128, 1, torch.int32),
            (2, 3, 8, 128, 4, torch.int32),
            (4, 2, 65, 128, 8, torch.int64),
            (8, 4, 513, 128, 16, torch.int32),
            (8, 3, 4096, 16, 32, torch.int64),
        )
        for case in cases:
            with self.subTest(case=case):
                self._run_case(*case)

    def test_num_blocks_le_topk_and_nan_scores(self) -> None:
        block_size = 128
        topk = 6
        score = torch.tensor(
            [[[float("nan"), 5.0, 5.0, 4.0, float("-inf"), 3.0, 2.0, 1.0]]],
            device="cuda",
            dtype=torch.float32,
        )
        seq_lens = torch.tensor([4 * block_size - 1], device="cuda", dtype=torch.int32)
        expected = _reference_topk(score, seq_lens, block_size, topk)

        from rtp_llm.ops.compute_ops import rtp_llm_ops

        actual = torch.empty_like(expected)
        rtp_llm_ops.minimax_decode_topk(
            score.contiguous(), seq_lens, actual, block_size, topk
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(actual.cpu(), expected.cpu()))


if __name__ == "__main__":
    unittest.main()
