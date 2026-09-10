import unittest

import torch


def _op_available() -> bool:
    try:
        from rtp_llm.ops.compute_ops import rtp_llm_ops
    except Exception:
        return False
    return hasattr(rtp_llm_ops, "minimax_decode_topk")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(_op_available(), "minimax_decode_topk op is unavailable")
class PagedIndexScoreSeqStrideTest(unittest.TestCase):
    def test_strided_request_final_lengths_match_contiguous(self) -> None:
        from rtp_llm.models_py.triton_kernels.sparse_msa.decode.flash_with_topk_idx import (
            flash_decode_with_topk_idx_paged,
        )

        block_size = 128
        head_dim = 64
        batch_size = 4
        max_blocks = 8
        topk = 2

        # Later logical blocks have strictly larger scores, so reading the
        # wrong request length changes the selected top-k set deterministically.
        k_paged = torch.stack(
            [
                torch.full(
                    (block_size, head_dim),
                    float(block + 1),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                for block in range(max_blocks)
            ]
        )
        block_table = torch.arange(max_blocks, device="cuda", dtype=torch.int32).repeat(
            batch_size, 1
        )

        for verify_width in (2, 3, 4):
            with self.subTest(verify_width=verify_width):
                q = torch.ones(
                    batch_size * verify_width,
                    1,
                    head_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                final_lens = (
                    torch.tensor([3, 4, 5, 6], device="cuda", dtype=torch.int32)
                    * block_size
                )
                offsets = torch.arange(
                    verify_width - 1, -1, -1, device="cuda", dtype=torch.int32
                )
                token_seq_lens = (final_lens[:, None] - offsets[None, :]).reshape(-1)
                request_final_lens = token_seq_lens.view(batch_size, verify_width)[
                    :, -1
                ]
                self.assertEqual(request_final_lens.stride(), (verify_width,))

                kwargs = dict(
                    q=q,
                    k_paged=k_paged,
                    block_table=block_table,
                    max_seqlen=int(final_lens.max().item()),
                    block_size=block_size,
                    topk=topk,
                    init_blocks=0,
                    local_blocks=0,
                    score_type="max",
                    decode_query_len=verify_width,
                    token_seq_lens=token_seq_lens,
                )
                _, actual = flash_decode_with_topk_idx_paged(
                    seq_lens=request_final_lens, **kwargs
                )
                _, control = flash_decode_with_topk_idx_paged(
                    seq_lens=request_final_lens.contiguous(), **kwargs
                )
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(actual.cpu(), control.cpu()),
                    f"strided seq_lens mismatch for verify_width={verify_width}",
                )


if __name__ == "__main__":
    unittest.main()
