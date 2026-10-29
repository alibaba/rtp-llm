import unittest

from rtp_llm.utils.token_capacity import (
    align_up,
    cp_padded_batch_tokens_per_rank_bound,
    cp_padded_tokens_per_rank_bound,
    sp_padded_tokens,
)


class TokenCapacityTest(unittest.TestCase):
    def test_sp_padding_preserves_complete_requests(self):
        for tp in (1, 2, 3, 4, 8, 16):
            for width in (1, 2, 3, 4, 5, 8):
                for batch in range(33):
                    padded_batch = batch
                    while padded_batch * width % tp:
                        padded_batch += 1
                    self.assertEqual(
                        sp_padded_tokens(batch, width, tp), padded_batch * width
                    )
        # Rounding the product alone would cut the last width-5 request.
        self.assertEqual(sp_padded_tokens(17, 5, 8), 120)
        self.assertEqual(align_up(17 * 5, 8), 88)

    def test_cp_single_sequence_includes_tail_padding(self):
        for cp in (1, 2, 3, 4, 8):
            for length in (*range(33), 200002):
                padded = length
                if cp > 1:
                    while padded % (2 * cp):
                        padded += 1
                self.assertEqual(
                    cp_padded_tokens_per_rank_bound(length, cp), padded // cp
                )

    def test_cp_ragged_batch_bound_matches_exhaustive_layouts(self):
        def layouts(tokens, rows):
            if tokens == 0:
                yield ()
            elif rows > 0:
                for length in range(1, tokens + 1):
                    for tail in layouts(tokens - length, rows - 1):
                        yield (length, *tail)

        for cp in (1, 2, 3, 4, 8):
            for budget in range(10):
                for rows in (1, 2, 3, 10):
                    sizes = []
                    for lengths in layouts(budget, rows):
                        local_rows = 0
                        for length in lengths:
                            if cp > 1:
                                while length % (2 * cp):
                                    length += 1
                            local_rows += length // cp
                        sizes.append(local_rows)
                    self.assertEqual(
                        cp_padded_batch_tokens_per_rank_bound(budget, cp, rows),
                        max(sizes),
                    )

    def test_cp_empty_and_disabled_bounds(self):
        for cp in (-1, 0, 1, 4):
            self.assertEqual(cp_padded_tokens_per_rank_bound(-1, cp), 0)
            self.assertEqual(cp_padded_batch_tokens_per_rank_bound(0, cp, 0), 0)
        for cp in (-1, 0, 1):
            self.assertEqual(cp_padded_tokens_per_rank_bound(7, cp), 7)
            self.assertEqual(cp_padded_batch_tokens_per_rank_bound(7, cp, 3), 7)


if __name__ == "__main__":
    unittest.main()
