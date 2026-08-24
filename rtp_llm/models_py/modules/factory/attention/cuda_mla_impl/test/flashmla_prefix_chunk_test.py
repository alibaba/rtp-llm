import unittest

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_prefix_chunk import (
    plan_flashmla_prefix_chunks,
)


class FlashMLAPrefixChunkPlannerTest(unittest.TestCase):
    def test_prefix_page_and_chunk_boundaries(self) -> None:
        chunk_tokens = 16384
        page_size = 128
        for prefix_len in (
            0,
            1,
            page_size - 1,
            page_size,
            page_size + 1,
            chunk_tokens - 1,
            chunk_tokens,
            chunk_tokens + 1,
            2 * chunk_tokens - 1,
            2 * chunk_tokens,
            2 * chunk_tokens + 1,
        ):
            with self.subTest(prefix_len=prefix_len):
                chunks = plan_flashmla_prefix_chunks(
                    [17],
                    [prefix_len],
                    chunk_tokens=chunk_tokens,
                    page_size=page_size,
                )
                expected_chunks = (
                    0
                    if prefix_len == 0
                    else (prefix_len + chunk_tokens - 1) // chunk_tokens
                )
                self.assertEqual(len(chunks), expected_chunks)
                self.assertEqual(sum(chunk.kv_tokens for chunk in chunks), prefix_len)
                self.assertTrue(
                    all(chunk.kv_tokens <= chunk_tokens for chunk in chunks)
                )
                self.assertTrue(
                    all(
                        start % page_size == 0
                        for chunk in chunks
                        for start in chunk.prefix_starts
                    )
                )

    def test_query_boundaries_do_not_change_prefix_partition(self) -> None:
        for q_len in (1, 17, 4096, 65536):
            with self.subTest(q_len=q_len):
                chunks = plan_flashmla_prefix_chunks(
                    [q_len],
                    [16385],
                    chunk_tokens=16384,
                    page_size=128,
                )
                self.assertEqual([chunk.kv_tokens for chunk in chunks], [16384, 1])
                self.assertEqual([chunk.q_start for chunk in chunks], [0, 0])
                self.assertEqual([chunk.q_tokens for chunk in chunks], [q_len, q_len])

    def test_mixed_batch_zero_prefix_gaps_preserve_query_ranges(self) -> None:
        chunks = plan_flashmla_prefix_chunks(
            [1, 17, 257, 4096],
            [1, 0, 16384, 16385],
            chunk_tokens=16384,
            page_size=128,
        )
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].request_indices, (0,))
        self.assertEqual((chunks[0].q_start, chunks[0].q_tokens), (0, 1))
        self.assertEqual(chunks[1].request_indices, (2,))
        self.assertEqual((chunks[1].q_start, chunks[1].q_tokens), (18, 257))
        self.assertEqual(chunks[2].request_indices, (3,))
        self.assertEqual(chunks[2].prefix_starts, (0,))
        self.assertEqual(chunks[3].request_indices, (3,))
        self.assertEqual(chunks[3].prefix_starts, (16384,))
        self.assertEqual((chunks[3].q_start, chunks[3].q_tokens), (275, 4096))

    def test_packed_mixed_prefixes(self) -> None:
        chunks = plan_flashmla_prefix_chunks(
            [17, 33, 65, 9],
            [0, 4096, 8192, 4096],
            chunk_tokens=8192,
            page_size=128,
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].request_indices, (1, 2))
        self.assertEqual(chunks[0].prefix_starts, (0, 0))
        self.assertEqual(chunks[0].prefix_lens, (4096, 4096))
        self.assertEqual(chunks[0].q_start, 17)
        self.assertEqual(chunks[0].q_tokens, 98)
        self.assertEqual(chunks[1].request_indices, (2, 3))
        self.assertEqual(chunks[1].prefix_starts, (4096, 0))
        self.assertEqual(chunks[1].prefix_lens, (4096, 4096))
        self.assertEqual(chunks[1].q_start, 50)
        self.assertEqual(chunks[1].q_tokens, 74)

    def test_non_page_aligned_tail(self) -> None:
        chunks = plan_flashmla_prefix_chunks(
            [1, 17, 65],
            [257, 0, 4097],
            chunk_tokens=256,
            page_size=128,
        )
        self.assertEqual(
            [chunk.kv_tokens for chunk in chunks], [256, 1] + [256] * 16 + [1]
        )
        self.assertEqual(chunks[1].prefix_starts, (256,))
        self.assertEqual(chunks[-1].prefix_starts, (4096,))

    def test_one_million_prefix_coverage_and_int32_range(self) -> None:
        prefix_len = 1024 * 1024
        chunks = plan_flashmla_prefix_chunks(
            [65536],
            [prefix_len],
            chunk_tokens=16384,
            page_size=4096,
        )
        self.assertEqual(len(chunks), 64)
        cursor = 0
        for chunk in chunks:
            self.assertEqual(chunk.request_indices, (0,))
            self.assertEqual(chunk.prefix_starts, (cursor,))
            self.assertLessEqual(chunk.kv_tokens, 16384)
            cursor += chunk.kv_tokens
            self.assertLess(cursor, 2**31)
        self.assertEqual(cursor, prefix_len)

    def test_disabled_and_invalid_capacity(self) -> None:
        self.assertEqual(
            plan_flashmla_prefix_chunks(
                [65536], [1024 * 1024], chunk_tokens=0, page_size=4096
            ),
            (),
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            plan_flashmla_prefix_chunks([1], [4096], chunk_tokens=5000, page_size=4096)
        with self.assertRaisesRegex(ValueError, "matching non-empty"):
            plan_flashmla_prefix_chunks([], [], chunk_tokens=16384, page_size=4096)
        with self.assertRaisesRegex(ValueError, "matching non-empty"):
            plan_flashmla_prefix_chunks([1, 2], [1], chunk_tokens=16384, page_size=4096)
        with self.assertRaisesRegex(ValueError, "Q lengths must be positive"):
            plan_flashmla_prefix_chunks([0], [1], chunk_tokens=16384, page_size=4096)
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            plan_flashmla_prefix_chunks([1], [-1], chunk_tokens=16384, page_size=4096)
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            plan_flashmla_prefix_chunks([1], [1], chunk_tokens=-1, page_size=4096)
        with self.assertRaisesRegex(ValueError, "page size must be positive"):
            plan_flashmla_prefix_chunks([1], [1], chunk_tokens=1, page_size=0)


if __name__ == "__main__":
    unittest.main()
