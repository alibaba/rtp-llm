import unittest

from rtp_llm.ops import cpp_get_block_cache_keys
from rtp_llm.server.cache_key_routing import (
    get_block_cache_keys,
    route_cache_keys_for_page_rr,
)


def _signed(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >= (1 << (bits - 1)) else value


def _hash_word(previous, value):
    # HashUtil.h: std::hash<int32_t>, then Jenkins' signed int64 rolling state.
    value = _signed(value, 32)
    return _signed(
        previous ^ (value + 0x9E3779B97F4A7C15 + (previous << 12) + (previous >> 32)),
        64,
    )


def _backend_full_keys(tokens, block_size, seed=0, spans=()):
    """Backend contract: tokens first, then overlapping image identities per page."""
    keys, rolling = [], seed
    for begin in range(0, len(tokens) - block_size + 1, block_size):
        for word in tokens[begin : begin + block_size]:
            rolling = _hash_word(rolling, word)
        for start, rows in spans:
            first = max(start, begin)
            last = min(start + len(rows), begin + block_size)
            if first < last:
                words = [-42, first - begin, len(rows), first - start, last - first]
                words += rows[first - start : last - start]
                for word in words:
                    rolling = _hash_word(rolling, word)
        keys.append(rolling)
    return keys


class CacheKeyRoutingTest(unittest.TestCase):
    def test_text_hashes_match_backend_complete_prefix_before_cp_projection(self):
        tokens = [(i * 997 + 31) % 129280 for i in range(2305)]
        tokens[17:20] = [-1, -(1 << 31), (1 << 31) - 1]
        for block in (128, 256):
            for cp in (1, 2, 4):
                virtual = block * cp
                lengths = sorted(
                    {
                        0,
                        1,
                        127,
                        128,
                        511,
                        512,
                        513,
                        1023,
                        1024,
                        block - 1,
                        block,
                        block + 1,
                        virtual - 1,
                        virtual,
                        virtual + 1,
                        2 * virtual + 17,
                    }
                )
                for sharded in (False, True):
                    for length in lengths:
                        with self.subTest(
                            block=block, cp=cp, sharded=sharded, length=length
                        ):
                            original = tokens[:length]
                            full = get_block_cache_keys(
                                original, block, cache_key_seed=41
                            )
                            expected = _backend_full_keys(original, block, seed=41)
                            self.assertEqual(full, expected)
                            selected = expected[cp - 1 :: cp] if sharded else expected
                            self.assertEqual(
                                route_cache_keys_for_page_rr(full, sharded, cp),
                                selected,
                            )
                            self.assertEqual(original, tokens[:length])

    def test_row_identity_matches_complete_token_ids_cpp_fixture(self):
        tokens = list(range(16))
        rows = [97, 98, 63, 112, 58, 229, 155, 190]
        spans = [(4, rows)]
        expected = _backend_full_keys(tokens, 4, seed=41, spans=spans)
        self.assertEqual(
            get_block_cache_keys(
                tokens, 4, cache_key_seed=41, multimodal_cache_spans=spans
            ),
            expected,
        )
        self.assertEqual(
            cpp_get_block_cache_keys(
                [
                    tokens[:4],
                    tokens[4:8] + [-42, 0, 8, 0, 4] + rows[:4],
                    tokens[8:12] + [-42, 0, 8, 4, 4] + rows[4:],
                    tokens[12:],
                ],
                41,
            ),
            expected,
        )

    def test_images_cross_physical_and_cp_boundaries_in_original_order(self):
        tokens = [(i * 17) % 129280 for i in range(2201)]
        for block in (128, 256):
            spans = [
                (block - 3, [i * 31 - 17 for i in range(block + 9)]),
                (3 * block + 1, [i * 43 - 3 for i in range(2 * block + 1)]),
            ]
            full = get_block_cache_keys(
                tokens, block, cache_key_seed=41, multimodal_cache_spans=spans
            )
            expected = _backend_full_keys(tokens, block, 41, spans)
            self.assertEqual(full, expected)
            for cp in (1, 2, 4):
                for sharded in (False, True):
                    with self.subTest(block=block, cp=cp, sharded=sharded):
                        self.assertEqual(
                            route_cache_keys_for_page_rr(full, sharded, cp),
                            expected[cp - 1 :: cp] if sharded else expected,
                        )

    def test_changed_image_rows_change_all_later_prefix_keys(self):
        tokens = list(range(1024))
        rows = list(range(32))
        original = get_block_cache_keys(
            tokens, 128, cache_key_seed=41, multimodal_cache_spans=[(128, rows)]
        )
        rows[0] += 1
        changed = get_block_cache_keys(
            tokens, 128, cache_key_seed=41, multimodal_cache_spans=[(128, rows)]
        )
        self.assertEqual(original[0], changed[0])
        self.assertTrue(all(a != b for a, b in zip(original[1:], changed[1:])))
        self.assertNotEqual(
            route_cache_keys_for_page_rr(original, True, 4),
            route_cache_keys_for_page_rr(changed, True, 4),
        )


if __name__ == "__main__":
    unittest.main()
