import unittest
from types import SimpleNamespace

import torch

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


def _backend_full_keys(tokens, block_size, seed=0, identities=()):
    """Backend contract: tokens first, then overlapping image identities per page."""
    keys, rolling = [], seed
    for begin in range(0, len(tokens) - block_size + 1, block_size):
        for word in tokens[begin : begin + block_size]:
            rolling = _hash_word(rolling, word)
        for start, end, words in identities:
            if end > begin and start < begin + block_size:
                for word in words:
                    rolling = _hash_word(rolling, word)
        keys.append(rolling)
    return keys


def _image(start, count, content="ab?", processor="p:图"):
    return SimpleNamespace(
        start=start,
        types=torch.zeros(count, dtype=torch.int32),
        n_vit_h=2,
        n_vit_w=2,
        content_sha256=content,
        processor_identity=processor,
    )


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

    def test_image_identity_matches_complete_token_ids_cpp_fixture(self):
        # Shared fixture with CompleteTokenIdsTest.ImageHashIdentityUsesLengthDelimitedBytes.
        # Non-hex and UTF-8 also prevent treating arbitrary RPC identities as hex digests.
        tokens = list(range(16))
        image = _image(4, 8)
        words = [-41, 4, 12, 2, 2, 3, 97, 98, 63, 5, 112, 58, 229, 155, 190]
        expected = _backend_full_keys(tokens, 4, seed=41, identities=[(4, 12, words)])
        self.assertEqual(
            expected,
            [
                556723330460289837,
                1094575162337917127,
                -8416381023089338704,
                2888375575027288070,
            ],
        )
        self.assertEqual(
            get_block_cache_keys(tokens, 4, SimpleNamespace(images=[image]), 41),
            expected,
        )
        # Exercise the actual native hash implementation in addition to the independent oracle.
        self.assertEqual(
            cpp_get_block_cache_keys(
                [tokens[:4], tokens[4:8] + words, tokens[8:12] + words, tokens[12:]], 41
            ),
            expected,
        )

    def test_images_cross_physical_and_cp_boundaries_in_original_order(self):
        tokens = [(i * 17) % 129280 for i in range(2201)]
        for block in (128, 256):
            first = _image(block - 3, block + 9)
            second = _image(3 * block + 1, 2 * block + 1, content="x", processor="v2")
            images = [first, second]
            identities = [
                (
                    block - 3,
                    2 * block + 6,
                    [
                        -41,
                        block - 3,
                        2 * block + 6,
                        2,
                        2,
                        3,
                        97,
                        98,
                        63,
                        5,
                        112,
                        58,
                        229,
                        155,
                        190,
                    ],
                ),
                (
                    3 * block + 1,
                    5 * block + 2,
                    [-41, 3 * block + 1, 5 * block + 2, 2, 2, 1, 120, 2, 118, 50],
                ),
            ]
            full = get_block_cache_keys(
                tokens, block, SimpleNamespace(images=images), 41
            )
            expected = _backend_full_keys(tokens, block, 41, identities)
            self.assertEqual(full, expected)
            for cp in (1, 2, 4):
                for sharded in (False, True):
                    with self.subTest(block=block, cp=cp, sharded=sharded):
                        self.assertEqual(
                            route_cache_keys_for_page_rr(full, sharded, cp),
                            expected[cp - 1 :: cp] if sharded else expected,
                        )

    def test_unknown_image_identity_only_routes_complete_safe_prefix(self):
        tokens = list(range(1025))
        for missing in ("content_sha256", "processor_identity"):
            image = _image(513, 32)
            setattr(image, missing, "")
            full = get_block_cache_keys(
                tokens, 128, SimpleNamespace(images=[image]), 41
            )
            self.assertEqual(full, _backend_full_keys(tokens[:513], 128, 41))
            self.assertEqual(route_cache_keys_for_page_rr(full, True, 4), [full[3]])

    def test_image_identity_changes_all_later_prefix_keys(self):
        tokens = list(range(1024))
        image = _image(128, 32)
        original = get_block_cache_keys(
            tokens, 128, SimpleNamespace(images=[image]), 41
        )
        image.processor_identity += "!"
        changed = get_block_cache_keys(tokens, 128, SimpleNamespace(images=[image]), 41)
        self.assertEqual(original[0], changed[0])
        self.assertTrue(all(a != b for a, b in zip(original[1:], changed[1:])))
        self.assertNotEqual(
            route_cache_keys_for_page_rr(original, True, 4),
            route_cache_keys_for_page_rr(changed, True, 4),
        )


if __name__ == "__main__":
    unittest.main()
