"""Route key parity against the real config binding; no mocked native hashes."""

import unittest

from rtp_llm.ops import cpp_get_block_cache_keys, get_block_cache_keys

_MASK64 = (1 << 64) - 1
_MTP_CACHE_KEY_SEED = 0x4D54505F4B455931


def _signed64(value):
    value &= _MASK64
    return value if value < (1 << 63) else value - (1 << 64)


def _reference_keys(token_ids, block_size, mtp_lookahead):
    """Hash the token stream once and record each complete KV dependency."""
    state = _MTP_CACHE_KEY_SEED if mtp_lookahead else 0
    keys = []
    lookahead = int(mtp_lookahead)
    for position, value in enumerate(token_ids, 1):
        # HashUtil uses std::hash<int32_t> and an arithmetic signed right shift.
        value &= (1 << 32) - 1
        if value >= (1 << 31):
            value -= 1 << 32
        combined = value + 0x9E3779B97F4A7C15 + (state << 12) + (state >> 32)
        state = _signed64(state ^ combined)
        if position > lookahead and (position - lookahead) % block_size == 0:
            keys.append(state)
    return keys


class BlockCacheKeyTest(unittest.TestCase):
    def test_wrapper_matches_native_token_stream_hash_at_every_boundary(self):
        for block_size in (1, 4, 64):
            for length in (
                0,
                1,
                block_size - 1,
                block_size,
                block_size + 1,
                2 * block_size,
                2 * block_size + 1,
                3 * block_size + 2,
            ):
                tokens = list(range(length))
                for mtp_lookahead in (False, True):
                    with self.subTest(
                        block_size=block_size,
                        length=length,
                        mtp_lookahead=mtp_lookahead,
                    ):
                        self.assertEqual(
                            get_block_cache_keys(tokens, block_size, mtp_lookahead),
                            _reference_keys(tokens, block_size, mtp_lookahead),
                        )

    def test_native_legacy_default_and_explicit_mode_are_identical(self):
        chunks = [[0, 1, 100000], [2147483647, -2147483648, -1]]
        expected = _reference_keys(sum(chunks, []), 3, False)
        self.assertEqual(cpp_get_block_cache_keys(chunks), expected)
        self.assertEqual(cpp_get_block_cache_keys(chunks, False), expected)
        self.assertEqual(get_block_cache_keys(sum(chunks, []), 3), expected)

    def test_mtp_native_chunk_contract_and_namespace(self):
        tokens = list(range(129))
        chunks = [tokens[:65], tokens[65:129]]
        keys = cpp_get_block_cache_keys(chunks, mtp_lookahead=True)
        self.assertEqual(keys, _reference_keys(tokens, 64, True))
        self.assertEqual(keys, get_block_cache_keys(tokens, 64, True))
        self.assertNotEqual(keys, cpp_get_block_cache_keys(chunks))

    def test_each_mtp_boundary_depends_on_its_lookahead(self):
        tokens = list(range(130))
        original = get_block_cache_keys(tokens, 64, True)
        for fork_position, expected_equal in (
            (64, [False, False]),
            (65, [True, False]),
            (128, [True, False]),
            (129, [True, True]),
        ):
            with self.subTest(fork_position=fork_position):
                fork = tokens.copy()
                fork[fork_position] += 1000
                changed = get_block_cache_keys(fork, 64, True)
                self.assertEqual(
                    [left == right for left, right in zip(original, changed)],
                    expected_equal,
                )

    def test_mtp_aligned_prompt_does_not_advertise_unknown_lookahead(self):
        self.assertEqual(get_block_cache_keys(list(range(64)), 64, True), [])
        for length in (128, 192):
            with self.subTest(length=length):
                tokens = list(range(length))
                keys = get_block_cache_keys(tokens, 64, True)
                self.assertEqual(len(keys), length // 64 - 1)
                # CP2 routes only complete pairs of stable physical block keys.
                self.assertEqual(len(keys[1::2]), (length - 1) // 128)


if __name__ == "__main__":
    unittest.main()
