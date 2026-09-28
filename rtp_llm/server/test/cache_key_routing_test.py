import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.ops import (
    DSV41_SWA_BOUNDED_REPLAY_CACHE_KEY_SEED,
    cpp_get_block_cache_keys,
    get_block_cache_keys,
)
from rtp_llm.server.cache_key_routing import (
    route_cache_key_seed_from_env,
    route_cache_keys_for_page_rr,
)


class CacheKeyLayoutTest(unittest.TestCase):
    def test_legacy_layout_is_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(route_cache_key_seed_from_env(), 0)
        for value in ("0", "false", "FALSE", "no", "off", ""):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {"DSV41_CED": "1", "DSV41_SWA_BOUNDED_REPLAY": value},
                clear=True,
            ):
                self.assertEqual(route_cache_key_seed_from_env(), 0)

    def test_bounded_layout_uses_worker_seed(self):
        for value in ("1", "true", "TRUE", "yes", "on"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {"DSV41_CED": value, "DSV41_SWA_BOUNDED_REPLAY": value},
                clear=True,
            ):
                self.assertEqual(
                    route_cache_key_seed_from_env(),
                    DSV41_SWA_BOUNDED_REPLAY_CACHE_KEY_SEED,
                )
        self.assertEqual(DSV41_SWA_BOUNDED_REPLAY_CACHE_KEY_SEED, 0x4453563431525031)

    def test_bounded_layout_requires_ced(self):
        for value in (None, "0", "false", "FALSE", "no", "off", ""):
            env = {"DSV41_SWA_BOUNDED_REPLAY": "1"}
            if value is not None:
                env["DSV41_CED"] = value
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, "requires DSV41_CED=1"):
                    route_cache_key_seed_from_env()

    def test_hash_default_remains_legacy(self):
        tokens = list(range(1025))
        chunks = [tokens[i : i + 128] for i in range(0, 1024, 128)]
        legacy = cpp_get_block_cache_keys(chunks)
        self.assertEqual(legacy, cpp_get_block_cache_keys(chunks, initial_hash=0))
        self.assertEqual(legacy, get_block_cache_keys(tokens, 128))
        self.assertEqual(legacy, get_block_cache_keys(tokens, 128, cache_key_seed=0))
        self.assertNotEqual(
            legacy,
            get_block_cache_keys(
                tokens, 128, cache_key_seed=DSV41_SWA_BOUNDED_REPLAY_CACHE_KEY_SEED
            ),
        )

    def test_text_keys_match_virtual_blocks_with_either_seed(self):
        for seed in (0, DSV41_SWA_BOUNDED_REPLAY_CACHE_KEY_SEED):
            for block_size, cp_size in ((128, 4), (128, 1), (256, 4), (128, 2)):
                for length in (127, 512, 8193):
                    with self.subTest(
                        seed=seed, block_size=block_size, cp_size=cp_size, length=length
                    ):
                        tokens = list(range(length))
                        physical = get_block_cache_keys(
                            tokens, block_size, cache_key_seed=seed
                        )
                        self.assertEqual(
                            route_cache_keys_for_page_rr(physical, True, cp_size),
                            get_block_cache_keys(
                                tokens, block_size * cp_size, cache_key_seed=seed
                            ),
                        )

    def test_image_identity_and_seed_precede_page_rr_sampling(self):
        tokens = list(range(1537))
        image = SimpleNamespace(
            start=600,
            types=SimpleNamespace(numel=lambda: 200),
            n_vit_h=10,
            n_vit_w=20,
            content_sha256="a" * 64,
            processor_identity="b" * 64,
        )
        prepared = SimpleNamespace(images=[image])
        identity = [-41, 600, 800, 10, 20] + [0xAAAAAAAA] * 8 + [0xBBBBBBBB] * 8
        chunks = [tokens[i : i + 128] for i in range(0, 1536, 128)]
        for index in (4, 5, 6):
            chunks[index].extend(identity)

        for seed in (0, DSV41_SWA_BOUNDED_REPLAY_CACHE_KEY_SEED):
            with self.subTest(seed=seed):
                actual = get_block_cache_keys(tokens, 128, prepared, seed)
                self.assertEqual(actual, cpp_get_block_cache_keys(chunks, seed))
                routed = route_cache_keys_for_page_rr(actual, True, 4)
                text = get_block_cache_keys(tokens, 512, cache_key_seed=seed)
                self.assertEqual(routed[:1], text[:1])
                self.assertNotEqual(routed[1:], text[1:])
                self.assertNotEqual(
                    routed, get_block_cache_keys(tokens, 512, prepared, seed)
                )
                image.content_sha256 = "c" * 64
                self.assertNotEqual(
                    actual, get_block_cache_keys(tokens, 128, prepared, seed)
                )
                image.content_sha256 = "a" * 64
                image.processor_identity = "d" * 64
                self.assertNotEqual(
                    actual, get_block_cache_keys(tokens, 128, prepared, seed)
                )
                image.processor_identity = "b" * 64
        self.assertEqual(tokens, list(range(1537)))


if __name__ == "__main__":
    unittest.main()
