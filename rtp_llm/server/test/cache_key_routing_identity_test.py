"""Frontend image key words must match CompleteTokenIds::imageCacheIdentity."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.server.cache_key_routing import get_block_cache_keys


class ImageCacheIdentityTest(unittest.TestCase):
    def test_length_delimited_content_identity_and_seed_reach_native_hash(self):
        image = SimpleNamespace(start=1, types=SimpleNamespace(numel=lambda: 2),
                                n_vit_h=3, n_vit_w=6, content_sha256="01", processor_identity="abc")
        native = SimpleNamespace(cpp_get_block_cache_keys=lambda chunks, seed: (chunks, seed))
        with patch.dict("sys.modules", {"rtp_llm.ops": native}):
            chunks, seed = get_block_cache_keys([10, 11, 12, 13], 2,
                                               SimpleNamespace(images=[image]), 123)
        # Image spans both blocks. C++ appends digest length then every unsigned byte,
        # including short/nonhex strings; it never parses groups of eight hex digits.
        identity = [-41, 1, 3, 3, 6, 2, 48, 49, 3, 97, 98, 99]
        self.assertEqual([[10, 11] + identity, [12, 13] + identity], chunks)
        self.assertEqual(123, seed)

    def test_empty_identity_limits_routing_to_trustworthy_text_prefix(self):
        image = SimpleNamespace(start=2, types=SimpleNamespace(numel=lambda: 2),
                                n_vit_h=3, n_vit_w=6, content_sha256="", processor_identity="abc")
        native = SimpleNamespace(cpp_get_block_cache_keys=lambda chunks, seed: chunks)
        with patch.dict("sys.modules", {"rtp_llm.ops": native}):
            result = get_block_cache_keys([1, 2, 3, 4, 5, 6], 2, SimpleNamespace(images=[image]))
        self.assertEqual([[1, 2]], result)


if __name__ == "__main__":
    unittest.main()
