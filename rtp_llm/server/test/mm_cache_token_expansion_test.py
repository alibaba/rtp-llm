"""Routing and wire expansion must agree without changing original prompt IDs."""
import unittest
from array import array

import torch

from rtp_llm.cpp.model_rpc.model_rpc_client import trans_input
from rtp_llm.server.mm_cache_key import linux_string_hash
from rtp_llm.server.mm_cache_routing import multimodal_routing_tokens
from rtp_llm.utils.base_model_datatypes import GenerateInput, MultimodalTokenExpansion
from rtp_llm.config.generate_config import GenerateConfig


class MultimodalTokenExpansionTest(unittest.TestCase):
    def test_linux_cache_identity_matches_libstdcpp(self):
        self.assertEqual(linux_string_hash("https://example.com/image.png"), 2000644620635833837)
        self.assertEqual(linux_string_hash("中文url"), 18006779371473338865)

    def test_duplicate_images_expand_in_prompt_order(self):
        spans = []
        ids, length = multimodal_routing_tokens(
            [1, 4, 6, 5, 2, 4, 6, 5, 3], [[4, 5]], False, ['a', 'a'],
            {'entries': [{'key': 'a', 'hash_hit': True, 'split_size': [2],
                          'feature_hashes': [-5, 100]}]}, 100, compact=True,
            expanded_spans=spans)
        self.assertEqual(list(ids), [1, 4, -5, 100, 5, 2, 4, -5, 100, 5, 3])
        self.assertEqual(length, 11)
        self.assertEqual(spans, [(2, 2), (7, 2)])

    def test_incomplete_metadata_never_claims_full_length(self):
        _, length = multimodal_routing_tokens([1, 4, 6, 5], [[4, 5]], False,
                                               ['a'], {'entries': []}, 100)
        self.assertIsNone(length)
        with self.assertRaises(ValueError):
            multimodal_routing_tokens([4, 6, 5], [[4, 5]], False, ['a'],
                {'entries': [{'key': 'a', 'hash_hit': True, 'split_size': [2],
                              'feature_hashes': [1]}]}, 100)

    def test_wire_uses_expansion_and_preserves_original_tensor(self):
        original = torch.tensor([[1, 6, 3]], dtype=torch.int32)
        request = GenerateInput(request_id=7, token_ids=original, mm_inputs=[],
                                generate_config=GenerateConfig(max_new_tokens=2))
        request.mm_token_expansion = MultimodalTokenExpansion(array('i', [1, -5, 100, 3]), [(1, 2)])
        wire = trans_input(request)
        self.assertEqual(list(wire.token_ids), [1, -5, 100, 3])
        self.assertEqual([(s.offset, s.length) for s in wire.multimodal_token_layout.spans], [(1, 2)])
        self.assertEqual(original.tolist(), [[1, 6, 3]])


if __name__ == '__main__':
    unittest.main()
