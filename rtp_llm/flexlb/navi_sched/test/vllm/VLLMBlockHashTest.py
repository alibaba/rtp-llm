import hashlib
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from navi_sched.adapters.vllm.VLLMBlockHash import (
    block_hash_to_uint64,
    calculate_vllm_block_hashes,
)


class VLLMBlockHashTest(unittest.TestCase):
    def test_request_metadata_matches_vllm_hasher(self):
        try:
            from vllm.utils.hashing import sha256_cbor
            from vllm.v1.core import kv_cache_utils
        except ImportError:
            self.skipTest("vLLM is not installed")
        token_ids = list(range(161))
        image = SimpleNamespace(
            identifier="image-test-content-hash",
            mm_position=SimpleNamespace(offset=48, length=32),
        )
        boundary_image = SimpleNamespace(
            identifier="different-image",
            mm_position=SimpleNamespace(offset=80, length=17),
        )
        cases = [
            {},
            {"mm_features": [image]},
            {"mm_features": [image, boundary_image]},
            {"lora_request": SimpleNamespace(lora_name="adapter")},
            {"cache_salt": "salt"},
            {"mm_features": [image], "cache_salt": "salt",
             "lora_request": SimpleNamespace(lora_name="adapter")},
        ]
        with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}):
            kv_cache_utils.init_none_hash(sha256_cbor)
            for metadata in cases:
                with self.subTest(metadata=metadata):
                    fields = dict(
                        mm_features=[], lora_request=None, cache_salt=None,
                        prompt_embeds=None, _prompt_embeds_per_block_hashes={},
                        block_hashes=[], all_token_ids=token_ids,
                        num_tokens=len(token_ids),
                    )
                    fields.update(metadata)
                    request = SimpleNamespace(**fields)
                    expected = kv_cache_utils.get_request_block_hasher(
                        16, sha256_cbor
                    )(request)
                    self.assertEqual(
                        [block_hash_to_uint64(value) for value in expected],
                        calculate_vllm_block_hashes(token_ids, 16, request=request),
                    )

    def test_multimodal_request_hashes_include_image_identity(self):
        try:
            import vllm.v1.core.kv_cache_utils  # noqa: F401
        except ImportError:
            self.skipTest("vLLM is not installed")
        request = SimpleNamespace(
            mm_features=[SimpleNamespace(
                identifier="image-test-content-hash",
                mm_position=SimpleNamespace(offset=48, length=32),
            )],
            lora_request=None,
            cache_salt=None,
            prompt_embeds=None,
        )
        expected = [
            12687895063098490012, 1814341696565857452, 2713165534665270607,
            18404408909527126274, 7865465297107673843, 10698632190875616625,
            13012976205636083705, 9965430180500714476, 7245807258087879451,
            12708805266631538567,
        ]
        self.assertEqual(
            expected,
            calculate_vllm_block_hashes(list(range(160)), 16, request=request),
        )

    def test_multimodal_hash_distinguishes_relative_image_offsets(self):
        from vllm.v1.core import kv_cache_utils

        def request(offset):
            return SimpleNamespace(
                mm_features=[SimpleNamespace(
                    identifier="same-image",
                    mm_position=SimpleNamespace(offset=offset, length=31),
                )],
                lora_request=None, cache_salt=None, prompt_embeds=None,
            )

        tokens = [1] * 160
        left = calculate_vllm_block_hashes(tokens, 16, request=request(48))
        right = calculate_vllm_block_hashes(tokens, 16, request=request(49))
        self.assertEqual(left[:3], right[:3])
        self.assertNotEqual(left[3:], right[3:])

        original = kv_cache_utils._gen_mm_extra_hash_keys

        def omit_relative_offset(*args):
            keys, index = original(*args)
            return [key[0] for key in keys], index

        # Dropping the offset recreates the old collision with identical tokens.
        with mock.patch.object(kv_cache_utils, "_gen_mm_extra_hash_keys", omit_relative_offset):
            left = calculate_vllm_block_hashes(tokens, 16, request=request(48))
            right = calculate_vllm_block_hashes(tokens, 16, request=request(49))
        self.assertEqual(left, right)

    def test_matches_vllm_fixed_vectors(self):
        self.assertEqual(
            [2164874634404590027],
            calculate_vllm_block_hashes([1, 2, 3, 4], 4),
        )
        self.assertEqual(
            [10918909127363516282, 10585920789087210302],
            calculate_vllm_block_hashes(list(range(128)), 64),
        )
        self.assertEqual(
            [2164874634404590027, 17270480062156288726],
            calculate_vllm_block_hashes(list(range(1, 10)), 4),
        )

    def test_drops_partial_final_block(self):
        self.assertEqual([], calculate_vllm_block_hashes([1, 2, 3], 4))
        self.assertEqual(
            calculate_vllm_block_hashes([1, 2, 3, 4], 4),
            calculate_vllm_block_hashes([1, 2, 3, 4, 5, 6, 7], 4),
        )

    def test_converts_full_hash_to_wire_uint64(self):
        full_hash = hashlib.sha256(b"navi_sched").digest()
        self.assertEqual(
            int.from_bytes(full_hash[-8:], "big"),
            block_hash_to_uint64(full_hash),
        )
        self.assertEqual((1 << 64) - 1, block_hash_to_uint64(-1))

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            calculate_vllm_block_hashes([1], 0)

    def test_matches_vllm_hash_block_tokens(self):
        try:
            from vllm.utils.hashing import sha256_cbor
            from vllm.v1.core import kv_cache_utils
        except ImportError:
            self.skipTest("vLLM is not installed")

        old_seed = os.environ.get("PYTHONHASHSEED")
        os.environ["PYTHONHASHSEED"] = "0"
        try:
            kv_cache_utils.init_none_hash(sha256_cbor)
            token_ids = list(range(513))
            block_size = 16
            parent_hash = None
            expected = []
            for begin in range(
                0,
                len(token_ids) // block_size * block_size,
                block_size,
            ):
                parent_hash = kv_cache_utils.hash_block_tokens(
                    sha256_cbor,
                    parent_hash,
                    token_ids[begin : begin + block_size],
                )
                expected.append(block_hash_to_uint64(parent_hash))
        finally:
            if old_seed is None:
                os.environ.pop("PYTHONHASHSEED", None)
            else:
                os.environ["PYTHONHASHSEED"] = old_seed

        self.assertEqual(
            expected,
            calculate_vllm_block_hashes(token_ids, block_size),
        )


if __name__ == "__main__":
    unittest.main()
