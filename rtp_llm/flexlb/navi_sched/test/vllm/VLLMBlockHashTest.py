import hashlib
import os
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

from navi_sched.adapters.vllm.NaviSchedEndpointRouter import NaviSchedEndpointRouter
from navi_sched.adapters.vllm.VLLMBlockHash import (
    block_hash_to_uint64,
    calculate_vllm_block_hashes,
    wrap_vllm_request_block_hasher,
)


class VLLMBlockHashTest(unittest.TestCase):

    def test_cpp_plain_route_hash_matches_python_block_chain(self):
        from navi_sched.util import naviSchedModule

        native = naviSchedModule()
        token_ids = [
            -(1 << 63), -65537, -256, -24, 0, 23, 24, 255, 256,
            65535, 65536, (1 << 32), (1 << 63) - 1,
        ] * 10
        for block_size in (1, 4, 16, 64, 4096):
            with self.subTest(block_size=block_size):
                self.assertEqual(
                    calculate_vllm_block_hashes(token_ids, block_size),
                    native.hash_vllm_plain_text_blocks(array("q", token_ids), block_size),
                )
        with self.assertRaises(ValueError):
            native.hash_vllm_plain_text_blocks(array("i", [1]), 1)

    def test_cpp_kv_hash_matches_native_and_continues_existing_blocks(self):
        from navi_sched.util import naviSchedModule
        from vllm.utils.hashing import sha256_cbor
        from vllm.v1.core import kv_cache_utils

        cpp_hash = naviSchedModule().hash_vllm_plain_text_kv_blocks
        block_size = 16
        native_hasher = kv_cache_utils.get_request_block_hasher(block_size, sha256_cbor)
        parent = sha256_cbor("a seed supplied by vLLM")
        native = mock.Mock(wraps=native_hasher)
        with mock.patch.object(kv_cache_utils, "NONE_HASH", parent, create=True):
            wrapped = wrap_vllm_request_block_hasher(native, block_size, parent, cpp_hash)
            token_ids = [-65537, -24, 23, 24, 65535, 65536, 1 << 32] * 7
            request = SimpleNamespace(
                all_token_ids=token_ids,
                num_tokens=len(token_ids),
                block_hashes=[],
                mm_features=[],
                lora_request=None,
                cache_salt=None,
                prompt_embeds=None,
            )
            expected = native_hasher(request)
            self.assertEqual(expected, wrapped(request))
            request.num_tokens = block_size + 1
            request.block_hashes = wrapped(request)
            self.assertEqual(expected[:1], request.block_hashes)
            self.assertEqual([], wrapped(request))
            request.num_tokens = len(token_ids)
            self.assertEqual(expected[1:], wrapped(request))
            self.assertEqual(expected[1:], native_hasher(request))
            native.assert_not_called()

        with self.assertRaises(ValueError):
            cpp_hash(array("q", token_ids), block_size, b"short")

    def test_cpp_kv_hash_preserves_native_special_request_handling(self):
        native = mock.Mock(return_value=[b"native"])
        cpp_hash = mock.Mock()
        wrapped = wrap_vllm_request_block_hasher(native, 16, b"x" * 32, cpp_hash)
        metadata = {
            "mm_features": [object()],
            "lora_request": object(),
            "cache_salt": "salt",
            "prompt_embeds": object(),
        }
        for field, value in metadata.items():
            request = SimpleNamespace(
                mm_features=[], lora_request=None, cache_salt=None, prompt_embeds=None
            )
            setattr(request, field, value)
            self.assertEqual([b"native"], wrapped(request))
            native.assert_called_with(request)
        cpp_hash.assert_not_called()

    def test_cpp_route_hash_only_handles_plain_native_pp_requests(self):
        env = {
            "VLLM_NAVI_SCHED_PP_ROUTE_HASH_CPP": "1",
            "VLLM_NAVI_SCHED_NATIVE_INGRESS": "1",
            "TPU_RAIDEN_JOB_NAME": "prefill",
            "NAVI_SCHED_PP_SIZE": "4",
        }
        token_ids = tuple(range(128))
        plain_request = SimpleNamespace()
        fields_requiring_vendor_hash = (
            {"mm_features": [object()]},
            {"lora_request": object()},
            {"cache_salt": "test"},
            {"prompt_embeds": object()},
        )
        client = mock.Mock()
        with mock.patch.dict(os.environ, env, clear=True):
            router = NaviSchedEndpointRouter(client, "prefill", block_size=64)
        self.addCleanup(router.close)
        with mock.patch(
            "navi_sched.adapters.vllm.NaviSchedEndpointRouter._hash_normalized_token_ids",
            return_value=[17],
        ) as vendor_hash:
            self.assertIs(
                router.route(token_ids, request_id=42, block_hash_request=plain_request),
                client.schedule.return_value,
            )
            client.schedule.assert_called_once_with(
                len(token_ids), block_hashes=[10918909127363516282, 10585920789087210302],
                block_size=64, forward_payload=b"", target_biz_name="prefill",
                timeout_ms=1000, request_id=42,
            )
            vendor_hash.assert_not_called()
            for metadata in fields_requiring_vendor_hash:
                request = SimpleNamespace(**metadata)
                self.assertEqual([17], router._route_hashes(token_ids, request))
                vendor_hash.assert_called_with(token_ids, 64, request=request)
            with mock.patch.dict(os.environ, dict(env, NAVI_SCHED_PP_SIZE="1"), clear=True):
                single_rank = NaviSchedEndpointRouter(mock.Mock(), "prefill", block_size=64)
            self.addCleanup(single_rank.close)
            self.assertEqual([17], single_rank._route_hashes(token_ids, plain_request))
            vendor_hash.assert_called_with(token_ids, 64, request=plain_request)

    def test_plain_runtime_request_uses_native_hashing(self):
        from navi_sched.adapters.vllm import VLLMBlockHash as hashes

        native_sha = mock.Mock()
        extra_keys = mock.Mock(return_value=(None, 0))
        native_hash = mock.Mock(
            side_effect=lambda function, parent, tokens, extra: hashes._hash_block(
                parent, tokens
            )
        )
        modules = {
            "vllm.utils.hashing": SimpleNamespace(sha256_cbor=native_sha),
            "vllm.v1.core.kv_cache_utils": SimpleNamespace(
                generate_block_hash_extra_keys=extra_keys,
                hash_block_tokens=native_hash,
            ),
        }
        request = SimpleNamespace(
            mm_features=[], lora_request=None, cache_salt=None, prompt_embeds=None
        )
        with mock.patch.dict("sys.modules", modules):
            actual = calculate_vllm_block_hashes(list(range(128)), 64, request=request)
        self.assertEqual([10918909127363516282, 10585920789087210302], actual)
        self.assertEqual(2, native_hash.call_count)
        self.assertEqual(2, extra_keys.call_count)
        self.assertTrue(
            all(call.args[0] is native_sha for call in native_hash.call_args_list)
        )

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
