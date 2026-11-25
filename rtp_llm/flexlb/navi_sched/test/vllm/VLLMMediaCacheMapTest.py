import unittest
from types import SimpleNamespace
from unittest import mock

from navi_sched.adapters.vllm import VLLMImageCacheMap as module
from navi_sched.adapters.vllm.VLLMImageCacheMap import (
    VLLMMediaCacheMap,
    media_expansion_spans,
    project_media_prefix,
)


def block(value):
    return value.to_bytes(32, "big")


def feature(modality, offset, length):
    return SimpleNamespace(
        modality=modality, mm_position=SimpleNamespace(offset=offset, length=length)
    )


class MediaCacheMapTest(unittest.TestCase):
    def test_audio_and_video_use_the_same_single_placeholder_expansion(self):
        for modality in ("image", "video", "audio"):
            with self.subTest(modality=modality):
                source = SimpleNamespace(
                    prompt_token_ids=list(range(12)),
                    mm_features=[feature(modality, 5, 1)],
                )
                native = SimpleNamespace(
                    prompt_token_ids=list(range(19)),
                    mm_features=[feature(modality, 5, 8)],
                    block_hashes=[block(i) for i in range(4)],
                )
                spans = media_expansion_spans(source, native)
                self.assertEqual(((5, 8, 1),), spans)
                self.assertEqual(
                    [4, 5, 5, 5, 6, 12],
                    [project_media_prefix(n, spans) for n in (4, 5, 6, 12, 13, 19)],
                )
                cache = SimpleNamespace(_full_ready=set(), _tail_ready=set())
                cache._flat_ready = cache._full_ready
                mapping = VLLMMediaCacheMap(cache, False)
                with mock.patch.object(
                    module, "calculate_vllm_block_hashes", return_value=[1, 2, 3]
                ):
                    mapping.register(7, source, native, 4)
                cache._full_ready.update(native.block_hashes[:-1])
                self.assertTrue(mapping.update(native.block_hashes[:-1]).empty())
                cache._full_ready.add(native.block_hashes[-1])
                self.assertEqual(
                    (block(2),), mapping.update([native.block_hashes[-1]]).added
                )

    def interleaved_requests(self):
        tokens = [
            10,
            11,
            12,
            13,
            900,
            14,
            15,
            16,
            901,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
        ]
        source = SimpleNamespace(
            prompt_token_ids=tokens,
            mm_features=[feature("video", 4, 1), feature("audio", 8, 1)],
        )
        native = SimpleNamespace(
            prompt_token_ids=tokens[:4]
            + [700, 701, 702]
            + [900] * 8
            + [901] * 8
            + [703, 704]
            + tokens[9:],
            mm_features=[feature("video", 7, 14), feature("audio", 9, 14)],
            block_hashes=[block(i) for i in range(9)],
        )
        return source, native

    def test_timestamps_and_interleaved_modalities_project_back_to_original_prefix(
        self,
    ):
        source, native = self.interleaved_requests()
        spans = media_expansion_spans(source, native)
        self.assertEqual(((4, 21, 5),), spans)
        for count in range(5, 25):
            self.assertEqual(4, project_media_prefix(count, spans))
        self.assertEqual(9, project_media_prefix(25, spans))
        self.assertEqual(12, project_media_prefix(28, spans))
        self.assertEqual(
            len(source.prompt_token_ids),
            project_media_prefix(len(native.prompt_token_ids), spans),
        )

    def test_interleaved_cache_aliases_require_complete_media_and_all_tail_aliases_update(
        self,
    ):
        source, native = self.interleaved_requests()
        cache = SimpleNamespace(_full_ready=set(), _tail_ready=set())
        mapping = VLLMMediaCacheMap(cache, True)
        with mock.patch.object(
            module, "calculate_vllm_block_hashes", return_value=[1, 2, 3, 4, 5]
        ):
            mapping.register(7, source, native, 4)
        record = next(iter(mapping._records.values()))
        self.assertEqual((7, 7, 8, 9), record.required_blocks)
        cache._full_ready.update(native.block_hashes[:6])
        self.assertTrue(mapping.update(native.block_hashes[:6]).empty())
        cache._full_ready.add(block(6))
        self.assertEqual(
            {block(2), block(3)}, set(mapping.update([block(6)]).added_full_ready)
        )
        cache._tail_ready.add(block(6))
        self.assertEqual(
            {block(2), block(3)}, set(mapping.update([block(6)]).added_tail_ready)
        )
        cache._tail_ready.remove(block(6))
        self.assertEqual(
            {block(2), block(3)}, set(mapping.update([block(6)]).removed_tail_ready)
        )
        cache._full_ready.remove(block(5))
        self.assertEqual(
            {block(2), block(3)}, set(mapping.update([block(5)]).removed_full_ready)
        )
        mapping.finish(7)
        self.assertEqual({}, mapping._records)
        self.assertEqual({}, mapping._owners)

    def test_mismatched_or_invalid_media_cannot_create_aliases(self):
        source, native = self.interleaved_requests()
        native.mm_features[1].modality = "image"
        with self.assertRaises(ValueError):
            media_expansion_spans(source, native)
        source, native = self.interleaved_requests()
        source.mm_features[0].mm_position.length = 2
        with self.assertRaises(ValueError):
            media_expansion_spans(source, native)


class AgentMediaResponseTest(unittest.TestCase):
    def test_registration_and_response_counters_use_the_same_media_spans(self):
        import ast
        import copy
        import logging
        import threading
        from pathlib import Path
        from types import MethodType

        source = Path(__file__).parents[2] / "adapters/vllm/VLLMEngineObserver.py"
        observer_class = next(
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "VLLMEngineObserver"
        )
        methods = [
            node
            for node in observer_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("_register_media_cache_mappings", "route_engine_outputs")
        ]
        namespace = dict(
            copy=copy,
            logger=logging.getLogger(__name__),
            media_expansion_spans=media_expansion_spans,
            project_media_prefix=project_media_prefix,
            forward_outputs_fields=lambda outputs: outputs,
        )
        exec(
            compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"),
            namespace,
        )
        prediction, actual = MediaCacheMapTest().interleaved_requests()
        cache = SimpleNamespace(_full_ready=set(), _tail_ready=set())
        cache._flat_ready = cache._full_ready
        observer = SimpleNamespace(
            _cache_lifecycle_lock=threading.RLock(),
            _response_lock=threading.RLock(),
            _response_request_ids={7: "request-7"},
            _response_media_spans={},
            _inactive_response_request_ids=set(),
            _block_size=4,
            _media_cache_map=VLLMMediaCacheMap(cache, False),
            _report_cache_delta=mock.Mock(),
            _bridge=mock.Mock(),
            _response_encoder=mock.Mock(),
        )
        for name in ("_register_media_cache_mappings", "route_engine_outputs"):
            setattr(observer, name, MethodType(namespace[name], observer))
        batch = SimpleNamespace(
            get_request_block_hashes=lambda request_id: [1, 2, 3, 4, 5]
        )
        observer._register_media_cache_mappings(
            batch, [(7, (actual,))], {7: prediction}
        )
        self.assertEqual(((4, 21, 5),), observer._response_media_spans[7])
        observer._report_cache_delta.assert_called_once()
        value = SimpleNamespace(
            request_id="request-7",
            finished=False,
            num_cached_tokens=28,
            num_external_computed_tokens=24,
        )
        outputs = SimpleNamespace(finished_requests=set(), outputs=[value])

        def encode_response(response):
            self.assertEqual(12, response.outputs[0].num_cached_tokens)
            self.assertEqual(8, response.outputs[0].num_external_computed_tokens)
            return [b"response"]

        observer._response_encoder.encode.side_effect = encode_response
        observer._bridge.publish_forward_response.return_value = True
        self.assertTrue(observer.route_engine_outputs(7, outputs))
        observer._bridge.publish_forward_response.assert_called_once_with(
            7, [b"response"], False
        )
        self.assertEqual(28, value.num_cached_tokens)
        self.assertEqual(24, value.num_external_computed_tokens)


if __name__ == "__main__":
    unittest.main()
