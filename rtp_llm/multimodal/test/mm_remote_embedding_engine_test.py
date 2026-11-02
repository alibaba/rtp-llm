import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from rtp_llm.multimodal.mm_embedding_cache import MMEmbeddingCache, MMHashKeyCache
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.mm_remote_embedding_cache import MMRemoteEmbeddingCache
from rtp_llm.multimodal.mm_scheduler import _run_embedding
from rtp_llm.multimodal.test.mm_remote_embedding_cache_test import FakeClient
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput


class RemoteEmbeddingEngineTest(unittest.TestCase):
    def setUp(self):
        self.value = (
            torch.ones(3, 8, dtype=torch.bfloat16),
            torch.ones(3, 3, dtype=torch.int64),
            torch.ones(72, dtype=torch.bfloat16),
        )
        self.engine = object.__new__(MMProcessEngine)
        self.engine.vit_config = SimpleNamespace(mm_timeout_ms=3000)
        self.engine.mm_preprocess_batch_size = 1
        self.engine._diagnostic_model_type = None
        self.engine._remote_embedding_cache = Mock()
        self.engine._remote_embedding_cache.key.side_effect = (
            lambda inputs, **kwargs: inputs[0].url
        )
        self.engine._embedding_cache = MMEmbeddingCache(0, 4096)
        self.engine._hash_key_cache = MMHashKeyCache(0)
        self.addCleanup(self.engine._embedding_cache.clear)
        self.addCleanup(self.engine._hash_key_cache.clear)
        self.engine._scheduler = Mock(max_request_images=10)
        self.engine.preprocess_executor = Mock()
        self.engine.report_vit_error = Mock()
        self.engine.is_proxy_mode = False
        self.part = Mock(supports_remote_source_identity=False)
        self.engine.mm_part = self.part
        self.part.batched_embedding.side_effect = lambda data, types: [
            self.value for _ in data
        ]
        self.engine._scheduler.submit_and_wait.side_effect = (
            lambda items, **kwargs: _run_embedding(self.part, items)
        )

    def inputs(self, count=1):
        return [
            MultimodalInput(
                "data:image/png;base64,AQI" + chr(68 + i),
                1,
                torch.empty(0),
                MMPreprocessConfig(),
            )
            for i in range(count)
        ]

    def test_remote_hit_skips_scheduler_and_preserves_deepstack(self):
        remote = self.engine._remote_embedding_cache
        remote.load.return_value = self.value
        items = self.engine._create_work_items(self.inputs())
        actual = self.engine._compute_embeddings(items)
        self.engine._scheduler.submit_and_wait.assert_not_called()
        self.assertEqual(items[0].cache_source, "remote")
        for got, expected in zip(actual, self.value):
            self.assertEqual(len(got), 1)
            self.assertIs(got[0], expected)
        remote.record.assert_called_with(
            "result_used",
            key=items[0].remote_cache_key,
            trace_id=items[0].remote_trace_id,
            source="remote",
        )

    def test_miss_encodes_into_local_cache_without_write_through(self):
        remote = self.engine._remote_embedding_cache
        remote.load.return_value = None
        items = self.engine._create_work_items(self.inputs())
        self.engine._compute_embeddings(items)
        self.engine._scheduler.submit_and_wait.assert_called_once_with(
            items, deadline=None
        )
        self.assertEqual(
            self.engine._embedding_cache.resident_tiers()[items[0].cache_key], "cpu"
        )
        remote.submit_eviction.assert_not_called()
        remote.save.assert_not_called()

    def test_mixed_remote_hits_and_misses_only_submit_misses(self):
        self.engine._remote_embedding_cache.load.side_effect = [self.value, None]
        items = self.engine._create_work_items(self.inputs(2))
        actual = self.engine._compute_embeddings(items)
        self.engine._scheduler.submit_and_wait.assert_called_once_with(
            [items[1]], deadline=None
        )
        self.assertEqual([item.cache_source for item in items], ["remote", "encoder"])
        self.assertEqual(len(actual[0]), 2)

    def test_in_progress_follower_preserves_producer_remote_context(self):
        remote = self.engine._remote_embedding_cache
        remote.load.return_value = None
        producer = self.engine._create_work_items(self.inputs())[0]
        follower = self.engine._create_work_items(self.inputs())[0]
        self.assertEqual(follower.cache_state, "in_progress")
        self.assertIs(producer.cache_entry, follower.cache_entry)
        remote.bind_entry.assert_called_once_with(
            producer.cache_entry, producer.remote_cache_key, producer.remote_trace_id
        )
        remote.load.assert_called_once()

    def test_disabled_keeps_existing_encoder_path(self):
        self.engine._remote_embedding_cache = None
        items = self.engine._create_work_items(self.inputs())
        self.engine._compute_embeddings(items)
        self.part.batched_embedding.assert_called_once()

    def real_remote(self, model_type):
        storage = {}
        remote = MMRemoteEmbeddingCache(
            None,
            client_factory=lambda: FakeClient(storage),
            namespace="weights-processor-v1",
            model_identity={"model_type": model_type},
            max_object_bytes=65536,
            max_inflight_bytes=1024 * 1024,
            read_timeout_ms=500,
        )
        self.engine._remote_embedding_cache = remote
        self.addCleanup(remote.close)
        return remote, storage

    def invoke_cpp(self, url, tensor, mm_type=1):
        return self.engine.mm_embedding_cpp(
            [url], [mm_type], [tensor], [(-1, -1, -1, -1, -1, -1, -1, [], -1)]
        )

    def test_cpp_url_placeholder_normalized_only_for_verified_models(self):
        for model in ("qwen3_vl", "deepseek_vl_v2"):
            with self.subTest(model=model):
                remote, _ = self.real_remote(model)
                self.engine._embed_with_greennet_sync = Mock(
                    return_value=SimpleNamespace(position_ids=[])
                )
                self.invoke_cpp("data:image/jpeg;base64,AQID", torch.empty(1))
                forwarded = self.engine._embed_with_greennet_sync.call_args.args[0][0]
                self.assertEqual(forwarded.tensor.numel(), 0)
                self.assertIsNotNone(remote.key([forwarded]))

    def test_cpp_does_not_normalize_real_tensor_mutable_url_or_unknown_model(self):
        for model, url, tensor, mm_type in (
            ("qwen3_vl", "data:image/jpeg;base64,AQID", torch.ones(3), 1),
            (
                "deepseek_vl_v2",
                "data:image/jpeg;base64,AQID",
                torch.ones(1, dtype=torch.int64),
                1,
            ),
            ("qwen3_vl", "data:image/jpeg;base64,AQID", torch.ones(1), 4),
            ("qwen3_vl", "https://host/mutable.jpg", torch.empty(1), 1),
            ("unverified", "data:image/jpeg;base64,AQID", torch.empty(1), 1),
        ):
            with self.subTest(model=model, mm_type=mm_type):
                remote, _ = self.real_remote(model)
                self.engine._embed_with_greennet_sync = Mock(
                    return_value=SimpleNamespace(position_ids=[])
                )
                self.invoke_cpp(url, tensor, mm_type)
                forwarded = self.engine._embed_with_greennet_sync.call_args.args[0][0]
                self.assertIs(forwarded.tensor, tensor)
                self.assertIsNone(remote.key([forwarded]))

    def test_deepseek_none_positions_remote_hit_skips_encoder(self):
        from rtp_llm.multimodal.kvcm.tensor_object import pack_object, plan_object

        remote, storage = self.real_remote("deepseek_vl_v2")
        value = (torch.arange(24, dtype=torch.bfloat16).reshape(3, 8), None)
        inputs = self.inputs()
        storage[remote.key(inputs)] = pack_object(plan_object(value))
        items = self.engine._create_work_items(inputs)
        actual = self.engine._compute_embeddings(items)
        self.assertEqual(items[0].cache_source, "remote")
        self.assertEqual(actual[1:], ([], []))
        torch.testing.assert_close(actual[0][0], value[0], rtol=0, atol=0)
        self.part.batched_embedding.assert_not_called()

    def test_encoder_event_at_actual_batched_embedding_call(self):
        remote = self.engine._remote_embedding_cache
        remote.load.return_value = None
        items = self.engine._create_work_items(self.inputs())
        _run_embedding(self.part, items)
        remote.record.assert_called_once_with(
            "encoder_forward",
            key=items[0].remote_cache_key,
            trace_id=items[0].remote_trace_id,
            batch_items=1,
        )
        self.assertIs(items[0].embedding_result, self.value)


if __name__ == "__main__":
    unittest.main()
