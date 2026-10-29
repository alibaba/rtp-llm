import json
import tempfile
import unittest
from pathlib import Path

import torch
from google.protobuf.json_format import MessageToDict
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import GenerateInputPB
from rtp_llm.cpp.model_rpc.replay_request import (
    load_model_inputs,
    load_request_log,
    prepare_local_request,
)


class ReplayRequestTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_request_log_preserves_tokens_sampling_and_correlation(self):
        request = GenerateInputPB(
            request_id=9223372036854770000, token_ids=range(20000)
        )
        request.generate_config.random_seed.value = 1234
        request.generate_config.top_p = 0.8
        request.generate_config.max_new_tokens = 128
        request.request_info.request_id = "dash-id"
        request.request_info.trace_id = 'trace\nwith"quotes'
        record = {"record_type": "prefill_request", "request": MessageToDict(request)}
        path = self.root / "prefill.log"
        path.write_text(json.dumps(record) + "\n")
        self.assertEqual(load_request_log(path, request.request_id), request)
        with self.assertRaisesRegex(ValueError, "No matching"):
            load_request_log(path, 42)
        path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "Multiple requests"):
            load_request_log(path, request.request_id)

    def test_local_replay_clears_production_routing_without_mutating_source(self):
        request = GenerateInputPB(
            request_id=123, token_ids=[1, 2, 3], start_time=1, group_size=2
        )
        request.group_id.value = 456
        request.generate_config.can_use_pd_separation = True
        request.generate_config.role_addrs.add(ip="production-decode", grpc_port=9000)
        request.generate_config.random_seed.value = 1234
        request.generate_config.max_new_tokens = 16
        replay = prepare_local_request(request)
        self.assertEqual(list(replay.token_ids), [1, 2, 3])
        self.assertEqual(replay.generate_config.random_seed.value, 1234)
        self.assertEqual(replay.generate_config.max_new_tokens, 16)
        self.assertFalse(replay.generate_config.role_addrs)
        self.assertFalse(replay.generate_config.can_use_pd_separation)
        self.assertFalse(replay.HasField("group_id"))
        self.assertEqual(replay.group_size, 0)
        self.assertGreater(replay.start_time, 1)
        self.assertEqual(request.generate_config.role_addrs[0].ip, "production-decode")

    def test_snapshot_uses_full_tokens_and_context_batch_index(self):
        snapshot = {
            "record_type": "gpt_model_inputs_snapshot",
            "record_sequence": 7,
            "combo_tokens": torch.tensor([90, 3, 4]),
            "all_input_token_ids": [torch.tensor([1, 2, 3]), torch.tensor([5, 6, 4])],
            "request_is_multimodal": [False, False],
            "request_id": torch.tensor([30, 40]),
            "trace_ids": ["decode-trace", "prefill-a", "prefill-b"],
        }
        path = self.root / "snapshot.pt"
        torch.save({"record_type": "model_inputs_chunk", "records": [snapshot]}, path)
        request = load_model_inputs(path, 7, 1)
        self.assertEqual(list(request.token_ids), [5, 6, 4])
        self.assertEqual(request.request_id, 40)
        self.assertEqual(request.generate_config.trace_id.value, "prefill-b")
        with self.assertRaisesRegex(ValueError, "outside"):
            load_model_inputs(path, 7, -1)
        with self.assertRaisesRegex(ValueError, "Select one"):
            load_model_inputs(path, 8)
        del snapshot["all_input_token_ids"]
        torch.save(snapshot, path)
        with self.assertRaisesRegex(ValueError, "older dumps"):
            load_model_inputs(path)

    def test_text_request_in_mixed_batch_with_image_features_is_replayable(self):
        for multimodal_index in (0, 1):
            with self.subTest(multimodal_index=multimodal_index):
                full_tokens = [torch.tensor([5, 6, 7]), torch.tensor([5, 6, 7])]
                full_tokens[multimodal_index] = torch.tensor([1, -1, -1, 2])
                modalities = [False, False]
                modalities[multimodal_index] = True
                snapshot = {
                    "record_type": "gpt_model_inputs_snapshot",
                    "record_sequence": 7,
                    "combo_tokens": torch.cat([torch.tensor([90]), *full_tokens]),
                    "all_input_token_ids": full_tokens,
                    "request_is_multimodal": modalities,
                    "multimodal_features": [torch.ones(2, 4)],
                    "input_embeddings": None,
                    "request_id": torch.tensor([30, 40]),
                    "trace_ids": ["decode-trace", "prefill-a", "prefill-b"],
                }
                path = self.root / "mixed.pt"
                torch.save(
                    {"record_type": "model_inputs_chunk", "records": [snapshot]}, path
                )
                text_index = 1 - multimodal_index
                request = load_model_inputs(path, 7, text_index)
                self.assertEqual(list(request.token_ids), [5, 6, 7])
                self.assertEqual(request.request_id, [30, 40][text_index])
                self.assertEqual(
                    request.generate_config.trace_id.value,
                    ["prefill-a", "prefill-b"][text_index],
                )
                with self.assertRaisesRegex(ValueError, "multimodal request"):
                    load_model_inputs(path, 7, multimodal_index)

    def test_snapshot_with_input_embeddings_is_rejected(self):
        snapshot = {
            "record_type": "gpt_model_inputs_snapshot",
            "record_sequence": 7,
            "all_input_token_ids": [torch.tensor([1, 2, 3])],
            "request_is_multimodal": [False],
            "input_embeddings": [torch.ones(3, 4)],
            "request_id": torch.tensor([30]),
        }
        path = self.root / "embeddings.pt"
        torch.save(snapshot, path)
        with self.assertRaisesRegex(ValueError, "input embeddings"):
            load_model_inputs(path)

    def test_cached_multimodal_request_is_rejected_without_forward_features(self):
        snapshot = {
            "record_type": "gpt_model_inputs_snapshot",
            "record_sequence": 7,
            "combo_tokens": torch.tensor([2, 3]),
            "all_input_token_ids": [torch.tensor([1, -1, -1, 2]), torch.tensor([3])],
            "request_is_multimodal": [True, False],
            "multimodal_features": None,
            "input_embeddings": None,
            "request_id": torch.tensor([30, 40]),
        }
        path = self.root / "cached_multimodal.pt"
        torch.save(snapshot, path)
        with self.assertRaisesRegex(ValueError, "multimodal request"):
            load_model_inputs(path, 7, 0)
        self.assertEqual(list(load_model_inputs(path, 7, 1).token_ids), [3])

        del snapshot["request_is_multimodal"]
        torch.save(snapshot, path)
        with self.assertRaisesRegex(ValueError, "request_is_multimodal"):
            load_model_inputs(path)
        snapshot["request_is_multimodal"] = [False]
        torch.save(snapshot, path)
        with self.assertRaisesRegex(ValueError, "request_is_multimodal"):
            load_model_inputs(path)


if __name__ == "__main__":
    unittest.main()
