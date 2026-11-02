import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel
from rtp_llm.models_py.modules import Embedding, MultimodalEmbeddingInjector
from rtp_llm.multimodal import emb_deepseek_diagnostics as diag
from rtp_llm.multimodal.kvcm.tensor_object import (
    pack_object,
    plan_object,
    unpack_object,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_vl2.deepseek_vl2_mixin import (
    DeepSeekVLV2ImageEmbedding,
)
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs


class Vision(nn.Module):
    def forward_features(self, pixels):
        return pixels + 2


class Projector(nn.Module):
    def forward(self, features):
        return features * 3


class DeepSeekDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env = patch.dict(
            os.environ,
            {
                "DEEPSEEK_EMB_DIAGNOSTICS": "1",
                "DEEPSEEK_EMB_DIAGNOSTICS_DIR": self.directory.name,
                "ENABLE_CUDA_GRAPH": "false",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        diag._observer = None

    def events(self):
        path = Path(self.directory.name) / "emb-chain.jsonl"
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )

    def vision_result(
        self, trace="writer", device="cpu", width=2, height=1, head="head"
    ):
        model = object.__new__(DeepSeekVLV2ImageEmbedding)
        model.vision = Vision()
        model.vision.register_parameter(
            "probe", nn.Parameter(torch.ones(1, dtype=torch.bfloat16, device=device))
        )
        model.projector = Projector()
        model.image_newline = nn.Parameter(
            torch.tensor([11, 13, 17, 19], dtype=torch.bfloat16, device=device)
        )
        model.view_seperator = nn.Parameter(
            torch.tensor([23, 29, 31, 37], dtype=torch.bfloat16, device=device)
        )
        model.global_view_pos = head
        model.tile_tag = "2D"
        pixels = (
            torch.arange((1 + width * height) * 4 * 4, dtype=torch.float32)
            .reshape(1 + width * height, 4, 4)
            .to(torch.bfloat16)
        )
        diag.request_start(trace, 1, "deepseek_vl_v2")
        with diag.encoder_scope([SimpleNamespace(deepseek_diagnostic_trace=trace)]):
            result = model.embedding([pixels, width, height], 1)
        diag.encoder_tuple(trace, result)
        return result

    def consume(self, trace, result, source):
        diag.delivered(trace, result, source)
        feature = result[0]
        hidden = torch.full(
            (feature.shape[0] + 4, feature.shape[1]),
            -1,
            dtype=feature.dtype,
            device=feature.device,
        )
        locs = torch.tensor([2], dtype=torch.int32)
        ticket = diag.model_begin([feature], locs, "deepseek_vl_v2", 1)
        expected = hidden.clone()
        expected[2 : 2 + feature.shape[0]].copy_(feature)
        actual = MultimodalEmbeddingInjector()(hidden, [feature], locs)
        diag.embedding_copied(ticket, actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return actual

    def test_real_embedding_head_and_tail_assembly_matches_independent_cpu_oracle(self):
        for width, height, head in ((2, 1, "head"), (1, 2, "tail"), (2, 2, "head")):
            with self.subTest(width=width, height=height, head=head):
                trace = f"{width}-{height}-{head}"
                result = self.vision_result(
                    trace, width=width, height=height, head=head
                )
                self.consume(trace, result, "encoder")
        self.assertFalse(any(e["event"] == "unproven" for e in self.events()))
        self.assertTrue(all(e["exact"] for e in self.events() if "exact" in e))

    def roundtrip(self, device):
        result = self.vision_result(device=device)
        before = result[0].clone()
        plan = plan_object(result)
        buffer = pack_object(plan)
        diag.serialized("writer", plan, buffer)
        self.consume("writer", result, "encoder")
        torch.testing.assert_close(result[0], before, rtol=0, atol=0)
        # A new observer has no writer ledger or file reads: only wire bytes are reused.
        diag._observer = None
        diag.request_start("reader", 1, "deepseek_vl_v2")
        loaded = unpack_object(buffer)
        diag.serialized("reader", plan_object(loaded), buffer, loading=True)
        self.consume("reader", loaded, "remote")
        self.assertIsNone(loaded[1])
        self.assertEqual(loaded[0].device.type, torch.device(device).type)
        events = self.events()
        self.assertFalse(any(e["event"] == "unproven" for e in events))
        self.assertTrue(all(e["exact"] for e in events if "exact" in e))
        self.assertEqual(
            [e["event"] for e in events if e["trace_id"] == "reader"],
            [
                "request_start",
                "unpack_slices",
                "tuple_delivered",
                "llm_inputs",
                "llm_copy",
            ],
        )
        self.assertEqual(events[3]["tensor"]["sha256"], events[-1]["tensor"]["sha256"])

    def test_cpu_pack_unpacked_tuple_and_real_llm_copy_are_exact(self):
        self.roundtrip("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_pack_restores_gpu_and_real_llm_copy_is_exact(self):
        self.roundtrip("cuda:0")

    def test_off_has_no_snapshot_ledger_or_file_side_effect(self):
        os.environ["DEEPSEEK_EMB_DIAGNOSTICS"] = "0"
        with patch.object(
            diag, "_snapshot", side_effect=AssertionError("must not snapshot")
        ):
            result = self.vision_result()
            diag.delivered("writer", result, "encoder")
            self.assertIsNone(
                diag.model_begin([result[0]], torch.tensor([0]), "deepseek_vl_v2", 1)
            )
        self.assertEqual(self.events(), [])
        self.assertIsNone(diag._observer)

    def test_observation_on_and_off_return_identical_production_embeddings(self):
        enabled_result = self.vision_result()
        self.consume("writer", enabled_result, "encoder")
        events_before = self.events()
        os.environ["DEEPSEEK_EMB_DIAGNOSTICS"] = "0"
        disabled_result = self.vision_result()
        torch.testing.assert_close(
            enabled_result[0], disabled_result[0], rtol=0, atol=0
        )
        self.assertIsNone(disabled_result[1])
        self.assertEqual(self.events(), events_before)

    def test_other_model_has_no_diagnostic_ledger(self):
        diag.request_start("qwen", 1, "qwen3_vl")
        self.assertIsNone(
            diag.model_begin([torch.ones(2, 4)], torch.tensor([0]), "qwen3_vl", 1)
        )
        self.assertEqual(self.events(), [])

    def test_missing_vision_or_projector_cannot_prove_tuple(self):
        diag.request_start("x", 1, "deepseek_vl_v2")
        diag.encoder_tuple("x", (torch.ones(2, 4), None))
        self.assertTrue(any(e["event"] == "unproven" for e in self.events()))
        self.assertFalse(self.events()[-1]["exact"])

    def test_positions_or_deepstack_are_not_silently_accepted(self):
        for result in (
            (torch.ones(2, 4), torch.ones(2, 3)),
            (torch.ones(2, 4), None, torch.ones(1)),
        ):
            diag._observer = None
            diag.request_start("x", 1, "deepseek_vl_v2")
            diag.encoder_tuple("x", result)
        self.assertTrue(all(e["event"] != "encoder_tuple" for e in self.events()))
        self.assertEqual(sum(e["event"] == "unproven" for e in self.events()), 2)

    def test_corrupt_packed_slice_is_not_exact(self):
        result = self.vision_result()
        plan = plan_object(result)
        buffer = pack_object(plan)
        buffer[-1] ^= 1
        diag.serialized("writer", plan, buffer)
        self.assertFalse(self.events()[-1]["exact"])
        self.assertTrue(diag._observer.requests["writer"]["unproven"])

    def test_wrong_delivered_feature_or_wrong_copy_is_not_exact(self):
        result = self.vision_result()
        diag.delivered("writer", result, "encoder")
        changed = result[0] + 1
        ticket = diag.model_begin([changed], torch.tensor([0]), "deepseek_vl_v2", 1)
        diag.embedding_copied(ticket, changed + 2)
        self.assertFalse(self.events()[-1]["exact"])
        self.assertTrue(self.events()[-1]["unproven"])

    def test_multiple_inflight_images_or_tp_are_unproven(self):
        diag.request_start("x", 2, "deepseek_vl_v2")
        self.assertEqual(diag._observer.requests, {})
        diag.request_start("a", 1, "deepseek_vl_v2")
        diag.request_start("b", 1, "deepseek_vl_v2")
        self.assertTrue(diag._observer.requests["a"]["unproven"])
        diag.model_begin([torch.ones(2, 4)], torch.tensor([0]), "deepseek_vl_v2", 2)
        self.assertEqual(
            self.events()[-1]["reason"], "single_image_segment_tp1_required"
        )

    def test_exception_and_broken_logging_do_not_escape_into_model(self):
        diag.request_start("a", 1, "deepseek_vl_v2")
        with patch.object(
            diag.Observer, "emit", side_effect=OSError("full")
        ), patch.object(diag.logging, "warning", side_effect=RuntimeError("handler")):
            diag.encoder_tuple("a", (torch.ones(2, 4), None))
        self.assertTrue(diag._observer.requests["a"]["unproven"])

    def test_deferred_cpu_eviction_after_llm_copy_preserves_origin(self):
        result = self.vision_result()
        origin = diag.eviction_origin("writer")
        self.consume("writer", result, "encoder")
        self.assertEqual(diag._observer.requests, {})
        cpu = (result[0].cpu().clone(), None)
        plan = plan_object(cpu, devices=(result[0].device, None))
        diag.eviction_serialized("writer", origin, plan, pack_object(plan))
        event = self.events()[-1]
        self.assertEqual(event["event"], "eviction_pack_slices")
        self.assertTrue(event["exact"])
        self.assertEqual(event["origin_event"], "encoder_tuple")
        self.assertEqual(
            event["tensors"][0]["origin_tensor"]["sha256"],
            event["tensors"][0]["staging_tensor"]["sha256"],
        )
        self.assertEqual(diag._observer.requests, {})

    def test_deferred_missing_or_mutated_origin_never_proves_bytes(self):
        result = self.vision_result()
        origin = diag.eviction_origin("writer")
        self.consume("writer", result, "encoder")
        plan = plan_object((result[0].clone() + 1, None))
        buffer = pack_object(plan)
        diag.eviction_serialized("writer", origin, plan, buffer)
        self.assertFalse(self.events()[-1]["exact"])
        diag.eviction_serialized("writer", None, plan, buffer)
        self.assertEqual(self.events()[-1]["event"], "unproven")

    def test_short_writes_produce_complete_json(self):
        original = os.write
        with patch.object(
            diag.os, "write", side_effect=lambda fd, data: original(fd, data[:7])
        ):
            diag.request_start("a", 1, "deepseek_vl_v2")
        self.assertEqual(len(self.events()), 1)


class TextEmbedding(nn.Embedding):
    def forward(
        self, inputs, position_ids=None, token_types=None, text_tokens_mask=None
    ):
        return super().forward(inputs)


class GenericMoeImageInjectionTest(unittest.TestCase):
    def model(self, multimodal):
        model = object.__new__(GenericMoeModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(model_type="deepseek_vl_v2")
        model.parallelism_config = SimpleNamespace(tp_size=1)
        model.embed_tokens = TextEmbedding.from_pretrained(
            torch.arange(40, dtype=torch.float32).reshape(10, 4)
        )
        model.multimodal_embedding_injector = (
            MultimodalEmbeddingInjector() if multimodal else None
        )
        model.layer_num = 0
        model.layers = nn.ModuleList()
        model.norm = lambda hidden, residual: (hidden, residual)
        return model

    def test_text_only_embedding_does_not_read_multimodal_inputs(self):
        model = self.model(False)
        inputs = SimpleNamespace(input_ids=torch.tensor([0, 1, 2, 3]))
        expected = model.embed_tokens(inputs.input_ids)
        actual = model.embedding(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_two_different_images_are_actually_consumed_by_embedding(self):
        model = self.model(True)
        tokens = torch.tensor([0, 1, 2, 3])
        original = model.embed_tokens(tokens)
        for value in (13.0, -7.0):
            feature = torch.full((2, 4), value)
            inputs = SimpleNamespace(
                input_ids=tokens,
                combo_position_ids=None,
                embedding_inputs=SimpleNamespace(
                    combo_tokens_type_ids=None, text_tokens_mask=None
                ),
                multimodal_inputs=SimpleNamespace(
                    multimodal_features=[feature],
                    mm_features_locs=torch.tensor([1], dtype=torch.int32),
                ),
            )
            actual = model.embedding(inputs)
            expected = original.clone()
            expected[1:3] = feature
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_native_masked_lookup_skips_feature_hash_ids_before_real_injection(self):
        model = self.model(True)
        weight = (
            torch.arange(32 * 1280, dtype=torch.float32, device="cuda")
            .reshape(32, 1280)
            .to(torch.bfloat16)
        )
        model.embed_tokens = Embedding(model.config, ParallelismConfig(), weight)
        # These image-slot hashes are deliberately outside the vocabulary. The
        # original native mask must skip them; never launch an unmasked lookup.
        ids = torch.tensor(
            [1, 2147483647, -2147483648, 2], dtype=torch.int32, device="cuda"
        )
        mask = torch.tensor([1, 0, 0, 1], dtype=torch.int32, device="cuda")
        feature = (
            torch.arange(2 * 1280, dtype=torch.float32, device="cuda")
            .reshape(2, 1280)
            .to(torch.bfloat16)
        )
        inputs = PyModelInputs(input_ids=ids)
        inputs.embedding_inputs.text_tokens_mask = mask
        inputs.multimodal_inputs.multimodal_features = [feature]
        inputs.multimodal_inputs.mm_features_locs = torch.tensor([1], dtype=torch.int32)
        actual = model.embedding(inputs)
        torch.cuda.synchronize()
        expected = torch.cat([weight[1:2], feature, weight[2:3]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            ids,
            torch.tensor(
                [1, 2147483647, -2147483648, 2], dtype=torch.int32, device="cuda"
            ),
        )

    def test_multimodal_decode_without_features_keeps_text_embedding(self):
        model = self.model(True)
        inputs = SimpleNamespace(
            input_ids=torch.tensor([2]),
            combo_position_ids=None,
            embedding_inputs=SimpleNamespace(
                combo_tokens_type_ids=None, text_tokens_mask=None
            ),
            multimodal_inputs=SimpleNamespace(
                multimodal_features=[],
                mm_features_locs=torch.empty(0, dtype=torch.int32),
            ),
        )
        actual = model.embedding(inputs)
        torch.testing.assert_close(
            actual, model.embed_tokens(inputs.input_ids), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
