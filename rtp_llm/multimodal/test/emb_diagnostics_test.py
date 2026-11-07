import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalDeepstackInjector,
    MultimodalEmbeddingInjector,
)
from rtp_llm.multimodal import emb_diagnostics as d
from rtp_llm.multimodal.kvcm.tensor_object import (
    pack_object,
    plan_object,
    unpack_object,
)


class EmbDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.env = patch.dict(
            os.environ,
            {
                "MM_EMB_DIAGNOSTICS": "1",
                "MM_EMB_DIAGNOSTICS_DIR": self.directory.name,
                "MM_EMB_DIAGNOSTICS_RUN_ID": "unit",
            },
        )
        self.env.start()
        d._observer = None
        self.trace = "unit-trace"

    def tearDown(self):
        d._observer = None
        self.env.stop()
        self.directory.cleanup()

    def events(self):
        path = Path(self.directory.name) / "emb-chain.jsonl"
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )

    def values(self, device="cpu"):
        e = (torch.arange(24, device=device).reshape(4, 6).float() / 16).to(
            torch.bfloat16
        )
        deep = [e + (i + 1) / 8 for i in range(3)]
        positions = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]],
            dtype=torch.int32,
            device=device,
        )
        grid = torch.tensor([[1, 4, 4]], dtype=torch.int64, device=device)
        return (
            (e, positions, torch.stack(deep).flatten()),
            SimpleNamespace(pooler_output=e, deepstack_features=deep),
            grid,
        )

    def encode(self, device="cpu"):
        result, vision, grid = self.values(device)
        d.request_start(self.trace, 1)
        wi = SimpleNamespace(emb_diagnostic_trace=self.trace)
        with d.encoder_scope([wi]):
            d.vision_output(vision, grid, 2)
        d.encoder_tuple(self.trace, result)
        return result

    def deliver(self, device="cpu"):
        result = self.encode(device)
        d.delivered(self.trace, result, "encoder")
        return result

    def test_deferred_eviction_keeps_three_tensor_origins_without_active_ledger(self):
        result = self.encode()
        origin = d.eviction_origin(self.trace)
        # A completed request no longer owns a live model ticket.
        d._observer.requests.clear()
        cpu = tuple(t.cpu().clone() for t in result)
        plan = plan_object(cpu, devices=tuple(t.device for t in result))
        d.eviction_serialized(self.trace, origin, plan, pack_object(plan))
        event = self.events()[-1]
        self.assertEqual(event["event"], "eviction_pack_slices")
        self.assertEqual(len(event["tensors"]), 3)
        self.assertTrue(event["exact"])
        bad = pack_object(plan)
        bad[-1] ^= 1
        d.eviction_serialized(self.trace, origin, plan, bad)
        self.assertFalse(self.events()[-1]["exact"])
        self.assertEqual(d._observer.requests, {})

    def test_disabled_does_not_allocate_copy_or_create_file(self):
        with patch.dict(os.environ, {"MM_EMB_DIAGNOSTICS": "0"}), patch.object(
            d, "_snapshot", side_effect=AssertionError("must not snapshot")
        ):
            d.request_start("off", 1)
            d.encoder_tuple("off", (None, None, None))
            with d.model_scope([], [], None, None, 1) as ticket:
                self.assertIsNone(ticket)
        self.assertEqual(self.events(), [])

    def test_flag_without_directory_is_disabled(self):
        with patch.dict(os.environ, {"MM_EMB_DIAGNOSTICS_DIR": ""}):
            self.assertFalse(d.enabled())

    def test_vision_tuple_pack_exact_and_inputs_unchanged(self):
        result = self.encode()
        original = [t.clone() for t in result]
        pointers = [t.data_ptr() for t in result]
        plan = plan_object(result)
        packed = pack_object(plan)
        saved = packed.clone()
        d.serialized(self.trace, plan, packed, result)
        self.assertEqual(pointers, [t.data_ptr() for t in result])
        for old, current in zip(original, result):
            self.assertTrue(torch.equal(old, current))
        self.assertTrue(torch.equal(packed, saved))
        self.assertFalse(any(e["event"] == "unproven" for e in self.events()))
        event = self.events()[-1]
        self.assertEqual(event["event"], "pack_slices")
        self.assertTrue(
            all(c["exact"] and c["origin_exact"] for c in event["comparisons"])
        )

    def test_pack_corrupt_slice_is_detected(self):
        result = self.encode()
        plan = plan_object(result)
        buffer = pack_object(plan)
        buffer[plan.payload_offset] ^= 1
        d.serialized(self.trace, plan, buffer, result)
        self.assertTrue(
            any(e.get("reason") == "serialized_tensor_mismatch" for e in self.events())
        )

    def test_mutated_encoder_tensor_before_pack_is_detected(self):
        result = self.encode()
        result[0][0, 0] += 1
        plan = plan_object(result)
        d.serialized(self.trace, plan, pack_object(plan), result)
        self.assertTrue(
            any(e.get("reason") == "tensor_changed_before_pack" for e in self.events())
        )

    def test_reader_has_no_writer_ledger_or_file_dependency(self):
        result, _, _ = self.values()
        buffer = pack_object(plan_object(result))
        d.request_start(self.trace, 1)
        with patch.object(
            Path, "read_text", side_effect=AssertionError("no writer files")
        ):
            restored = unpack_object(buffer)
            d.serialized(
                self.trace, plan_object(restored), buffer, restored, loading=True
            )
            d.delivered(self.trace, restored, "remote")
        self.assertFalse(any(e["event"] == "unproven" for e in self.events()))
        self.assertEqual(
            [e["event"] for e in self.events()],
            ["request_start", "unpack_slices", "tuple_delivered"],
        )

    def test_wrong_restored_value_detected(self):
        result, _, _ = self.values()
        buffer = pack_object(plan_object(result))
        restored = list(unpack_object(buffer))
        # CPU unpack views source buffer: clone before deliberately corrupting
        # restored value so the comparison retains independent wire evidence.
        restored[0] = restored[0].clone() + 1
        d.request_start(self.trace, 1)
        d.serialized(self.trace, plan_object(restored), buffer, restored, loading=True)
        self.assertTrue(
            any(e.get("reason") == "serialized_tensor_mismatch" for e in self.events())
        )

    def test_wrong_restored_shape_detected_even_if_bytes_equal(self):
        result, _, _ = self.values()
        buffer = pack_object(plan_object(result))
        restored = list(unpack_object(buffer))
        restored[0] = restored[0].flatten()
        d.request_start(self.trace, 1)
        d.serialized(self.trace, plan_object(restored), buffer, restored, loading=True)
        self.assertTrue(
            any(e.get("reason") == "serialized_tensor_mismatch" for e in self.events())
        )

    def test_wrong_positions_from_grid_are_detected(self):
        result, vision, grid = self.values()
        d.request_start(self.trace, 1)
        with d.encoder_scope([SimpleNamespace(emb_diagnostic_trace=self.trace)]):
            d.vision_output(vision, grid, 2)
        result[1][0, 0] = 1
        d.encoder_tuple(self.trace, result)
        self.assertTrue(
            any(e.get("reason") == "vision_to_tuple_mismatch" for e in self.events())
        )

    def test_concurrent_requests_are_unproven(self):
        d.request_start("a", 1)
        d.request_start("b", 1)
        failures = [e for e in self.events() if e["event"] == "unproven"]
        self.assertEqual({e["trace_id"] for e in failures}, {"a", "b"})

    def test_multiple_images_are_unproven(self):
        d.request_start("multi", 2)
        self.assertEqual(self.events()[0]["reason"], "single_image_required")

    def _consume(self, device="cpu", corrupt=None, split=False):
        result = self.deliver(device)
        e, positions, extra = result
        deep = extra.reshape(3, 4, 6)
        features = [e[:2], e[2:]] if split else [e]
        stacks = [deep[:, :2], deep[:, 2:]] if split else [deep]
        locs = torch.tensor([2, 4] if split else [2], dtype=torch.int32, device=device)
        combo = torch.zeros(8, 3, dtype=torch.int32, device=device)
        combo[2:6] = positions + 2
        config = SimpleNamespace(
            style=7,
            dim=128,
            base=5000000.0,
            scale=1.0,
            index_factor=3,
            mrope_dim1=24,
            mrope_dim2=20,
            mrope_dim3=20,
            mrope_interleaved=True,
        )
        inputs = torch.zeros(8, 6, dtype=torch.bfloat16, device=device)
        reference = inputs.clone()
        if corrupt == "positions":
            combo[2, 0] += 1
        with d.model_scope(features, stacks, locs, combo, 1) as ticket:
            hidden = MultimodalEmbeddingInjector()(inputs, features, locs)
            reference = MultimodalEmbeddingInjector()(reference, features, locs)
            if corrupt == "copy":
                hidden[2, 0] += 1
            d.embedding_copied(ticket, hidden)
            d.rope_input(combo.flatten(), config, torch.zeros(1, dtype=torch.int32))
            for layer in range(3):
                before = d.before_deepstack(ticket, hidden, layer)
                hidden = MultimodalDeepstackInjector()(
                    hidden, stacks, locs.cpu().tolist(), layer
                )
                reference = MultimodalDeepstackInjector()(
                    reference, stacks, locs.cpu().tolist(), layer
                )
                if corrupt == "add" and layer == 1:
                    hidden[2, 0] += 1
                d.after_deepstack(ticket, before, hidden, layer)
        if corrupt is None:
            self.assertTrue(torch.equal(hidden, reference))
            self.assertEqual(hidden.data_ptr(), inputs.data_ptr())
            self.assertFalse(
                any(e["event"] == "unproven" for e in self.events()), self.events()
            )
            self.assertEqual(
                [e["layer"] for e in self.events() if e["event"] == "deepstack_add"],
                [0, 1, 2],
            )
            self.assertEqual(self.events()[-1]["event"], "llm_complete")
        return self.events()

    def test_actual_injectors_unchanged_and_all_steps_present(self):
        self._consume()

    def test_cp1_segments_reassemble_layer_major_deepstack(self):
        self._consume(split=True)

    def test_embedding_copy_error_detected(self):
        self.assertTrue(
            any(
                e.get("reason") == "embedding_copy_mismatch"
                for e in self._consume(corrupt="copy")
            )
        )

    def test_deepstack_add_error_detected(self):
        self.assertTrue(
            any(
                e.get("reason") == "deepstack_add_mismatch"
                for e in self._consume(corrupt="add")
            )
        )

    def test_llm_position_offset_error_detected(self):
        self.assertTrue(
            any(
                e.get("reason") == "llm_input_mismatch"
                for e in self._consume(corrupt="positions")
            )
        )

    def test_missing_steps_do_not_pass_completion(self):
        e, pos, extra = self.deliver()
        with d.model_scope([e], [extra.reshape(3, 4, 6)], torch.tensor([0]), pos, 1):
            pass
        self.assertTrue(
            any(e.get("reason") == "missing_llm_observation" for e in self.events())
        )

    def test_observer_failure_does_not_escape(self):
        d.request_start(self.trace, 1)
        with patch.object(d, "_snapshot", side_effect=RuntimeError("private-content")):
            d.encoder_tuple(self.trace, self.values()[0])
        self.assertTrue(
            any(e.get("reason") == "diagnostic_exception" for e in self.events())
        )
        self.assertNotIn("private-content", json.dumps(self.events()))
        self.assertTrue(d._observer.requests[self.trace]["unproven"])
        self.assertTrue(all(e["trace_id"] == self.trace for e in self.events()))

    def test_broken_logging_does_not_escape(self):
        d.request_start(self.trace, 1)
        with patch.object(
            d.Observer, "emit", side_effect=OSError("disk full")
        ), patch.object(d.logging, "warning", side_effect=RuntimeError("bad handler")):
            d.encoder_tuple(self.trace, self.values()[0])
        self.assertTrue(d._observer.requests[self.trace]["unproven"])

    def test_graph_configuration_is_unproven(self):
        with patch.dict(os.environ, {"ENABLE_CUDA_GRAPH": "true"}):
            d.request_start(self.trace, 1)
        self.assertEqual(
            self.events()[0]["reason"], "cuda_graph_configuration_unsupported"
        )

    def test_qwen_model_hook_off_on_preserves_forward_output(self):
        from rtp_llm.models_py.model_desc.hidden_state_capture import CaptureContext
        from rtp_llm.models_py.model_desc.module_base import GptModelBase
        from rtp_llm.models_py.model_desc.qwen3vl import Qwen3VLModel

        e, positions, extra = self.deliver()
        combo = torch.zeros(8, 3, dtype=torch.int32)
        combo[2:6] = positions + 2
        config = SimpleNamespace(
            style=7,
            dim=128,
            base=5000000.0,
            scale=1.0,
            index_factor=3,
            mrope_dim1=24,
            mrope_dim2=20,
            mrope_dim3=20,
            mrope_interleaved=True,
        )

        def layer(hidden, fmha, kv_cache=None):
            d.rope_input(combo.flatten(), config, torch.zeros(1, dtype=torch.int32))
            return hidden * 1.125

        model = SimpleNamespace(
            embed_tokens=lambda *args: torch.zeros(8, 6, dtype=torch.bfloat16),
            multimodal_embedding_injector=MultimodalEmbeddingInjector(),
            multimodal_deepstack_injector=MultimodalDeepstackInjector(),
            layers=[layer] * 3,
            layer_num=3,
            kv_cache=None,
            norm=lambda value: value,
            parallelism_config=SimpleNamespace(tp_size=1),
        )
        # Use the baseline's real capture plan and canonical hooks; diagnostics
        # must preserve ordinary forward output through that existing path.
        model._capture_context = CaptureContext.configured(
            (),
            lambda hidden, residual: Qwen3VLModel._capture_canonical_layer(
                model, hidden, residual
            ),
            lambda hidden, residual: Qwen3VLModel._capture_canonical_final(
                model, hidden, residual
            ),
        )
        model.capture_context = lambda requested: GptModelBase.capture_context(
            model, requested
        )
        inputs = SimpleNamespace(
            capture_hidden_states=False,
            input_ids=torch.zeros(8, dtype=torch.int32),
            combo_position_ids=combo,
            embedding_inputs=SimpleNamespace(
                combo_tokens_type_ids=None, text_tokens_mask=None
            ),
            multimodal_inputs=SimpleNamespace(
                multimodal_features=[e],
                mm_features_locs=torch.tensor([2], dtype=torch.int32),
                mm_extra_input=[extra],
            ),
        )
        with patch.dict(os.environ, {"MM_EMB_DIAGNOSTICS": "0"}):
            # Disabled hooks do not even require a diagnostics-only config read.
            del model.parallelism_config
            reference = Qwen3VLModel.forward(
                model, inputs, fmha_impl=object()
            ).hidden_states.clone()
        model.parallelism_config = SimpleNamespace(tp_size=1)
        actual = Qwen3VLModel.forward(model, inputs, fmha_impl=object()).hidden_states
        self.assertTrue(torch.equal(reference, actual))
        self.assertFalse(any(event["event"] == "unproven" for event in self.events()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_actual_injection_and_add_exact(self):
        self._consume("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_vision_pack_restore_exact(self):
        result = self.encode("cuda")
        plan = plan_object(result)
        buffer = pack_object(plan)
        d.serialized(self.trace, plan, buffer, result)
        restored = unpack_object(buffer, cuda_device="cuda")
        d.serialized(self.trace, plan_object(restored), buffer, restored, loading=True)
        for a, b in zip(result, restored):
            self.assertTrue(torch.equal(a, b))
        self.assertFalse(any(e["event"] == "unproven" for e in self.events()))


if __name__ == "__main__":
    unittest.main()
