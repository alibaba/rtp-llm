"""Model boundary ownership, nested MTP, and live CUDA Graph observations."""

import importlib.util
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


recorder = load("k3_model_test_recorder", "k3_tensor_trace.py")
with patch.dict(sys.modules, {"rtp_llm.utils.k3_tensor_trace": recorder}):
    tracing = load("k3_model_under_test", "k3_model_trace.py")


def inputs(value):
    return types.SimpleNamespace(
        input_ids=value,
        input_hiddens=None,
        combo_position_ids=None,
        attention_inputs=types.SimpleNamespace(
            is_prefill=False,
            is_target_verify=False,
            input_lengths=torch.tensor([value.shape[0]]),
        ),
    )


class ToyModel(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(2))
        tracing.install_model_trace(self, name)

    def forward(self, inputs):
        value = self.projection(inputs.input_ids)
        tracing.record_model("explicit_projection", value)
        value.add_(5)
        return types.SimpleNamespace(hidden_states=value)


class ModelTraceTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        env = patch.dict(
            os.environ,
            {"K3_TRACE_ROOT": str(self.root), "K3_TRACE_RUN_ID": "model-test"},
        )
        env.start()
        self.addCleanup(env.stop)
        managers = patch.object(tracing, "_managers", [])
        managers.start()
        self.addCleanup(managers.stop)
        self.addCleanup(tracing.close_models)

    def frames(self, model="main"):
        return [
            torch.load(path, weights_only=True)
            for path in sorted(self.root.glob(f"model-{model}-*/frame-*.pt"))
        ]

    def test_module_snapshot_precedes_in_place_overwrite(self):
        model = ToyModel("main")
        model(inputs(torch.tensor([[1.0, 2.0]])))
        tracing.close_models()
        frame = self.frames()[0]
        tensors = {item["name"]: item["value"] for item in frame["tensors"]}
        torch.testing.assert_close(
            tensors["main.projection.output"], torch.tensor([[1.0, 2.0]])
        )
        torch.testing.assert_close(
            tensors["main.output.hidden_states"], torch.tensor([[6.0, 7.0]])
        )

    def test_warmup_does_not_produce_request_frames(self):
        model = ToyModel("main")
        model._k3_trace_warmup(True)
        model(inputs(torch.ones(1, 2)))
        model._k3_trace_warmup(False)
        model(inputs(torch.ones(1, 2)))
        tracing.close_models()
        self.assertEqual(len(self.frames()), 1)

    def test_module_replaced_during_initialization_is_observed(self):
        model = ToyModel("main")
        model.projection = torch.nn.Identity()
        model(inputs(torch.ones(1, 2)))
        tracing.close_models()
        names = [item["name"] for item in self.frames()[0]["tensors"]]
        self.assertIn("main.projection.output", names)

    def test_nested_draft_uses_the_active_main_frame(self):
        main, draft = ToyModel("main"), ToyModel("mtp")
        manager = main._k3_trace_replay.__self__

        def operation(value):
            return draft(value)

        manager.warmup(True)
        manager.forward(operation, inputs(torch.ones(1, 2)))
        manager.warmup(False)
        manager.forward(operation, inputs(torch.ones(1, 2)))
        tracing.close_models()
        names = [item["name"] for item in self.frames()[0]["tensors"]]
        self.assertIn("mtp.projection.output", names)
        self.assertIn("main.output.hidden_states", names)
        self.assertEqual(self.frames("mtp"), [])
        self.assertEqual(len(self.frames()), 1)

    def test_serving_can_follow_initialization_on_another_thread(self):
        model = ToyModel("main")
        errors = []

        def serve():
            try:
                model(inputs(torch.ones(1, 2)))
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=serve)
        thread.start()
        thread.join()
        self.assertEqual(errors, [])
        tracing.close_models()
        self.assertEqual(len(self.frames()), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA Graph")
    def test_each_replay_has_current_values_and_live_inputs(self):
        model = ToyModel("main").cuda()
        value = torch.ones(1, 2, device="cuda")
        model._k3_trace_warmup(True)
        model(inputs(value))
        model._k3_trace_warmup(False)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        model._k3_trace_capture_begin("runner0:1")
        with torch.cuda.graph(graph):
            model(inputs(value))
        model._k3_trace_capture_end()
        for step in range(3):
            value.fill_(step)
            graph.replay()
            model._k3_trace_replay("runner0:1", inputs(value))
        tracing.close_models()
        for step, frame in enumerate(self.frames()):
            tensors = {item["name"]: item["value"] for item in frame["tensors"]}
            torch.testing.assert_close(
                tensors["main.projection.output"], torch.full((1, 2), float(step))
            )
            torch.testing.assert_close(
                tensors["input.input_ids"], torch.full((1, 2), float(step))
            )
        self.assertEqual(len(self.frames()), 3)


if __name__ == "__main__":
    unittest.main()
