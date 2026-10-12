"""Explicit eager routing, isolation, restoration and changing graphs on SM120."""

import threading
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.fp8.test.v41_router_eager_reference import (
    router_eager_reference,
)
from rtp_llm.models_py.modules.dsv4.moe import gate as native_module
from rtp_llm.utils.model_weight import W


def make_gate(width=512):
    return native_module.Gate(
        8,
        width,
        384,
        6,
        "sqrtsoftplus",
        1.5,
        layer_weights={
            W.v4_router_w: torch.randn(384, width, device="cuda", dtype=torch.bfloat16)
            / width**0.5,
            W.v4_router_bias: torch.randn(384, device="cuda", dtype=torch.float32) / 16,
        },
        fp32_logits=True,
    )


@unittest.skipUnless(torch.cuda.is_available(), "Requires SM120")
class RouterEagerReferenceTest(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(41307)
        torch.set_float32_matmul_precision("highest")
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(native_module._use_fused_gate("sqrtsoftplus", 1))
        self.owner, self.other = make_gate(), make_gate()
        self.x = torch.randn(5, 512, device="cuda", dtype=torch.bfloat16)

    def exact(self, left, right):
        if isinstance(left, tuple):
            self.assertEqual(len(left), len(right))
            for expected, actual in zip(left, right):
                self.exact(expected, actual)
            return
        self.assertEqual(left.shape, right.shape)
        self.assertEqual(left.dtype, right.dtype)
        self.assertTrue(
            torch.equal(
                left.contiguous().reshape(-1).view(torch.uint8),
                right.contiguous().reshape(-1).view(torch.uint8),
            )
        )

    def eager(self, owner, value):
        with patch.object(native_module, "_use_fused_gate", return_value=False):
            return tuple(t.clone() for t in owner(value))

    @torch.inference_mode()
    def test_both_widths_selected_and_unselected_owners(self):
        original_forward = native_module.Gate._forward_local
        original_policy = native_module._use_fused_gate
        other = tuple(t.clone() for t in self.other(self.x))
        for width in (512, 5120):
            owner = self.owner if width == 512 else make_gate(width)
            value = (
                self.x
                if width == 512
                else torch.randn(5, width, device="cuda", dtype=torch.bfloat16)
            )
            before = [t.clone() for t in (value, owner.weight, owner.bias)]
            baseline = tuple(t.clone() for t in owner(value))
            expected = self.eager(owner, value)
            with patch.object(
                native_module, "_use_fused_gate", side_effect=original_policy
            ) as delegated:
                with router_eager_reference(True, gates=[owner]):
                    self.exact(expected, owner(value))
                    self.exact(other, self.other(self.x))
                delegated.assert_called_once_with("sqrtsoftplus", len(self.x))
            self.exact(baseline, owner(value))
            for retained, current in zip(before, (value, owner.weight, owner.bias)):
                self.exact(retained, current)
        self.assertIs(native_module.Gate._forward_local, original_forward)
        self.assertIs(native_module._use_fused_gate, original_policy)

    @torch.inference_mode()
    def test_disabled_nested_and_exception_restoration(self):
        original_forward = native_module.Gate._forward_local
        original_policy = native_module._use_fused_gate
        expected, other_expected = self.eager(self.owner, self.x), self.eager(
            self.other, self.x
        )
        with router_eager_reference(False, gates=[object()]):
            self.assertIs(native_module.Gate._forward_local, original_forward)
            self.assertIs(native_module._use_fused_gate, original_policy)
        with router_eager_reference(True, gates=[self.owner]):
            outer_forward, outer_policy = (
                native_module.Gate._forward_local,
                native_module._use_fused_gate,
            )
            with router_eager_reference(False):
                self.assertIs(native_module.Gate._forward_local, outer_forward)
                self.assertIs(native_module._use_fused_gate, outer_policy)
            with router_eager_reference(True, gates=[self.other]):
                self.exact(expected, self.owner(self.x))
                self.exact(other_expected, self.other(self.x))
            self.assertIs(native_module.Gate._forward_local, outer_forward)
            self.assertIs(native_module._use_fused_gate, outer_policy)
            with self.assertRaisesRegex(RuntimeError, "scope-control"):
                with router_eager_reference(True, gates=[self.other]):
                    raise RuntimeError("scope-control")
            with patch.object(F, "linear", side_effect=RuntimeError("call-control")):
                with self.assertRaisesRegex(RuntimeError, "call-control"):
                    self.owner(self.x)
            self.assertTrue(native_module._use_fused_gate("sqrtsoftplus", 1))
            self.exact(expected, self.owner(self.x))
        self.assertIs(native_module.Gate._forward_local, original_forward)
        self.assertIs(native_module._use_fused_gate, original_policy)

    def test_rejected_contracts_leave_methods_unchanged(self):
        original_forward = native_module.Gate._forward_local
        original_policy = native_module._use_fused_gate
        for enabled, owners in (
            (1, [self.owner]),
            (True, []),
            (True, [self.owner, self.owner]),
            (True, [object()]),
        ):
            with self.subTest(owners=owners), self.assertRaises(ValueError):
                with router_eager_reference(enabled, gates=owners):
                    pass
        weight, bias = self.owner.weight, self.owner.bias
        for name, value in (
            ("dim", 1024),
            ("dim", 512.0),
            ("hash", True),
            ("fp32_logits", False),
            ("fp32_logits", 1),
            ("score_func", "softmax"),
            ("_v41_cp_ctx", object()),
            ("topk", 0),
            ("topk", 385),
            ("topk", True),
            ("route_scale", float("nan")),
            ("route_scale", float("inf")),
            ("route_scale", 0),
            ("route_scale", True),
            ("weight", weight.float()),
            ("weight", weight.cpu()),
            ("weight", weight.t().contiguous().t()),
            ("weight", weight[:, :256]),
            ("bias", None),
            ("bias", bias.bfloat16()),
            ("bias", bias.cpu()),
            ("bias", bias[:-1]),
            ("bias", torch.empty(768, device="cuda", dtype=torch.float32)[::2]),
        ):
            with self.subTest(name=name, value=type(value)), patch.object(
                self.owner, name, value
            ), self.assertRaisesRegex(ValueError, "admitted SM120"):
                with router_eager_reference(True, gates=[self.owner]):
                    pass
        with router_eager_reference(True, gates=[self.owner]):
            for value in (
                self.x.float(),
                self.x.cpu(),
                self.x[:, :256],
                self.x.t().contiguous().t(),
                self.x.reshape(-1),
            ):
                with self.subTest(shape=value.shape), self.assertRaisesRegex(
                    ValueError, "token rows"
                ):
                    self.owner(value)
        self.assertIs(native_module.Gate._forward_local, original_forward)
        self.assertIs(native_module._use_fused_gate, original_policy)

    @torch.inference_mode()
    def test_empty_rows_and_nonfinite_fallback(self):
        empty = self.x[:0].contiguous()
        bad = self.x.clone()
        bad[0, 0], bad[1, 0] = float("nan"), float("inf")
        expected = self.eager(self.owner, bad)
        with router_eager_reference(True, gates=[self.owner]):
            outputs = self.owner(empty)
            self.assertEqual(outputs[0].shape, (0, 6))
            self.assertEqual(outputs[0].dtype, torch.float32)
            self.assertEqual(outputs[1].dtype, torch.int64)
            actual = self.owner(bad)
            self.exact(expected, actual)
            self.assertTrue(torch.isfinite(actual[0]).all())
            self.exact(actual[0][:2], torch.full_like(actual[0][:2], 0.25))
            self.assertTrue(((actual[1] >= 0) & (actual[1] < 384)).all())

    def test_nested_unselected_call_uses_its_own_policy(self):
        calls = []

        def local(owner, value, input_ids=None):
            before = native_module._use_fused_gate("sqrtsoftplus", len(value))
            if owner is self.owner:
                self.other(value)
            after = native_module._use_fused_gate("sqrtsoftplus", len(value))
            calls.append((owner, before, after))
            return before

        with patch.object(native_module.Gate, "_forward_local", local):
            with router_eager_reference(True, gates=[self.owner]):
                self.assertFalse(self.owner(self.x))
                self.assertTrue(native_module._use_fused_gate("sqrtsoftplus", 1))
        self.assertEqual(calls, [(self.other, True, True), (self.owner, False, False)])

    def test_context_isolated_from_other_thread(self):
        entered, release = threading.Event(), threading.Event()
        calls, errors = [], []

        def local(owner, value, input_ids=None):
            if owner is self.owner:
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("thread-control timed out")
            result = native_module._use_fused_gate("sqrtsoftplus", len(value))
            calls.append((owner, result))
            return result

        def selected_call():
            try:
                self.owner(self.x)
            except BaseException as error:
                errors.append(error)

        with patch.object(native_module.Gate, "_forward_local", local):
            with router_eager_reference(True, gates=[self.owner]):
                worker = threading.Thread(target=selected_call)
                worker.start()
                try:
                    self.assertTrue(entered.wait(10))
                    self.assertTrue(native_module._use_fused_gate("sqrtsoftplus", 1))
                    self.assertTrue(self.other(self.x))
                finally:
                    release.set()
                    worker.join(10)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
        self.assertEqual(calls, [(self.other, True), (self.owner, False)])

    @torch.inference_mode()
    def test_two_graphs_and_six_changed_input_replays(self):
        replay_count = 0
        for width in (512, 5120):
            owner = self.owner if width == 512 else make_gate(width)
            value = torch.randn(5, width, device="cuda", dtype=torch.bfloat16)
            retained = [owner.weight.clone(), owner.bias.clone()]
            other = tuple(t.clone() for t in self.other(self.x))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with router_eager_reference(True, gates=[owner]):
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        owner(value)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = owner(value)
            self.exact(other, self.other(self.x))
            previous = value.clone()
            for _ in range(3):
                value.copy_(torch.randn_like(value))
                self.assertFalse(torch.equal(previous, value))
                before = value.clone()
                expected = self.eager(owner, value)
                graph.replay()
                self.exact(expected, captured)
                self.exact(before, value)
                previous.copy_(value)
                replay_count += 1
            self.exact(retained[0], owner.weight)
            self.exact(retained[1], owner.bias)
        self.assertEqual(replay_count, 6)
        print(
            'DSV41_ROUTER_EAGER_GRAPHS {"graphs": 2, "changed_replays": 6}', flush=True
        )


if __name__ == "__main__":
    unittest.main()
