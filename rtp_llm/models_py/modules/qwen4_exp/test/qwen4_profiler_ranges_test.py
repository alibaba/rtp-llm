import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

import rtp_llm.models_py.model_desc.qwen4_exp as qwen4_exp_module
from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpDecoderLayer
from rtp_llm.models_py.modules.qwen4_exp import _profiler


class _RecordingUnit:
    """Stand-in for a hyper-connection unit that records its call."""

    def __init__(self, name, events):
        self._name = name
        self._events = events

    def __call__(self, hyper_states):
        self._events.append(("call", self._name))
        return hyper_states, hyper_states, None


class _RecordingModule:
    def __init__(self, name, events):
        self._name = name
        self._events = events

    def __call__(self, *args, **kwargs):
        self._events.append(("call", self._name))
        return args[0] if args else None


class Qwen4ProfilerRangesTest(unittest.TestCase):
    def test_factory_is_noop_without_active_profiler(self):
        with patch.object(_profiler, "_RANGES_ENABLED", True), patch.object(
            torch.autograd, "_profiler_enabled", return_value=False
        ), patch.object(torch._C._profiler, "_RecordFunctionFast") as record_function:
            layer_range = _profiler.make_layer_forward_range()
            with layer_range(3):
                pass

        record_function.assert_not_called()

    def test_factory_uses_qwen4_layer_names(self):
        names = []

        @contextmanager
        def record_function_fast(name):
            names.append(name)
            yield

        with patch.object(_profiler, "_RANGES_ENABLED", True), patch.object(
            torch.autograd, "_profiler_enabled", return_value=True
        ), patch.object(
            torch._C._profiler,
            "_RecordFunctionFast",
            record_function_fast,
        ):
            layer_range = _profiler.make_layer_forward_range()
            with layer_range(7):
                pass

        self.assertEqual(names, ["qwen4.layer7"])

    def test_factory_respects_outer_disable(self):
        with patch.object(_profiler, "_RANGES_ENABLED", True), patch.object(
            torch.autograd, "_profiler_enabled", return_value=True
        ), patch.object(torch._C._profiler, "_RecordFunctionFast") as record_function:
            with _profiler.disable_record_function_ranges():
                layer_range = _profiler.make_layer_forward_range()
            with layer_range(2):
                pass

        record_function.assert_not_called()

    def test_layer_range_is_cpu_function_scope(self):
        with patch.object(_profiler, "_RANGES_ENABLED", True), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            layer_range = _profiler.make_layer_forward_range()
            with layer_range(5):
                torch.ones(1)

        layer_events = [
            event for event in prof.events() if event.name == "qwen4.layer5"
        ]
        self.assertEqual(len(layer_events), 1)
        self.assertEqual(
            layer_events[0].scope,
            torch.profiler.RecordScope.FUNCTION.value,
        )
        self.assertFalse(layer_events[0].is_user_annotation)

    def _make_layer(self, events):
        layer = Qwen4ExpDecoderLayer.__new__(Qwen4ExpDecoderLayer)
        layer.attn_hyper_connection = _RecordingUnit("hc_attn", events)
        layer.mlp_hyper_connection = _RecordingUnit("hc_mlp", events)
        layer.self_attn = _RecordingModule("attn", events)
        layer.mlp = _RecordingModule("mlp", events)
        layer._attn_profile_range = "qwen4.gdn"
        return layer

    @staticmethod
    def _recording_range(events):
        @contextmanager
        def record_function_range(name):
            events.append(("range", name))
            try:
                yield
            finally:
                events.append(("range_exit", name))

        return record_function_range

    def test_decoder_layer_ranges_wrap_each_block_in_order(self):
        events = []
        layer = self._make_layer(events)

        with patch.object(
            qwen4_exp_module,
            "record_function_range",
            self._recording_range(events),
        ), patch.object(
            qwen4_exp_module,
            "inject_into_residual",
            lambda hyper_states, output, gate: (hyper_states, output, gate),
        ):
            layer.forward_deferred(torch.zeros(1, 4), fmha_impl=object())

        self.assertEqual(
            [event for event in events if event[0] in ("range", "call")],
            [
                ("range", "qwen4.hc.attn"),
                ("call", "hc_attn"),
                ("range", "qwen4.gdn"),
                ("call", "attn"),
                ("range", "qwen4.hc.mlp"),
                ("call", "hc_mlp"),
                ("range", "qwen4.moe"),
                ("call", "mlp"),
            ],
        )

    def test_forward_wraps_residual_inject_in_hc_mlp_range(self):
        events = []
        layer = self._make_layer(events)

        with patch.object(
            qwen4_exp_module,
            "record_function_range",
            self._recording_range(events),
        ), patch.object(
            qwen4_exp_module,
            "inject_into_residual",
            lambda hyper_states, output, gate: (hyper_states, output, gate),
        ):
            layer.forward(torch.zeros(1, 4), fmha_impl=object())

        self.assertEqual(events.count(("range", "qwen4.hc.mlp")), 2)
        self.assertEqual(events[-2], ("range", "qwen4.hc.mlp"))

    def test_all_reduce_target_graph_is_ranged(self):
        events = []
        import rtp_llm.models_py.modules.qwen4_exp.tp_reduce as tp_reduce_module

        captured = {}

        def fake_collective_all_reduce(tensor, group, inplace=True):
            captured["group"] = group
            return tensor

        with patch.object(
            tp_reduce_module,
            "record_function_range",
            self._recording_range(events),
        ), patch.object(
            tp_reduce_module,
            "collective_torch",
            type(
                "M",
                (),
                {
                    "Group": type("G", (), {"TP": "tp"}),
                    "all_reduce": staticmethod(fake_collective_all_reduce),
                },
            ),
        ), patch.object(
            tp_reduce_module,
            "get_flashinfer_allreduce",
            return_value=None,
        ):
            tp_reduce_module.all_reduce_target_graph(torch.zeros(1), "tp")

        self.assertEqual(events[:1], [("range", "qwen4.tp_all_reduce")])
        self.assertEqual(captured.get("group"), "tp")


if __name__ == "__main__":
    unittest.main()
