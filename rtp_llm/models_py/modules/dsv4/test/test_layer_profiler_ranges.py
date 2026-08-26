import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import rtp_llm.models_py.model_desc.deepseek_v4_dspark_model as dspark_module
from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import (
    DeepSeekV4DSparkModel,
)
from rtp_llm.models_py.modules.dsv4 import _profiler
from rtp_llm.models_py.modules.dsv4.decode import forward as decode_forward
from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


def _logging_layer_range(events):
    @contextmanager
    def layer_range(layer_idx):
        events.append(("enter", layer_idx))
        try:
            yield
        finally:
            events.append(("exit", layer_idx))

    return layer_range


class _DecodeLayer:
    def __init__(self, layer_idx, events):
        self.layer_id = layer_idx
        self._events = events

    def forward_decode(self, hidden, *_args, **_kwargs):
        self._events.append(("layer", self.layer_id))
        return hidden + self.layer_id + 1


class _PrefillLayer:
    def __init__(self, layer_idx, events):
        self.layer_id = layer_idx
        self._events = events

    def __call__(self, hidden, *_args, **_kwargs):
        self._events.append(("layer", self.layer_id))
        return hidden + self.layer_id + 1


class LayerProfilerRangeTest(unittest.TestCase):
    def test_factory_is_noop_without_active_profiler(self):
        with patch.object(_profiler, "_RANGES_ENABLED", True), patch.object(
            torch.autograd, "_profiler_enabled", return_value=False
        ), patch.object(
            torch._C._profiler, "_RecordFunctionFast"
        ) as record_function:
            layer_range = _profiler.make_layer_forward_range()
            with layer_range(3):
                pass

        record_function.assert_not_called()

    def test_factory_captures_active_state_before_nested_suppression(self):
        names = []

        @contextmanager
        def record_function(name):
            names.append(name)
            yield

        with patch.object(_profiler, "_RANGES_ENABLED", True), patch.object(
            torch.autograd, "_profiler_enabled", return_value=True
        ), patch.object(
            torch._C._profiler,
            "_RecordFunctionFast",
            record_function,
        ):
            layer_range = _profiler.make_layer_forward_range()
            with self.assertRaisesRegex(RuntimeError, "layer failure"):
                with _profiler.disable_record_function_ranges():
                    with layer_range(7):
                        self.assertFalse(
                            _profiler.record_function_ranges_enabled()
                        )
                        raise RuntimeError("layer failure")

        self.assertEqual(names, ["forward(layer=7)"])
        self.assertTrue(_profiler.record_function_ranges_enabled())

    def test_factory_respects_outer_disable(self):
        with patch.object(_profiler, "_RANGES_ENABLED", True), patch.object(
            torch.autograd, "_profiler_enabled", return_value=True
        ), patch.object(
            torch._C._profiler, "_RecordFunctionFast"
        ) as record_function:
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
            event for event in prof.events() if event.name == "forward(layer=5)"
        ]
        self.assertEqual(len(layer_events), 1)
        self.assertEqual(
            layer_events[0].scope,
            torch.profiler.RecordScope.FUNCTION.value,
        )
        self.assertFalse(layer_events[0].is_user_annotation)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_layer_range_has_no_gpu_user_annotation_projection(self):
        x = torch.ones(32, device="cuda")
        torch.cuda.synchronize()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "layer_range.json"
            with patch.object(
                _profiler, "_RANGES_ENABLED", True
            ), torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                layer_range = _profiler.make_layer_forward_range()
                with layer_range(9):
                    torch.add(x, 1)
                    torch.cuda.synchronize()
            prof.export_chrome_trace(str(trace_path))

            trace = json.loads(trace_path.read_text())
            events = trace if isinstance(trace, list) else trace["traceEvents"]
            layer_events = [
                event
                for event in events
                if event.get("name") == "forward(layer=9)"
            ]

        self.assertEqual(
            [event.get("cat") for event in layer_events],
            ["cpu_op"],
        )
        self.assertTrue(any(event.get("cat") == "kernel" for event in events))
        self.assertFalse(
            any(
                event.get("cat") == "gpu_user_annotation"
                and event.get("name") == "forward(layer=9)"
                for event in events
            )
        )

    def test_production_decode_wraps_each_layer(self):
        events = []
        meta = SimpleNamespace(
            batch_size=1,
            q_len_per_req=2,
            is_cuda_graph=False,
        )
        v4 = SimpleNamespace(
            embed=lambda ids: torch.zeros((ids.numel(), 2)),
            hc_mult=1,
            layers=[_DecodeLayer(0, events), _DecodeLayer(1, events)],
            capture_aux_hidden_layer_ids=(),
            _mtp_hidden_buffer=None,
            _hc_head_reduce=lambda hidden: hidden.squeeze(2),
            norm=lambda hidden: hidden,
        )

        with patch.object(decode_forward._rt, "ENABLED", False), patch.object(
            _profiler,
            "make_layer_forward_range",
            return_value=_logging_layer_range(events),
        ):
            decode_forward.forward_layers(
                v4,
                kv_cache=None,
                input_ids=torch.tensor([1, 2]),
                attn_metadata=meta,
            )

        self.assertEqual(
            events,
            [
                ("enter", 0),
                ("layer", 0),
                ("exit", 0),
                ("enter", 1),
                ("layer", 1),
                ("exit", 1),
            ],
        )

    def test_standalone_decode_and_prefill_wrap_each_layer(self):
        decode_events = []
        decode_model = SimpleNamespace(
            embed=lambda ids: torch.zeros((*ids.shape, 2)),
            hc_mult=1,
            layers=[
                _DecodeLayer(0, decode_events),
                _DecodeLayer(1, decode_events),
            ],
            _hc_head_reduce=lambda hidden: hidden.squeeze(2),
            norm=lambda hidden: hidden,
            args=SimpleNamespace(dim=2),
        )
        meta = SimpleNamespace(batch_size=1, q_len_per_req=2)

        with patch.object(
            _profiler,
            "make_layer_forward_range",
            return_value=_logging_layer_range(decode_events),
        ):
            V4Transformer.forward_decode(
                decode_model,
                torch.tensor([1, 2]),
                meta,
            )

        prefill_events = []
        prefill_model = SimpleNamespace(
            args=SimpleNamespace(ep_size=1, dim=2),
            embed=lambda ids: torch.zeros((*ids.shape, 2)),
            hc_mult=1,
            layers=[
                _PrefillLayer(0, prefill_events),
                _PrefillLayer(1, prefill_events),
            ],
            _propagate_cp_ctx=lambda _ctx: None,
            _hc_head_reduce=lambda hidden: hidden.squeeze(2),
            norm=lambda hidden: hidden,
        )
        with patch.object(V4Transformer, "_propagate_cp_ctx", lambda _self, _ctx: None), patch.object(
            _profiler,
            "make_layer_forward_range",
            return_value=_logging_layer_range(prefill_events),
        ), patch.object(
            decode_forward._rt, "ENABLED", False
        ):
            V4Transformer.forward(
                prefill_model,
                torch.tensor([[1, 2]]),
                apply_lm_head=False,
            )

        expected = [
            ("enter", 0),
            ("layer", 0),
            ("exit", 0),
            ("enter", 1),
            ("layer", 1),
            ("exit", 1),
        ]
        self.assertEqual(decode_events, expected)
        self.assertEqual(prefill_events, expected)

    def test_dspark_proposal_and_commit_wrap_each_layer(self):
        proposal_events = []
        model = DeepSeekV4DSparkModel.__new__(DeepSeekV4DSparkModel)
        model.v4 = SimpleNamespace(
            embed=lambda ids: torch.zeros((*ids.shape, 2)),
            hc_mult=1,
            layers=[
                _DecodeLayer(0, proposal_events),
                _DecodeLayer(1, proposal_events),
            ],
        )
        with patch.object(
            _profiler,
            "make_layer_forward_range",
            return_value=_logging_layer_range(proposal_events),
        ):
            model._forward_layers(
                query_ids=torch.tensor([[1, 2]]),
                query_positions=torch.tensor([[3, 4]]),
                prefix_lengths=torch.tensor([3]),
                active_requests=torch.tensor([True]),
                block_table=torch.tensor([[1]]),
                tokens_per_block=16,
                graph_metadata=SimpleNamespace(),
            )

        expected = [
            ("enter", 0),
            ("layer", 0),
            ("exit", 0),
            ("enter", 1),
            ("layer", 1),
            ("exit", 1),
        ]
        self.assertEqual(proposal_events, expected)

        commit_events = []
        model.v4 = SimpleNamespace(layers=[object(), object()])
        model.kv_cache = SimpleNamespace(
            get_layer_caches=lambda idx: commit_events.append(("cache", idx)) or idx
        )
        model._swa_block_table = lambda *_args: torch.tensor([[1]])
        model._commit_layer_features = (
            lambda layer_idx, *_args, **_kwargs: commit_events.append(
                ("layer", layer_idx)
            )
        )
        inputs = SimpleNamespace(attention_inputs=SimpleNamespace())

        with patch.object(
            _profiler,
            "make_layer_forward_range",
            return_value=_logging_layer_range(commit_events),
        ), patch.object(
            dspark_module, "require_pool_tokens_per_block", return_value=16
        ), patch.object(
            dspark_module,
            "create_write_cache_store_impl",
            return_value=lambda cache: commit_events.append(("store", cache)),
        ):
            model.commit_feature_rows(
                main_x=torch.zeros((1, 2)),
                context_req_ids=torch.tensor([0]),
                context_positions=torch.tensor([0]),
                committed_ends=torch.tensor([1]),
                inputs=inputs,
            )

        self.assertEqual(
            commit_events,
            [
                ("enter", 0),
                ("layer", 0),
                ("cache", 0),
                ("store", 0),
                ("exit", 0),
                ("enter", 1),
                ("layer", 1),
                ("cache", 1),
                ("store", 1),
                ("exit", 1),
            ],
        )


if __name__ == "__main__":
    unittest.main()
