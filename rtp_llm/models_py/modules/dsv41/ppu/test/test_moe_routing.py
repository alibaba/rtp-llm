"""Live EPLB routing and the two existing prepared-buffer validation stages.

Device tests execute real lookup/rotation and Graph replay, not expert GEMMs
or WORLD8 collectives. CPU protocol regressions live in the capacity suite.
"""

import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from rtp_llm.eplb.runtime import EplbLayer
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.moe import PPUMoE, _DPForwardMegaBuffer


def _layer(physical, *, device="cpu", dtype=torch.int32):
    logical = max(physical) + 1
    mapping = torch.full((logical, len(physical) - logical + 1), -1, dtype=dtype)
    counts = torch.zeros(logical, dtype=dtype)
    for slot, expert in enumerate(physical):
        mapping[expert, counts[expert]] = slot
        counts[expert] += 1
    return EplbLayer(
        0,
        mapping.to(device),
        counts.to(device),
        [torch.empty(len(physical), 1, device=device)],
        replica_offset=3,
    )


def _engine(layer=None):
    # No allocator/communication setup is needed for the production methods
    # exercised here. The mapping itself is a real validated EplbLayer.
    engine = PPUMoE.__new__(PPUMoE)
    engine._weights, engine._current_layer = {0: ()}, 0
    engine._eplb_layers = {} if layer is None else {0: layer}
    engine._eplb_routes = {}
    engine.backend, engine._fake_balance_config = "megamoe", None
    engine._mega_buffer, engine._mega_capacity = object(), 128
    engine._mega_generation, engine._dp_prepare_epoch = 1, 9
    engine._dp_active_preparation = None
    return engine


def _ids(device="cpu", dtype=torch.int64):
    return torch.tensor(
        [[0, 2, -1, 1, 3, -7, 0, 2], [3, 0, 1, -1, 2, 1, -1, -1]],
        device=device,
        dtype=dtype,
    )


class MoERoutingContract(unittest.TestCase):
    def test_absent_eplb_or_layer_preserves_input_identity(self):
        engine, ids = _engine(), _ids()
        weights = (ids >= 0).float()
        self.assertIs(engine._map_expert_ids(ids, weights, 0), ids)
        engine._eplb_layers[1] = _layer((2, 0, 3, 1))
        self.assertIs(engine._map_expert_ids(ids, weights, 0), ids)
        self.assertEqual(engine._eplb_routes, {})

    def test_single_and_multi_maps_preserve_dtype_values_padding_and_diagnostics(self):
        for physical in ((2, 0, 3, 1), (0, 1, 2, 3, 0, 2)):
            for map_dtype in (torch.int32, torch.int64):
                for dtype in (torch.int32, torch.int64):
                    layer = _layer(physical, dtype=map_dtype)
                    engine = _engine(layer)
                    for ids in (
                        _ids(dtype=dtype),
                        _ids(dtype=dtype).T,
                        _ids(dtype=dtype)[:0],
                    ):
                        weights = (ids >= 0).float()
                        expected = layer.map_ids(ids)
                        with patch.object(
                            layer, "map_ids", wraps=layer.map_ids
                        ) as general:
                            actual = engine._map_expert_ids(ids, weights, 0)
                        self.assertEqual(general.call_count, int(len(physical) != 4))
                        self.assertEqual(actual.dtype, dtype)
                        self.assertEqual(actual.shape, ids.shape)
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertIs(engine._eplb_routes[0][0], ids)
                        self.assertIs(engine._eplb_routes[0][1], actual)
                        self.assertIs(engine._eplb_routes[0][2], weights)

    def test_fast_profile_and_preparation_type_do_not_select_mapping(self):
        layer = _layer((2, 0, 3, 1))
        engine, ids = _engine(layer), _ids()
        weights, x = (ids >= 0).float(), torch.zeros(2, 16, dtype=torch.bfloat16)
        receipt = _DPForwardMegaBuffer(engine, engine._mega_buffer, 128, 2, 1, 9)
        engine._dp_active_preparation = receipt
        expected = layer.map_ids(ids)
        for flag in (None, "0", "1"):
            for prepared in (None, engine._mega_buffer, receipt):
                with patch.dict(os.environ), patch.object(
                    engine, "_forward_megamoe", return_value=x
                ) as native:
                    os.environ.pop("RTP_V41_PPU_CP8_YOCO_FAST_PREFILL", None)
                    if flag is not None:
                        os.environ["RTP_V41_PPU_CP8_YOCO_FAST_PREFILL"] = flag
                    engine.forward(
                        x, ids, weights, layer_id=0, prepared_buffer=prepared
                    )
                actual = native.call_args.args[1]
                self.assertEqual(actual.dtype, ids.dtype)
                self.assertTrue(torch.equal(actual, expected))
                self.assertIs(
                    native.call_args.kwargs["prepared_buffer"],
                    receipt.buffer if prepared is receipt else prepared,
                )

    def test_invalid_positive_ids_and_noninteger_general_fallback(self):
        for physical in ((2, 0, 3, 1), (0, 1, 2, 3, 0, 2)):
            layer = _layer(physical)
            engine = _engine(layer)
            with self.assertRaises((IndexError, RuntimeError)):
                engine._map_expert_ids(torch.tensor([[4, -1]]), None, 0)
        layer = _layer((2, 0, 3, 1))
        ids = _ids().float()
        self.assertTrue(
            torch.equal(
                _engine(layer)._map_expert_ids(ids, None, 0), layer.map_ids(ids)
            )
        )

    def test_live_mapping_updates_are_not_cached(self):
        layer = _layer((2, 0, 3, 1))
        engine, ids = _engine(layer), _ids()
        weights = (ids >= 0).float()
        first = engine._map_expert_ids(ids, weights, 0)
        updated = _layer((0, 2, 1, 3))
        layer.logical_to_physical.copy_(updated.logical_to_physical)
        second = engine._map_expert_ids(ids, weights, 0)
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(torch.equal(second, layer.map_ids(ids)))

    def test_common_buffer_consumer_preserves_allocation_and_raw_checks(self):
        engine = _engine()
        x = torch.zeros(2, 16)
        raw = engine._mega_buffer
        engine._ensure_mega_capacity = Mock(return_value=raw)
        self.assertIs(engine._ensure_mega_buffer(x), raw)
        engine._ensure_mega_capacity.assert_called_once_with(x.device, 2)
        engine._ensure_mega_capacity.reset_mock()
        for rows in (0, 2, 128):
            self.assertIs(
                engine._ensure_mega_buffer(torch.empty(rows, 16), prepared_buffer=raw),
                raw,
            )
        for value, rows in ((object(), 2), (raw, 129)):
            with self.assertRaisesRegex(ValueError, "prepared MegaMoE capacity"):
                engine._ensure_mega_buffer(torch.empty(rows, 16), prepared_buffer=value)
        engine._ensure_mega_capacity.assert_not_called()

    def test_dp_receipt_guards_and_error_stage_stay_before_routing(self):
        engine = _engine(_layer((2, 0, 3, 1)))
        ids, x = _ids(), torch.zeros(2, 16, dtype=torch.bfloat16)
        receipt = _DPForwardMegaBuffer(engine, engine._mega_buffer, 128, 2, 1, 9)
        engine._dp_active_preparation = receipt
        for field, value in (
            ("engine", object()),
            ("buffer", object()),
            ("generation", 2),
            ("epoch", 10),
            ("capacity", 256),
            ("bound", 1),
        ):
            bad = replace(receipt, **{field: value})
            engine._dp_active_preparation = bad
            with patch.object(
                engine, "_apply_fake_balance", side_effect=AssertionError("late stage")
            ) as later:
                with self.assertRaisesRegex(ValueError, "stale or insufficient DP"):
                    engine.forward(
                        x, ids, (ids >= 0).float(), layer_id=0, prepared_buffer=bad
                    )
                later.assert_not_called()
        engine._dp_active_preparation = receipt
        engine.finish_dp_prefill_forward(receipt)
        with self.assertRaisesRegex(ValueError, "stale or insufficient DP"):
            engine.forward(
                x, ids, (ids >= 0).float(), layer_id=0, prepared_buffer=receipt
            )

    def test_raw_buffer_error_stays_after_input_weight_and_padding_work(self):
        engine, ids = _engine(), _ids()
        x, weights, events = torch.zeros(2, 16), (ids >= 0).float(), []
        engine._validate_megamoe_input = Mock(
            side_effect=lambda x: events.append("input")
        )
        engine._processed_weights = Mock(side_effect=RuntimeError("weights first"))
        with self.assertRaisesRegex(RuntimeError, "weights first"):
            engine._forward_megamoe(x, ids, weights, 0, prepared_buffer=object())
        self.assertEqual(events, ["input"])
        engine._processed_weights.side_effect = (
            lambda *a: events.append("weights") or (None,) * 6
        )
        engine._get_mega_module = Mock(
            side_effect=AssertionError("native must not launch")
        )
        with self.assertRaisesRegex(ValueError, "prepared MegaMoE capacity"):
            engine._forward_megamoe(x, ids, weights, 0, prepared_buffer=object())
        self.assertEqual(events, ["input", "input", "weights"])
        engine._get_mega_module.assert_not_called()


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU test"
)
class MoERoutingDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(is_ppu())
        scope = torch.inference_mode()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def test_native_single_multi_and_empty_ids_match_public_mapper_exactly(self):
        for physical in ((2, 0, 3, 1), (0, 1, 2, 3, 0, 2)):
            for map_dtype in (torch.int32, torch.int64):
                layer = _layer(physical, device="cuda:0", dtype=map_dtype)
                engine = _engine(layer)
                for dtype in (torch.int32, torch.int64):
                    for ids in (
                        _ids("cuda:0", dtype),
                        _ids("cuda:0", dtype).T,
                        _ids("cuda:0", dtype)[:0],
                    ):
                        actual = engine._map_expert_ids(ids, (ids >= 0).float(), 0)
                        self.assertEqual(actual.dtype, dtype)
                        self.assertTrue(torch.equal(actual, layer.map_ids(ids)))

    def test_graph_replay_reads_live_maps_ids_counts_and_preserves_output_storage(self):
        for old, new in (
            ((2, 0, 3, 1), (0, 2, 1, 3)),
            ((0, 1, 2, 3, 0, 2), (1, 1, 2, 3, 0, 3)),
        ):
            layer, updated = _layer(old, device="cuda:0"), _layer(new, device="cuda:0")
            engine, ids = _engine(layer), _ids("cuda:0")
            weights = (ids >= 0).float()
            engine._map_expert_ids(ids, weights, 0)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = engine._map_expert_ids(ids, weights, 0)
            pointer = actual.data_ptr()
            for step in range(3):
                if step:
                    layer.logical_to_physical.copy_(updated.logical_to_physical)
                    layer.logical_count.copy_(updated.logical_count)
                    ids.copy_(torch.roll(ids, 1, 0))
                graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(actual.data_ptr(), pointer)
                self.assertTrue(torch.equal(actual, layer.map_ids(ids)))
                self.assertIs(engine._eplb_routes[0][1], actual)

    def test_single_replica_profile_removes_rotation_without_host_tensor_reads(self):
        layer = _layer((2, 0, 3, 1), device="cuda:0")
        engine, ids = _engine(layer), _ids("cuda:0")
        weights = (ids >= 0).float()
        engine._map_expert_ids(ids, weights, 0)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            actual = engine._map_expert_ids(ids, weights, 0)
            torch.cuda.synchronize()
        names = {item.key for item in profile.key_averages()}
        self.assertIn("aten::index_select", names)
        for absent in (
            "aten::arange",
            "aten::remainder",
            "aten::item",
            "aten::_local_scalar_dense",
        ):
            self.assertNotIn(absent, names)
        self.assertTrue(torch.equal(actual, layer.map_ids(ids)))
        directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if directory:
            profile.export_chrome_trace(
                str(Path(directory) / "single-replica-routing.trace.json")
            )


if __name__ == "__main__":
    unittest.main()
