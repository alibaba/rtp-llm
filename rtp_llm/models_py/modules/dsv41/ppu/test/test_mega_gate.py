# SPDX-License-Identifier: Apache-2.0
"""Explicit MegaGate candidate contracts and opt-in PPU qualification.

The CPU native double proves adapter behavior only. Device qualification
compares exact logical IDs and slot order with an independent FP32-logit
oracle; it does not claim EP8/model correctness or nonfinite support.
"""

import importlib.util
import json
import os
import sys
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F


_PATH = Path(__file__).resolve().parent.parent / "mega_gate.py"
_SPEC = importlib.util.spec_from_file_location("_ppu_mega_gate_candidate", _PATH)
GATE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(GATE)


def _reference(x, weight, bias, *, top_k=GATE.TOP_K, scale=1.5, image=None, image_bias=None, valid=None):
    """Original RTP FP32 projection, stable softplus and ranking-only bias."""
    logits = (
        torch.mm(x, weight.t(), out_dtype=torch.float32)
        if x.is_cuda
        else F.linear(x.float(), weight.float())
    )
    scores = F.softplus(logits).sqrt()
    ranking = scores if bias is None else scores + bias
    if image is not None:
        text_bias = torch.zeros_like(image_bias) if bias is None else bias
        ranking = scores + torch.where(image[:, None], image_bias, text_bias)
    ids = ranking.topk(top_k, dim=-1).indices
    values = scores.gather(1, ids)
    values = values / (values.sum(-1, keepdim=True) + 1e-12) * scale
    if valid is not None:
        ids = ids.masked_fill(~valid[:, None], -1)
        values = values.masked_fill(~valid[:, None], 0)
    return ids, values


def _native_double(x, weight, top_k, **kwargs):
    ids, values = _reference(
        x,
        weight,
        kwargs["bias"],
        top_k=top_k,
        scale=kwargs["routed_scaling_factor"],
        image=kwargs["image_token_mask"],
        image_bias=kwargs["image_bias"],
        valid=kwargs["mask"],
    )
    kwargs["out"][0].copy_(ids)
    kwargs["out"][1].copy_(values)


def _case(rows=3, *, device="cpu", generator=None, experts=GATE.NUM_EXPERTS):
    x = torch.randn(
        (rows, GATE.HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        (experts, GATE.HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ) * 0.001
    bias = torch.arange(experts, device=device, dtype=torch.float32) * 0.25
    return x, weight, bias


class MegaGateAdapterContracts(unittest.TestCase):
    def setUp(self):
        scope = torch.inference_mode()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        hardware = patch.object(GATE, "_hardware_supported", return_value=True)
        hardware.start()
        self.addCleanup(hardware.stop)
        self.x, self.weight, self.bias = _case()

    def candidate(self, **kwargs):
        return GATE.try_mega_gate(
            self.x,
            self.weight,
            self.bias,
            finite_domain_qualified=True,
            native=_native_double,
            **kwargs,
        )

    def test_default_declines_without_dispatch_even_for_nonfinite_input(self):
        self.x.fill_(float("nan"))
        native = Mock(side_effect=AssertionError("must not dispatch"))
        self.assertIsNone(GATE.try_mega_gate(self.x, self.weight, self.bias, native=native))
        native.assert_not_called()

    def test_native_abi_logical_ids_owned_outputs_and_unchanged_inputs(self):
        original = tuple(t.clone() for t in (self.x, self.weight, self.bias))
        native = Mock(side_effect=_native_double)
        actual = GATE.try_mega_gate(
            self.x, self.weight, self.bias,
            finite_domain_qualified=True, native=native, ep_rank=7,
        )
        expected = _reference(self.x, self.weight, self.bias)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        self.assertEqual(actual[0].dtype, torch.int64)
        self.assertEqual(actual[1].dtype, torch.float32)
        kwargs = native.call_args.kwargs
        self.assertEqual(kwargs["ep_rank"], 7)
        self.assertFalse(kwargs["use_shared_as_routed"])
        self.assertEqual(kwargs["num_shared_experts"], 0)
        self.assertEqual(kwargs["scoring_func"], "sqrtsoftplus")
        self.assertNotIn("to_physical_map", kwargs)
        self.assertNotIn("logical_count", kwargs)
        for tensor, old in zip((self.x, self.weight, self.bias), original):
            torch.testing.assert_close(tensor, old, rtol=0, atol=0)
        retained = tuple(t.clone() for t in actual)
        self.x.add_(1)
        later = self.candidate()
        for value, old, new in zip(actual, retained, later):
            torch.testing.assert_close(value, old, rtol=0, atol=0)
            self.assertNotEqual(value.data_ptr(), new.data_ptr())

    def test_visual_bias_is_alternative_ranking_bias_and_weights_are_unbiased(self):
        visual = torch.tensor([True, False, True])
        visual_bias = self.bias.flip(0).contiguous()
        actual = self.candidate(image_rows=visual, router_bias_vl=visual_bias)
        expected = _reference(
            self.x, self.weight, self.bias, image=visual, image_bias=visual_bias
        )
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)

    def test_no_bias_and_explicit_route_scale(self):
        actual = GATE.try_mega_gate(
            self.x, self.weight, None, routed_scaling_factor=2.0,
            finite_domain_qualified=True, native=_native_double,
        )
        expected = _reference(self.x, self.weight, None, scale=2.0)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)

    def test_dispatch_slots_valid_rows_and_counter(self):
        valid = torch.tensor([True, False, True])
        count = torch.tensor(91, dtype=torch.int64)
        ids, weights = self.candidate(graph_routes=(valid, count))
        expected_ids, expected_weights = _reference(
            self.x, self.weight, self.bias, valid=valid
        )
        self.assertEqual(ids.shape, (3, 8))
        torch.testing.assert_close(ids[:, :6], expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(weights[:, :6], expected_weights, rtol=0, atol=0)
        self.assertTrue(torch.equal(ids[:, 6:], torch.full((3, 2), -1)))
        self.assertTrue(torch.equal(weights[:, 6:], torch.zeros(3, 2)))
        self.assertEqual(count.item(), 2)
        valid.zero_()
        ids, weights = self.candidate(graph_routes=(valid, count))
        self.assertTrue(torch.equal(ids, torch.full((3, 8), -1)))
        self.assertTrue(torch.equal(weights, torch.zeros(3, 8)))
        self.assertEqual(count.item(), 0)

    def test_fused_pack_selection_keeps_native_six_slots_and_borrows_metadata(self):
        valid = torch.tensor([True, False, True])
        count = torch.tensor(91, dtype=torch.int64)

        def injected_pack(ids, weights, borrowed_valid, borrowed_count, output_slots):
            self.assertEqual(ids.shape, (3, 6))
            self.assertEqual(weights.shape, (3, 6))
            self.assertIs(borrowed_valid, valid)
            self.assertIs(borrowed_count, count)
            self.assertEqual(output_slots, 8)
            # This double validates only adapter dispatch. Existing PPU tests
            # exercise the actual fused kernel and its complete-chain timing.
            with patch.object(GATE, "_use_triton_pack", return_value=False):
                return GATE._pack_dispatch(ids, weights, borrowed_valid, borrowed_count)

        with patch.object(GATE, "_use_triton_pack", return_value=True), patch.object(
            GATE, "_pack_dispatch_triton", side_effect=injected_pack
        ) as fused:
            actual = self.candidate(graph_routes=(valid, count))
        fused.assert_called_once()
        expected = self.candidate(graph_routes=(valid, count))
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        self.assertEqual(count.item(), 2)
        with patch.object(GATE, "_use_triton_pack", return_value=True), patch.object(
            GATE, "_pack_dispatch_triton", side_effect=RuntimeError("pack dispatch failed")
        ) as fused:
            with self.assertRaisesRegex(RuntimeError, "pack dispatch failed"):
                self.candidate(graph_routes=(valid, count))
        fused.assert_called_once()

    def test_empty_does_not_discover_or_launch_native(self):
        count = torch.tensor(91, dtype=torch.int64)
        with patch.object(GATE, "_native_gate", side_effect=AssertionError("empty")):
            ids, weights = GATE.try_mega_gate(
                self.x[:0], self.weight, self.bias,
                graph_routes=(torch.empty(0, dtype=torch.bool), count),
                finite_domain_qualified=True,
            )
        self.assertEqual(ids.shape, (0, 8))
        self.assertEqual(weights.shape, (0, 8))
        self.assertEqual(count.item(), 0)

    def test_metadata_rejections_precede_native_and_preserve_counter(self):
        count = torch.tensor(91, dtype=torch.int64)
        common = dict(
            hidden=self.x, router_weight=self.weight, router_bias=self.bias,
            finite_domain_qualified=True,
            graph_routes=(torch.ones(3, dtype=torch.bool), count),
        )
        overrides = (
            {"finite_domain_qualified": False},
            {"hidden": self.x.float()},
            {"hidden": self.x[:, ::2]},
            {"router_weight": self.weight.float()},
            {"router_weight": self.weight[:-1]},
            {"router_bias": self.bias.bfloat16()},
            {"top_k": 3}, {"ep_rank": 8}, {"ep_rank": True},
            {"routed_scaling_factor": float("nan")},
            {"image_rows": torch.ones(3, dtype=torch.bool)},
            {"graph_routes": (torch.ones(3, dtype=torch.int32), count)},
            {"graph_routes": (torch.ones(3, dtype=torch.bool), count.float())},
        )
        for override in overrides:
            with self.subTest(override=tuple(override)):
                native = Mock(side_effect=AssertionError("unsupported"))
                self.assertIsNone(GATE.try_mega_gate(**(common | override), native=native))
                native.assert_not_called()
                self.assertEqual(count.item(), 91)
        with patch.object(GATE, "_hardware_supported", return_value=False):
            self.assertIsNone(GATE.try_mega_gate(**common, native=_native_double))

    def test_missing_api_declines_and_native_failure_never_retries(self):
        with patch.object(GATE, "_native_gate", return_value=None):
            self.assertIsNone(GATE.try_mega_gate(
                self.x, self.weight, self.bias, finite_domain_qualified=True,
            ))
        count = torch.tensor(91, dtype=torch.int64)
        native = Mock(side_effect=RuntimeError("native dispatch failed"))
        with self.assertRaisesRegex(RuntimeError, "native dispatch failed"):
            GATE.try_mega_gate(
                self.x, self.weight, self.bias, finite_domain_qualified=True,
                native=native, graph_routes=(torch.ones(3, dtype=torch.bool), count),
            )
        native.assert_called_once()
        self.assertEqual(count.item(), 91)

    def test_actual_prefill_shapes_are_metadata_only_and_no_finite_scan(self):
        with patch.object(torch, "isfinite", side_effect=AssertionError("value scan")):
            for rows in (0, 140, 1024, 2048, 4096, 8192):
                x = torch.empty((rows, 5120), dtype=torch.bfloat16, device="meta")
                weight = torch.empty((384, 5120), dtype=torch.bfloat16, device="meta")
                bias = torch.empty(384, dtype=torch.float32, device="meta")
                self.assertIsNone(GATE.eligibility_reason(
                    x, weight, bias, finite_domain_qualified=True,
                ))

    def test_autograd_and_capture_qualification_are_separate_from_native_availability(self):
        with torch.enable_grad():
            self.assertEqual(GATE.eligibility_reason(
                self.x, self.weight, self.bias, finite_domain_qualified=True,
            ), "autograd")
        with patch.object(GATE, "_hardware_supported", return_value=False):
            self.assertEqual(GATE.eligibility_reason(
                self.x, self.weight, self.bias, finite_domain_qualified=True,
            ), "hardware_or_capture")


class MegaGateProposalContracts(unittest.TestCase):
    """Exercise the real adapter and Gate caller without a device runtime."""

    def setUp(self):
        scope = torch.inference_mode()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        self.stream = NS(cuda_stream=7)
        for name, kwargs in (
            ("_hardware_supported", {"return_value": True}),
        ):
            mocked = patch.object(GATE, name, **kwargs)
            mocked.start()
            self.addCleanup(mocked.stop)
        for name, kwargs in (
            ("device", {"side_effect": lambda device: nullcontext()}),
            ("current_stream", {"side_effect": lambda: self.stream}),
            ("is_current_stream_capturing", {"return_value": False}),
        ):
            mocked = patch.object(torch.cuda, name, **kwargs)
            value = mocked.start()
            self.addCleanup(mocked.stop)
            if name == "is_current_stream_capturing":
                self.capturing = value
        self.x, self.weight, self.bias = _case(experts=128)
        self.native = Mock(side_effect=_native_double)
        self.adapter = GATE.PPUMegaGate(enabled=True, finite_domain_qualified=True)
        self.adapter._operation = self.native

    def call(self, **kwargs):
        return self.adapter.forward(
            self.x, self.weight, self.bias, phase="proposal", top_k=3, **kwargs,
        )

    def test_top3_native_width_padding_mask_and_count(self):
        actual = self.call()
        expected = _reference(self.x, self.weight, self.bias, top_k=3)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        self.assertEqual(self.native.call_args.args[2], 3)
        self.assertEqual(self.native.call_args.kwargs["out"][0].shape, (3, 3))
        valid, count = torch.tensor([True, False, True]), torch.tensor(91)
        actual = self.call(is_graph=True, graph_routes=(valid, count))
        expected = _reference(self.x, self.weight, self.bias, top_k=3, valid=valid)
        for value, reference in zip(actual, expected):
            self.assertEqual(value.shape, (3, 8))
            torch.testing.assert_close(value[:, :3], reference, rtol=0, atol=0)
        self.assertTrue(bool((actual[0][:, 3:] == -1).all()))
        self.assertTrue(bool((actual[1][:, 3:] == 0).all()))
        self.assertEqual(count.item(), 2)
        retained = tuple(value.clone() for value in actual)
        valid.zero_()
        later = self.call(is_graph=True, graph_routes=(valid, count))
        self.assertTrue(bool((later[0] == -1).all()))
        self.assertTrue(bool((later[1] == 0).all()))
        self.assertEqual(count.item(), 0)
        for value, reference in zip(actual, retained):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)

    def test_geometry_phase_and_optin_rejections_do_not_dispatch(self):
        target_x, target_weight, target_bias = _case()
        cases = (
            (self.weight, self.bias, "decode", 3),
            (self.weight, self.bias, "verify", 3),
            (self.weight, self.bias, "proposal", 6),
            (target_weight, target_bias, "proposal", 3),
            (target_weight, target_bias, "proposal", 6),
        )
        for weight, bias, phase, top_k in cases:
            with self.subTest(phase=phase, experts=weight.shape[0], top_k=top_k):
                self.assertIsNone(self.adapter.forward(
                    target_x, weight, bias, phase=phase, top_k=top_k,
                ))
        for enabled, finite in ((False, True), (True, False)):
            adapter = GATE.PPUMegaGate(enabled=enabled, finite_domain_qualified=finite)
            adapter._operation = self.native
            self.assertIsNone(adapter.forward(
                self.x, self.weight, self.bias, phase="proposal", top_k=3,
            ))
        self.native.assert_not_called()

    def test_capture_requires_matching_stream_rows_and_optional_operands(self):
        valid, count = torch.ones(3, dtype=torch.bool), torch.tensor(91)
        self.capturing.return_value = True
        self.assertIsNone(self.call(is_graph=True, graph_routes=(valid, count)))
        self.native.assert_not_called()
        self.assertEqual(count.item(), 91)
        self.capturing.return_value = False
        self.call(is_graph=True, graph_routes=(valid, count))
        self.capturing.return_value = True
        self.assertIsNotNone(self.call(is_graph=True, graph_routes=(valid, count)))
        calls = self.native.call_count
        count.fill_(91)
        self.stream = NS(cuda_stream=8)
        self.assertIsNone(self.call(is_graph=True, graph_routes=(valid, count)))
        self.stream = NS(cuda_stream=7)
        self.assertIsNone(self.call(is_graph=True))
        self.assertIsNone(self.adapter.forward(
            self.x[:2], self.weight, self.bias, phase="proposal", top_k=3,
            is_graph=True, graph_routes=(valid[:2], count),
        ))
        self.assertEqual(self.native.call_count, calls)
        self.assertEqual(count.item(), 91)

    def test_top_k_packing_requires_its_own_graph_warmup(self):
        valid = torch.tensor([True, False, True])
        count = torch.tensor(91)
        routes = (valid, count)
        self.assertIsNone(self.call(pack_graph_top_k=True, graph_routes=routes))
        self.assertIsNone(self.call(is_graph=True, pack_graph_top_k=True))
        self.assertIsNone(self.call(
            is_graph=True, graph_routes=routes,
            pack_graph_top_k=True, compact_graph_routes=True,
        ))
        self.native.assert_not_called()
        self.assertEqual(count.item(), 91)

        padded = self.call(is_graph=True, graph_routes=routes)
        count.fill_(91)
        self.capturing.return_value = True
        self.assertIsNone(
            self.call(is_graph=True, graph_routes=routes, pack_graph_top_k=True)
        )
        self.assertEqual(self.native.call_count, 1)
        self.assertEqual(count.item(), 91)
        self.capturing.return_value = False
        self.call(is_graph=True, graph_routes=routes, pack_graph_top_k=True)
        self.capturing.return_value = True
        compact = self.call(is_graph=True, graph_routes=routes, pack_graph_top_k=True)
        for actual, expected in zip(compact, padded):
            self.assertTrue(actual.is_contiguous())
            torch.testing.assert_close(actual, expected[:, :3], rtol=0, atol=0)
        self.assertEqual(count.item(), 2)

    def test_top_k_packing_empty_graph_writes_zero_without_native_launch(self):
        count = torch.tensor(91)
        ids, weights = self.adapter.forward(
            self.x[:0], self.weight, self.bias, phase="proposal", top_k=3,
            is_graph=True, pack_graph_top_k=True,
            graph_routes=(torch.empty(0, dtype=torch.bool), count),
        )
        self.native.assert_not_called()
        self.assertEqual(count.item(), 0)
        self.assertEqual(ids.shape, (0, 3))
        self.assertEqual(weights.shape, (0, 3))
        self.assertEqual(ids.dtype, torch.int64)
        self.assertEqual(weights.dtype, torch.float32)

    def test_top_k_packing_selector_requires_unmapped_real_megamoe(self):
        from rtp_llm.models_py.modules.dsv41.ppu.decode_fusions import (
            pack_graph_top_k_enabled,
        )

        engine = dict(
            backend="megamoe", precision="w4a4", top_k=6,
            num_routed_experts=384, num_physical_experts=384,
            _eplb_layers={}, _fake_balance_config=(False, 6, 0, 32),
        )
        flag = "RTP_V41_PPU_MEGA_GATE_PACK_TOP_K"
        with patch.dict(os.environ, {flag: "0"}):
            self.assertFalse(pack_graph_top_k_enabled(NS(**engine), 6))
        with patch.dict(os.environ, {flag: "1"}):
            self.assertTrue(pack_graph_top_k_enabled(NS(**engine), 6))
            self.assertFalse(pack_graph_top_k_enabled(NS(**engine), 3))
            for override in (
                {"backend": "grouped"},
                {"precision": "w4a8"},
                {"num_physical_experts": 416},
                {"_eplb_layers": {0: object()}},
                {"_fake_balance_config": (True, 6, 0, 32)},
            ):
                with self.subTest(override=override):
                    self.assertFalse(pack_graph_top_k_enabled(NS(**(engine | override)), 6))

    def test_native_failure_propagates_without_router_retry(self):
        self.native.side_effect = RuntimeError("proposal native failed")
        count = torch.tensor(91)
        with self.assertRaisesRegex(RuntimeError, "proposal native failed"):
            self.call(graph_routes=(torch.ones(3, dtype=torch.bool), count))
        self.native.assert_called_once()
        self.assertEqual(count.item(), 91)

    def test_real_shared_gate_passes_proposal_phase_and_masks(self):
        from rtp_llm.models_py.modules.dsv41.ppu.layers import PPUSharedGate

        gate = PPUSharedGate.__new__(PPUSharedGate)
        torch.nn.Module.__init__(gate)
        gate.weight, gate.bias, gate.bias_vl = self.weight, self.bias, None
        gate.top_k, gate.route_scale = 3, 1.5
        gate._mega_gate = self.adapter
        valid, count = torch.tensor([True, False, True]), torch.tensor(91)
        for graph, compact in ((False, False), (True, False), (True, True)):
            state = NS(proposal=True, is_graph=graph, prefill=False, idle=False,
                       active_token_mask=valid, metadata=NS(valid_mask=valid))
            gate.runtime = NS(current=state)
            kwargs = dict(graph_routes=(valid, count), allow_top3_graph_routes=True) if graph else {}
            if compact:
                kwargs["pack_graph_top_k"] = True
            actual = gate(self.x, **kwargs)
            expected = _reference(self.x, self.weight, self.bias, top_k=3, valid=valid)
            for value, reference in zip(actual, expected):
                torch.testing.assert_close(value[:, :3], reference, rtol=0, atol=0)
            if graph:
                self.assertEqual(actual[0].shape, (3, 3 if compact else 8))
                self.assertEqual(count.item(), 2)
        self.assertEqual(self.native.call_count, 3)

        self.adapter.enabled = False
        count.fill_(-1)
        ids, weights = gate(self.x, **kwargs)
        self.assertEqual(ids.shape, (3, 8))
        self.assertEqual(weights.shape, (3, 8))
        self.assertEqual(count.item(), 2)
        self.assertEqual(self.native.call_count, 3)


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU test"
)
class MegaGatePPUQualification(unittest.TestCase):
    def setUp(self):
        from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu

        self.assertTrue(is_ppu())
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))
        native = GATE._native_gate()
        self.assertIsNotNone(native, "PPU bf16_mega_gate API is required")
        scope = torch.inference_mode()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        self.generator = torch.Generator(device="cuda").manual_seed(20261116)
        provider_record = dict(
            case="native_provider", native_module=getattr(native, "__module__", None)
        )
        provider = sys.modules.get("rtp_ppu_mega_gate_native")
        if provider is not None:
            provider_record["provider_file"] = getattr(provider, "__file__", None)
            commit = getattr(provider, "upstream_commit", None)
            if commit is not None:
                provider_record["upstream_commit"] = commit
            runtime_paths = getattr(provider, "runtime_paths", None)
            if callable(runtime_paths):
                provider_record["runtime_paths"] = runtime_paths()
        self.records = [provider_record]
        self.addCleanup(self._write_records)

    def _write_records(self):
        directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if directory:
            method = self.id().rsplit(".", 1)[-1]
            Path(directory, f"mega-gate-qualification.{method}.json").write_text(
                json.dumps(self.records, indent=2) + "\n"
            )

    def _assert_routes(self, actual, expected, **metadata):
        ids, weights = actual
        expected_ids, expected_weights = expected
        sorted_ids, order = ids.sort(-1)
        sorted_expected, expected_order = expected_ids.sort(-1)
        raw_equal = torch.equal(ids, expected_ids)
        members_equal = torch.equal(sorted_ids, sorted_expected)
        error = (weights.gather(1, order) - expected_weights.gather(1, expected_order)).abs()
        self.records.append(dict(
            metadata, raw_slot_ids_equal=raw_equal, members_equal=members_equal,
            maximum_weight_error=float(error.max()) if error.numel() else 0.0,
        ))
        torch.testing.assert_close(sorted_ids, sorted_expected, rtol=0, atol=0)
        # Existing public dsv41/test/test_mega_gate.py uses this weight budget;
        # discrete membership and raw slot order remain exact requirements.
        torch.testing.assert_close(
            weights.gather(1, order), expected_weights.gather(1, expected_order),
            rtol=1e-4, atol=1e-5,
        )
        self.assertTrue(raw_equal, "native slot order differs from the RTP top-k order")

    def test_random_prefill_shapes_visual_masks_and_empty_rows(self):
        for rows in (140, 1024, 2048, 4096, 8192):
            x, weight, bias = _case(rows, device="cuda", generator=self.generator)
            valid = torch.arange(rows, device="cuda") % 3 != 0
            image = torch.arange(rows, device="cuda") % 2 == 0
            visual_bias = bias.flip(0).contiguous()
            count = torch.full((), -1, device="cuda", dtype=torch.int64)
            for visual in (False, True):
                actual = GATE.try_mega_gate(
                    x, weight, bias, finite_domain_qualified=True,
                    image_rows=image if visual else None,
                    router_bias_vl=visual_bias if visual else None,
                    graph_routes=(valid, count), ep_rank=7,
                )
                self.assertIsNotNone(actual)
                expected = _reference(
                    x, weight, bias, valid=valid,
                    image=image if visual else None,
                    image_bias=visual_bias if visual else None,
                )
                self._assert_routes(
                    (actual[0][:, :6], actual[1][:, :6]), expected,
                    rows=rows, visual=visual, case="separated_bias",
                )
                self.assertEqual(int(count), int(valid.sum()))
                self.assertTrue(bool((actual[0][:, 6:] == -1).all()))
                self.assertTrue(bool((actual[1][:, 6:] == 0).all()))
            empty = GATE.try_mega_gate(
                x[:0], weight, bias, finite_domain_qualified=True,
                graph_routes=(valid[:0], count),
            )
            self.assertEqual(empty[0].shape, (0, 8))
            self.assertEqual(int(count), 0)

    def test_near_cutoff_keeps_fp32_projection_information(self):
        x = torch.zeros((140, 5120), device="cuda", dtype=torch.bfloat16)
        x[:, 0], x[:, 1] = 1, 2 ** -7
        weight = torch.zeros((384, 5120), device="cuda", dtype=torch.bfloat16)
        weight[:, 0], weight[:, 1] = 1, -1
        weight[5, 1], weight[6, 1] = 2 ** -8, -(2 ** -8)
        bias = torch.zeros(384, device="cuda", dtype=torch.float32)
        bias[:5] = torch.arange(5, device="cuda") + 2
        actual = GATE.try_mega_gate(x, weight, bias, finite_domain_qualified=True)
        self.assertIsNotNone(actual)
        expected = _reference(x, weight, bias)
        self.assertTrue(bool((expected[0] == 5).any(-1).all()))
        self.assertFalse(bool((expected[0] == 6).any()))
        self._assert_routes(actual, expected, rows=140, case="fp32_cutoff")

    def test_proposal_top3_changed_input_graph_replay_and_padding(self):
        """Real native E128/top3 and fused 3-to-8 packing, including idle rows."""
        for rows in (1, 3, 24, 96):
            with self.subTest(rows=rows):
                x = torch.zeros((rows, 5120), device="cuda", dtype=torch.bfloat16)
                x[:, 0] = 4
                weight = torch.zeros((128, 5120), device="cuda", dtype=torch.bfloat16)
                weight[:, 0] = torch.arange(128, device="cuda").to(torch.bfloat16) / 128
                bias = torch.arange(128, device="cuda", dtype=torch.float32) * 0.001
                valid = torch.arange(rows, device="cuda") % 3 != 1
                count = torch.full((), -1, device="cuda", dtype=torch.int64)
                adapter = GATE.PPUMegaGate(enabled=True, finite_domain_qualified=True)

                def run():
                    result = adapter.forward(
                        x, weight, bias, phase="proposal", top_k=3, is_graph=True,
                        graph_routes=(valid, count),
                    )
                    self.assertIsNotNone(result, "Proposal MegaGate declined")
                    return result

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = run()
                graph.replay()
                torch.cuda.synchronize()
                first_ids = actual[0].clone()
                for changed, idle in ((False, False), (True, False), (True, True)):
                    if changed:
                        x[:, 0] = -4
                    if idle:
                        valid.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = _reference(x, weight, bias, top_k=3, valid=valid)
                    self._assert_routes(
                        (actual[0][:, :3], actual[1][:, :3]), expected,
                        rows=rows, case="proposal_graph", changed=changed, idle=idle,
                    )
                    self.assertEqual(actual[0].shape, (rows, 8))
                    self.assertEqual(actual[1].shape, (rows, 8))
                    self.assertTrue(bool((actual[0][:, 3:] == -1).all()))
                    self.assertTrue(bool((actual[1][:, 3:] == 0).all()))
                    self.assertEqual(int(count), int(valid.sum()))
                    if changed and not idle:
                        self.assertFalse(torch.equal(actual[0][valid], first_ids[valid]))

    def test_top_k_packing_matches_padded_routes_on_graph_replay(self):
        stream = torch.cuda.Stream()
        for phase, q_len, top_k, experts in (
            ("proposal", 3, 3, 128),
            ("verify", 4, 6, 384),
        ):
            for batch in (4, 8, 16, 32):
                with self.subTest(phase=phase, batch=batch):
                    rows = batch * q_len
                    x, weight, bias = _case(
                        rows, device="cuda", generator=self.generator, experts=experts
                    )
                    pattern = torch.arange(rows, device="cuda") % q_len
                    valid = pattern != 0
                    count = torch.full((), -1, dtype=torch.int64, device="cuda")
                    padded_count = torch.empty_like(count)
                    adapter = GATE.PPUMegaGate(enabled=True, finite_domain_qualified=True)

                    def run(pack_top_k, route_count):
                        return adapter.forward(
                            x, weight, bias, phase=phase, top_k=top_k, is_graph=True,
                            graph_routes=(valid, route_count), pack_graph_top_k=pack_top_k,
                        )

                    caller = torch.cuda.current_stream()
                    stream.wait_stream(caller)
                    with torch.cuda.stream(stream):
                        self.assertIsNotNone(run(True, count))
                    caller.wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        compact = run(True, count)
                    self.assertIsNotNone(compact)
                    ids, weights = compact
                    self.assertEqual(ids.shape, (rows, top_k))
                    self.assertEqual(weights.shape, ids.shape)
                    self.assertEqual(ids.dtype, torch.int64)
                    self.assertEqual(weights.dtype, torch.float32)
                    self.assertTrue(ids.is_contiguous())
                    self.assertTrue(weights.is_contiguous())

                    excluded_slots = (0, q_len - 1, None, -1, 1)
                    for excluded in excluded_slots:
                        if excluded is None:
                            valid.zero_()
                        else:
                            valid.copy_(pattern != excluded)
                        x.neg_()
                        count.fill_(-1)
                        graph.replay()
                        padded_ids, padded_weights = run(False, padded_count)
                        self.assertTrue(torch.equal(ids, padded_ids[:, :top_k]))
                        self.assertTrue(
                            torch.equal(
                                weights.view(torch.int32),
                                padded_weights[:, :top_k].view(torch.int32),
                            )
                        )
                        self.assertTrue(bool((ids[~valid] == -1).all()))
                        self.assertTrue(bool((weights[~valid] == 0).all()))
                        self.assertEqual(int(count), int(valid.sum()))
                        self.assertEqual(int(count), int(padded_count))

                    self.records.append(dict(
                        case="top_k_graph_pack", phase=phase, batch=batch,
                        rows=rows, top_k=top_k, exact_replays=len(excluded_slots),
                    ))

    def test_existing_ptpc_dense_bytes_survive_native_mega_gate_call(self):
        """Prove runtime coexistence; setUp has already resolved MegaGate.

        This does not prove the reverse import order or full-model behavior.
        The actual resident FP8 dense consumer runs before and after MegaGate;
        neither native operation is replaced with a double.
        """
        import deep_gemm

        from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear

        old_module = deep_gemm
        old_file = deep_gemm.__file__
        old_gemm = deep_gemm.fp8_gemm_nt
        weight = torch.randn(
            (128, 512), generator=torch.Generator().manual_seed(20261117)
        ).to(torch.float8_e4m3fn).to("cuda")
        scale = torch.linspace(
            0.002, 0.012, 128, device="cuda", dtype=torch.float32
        )[:, None]
        dense = PPUPTPCLinear(weight, scale)
        self.assertIs(dense._gemm, old_gemm)
        dense_input = torch.randn(
            (140, 512), device="cuda", dtype=torch.bfloat16,
            generator=self.generator,
        )
        before = dense(dense_input)
        torch.cuda.synchronize()
        retained_bytes = before.contiguous().view(torch.uint8).clone()
        x, gate_weight, bias = _case(140, device="cuda", generator=self.generator)
        candidate = GATE.try_mega_gate(
            x, gate_weight, bias, finite_domain_qualified=True,
        )
        self.assertIsNotNone(candidate)
        torch.cuda.synchronize()
        after = dense(dense_input)
        torch.cuda.synchronize()
        same_module = sys.modules.get("deep_gemm") is old_module
        same_file = deep_gemm.__file__ == old_file
        exact_bytes = torch.equal(after.contiguous().view(torch.uint8), retained_bytes)
        self.records.append(dict(
            case="existing_ptpc_dense_runtime_coexistence", dense_shape=[140, 512, 128],
            deep_gemm_file_before=old_file, deep_gemm_file_after=deep_gemm.__file__,
            same_module=same_module, same_file=same_file, exact_bytes=exact_bytes,
            dense_native_module=getattr(old_gemm, "__module__", None),
            reverse_import_order_covered=False,
        ))
        self.assertTrue(same_module)
        self.assertTrue(same_file)
        self.assertIs(deep_gemm.fp8_gemm_nt, old_gemm)
        self.assertEqual(before.dtype, after.dtype)
        self.assertEqual(before.shape, after.shape)
        self.assertTrue(exact_bytes, "existing PTPC dense output changed after MegaGate")

    @unittest.skipUnless(
        os.environ.get("RTP_V41_TEST_MEGAGATE_PERF") == "1", "explicit component timing"
    )
    def test_eager_whole_router_timing(self):
        """Time actual baseline/candidate, including padding and route counter.

        L0 supplies the immutable source/weight manifest and device lease. An
        optional fixture contains BF16 ``hidden``/``weight``, FP32 ``bias``
        and optional boolean ``valid``/``image_rows`` plus FP32 ``bias_vl``.
        Host wall time includes submission and synchronization; device events
        cover the submitted local chain. No profiler or CUDA graph is used.
        """
        from rtp_llm.models_py.modules.dsv41.ppu.moe import sqrtsoftplus_gate

        fixture_path = os.environ.get("RTP_V41_TEST_MEGAGATE_FIXTURE")
        fixture = (
            torch.load(fixture_path, map_location="cuda", weights_only=True)
            if fixture_path else None
        )
        shapes = (fixture["hidden"].shape[0],) if fixture else (140, 1024, 2048, 4096, 8192)
        for rows in shapes:
            if fixture:
                x, weight, bias = fixture["hidden"], fixture["weight"], fixture["bias"]
                image, visual_bias = fixture.get("image_rows"), fixture.get("bias_vl")
                valid = fixture.get("valid", torch.ones(rows, device="cuda", dtype=torch.bool))
            else:
                x, weight, bias = _case(rows, device="cuda", generator=self.generator)
                image = visual_bias = None
                valid = torch.arange(rows, device="cuda") % 3 != 0
            count = torch.zeros((), device="cuda", dtype=torch.int64)
            kwargs = dict(
                image_rows=image, router_bias_vl=visual_bias,
                graph_routes=(valid, count),
            )

            def baseline():
                return sqrtsoftplus_gate(x, weight, bias, **kwargs)

            def candidate():
                result = GATE.try_mega_gate(
                    x, weight, bias, **kwargs, finite_domain_qualified=True,
                )
                if result is None:
                    self.fail("timing candidate declined; no fallback timings are valid")
                return result

            self._assert_routes(
                candidate(), baseline(), rows=rows, case="timing_actual_baseline",
                fixture=fixture_path,
            )
            for _ in range(5):
                baseline()
                candidate()
            for block in range(3):
                operations = (("baseline", baseline), ("candidate", candidate))
                if block % 2:
                    operations = operations[::-1]
                for name, operation in operations:
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    stop = torch.cuda.Event(enable_timing=True)
                    wall_start = time.perf_counter()
                    start.record()
                    for _ in range(20):
                        operation()
                    stop.record()
                    stop.synchronize()
                    self.records.append(dict(
                        case="eager_whole_router_timing", arm=name, rows=rows,
                        block=block, iterations=20, fixture=fixture_path,
                        gpu_ms_per_call=start.elapsed_time(stop) / 20,
                        wall_ms_per_call=(time.perf_counter() - wall_start) * 1000 / 20,
                        graph=False, profiler=False,
                    ))


if __name__ == "__main__":
    unittest.main()
