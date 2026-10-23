"""Exactness, ownership and fallback gates for routed SiLU lookup fusion."""

import gc
import os
import unittest
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.dsv4._v41_routed_lut_triton import (
    prepare_silu_table as prepare_table,
)
from rtp_llm.models_py.modules.dsv4._v41_routed_lut_triton import quantize_swiglu_lut
from rtp_llm.models_py.modules.dsv4.v41_expert import quantize_swiglu
from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts


def exact(actual, expected):
    if actual is None:
        raise AssertionError("The routed LUT unexpectedly fell back")
    for left, right in zip(actual, expected):
        torch.testing.assert_close(
            left.view(torch.uint8), right.view(torch.uint8), rtol=0, atol=0
        )


def reference(gate, up, valid, weights, clamp):
    return quantize_swiglu(
        gate,
        up,
        valid,
        route_weights=weights,
        clamp_limit=clamp,
        allow_contiguous=True,
    )


def invalid_child(_rank, case):
    gate = torch.ones(3, 32, device="cuda", dtype=torch.bfloat16)
    up = torch.ones_like(gate)
    valid = torch.ones(3, device="cuda", dtype=torch.bool)
    weights = torch.ones(3, device="cuda")
    if case == "gate":
        gate[0, 0] = float("nan")
    elif case == "up":
        up[0, 0] = float("nan")
    elif case == "weight":
        weights[0] = float("nan")
    else:
        raise AssertionError(case)
    quantize_swiglu_lut(gate, up, valid, weights, prepare_table("cuda"))
    torch.cuda.synchronize()
    raise AssertionError("Nonfinite activation was accepted")


class RoutedLutTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(41729)

    def compare(self, gate, up, valid, weights, clamp, table=None, expected=None):
        if table is None:
            table = prepare_table(gate.device, clamp)
        if expected is None:
            expected = reference(gate, up, valid, weights, clamp)
        exact(
            quantize_swiglu_lut(gate, up, valid, weights, table, clamp_limit=clamp),
            expected,
        )

    def test_every_finite_bf16_gate_pattern(self):
        bits = torch.arange(65536, device="cuda", dtype=torch.int32)
        finite = bits[(bits & 0x7F80) != 0x7F80].to(torch.uint16)
        self.assertEqual(finite.numel(), 65280)
        gate = finite.view(torch.bfloat16).reshape(-1, 32)
        valid = torch.ones(gate.shape[0], device="cuda", dtype=torch.bool)
        weights = torch.full((gate.shape[0],), 0.25, device="cuda")
        for clamp in (0.0, 10.0, 10.1):
            for sign in (-1, 1):
                with self.subTest(clamp=clamp, sign=sign):
                    self.compare(
                        gate, torch.full_like(gate, sign * 0.5), valid, weights, clamp
                    )

    def test_strides_padding_inactive_nan_and_zero_routes(self):
        for shape in ((1, 32), (3, 96), (129, 2304), (1021, 2304), (3, 33, 96)):
            storage = (
                torch.randn(*shape[:-1], shape[-1] * 2 + 32, device="cuda") * 8
            ).bfloat16()
            gate = storage[..., : shape[-1]]
            up = storage[..., shape[-1] : 2 * shape[-1]]
            valid = (
                torch.arange(gate.numel() // shape[-1], device="cuda").reshape(
                    shape[:-1]
                )
                % 7
                != 0
            )
            weights = torch.rand(shape[:-1], device="cuda") * 1.5
            weights.reshape(-1)[::11] = 0
            weights.reshape(-1)[1::11] = -0.0
            storage[~valid | (weights == 0)] = float("nan")
            weights[~valid] = float("nan")
            for clamp in (0.0, 10.0, 10.1):
                with self.subTest(shape=shape, clamp=clamp):
                    self.compare(gate, up, valid, weights, clamp)

    def test_rounding_scale_transitions_and_signed_zero(self):
        gates = torch.tensor(
            [-100, -90, -20, -10, -1, -0.0, 0.0, 1, 10, 20],
            device="cuda",
            dtype=torch.bfloat16,
        )
        gate = gates[torch.arange(129 * 96, device="cuda") % gates.numel()].reshape(
            129, 96
        )
        valid = torch.ones(129, device="cuda", dtype=torch.bool)
        weights = torch.tensor([0, 2**-120, 0.5, 1, 1.5], device="cuda")[
            torch.arange(129, device="cuda") % 5
        ]
        for clamp in (0.0, 10.0):
            table = prepare_table("cuda", clamp)
            for exponent in (-120, -30, -20, -8, 0, 8):
                for multiplier in (0.99609375, 1.0, 1.0078125):
                    with self.subTest(
                        clamp=clamp, exponent=exponent, multiplier=multiplier
                    ):
                        up = torch.full_like(gate, multiplier * 2**exponent)
                        self.compare(gate, up, valid, weights, clamp, table)

    def test_table_handoff_retirement_and_graph_replay(self):
        gate = torch.randn(129, 96, device="cuda").bfloat16()
        up = torch.randn_like(gate)
        valid = torch.ones(129, device="cuda", dtype=torch.bool)
        weights = torch.rand(129, device="cuda")
        expected = reference(gate, up, valid, weights, 10.0)
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        torch.cuda.synchronize()
        for _ in range(3):
            with torch.cuda.stream(producer):
                table = prepare_table("cuda")
            with torch.cuda.stream(consumer):
                torch.cuda._sleep(1000000)
                actual = quantize_swiglu_lut(gate, up, valid, weights, table)
            del table
            gc.collect()
            with torch.cuda.stream(producer):
                pressure = [
                    torch.full((65536,), -123.0, device="cuda") for _ in range(8)
                ]
            consumer.synchronize()
            exact(actual, expected)
            producer.synchronize()
            del pressure
        table = prepare_table("cuda")
        consumer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(consumer):
            for _ in range(3):
                quantize_swiglu_lut(gate, up, valid, weights, table)
        consumer.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=consumer):
            captured = quantize_swiglu_lut(gate, up, valid, weights, table)
        for shift in (0.0, 0.25, -0.75):
            gate.add_(shift)
            expected = reference(gate, up, valid, weights, 10.0)
            graph.replay()
            torch.cuda.synchronize()
            exact(captured, expected)

    def test_unsupported_inputs_return_none(self):
        gate = torch.ones(3, 32, device="cuda", dtype=torch.bfloat16)
        valid = torch.ones(3, device="cuda", dtype=torch.bool)
        weights = torch.ones(3, device="cuda")
        table = prepare_table("cuda")
        self.assertIsNone(quantize_swiglu_lut(gate, gate, valid, weights, None))
        self.assertIsNone(
            quantize_swiglu_lut(gate, gate, valid, weights, table, clamp_limit=0)
        )
        self.assertIsNone(
            quantize_swiglu_lut(gate.float(), gate, valid, weights, table)
        )
        self.assertIsNone(
            quantize_swiglu_lut(gate[:0], gate[:0], valid[:0], weights[:0], table)
        )
        self.assertIsNone(
            quantize_swiglu_lut(
                gate.cpu(), gate.cpu(), valid.cpu(), weights.cpu(), table
            )
        )

    def test_nonfinite_output_asserts_in_isolated_contexts(self):
        # Triton snapshots debug mode on import in each spawned interpreter.
        with patch.dict(os.environ, {"TRITON_DEBUG": "1"}):
            for case in ("gate", "up", "weight"):
                with self.subTest(case=case):
                    with self.assertRaises(mp.ProcessRaisedException) as failure:
                        mp.spawn(invalid_child, args=(case,), nprocs=1, join=True)
                    self.assertIn("device-side assert", str(failure.exception))

    def test_owner_selection_fallback_and_graph_replay(self):
        weights = {
            name: torch.randint(low, high, shape, device="cuda", dtype=torch.uint8)
            for name, shape, low, high in (
                ("w13_w", (2, 256, 64), 0, 256),
                ("w2_w", (2, 128, 64), 0, 256),
                ("w13_s", (2, 256, 4), 119, 124),
                ("w2_s", (2, 128, 4), 119, 124),
            )
        }
        with patch.dict(os.environ, {"DSV41_ROUTED_ACT_QUANT_LUT": "0"}):
            control = V41RoutedExperts(weights)
        with patch.dict(os.environ, {"DSV41_ROUTED_ACT_QUANT_LUT": "1"}):
            candidate = V41RoutedExperts(weights)
        self.assertIsNone(control._silu_table)
        self.assertIsNotNone(candidate._silu_table)
        x = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16)
        counts = torch.tensor([3, 5], device="cuda", dtype=torch.int32)
        valid = torch.arange(128, device="cuda")[None] < counts[:, None]
        route = torch.rand(2, 128, device="cuda")
        route[~valid], x[~valid] = float("nan"), float("nan")
        aliased = candidate._silu_table.values.view(torch.bfloat16)[
            : x.numel()
        ].view_as(x)
        with self.assertRaisesRegex(ValueError, "must not share"):
            candidate.forward_masked(x, counts, route, out=aliased)
        module = "rtp_llm.models_py.modules.dsv4._v41_routed_lut_triton"

        def require_lut(*args, **kwargs):
            result = quantize_swiglu_lut(*args, **kwargs)
            self.assertIsNotNone(result, "The routed owner unexpectedly fell back")
            return result

        with patch.dict(os.environ, {"DSV41_ROUTED_ACT_QUANT_LUT": "0"}), patch(
            module + ".quantize_swiglu_lut", side_effect=require_lut
        ) as selected:
            for op in ("masked", "contiguous"):
                if op == "masked":
                    invoke = lambda owner: owner.forward_masked(x, counts, route)
                else:
                    invoke = lambda owner: owner.forward_contiguous(
                        x.flatten(0, 1), (128, 128), valid.flatten(), route.flatten()
                    )
                before = selected.call_count
                expected = invoke(control)
                actual = invoke(candidate)
                self.assertEqual(selected.call_count, before + 1)
                torch.testing.assert_close(
                    actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
                )
                with patch(module + ".quantize_swiglu_lut", return_value=None):
                    fallback = invoke(candidate)
                torch.testing.assert_close(
                    fallback.view(torch.uint8),
                    expected.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                candidate.forward_masked(x, counts, route)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = candidate.forward_masked(x, counts, route)
        gc.collect()
        for shift in (0.0, 0.25):
            x[valid] += shift
            expected = control.forward_masked(x, counts, route)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                captured.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
            )

    def test_public_quantizer_keeps_existing_path_without_a_supported_table(self):
        gate = torch.randn(129, 96, device="cuda", dtype=torch.bfloat16)
        up = torch.randn_like(gate)
        valid = torch.ones(129, device="cuda", dtype=torch.bool)
        route = torch.rand(129, device="cuda")
        expected = reference(gate, up, valid, route, 10.0)
        for table in (None, prepare_table("cuda", 0.0), prepare_table("cuda", 10.0)):
            actual = quantize_swiglu(
                gate,
                up,
                valid,
                route_weights=route,
                clamp_limit=10.0,
                allow_contiguous=True,
                silu_table=table,
            )
            exact(actual, expected)


if __name__ == "__main__":
    unittest.main()
