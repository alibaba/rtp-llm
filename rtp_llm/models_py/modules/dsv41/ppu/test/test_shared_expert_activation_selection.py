"""Real CPU metadata/consumer contracts; doubles do not validate PPU math."""

import importlib.util
import os
import sys
import unittest
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.ppu import local_prefill_sg_ops as sg
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear
from rtp_llm.models_py.modules.dsv41.ppu.moe import shared_expert_forward


def reference(gate_up, clamp_limit=10.0):
    gate, up = gate_up.chunk(2, -1)
    if clamp_limit and clamp_limit > 0:
        gate = gate.clamp(max=float(clamp_limit))
        up = up.clamp(-float(clamp_limit), float(clamp_limit))
    return (F.silu(gate.float()) * up.float()).bfloat16()


class QuantizedConsumer(PPUPTPCLinear):
    """PTPC double distinguishing direct payload consumption from BF16 input."""

    def __init__(self):
        self.forward_quantized = Mock(return_value=object())
        self.unquantized = Mock(side_effect=lambda hidden: hidden)

    def __call__(self, hidden):
        return self.unquantized(hidden)


@contextmanager
def providers(*, ppu=True, capture=False, triton=True, capability=(8, 9)):
    activation = ModuleType("activation_double")
    activation.silu_mul_split_bf16 = Mock(side_effect=reference)
    activation.silu_mul_split_bf16_torch_exact = Mock(
        side_effect=AssertionError("retired exact provider")
    )
    quant = ModuleType("quant_double")
    quant.is_ppu = lambda: ppu
    original_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "triton":
            return object() if triton else None
        return original_find_spec(name, *args, **kwargs)

    with (
        patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.modules.dsv41.moe._silu_mul_bf16_triton": activation,
                "rtp_llm.models_py.modules.dsv41.ppu._quant": quant,
            },
        ),
        patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
        patch.object(torch.cuda, "device", side_effect=lambda device: nullcontext()),
        patch.object(torch.cuda, "get_device_capability", return_value=capability),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=capture),
        patch.object(torch.version, "hip", None),
        patch.object(importlib.util, "find_spec", side_effect=find_spec),
        patch.object(sg, "swiglu_quant", return_value=(object(), object())) as quantize,
    ):
        yield SimpleNamespace(activation=activation, quantize=quantize)


def call(value, w2=None, limit=10.0, **options):
    return shared_expert_forward(
        value, lambda x: x, (lambda x: x) if w2 is None else w2, limit, **options
    )


class SharedExpertActivationSelectionTest(unittest.TestCase):
    @torch.no_grad()
    def test_quantized_consumer_uses_sg_without_requantization(self):
        for rows in (0, 1, 8, 24, 32, 140):
            with self.subTest(rows=rows), providers() as provider:
                value = torch.randn(rows, 4608).bfloat16()
                before = value.clone()
                consumer = QuantizedConsumer()
                output = call(value, consumer)
                provider.quantize.assert_called_once_with(value, 10.0)
                consumer.forward_quantized.assert_called_once_with(
                    *provider.quantize.return_value
                )
                self.assertIs(output, consumer.forward_quantized.return_value)
                consumer.unquantized.assert_not_called()
                provider.activation.silu_mul_split_bf16.assert_not_called()
                provider.activation.silu_mul_split_bf16_torch_exact.assert_not_called()
                self.assertTrue(torch.equal(value, before))

    @torch.no_grad()
    def test_non_ptpc_consumer_uses_fast(self):
        value = torch.randn(3, 4608).bfloat16()
        for consumer in (None, Mock(side_effect=lambda hidden: hidden)):
            with self.subTest(consumer=consumer), providers() as provider:
                if consumer is not None:
                    consumer.forward_quantized = Mock()
                output = call(value, consumer)
                provider.activation.silu_mul_split_bf16.assert_called_once_with(
                    value, clamp_limit=10.0
                )
                provider.quantize.assert_not_called()
                self.assertTrue(torch.equal(output, reference(value)))

    @torch.no_grad()
    def test_sg_decline_and_graph_warmup_use_fast(self):
        value = torch.randn(3, 4608).bfloat16()
        for settings, options in (
            ({"ppu": False}, {}),
            ({"capability": (8, 0)}, {}),
            ({"capture": True}, {}),
            ({}, {"allow_quant": False}),
            ({"capture": True}, {"allow_quant": False}),
        ):
            with (
                self.subTest(settings=settings, options=options),
                providers(**settings) as provider,
            ):
                consumer = QuantizedConsumer()
                output = call(value, consumer, **options)
                provider.quantize.assert_not_called()
                consumer.forward_quantized.assert_not_called()
                provider.activation.silu_mul_split_bf16.assert_called_once_with(
                    value, clamp_limit=10.0
                )
                self.assertTrue(torch.equal(output, reference(value)))

    @torch.no_grad()
    def test_sg_unsupported_width_limit_and_alignment_use_fast(self):
        good = torch.randn(3, 4608).bfloat16()
        unaligned = torch.empty(2 * 4608 + 1, dtype=torch.bfloat16)[1:].view(2, 4608)
        unaligned.copy_(good[:2])
        for value, limit in (
            (good[:, :32].contiguous(), 10.0),
            (good, 0.0),
            (good, 9.0),
            (unaligned, 10.0),
        ):
            with self.subTest(shape=value.shape, limit=limit), providers() as provider:
                output = call(value, QuantizedConsumer(), limit)
                provider.quantize.assert_not_called()
                provider.activation.silu_mul_split_bf16.assert_called_once_with(
                    value, clamp_limit=limit
                )
                self.assertTrue(torch.equal(output, reference(value, limit)))

    @torch.no_grad()
    def test_legacy_flags_cannot_restore_exact_or_disable_sg(self):
        flags = (
            "EXACT_SHARED_SWIGLU",
            "CP8_FUSED_SHARED_SWIGLU",
            "LOCAL_PREFILL_EXACT_SHARED_SWIGLU",
            "VERIFY_EXACT_SWIGLU",
            "CP8_YOCO_FUSED_SWIGLU",
            "LOCAL_PREFILL_SG_ALIGN",
            "PROPOSAL_EXACT_OPS",
        )
        value = torch.randn(1, 4608).bfloat16()
        for flag in flags:
            for setting in ("0", "1"):
                with (
                    self.subTest(flag=flag, setting=setting),
                    patch.dict(os.environ, {"RTP_V41_PPU_" + flag: setting}),
                    providers() as provider,
                ):
                    call(value, QuantizedConsumer())
                    provider.quantize.assert_called_once()
                    call(value)
                    provider.activation.silu_mul_split_bf16.assert_called_once()
                    provider.activation.silu_mul_split_bf16_torch_exact.assert_not_called()

    @torch.no_grad()
    def test_cpu_and_unsupported_fast_metadata_keep_portable_reference(self):
        good = torch.randn(3, 4608).bfloat16()
        self.assertTrue(torch.equal(call(good), reference(good)))
        for value in (good.float(), good[:, ::2], torch._neg_view(good)):
            with (
                self.subTest(shape=value.shape, dtype=value.dtype),
                providers() as provider,
            ):
                output = call(value, QuantizedConsumer())
                provider.quantize.assert_not_called()
                provider.activation.silu_mul_split_bf16.assert_not_called()
                self.assertTrue(torch.equal(output, reference(value)))
        with providers(triton=False) as provider:
            self.assertTrue(torch.equal(call(good), reference(good)))
            provider.activation.silu_mul_split_bf16.assert_not_called()

    def test_grad_enabled_keeps_autograd_reference(self):
        value = torch.randn(3, 32, requires_grad=True)
        with providers() as provider:
            call(value).float().sum().backward()
            self.assertIsNotNone(value.grad)
            provider.quantize.assert_not_called()
            provider.activation.silu_mul_split_bf16.assert_not_called()

    @torch.no_grad()
    def test_provider_launch_errors_propagate(self):
        value = torch.randn(1, 4608).bfloat16()
        with providers() as provider:
            provider.quantize.side_effect = RuntimeError("SG launch failed")
            with self.assertRaisesRegex(RuntimeError, "SG launch failed"):
                call(value, QuantizedConsumer())
            provider.activation.silu_mul_split_bf16.assert_not_called()
        with providers() as provider:
            provider.activation.silu_mul_split_bf16.side_effect = RuntimeError(
                "fast launch failed"
            )
            with self.assertRaisesRegex(RuntimeError, "fast launch failed"):
                call(value)


if __name__ == "__main__":
    unittest.main()
