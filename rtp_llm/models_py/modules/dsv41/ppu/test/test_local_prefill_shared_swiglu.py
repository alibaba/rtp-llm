"""Shared-expert caller routing and opt-in SG/fast device contracts.

CPU tests execute the real caller classes with unrelated model dependencies
replaced. Device tests compare the selected providers, not the retired ATen
activation contract or a CUDA golden.
"""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.ppu import moe, optimization_config


def caller_classes():
    """Load actual classes without importing unrelated compiled model bindings."""
    path = Path(__file__).resolve().parents[1] / "layers.py"
    tree = ast.parse(path.read_text())
    names = {
        "PPUSharedExpert",
        "PPUSharedMoE",
        "_ordinary_decode_graph",
        "_verify_packed_attention_graph",
        "_proposal_exact_ops",
        "_full_cp8_prefill_context",
        "cp8_yoco_operator_eligible",
        "cp8_yoco_fast_prefill_enabled",
    }
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    namespace = {
        "os": os,
        "torch": torch,
        "nn": nn,
        "shared_expert_forward": moe.shared_expert_forward,
        "combine_routed_and_shared": moe.combine_routed_and_shared,
        "pad_topk_for_dispatch": moe.pad_topk_for_dispatch,
        "optimization_enabled": optimization_config.enabled,
        "DIRECT_MOE_ROUTES": optimization_config.DIRECT_MOE_ROUTES,
        "_local_prefill_direct_moe_real": lambda state, flat: None,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def owner_fixture(namespace, *, phase="local", rows=3):
    unit = namespace["PPUSharedMoE"].__new__(namespace["PPUSharedMoE"])
    nn.Module.__init__(unit)
    unit.routed_enabled, unit.calls, unit.layer_id = True, 0, 0
    unit.shared_expert = Mock(side_effect=lambda value, **options: value)
    unit.gate = Mock(
        return_value=(
            torch.zeros(rows, 6, dtype=torch.int64),
            torch.ones(rows, 6),
        )
    )
    runtime = NS(
        model=NS(_is_speculative=phase in ("verify", "proposal")),
        policy=NS(mode="compact"),
    )
    state = NS(
        runtime=runtime,
        context=None,
        is_graph=phase in ("decode", "verify", "proposal"),
        verify=phase == "verify",
        proposal=phase == "proposal",
        prefill=phase in ("local", "cp8"),
        idle=False,
        compacted=False,
        grouped_moe=None,
        verify_moe_overlap=None,
        routed_rows={},
        packed_attention=object() if phase == "verify" else None,
        metadata=NS(valid_mask=torch.ones(rows, dtype=torch.bool)),
        routed_rows_device=[torch.zeros((), dtype=torch.int64)],
        moe_calls=torch.zeros(1, dtype=torch.int64),
        bulk_counters=False,
    )
    if phase == "cp8":
        state.context = NS(
            cp_size=8,
            kv_cache_sharded=True,
            gather_restore_positions=None,
            local_is_real=torch.ones(rows, dtype=torch.bool),
        )
    runtime.current = state
    unit.runtime = runtime
    engine = NS(
        forward_shared_graph=Mock(
            side_effect=lambda value, *args, **options: torch.zeros_like(value)
        )
    )
    unit.ops = NS(
        moe_engine=engine,
        moe=Mock(side_effect=lambda value, *args, **options: torch.zeros_like(value)),
    )
    return unit, state


class LocalPrefillSharedSwiGLUCPU(unittest.TestCase):
    @torch.no_grad()
    def test_shared_expert_forwards_only_quant_capability(self):
        namespace = caller_classes()
        expert = namespace["PPUSharedExpert"].__new__(namespace["PPUSharedExpert"])
        nn.Module.__init__(expert)
        expert.w13, expert.w2, expert.swiglu_limit = object(), object(), 10.0
        value = torch.randn(3, 32).bfloat16()
        operation = Mock(return_value=value)
        namespace["shared_expert_forward"] = operation
        for allow_quant in (True, False):
            operation.reset_mock()
            self.assertIs(expert(value, allow_quant=allow_quant), value)
            operation.assert_called_once_with(
                value, expert.w13, expert.w2, 10.0, allow_quant=allow_quant
            )

    @torch.no_grad()
    def test_all_phase_callers_share_policy_independent_of_legacy_flags(self):
        namespace = caller_classes()
        flags = (
            "EXACT_SHARED_SWIGLU",
            "CP8_FUSED_SHARED_SWIGLU",
            "LOCAL_PREFILL_EXACT_SHARED_SWIGLU",
            "VERIFY_EXACT_SWIGLU",
            "CP8_YOCO_FUSED_SWIGLU",
            "CP8_YOCO_FAST_PREFILL",
            "LOCAL_PREFILL_SG_ALIGN",
            "PROPOSAL_EXACT_OPS",
        )
        value = torch.randn(3, 32).bfloat16()
        for phase in ("local", "cp8", "decode", "verify", "proposal"):
            for setting in ("0", "1"):
                with self.subTest(phase=phase, setting=setting), patch.dict(
                    os.environ, {"RTP_V41_PPU_" + flag: setting for flag in flags}
                ):
                    unit, state = owner_fixture(namespace, phase=phase)
                    state.local_prefill_sg_align = setting == "1"
                    output = unit(value)
                    self.assertTrue(torch.equal(output, value))
                    self.assertEqual(unit.shared_expert.call_count, 1)
                    self.assertEqual(
                        unit.shared_expert.call_args.kwargs,
                        {"allow_quant": False} if state.is_graph else {},
                    )
                    argument = unit.shared_expert.call_args.args[0]
                    self.assertEqual(argument.data_ptr(), value.data_ptr())

    @torch.no_grad()
    def test_graph_separate_shared_return_keeps_both_branches(self):
        namespace = caller_classes()
        value = torch.randn(3, 32).bfloat16()
        for phase in ("decode", "verify", "proposal"):
            with self.subTest(phase=phase):
                unit, _ = owner_fixture(namespace, phase=phase)
                routed, shared = unit(value, _return_shared=True)
                self.assertTrue(torch.equal(routed, torch.zeros_like(value)))
                self.assertTrue(torch.equal(shared, value))
                unit.shared_expert.assert_called_once()
                self.assertEqual(
                    unit.shared_expert.call_args.kwargs, {"allow_quant": False}
                )


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1",
    "explicit PPU device test",
)
class LocalPrefillSharedSwiGLUDevice(unittest.TestCase):
    def setUp(self):
        from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu

        self.assertTrue(is_ppu(), "requires an actual PPU")
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))

    def assert_bits(self, actual, expected):
        self.assertEqual((actual.shape, actual.dtype), (expected.shape, expected.dtype))
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    @torch.no_grad()
    def test_fast_activation_for_non_quantized_consumer(self):
        from rtp_llm.models_py.modules.dsv41.moe._silu_mul_bf16_triton import (
            silu_mul_split_bf16,
        )

        for rows in (0, 1, 8, 24, 32, 140, 4096):
            with self.subTest(rows=rows):
                value = torch.randn(rows, 4608, device="cuda", dtype=torch.bfloat16)
                before = value.clone()
                actual = moe.shared_expert_forward(value, lambda x: x, lambda x: x)
                self.assert_bits(actual, silu_mul_split_bf16(value, 10.0))
                self.assert_bits(value, before)

    @torch.no_grad()
    def test_real_sg_payload_and_ptpc_w2(self):
        from rtp_llm.models_py.modules.dsv41.ppu import local_prefill_sg_ops as sg
        from rtp_llm.models_py.modules.dsv41.ppu.linear import (
            PPUPTPCLinear,
            PPUResidentBF16Linear,
        )

        w13 = PPUResidentBF16Linear(
            torch.randn(4608, 64, device="cuda", dtype=torch.bfloat16) * 0.1
        )
        w2 = PPUPTPCLinear(
            (torch.randn(32, 2304, device="cuda") * 0.1).to(torch.float8_e4m3fn),
            torch.ones(32, 1, device="cuda"),
        )
        for rows in (0, 1, 140, 4096):
            with self.subTest(rows=rows):
                value = torch.randn(rows, 64, device="cuda", dtype=torch.bfloat16)
                payload, scale = sg.swiglu_quant(w13(value), 10.0)
                expected = w2.forward_quantized(payload, scale)
                with patch.object(
                    w2,
                    "quantize_input",
                    side_effect=AssertionError("unexpected requant"),
                ):
                    actual = moe.shared_expert_forward(value, w13, w2)
                self.assert_bits(actual, expected)

    @torch.no_grad()
    def test_graph_warmup_capture_and_replay_use_fast(self):
        from rtp_llm.models_py.modules.dsv41.moe._silu_mul_bf16_triton import (
            silu_mul_split_bf16,
        )
        from rtp_llm.models_py.modules.dsv41.ppu import local_prefill_sg_ops as sg
        from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear

        value = torch.randn(32, 4608, device="cuda", dtype=torch.bfloat16)
        w2 = PPUPTPCLinear(
            (torch.randn(32, 2304, device="cuda") * 0.1).to(torch.float8_e4m3fn),
            torch.ones(32, 1, device="cuda"),
        )
        with patch.object(sg, "swiglu_quant", side_effect=AssertionError("Graph SG")):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    moe.shared_expert_forward(value, lambda x: x, w2, allow_quant=False)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = moe.shared_expert_forward(
                    value, lambda x: x, w2, allow_quant=False
                )
            for _ in range(2):
                value.copy_(torch.randn_like(value))
                graph.replay()
                self.assert_bits(actual, w2(silu_mul_split_bf16(value, 10.0)))


if __name__ == "__main__":
    unittest.main()
