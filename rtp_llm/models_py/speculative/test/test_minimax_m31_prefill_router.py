"""CPU ownership/dispatch gates and focused CUDA arithmetic checks."""

import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.minimax_m3 import (
    MiniMaxM3DecoderLayer,
    MiniMaxM3Model,
)
from rtp_llm.models_py.model_desc.minimax_m31 import (
    MiniMaxM31DecoderLayer,
    MiniMaxM31Model,
    MiniMaxM31MoeLayer,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.triton_kernels.minimax_m31_prefill_router import (
    expand_fp32_router_weight,
    minimax_m31_prefill_bf16_router_logits,
    minimax_m31_prefill_router_logits,
)
from rtp_llm.ops import RoleType


def bare_mlp():
    mlp = object.__new__(MiniMaxM31MoeLayer)
    nn.Module.__init__(mlp)
    mlp.gate = CudaF16Linear(torch.randn(6144, 128, dtype=torch.float32) * 0.01)
    return mlp


class PrefillRouterOwnershipTest(unittest.TestCase):
    def test_prepared_bf16_gate_dispatches_only_prefill(self):
        for expanded_first in (False, True):
            with self.subTest(expanded_first=expanded_first):
                mlp = bare_mlp()
                if expanded_first:
                    mlp.prepare_prefill_router()
                mlp.gate = CudaF16Linear(
                    torch.randn(6144, 128, dtype=torch.bfloat16) * 0.01
                )
                mlp.prepare_prefill_router()
                self.assertFalse(any("_prefill_gate" in name for name in mlp._buffers))
                self.assertTrue(mlp._prefill_bf16_router_ready)
                x = torch.randn(17, 6144, dtype=torch.bfloat16)
                for prefill, invariant in ((True, False), (False, True)):
                    mlp._prefill_router_active = prefill
                    mlp._batch_invariant_router = invariant
                    expected = mlp.gate(x)
                    with patch(
                        "rtp_llm.models_py.triton_kernels.minimax_m31_prefill_router.minimax_m31_prefill_bf16_router_logits",
                        return_value=expected,
                    ) as kernel:
                        torch.testing.assert_close(
                            mlp._compute_router_logits(x), expected, rtol=0, atol=0
                        )
                        self.assertEqual(kernel.call_count, int(prefill))

    def test_unprepared_bf16_gate_keeps_legacy_linear(self):
        mlp = bare_mlp()
        mlp.gate = CudaF16Linear(torch.randn(6144, 128, dtype=torch.bfloat16))
        mlp._prefill_router_active = True
        x = torch.randn(3, 6144, dtype=torch.bfloat16)
        with patch(
            "rtp_llm.models_py.triton_kernels.minimax_m31_prefill_router.minimax_m31_prefill_bf16_router_logits"
        ) as kernel:
            torch.testing.assert_close(
                mlp._compute_router_logits(x), mlp.gate(x), rtol=0, atol=0
            )
            kernel.assert_not_called()

    def test_exact_expansion_and_layout(self):
        for weight in (
            torch.randn(128, 6144) * 0.01,
            (torch.randn(6144, 128) * 0.01).T,
        ):
            parts = expand_fp32_router_weight(weight)
            self.assertTrue(
                torch.equal(
                    (parts[0].float() + parts[1].float()) + parts[2].float(), weight
                )
            )
            self.assertTrue(all(p.stride() == weight.stride() for p in parts))

    def test_invalid_weight_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "FP32"):
            expand_fp32_router_weight(torch.zeros(128, 6144, dtype=torch.bfloat16))
        for value in (float("nan"), float("inf")):
            weight = torch.zeros(128, 6144)
            weight[0, 0] = value
            with self.assertRaisesRegex(ValueError, "finite"):
                expand_fp32_router_weight(weight)
        weight = torch.zeros(128, 6144)
        weight[0, 0] = torch.finfo(torch.float32).tiny * 2**-23
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            expand_fp32_router_weight(weight)

    def test_nonpersistent_model_owned_parts_and_reload(self):
        mlp = bare_mlp()
        raw = mlp.gate.weight.clone()
        mlp.prepare_prefill_router()
        self.assertTrue(torch.equal(raw, mlp.gate.weight))
        self.assertFalse(any("_prefill_gate" in name for name in mlp.state_dict()))
        first = mlp._prefill_gate_high
        mlp.gate.weight.mul_(2)
        mlp.prepare_prefill_router()
        self.assertIsNot(first, mlp._prefill_gate_high)
        self.assertTrue(
            torch.equal(
                (mlp._prefill_gate_high.float() + mlp._prefill_gate_middle.float())
                + mlp._prefill_gate_low.float(),
                mlp.gate.weight,
            )
        )

    def test_role_and_native_cache_ownership(self):
        for role in (RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION):
            for native in (False, True):
                model = object.__new__(MiniMaxM31Model)
                nn.Module.__init__(model)
                mlp = bare_mlp()
                model.parallelism_config = types.SimpleNamespace(
                    role_type=role, dp_rank=0
                )
                model.layers = [
                    types.SimpleNamespace(
                        mlp=mlp, self_attn=types.SimpleNamespace(nvfp4_kv_cache=native)
                    )
                ]
                with patch.object(
                    MiniMaxM3Model, "initialize", return_value=True
                ), patch.object(mlp, "prepare_prefill_router") as prepare:
                    model.initialize(
                        types.SimpleNamespace(is_decode_role=role == RoleType.DECODE)
                    )
                    self.assertEqual(
                        prepare.call_count, int(role == RoleType.PREFILL and native)
                    )

    def test_phase_and_dispatch(self):
        layer = object.__new__(MiniMaxM31DecoderLayer)
        nn.Module.__init__(layer)
        layer.mlp = bare_mlp()
        layer.mlp.prepare_prefill_router()
        x = torch.randn(2, 6144, dtype=torch.bfloat16)
        kernel_path = "rtp_llm.models_py.triton_kernels.minimax_m31_prefill_router.minimax_m31_prefill_router_logits"
        with patch.object(
            MiniMaxM3DecoderLayer, "_forward_attention", return_value=(x, None)
        ):
            for prefill, verify in (
                (True, False),
                (True, True),
                (False, False),
                (True, False),
            ):
                layer._forward_attention(
                    x,
                    None,
                    None,
                    None,
                    False,
                    types.SimpleNamespace(is_prefill=prefill, is_target_verify=verify),
                )
                self.assertEqual(
                    layer.mlp._prefill_router_active, prefill and not verify
                )
        layer.mlp._prefill_router_active = True
        with patch(kernel_path, return_value=x.float()) as kernel:
            layer.mlp._compute_router_logits(x)
            kernel.assert_called_once()
        with patch(kernel_path) as kernel:
            result = layer.mlp._compute_router_logits(x.float())
            kernel.assert_not_called()
            torch.testing.assert_close(
                result, layer.mlp.gate(x.float()), rtol=0, atol=0
            )

    def test_graph_clone_does_not_own_parts(self):
        mlp = bare_mlp()
        mlp.prepare_prefill_router()
        mlp._prefill_router_active = True
        # Populate the actual GenericMoeLayer clone contract without loading experts.
        for name in (
            "config",
            "parallelism_config",
            "hidden_dim",
            "ffn_dim",
            "num_experts",
            "top_k",
            "select_topk",
            "fake_balance_expert",
            "w1",
            "w2",
            "num_local_experts",
            "add_shared_expert",
            "ffn_tp_size",
            "ep_size",
            "shared_expert",
            "_shared_overlap",
            "shared_expert_gate",
            "sigmoid_gate_scale_add",
            "correction_bias",
            "_use_mega_moe_fused_shared",
        ):
            setattr(mlp, name, None)
        mlp.fused_moe = nn.Identity()
        clone = mlp.clone_for_cuda_graph()
        self.assertIs(clone.gate, mlp.gate)
        self.assertFalse(clone._prefill_router_active)
        self.assertFalse(clone._prefill_bf16_router_ready)
        self.assertFalse(hasattr(clone, "_prefill_gate_high"))
        self.assertEqual(list(clone.named_buffers()), [])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PrefillRouterCudaTest(unittest.TestCase):
    def test_bf16_graph_replay_updates_padded_rows(self):
        torch.manual_seed(1017)
        weight = torch.randn(128, 6144, device="cuda", dtype=torch.bfloat16) * 0.01
        x = torch.randn(129, 6144, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                minimax_m31_prefill_bf16_router_logits(x, weight)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = minimax_m31_prefill_bf16_router_logits(x, weight)
        address = output.data_ptr()
        for live_rows in (129, 17, 1, 128):
            x.zero_()
            x[:live_rows].normal_()
            expected = minimax_m31_prefill_bf16_router_logits(x, weight)
            graph.replay()
            self.assertEqual(output.data_ptr(), address)
            self.assertTrue(torch.equal(output, expected))
            if live_rows < x.shape[0]:
                self.assertEqual(torch.count_nonzero(output[live_rows:]).item(), 0)

    def test_bf16_rows_layouts_and_padding(self):
        torch.manual_seed(1015)
        for weight in (
            (torch.randn(6144, 128, device="cuda", dtype=torch.bfloat16) * 0.01).T,
            torch.randn(128, 6144, device="cuda", dtype=torch.bfloat16) * 0.01,
        ):
            row = torch.randn(1, 6144, device="cuda", dtype=torch.bfloat16)
            expected = minimax_m31_prefill_bf16_router_logits(row, weight)
            for count in (0, 1, 15, 16, 17, 127, 129, 1934):
                x = torch.randn(count, 6144, device="cuda", dtype=torch.bfloat16)
                if count:
                    x[-1].copy_(row[0])
                output = minimax_m31_prefill_bf16_router_logits(x, weight)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(tuple(output.shape), (count, 128))
                if count:
                    self.assertTrue(torch.equal(output[-1], expected[0]))
                    torch.testing.assert_close(
                        output.float(),
                        (x.double() @ weight.double().T).float(),
                        rtol=2**-7,
                        atol=2**-14,
                    )

    def test_bf16_int32_boundary_and_one_million_rows(self):
        torch.manual_seed(1016)
        row = torch.randn(1, 6144, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 6144, device="cuda", dtype=torch.bfloat16) * 0.01
        expected = minimax_m31_prefill_bf16_router_logits(row, weight)
        for count in (349525, 349526, 1048576):
            with self.subTest(rows=count):
                x = torch.zeros(count, 6144, device="cuda", dtype=torch.bfloat16)
                x[-1].copy_(row[0])
                output = minimax_m31_prefill_bf16_router_logits(x, weight)
                self.assertTrue(torch.equal(output[-1], expected[0]))
                self.assertTrue(torch.isfinite(output).all().item())
                self.assertTrue(torch.equal(output[0], torch.zeros_like(output[0])))
                del x, output

    def test_int32_boundary_and_one_million_rows(self):
        torch.manual_seed(1014)
        row = torch.randn(1, 6144, device="cuda", dtype=torch.bfloat16)
        for weight in (
            (torch.randn(6144, 128, device="cuda") * 0.01).T,
            torch.randn(128, 6144, device="cuda") * 0.01,
        ):
            parts = expand_fp32_router_weight(weight)
            expected = minimax_m31_prefill_router_logits(row, parts)
            for count in (349525, 349526, 1048576):
                with self.subTest(rows=count, stride=weight.stride()):
                    x = torch.zeros(count, 6144, device="cuda", dtype=torch.bfloat16)
                    x[-1].copy_(row[0])
                    result = minimax_m31_prefill_router_logits(x, parts)
                    self.assertTrue(torch.equal(result[-1], expected[0]))
                    self.assertTrue(
                        torch.equal(
                            result[[0, count // 2]],
                            torch.zeros_like(result[[0, count // 2]]),
                        )
                    )
                    self.assertTrue(torch.isfinite(result).all().item())
                    del x, result

    def test_rows_layouts_and_padding(self):
        torch.manual_seed(1013)
        for weight in (
            (torch.randn(6144, 128, device="cuda") * 0.01).T,
            torch.randn(128, 6144, device="cuda") * 0.01,
        ):
            parts = expand_fp32_router_weight(weight)
            row = torch.randn(1, 6144, device="cuda", dtype=torch.bfloat16)
            first = minimax_m31_prefill_router_logits(row, parts)
            for count in (0, 1, 15, 16, 17, 127, 129, 1934):
                x = torch.randn(count, 6144, device="cuda", dtype=torch.bfloat16)
                if count:
                    x[-1].copy_(row[0])
                result = minimax_m31_prefill_router_logits(x, parts)
                self.assertEqual(result.dtype, torch.float32)
                self.assertEqual(tuple(result.shape), (count, 128))
                if count:
                    self.assertTrue(torch.equal(result[-1], first[0]))
                    torch.testing.assert_close(
                        result.double(),
                        x.double() @ weight.double().T,
                        rtol=2e-5,
                        atol=2e-5,
                    )


if __name__ == "__main__":
    unittest.main()
