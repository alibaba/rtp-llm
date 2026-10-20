# SPDX-License-Identifier: Apache-2.0
"""MegaGate eligibility, native call contract and optional CUDA replay checks."""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch
import torch.nn.functional as F

# Exercise the wrapper without importing the engine's native bindings.
_PATH = Path(__file__).resolve().parent.parent / "moe" / "mega_gate.py"
_SPEC = importlib.util.spec_from_file_location("_mega_gate_under_test", _PATH)
_MEGA_GATE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MEGA_GATE)


def _gate(weight, bias, topk=6):
    return SimpleNamespace(
        hash=False,
        score_func="sqrtsoftplus",
        weight=weight,
        bias=bias,
        topk=topk,
        route_scale=2.5,
        _weight_bf16=lambda: weight.to(torch.bfloat16),
    )


class MegaGateHostTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ,
            {"DSV4_GATE_FP32": "0", "DSV4_GATE_FUSED": "1", "MOEDBG": "0"},
        )
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("DSV4_MEGA_GATE", None)
        self.x = torch.empty((8, 256), dtype=torch.bfloat16)
        self.gate = _gate(
            torch.empty((384, 256), dtype=torch.bfloat16), torch.zeros(384)
        )

    def test_cpu_rejects_before_gate_or_cuda_access(self):
        with patch.object(_MEGA_GATE, "_native_gate") as native, patch.object(
            torch.cuda, "get_device_capability"
        ) as capability:
            self.assertFalse(_MEGA_GATE.can_use_mega_gate(self.x, object()))
        native.assert_not_called()
        capability.assert_not_called()

    def test_invalid_input_metadata_rejects_before_gate_access(self):
        cases = (
            self.x.unsqueeze(0),
            self.x.float(),
            self.x[:0],
            self.x[:, ::2],
            torch.empty(((1 << 20) + 1, 256), device="meta", dtype=torch.bfloat16),
        )
        with patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ), patch.object(_MEGA_GATE, "_native_gate") as native, patch.object(
            torch.cuda, "get_device_capability"
        ) as capability:
            for x in cases:
                with self.subTest(shape=x.shape, dtype=x.dtype):
                    self.assertFalse(_MEGA_GATE.can_use_mega_gate(x, object()))
        native.assert_not_called()
        capability.assert_not_called()

    def test_default_enabled_for_small_decode_and_prefill_sizes(self):
        with patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ), patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 0)
        ), patch.object(
            _MEGA_GATE, "_native_gate", return_value=Mock()
        ):
            for batch in (1, 32, 384, 2304, 4096):
                with self.subTest(batch=batch):
                    x = torch.empty((batch, 256), dtype=torch.bfloat16)
                    self.assertTrue(_MEGA_GATE.can_use_mega_gate(x, self.gate))

    def test_routing_and_environment_overrides_keep_old_path(self):
        with patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ), patch.object(_MEGA_GATE, "_native_gate") as native:
            for name, value in (
                ("DSV4_MEGA_GATE", "0"),
                ("DSV4_GATE_FP32", "1"),
                ("DSV4_GATE_FUSED", "0"),
                ("MOEDBG", "1"),
            ):
                with self.subTest(env=name), patch.dict(os.environ, {name: value}):
                    self.assertFalse(_MEGA_GATE.can_use_mega_gate(self.x, self.gate))
            for name, value in (
                ("hash", True),
                ("score_func", "sigmoid"),
                ("_ced_row_projection", object()),
                ("_dbg_prefix", "debug"),
                ("_has_visual_tokens", True),
            ):
                with self.subTest(router=name), patch.object(
                    self.gate, name, value, create=True
                ):
                    self.assertFalse(_MEGA_GATE.can_use_mega_gate(self.x, self.gate))
        native.assert_not_called()

    def test_missing_native_or_unsupported_device_keeps_old_path(self):
        with patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ):
            for arch, kernel in (((9, 0), Mock()), ((10, 0), None)):
                with self.subTest(arch=arch, kernel=kernel), patch.object(
                    torch.cuda, "get_device_capability", return_value=arch
                ), patch.object(_MEGA_GATE, "_native_gate", return_value=kernel):
                    self.assertFalse(_MEGA_GATE.can_use_mega_gate(self.x, self.gate))

    def test_native_out_order_and_global_ids_are_preserved(self):
        global_ids = torch.tensor([0, 63, 64, 127, 128, 383])

        def fill_out(*args, **kwargs):
            indices, weights = kwargs["out"]
            indices.copy_(global_ids.expand_as(indices))
            weights.fill_(self.gate.route_scale / self.gate.topk)

        kernel = Mock(side_effect=fill_out)
        with patch.object(_MEGA_GATE, "_native_gate", return_value=kernel):
            weights, indices = _MEGA_GATE.mega_gate(self.x, self.gate, ep_rank=3)
        args, kwargs = kernel.call_args
        self.assertIs(args[0], self.x)
        self.assertIs(args[1], self.gate.weight)
        self.assertEqual(args[2], 6)
        self.assertIs(kwargs.pop("bias"), self.gate.bias)
        out_indices, out_weights = kwargs.pop("out")
        self.assertIs(indices, out_indices)
        self.assertIs(weights, out_weights)
        self.assertEqual(
            kwargs,
            dict(
                use_shared_as_routed=False,
                num_shared_experts=0,
                routed_scaling_factor=2.5,
                ep_rank=3,
                scoring_func="sqrtsoftplus",
            ),
        )
        self.assertEqual(indices.dtype, torch.int64)
        self.assertEqual(weights.dtype, torch.float32)
        torch.testing.assert_close(indices, global_ids.expand(8, 6))
        torch.testing.assert_close(weights.sum(-1), torch.full((8,), 2.5))

    def test_native_errors_propagate_without_retry(self):
        kernel = Mock(side_effect=RuntimeError("native failure"))
        with patch.object(_MEGA_GATE, "_native_gate", return_value=kernel):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                _MEGA_GATE.mega_gate(self.x, self.gate)
        kernel.assert_called_once()

    def test_caller_owned_outputs_receive_routes(self):
        indices = torch.full((8, 6), -1, dtype=torch.int64)
        weights = torch.zeros((8, 6), dtype=torch.float32)

        def fill_out(*args, out, **kwargs):
            out[0].copy_(torch.arange(6).expand(8, 6))
            out[1].fill_(2.5 / 6)

        with patch.object(_MEGA_GATE, "_native_gate", return_value=fill_out):
            actual_weights, actual_indices = _MEGA_GATE.mega_gate(
                self.x, self.gate, out=(indices, weights)
            )
        self.assertIs(actual_indices, indices)
        self.assertIs(actual_weights, weights)
        torch.testing.assert_close(indices, torch.arange(6).expand(8, 6))
        torch.testing.assert_close(weights.sum(-1), torch.full((8,), 2.5))


class MegaGateMoETest(unittest.TestCase):
    def test_forward_and_chunk_preserve_strategy_and_shared_expert_paths(self):
        from rtp_llm.models_py.modules.dsv41.moe import mega_gate as gate_module
        from rtp_llm.models_py.modules.dsv41.moe.moe_layer import MoE

        x = torch.ones((8, 256), dtype=torch.bfloat16)
        input_ids = torch.arange(8)
        weights = torch.full((8, 6), 1 / 6)
        indices = torch.arange(6).expand(8, 6)
        for chunk in (False, True):
            for includes_shared in (False, True):
                for enabled in (False, True):
                    with self.subTest(
                        chunk=chunk, includes_shared=includes_shared, enabled=enabled
                    ):
                        moe = MoE.__new__(MoE)
                        torch.nn.Module.__init__(moe)
                        moe.layer_id, moe.dim, moe.max_tokens_per_rank = 0, 256, 8
                        moe._is_decode_role = True
                        moe._gate_pack_static = True
                        moe._has_visual_tokens = False
                        moe._routed_includes_shared = includes_shared
                        moe.gate = object()
                        strategy = Mock(return_value=x * 2)
                        strategy.cfg = SimpleNamespace(ep_rank=3)
                        strategy.forward_with_gate_pack.return_value = x * 2
                        moe._strategy = strategy
                        moe.shared_experts = object()
                        moe._shared_executor = Mock()
                        moe._shared_executor.finish.return_value = torch.ones_like(x)
                        with patch.object(
                            gate_module, "can_use_mega_gate", return_value=enabled
                        ), patch.object(
                            gate_module, "mega_gate", return_value=(weights, indices)
                        ) as native, patch.dict(
                            os.environ, {"DSV4_SHARED_EXPERT_BF16_ADD": "1"}
                        ):
                            if chunk:
                                actual = torch.empty_like(x)
                                moe._run_chunk(x, input_ids, actual)
                            else:
                                actual = moe(x, input_ids)
                        torch.testing.assert_close(
                            actual, x * (2 if includes_shared else 3)
                        )
                        if enabled:
                            native.assert_called_once_with(x, moe.gate, 3)
                            strategy.assert_called_once_with(x, weights, indices)
                            strategy.forward_with_gate_pack.assert_not_called()
                        else:
                            native.assert_not_called()
                            strategy.assert_not_called()
                            strategy.forward_with_gate_pack.assert_called_once_with(
                                x, moe.gate, input_ids
                            )
                        self.assertEqual(
                            moe._shared_executor.finish.call_count,
                            0 if includes_shared else 1,
                        )


class MegaGateCudaTest(unittest.TestCase):
    def test_finite_reference_and_graph_replay(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("SM10x required")
        self.assertIsNotNone(_MEGA_GATE._native_gate(), "bf16_mega_gate required")
        generator = torch.Generator(device="cuda").manual_seed(17)
        stream = torch.cuda.Stream()
        with torch.inference_mode(), torch.cuda.stream(stream):
            for experts, topk in ((384, 6), (128, 3)):
                weight = (
                    torch.randn(
                        (experts, 5120),
                        device="cuda",
                        dtype=torch.bfloat16,
                        generator=generator,
                    )
                    * 0.01
                )
                # Separated ranks avoid making cutoff-tie behavior the contract.
                bias = torch.arange(experts, device="cuda", dtype=torch.float32) * 4
                gate = _gate(weight, bias, topk)
                for batch in (1, 64, 384):
                    with self.subTest(experts=experts, batch=batch):
                        x = torch.randn(
                            (batch, 5120),
                            device="cuda",
                            dtype=torch.bfloat16,
                            generator=generator,
                        )
                        for _ in range(2):
                            _MEGA_GATE.mega_gate(x, gate, ep_rank=3)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            weights, indices = _MEGA_GATE.mega_gate(x, gate, ep_rank=3)
                        addresses = (weights.data_ptr(), indices.data_ptr())
                        for _ in range(2):
                            x.copy_(
                                torch.randn(
                                    x.shape,
                                    device="cuda",
                                    dtype=x.dtype,
                                    generator=generator,
                                )
                            )
                            graph.replay()
                            scores = F.softplus(
                                F.linear(x.float(), weight.float())
                            ).sqrt()
                            expected_ids = (scores + bias).topk(topk, dim=-1).indices
                            expected_ids = expected_ids.sort(-1).values
                            expected_weights = scores.gather(1, expected_ids)
                            expected_weights = (
                                expected_weights
                                / (expected_weights.sum(-1, keepdim=True) + 1e-20)
                                * gate.route_scale
                            )
                            sorted_ids, order = indices.sort(-1)
                            torch.testing.assert_close(sorted_ids, expected_ids)
                            torch.testing.assert_close(
                                weights.gather(1, order),
                                expected_weights,
                                rtol=1e-4,
                                atol=1e-5,
                            )
                            self.assertEqual(
                                addresses, (weights.data_ptr(), indices.data_ptr())
                            )
        stream.synchronize()


if __name__ == "__main__":
    unittest.main()
