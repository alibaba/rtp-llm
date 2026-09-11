"""V4.1 weight/router component checks, not Mega collective certification."""

import ast
import hashlib
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.models_py.modules.dsv41.linear import is_supported
from rtp_llm.models_py.modules.dsv41.moe import (
    V41MoE,
    V41MoERouter,
    pack_v41_moe_weights,
)
from rtp_llm.models_py.modules.dsv41.test.fixture import flash_config
from rtp_llm.models_py.modules.dsv41.test.official_compressor import (
    OFFICIAL_MODEL_SHA256,
    load_official_compressor,
)
from rtp_llm.utils.model_weight import W


def official_gate_forward():
    configured = os.environ.get("DSV41_OFFICIAL_MODEL_PATH")
    if configured is None:
        configured = str(Path(os.environ["DSV41_MODEL_PATH"]) / "inference/model.py")
    source = Path(configured).read_bytes()
    if hashlib.sha256(source).hexdigest() != OFFICIAL_MODEL_SHA256:
        raise RuntimeError("the official V4.1 router source changed")
    tree = ast.parse(source, filename=configured)
    functions = [
        method
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Gate"
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "forward"
    ]
    if len(functions) != 1:
        raise RuntimeError("the official V4.1 router forward definition is missing")
    namespace = dict(load_official_compressor().__dict__)
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), configured, "exec"),
        namespace,
    )
    return namespace["forward"]


class RecordExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(max_tokens_per_rank=33)
        self.calls = []

    def forward(self, hidden, weights, indices):
        self.calls.append((hidden, weights.clone(), indices.clone()))
        return hidden.clone()


class V41MoEBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(os.environ.get("DSV41_TEST_DEVICE", "cuda"))
        if not is_supported(torch.empty(0, device=cls.device)):
            raise RuntimeError("V4.1 MoE component checks require CUDA13/Blackwell")
        cls.config = V41Config.from_dict(flash_config())
        cls.reference_gate = staticmethod(official_gate_forward())
        torch.backends.cuda.matmul.allow_tf32 = False

    def router_weights(self, draft=False):
        experts = 128 if draft else 384
        generator = torch.Generator(device=self.device).manual_seed(41071)
        return {
            "ffn.gate.weight": torch.randn(
                experts, 5120, generator=generator, device=self.device
            )
            .mul_(0.0625)
            .bfloat16(),
            "ffn.gate.bias": torch.linspace(-0.25, 0.25, experts, device=self.device),
            "ffn.gate.bias_vl": torch.linspace(
                0.75, -0.75, experts, device=self.device
            ),
        }

    def complete_weights(self, ep_size, ep_rank, draft=False):
        weights = self.router_weights(draft)
        experts = 128 if draft else 384
        local = range(ep_rank * experts // ep_size, (ep_rank + 1) * experts // ep_size)
        for name, (rows, columns) in {
            "w1": (2304, 5120),
            "w3": (2304, 5120),
            "w2": (5120, 2304),
        }.items():
            part = ("w1", "w3", "w2").index(name)
            for expert in local:
                prefix = f"ffn.experts.{expert}.{name}."
                weights[prefix + "weight"] = torch.full(
                    (rows, columns // 2),
                    (expert * 3 + part) % 127,
                    dtype=torch.int8,
                    device=self.device,
                )
                weights[prefix + "scale"] = torch.full(
                    (rows, columns // 32),
                    (expert + part) % 63 + 64,
                    dtype=torch.uint8,
                    device=self.device,
                ).view(torch.float8_e8m0fnu)
            prefix = "ffn.shared_experts." + name + "."
            weights[prefix + "weight"] = torch.full(
                (rows, columns),
                1 << part,
                dtype=torch.uint8,
                device=self.device,
            ).view(torch.float8_e4m3fn)
            weights[prefix + "scale"] = torch.full(
                (rows // 32, columns // 32),
                120 + part,
                dtype=torch.uint8,
                device=self.device,
            ).view(torch.float8_e8m0fnu)
        return weights, local

    def test_router_matches_unchanged_official_forward_for_text_and_image_rows(self):
        for draft in (False, True):
            weights = self.router_weights(draft)
            router = V41MoERouter(self.config, weights, draft=draft)
            reference = SimpleNamespace(
                weight=weights["ffn.gate.weight"],
                bias=weights["ffn.gate.bias"],
                bias_vl=weights["ffn.gate.bias_vl"],
                gate_temp=1.0,
                score_func="sqrtsoftplus",
                norm_topk_prob=True,
                topk=3 if draft else 6,
                route_scale=1.5,
            )
            self.assertIs(router.weight, weights["ffn.gate.weight"])
            self.assertIs(router.bias, weights["ffn.gate.bias"])
            self.assertIs(router.bias_vl, weights["ffn.gate.bias_vl"])
            for rows in (0, 1, 13, 33):
                for mixed in (False, True):
                    with self.subTest(draft=draft, rows=rows, mixed=mixed):
                        generator = torch.Generator(device=self.device).manual_seed(
                            41083 + rows
                        )
                        hidden = torch.randn(
                            rows, 5120, generator=generator, device=self.device
                        ).bfloat16()
                        image_mask = (
                            (torch.arange(rows, device=self.device) % 3 != 0)
                            if mixed
                            else None
                        )
                        actual = router(hidden, image_mask)
                        expected = self.reference_gate(reference, hidden, image_mask)
                        self.assertEqual(actual[0].dtype, torch.float32)
                        self.assertEqual(actual[1].dtype, torch.int64)
                        for candidate, official in zip(actual, expected):
                            torch.testing.assert_close(
                                candidate, official, rtol=0, atol=0
                            )

    def test_image_bias_changes_expert_ids_without_scaling_unbiased_weights(self):
        weights = self.router_weights()
        weights["ffn.gate.weight"].zero_()
        weights["ffn.gate.bias"].copy_(torch.arange(384, device=self.device))
        weights["ffn.gate.bias_vl"].copy_(torch.arange(383, -1, -1, device=self.device))
        router = V41MoERouter(self.config, weights)
        hidden = torch.ones(6, 5120, device=self.device, dtype=torch.bfloat16)
        image_mask = torch.tensor(
            [False, True, True, True, True, False], device=self.device
        )
        route, indices = router(hidden, image_mask)
        expected = torch.tensor(
            [[383, 382, 381, 380, 379, 378]]
            + [[0, 1, 2, 3, 4, 5]] * 4
            + [[383, 382, 381, 380, 379, 378]],
            device=self.device,
        )
        torch.testing.assert_close(indices, expected, rtol=0, atol=0)
        torch.testing.assert_close(route, torch.full_like(route, 0.25), rtol=0, atol=0)

    def test_packed_expert_bytes_global_order_and_shared32_layout(self):
        for draft, ep_size, rank in (
            (False, 8, 0),
            (False, 8, 7),
            (False, 16, 15),
            (True, 8, 7),
            (True, 16, 15),
        ):
            with self.subTest(draft=draft, ep_size=ep_size, rank=rank):
                weights, local = self.complete_weights(ep_size, rank, draft)
                pointers = {key: value.data_ptr() for key, value in weights.items()}
                packed = pack_v41_moe_weights(
                    self.config, weights, ep_size=ep_size, ep_rank=rank, draft=draft
                )
                self.assertEqual(len(packed), 10)
                for name in ("w1", "w3", "w2"):
                    for part, suffix in (("weight", "w"), ("scale", "s")):
                        stack = packed[getattr(W, f"v4_routed_{name}_{suffix}")]
                        for index, expert in enumerate(local):
                            original = weights[f"ffn.experts.{expert}.{name}.{part}"]
                            self.assertEqual(stack.dtype, original.dtype)
                            torch.testing.assert_close(
                                stack[index].view(torch.uint8),
                                original.view(torch.uint8),
                                rtol=0,
                                atol=0,
                            )
                for part, suffix, rows in (("weight", "w", 2304), ("scale", "s", 72)):
                    merged = packed[getattr(W, "v4_shared_w13_" + suffix)]
                    for index, name in enumerate(("w1", "w3")):
                        original = weights[f"ffn.shared_experts.{name}.{part}"]
                        self.assertEqual(merged.dtype, original.dtype)
                        torch.testing.assert_close(
                            merged[index * rows : (index + 1) * rows].view(torch.uint8),
                            original.view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                    self.assertIs(
                        packed[getattr(W, "v4_shared_w2_" + suffix)],
                        weights["ffn.shared_experts.w2." + part],
                    )
                self.assertEqual(
                    pointers, {key: value.data_ptr() for key, value in weights.items()}
                )
                del packed, weights

    def test_invalid_inventory_or_layout_fails_before_packing(self):
        weights, local = self.complete_weights(16, 15)
        key = f"ffn.experts.{local.start}.w1.weight"
        missing = dict(weights)
        del missing[key]
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            pack_v41_moe_weights(self.config, missing, ep_size=16, ep_rank=15)
        wrong_owner = dict(weights)
        wrong_owner["ffn.experts.0.w1.weight"] = wrong_owner.pop(key)
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            pack_v41_moe_weights(self.config, wrong_owner, ep_size=16, ep_rank=15)
        for wrong in (weights[key].to(torch.bfloat16), weights[key][:, :128]):
            invalid = dict(weights, **{key: wrong})
            with self.assertRaisesRegex(
                ValueError, "checkpoint shape/dtype/device/layout"
            ):
                pack_v41_moe_weights(self.config, invalid, ep_size=16, ep_rank=15)
        scale_key = f"ffn.experts.{local.start}.w1.scale"
        invalid = dict(weights, **{scale_key: weights[scale_key].float()})
        with self.assertRaisesRegex(ValueError, "checkpoint shape/dtype/device/layout"):
            pack_v41_moe_weights(self.config, invalid, ep_size=16, ep_rank=15)
        for size, rank in ((1, 0), (4, 0), (8, 8), (16, -1)):
            with self.subTest(size=size, rank=rank), self.assertRaises(ValueError):
                pack_v41_moe_weights(self.config, {}, ep_size=size, ep_rank=rank)

    def test_world_ownership_and_b300_ep16_are_rejected_before_weight_setup(self):
        weights = self.router_weights()
        with patch("torch.distributed.is_initialized", return_value=False):
            with self.assertRaisesRegex(ValueError, "actual WORLD EP group"):
                V41MoE.from_weights(
                    self.config,
                    0,
                    weights,
                    ep_size=8,
                    ep_rank=0,
                    max_tokens_per_rank=33,
                )
        with patch("torch.cuda.get_device_capability", return_value=(10, 3)):
            with self.assertRaisesRegex(ValueError, "GB200 NVLink deployment"):
                V41MoE.from_weights(
                    self.config,
                    0,
                    weights,
                    ep_size=16,
                    ep_rank=0,
                    max_tokens_per_rank=33,
                )

    def test_wrapper_delivers_routes_and_empty_rows_to_the_expert_backend(self):
        router = V41MoERouter(self.config, self.router_weights())
        experts = RecordExperts()
        model = V41MoE(router, experts)
        for rows in (0, 13):
            hidden = torch.ones(rows, 5120, device=self.device, dtype=torch.bfloat16)
            image_mask = torch.arange(rows, device=self.device) % 2 == 0
            expected_routes = router(hidden, image_mask)
            actual = model(hidden, image_mask)
            torch.testing.assert_close(actual, hidden, rtol=0, atol=0)
            self.assertIs(experts.calls[-1][0], hidden)
            for delivered, expected in zip(experts.calls[-1][1:], expected_routes):
                torch.testing.assert_close(delivered, expected, rtol=0, atol=0)
        self.assertEqual(len(experts.calls), 2)
        with self.assertRaisesRegex(ValueError, "startup token budget"):
            model(torch.empty(34, 5120, device=self.device, dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "image mask"):
            model(
                torch.empty(1, 5120, device=self.device, dtype=torch.bfloat16),
                torch.zeros(1, device=self.device, dtype=torch.int32),
            )


if __name__ == "__main__":
    unittest.main()
