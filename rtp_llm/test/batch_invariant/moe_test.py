"""Check Qwen3 MoE routing and expert arithmetic against a per-token CPU reference."""

import unittest
import weakref
from unittest.mock import patch

import torch
import torch.nn.functional as F


def _expert_reference(hidden, w1, w2, ids, weights):
    result = torch.zeros_like(hidden, dtype=torch.float32)
    for row in range(hidden.size(0)):
        for slot, expert in enumerate(ids[row].tolist()):
            proj = F.linear(hidden[row].float(), w1[expert].float()).bfloat16()
            up, gate = proj.float().chunk(2)
            activation = (F.silu(gate) * up).bfloat16()
            down = F.linear(activation.float(), w2[expert].float())
            result[row] += (down * weights[row, slot]).bfloat16().float()
    return result.bfloat16()


def _make_layer(gate_weight, w1, w2, top_k, *, batch_invariant):
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
    from rtp_llm.ops import ActivationType, HWKernelConfig, MoeConfig, ParallelismConfig
    from rtp_llm.utils.model_weight import W

    config = ModelConfig()
    config.model_type = "qwen_3_moe"
    config.expert_num, config.hidden_size, config.moe_inter_size = w2.shape
    config.inter_size = config.moe_inter_size
    config.moe_k = top_k
    config.moe_style = 1
    config.has_moe_norm = True
    config.activation_type = ActivationType.Swiglu
    moe = MoeConfig()
    moe.moe_strategy = "no_quant_cpp"
    moe.use_all_gather = True
    hw = HWKernelConfig()
    hw.batch_invariant = batch_invariant
    weights = {W.moe_gate: gate_weight, W.moe_w1: w1, W.moe_w2: w2}
    return GenericMoeLayer(
        config, ParallelismConfig(), weights, moe, hw_kernel_config=hw
    )


@unittest.skipUnless(
    torch.version.cuda is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (10, 3),
    "requires an SM103 CUDA GPU",
)
class BatchInvariantMoeTest(unittest.TestCase):
    @torch.inference_mode()
    def test_routing_and_experts_alone_and_in_mixed_batch(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
            triton_fused_executor,
        )

        experts, hidden_size, inter_size, top_k = 128, 128, 128, 8

        gen = torch.Generator().manual_seed(561)

        def values(*shape):
            return torch.randint(-4, 5, shape, generator=gen).bfloat16() / 8

        # The 8th and 9th candidates tie across warp lanes. The second
        # token uses a different set of experts, including high expert IDs.
        gate_weight = torch.zeros(hidden_size, experts, dtype=torch.bfloat16)
        scores = torch.full((experts,), -8.0, dtype=torch.bfloat16)
        scores[[126, 17, 99, 4, 68, 2, 31]] = torch.arange(7, 0, -1).bfloat16() / 4
        scores[[32, 63]] = 0
        gate_weight[0], gate_weight[1] = scores, scores.flip(0)
        inputs = values(2, hidden_size)
        inputs[:, :2] = torch.eye(2)
        w1 = values(experts, 2 * inter_size, hidden_size)
        w2 = values(experts, hidden_size, inter_size)

        probs = F.softmax(inputs.float() @ gate_weight.float(), dim=-1)
        ids = probs.argsort(dim=-1, descending=True, stable=True)[:, :top_k]
        self.assertEqual(ids[0, -1].item(), 32)
        weights = probs.gather(1, ids)
        weights /= weights.sum(dim=-1, keepdim=True)
        reference = _expert_reference(inputs, w1, w2, ids, weights)

        def forward_with_configs(model, batch):
            configs = []
            gate_up = None
            invoke = triton_fused_executor.invoke_fused_moe_kernel

            def record(*args, **kwargs):
                nonlocal gate_up
                configs.append(kwargs["config"])
                if kwargs["mul_routed_weight"]:
                    self.assertEqual(
                        gate_up() is None, model.fused_moe.fused_experts._batch_invariant
                    )
                else:
                    gate_up = weakref.ref(args[2])
                return invoke(*args, **kwargs)

            with patch.object(triton_fused_executor, "invoke_fused_moe_kernel", record):
                output = model(batch)
            self.assertEqual(len(configs), 2)
            return output.cpu(), configs

        inputs, gate_weight, w1, w2 = (t.cuda() for t in (inputs, gate_weight, w1, w2))
        layer = _make_layer(gate_weight, w1, w2, top_k, batch_invariant=True)
        ordinary = _make_layer(gate_weight, w1, w2, top_k, batch_invariant=False)
        for instance in (layer, ordinary):
            self.assertEqual(instance.fused_moe.strategy_name, "no_quant_cpp")
        self.assertIs(
            type(layer.fused_moe.fused_experts), type(ordinary.fused_moe.fused_experts)
        )

        # Both modes execute real gate/up and down projections. A 35-token batch
        # crosses the ordinary tile-selection threshold for these 128 experts.
        order = torch.arange(35) % 2
        batch = inputs.index_select(0, order.cuda())
        first, solo_configs = forward_with_configs(layer, inputs[:1])
        regular, regular_solo_configs = forward_with_configs(ordinary, inputs[:1])
        mixed, mixed_configs = forward_with_configs(layer, batch)
        regular_mixed, regular_mixed_configs = forward_with_configs(ordinary, batch)
        self.assertEqual(solo_configs, mixed_configs)
        self.assertNotEqual(regular_solo_configs, regular_mixed_configs)
        torch.testing.assert_close(
            regular.float(), reference[:1].float(), rtol=0.01, atol=0.004
        )
        torch.testing.assert_close(
            regular_mixed.float(), reference[order].float(), rtol=0.01, atol=0.004
        )

        solo_ids, solo_weights, solo_outputs = [], [], []
        for token in inputs.split(1):
            routed_ids, routed_weights = layer._route(token)
            solo_ids.append(routed_ids.cpu())
            solo_weights.append(routed_weights.cpu())
            solo_outputs.append(layer(token).cpu())
        solo_ids, solo_weights, solo_outputs = map(
            torch.cat, (solo_ids, solo_weights, solo_outputs)
        )
        torch.testing.assert_close(solo_ids.long(), ids, rtol=0, atol=0)
        torch.testing.assert_close(solo_weights, weights, rtol=2e-6, atol=1e-7)
        torch.testing.assert_close(
            solo_outputs.float(), reference.float(), rtol=0.01, atol=0.004
        )

        # Both tokens change row index and expert occupancy across multiple tiles.
        batch_ids, batch_weights = layer._route(batch)
        self.assertTrue(torch.equal(batch_ids.cpu(), solo_ids[order]))
        self.assertTrue(torch.equal(batch_weights.cpu(), solo_weights[order]))
        self.assertTrue(torch.equal(first, solo_outputs[:1]))
        self.assertTrue(torch.equal(mixed, solo_outputs[order]))

    @torch.inference_mode()
    def test_weight_rounding_and_slot_order(self):
        from rtp_llm.models_py.determinism import batch_invariant
        from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
            ExpertForwardPayload,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
            triton_fused_executor,
        )

        experts, hidden_size, inter_size = 8, 128, 128
        hidden = torch.zeros(2, hidden_size, dtype=torch.bfloat16)
        hidden[:, 0] = 1
        w1 = torch.zeros(experts, 2 * inter_size, hidden_size, dtype=torch.bfloat16)
        w1[:, 0, 0], w1[:, 1, 0] = 1 / 32, 1 / 4096
        w1[:, inter_size : inter_size + 2, 0] = 32
        w2 = torch.zeros(experts, hidden_size, inter_size, dtype=torch.bfloat16)
        w2[:4, 0, 0] = torch.tensor([2**24, 1, -(2**24), 6]).bfloat16() * 8
        w2[0, 1, :2] = torch.tensor([1.0, 0.5])
        ids = torch.arange(experts, dtype=torch.int32).repeat(2, 1)
        weights = torch.full((2, experts), 1 / 8)
        weights[1] = torch.tensor([3 / 8, 1 / 8, 1 / 8, 1 / 8] + [1 / 16] * 4)

        # Slot order gives ((2**24 + 1) - 2**24) + 6 = 6 in FP32.
        # The down-projection's 1 + 1/256 must be weighted before rounding.
        reference = _expert_reference(hidden, w1, w2, ids, weights)
        self.assertEqual(reference[0, 0].item(), 6)
        self.assertEqual(reference[1, 1].item(), 0.376953125)
        hidden, w1, w2, ids, weights = (
            t.cuda() for t in (hidden, w1, w2, ids, weights)
        )
        gate = hidden.new_zeros((hidden_size, experts))
        executor = _make_layer(
            gate, w1, w2, experts, batch_invariant=True
        ).fused_moe.fused_experts

        def execute(x, route_ids, route_weights):
            payload = ExpertForwardPayload(
                expert_x=x, expert_topk_ids=route_ids, expert_topk_weights=route_weights
            )
            return executor.execute(
                payload,
                activation="silu",
                expert_map=None,
                a2_scale=None,
                apply_router_weight_on_input=False,
                extra_expert_args=None,
            ).fused_expert_output

        actual = execute(hidden, ids, weights)
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
        with (
            patch.object(batch_invariant, "_EXPERT_WORKSPACE_BYTES", 1),
            patch.object(
                triton_fused_executor,
                "invoke_fused_moe_kernel",
                wraps=triton_fused_executor.invoke_fused_moe_kernel,
            ) as gemm,
        ):
            chunked = execute(hidden, ids, weights)
            self.assertEqual(gemm.call_count, 4)
            gemm.reset_mock()
            empty = execute(hidden[:0], ids[:0], weights[:0])
            gemm.assert_not_called()
        self.assertTrue(torch.equal(chunked, actual))
        self.assertEqual(empty.shape, (0, hidden_size))
        self.assertEqual(empty.dtype, hidden.dtype)


if __name__ == "__main__":
    unittest.main()
