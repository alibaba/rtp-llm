"""4096-token MegaMoE gate-pack boundary and real CUDA output equivalence."""

import os
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import Mock, patch

import torch
from rtp_llm.config.model_config import ModelConfig

from rtp_llm.models_py.model_desc import generic_moe
from rtp_llm.models_py.modules.base.cuda.select_topk import SelectTopk


class _Backend(torch.nn.Module):
    includes_shared_expert = False
    supports_gate_pack = True
    topk_ids_dtype = torch.int64
    router = SimpleNamespace(tp_collective_size=1, supports_skip_tp_allreduce=False)
    fused_experts = SimpleNamespace(
        gated_shared_expert_requested=False, uses_shared_expert_gates=False
    )

    def __init__(self, real_pack=False):
        super().__init__()
        self.real_pack = real_pack
        self.calls = []
        self.buffer = None

    def _buffer(self, x):
        n, dim = x.shape
        self.buffer = (
            torch.empty((n, dim), device=x.device, dtype=torch.float8_e4m3fn),
            torch.empty((n, dim // 128), device=x.device, dtype=torch.int32),
            torch.empty((n, 10), device=x.device, dtype=torch.int64),
            torch.empty((n, 10), device=x.device, dtype=torch.float32),
        )
        return self.buffer

    def forward_gate_pack(self, hidden_states, gate_payload, **kwargs):
        self.calls.append("fused")
        if self.real_pack:
            from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
                fused_pack_mega_moe_gate_inputs,
            )

            fused_pack_mega_moe_gate_inputs(
                hidden_states,
                gate_payload.scores,
                *self._buffer(hidden_states),
                topk=10,
                score_func="softmax",
                route_scale=1.0,
            )
        return hidden_states

    def forward(self, hidden_states, topk_ids, topk_weights, **kwargs):
        self.calls.append("separate")
        if self.real_pack:
            from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
                fused_pack_mega_moe_inputs_optimized,
            )

            fused_pack_mega_moe_inputs_optimized(
                hidden_states, topk_weights, topk_ids, *self._buffer(hidden_states)
            )
        return hidden_states


def make_layer(strategy="mega_moe_fp8", *, real_pack=False, **overrides):
    config = ModelConfig()
    config.hidden_size = 4096
    config.inter_size = 128
    config.expert_num = 512
    config.moe_k = 10
    config.moe_style = 1
    config.scoring_func = 0
    config.has_moe_norm = True
    config.routed_scaling_factor = 1.0
    for key, value in overrides.items():
        setattr(config, key, value)
    parallel = SimpleNamespace(ep_size=4, get_ffn_tp_size=lambda: 1)
    backend = _Backend(real_pack)
    prefix = "rtp_llm.models_py.model_desc.generic_moe."
    with patch(
        prefix + "LinearFactory.create_linear_from_weights", return_value=Mock()
    ), patch(prefix + "MoEConfigAdapter"), patch(prefix + "FusedMoeFactory") as factory:
        factory.return_value.create_fused_moe.return_value = backend
        layer = generic_moe.GenericMoeLayer(
            config,
            parallel,
            {},
            SimpleNamespace(moe_strategy=strategy, fake_balance_expert=False),
        )
    return layer, backend


class GenericMoeGatePackRoutingTest(TestCase):
    def test_boundary_and_backend_scope(self):
        for strategy in ("mega_moe_fp8", "mega_moe_fp8_se", "auto", "mega_moe"):
            with self.subTest(strategy=strategy), patch.object(
                generic_moe, "SelectTopk"
            ) as select:
                layer, backend = make_layer(strategy)
                split = strategy in ("mega_moe_fp8", "mega_moe_fp8_se")
                self.assertEqual(
                    select.call_args.kwargs.get("use_fused_512"),
                    True if split else None,
                )
                for n in (0, 1, 4095, 4096, 4097, 8192):
                    x = torch.zeros(n, 1)
                    layer.gate = Mock(
                        return_value=torch.zeros(n, 512, dtype=torch.bfloat16)
                    )
                    layer(x)
                    self.assertEqual(
                        backend.calls[-1], "separate" if split and n > 4096 else "fused"
                    )

    def test_other_routing_semantics_keep_existing_path(self):
        for overrides in (
            dict(expert_num=256),
            dict(scoring_func=2),
            dict(has_moe_norm=False),
            dict(routed_scaling_factor=2.5),
        ):
            with self.subTest(overrides=overrides), patch.object(
                generic_moe, "SelectTopk"
            ) as select:
                layer, _ = make_layer(**overrides)
                self.assertFalse(layer._split_mega_moe_gate_pack)
                self.assertNotIn("use_fused_512", select.call_args.kwargs)

    def test_explicit_topk_override_and_environment_default(self):
        config = ModelConfig()
        for env in ("0", "1"):
            with patch.dict(os.environ, {"RTP_FUSED_TOPK_512": env}), patch(
                "rtp_llm.models_py.modules.base.cuda.select_topk.compute_ops.SelectTopkOp"
            ) as op:
                SelectTopk(config)
                self.assertEqual(op.call_args.kwargs["use_fused_512"], env == "1")
                SelectTopk(config, use_fused_512=True)
                self.assertTrue(op.call_args.kwargs["use_fused_512"])
                SelectTopk(config, use_fused_512=False)
                self.assertFalse(op.call_args.kwargs["use_fused_512"])

    @skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_boundary_matches_fused_reference(self):
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
            fused_pack_mega_moe_gate_inputs,
        )

        # The production branch must force topk512 even if the environment opts out.
        with patch.dict(os.environ, {"RTP_FUSED_TOPK_512": "0"}):
            layer, backend = make_layer(real_pack=True)
        torch.manual_seed(20260920)
        for n in (4096, 4097, 8192, 4096):
            with self.subTest(tokens=n):
                x = torch.randn(n, 4096, device="cuda", dtype=torch.bfloat16) * 0.3
                scores = torch.randn(n, 512, device="cuda", dtype=torch.bfloat16)
                layer.gate = Mock(return_value=scores)
                with patch.object(
                    layer.select_topk, "forward", wraps=layer.select_topk.forward
                ) as topk:
                    output = layer(x)
                    self.assertEqual(topk.call_count, int(n > 4096))
                self.assertIs(output, x)
                self.assertEqual(
                    backend.calls[-1], "fused" if n <= 4096 else "separate"
                )
                got = backend.buffer
                ref = tuple(torch.empty_like(t) for t in got)
                fused_pack_mega_moe_gate_inputs(
                    x, scores, *ref, topk=10, score_func="softmax", route_scale=1.0
                )
                self.assertTrue(
                    torch.equal(got[0].view(torch.uint8), ref[0].view(torch.uint8))
                )
                self.assertTrue(torch.equal(got[1], ref[1]))
                self.assertTrue(torch.equal(got[2], ref[2]))
                torch.testing.assert_close(got[3], ref[3], rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    main()
